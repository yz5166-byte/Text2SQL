"""Load and validate the semantic layer: physical catalog, join graph, dimensions, metrics.

The three YAML files under semantic/ are the single source of truth. This module turns them
into typed objects and answers structural questions the compiler needs:
which join path connects a metric's table to a dimension, and which dimensions a metric
can be sliced by (reachability over many-to-one edges only).
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from text2sql.config import SEMANTIC_DIR

Code = str | int | None


class Column(BaseModel):
    type: str
    name_zh: str
    description: str = ""
    dimension: str | None = None
    distinct: int | None = None  # distinct values in the full table, when the database holds only a sample


class Table(BaseModel):
    key: str
    name_zh: str
    layer: str
    grain: str
    description: str
    notes: str = ""
    columns: dict[str, Column]


class JoinEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    condition: str  # SQL join condition (YAML 1.1 would read a bare `on` key as boolean True)
    table: str | None = None  # physical table when `to` is an alias such as owner_disp

    @property
    def physical_table(self) -> str:
        return self.table or self.to


class TimeGrain(BaseModel):
    label: str
    sql: str  # template with {col}

    def render(self, col: str) -> str:
        return self.sql.format(col=col)


class DimValue(BaseModel):
    codes: list[Code]
    label: str
    aliases: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def _single_code(cls, data: Any) -> Any:
        if isinstance(data, dict) and "code" in data:
            data = dict(data)
            data["codes"] = [data.pop("code")]
        return data


class Dimension(BaseModel):
    key: str
    name: str
    aliases: list[str] = []
    column: str  # "<table or alias>.<column>"
    description: str = ""
    values: list[DimValue] = []

    @property
    def alias(self) -> str:
        return self.column.split(".", 1)[0]

    def label_for(self, code: Code) -> str:
        for value in self.values:
            if len(value.codes) == 1 and value.codes[0] == code:
                return value.label
        return "" if code is None else str(code)


class Metric(BaseModel):
    key: str
    name: str
    aliases: list[str] = []
    domain: str
    table: str
    time_column: str
    agg: Literal["count", "sum", "count_distinct", "ratio"]
    column: str | None = None
    filters: list[str] = []
    numerator: str | None = None
    denominator: str | None = None
    okr: bool = False
    unit: str = ""
    description: str = ""

    @property
    def additivity(self) -> Literal["additive", "non_additive", "ratio"]:
        if self.agg == "ratio":
            return "ratio"
        return "non_additive" if self.agg == "count_distinct" else "additive"


class SemanticModel(BaseModel):
    tables: dict[str, Table]
    foreign_keys: list[str]
    join_graph: list[JoinEdge]
    time_grains: dict[str, TimeGrain]
    dimensions: dict[str, Dimension]
    metrics: dict[str, Metric]
    ambiguous_terms: dict[str, list[str]]

    # ------------------------------------------------------------------ structure
    def physical_table(self, alias: str) -> str:
        """Map a join-graph alias (owner_client) to its physical table (client)."""
        if alias in self.tables:
            return alias
        for edge in self.join_graph:
            if edge.to == alias:
                return edge.physical_table
        raise KeyError(f"unknown table alias: {alias}")

    def join_path(self, base: str, target: str) -> list[JoinEdge] | None:
        """Shortest many-to-one join path from `base` to `target` (BFS), [] if same table."""
        if base == target:
            return []
        parents: dict[str, JoinEdge] = {}
        queue, seen = deque([base]), {base}
        while queue:
            node = queue.popleft()
            for edge in self.join_graph:
                if edge.from_ != node or edge.to in seen:
                    continue
                parents[edge.to] = edge
                if edge.to == target:
                    path, cur = [], target
                    while cur != base:
                        path.append(parents[cur])
                        cur = parents[cur].from_
                    return list(reversed(path))
                seen.add(edge.to)
                queue.append(edge.to)
        return None

    def restricted_to(self, tables: set[str]) -> "SemanticModel":
        """A consistent copy that knows only `tables` (the BIRD benchmark has no derived tables):
        metrics, dimensions, joins, foreign keys and ambiguous terms on other tables are dropped."""
        def on_kept_table(name: str) -> bool:  # a table or join alias, optionally with ".column"
            try:
                return self.physical_table(name.split(".")[0]) in tables
            except KeyError:
                return False

        metrics = {k: m for k, m in self.metrics.items() if m.table in tables}
        metrics = {k: m for k, m in metrics.items() if m.agg != "ratio" or {m.numerator, m.denominator} <= set(metrics)}
        return self.model_copy(update={
            "tables": {k: v for k, v in self.tables.items() if k in tables},
            "metrics": metrics,
            "dimensions": {k: d for k, d in self.dimensions.items() if on_kept_table(d.column)},
            "join_graph": [e for e in self.join_graph if on_kept_table(e.from_) and e.physical_table in tables],
            "foreign_keys": [fk for fk in self.foreign_keys if all(on_kept_table(s.strip()) for s in fk.split("->"))],
            "ambiguous_terms": {t: [c for c in cs if c in metrics] for t, cs in self.ambiguous_terms.items()
                                if any(c in metrics for c in cs)},
        })

    def uncovered_tables(self) -> list[str]:
        """Tables that no metric reads and no dimension groups by (代扣指令表 order, for example)."""
        used = {m.table for m in self.metrics.values()}
        used |= {self.physical_table(d.column.split(".")[0]) for d in self.dimensions.values()}
        return [t for t in self.tables if t not in used]

    def base_metric(self, key: str) -> str | None:
        """The unconditioned metric with the same table and aggregation (cash_withdraw_cnt -> trans_cnt)."""
        m = self.metrics[key]
        return next((k for k, x in self.metrics.items() if k != key and not x.filters
                     and (x.table, x.agg, x.column) == (m.table, m.agg, m.column)), None)

    def reachable_dimensions(self, metric_key: str) -> list[str]:
        metric = self.metrics[metric_key]
        return [
            key for key, dim in self.dimensions.items()
            if self.join_path(metric.table, dim.alias) is not None
        ]

    def component_metrics(self, metric_key: str) -> list[Metric]:
        """Additive building blocks of a metric (itself unless it is a ratio)."""
        metric = self.metrics[metric_key]
        if metric.agg != "ratio":
            return [metric]
        return [self.metrics[metric.numerator], self.metrics[metric.denominator]]

    def okr_metrics(self) -> list[Metric]:
        return [m for m in self.metrics.values() if m.okr]

    # ------------------------------------------------------------------ validation
    def validate_references(self) -> list[str]:
        """Return a list of broken references; empty means the model is consistent."""
        errors: list[str] = []

        def check_column(ref: str, where: str) -> None:
            alias, _, col = ref.partition(".")
            try:
                table = self.physical_table(alias)
            except KeyError:
                errors.append(f"{where}: unknown table alias '{alias}'")
                return
            if table not in self.tables or col not in self.tables[table].columns:
                errors.append(f"{where}: unknown column '{ref}'")

        for fk in self.foreign_keys:
            left, _, right = (s.strip() for s in fk.partition("->"))
            check_column(left, f"foreign_key {fk}")
            check_column(right, f"foreign_key {fk}")
        for edge in self.join_graph:
            if edge.from_ not in self.tables and edge.from_ not in {e.to for e in self.join_graph}:
                errors.append(f"join {edge.from_}->{edge.to}: unknown source")
            if edge.physical_table not in self.tables:
                errors.append(f"join {edge.from_}->{edge.to}: unknown table {edge.physical_table}")
        for key, dim in self.dimensions.items():
            check_column(dim.column, f"dimension {key}")
        for key, metric in self.metrics.items():
            if metric.table not in self.tables:
                errors.append(f"metric {key}: unknown table {metric.table}")
            check_column(metric.time_column, f"metric {key} time_column")
            if metric.column:
                check_column(metric.column, f"metric {key} column")
            if metric.agg in ("sum", "count_distinct") and not metric.column:
                errors.append(f"metric {key}: agg {metric.agg} requires column")
            if metric.agg == "ratio":
                for part in (metric.numerator, metric.denominator):
                    if part not in self.metrics:
                        errors.append(f"metric {key}: unknown component {part}")
                    elif self.metrics[part].table != metric.table:
                        errors.append(f"metric {key}: component {part} on a different table")
                    elif self.metrics[part].agg not in ("count", "sum"):
                        errors.append(f"metric {key}: component {part} must be additive")
        for term, candidates in self.ambiguous_terms.items():
            for cand in candidates:
                if cand not in self.metrics:
                    errors.append(f"ambiguous term {term}: unknown metric {cand}")
        return errors


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=4)
def load_semantic_model(semantic_dir: Path = SEMANTIC_DIR) -> SemanticModel:
    catalog = _read(semantic_dir / "catalog.yaml")
    dims = _read(semantic_dir / "dimensions.yaml")
    metrics = _read(semantic_dir / "metrics.yaml")
    return SemanticModel(
        tables={k: Table(key=k, **v) for k, v in catalog["tables"].items()},
        foreign_keys=catalog["foreign_keys"],
        join_graph=[JoinEdge(**e) for e in dims["join_graph"]],
        time_grains={k: TimeGrain(**v) for k, v in dims["time_grains"].items()},
        dimensions={k: Dimension(key=k, **v) for k, v in dims["dimensions"].items()},
        metrics={k: Metric(key=k, **v) for k, v in metrics["metrics"].items()},
        ambiguous_terms=metrics.get("ambiguous_terms", {}),
    )
