"""Layer 2: compile a QueryIR into parameterized SQL over the base tables.

One template per query type (semantic/templates): point, breakdown, trend and compare, sharing
one FROM / JOIN / WHERE block. Python builds only the fragments they place: SELECT expressions,
joins found on the many-to-one join graph, and WHERE conditions. Identifiers come from the
semantic model and every user-supplied value is a bound parameter.

The public building blocks (Params, render, conjoin, in_list, between, order_keys, compare_*)
are shared with layer 1 (okr.py).

Decisions (see the progress board): one template per query type; a metric's own conditions
go to WHERE when every metric of the query shares them, otherwise into conditional
aggregates; metrics on different tables compile to one query per table, merged by the
caller; a ratio is numerator / denominator and NULL when the denominator is 0.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from text2sql.semantic.ir import Code, QueryIR
from text2sql.semantic.registry import JoinEdge, Metric, SemanticModel
from text2sql.semantic.timeparse import baseline_range

_env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"),
                   undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


class CompileError(ValueError):
    pass


@dataclass
class CompiledQuery:
    sql: str
    params: dict[str, Any]
    layer: str  # "L1" or "L2"
    metrics: list[str]
    keys: list[str]  # period and dimension columns; results of several tables are merged on them
    columns: list[str]

    def display_sql(self) -> str:
        """SQL with parameters inlined, for display only; execution always binds them."""
        def literal(v: Any) -> str:
            if v is None:
                return "NULL"
            return str(v) if isinstance(v, (int, float)) else "'" + str(v).replace("'", "''") + "'"
        return re.sub(r":(p\d+)\b", lambda m: literal(self.params[m[1]]), self.sql)


class Params:
    """Allocates named parameters :p0, :p1 ... so that no value is ever written into the SQL text."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def add(self, value: Any) -> str:
        name = f"p{len(self.values)}"
        self.values[name] = value
        return f":{name}"


def render(query_type: str, **parts: Any) -> str:
    return _env.get_template(f"{query_type}.sql.j2").render(**parts).strip()


# ---------------------------------------------------------------- shared SQL fragments
def conjoin(*conds: str | None) -> str | None:
    """AND of the non-empty conditions; a condition containing OR is parenthesized."""
    parts = [c if " OR " not in c or c.startswith("(") else f"({c})" for c in conds if c]
    return " AND ".join(parts) or None


def in_list(column: str, codes: list[Code], p: Params) -> str:
    """column IN (...) with bound values; a None code adds IS NULL."""
    values = [c for c in codes if c is not None]
    parts = [f"{column} IN ({', '.join(p.add(v) for v in values)})"] if values else []
    if None in codes:
        parts.append(f"{column} IS NULL")
    return parts[0] if len(parts) == 1 else f"({' OR '.join(parts)})"


def between(column: str, rng: tuple[Any, Any], p: Params) -> str:
    return f"{column} BETWEEN {p.add(str(rng[0]))} AND {p.add(str(rng[1]))}"


def order_keys(ir: QueryIR, group_by: list[str], first_value: str) -> list[str]:
    """Ranked questions sort by the value (keys break ties); everything else by its keys."""
    if ir.order and ir.query_type != "trend":
        return [f"{first_value} {ir.order.upper()}", *group_by]
    return group_by


def compare_outer(keys: list[str], metrics: list[str]) -> list[str]:
    outer = list(keys)
    for m in metrics:
        cur, base = f"{m}_current", f"{m}_baseline"
        outer += [cur, base, f"{cur} - {base} AS {m}_change",
                  f"({cur} - {base}) * 1.0 / NULLIF({base}, 0) AS {m}_change_rate"]
    return outer


def compare_columns(keys: list[str], metrics: list[str]) -> list[str]:
    return keys + [f"{m}_{s}" for m in metrics for s in ("current", "baseline", "change", "change_rate")]


# ---------------------------------------------------------------- layer 2: base tables
def _agg(model: SemanticModel, metric: Metric, cond: str | None) -> str:
    """Aggregate of one metric, restricted to `cond` when given (conditional aggregation)."""
    if metric.agg == "ratio":
        num, den = model.metrics[metric.numerator], model.metrics[metric.denominator]
        n = _agg(model, num, conjoin(cond, *num.filters))
        d = _agg(model, den, conjoin(cond, *den.filters))
        return f"{n} * 1.0 / NULLIF({d}, 0)"
    if metric.agg == "count":
        return f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END)" if cond else "COUNT(*)"
    inner = f"CASE WHEN {cond} THEN {metric.column} END" if cond else metric.column
    if metric.agg == "sum":
        return f"COALESCE(SUM({inner}), 0)"
    return f"COUNT(DISTINCT {inner})"


def _joins(model: SemanticModel, base: str, dims: list[str]) -> list[JoinEdge]:
    """Union of the many-to-one paths from `base` to every dimension, without duplicates."""
    edges: OrderedDict[str, JoinEdge] = OrderedDict()
    for dim in dims:
        path = model.join_path(base, model.dimensions[dim].alias)
        if path is None:
            raise CompileError(f"维度 {dim} 无法从 {base} 沿多对一关系到达")
        for edge in path:
            edges.setdefault(edge.to, edge)
    return list(edges.values())


def compile_l2(ir: QueryIR, model: SemanticModel) -> list[CompiledQuery]:
    """One query per base table; the caller merges their results on the key columns."""
    tables: OrderedDict[str, list[Metric]] = OrderedDict()
    for key in ir.metrics:
        tables.setdefault(model.metrics[key].table, []).append(model.metrics[key])
    return [_compile_table(ir, model, base, metrics) for base, metrics in tables.items()]


def _compile_table(ir: QueryIR, model: SemanticModel, base: str, metrics: list[Metric]) -> CompiledQuery:
    p = Params()
    time_col = metrics[0].time_column
    names = [m.key for m in metrics]
    where = [in_list(model.dimensions[d].column, codes, p) for d, codes in ir.filters.items()]

    # A condition shared by every metric goes to WHERE; otherwise each metric filters itself.
    own = {m.key: () if m.agg == "ratio" else tuple(m.filters) for m in metrics}
    shared = len(set(own.values())) == 1
    if shared:
        where += next(iter(own.values()))

    def value(m: Metric, period: str | None = None) -> str:
        return _agg(model, m, conjoin(period, *([] if shared else own[m.key])))

    keys = [(model.time_grains[ir.grain].render(time_col), "period")] if ir.query_type == "trend" else []
    keys += [(model.dimensions[d].column, d) for d in ir.dimensions]
    select = [f"{expr} AS {alias}" for expr, alias in keys]
    group_by = [alias for _, alias in keys]
    parts = dict(base=base, joins=_joins(model, base, [*ir.dimensions, *ir.filters]), where=where,
                 group_by=group_by, limit=ir.limit)

    if ir.query_type == "compare":
        current = between(time_col, ir.time_range, p)
        baseline = between(time_col, baseline_range(*ir.time_range, ir.compare), p)
        where.append(f"({current} OR {baseline})")
        select += [f"{value(m, current)} AS {m.key}_current" for m in metrics]
        select += [f"{value(m, baseline)} AS {m.key}_baseline" for m in metrics]
        sql = render("compare", outer=compare_outer(group_by, names), select=select,
                     order_by=order_keys(ir, group_by, f"{names[0]}_current"), **parts)
        return CompiledQuery(sql, p.values, "L2", names, group_by, compare_columns(group_by, names))

    if ir.time_range:
        where.append(between(time_col, ir.time_range, p))
    select += [f"{value(m)} AS {m.key}" for m in metrics]
    sql = render(ir.query_type, select=select, order_by=order_keys(ir, group_by, names[0]), **parts)
    return CompiledQuery(sql, p.values, "L2", names, group_by, group_by + names)
