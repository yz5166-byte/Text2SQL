"""Evaluation datasets: the self-built domain set and the BIRD financial subset."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from text2sql.config import BIRD_DB_PATH, EVAL_DIR, RAW_DIR
from text2sql.semantic.ir import QueryIR
from text2sql.semantic.registry import SemanticModel

Route = Literal["L1", "L2", "L3", "CLARIFY", "OOD", "META"]
DATA_ROUTES = ("L1", "L2", "L3")


class EvalItem(BaseModel):
    id: str
    route: Route
    question: str
    split: Literal["dev", "test"] = "test"
    gold_sql: str | None = None
    gold_ir: QueryIR | None = None
    alt_irs: list[QueryIR] = []
    order_matters: bool = False
    clarify_candidates: list[str] = []
    meta_target: str | None = None
    evidence: str | None = None  # BIRD external knowledge
    difficulty: str | None = None  # BIRD difficulty


def _assign_splits(items: list[EvalItem]) -> None:
    """Within each route, item i (1-based) goes to dev when i % 5 is 1 or 3 (about 40%)."""
    counters: dict[str, int] = defaultdict(int)
    for item in items:
        counters[item.route] += 1
        item.split = "dev" if counters[item.route] % 5 in (1, 3) else "test"


def load_domain_eval(path: Path = EVAL_DIR / "domain_eval.yaml") -> list[EvalItem]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)["items"]
    items = [EvalItem(**r) for r in raw]
    _assign_splits(items)
    return items


def load_bird_financial(path: Path = RAW_DIR / "bird_dev_20251106.json") -> list[EvalItem]:
    """BIRD dev (2025-11-06 cleaned release) questions for the financial database."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [
        EvalItem(
            id=f"BIRD-{d['question_id']}",
            route="L3",
            question=d["question"],
            gold_sql=d["SQL"],
            evidence=d.get("evidence") or None,
            difficulty=d.get("difficulty"),
        )
        for d in data
        if d["db_id"] == "financial"
    ]


def bird_tables() -> set[str]:
    """The 8 tables of BIRD's own financial database, without the derived ones (experiments X3 to X5)."""
    con = sqlite3.connect(f"file:{BIRD_DB_PATH}?mode=ro", uri=True)
    try:
        return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        con.close()


def validate_items(items: list[EvalItem], model: SemanticModel) -> list[str]:
    """Structural checks: ids, required fields, and IR references into the semantic layer."""
    errors: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            errors.append(f"{item.id}: duplicate id")
        seen.add(item.id)
        if item.route in DATA_ROUTES and not item.gold_sql:
            errors.append(f"{item.id}: missing gold_sql")
        if item.route in ("L1", "L2") and item.gold_ir is None:
            errors.append(f"{item.id}: missing gold_ir")
        if item.route == "META" and not item.meta_target:
            errors.append(f"{item.id}: missing meta_target")
        for cand in item.clarify_candidates:
            if cand not in model.metrics:
                errors.append(f"{item.id}: unknown clarify candidate {cand}")
        for ir in filter(None, [item.gold_ir, *item.alt_irs]):
            errors.extend(f"{item.id}: {e}" for e in ir_reference_errors(ir, model))
    return errors


def ir_reference_errors(ir: QueryIR, model: SemanticModel) -> list[str]:
    errors: list[str] = []
    tables = set()
    for metric in ir.metrics:
        if metric not in model.metrics:
            errors.append(f"unknown metric {metric}")
            continue
        tables.add(model.metrics[metric].table)
    reachable = set.intersection(*(set(model.reachable_dimensions(m)) for m in ir.metrics if m in model.metrics)) \
        if tables else set()
    for dim in [*ir.dimensions, *ir.filters]:
        if dim not in model.dimensions:
            errors.append(f"unknown dimension {dim}")
        elif dim not in reachable:
            errors.append(f"dimension {dim} not reachable from {sorted(ir.metrics)}")
    for dim, codes in ir.filters.items():
        if dim in model.dimensions:
            declared = {c for v in model.dimensions[dim].values for c in v.codes}
            for code in codes:
                if code not in declared:
                    errors.append(f"filter {dim}: unknown code {code!r}")
    if ir.query_type == "trend" and ir.grain is None:
        errors.append("trend requires grain")
    if ir.query_type == "compare" and (ir.compare is None or ir.time_range is None):
        errors.append("compare requires compare type and time_range")
    if ir.time_range and ir.time_range[0] > ir.time_range[1]:
        errors.append("time_range start after end")
    return errors
