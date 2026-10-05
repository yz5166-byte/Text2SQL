"""Evaluation records and metric computation (routing, slots, execution, clarification, cost)."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

from pydantic import BaseModel

from text2sql.eval.compare import bird_ex, lenient_ex
from text2sql.eval.dataset import DATA_ROUTES, EvalItem
from text2sql.semantic.ir import QueryIR

SLOT_FIELDS = ("metrics", "query_type", "time_range", "grain", "dimensions", "filters", "compare", "order_limit")


class Prediction(BaseModel):
    item_id: str
    route: str | None = None
    ir: QueryIR | None = None
    sql: str | None = None
    rows: list[list[Any]] | None = None
    error: str | None = None
    latency_s: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    attempts: int = 0  # layer-3 generation attempts
    clarify_candidates: list[str] = []
    answer: str | None = None
    trace: list[dict] = []  # every step of the request: name, detail, elapsed_ms


def slot_scores(pred: QueryIR, item: EvalItem) -> dict[str, bool]:
    """Per-slot correctness against the best-matching gold or alternative IR."""
    best: dict[str, bool] = {}
    for gold in filter(None, [item.gold_ir, *item.alt_irs]):
        pv, gv = pred.slot_view(), gold.slot_view()
        scores = {f: pv[f] == gv[f] for f in SLOT_FIELDS}
        if sum(scores.values()) > sum(best.values()):
            best = scores
    return best or {f: False for f in SLOT_FIELDS}


def _pct(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 3)
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(q * 100) - 1], 3)


def compute_metrics(
    items: list[EvalItem], preds: dict[str, Prediction], gold_rows: dict[str, list]
) -> dict[str, Any]:
    by_id = {i.id: i for i in items}
    out: dict[str, Any] = {"n_items": len(items)}

    # routing
    confusion = Counter((i.route, preds[i.id].route) for i in items)
    out["route_accuracy"] = _pct(sum(n for (g, p), n in confusion.items() if g == p), len(items))
    out["confusion"] = {f"{g}->{p}": n for (g, p), n in sorted(confusion.items(), key=str)}

    # execution accuracy on data questions
    ex_by_route: dict[str, list[bool]] = defaultdict(list)
    executable: list[bool] = []
    strict: list[bool] = []
    for item in items:
        if item.route not in DATA_ROUTES:
            continue
        pred = preds[item.id]
        ok = pred.rows is not None and lenient_ex(pred.rows, gold_rows[item.id], item.order_matters)
        ex_by_route[item.route].append(ok)
        executable.append(pred.rows is not None)
        strict.append(pred.rows is not None and bird_ex(pred.rows, gold_rows[item.id]))
    all_ex = [x for v in ex_by_route.values() for x in v]
    out["ex"] = _pct(sum(all_ex), len(all_ex))
    out["ex_strict"] = _pct(sum(strict), len(strict))
    out["ex_by_route"] = {r: _pct(sum(v), len(v)) for r, v in sorted(ex_by_route.items())}
    out["executable_rate"] = _pct(sum(executable), len(executable))

    # slot accuracy on semantic-layer questions that were parsed into an IR
    slot_hits: dict[str, list[bool]] = defaultdict(list)
    exact: list[bool] = []
    for item in items:
        if item.route not in ("L1", "L2"):
            continue
        pred = preds[item.id]
        scores = slot_scores(pred.ir, item) if pred.ir else {f: False for f in SLOT_FIELDS}
        for f, ok in scores.items():
            slot_hits[f].append(ok)
        exact.append(all(scores.values()))
    out["slot_accuracy"] = {f: _pct(sum(v), len(v)) for f, v in slot_hits.items()}
    out["ir_exact_match"] = _pct(sum(exact), len(exact))

    # clarification
    gold_c = {i.id for i in items if i.route == "CLARIFY"}
    pred_c = {pid for pid, p in preds.items() if p.route == "CLARIFY" and pid in by_id}
    out["clarify_precision"] = _pct(len(gold_c & pred_c), len(pred_c))
    out["clarify_recall"] = _pct(len(gold_c & pred_c), len(gold_c))

    # cost and latency per predicted route
    per_route: dict[str, list[Prediction]] = defaultdict(list)
    for item in items:
        per_route[preds[item.id].route or "NONE"].append(preds[item.id])
    out["by_pred_route"] = {
        r: {
            "n": len(ps),
            "latency_p50": _quantile([p.latency_s for p in ps], 0.5),
            "latency_p95": _quantile([p.latency_s for p in ps], 0.95),
            "llm_calls_mean": round(statistics.mean(p.llm_calls for p in ps), 2),
            "tokens_mean": round(statistics.mean(p.input_tokens + p.output_tokens for p in ps), 1),
        }
        for r, ps in sorted(per_route.items())
    }
    return out
