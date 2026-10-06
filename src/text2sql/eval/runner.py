"""Run an evaluation: predictions for every item, gold result sets, metrics, and run artifacts."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from text2sql.config import RUNS_DIR
from text2sql.eval.cost import price_for, request_cost
from text2sql.eval.dataset import DATA_ROUTES, EvalItem
from text2sql.eval.metrics import Prediction, compute_metrics
from text2sql.pipeline.decision import RouteDecision
from text2sql.pipeline.layer3 import GenerativeSQL
from text2sql.pipeline.semantic import run_queries
from text2sql.pipeline.workflow import Text2SQLAgent
from text2sql.semantic.okr import compile_ir, l1_ineligibility

Predictor = Callable[[EvalItem], Prediction]


def gold_result_sets(items: list[EvalItem], db_path: Path) -> dict[str, list]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {i.id: [list(r) for r in con.execute(i.gold_sql)] for i in items if i.route in DATA_ROUTES}
    finally:
        con.close()


def agent_predictor(agent: Text2SQLAgent) -> Predictor:
    price = price_for(getattr(agent.llm, "name", None))

    def predict(item: EvalItem) -> Prediction:
        r = agent.ask(item.question)
        ok_rows = r.rows if r.route in DATA_ROUTES and r.error is None else None
        u = r.usage
        return Prediction(
            item_id=item.id, route=r.route, source=r.source, ir=r.ir, sql="\n;\n".join(r.sql) or None, rows=ok_rows,
            error=r.error, latency_s=r.latency_s, llm_calls=u.calls, input_tokens=u.input_tokens,
            cached_input_tokens=u.cached_input_tokens, output_tokens=u.output_tokens,
            cost_usd=request_cost(price, u.input_tokens, u.cached_input_tokens, u.output_tokens) if price else 0.0,
            attempts=r.attempts,
            clarify_candidates=[o["metric"] for o in r.clarify_options], answer=r.answer,
            trace=[asdict(step) for step in r.trace],
        )
    return predict


def generation_predictor(layer3: GenerativeSQL) -> Predictor:
    """Every question straight to layer 3 (experiment X2): no rules, no semantic layer, no router."""
    price = price_for(layer3.llm.name)

    def predict(item: EvalItem) -> Prediction:
        t0 = time.perf_counter()
        r = layer3.answer(item.question, RouteDecision("L3", "llm", "直接进入第三层"), item.evidence)
        u = r.usage
        return Prediction(
            item_id=item.id, route="L3", source="llm", sql=r.sql, rows=[list(x) for x in r.rows] if r.ok else None,
            error=r.error, latency_s=time.perf_counter() - t0, llm_calls=u.calls, input_tokens=u.input_tokens,
            cached_input_tokens=u.cached_input_tokens, output_tokens=u.output_tokens,
            cost_usd=request_cost(price, u.input_tokens, u.cached_input_tokens, u.output_tokens) if price else 0.0,
            attempts=r.attempts, trace=[asdict(step) for step in r.trace],
        )
    return predict


def oracle_ir_predictor(agent: Text2SQLAgent) -> Predictor:
    """Decoupled evaluation of layers 1 and 2: feed the gold IR, measure compile + execute only."""
    def predict(item: EvalItem) -> Prediction:
        t0 = time.perf_counter()
        queries = compile_ir(item.gold_ir, agent.model)
        res = run_queries(agent.executor, queries)
        route = "L1" if l1_ineligibility(item.gold_ir, agent.model) is None else "L2"
        return Prediction(item_id=item.id, route=route, ir=item.gold_ir, sql=queries[0].display_sql(),
                          rows=[list(r) for r in res.rows] if res.ok else None, error=res.error,
                          latency_s=time.perf_counter() - t0)
    return predict


def run(items: list[EvalItem], predictor: Predictor, gold_db: Path, name: str, meta: dict,
        progress: bool = True) -> dict:
    out_dir = RUNS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    gold = gold_result_sets(items, gold_db)
    preds: dict[str, Prediction] = {}
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for n, item in enumerate(items, 1):
            try:
                pred = predictor(item)
            except Exception as exc:  # one failing item must not abort the whole run
                pred = Prediction(item_id=item.id, error=f"{type(exc).__name__}: {exc}")
            preds[item.id] = pred
            f.write(pred.model_dump_json() + "\n")
            if progress:
                print(f"[{n}/{len(items)}] {item.id} route={pred.route} err={bool(pred.error)}", flush=True)
    metrics = compute_metrics(items, preds, gold)
    metrics["meta"] = meta
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")
    return metrics
