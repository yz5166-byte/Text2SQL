"""Gold replay: feeding gold answers as predictions must score 100% on every metric."""

import sqlite3

import pytest

from text2sql.config import RISK_DB_PATH
from text2sql.eval.dataset import DATA_ROUTES, load_domain_eval, validate_items
from text2sql.eval.metrics import Prediction, compute_metrics
from text2sql.semantic.registry import load_semantic_model

items = load_domain_eval()


def test_domain_eval_is_valid():
    assert validate_items(items, load_semantic_model()) == []
    assert len(items) >= 100
    assert {i.split for i in items} == {"dev", "test"}


@pytest.mark.skipif(not RISK_DB_PATH.exists(), reason="data/bank_risk.sqlite missing")
def test_gold_replay_scores_perfectly():
    con = sqlite3.connect(f"file:{RISK_DB_PATH}?mode=ro", uri=True)
    gold_rows = {i.id: [list(r) for r in con.execute(i.gold_sql)] for i in items if i.route in DATA_ROUTES}
    con.close()
    preds = {
        i.id: Prediction(item_id=i.id, route=i.route, ir=i.gold_ir, rows=gold_rows.get(i.id))
        for i in items
    }
    m = compute_metrics(items, preds, gold_rows)
    assert m["route_accuracy"] == 1.0
    assert m["ex"] == 1.0 and m["ex_strict"] == 1.0
    assert m["ir_exact_match"] == 1.0
    assert m["clarify_precision"] == 1.0 and m["clarify_recall"] == 1.0
