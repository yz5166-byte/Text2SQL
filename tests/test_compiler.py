"""Compiler correctness against independently written gold SQL, and layer-1 / layer-2 consistency."""

import sqlite3

import pytest

from text2sql.config import RISK_DB_PATH
from text2sql.db.executor import SQLiteExecutor
from text2sql.eval.compare import lenient_ex
from text2sql.eval.dataset import load_domain_eval
from text2sql.pipeline.semantic import run_queries
from text2sql.semantic.compiler import compile_l2
from text2sql.semantic.okr import compile_ir, compile_l1, l1_ineligibility
from text2sql.semantic.ir import QueryIR
from text2sql.semantic.registry import load_semantic_model

pytestmark = pytest.mark.skipif(not RISK_DB_PATH.exists(), reason="data/bank_risk.sqlite missing")

model = load_semantic_model()
semantic_items = [i for i in load_domain_eval() if i.route in ("L1", "L2")]


@pytest.fixture(scope="module")
def ex():
    return SQLiteExecutor(RISK_DB_PATH, max_rows=5000)


def gold_rows(item):
    con = sqlite3.connect(f"file:{RISK_DB_PATH}?mode=ro", uri=True)
    rows = con.execute(item.gold_sql).fetchall()
    con.close()
    return rows


@pytest.mark.parametrize("item", semantic_items, ids=lambda i: i.id)
def test_layer2_matches_gold(item, ex):
    res = run_queries(ex, compile_l2(item.gold_ir, model))
    assert res.ok, res.error
    assert lenient_ex(res.rows, gold_rows(item), item.order_matters), (res.columns, res.rows[:5])


@pytest.mark.parametrize("item", semantic_items, ids=lambda i: i.id)
def test_routing_label_matches_l1_eligibility(item, ex):
    eligible = l1_ineligibility(item.gold_ir, model) is None
    assert eligible == (item.route == "L1"), l1_ineligibility(item.gold_ir, model)
    if eligible:
        res = run_queries(ex, [compile_l1(item.gold_ir, model)])
        assert res.ok, res.error
        assert lenient_ex(res.rows, gold_rows(item), item.order_matters), (res.columns, res.rows[:5])


@pytest.mark.parametrize("metric", [m.key for m in model.okr_metrics()])
@pytest.mark.parametrize("shape", [
    {"query_type": "trend", "grain": "month"},
    {"query_type": "trend", "grain": "month", "dimensions": ["region"]},
    {"query_type": "trend", "grain": "quarter"},
    {"query_type": "trend", "grain": "year", "dimensions": ["region"]},
    {"query_type": "compare", "compare": "yoy", "time_range": ["1997-01-01", "1997-12-31"]},
    {"query_type": "breakdown", "dimensions": ["region"], "time_range": ["1998-03-01", "1998-03-31"]},
])
def test_layer1_equals_layer2(metric, shape, ex):
    ir = QueryIR(metrics=[metric], **shape)
    if l1_ineligibility(ir, model):
        pytest.skip(l1_ineligibility(ir, model))
    l1 = run_queries(ex, [compile_l1(ir, model)])
    l2 = run_queries(ex, compile_l2(ir, model))
    assert l1.ok and l2.ok
    assert lenient_ex(l1.rows, l2.rows) and lenient_ex(l2.rows, l1.rows)


def test_multi_table_metrics_are_merged(ex):
    ir = QueryIR(metrics=["risk_event_cnt", "loan_cnt", "trans_cnt"], query_type="trend", grain="year",
                 time_range=["1997-01-01", "1998-12-31"])
    queries = compile_ir(ir, model)
    assert len(queries) == 3  # trans_cnt is not an OKR metric: layer 2, one query per base table
    res = run_queries(ex, queries)
    assert res.columns == ["period", "risk_event_cnt", "loan_cnt", "trans_cnt"]
    assert res.rows == [("1997", 2333, 196, 284409), ("1998", 3055, 158, 322277)]


def test_values_are_bound_not_inlined():
    ir = QueryIR(metrics=["trans_amt"], query_type="point", filters={"district_name": ["Hl.m. Praha' OR 1=1 --"]})
    (q,) = compile_l2(ir, model)
    assert "OR 1=1" not in q.sql and "OR 1=1" in str(q.params)
