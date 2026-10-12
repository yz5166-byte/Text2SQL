"""Experiment infrastructure: request cost, repeated-run summaries, and report sections."""

import pytest

from text2sql.eval.cost import PRICES, price_for, request_cost
import json

from text2sql.eval.repeats import comparison_table, linking_table, metric_table, stability, write_section


def test_request_cost_splits_cached_input():
    price = PRICES["deepseek-flash"]
    # 1M input of which 0.9M cached, plus 0.1M output: 0.1 * 0.30 + 0.9 * 0.006 + 0.1 * 1.20
    assert request_cost(price, 1_000_000, 900_000, 100_000) == pytest.approx(0.1554)


def test_price_follows_the_client_name():
    assert price_for("deepseek:deepseek-flash") is PRICES["deepseek-flash"]
    assert price_for("claude:claude-opus-5") is PRICES["claude-opus-5"]
    assert price_for("claude:claude-sonnet-5-5") is PRICES["claude-sonnet-5-5"]
    assert price_for(None) is None and price_for("deepseek:unknown") is None


def run(ex: float, per_item: dict) -> dict:
    return {"ex": ex, "per_item": per_item, "efficiency": {"cost_usd_mean": 0.001}}


def test_summary_reports_mean_and_range_and_stability():
    runs = [run(0.9, {"a": True, "b": False, "c": True}), run(1.0, {"a": True, "b": False, "c": False})]
    table = metric_table(runs)
    assert "| 执行准确率 EX | 95.0% | 90.0% | 100.0% |" in table and "第三层 EX" not in table
    assert stability(runs) == {"always": ["a"], "never": ["b"], "sometimes": ["c"]}


def test_write_section_replaces_only_between_markers(tmp_path):
    doc = tmp_path / "report.md"
    doc.write_text("intro\n<!-- X1:BEGIN -->\nold\n<!-- X1:END -->\noutro\n", encoding="utf-8")
    write_section(doc, "X1", "new table")
    assert doc.read_text(encoding="utf-8") == "intro\n<!-- X1:BEGIN -->\nnew table\n<!-- X1:END -->\noutro\n"
    with pytest.raises(ValueError):
        write_section(doc, "X9", "x")


def test_comparison_table_uses_only_the_given_items(tmp_path):
    for name, hits, sql in (("a_r1", [True, False, True], "S1"), ("a_r2", [True, False, True], "S2"),
                            ("b_r1", [False, False, True], "S1"), ("b_r2", [False, False, True], "S1")):
        (tmp_path / name).mkdir()
        per_item = dict(zip(["x", "y", "z"], hits))
        (tmp_path / name / "metrics.json").write_text(json.dumps({"per_item": per_item}), encoding="utf-8")
        preds = [{"item_id": i, "sql": sql, "rows": [[1]], "latency_s": 1.0, "llm_calls": 1, "cost_usd": 0.001}
                 for i in ("x", "y", "z")]
        (tmp_path / name / "predictions.jsonl").write_text("\n".join(map(json.dumps, preds)), encoding="utf-8")
    table = comparison_table({"A": ["a_r1", "a_r2"], "B": ["b_r1", "b_r2"]}, {"x": "L1", "y": "L3"},
                             format_only=frozenset({"y"}), runs_dir=tmp_path)
    assert "| A | 50.0%（50.0% 至 50.0%） | 100.0% | 100.0% | 0.0% | 0/2 | 2/2 |" in table and "第一层题" in table
    assert "| B | 0.0%（0.0% 至 0.0%） | 50.0% | 0.0% | 0.0% | 2/2 | 2/2 |" in table
    single = comparison_table({"A": ["a_r1"]}, {"x": "L1", "y": "L3"}, runs_dir=tmp_path)
    assert "| A | 50.0%（单轮） | 100.0% | 0.0% | 单轮不适用 | 单轮不适用 |" in single


def test_bird_restricted_model_is_consistent():
    from text2sql.pipeline.layer3 import schema_text
    from text2sql.semantic.registry import load_semantic_model

    base = {"account", "card", "client", "disp", "district", "loan", "order", "trans"}
    m = load_semantic_model().restricted_to(base)
    assert set(m.tables) == base and m.validate_references() == []
    assert "risk_event_cnt" not in m.metrics and "risk_rule" not in m.dimensions and "loan_amt" in m.metrics
    assert "### trans（" in schema_text(m, list(m.tables))  # renders without the derived tables


def test_failure_table_sorts_answers_by_kind(tmp_path):
    from text2sql.eval.repeats import failure_table

    (tmp_path / "a_r1").mkdir()
    per_item = {"ok": True, "err": False, "empty": False, "cols": False, "vals": False}
    (tmp_path / "a_r1" / "metrics.json").write_text(json.dumps({"per_item": per_item}), encoding="utf-8")
    rows = {"ok": [[1]], "err": None, "empty": [], "cols": [[1, 2]], "vals": [[9]]}
    preds = [{"item_id": i, "rows": r} for i, r in rows.items()]
    (tmp_path / "a_r1" / "predictions.jsonl").write_text("\n".join(map(json.dumps, preds)), encoding="utf-8")
    gold = {i: [[1]] for i in rows}
    assert "| A | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 |" in failure_table({"A": ["a_r1"]}, gold, runs_dir=tmp_path)


def test_coverage_marks_questions_whose_tables_were_all_retrieved():
    from types import SimpleNamespace

    from text2sql.eval.retrieval import coverage

    class Stub:
        graph = {"loan": [], "client": [], "district": []}

        def retrieve(self, text):
            return SimpleNamespace(all=["loan", "client"])

    cases = {"a": ("q", "SELECT * FROM loan JOIN client USING (client_id)"),
             "b": ("q", "WITH x AS (SELECT * FROM district) SELECT * FROM x JOIN loan USING (district_id)")}
    assert coverage(Stub(), cases) == {"a": True, "b": False}


def test_linking_table_compares_the_context_with_the_gold_needs(tmp_path):
    preds = [{"item_id": "x", "context_tables": ["loan", "account"], "context_columns": ["loan.amount", "loan.loan_id"]},
             {"item_id": "y", "context_tables": ["loan"], "context_columns": None}]
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "predictions.jsonl").write_text("\n".join(map(json.dumps, preds)), encoding="utf-8")
    gold = {"x": ({"loan"}, {"loan.amount", "loan.status"}), "y": ({"loan", "client"}, {"loan.status"})}
    columns_of = {"loan": ["loan.loan_id", "loan.amount", "loan.status"]}
    table = linking_table({"A": ["r"]}, gold, columns_of, runs_dir=tmp_path)
    # x: tables complete, 1 of 2 columns kept (pruned); y: client missing, status kept (not pruned)
    assert "| A | 50.0% | 66.7% | 1.5 | 2.5 |" in table
