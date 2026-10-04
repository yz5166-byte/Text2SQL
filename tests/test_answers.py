"""Handlers after routing: clarification kinds, the model-written summary and its number check, answer display."""

import pytest

from text2sql.config import RISK_DB_PATH
from text2sql.pipeline.prompts import example_output
from text2sql.pipeline.summary import SUMMARY_MAX_ROWS, summarize, unsupported_numbers
from text2sql.pipeline.workflow import Text2SQLAgent

from scripted_llm import ScriptedLLM

needs_db = pytest.mark.skipif(not RISK_DB_PATH.exists(), reason="data/bank_risk.sqlite missing")
BY_REGION_1997 = example_output(dict(metrics=["bad_loan_rate"], time_text="97年", dimensions=["region"],
                                     query_type="breakdown", status="covered"))


def ask(question: str, *replies) -> object:
    return Text2SQLAgent(llm=ScriptedLLM(*replies)).ask(question)


# ---------------------------------------------------------------- clarification
@needs_db
def test_time_clarification_asks_for_a_time_not_a_metric():
    r = ask("风险事件数同比变化", example_output(dict(metrics=["risk_event_cnt"], query_type="compare", status="covered")))
    assert r.route == "CLARIFY" and r.clarify_options == []
    assert r.answer.startswith("需要确认：同比或环比需要明确的统计期间。请给出具体的时间范围")
    vague = Text2SQLAgent(llm=None).ask("最近风险事件数是多少")
    assert vague.answer.count("时间范围") == 1 and "指标" not in vague.answer


@needs_db
def test_metric_clarification_lists_candidates():
    r = Text2SQLAgent(llm=None).ask("1997年的交易量是多少？")
    assert [o["metric"] for o in r.clarify_options] == ["trans_cnt", "trans_amt"]
    assert "请选择要查询的指标" in r.answer


# ---------------------------------------------------------------- summary number check
ROWS = [["中波希米亚", "23.08%"], ["布拉格", "0.00%"], ["北摩拉维亚", "1,234"]]


@pytest.mark.parametrize("text, unsupported", [
    ("不良率最高为中波希米亚 23.08%，最低为布拉格 0.00%。", []),
    ("北摩拉维亚为 1234，问题所问为97年。", []),  # thousands separators ignored; numbers from the question allowed
    ("中波希米亚比布拉格高 23.08 个百分点，共 3 个地区。", ["3"]),  # a computed count is rejected
    ("中波希米亚为 0.2308。", ["0.2308"]),
])
def test_every_number_must_come_from_the_table(text, unsupported):
    assert unsupported_numbers(text, "97年各大区的不良率", ROWS) == unsupported


def test_summary_falls_back_without_a_model_or_on_large_tables():
    assert summarize(None, "q", ["地区"], [["布拉格"]]).text is None
    rows = [[str(i)] for i in range(SUMMARY_MAX_ROWS + 1)]
    assert summarize(ScriptedLLM("不会被调用"), "q", ["n"], rows).text is None


@needs_db
def test_checked_summary_becomes_the_first_sentence():
    r = ask("97年各大区的不良率", BY_REGION_1997, "不良率最高为中波希米亚 23.08%，最低为布拉格与北波希米亚 0.00%。")
    assert r.answer.startswith("不良率最高为中波希米亚 23.08%")
    assert "说明：首句总结由模型根据结果表生成" in r.answer and r.usage.calls == 2
    assert r.trace[-1].name == "summary" and r.trace[-1].detail == "通过数字校验"


@needs_db
def test_summary_with_a_computed_number_is_dropped():
    r = ask("97年各大区的不良率", BY_REGION_1997, "中波希米亚比布拉格高 23.08 个百分点，8 个地区中有 2 个为零。")
    assert r.answer.startswith("共 8 行结果，见下表。") and "首句总结" not in r.answer
    assert "不在结果表中" in r.trace[-1].detail and r.usage.calls == 2


# ---------------------------------------------------------------- display
@needs_db
def test_baseline_outside_the_data_shows_no_data():
    r = ask("1993年放款金额同比增长多少",
            example_output(dict(metrics=["loan_amt"], time_text="1993年", query_type="compare", compare="yoy",
                                status="covered")))
    assert "基期 无数据，变化率 无法计算" in r.answer and r.rows[0][1] == 0  # raw rows untouched for evaluation
    assert r.display_rows[0][1:] == ["无数据", "无数据", "无法计算"]
