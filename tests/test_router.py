"""Whole routing (rules, parser, normalization, data range) and the agent on rule-decided questions."""

import pytest

from text2sql.config import RISK_DB_PATH
from text2sql.llm.base import LLMError
from text2sql.pipeline.router import Router
from text2sql.pipeline.workflow import Text2SQLAgent
from text2sql.semantic.registry import load_semantic_model

model = load_semantic_model()
router = Router(model, llm=None)


@pytest.mark.parametrize("q, route", [
    ("上月不良率多少", "L1"),
    ("请问1996年放款笔数一共是多少", "L1"),
    ("1998年12月入账金额", "L2"),
    ("中波希米亚1997年的风险涉及金额是多少？", "L1"),
    ("出账金额的计算公式是什么", "META"),
    ("帮我订一张机票", "OOD"),
    ("1998年损失多少", "CLARIFY"),
    ("最近风险事件数是多少", "CLARIFY"),
])
def test_rules_and_fast_path(q, route):
    assert router.route(q).route == route


def test_complex_questions_need_the_model():
    with pytest.raises(LLMError):
        router.route("1998年各月风险事件数，按规则拆分")


def test_fast_path_answers_are_fitted_to_the_data_range():
    assert router.route("1999年的放款金额是多少？").route == "CLARIFY"
    d = router.route("1990年到1994年的放款笔数")
    assert d.route == "L1" and any("实际统计 1993-01-01 至 1994-12-31" in a for a in d.assumptions)


@pytest.mark.skipif(not RISK_DB_PATH.exists(), reason="data/bank_risk.sqlite missing")
def test_agent_answers_without_model_for_rule_decided_questions():
    agent = Text2SQLAgent(llm=None)
    r = agent.ask("1998年11月的风险事件数是多少？")
    assert r.route == "L1" and r.rows == [[265]] and r.usage.calls == 0
    assert "OKR" in r.answer and "ads_okr_monthly" in r.sql[0]
    assert "VYBER" in agent.ask("出账金额包含哪些交易类型？").answer
    assert "30,000" in agent.ask("快进快出规则是什么意思？").answer
    assert "trans" in agent.ask("哪张表记录了交易对手的信息？").answer
    assert "不良贷款笔数 ÷ 放款笔数" in agent.ask("不良率是怎么计算的？").answer
    clar = agent.ask("上个月的损失有多少？")
    assert clar.route == "CLARIFY" and [o["metric"] for o in clar.clarify_options] == ["bad_loan_amt", "risk_amt"]
    follow = agent.resolve_clarification("上个月的损失有多少？", "risk_amt")
    assert follow.route == "L1" and follow.rows == [[9609493]]
