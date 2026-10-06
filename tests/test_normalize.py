"""Deterministic normalization of parse results into a route and a QueryIR (no model calls)."""

from datetime import date

from text2sql.pipeline.decision import RouteDecision
from text2sql.pipeline.normalize import fit_to_data_range, normalize_parse
from text2sql.pipeline.parser import ParseResult
from text2sql.semantic.registry import load_semantic_model

model = load_semantic_model()


def normalized(question: str, **slots) -> RouteDecision:
    parse = ParseResult.model_validate({"intent": "data_query", "status": "covered", **slots}, context={"model": model})
    return normalize_parse(question, parse, model)


# ---------------------------------------------------------------- time: the question is authoritative when covered
def test_question_time_corrects_an_omitted_reworded_or_truncated_extraction():
    for time_text in (None, "1998-11", "1998年"):
        d = normalized("1998年11月风险事件数", metrics=["risk_event_cnt"], query_type="point", time_text=time_text)
        assert d.ir.time_range == (date(1998, 11, 1), date(1998, 11, 30))


def test_model_chooses_among_several_time_expressions():
    q = "1998年11月风险事件数，与1998年10月相比"
    d = normalized(q, metrics=["risk_event_cnt"], query_type="compare", compare="mom", time_text="1998年11月")
    assert d.ir.time_range == (date(1998, 11, 1), date(1998, 11, 30))
    d = normalized(q, metrics=["risk_event_cnt"], query_type="compare", compare="mom")
    assert d.route == "CLARIFY" and "多个时间表达" in d.reason


def test_time_the_question_does_not_contain_is_not_trusted():
    d = normalized("风险事件数是多少", metrics=["risk_event_cnt"], query_type="point", time_text="1998年")
    assert d.route == "CLARIFY" and "无法解析时间表达" in d.reason


def test_uncovered_question_uses_only_the_model_extraction():
    d = normalized("1998年余额为负的账户数", status="uncovered", time_text="1998年")
    assert d.route == "L3" and d.time_hint == (date(1998, 1, 1), date(1998, 12, 31))
    d = normalized("账户所有人在1970年及以后出生的贷款有多少笔？", status="uncovered", metrics=["loan_cnt"],
                   reason="出生年份不是语义层维度")
    assert d.time_hint is None and d.reason.endswith("出生年份不是语义层维度")


# ---------------------------------------------------------------- defaults
def test_trend_grain_comes_from_the_question_else_month_with_a_note():
    d = normalized("1995年到1997年每年的放款笔数", metrics=["loan_cnt"], query_type="trend")
    assert d.ir.grain == "year" and d.assumptions == []
    d = normalized("1998年风险事件数趋势", metrics=["risk_event_cnt"], query_type="trend")
    assert d.ir.grain == "month" and "未指定趋势粒度，按月展示" in d.assumptions and d.route == "L1"


def test_point_with_dimensions_becomes_breakdown():
    d = normalized("1998年各规则事件数", metrics=["risk_event_cnt"], query_type="point", time_text="1998年",
                   dimensions=["risk_rule"])
    assert d.ir.query_type == "breakdown" and d.route == "L2"


def test_clarification_paths():
    assert normalized("看一下数据", status="clarify").route == "CLARIFY"
    d = normalized("风险事件数同比", metrics=["risk_event_cnt"], query_type="compare")
    assert d.route == "CLARIFY" and d.clarify_kind == "time" and d.clarify_options == []


# ---------------------------------------------------------------- reachability and value linking
def test_unreachable_dimension_degrades_to_layer3():
    d = normalized("各卡等级的放款金额", metrics=["loan_amt"], query_type="breakdown", dimensions=["card_type"])
    assert d.route == "L3"


def test_misplaced_filter_is_repaired():
    d = normalized("1997年12月透支罚息交易笔数", metrics=["trans_cnt"], query_type="point", time_text="1997年12月",
                   filters=[{"dimension": "trans_operation", "values": ["透支罚息"]}])
    assert d.route == "L2" and d.ir.filters == {"trans_purpose": ["SANKC. UROK"]}
    assert any("归入维度" in a for a in d.assumptions)


def test_fuzzy_value_match_is_disclosed():
    d = normalized("1998年Plzen jh的交易金额", metrics=["trans_amt"], query_type="point", time_text="1998年",
                   filters=[{"dimension": "district_name", "values": ["Plzen jh"]}])
    assert d.ir.filters == {"district_name": ["Plzen - jih"]}
    assert "筛选值“Plzen jh”按近似匹配为“Plzen jih”" in d.assumptions


# ---------------------------------------------------------------- data range
def test_period_outside_the_data_is_not_queried():
    d = fit_to_data_range(normalized("1999年放款金额", metrics=["loan_amt"], query_type="point"), model)
    assert d.route == "CLARIFY" and d.ir is None and "不在数据范围" in d.reason


def test_partly_covered_period_is_clipped_and_disclosed():
    d = fit_to_data_range(normalized("1990年到1994年每年的放款笔数", metrics=["loan_cnt"], query_type="trend"), model)
    assert d.ir.time_range == (date(1993, 1, 1), date(1994, 12, 31)) and d.route == "L1"
    assert "实际统计 1993-01-01 至 1994-12-31" in d.assumptions[-1]


def test_baseline_outside_the_data_is_disclosed():
    d = fit_to_data_range(normalized("1993年放款金额同比", metrics=["loan_amt"], query_type="compare", compare="yoy"),
                          model)
    assert d.route == "L1" and "基期 1992-01-01 至 1992-12-31 不在数据范围内" in d.assumptions[-1]


# ---------------------------------------------------------------- contradictions found in the M4-4 dev run
def test_every_conditioned_metric_has_a_base_metric():
    assert all(model.base_metric(k) for k, m in model.metrics.items() if m.filters)


def test_metric_condition_contradicting_a_filter_is_clarified():
    d = normalized("1997年信用卡取现的笔数和金额", metrics=["cash_withdraw_cnt", "cash_withdraw_amt"], query_type="point",
                   time_text="1997年", filters=[{"dimension": "trans_operation", "values": ["信用卡取现"]}])
    assert d.route == "CLARIFY" and d.clarify_kind == "metric" and d.clarify_options == ["trans_cnt", "trans_amt"]
    d = normalized("1997年南摩拉维亚现金取现笔数", metrics=["cash_withdraw_cnt"], query_type="point", time_text="1997年",
                   filters=[{"dimension": "region", "values": ["南摩拉维亚"]}])
    assert d.route == "L2"  # a filter on another column is no contradiction


def test_period_by_period_comparison_goes_to_layer3():
    d = normalized("1997年各月现金取现金额环比", metrics=["cash_withdraw_amt"], query_type="compare", compare="mom",
                   time_text="1997年")
    assert d.route == "L3" and "逐期对比" in d.reason


# ---------------------------------------------------------------- found in the X1 test run (after-the-fact fixes)
def test_tables_outside_the_semantic_layer_go_to_layer3():
    assert set(model.uncovered_tables()) >= {"order", "disp"}
    d = normalized("代扣指令中用途为家庭缴费的指令有几条", metrics=["trans_cnt"], query_type="point",
                   filters=[{"dimension": "trans_purpose", "values": ["家庭缴费"]}])
    assert d.route == "L3" and "代扣指令表" in d.reason
