"""LLM structured parsing: output validation, retry with errors, degradation, and the prompt examples."""

from difflib import SequenceMatcher

import pytest
from pydantic import ValidationError

from text2sql.eval.dataset import load_domain_eval
from text2sql.llm.base import extract_json
from text2sql.pipeline.normalize import normalize_parse
from text2sql.pipeline.parser import ParseResult
from text2sql.pipeline.prompts import EXAMPLES, example_output, parse_schema
from text2sql.pipeline.router import Router
from text2sql.semantic.registry import load_semantic_model

from scripted_llm import ScriptedLLM

model = load_semantic_model()
QUESTION = "1998年各月风险事件数，按规则拆分"  # passes the rules, so it reaches the model
VALID = example_output(dict(metrics=["risk_event_cnt"], time_text="1998年", dimensions=["risk_rule"],
                            query_type="trend", grain="month", status="covered"))


def validate(**slots) -> ParseResult:
    return ParseResult.model_validate(example_output(slots), context={"model": model})


@pytest.mark.parametrize("slots, loc, message", [
    (dict(metrics=["loan_balance"], status="covered"), ("metrics",), "不在可用指标中"),
    (dict(metrics=["loan_amt"], dimensions=["province"], status="covered"), ("dimensions",), "不在可用维度中"),
    (dict(metrics=["loan_amt"], filters=[{"dimension": "province", "values": ["南摩拉维亚"]}], status="covered"),
     ("filters", 0, "dimension"), "不在可用维度中"),
    (dict(metrics=["loan_amt"], query_type="point", limit=-3, status="covered"), ("limit",), "greater than or equal to 1"),
    (dict(metrics=["loan_amt"], query_type="point"), (), "必须给出 status"),
    (dict(metrics=["loan_amt"], status="covered"), (), "必须给出 query_type"),
])
def test_invalid_outputs_are_rejected(slots, loc, message):
    with pytest.raises(ValidationError) as exc:
        validate(**slots)
    err = exc.value.errors()[0]
    assert err["loc"] == loc and message in err["msg"]


def test_metadata_and_out_of_domain_need_no_status():
    assert validate(intent="metadata").status is None


def test_schema_and_parse_result_share_field_order():
    assert list(parse_schema(model)["properties"]) == list(ParseResult.model_fields)


def test_retry_carries_readable_errors():
    llm = ScriptedLLM({**VALID, "metrics": ["loan_balance"]}, VALID)
    decision = Router(model, llm).route(QUESTION)
    assert decision.route == "L2" and decision.usage.calls == 2
    assert "1. 字段 metrics：['loan_balance'] 不在可用指标中" in llm.prompts[1]


def test_two_invalid_replies_degrade_to_layer3():
    llm = ScriptedLLM(None, {**VALID, "limit": 0})
    decision = Router(model, llm).route(QUESTION)
    assert decision.route == "L3" and decision.usage.calls == 2
    assert decision.time_hint is None  # without a parse there is no model extraction to hint the period
    assert "输出不是合法的 JSON 对象" in llm.prompts[1]


def test_text_without_a_json_object_yields_none():
    assert extract_json("抱歉，我无法解析") is None
    assert extract_json('```json\n{"intent": "metadata"}\n```') == {"intent": "metadata"}


EXPECTED_ROUTES = {"covered": {"L1", "L2"}, "uncovered": {"L3"}, "clarify": {"CLARIFY"}}


@pytest.mark.parametrize("question, slots", EXAMPLES)
def test_examples_are_valid_and_route_by_status(question, slots):
    parse = validate(**slots)
    assert normalize_parse(question, parse, model).route in EXPECTED_ROUTES[parse.status]


MAX_SIMILARITY = 0.57  # the original four examples peak at 0.5625; added ones may not exceed them


def test_examples_do_not_resemble_evaluation_questions():
    questions = [item.question for item in load_domain_eval()]
    for example, _ in EXAMPLES:
        closest = max(SequenceMatcher(None, example, q).ratio() for q in questions)
        assert closest < MAX_SIMILARITY, example
