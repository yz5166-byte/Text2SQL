"""Layer 3: the example library, and the generate, guard, dry-run, execute and repair loop."""

import sqlite3
from datetime import date
from difflib import SequenceMatcher

import pytest

from text2sql.config import RISK_DB_PATH
from text2sql.db.executor import SQLiteExecutor
from text2sql.db.guard import guard_sql
from text2sql.eval.dataset import load_domain_eval
from text2sql.knowledge.examples import ExamplePicker, load_examples
from text2sql.knowledge.retriever import TableRetriever
from text2sql.pipeline.decision import RouteDecision
from text2sql.pipeline.layer3 import EMPTY_NOTE, GenerativeSQL, schema_text
from text2sql.pipeline.prompts import example_output
from text2sql.pipeline.workflow import Text2SQLAgent
from text2sql.semantic.registry import load_semantic_model

from scripted_llm import ScriptedLLM

pytestmark = pytest.mark.skipif(not RISK_DB_PATH.exists(), reason="data/bank_risk.sqlite missing")
model = load_semantic_model()
L3 = RouteDecision("L3", "llm", "语义层无法表达")


def layer3(*replies, max_repairs: int = 3) -> tuple[GenerativeSQL, ScriptedLLM]:
    llm = ScriptedLLM(*replies)
    gen = GenerativeSQL(llm, model, SQLiteExecutor(RISK_DB_PATH), retriever=TableRetriever(model, "bm25"),
                        examples=ExamplePicker(load_examples(), "bm25"), max_repairs=max_repairs)
    return gen, llm


def sql(text: str) -> dict:
    return {"sql": text, "explanation": "示例说明"}


# ---------------------------------------------------------------- the example library
def test_examples_run_pass_the_guard_and_differ_from_the_evaluation_set():
    questions = [i.question for i in load_domain_eval()]
    con = sqlite3.connect(f"file:{RISK_DB_PATH}?mode=ro", uri=True)
    for e in load_examples():
        assert con.execute(e.sql).fetchall(), e.id
        assert guard_sql(e.sql, model.tables, 1000).ok, e.id
        assert max(SequenceMatcher(None, e.question, q).ratio() for q in questions) < 0.57, e.id


def test_picker_returns_the_three_closest_examples():
    picked = ExamplePicker(load_examples(), "bm25").pick("没有代扣指令的账户数量")
    assert len(picked) == 3 and picked[0].id == "E07"


# ---------------------------------------------------------------- the loop
def test_first_attempt_succeeds_with_the_row_cap_and_explanation():
    gen, _ = layer3(sql("SELECT COUNT(*) FROM loan"))
    r = gen.answer("一共放了多少笔贷款", L3)
    assert r.ok and r.attempts == 1 and r.rows == [(682,)]
    assert r.sql.endswith("LIMIT 1001") and r.notes == ["口径说明：示例说明"]


def test_missing_column_is_repaired_with_the_full_schema():
    question = "每个地区的平均工资是多少"
    retrieved = TableRetriever(model, "bm25").retrieve(question).all
    absent = next(t for t in model.tables if t not in retrieved)
    gen, llm = layer3(sql("SELECT A3, AVG(salary) FROM district GROUP BY A3"),
                      sql("SELECT A3, AVG(A11) FROM district GROUP BY A3"))
    r = gen.answer(question, L3)
    assert r.ok and r.attempts == 2 and any(s.name == "widen" for s in r.trace)
    assert f"### {absent}（" not in llm.prompts[0] and f"### {absent}（" in llm.prompts[1]
    assert "上一次的 SQL" in llm.prompts[1] and "no such column: salary" in llm.prompts[1]


def test_a_write_is_refused_without_a_rewrite():
    gen, _ = layer3(sql("DELETE FROM loan"))
    r = gen.answer("删除全部贷款", L3)
    assert not r.ok and r.attempts == 1 and "只允许查询语句" in r.error


def test_guard_violation_is_rewritten():
    gen, llm = layer3(sql("SELECT name FROM sqlite_master"), sql("SELECT COUNT(*) FROM account"))
    r = gen.answer("一共有多少个账户", L3)
    assert r.ok and r.attempts == 2 and "不在数据字典中" in llm.prompts[1]


def test_empty_result_is_returned_with_a_note():
    gen, _ = layer3(sql("SELECT loan_id FROM loan WHERE amount < 0"))
    r = gen.answer("金额为负的贷款", L3)
    assert r.ok and r.attempts == 1 and r.rows == [] and EMPTY_NOTE in r.notes


def test_repairs_are_bounded():
    gen, _ = layer3(*[sql("SELEC 1")] * 3, max_repairs=2)
    r = gen.answer("随便查点什么", L3)
    assert not r.ok and r.attempts == 3 and r.error.startswith("修复 2 次后仍未成功：护栏拒绝：无法解析 SQL")


def test_malformed_output_is_retried():
    gen, llm = layer3(None, sql("SELECT COUNT(*) FROM card"))
    r = gen.answer("发了多少张卡", L3)
    assert r.ok and r.attempts == 2 and "输出不是包含 sql 与 explanation 的 JSON 对象" in llm.prompts[1]


def test_prompt_carries_definitions_time_and_evidence():
    gen, llm = layer3(sql("SELECT 1"))
    decision = RouteDecision("L3", "llm", "语义层无法表达", time_hint=(date(1997, 1, 1), date(1997, 12, 31)))
    gen.answer("1997年各卡等级持卡账户的不良率", decision, evidence="gold 指金卡")
    prompt = llm.prompts[0]
    assert "不良贷款笔数 ÷ 放款笔数" in prompt and "统计期间（已按参考日期换算）：1997-01-01 至 1997-12-31" in prompt
    assert "外部知识：gold 指金卡" in prompt and prompt.count("问题：") == 4  # three examples and the question


def test_prompt_carries_linked_values_and_every_code():
    gen, llm = layer3(sql("SELECT 1"))
    gen.answer("1998年各地区单笔1万及以上的现金取现交易，平均金额是多少？", L3)
    assert "- 现金取现 → trans.operation = 'VYBER'（交易方式）" in llm.prompts[0]
    assert "出账='VYDAJ'/'VYBER'" in schema_text(model, ["trans"])


# ---------------------------------------------------------------- through the agent
def test_agent_routes_an_uncovered_question_to_layer3():
    parse = example_output(dict(metrics=["loan_cnt"], status="uncovered", reason="本金阈值不是语义层维度"))
    agent = Text2SQLAgent(llm=ScriptedLLM(parse, sql("SELECT COUNT(*) FROM loan WHERE amount > 300000")))
    r = agent.ask("本金超过30万的贷款有几笔")
    assert r.route == "L3" and r.rows == [[82]] and r.usage.calls == 2
    assert "说明：口径说明：示例说明" in r.answer and "第三层" in r.answer
