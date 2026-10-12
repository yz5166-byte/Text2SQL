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
from text2sql.llm.base import LLMError
from text2sql.pipeline.decision import RouteDecision
from text2sql.pipeline.layer3 import EMPTY_NOTE, NAIVE, GenerativeSQL, Layer3Options, schema_text
from text2sql.pipeline.prompts import example_output
from text2sql.pipeline.workflow import Text2SQLAgent
from text2sql.semantic.registry import load_semantic_model

from scripted_llm import ScriptedLLM

pytestmark = pytest.mark.skipif(not RISK_DB_PATH.exists(), reason="data/bank_risk.sqlite missing")
model = load_semantic_model()
L3 = RouteDecision("L3", "llm", "语义层无法表达")


def layer3(*replies, options: Layer3Options = Layer3Options(max_repairs=3),
           retriever: TableRetriever | None = None) -> tuple[GenerativeSQL, ScriptedLLM]:
    llm = ScriptedLLM(*replies)
    gen = GenerativeSQL(llm, model, SQLiteExecutor(RISK_DB_PATH), options,
                        retriever=retriever or TableRetriever(model, "bm25"), examples=ExamplePicker(load_examples(), "bm25"))
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
    gen, _ = layer3(*[sql("SELEC 1")] * 3, options=Layer3Options(max_repairs=2))
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



def test_column_linking_adds_value_tables_and_names_their_columns():
    options = Layer3Options(retrieval="bm25", column_linking=True)
    gen, llm = layer3(sql("SELECT 1"), options=options, retriever=TableRetriever(model, "bm25", k=1))
    r = gen.answer("How many loans went to accounts in Hl.m. Praha?", L3)
    prompt = llm.prompts[0]
    assert "### district（" in prompt and "- Hl.m. Praha → district.A2 中的取值 'Hl.m. Praha'" in prompt
    assert "链接命中 ['district']" in r.trace[0].detail
    assert r.trace[1].name == "link" and "district.A2" in r.trace[1].detail


def test_column_linking_is_off_by_default():
    gen, llm = layer3(sql("SELECT 1"))
    r = gen.answer("How many loans went to accounts in Hl.m. Praha?", L3)
    assert "数据库取值" not in llm.prompts[0] and [s.name for s in r.trace][:2] == ["retrieve", "generate 1"]


# ---------------------------------------------------------------- table selection
SELECT = Layer3Options(retrieval="bm25", table_selection=True)
FULL_SELECT = Layer3Options(schema="full", table_selection=True)


def tables_in(prompt: str) -> list[str]:
    return [t for t in model.tables if f"### {t}（" in prompt]


def test_selection_keeps_the_chosen_tables_for_generation():
    gen, llm = layer3({"tables": ["LOAN"], "reason": "月还款额与还款状态都在贷款表"},
                      sql("SELECT COUNT(*) FROM loan WHERE payments > 5000 AND status = 'D'"), options=SELECT)
    r = gen.answer("月还款额超过5000且处于履约中欠款状态的贷款有多少笔？", L3)
    assert "## 候选表\n- " in llm.prompts[0] and "- loan（贷款表）：" in llm.prompts[0]
    assert tables_in(llm.prompts[1]) == ["loan"] and r.ok and r.usage.calls == 2
    step = next(s for s in r.trace if s.name == "select")
    assert step.detail.startswith("保留 ['loan']，连接补全 []，删去 [") and "理由：月还款额" in step.detail


def test_selection_gets_the_join_bridge_back():
    gen, llm = layer3({"tables": ["loan", "client"], "reason": "客户与贷款"}, sql("SELECT 1"), options=FULL_SELECT)
    r = gen.answer("女性客户的贷款总额", L3)
    # two shortest paths of three hops: client - disp - account - loan and client - district - account - loan
    assert tables_in(llm.prompts[1]) == ["account", "client", "disp", "loan", "district"]
    assert "保留 ['client', 'loan']，连接补全 ['account', 'disp', 'district']" in r.trace[-2].detail


@pytest.mark.parametrize("reply", [{"tables": ["no_such_table"], "reason": ""}, {"tables": [], "reason": ""},
                                   None, LLMError("connection reset")])
def test_a_useless_or_failed_selection_keeps_the_candidates(reply):
    gen, llm = layer3(reply, sql("SELECT 1"), options=FULL_SELECT)
    r = gen.answer("女性客户的贷款总额", L3)
    assert tables_in(llm.prompts[-1]) == list(model.tables) and r.ok
    assert "沿用候选表" in next(s for s in r.trace if s.name == "select").detail


def test_selection_sees_the_linking_hints():
    options = Layer3Options(retrieval="bm25", column_linking=True, table_selection=True)
    gen, llm = layer3({"tables": ["loan"], "reason": ""}, sql("SELECT 1"), options=options)
    gen.answer("How many loans went to accounts in Hl.m. Praha?", L3)
    assert "## 问题中出现的数据库取值\n- Hl.m. Praha → district.A2" in llm.prompts[0]
    assert "## 与问题最相关的字段（检索排序）\n" in llm.prompts[0]


# ---------------------------------------------------------------- column pruning
PRUNE = Layer3Options(retrieval="bm25", column_pruning=True)
L3_11 = "月还款额超过5000且处于履约中欠款状态的贷款有多少笔？"


def columns_in(prompt: str, table: str) -> list[str]:
    """Columns listed under a table's header in the generation prompt."""
    columns = []
    for line in prompt.split(f"### {table}（", 1)[1].split("\n")[1:]:
        if not line.startswith("- "):
            break
        columns.append(line[2:].split(" ")[0])
    return columns


def test_pruning_keeps_the_choice_keys_and_value_columns():
    gen, llm = layer3({"columns": ["LOAN.PAYMENTS", "no_such.column"], "reason": "月还款额与状态"}, sql("SELECT 1"),
                      options=PRUNE)
    r = gen.answer(L3_11, L3)
    assert "## 可用的表\n### " in llm.prompts[0] and "- duration INTEGER 贷款期限" in llm.prompts[0]
    assert columns_in(llm.prompts[1], "loan") == ["loan_id", "account_id", "payments", "status"]  # status: linked value
    others = [t for t in tables_in(llm.prompts[1]) if t != "loan"]
    assert others and all(f"{t}.{c}" in gen.keys for t in others for c in columns_in(llm.prompts[1], t))
    step = next(s for s in r.trace if s.name == "prune")
    assert "模型选择 ['loan.payments']" in step.detail and "理由：月还款额与状态" in step.detail


def test_pruning_works_on_the_selected_tables():
    options = Layer3Options(retrieval="bm25", table_selection=True, column_pruning=True)
    gen, llm = layer3({"tables": ["loan"], "reason": ""}, {"columns": ["loan.payments"], "reason": ""}, sql("SELECT 1"),
                      options=options)
    r = gen.answer(L3_11, L3)
    assert tables_in(llm.prompts[1]) == tables_in(llm.prompts[2]) == ["loan"]
    assert columns_in(llm.prompts[2], "loan") == ["loan_id", "account_id", "payments", "status"]
    assert r.usage.calls == 3


@pytest.mark.parametrize("reply", [{"columns": ["loan.no_such"], "reason": ""}, {"columns": [], "reason": ""},
                                   None, LLMError("connection reset")])
def test_a_useless_or_failed_pruning_keeps_every_column(reply):
    gen, llm = layer3(reply, sql("SELECT 1"), options=PRUNE)
    r = gen.answer(L3_11, L3)
    assert columns_in(llm.prompts[-1], "loan") == list(model.tables["loan"].columns)
    assert "保留全部字段" in next(s for s in r.trace if s.name == "prune").detail


def test_a_column_error_drops_the_pruning_with_the_widening():
    gen, llm = layer3({"columns": ["loan.payments"], "reason": ""}, sql("SELECT monthly_payment FROM loan"),
                      sql("SELECT COUNT(*) FROM loan"), options=PRUNE)
    r = gen.answer(L3_11, L3)
    assert r.ok and "duration" not in columns_in(llm.prompts[1], "loan")
    assert tables_in(llm.prompts[2]) == list(model.tables) and "duration" in columns_in(llm.prompts[2], "loan")
    assert any(s.name == "widen" and s.detail.endswith("取消字段裁剪") for s in r.trace)


def test_the_pool_size_and_the_context_are_recorded():
    llm = ScriptedLLM({"columns": ["loan.payments"], "reason": ""}, sql("SELECT 1"))
    gen = GenerativeSQL(llm, model, SQLiteExecutor(RISK_DB_PATH), Layer3Options(retrieval="bm25", top_k=2,
                        column_pruning=True), examples=ExamplePicker(load_examples(), "bm25"))
    r = gen.answer(L3_11, L3)
    top = r.trace[0].detail.split("表 ", 1)[1].split("]")[0]
    assert top.count("'") == 4 and set(r.context_tables) == set(tables_in(llm.prompts[1]))  # two, then the bridge
    assert r.context_columns == sorted(r.context_columns) and "loan.payments" in r.context_columns
    plain, _ = layer3(sql("SELECT 1"))
    assert plain.answer(L3_11, L3).context_columns is None


def test_a_named_dimension_joins_the_candidates_and_survives_pruning():
    options = Layer3Options(retrieval="bm25", column_linking=True, column_pruning=True)
    gen, llm = layer3({"columns": ["district.A2"], "reason": ""}, sql("SELECT 1"), options=options,
                      retriever=TableRetriever(model, "bm25", k=1))
    r = gen.answer("1998年12月各地区现金存入减去现金取现的净额是多少？", L3)
    assert "## 问题中提到的维度（分组或筛选所在的字段）\n- 各地区 → district.A3" in llm.prompts[0]
    assert "district" in r.context_tables and {"district.A2", "district.A3"} <= set(r.context_columns)
    assert "维度 ['各地区 → district.A3']" in next(s for s in r.trace if s.name == "link").detail

def test_naive_baseline_sends_the_whole_schema_and_nothing_else():
    gen, llm = layer3(sql("SELEC 1"), options=NAIVE)
    r = gen.answer("1998年各地区单笔1万及以上的现金取现交易，平均金额是多少？", L3)
    prompt = llm.prompts[0]
    assert not r.ok and r.attempts == 1  # no repair
    assert all(f"### {t}（" in prompt for t in model.tables)
    assert "## 示例" not in prompt and "业务取值" not in prompt and "官方口径" not in prompt


# ---------------------------------------------------------------- through the agent
def test_agent_routes_an_uncovered_question_to_layer3():
    parse = example_output(dict(metrics=["loan_cnt"], status="uncovered", reason="本金阈值不是语义层维度"))
    agent = Text2SQLAgent(llm=ScriptedLLM(parse, sql("SELECT COUNT(*) FROM loan WHERE amount > 300000")))
    r = agent.ask("本金超过30万的贷款有几笔")
    assert r.route == "L3" and r.rows == [[82]] and r.usage.calls == 2
    assert "说明：口径说明：示例说明" in r.answer and "第三层" in r.answer
