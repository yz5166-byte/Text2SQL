"""Command line and clarification follow-up: time rewriting, chained clarifications, titles, failures."""

import httpx
import openai
import pytest
from rich.console import Console

from text2sql import cli
from text2sql.config import RISK_DB_PATH
from text2sql.llm.base import LLMError
from text2sql.llm.deepseek import DeepSeekClient
from text2sql.pipeline.answer import response_title
from text2sql.pipeline.prompts import example_output
from text2sql.pipeline.workflow import AgentResponse, Text2SQLAgent, with_time
from text2sql.semantic.timeparse import find_times

from scripted_llm import ScriptedLLM

needs_db = pytest.mark.skipif(not RISK_DB_PATH.exists(), reason="data/bank_risk.sqlite missing")


# ---------------------------------------------------------------- time follow-up
@pytest.mark.parametrize("question, time_text, expected", [
    ("最近的风险事件数是多少？", "1998年11月", "1998年11月的风险事件数是多少？"),  # vague word replaced
    ("1999年的风险事件数是多少？", "1998年", "1998年的风险事件数是多少？"),  # out-of-range time replaced
    ("风险事件数同比变化", "1998年11月", "1998年11月风险事件数同比变化"),  # no time: put in front
    ("目前南摩拉维亚的风险涉及金额", "1998年第3季度", "1998年第3季度南摩拉维亚的风险涉及金额"),
])
def test_the_users_time_replaces_the_questions_own(question, time_text, expected):
    assert with_time(question, time_text) == expected


def test_several_times_leave_only_the_users():
    rewritten = with_time("1997年和1998年3月的放款金额", "1998年")
    assert [(s.start.isoformat(), s.end.isoformat()) for s in find_times(rewritten)] == [("1998-01-01", "1998-12-31")]


@needs_db
def test_time_reply_is_checked_before_re_asking():
    agent = Text2SQLAgent(llm=None)
    assert agent.resolve_time_clarification("最近的风险事件数是多少？", "最近") is None
    assert agent.resolve_time_clarification("最近的风险事件数是多少？", "下下个世纪") is None
    r = agent.resolve_time_clarification("最近的风险事件数是多少？", "就看1998年11月吧")
    assert r.question == "1998年11月的风险事件数是多少？"
    assert r.route == "L1" and r.rows == [[265]] and r.usage.calls == 0


# ---------------------------------------------------------------- titles
@pytest.mark.parametrize("route, source, failed, title", [
    ("L1", "fast_path", False, "第一层 OKR 物化表 · 快速通道"),
    ("CLARIFY", "rule", False, "需要澄清 · 规则"),
    ("L3", "llm", True, "未完成 · 第三层 生成式 SQL"),
    (None, None, True, "未完成 · 路由失败"),
])
def test_titles_name_the_layer_and_who_decided(route, source, failed, title):
    assert response_title(route, source, failed) == title


# ---------------------------------------------------------------- model failures
class FailingLayer3:
    def answer(self, question, decision):
        raise LLMError("connection reset")


def test_a_failed_model_call_becomes_an_answer_not_a_crash():
    parse = example_output(dict(metrics=["loan_cnt"], status="uncovered", reason="本金阈值不是语义层维度"))
    agent = Text2SQLAgent(llm=ScriptedLLM(parse), layer3=FailingLayer3())
    r = agent.ask("本金超过30万的贷款有几笔")
    assert r.route == "L3" and r.error == "模型调用失败：connection reset" and r.answer == r.error


@pytest.mark.parametrize("error", [
    openai.APIConnectionError(request=httpx.Request("POST", "https://api.deepseek.com")),
    openai.APIStatusError("busy", response=httpx.Response(503, request=httpx.Request("POST", "https://x")), body=None),
])
def test_deepseek_errors_are_llm_errors(monkeypatch, error):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    client = DeepSeekClient()

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(client.client.chat.completions, "create", fail)
    with pytest.raises(LLMError, match="DeepSeek"):
        client.complete("system", "user")


# ---------------------------------------------------------------- chat session
def run_chat(monkeypatch, *lines: str) -> str:
    """Feed `lines` to `t2s chat` (then end of input) and return everything it printed."""
    console = Console(record=True, width=120)
    replies = list(lines)

    def fake_input(prompt=""):
        console.print(prompt, end="")
        if not replies:
            raise EOFError
        return replies.pop(0)

    monkeypatch.setattr(console, "input", fake_input)
    monkeypatch.setattr(cli, "console", console)
    cli.chat(Text2SQLAgent(llm=None), trace=False)
    return console.export_text()


@needs_db
def test_chat_answers_a_metric_then_a_time_clarification(monkeypatch):
    out = run_chat(monkeypatch, "最近的交易量是多少？", "abc", "9", "1", "下下个世纪", "1998年11月")
    assert out.count("请输入 1 至 2 之间的编号") == 2 and "没有识别出时间" in out
    assert "重新提问：最近的交易笔数是多少？" in out
    assert "重新提问：1998年11月的交易笔数是多少？" in out and "25,341" in out


@needs_db
def test_chat_stops_after_three_rounds_and_survives_blank_lines(monkeypatch):
    out = run_chat(monkeypatch, "", "1999年的风险事件数是多少？", "2000年", "2001年", "2002年")
    assert out.count("需要澄清") == 4 and "已追问 3 轮" in out


def test_brackets_in_answers_and_sql_are_shown_as_text(monkeypatch):
    console = Console(record=True, width=120)
    monkeypatch.setattr(cli, "console", console)
    cli.show(AgentResponse(question="q", route="L3", source="llm", answer="说明 [/b] 原样显示",
                           sql=["SELECT [amount] FROM trans"]), trace=False)
    out = console.export_text()
    assert "[/b]" in out and "SELECT [amount] FROM trans" in out
