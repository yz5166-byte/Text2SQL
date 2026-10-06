"""Streamlit demo of the tiered Text2SQL agent: a chat whose answers fold out their route, trace, SQL and cost.

Run with: uv run --extra app streamlit run app/streamlit_app.py --server.address localhost
(localhost keeps others on the network from querying with your API key).

Streamlit reruns this script on every click. A click only queues an action (queue); the next run
handles it once (handle), then the whole conversation is drawn again from session state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from text2sql.config import DATA_END_DATE, DATA_START_DATE, settings
from text2sql.eval.cost import price_for, request_cost
from text2sql.llm import LLMError, get_llm
from text2sql.pipeline.answer import response_title
from text2sql.pipeline.workflow import MAX_CLARIFY_ROUNDS, AgentResponse, Text2SQLAgent

PROVIDERS = {
    f"DeepSeek（{settings.deepseek_model}）": "deepseek",
    f"Claude（{settings.claude_model}）": "claude",
    "无模型（仅规则）": "none",
}
# (evaluation id, what it shows, question): the dev split, plus C-05 from test for two clarification rounds
EXAMPLES = [
    ("L1-01", "第一层 · 快速通道", "1998年11月的风险事件数是多少？"),
    ("L1-13", "第一层 · 模型解析", "97年各大区的不良率"),
    ("L2-11", "第二层 · 趋势", "1998年上半年每月的现金取现笔数"),
    ("L3-03", "第三层 · 生成式 SQL", "1996年犯罪数最多的区县，1998年发生了多少起风险事件？"),
    ("C-01", "澄清 · 指标多义", "上个月的损失有多少？"),
    ("C-05", "澄清 · 先指标后时间（test 集）", "最近的风险情况如何？"),
    ("M-01", "口径问答", "不良率是怎么计算的？"),
    ("O-03", "域外 · 模型判定", "如何申请一张信用卡？"),
]
MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")
TIME_ERROR = "没有识别出时间，请换一种写法，例如：1998年11月、1998年第三季度。"


@dataclass
class Turn:
    said: str  # what the user typed, clicked or answered
    resp: AgentResponse
    cost_usd: float | None  # upper bound at peak prices; None without a model
    rounds: int = 0  # clarification answers that led to this turn


@st.cache_resource
def load_agent(provider: str) -> tuple[Text2SQLAgent, str | None]:
    """One agent per provider for the whole server, so the embedding model loads once; a warning if it has no model."""
    if provider == "none":
        return Text2SQLAgent(llm=None), None
    try:
        return Text2SQLAgent(llm=get_llm(provider)), None
    except LLMError as exc:
        return Text2SQLAgent(llm=None), f"未启用大模型（{exc}），仅规则可判定的问题可以回答。"


def plain(text: str) -> str:
    """Markdown that shows `text` as written: COUNT(*), a_b or $ must not turn into formatting."""
    return MARKDOWN_SPECIAL.sub(r"\\\1", text).replace("\n", "  \n")


def unique_columns(columns: list[str]) -> list[str]:
    """Generated SQL may repeat a column name, which a data frame cannot show."""
    seen: dict[str, int] = {}
    out = []
    for c in columns:
        seen[c] = seen.get(c, 0) + 1
        out.append(c if seen[c] == 1 else f"{c}（{seen[c]}）")
    return out


# ---------------------------------------------------------------- actions
def queue(action: tuple[str, str]) -> None:
    """("ask", question), ("metric", metric key) or ("time", the user's reply)."""
    st.session_state.action = action


def queue_time(input_key: str) -> None:
    queue(("time", st.session_state[input_key]))


def handle(agent: Text2SQLAgent, action: tuple[str, str]) -> None:
    kind, value = action
    turns: list[Turn] = st.session_state.turns
    if kind == "ask":
        turns.append(Turn(value, resp := agent.ask(value), cost_of(agent, resp)))
        return
    last = turns[-1]
    if kind == "metric":
        resp = agent.resolve_metric_clarification(last.resp.question, value)
        said = f"选择指标：{agent.model.metrics[value].name}"
    else:
        resp = agent.resolve_time_clarification(last.resp.question, value)
        if resp is None:
            st.session_state.time_error = TIME_ERROR
            return
        said = f"补充时间：{value}"
    turns.append(Turn(said, resp, cost_of(agent, resp), last.rounds + 1))


def cost_of(agent: Text2SQLAgent, resp: AgentResponse) -> float | None:
    price = price_for(agent.llm.name if agent.llm else None)
    u = resp.usage
    return request_cost(price, u.input_tokens, u.cached_input_tokens, u.output_tokens) if price else None


# ---------------------------------------------------------------- drawing
def show_turn(i: int, turn: Turn, last: bool) -> None:
    resp = turn.resp
    with st.chat_message("user"):
        st.markdown(plain(turn.said))
    with st.chat_message("assistant"):
        if turn.rounds:
            st.caption(plain(f"重新提问：{resp.question}"))
        title = response_title(resp.route, resp.source, resp.error is not None)
        if resp.error is not None:
            st.error(f"**{title}**  \n{plain(resp.answer)}")
        else:
            head, *notes = resp.answer.split("\n")
            st.markdown(f"**{title}**  \n{plain(head)}")
            if notes and not resp.clarify_options:  # metric options are drawn as buttons instead
                st.caption(plain("\n".join(n for n in notes if n)))
        if resp.display_rows:
            st.dataframe(pd.DataFrame(resp.display_rows, columns=unique_columns(resp.display_columns)),
                         hide_index=True)
        if last and resp.route == "CLARIFY":
            show_clarification(i, turn)
        with st.expander("链路详情"):
            show_details(turn)


def show_clarification(i: int, turn: Turn) -> None:
    if turn.rounds >= MAX_CLARIFY_ROUNDS:
        st.warning(f"已追问 {MAX_CLARIFY_ROUNDS} 轮，请换一种说法重新提问。")
        return
    resp = turn.resp
    if resp.clarify_kind == "metric":
        for j, option in enumerate(resp.clarify_options):
            left, right = st.columns([1, 3], vertical_alignment="center")
            left.button(option["name"], key=f"metric-{i}-{j}", on_click=queue, args=(("metric", option["metric"]),),
                        use_container_width=True)
            right.caption(plain(option["description"]))
        return
    with st.form(f"time-{i}", clear_on_submit=True, border=False):
        st.text_input("补充时间", key=f"time-input-{i}", placeholder="例如：1998年11月、1998年第三季度")
        st.form_submit_button("提交", on_click=queue_time, args=(f"time-input-{i}",))
    if error := st.session_state.pop("time_error", None):
        st.warning(error)


def show_details(turn: Turn) -> None:
    resp, u = turn.resp, turn.resp.usage
    st.markdown("\n".join(f"{n}. {plain(f'{s.name}（{s.elapsed_ms:.1f} 毫秒）：{s.detail}')}"
                          for n, s in enumerate(resp.trace, 1)) or "无轨迹")
    for sql in resp.sql:
        st.code(sql, language="sql", wrap_lines=True)
    if resp.ir:
        st.caption("中间表示（IR）：模型解析与规范化的结果，第一、二层据此编译 SQL")
        st.json(resp.ir, expanded=False)
    cost = "未使用模型" if turn.cost_usd is None else f"费用上限 {turn.cost_usd:.4f} 美元"
    st.caption(f"总耗时 {resp.latency_s:.2f} 秒；模型调用 {u.calls} 次；token 输入 {u.input_tokens}"
               f"（缓存命中 {u.cached_input_tokens}）、输出 {u.output_tokens}；{cost}")


def main() -> None:
    st.set_page_config(page_title="风控取数助手")
    st.session_state.setdefault("turns", [])

    with st.sidebar:
        agent, warning = load_agent(PROVIDERS[st.selectbox("模型", list(PROVIDERS))])
        if warning:
            st.warning(warning)
        st.subheader("示例问题")
        for item_id, shows, question in EXAMPLES:
            st.button(question, key=f"example-{item_id}", help=f"{item_id}：{shows}", on_click=queue,
                      args=(("ask", question),), use_container_width=True)
        st.caption("示例取自自建评估集的 dev 集，C-05 取自 test 集；正式成绩以 test 集为准，见 docs/EXPERIMENTS.md。")
        st.button("清空对话", on_click=st.session_state.turns.clear)

    st.title("风控取数助手")
    st.caption(f"分层 Text2SQL：语义层优先，生成式兜底。数据为 PKDD'99 捷克银行（BIRD financial），"
               f"{DATA_START_DATE} 至 {DATA_END_DATE}；\"上个月\"等相对时间以 {DATA_END_DATE} 为参考日期。")

    if question := st.chat_input("输入问题，例如：1998年11月的风险事件数是多少？"):
        queue(("ask", question))
    action = st.session_state.pop("action", None)
    turns = st.session_state.turns
    for i, turn in enumerate(turns):
        show_turn(i, turn, last=i == len(turns) - 1 and action is None)
    if action:
        with st.chat_message("assistant"), st.spinner("查询中……"):
            handle(agent, action)
        st.rerun()


if __name__ == "__main__":
    main()
