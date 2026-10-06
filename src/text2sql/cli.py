"""Command line: `t2s ask "问题"` for one question, `t2s chat` for an interactive session.

In chat a clarification is answered in place: a metric number or a time re-asks the question,
up to MAX_CLARIFY_ROUNDS times (最近的风险情况如何 needs two: the metric, then the time).
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from text2sql.llm import LLMError, get_llm
from text2sql.pipeline.answer import response_title
from text2sql.pipeline.workflow import MAX_CLARIFY_ROUNDS, AgentResponse, Text2SQLAgent

console = Console()
EXIT_WORDS = ("exit", "quit", "退出")


def build_agent(provider: str) -> Text2SQLAgent:
    llm = None
    if provider != "none":
        try:
            llm = get_llm(provider)
        except LLMError as exc:
            console.print(f"[yellow]未启用大模型（{exc}），仅规则可判定的问题可以回答。[/yellow]")
    return Text2SQLAgent(llm=llm)  # layer 3 is enabled whenever a model is configured


def show(resp: AgentResponse, trace: bool) -> None:
    failed = resp.error is not None
    # metric options go in a table below; in the panel text a long option wraps after its number
    text = resp.answer.split("\n", 1)[0] if resp.clarify_options else resp.answer
    # Text, not a markup string: brackets in SQL or in the question ([amount], [/b]) are content
    console.print(Panel(Text(text), title=response_title(resp.route, resp.source, failed),
                        border_style="red" if failed else "none", expand=False))
    if resp.clarify_options:
        options = Table("编号", "指标")
        options.add_column("说明", overflow="fold")  # the whole description, wrapped inside its column
        for i, o in enumerate(resp.clarify_options, 1):
            options.add_row(str(i), Text(o["name"]), Text(o["description"]))
        console.print(options)
    if resp.display_rows:
        table = Table()
        for column in resp.display_columns:
            table.add_column(Text(column))
        for row in resp.display_rows[:30]:
            table.add_row(*map(Text, row))
        console.print(table)
        if len(resp.display_rows) > 30:
            console.print(f"仅显示前 30 行，共 {len(resp.display_rows)} 行")
    for sql in resp.sql:
        console.print(Panel(Text(sql), title="SQL", expand=False))
    if trace:
        for step in resp.trace:
            console.print(f"  {step.name:<10} {step.elapsed_ms:8.1f} ms  {step.detail}", markup=False)
        console.print(f"  总耗时 {resp.latency_s:.2f}s；LLM 调用 {resp.usage.calls} 次；"
                      f"token {resp.usage.input_tokens}+{resp.usage.output_tokens}")


def read(prompt: str) -> str | None:
    """One line of input; None at end of input (Ctrl+D)."""
    try:
        return console.input(prompt).strip()
    except EOFError:
        console.print()
        return None


def answer_clarification(agent: Text2SQLAgent, resp: AgentResponse) -> AgentResponse | None:
    """Ask the user for what the clarification needs and re-ask; None when the user skips."""
    if resp.clarify_kind == "metric":
        n = len(resp.clarify_options)
        while choice := read(f"选择指标编号（1 至 {n}，回车跳过）> "):
            if choice.isdigit() and 1 <= int(choice) <= n:
                return agent.resolve_metric_clarification(resp.question, resp.clarify_options[int(choice) - 1]["metric"])
            console.print(f"[yellow]请输入 1 至 {n} 之间的编号。[/yellow]")
        return None
    while reply := read("补充时间（回车跳过）> "):
        follow = agent.resolve_time_clarification(resp.question, reply)
        if follow is not None:
            return follow
        console.print("[yellow]没有识别出时间，请换一种写法，例如：1998年11月、1998年第三季度。[/yellow]")
    return None


def chat(agent: Text2SQLAgent, trace: bool) -> None:
    console.print("输入问题后回车查询；输入 exit 或按 Ctrl+D 退出。")
    while True:
        question = read("[bold]问题> [/bold]")
        if question is None or question in EXIT_WORDS:
            return
        if not question:  # a stray Enter does not end the session
            continue
        resp = agent.ask(question)
        show(resp, trace)
        rounds = 0
        while resp.route == "CLARIFY" and rounds < MAX_CLARIFY_ROUNDS:
            follow = answer_clarification(agent, resp)
            if follow is None:
                break
            resp, rounds = follow, rounds + 1
            console.print(Text(f"重新提问：{resp.question}", style="dim"))
            show(resp, trace)
        if resp.route == "CLARIFY" and rounds == MAX_CLARIFY_ROUNDS:
            console.print(f"[yellow]已追问 {MAX_CLARIFY_ROUNDS} 轮，请换一种说法重新提问。[/yellow]")


def main() -> None:
    ap = argparse.ArgumentParser(prog="t2s")
    ap.add_argument("--provider", choices=["deepseek", "claude", "none"], default="deepseek")
    ap.add_argument("--trace", action="store_true", help="显示每一步的耗时与决策")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    sub.add_parser("chat")
    args = ap.parse_args()

    try:
        agent = build_agent(args.provider)
        if args.cmd == "chat":
            chat(agent, args.trace)
            return
        resp = agent.ask(args.question)
        show(resp, args.trace)
        if resp.route == "CLARIFY":
            console.print("[dim]使用 t2s chat 可以直接回答追问。[/dim]")
    except KeyboardInterrupt:  # Ctrl+C, also during a model call
        console.print("\n已退出。")


if __name__ == "__main__":
    main()
