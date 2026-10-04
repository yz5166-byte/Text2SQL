"""Command line: `t2s ask "问题"` for one question, `t2s chat` for an interactive session."""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from text2sql.llm import LLMError, get_llm
from text2sql.pipeline.workflow import AgentResponse, Text2SQLAgent

console = Console()


def build_agent(provider: str) -> Text2SQLAgent:
    llm = None
    if provider != "none":
        try:
            llm = get_llm(provider)
        except LLMError as exc:
            console.print(f"[yellow]未启用大模型（{exc}），仅规则可判定的问题可以回答。[/yellow]")
    layer3 = None
    if llm is not None:
        try:
            from text2sql.pipeline.layer3 import GenerativeSQL

            layer3 = GenerativeSQL(llm)
        except ImportError:
            pass
    return Text2SQLAgent(llm=llm, layer3=layer3)


def show(resp: AgentResponse, trace: bool) -> None:
    console.print(Panel(resp.answer or "", title=f"路由 {resp.route}（{resp.source}）", expand=False))
    if resp.display_rows:
        table = Table(*resp.display_columns, show_lines=False)
        for row in resp.display_rows[:30]:
            table.add_row(*row)
        console.print(table)
        if len(resp.display_rows) > 30:
            console.print(f"仅显示前 30 行，共 {len(resp.display_rows)} 行")
    for sql in resp.sql:
        console.print(Panel(sql, title="SQL", expand=False))
    if trace:
        for step in resp.trace:
            console.print(f"  {step.name:<10} {step.elapsed_ms:8.1f} ms  {step.detail}")
        console.print(f"  总耗时 {resp.latency_s:.2f}s；LLM 调用 {resp.usage.calls} 次；"
                      f"token {resp.usage.input_tokens}+{resp.usage.output_tokens}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="t2s")
    ap.add_argument("--provider", choices=["deepseek", "claude", "none"], default="deepseek")
    ap.add_argument("--trace", action="store_true", help="显示每一步的耗时与决策")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    sub.add_parser("chat")
    args = ap.parse_args()

    agent = build_agent(args.provider)
    if args.cmd == "ask":
        show(agent.ask(args.question), args.trace)
        return
    console.print("输入问题，回车查询；输入 exit 退出。")
    while True:
        question = console.input("[bold]问题> [/bold]").strip()
        if question in ("exit", "quit", ""):
            break
        resp = agent.ask(question)
        show(resp, args.trace)
        if resp.route == "CLARIFY" and resp.clarify_options:
            choice = console.input("选择指标编号> ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(resp.clarify_options):
                show(agent.resolve_clarification(question, resp.clarify_options[int(choice) - 1]["metric"]),
                     args.trace)


if __name__ == "__main__":
    main()
