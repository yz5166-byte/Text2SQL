"""The agent: an explicit workflow (route, then one handler) with a full per-request trace."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from text2sql.config import DATA_END_DATE, DATA_START_DATE, RISK_DB_PATH
from text2sql.db.executor import SQLiteExecutor
from text2sql.llm.base import LLMClient, LLMError, Usage
from text2sql.pipeline import answer as ans
from text2sql.pipeline.meta import answer_meta
from text2sql.pipeline.decision import RouteDecision
from text2sql.pipeline.layer3 import GenerativeSQL, L3Result
from text2sql.pipeline.router import Router
from text2sql.pipeline.semantic import run_queries
from text2sql.pipeline.summary import summarize
from text2sql.pipeline.trace import TraceStep
from text2sql.semantic.compiler import CompileError
from text2sql.semantic.ir import QueryIR
from text2sql.semantic.okr import compile_ir
from text2sql.semantic.registry import SemanticModel, load_semantic_model

OOD_TEXT = (
    "这个问题不在取数助手的服务范围内。我可以回答银行风控与经营数据问题，例如：\n"
    "1. 上个月风险事件数是多少？\n2. 1997年各地区的不良率\n3. 1998年每月现金取现金额趋势\n"
    "也可以询问指标口径，例如：不良率是怎么计算的？"
)
SUMMARY_NOTE = "首句总结由模型根据结果表生成，其中数字均已与结果表核对"
TIME_HINT = f"请给出具体的时间范围，例如：1998年11月、1998年第三季度、1997年到1998年（数据范围 {DATA_START_DATE} 至 {DATA_END_DATE}）。"


class Layer3(Protocol):
    def answer(self, question: str, decision: RouteDecision) -> L3Result: ...


@dataclass
class AgentResponse:
    question: str
    route: str | None = None
    source: str | None = None
    answer: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    display_columns: list[str] = field(default_factory=list)
    display_rows: list[list[str]] = field(default_factory=list)
    sql: list[str] = field(default_factory=list)
    ir: dict | None = None
    clarify_options: list[dict] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    trace: list[TraceStep] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    attempts: int = 0
    latency_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Text2SQLAgent:
    def __init__(self, llm: LLMClient | None = None, db_path: Path = RISK_DB_PATH,
                 model: SemanticModel | None = None, layer3: Layer3 | None = None, use_fast_path: bool = True,
                 max_rows: int | None = None):
        self.model = model or load_semantic_model()
        self.llm = llm
        self.executor = SQLiteExecutor(db_path, max_rows=max_rows)
        self.router = Router(self.model, llm, use_fast_path=use_fast_path)
        # layer 3 needs a model; tests may inject their own
        self.layer3 = layer3 if layer3 is not None else (GenerativeSQL(llm, self.model, self.executor) if llm else None)

    def ask(self, question: str) -> AgentResponse:
        t0 = time.perf_counter()
        resp = AgentResponse(question=question)
        step_t = time.perf_counter()
        try:
            decision = self.router.route(question)
        except LLMError as exc:
            resp.error = f"路由失败：{exc}"
            resp.answer = resp.error
            resp.latency_s = time.perf_counter() - t0
            return resp
        resp.usage.add(decision.usage)
        resp.route, resp.source = decision.route, decision.source
        resp.assumptions = list(decision.assumptions)
        resp.ir = decision.ir.model_dump(mode="json") if decision.ir else None
        parse_calls = f"；解析调用 {decision.usage.calls} 次" if decision.usage.calls else ""
        if decision.parse_problems:
            parse_calls += "；不合格原因：" + " / ".join(p.replace("\n", " ") for p in decision.parse_problems)
        resp.trace.append(TraceStep("route", f"{decision.route}（{decision.source}）：{decision.reason}{parse_calls}",
                                    (time.perf_counter() - step_t) * 1000))

        handler = {
            "L1": self._semantic, "L2": self._semantic, "L3": self._generative,
            "META": self._meta, "OOD": self._ood, "CLARIFY": self._clarify,
        }[decision.route]
        handler(question, decision, resp)
        resp.latency_s = time.perf_counter() - t0
        return resp

    # ------------------------------------------------------------------ handlers
    def _semantic(self, question: str, decision: RouteDecision, resp: AgentResponse) -> None:
        t = time.perf_counter()
        try:
            queries = compile_ir(decision.ir, self.model)
        except CompileError as exc:  # should not happen after routing checks; degrade to layer 3
            resp.trace.append(TraceStep("compile", f"编译失败，降级第三层：{exc}", 0))
            decision.route = resp.route = "L3"
            return self._generative(question, decision, resp)
        resp.sql = [q.display_sql() for q in queries]
        resp.trace.append(TraceStep("compile", f"{len(queries)} 条 SQL（{queries[0].layer}）",
                                    (time.perf_counter() - t) * 1000))
        t = time.perf_counter()
        result = run_queries(self.executor, queries)
        resp.trace.append(TraceStep("execute", "成功" if result.ok else f"失败：{result.error}",
                                    (time.perf_counter() - t) * 1000))
        if not result.ok:
            resp.error = result.error
            resp.answer = f"查询执行失败：{result.error}"
            return
        ir = decision.ir
        self._fill_table(resp, result.columns, result.rows, ir)
        first = ans.conclusion(ir, result.columns, result.rows, self.model)
        if ir.query_type in ("breakdown", "trend") and result.rows:
            first = self._summarize(question, resp) or first
        lines = [first, "",
                 f"路由：{ans.LAYER_ZH[resp.route]}", f"时间范围：{ans.time_scope(ir)}；{ans.data_note()}"]
        lines += [c if c.startswith("筛选") else f"口径：{c}" for c in ans.caliber_lines(ir, self.model)]
        lines += [f"说明：{a}" for a in resp.assumptions]
        resp.answer = "\n".join(lines)

    def _generative(self, question: str, decision: RouteDecision, resp: AgentResponse) -> None:
        if self.layer3 is None:
            resp.error = "第三层未启用"
            resp.answer = "该问题需要生成式 SQL（第三层），当前未启用。"
            return
        result = self.layer3.answer(question, decision)
        resp.trace += result.trace
        resp.usage.add(result.usage)
        resp.attempts = result.attempts
        resp.sql = [result.sql] if result.sql else []
        if not result.ok:
            resp.error = result.error
            resp.answer = f"未能生成可执行的查询：{result.error}"
            return
        self._fill_table(resp, result.columns, result.rows)
        first = f"共 {len(result.rows)} 行结果，见下表。" if result.rows else "查询结果为空。"
        if len(result.rows) > 1:
            first = self._summarize(question, resp) or first
        lines = [first, "", f"路由：{ans.LAYER_ZH['L3']}（非标准口径，请结合 SQL 核对）", ans.data_note()]
        lines += [f"说明：{n}" for n in [*result.notes, *resp.assumptions]]
        resp.answer = "\n".join(lines)

    def _meta(self, question: str, decision: RouteDecision, resp: AgentResponse) -> None:
        resp.answer = answer_meta(question, self.model)

    def _ood(self, question: str, decision: RouteDecision, resp: AgentResponse) -> None:
        resp.answer = OOD_TEXT

    def _clarify(self, question: str, decision: RouteDecision, resp: AgentResponse) -> None:
        if decision.clarify_kind == "time":
            resp.answer = f"需要确认：{decision.reason}。{TIME_HINT}"
            return
        resp.clarify_options = [
            {"metric": k, "name": self.model.metrics[k].name, "description": self.model.metrics[k].description}
            for k in decision.clarify_options
        ]
        options = "\n".join(f"{i}. {o['name']}：{o['description']}" for i, o in enumerate(resp.clarify_options, 1))
        resp.answer = f"需要确认：{decision.reason}。请选择要查询的指标：\n{options}"

    def _summarize(self, question: str, resp: AgentResponse) -> str | None:
        """Model-written first sentence for a breakdown or trend; None keeps the plain conclusion."""
        t = time.perf_counter()
        s = summarize(self.llm, question, resp.display_columns, resp.display_rows)
        resp.usage.add(s.usage)
        resp.trace.append(TraceStep("summary", s.note, (time.perf_counter() - t) * 1000))
        if s.text:
            resp.assumptions.append(SUMMARY_NOTE)
        return s.text

    def _fill_table(self, resp: AgentResponse, columns: list[str], rows: list[tuple],
                    ir: QueryIR | None = None) -> None:
        resp.columns, resp.rows = list(columns), [list(r) for r in rows]
        resp.display_columns, resp.display_rows = ans.display_table(columns, rows, self.model, ir)

    # ------------------------------------------------------------------ clarification follow-up
    def resolve_clarification(self, question: str, metric_key: str) -> AgentResponse:
        """Re-ask with the chosen metric substituted for the ambiguous term (or appended)."""
        name = self.model.metrics[metric_key].name
        for term in self.model.ambiguous_terms:
            if term in question:
                return self.ask(question.replace(term, name, 1))
        return self.ask(f"{question}（指标：{name}）")
