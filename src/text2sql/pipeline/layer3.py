"""Layer 3: generated SQL for questions the semantic layer cannot express.

The context is built once per question: the retrieved tables (top 5 plus the join bridge),
official definitions of the metrics the question names, the business values the linker
recognises in the question with their columns and codes (the same deterministic translation
as layer 2: 现金取现 is trans.operation = 'VYBER', not trans.type), the resolved time hint,
BIRD evidence when given, and the three most similar examples. Every attempt passes the guard,
the dry run and execution. A failure goes back to the model with its reason unless a
rewrite cannot help (NOT_REPAIRABLE); a missing or ambiguous column also widens the context
to the full schema, since the table the question needs may not have been retrieved. An
empty result is returned as it is, with a note.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from text2sql.config import settings
from text2sql.db.executor import ErrorKind, ExecResult, SQLiteExecutor, is_repairable
from text2sql.db.guard import guard_sql
from text2sql.knowledge.examples import ExamplePicker, SQLExample, load_examples
from text2sql.knowledge.retriever import Method, TableRetriever
from text2sql.llm.base import LLMClient, Usage
from text2sql.pipeline.decision import RouteDecision
from text2sql.pipeline.meta import formula
from text2sql.pipeline.trace import TraceStep
from text2sql.semantic.linker import Code, find_metric_mentions, find_value_mentions
from text2sql.semantic.registry import SemanticModel

SYSTEM = """你是 SQLite 专家，为银行风控数据库编写查询。
1. 只写一条 SELECT 查询（可以使用 WITH），不得修改数据。
2. 只使用给出的表和字段；表名 order 是 SQL 关键字，须写作 "order"。
3. 日期是 YYYY-MM-DD 格式的文本：按月或年汇总用 strftime，日期差用 julianday，日期推算用 date() 的修饰符。
4. 整数相除前先 CAST(... AS REAL)。
5. 只输出问题要求的列；排名类问题用 ORDER BY 与 LIMIT。
6. 问题涉及给出的官方口径时，按口径计算。
7. 计算环比、同比或增长率时，基期数据可能在要输出的期间之外：先在包含基期的范围内计算，再筛选要输出的期间。
以 json 输出：{"sql": "一条 SQL", "explanation": "一句话说明口径与关键假设"}"""

SQL_SCHEMA = {
    "type": "object",
    "properties": {"sql": {"type": "string"}, "explanation": {"type": "string"}},
    "required": ["sql", "explanation"],
}
WIDEN_ON = {ErrorKind.NO_SUCH_COLUMN, ErrorKind.AMBIGUOUS_COLUMN}  # the needed table may not have been retrieved
MAX_VALUES_SHOWN = 20
EMPTY_NOTE = "查询结果为空，未做修复；如与预期不符，请核对 SQL 中的筛选条件与取值写法"


@dataclass(frozen=True)
class Layer3Options:
    """Switches for the experiments (X2 to X4); the defaults are the production layer 3."""

    schema: Literal["retrieved", "full"] = "retrieved"
    retrieval: Method = "hybrid"  # how tables are retrieved when schema is "retrieved" (X4)
    examples: int = 3  # similar examples in the prompt; 0 turns them off
    knowledge: bool = True  # official metric definitions and linked values
    max_repairs: int | None = None  # None: settings.max_repair_attempts


# Plain text-to-SQL: the whole schema and the question, one generation, no repair (X2 baseline).
NAIVE = Layer3Options(schema="full", examples=0, knowledge=False, max_repairs=0)


class GeneratedSQL(BaseModel):
    sql: str = Field(min_length=1)
    explanation: str


@dataclass
class L3Result:
    ok: bool
    sql: str | None = None  # the statement that ran (with the guard's row cap)
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    attempts: int = 0
    error: str | None = None
    notes: list[str] = field(default_factory=list)  # the model's explanation, the empty-result note
    trace: list[TraceStep] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


@dataclass
class Attempt:
    sql: str  # as run, or as generated when rejected before running
    result: ExecResult | None = None
    error_kind: ErrorKind | None = None
    reason: str | None = None  # why it failed, phrased for the repair prompt


class GenerativeSQL:
    def __init__(self, llm: LLMClient, model: SemanticModel, executor: SQLiteExecutor,
                 options: Layer3Options = Layer3Options(), retriever: TableRetriever | None = None,
                 examples: ExamplePicker | None = None):
        self.llm, self.model, self.executor, self.options = llm, model, executor, options
        self.retriever = retriever or TableRetriever(model, options.retrieval)
        self.examples = examples or ExamplePicker(load_examples())
        self.max_repairs = settings.max_repair_attempts if options.max_repairs is None else options.max_repairs

    def answer(self, question: str, decision: RouteDecision, evidence: str | None = None) -> L3Result:
        out = L3Result(ok=False)
        t = time.perf_counter()
        if self.options.schema == "full":
            tables, context = list(self.model.tables), "全量表结构"
        else:
            retrieval = self.retriever.retrieve(f"{question} {evidence or ''}")
            tables, context = retrieval.all, f"表 {retrieval.tables}，连接补全 {retrieval.bridged}"
        examples = self.examples.pick(question, self.options.examples) if self.options.examples else []
        out.trace.append(TraceStep("retrieve", f"{context}，示例 {[e.id for e in examples]}", _ms(t)))
        request = self._request(question, decision, evidence, examples)
        feedback, last = "", None
        for attempt_no in range(1, self.max_repairs + 2):
            out.attempts = attempt_no
            t = time.perf_counter()
            reply = self.llm.complete(SYSTEM, f"## 可用的表\n{schema_text(self.model, tables)}\n\n{request}{feedback}",
                                      schema=SQL_SCHEMA, purpose="generate")
            out.usage.add(reply.usage)
            if reply.refused:
                return _failed(out, "模型拒绝回答该问题")
            generated = _parse(reply.data)
            if generated is None:
                feedback = _feedback(None, "输出不是包含 sql 与 explanation 的 JSON 对象")
                out.trace.append(TraceStep(f"generate {attempt_no}", "输出格式不合格", _ms(t)))
                continue
            attempt = last = self._run(generated.sql)
            out.trace.append(TraceStep(f"generate {attempt_no}", attempt.reason or "执行成功", _ms(t)))
            if attempt.result is not None:
                return _succeeded(out, attempt, generated.explanation)
            if not is_repairable(attempt.error_kind):
                return _failed(out, attempt.reason, attempt.sql)
            if attempt_no > self.max_repairs:  # no repair left: widening would change nothing
                break
            if attempt.error_kind in WIDEN_ON and len(tables) < len(self.model.tables):
                tables = list(self.model.tables)
                out.trace.append(TraceStep("widen", "字段相关错误，修复改用全量表结构", 0))
            feedback = _feedback(generated.sql, attempt.reason)
        reason = f"修复 {self.max_repairs} 次后仍未成功" + (f"：{last.reason}" if last else "")
        return _failed(out, reason, last.sql if last else None)

    def _run(self, sql: str) -> Attempt:
        """Guard, dry run, execute; stop at the first failure."""
        guard = guard_sql(sql, self.model.tables, self.executor.max_rows)
        if not guard.ok:
            return Attempt(sql, error_kind=guard.error_kind, reason=f"护栏拒绝：{guard.reason}")
        for step in (self.executor.explain, self.executor.execute):
            result = step(guard.sql)
            if not result.ok:
                return Attempt(guard.sql, error_kind=result.error_kind, reason=self._error_reason(result))
        return Attempt(guard.sql, result=result)

    def _error_reason(self, result: ExecResult) -> str:
        if result.error_kind == ErrorKind.TIMEOUT:
            return f"执行超过 {self.executor.timeout_s:g} 秒：请缩小时间范围、补充筛选条件，避免笛卡尔积"
        return f"执行报错（{result.error_kind.value}）：{result.error}"

    def _request(self, question: str, decision: RouteDecision, evidence: str | None,
                 examples: list[SQLExample]) -> str:
        """Everything after the tables: definitions, examples and the question itself."""
        parts = []
        definitions = self._definitions(question, decision) if self.options.knowledge else ""
        if definitions:
            parts.append("## 官方口径（问题涉及这些指标时按此计算）\n" + definitions)
        values = self._values(question) if self.options.knowledge else ""
        if values:
            parts.append("## 问题中识别到的业务取值（写筛选条件时使用这些字段与编码）\n" + values)
        if examples:
            shots = [f"问题：{e.question}\n输出：{json.dumps({'sql': e.sql, 'explanation': e.explanation}, ensure_ascii=False)}"
                     for e in examples]
            parts.append("## 示例\n" + "\n".join(shots))
        ask = [f"问题：{question}"]
        if evidence:
            ask.append(f"外部知识：{evidence}")
        if decision.time_hint:
            ask.append(f"统计期间（已按参考日期换算）：{decision.time_hint[0]} 至 {decision.time_hint[1]}")
        parts.append("## 问题\n" + "\n".join(ask))
        return "\n\n".join(parts)

    def _definitions(self, question: str, decision: RouteDecision) -> str:
        keys = [m.key for m in find_metric_mentions(question, self.model)]
        if decision.parse:
            keys += decision.parse.metrics
        return "\n".join(f"- {self.model.metrics[k].name}：{self.model.metrics[k].description}"
                         f"计算方式：{formula(self.model, self.model.metrics[k])}" for k in dict.fromkeys(keys))


    def _values(self, question: str) -> str:
        lines = []
        for v in find_value_mentions(question, self.model, list(self.model.dimensions)):
            dim = self.model.dimensions[v.key]
            alias, column = dim.column.split(".")
            lines.append(f"- {v.surface} → {self.model.physical_table(alias)}.{column} {_condition(v.codes)}（{dim.name}）")
        return "\n".join(dict.fromkeys(lines))


def _condition(codes: tuple[Code, ...]) -> str:
    """SQL predicate text for a value's codes: = 'x', IN ('x', 'y'), with IS NULL for a missing code."""
    literals = [repr(c) if isinstance(c, str) else str(c) for c in codes if c is not None]
    parts = [f"= {literals[0]}" if len(literals) == 1 else f"IN ({', '.join(literals)})"] if literals else []
    if None in codes:
        parts.append("IS NULL")
    return " 或 ".join(parts)


def schema_text(model: SemanticModel, tables: list[str]) -> str:
    """Table header, then one line per column: type, Chinese name, description, value codes."""
    codes = {}
    for dim in model.dimensions.values():
        alias, column = dim.column.split(".")
        shown = "；".join(f"{v.label}={'/'.join(map(repr, v.codes))}" for v in dim.values[:MAX_VALUES_SHOWN])
        more = f" 等 {len(dim.values)} 个" if len(dim.values) > MAX_VALUES_SHOWN else ""
        codes[(model.physical_table(alias), column)] = shown + more
    lines = []
    for t in tables:
        table = model.tables[t]
        notes = f" 注意：{table.notes}" if table.notes else ""
        lines.append(f"### {t}（{table.name_zh}）：{table.description}{notes}")
        for c, col in table.columns.items():
            values = f"；取值 {codes[(t, c)]}" if (t, c) in codes else ""
            lines.append(f"- {c} {col.type} {col.name_zh}：{col.description}{values}")
    return "\n".join(lines)


def _parse(data: dict | None) -> GeneratedSQL | None:
    try:
        return GeneratedSQL.model_validate(data) if data is not None else None
    except ValidationError:
        return None


def _feedback(sql: str | None, reason: str) -> str:
    previous = f"\n\n## 上一次的 SQL\n{sql}" if sql else ""
    return f"{previous}\n\n## 需要修正\n{reason}\n请修正后重新输出。"


def _succeeded(out: L3Result, attempt: Attempt, explanation: str) -> L3Result:
    out.ok, out.sql = True, attempt.sql
    out.columns, out.rows = attempt.result.columns, attempt.result.rows
    out.notes = [f"口径说明：{explanation}"] + ([] if out.rows else [EMPTY_NOTE])
    return out


def _failed(out: L3Result, reason: str, sql: str | None = None) -> L3Result:
    out.error, out.sql = reason, sql
    return out


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
