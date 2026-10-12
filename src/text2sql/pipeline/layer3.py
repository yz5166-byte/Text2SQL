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

Column-level linking (off by default) finds the stored values the question names
(knowledge/values.py) and the dimensions it names by their semantic-layer names (地区 is
district.A3): their tables join the candidates, the prompt says which column holds each value,
and pruning keeps those columns. It also ranks the columns of the candidate tables, for the trace and for
the steps that select tables and columns.

Table selection (off by default) is one more model call before generation: the model sees a
card per candidate table (names, description, columns with their Chinese names) and the
linking hints, and keeps the tables the question needs. It only filters the candidates; the
join bridge is added back over the kept tables. An empty or malformed selection, a refusal or
a failed call leaves the candidates as they were.

Column pruning (off by default) is another call, on the final tables: the model sees their
full column lines and keeps the columns the question needs. Keys and the columns holding a
linked value are always kept. Generation then sees only the kept columns; a missing or
ambiguous column drops the pruning together with widening to the full schema. A failed or
empty pruning keeps every column.
"""

from __future__ import annotations

import json
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError

from text2sql.config import settings
from text2sql.db.executor import ErrorKind, ExecResult, SQLiteExecutor, is_repairable
from text2sql.db.guard import guard_sql
from text2sql.knowledge.examples import ExamplePicker, SQLExample, load_examples
from text2sql.knowledge.retriever import ColumnRetriever, Method, TableRetriever, bridge_tables, key_columns
from text2sql.knowledge.values import ValueHit, ValueIndex
from text2sql.llm.base import LLMClient, LLMError, Usage
from text2sql.pipeline.decision import RouteDecision
from text2sql.pipeline.meta import formula
from text2sql.pipeline.trace import TraceStep
from text2sql.semantic.linker import Code, find_dimension_mentions, find_metric_mentions, find_value_mentions
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
SELECT_SYSTEM = """你是数据库专家，为一个问题挑选写 SQL 必需的表。
1. 只能从给出的候选表中选择，表名照抄。
2. 选出取数、筛选条件和分组维度所在的表；只用于连接的中间表会自动补上，不必选。
3. 拿不准是否需要的表，保留。
以 json 输出：{"tables": ["表名"], "reason": "一句话说明"}"""

PRUNE_SYSTEM = """你是数据库专家，为一个问题挑选写 SQL 需要的字段。
1. 只能从给出的字段中选择，按"表名.字段名"照抄。
2. 选出要输出、筛选、分组、排序和参与计算的字段；主键、关联字段和问题中取值所在的字段会自动保留，不必选。
3. 拿不准是否需要的字段，保留。
以 json 输出：{"columns": ["表名.字段名"], "reason": "一句话说明"}"""

WIDEN_ON = {ErrorKind.NO_SUCH_COLUMN, ErrorKind.AMBIGUOUS_COLUMN}  # the needed table may not have been retrieved
MAX_VALUES_SHOWN = 20
COLUMNS_TRACED = 10
EMPTY_NOTE = "查询结果为空，未做修复；如与预期不符，请核对 SQL 中的筛选条件与取值写法"


@dataclass(frozen=True)
class Layer3Options:
    """Switches for the experiments (X2 to X4); the defaults are the production layer 3."""

    schema: Literal["retrieved", "full"] = "retrieved"
    retrieval: Method = "hybrid"  # how tables are retrieved when schema is "retrieved" (X4)
    top_k: int = 5  # tables kept from the ranking before value hits and the join bridge (X6)
    examples: int = 3  # similar examples in the prompt; 0 turns them off
    knowledge: bool = True  # official metric definitions and linked values
    max_repairs: int | None = None  # None: settings.max_repair_attempts
    column_linking: bool = False  # value index and column ranking (X6)
    table_selection: bool = False  # the model keeps the candidate tables the question needs (X6)
    column_pruning: bool = False  # the model keeps the columns of the final tables the question needs (X6)


# Plain text-to-SQL: the whole schema and the question, one generation, no repair (X2 baseline).
NAIVE = Layer3Options(schema="full", examples=0, knowledge=False, max_repairs=0)


M = TypeVar("M", bound=BaseModel)


class GeneratedSQL(BaseModel):
    sql: str = Field(min_length=1)
    explanation: str


class TableSelection(BaseModel):
    tables: list[str]
    reason: str = ""


class ColumnSelection(BaseModel):
    columns: list[str]
    reason: str = ""


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
    context_tables: list[str] = field(default_factory=list)  # handed to the first generation
    context_columns: list[str] | None = None  # "table.column" after pruning; None: every column of the tables


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
        self.retriever = retriever or TableRetriever(model, options.retrieval, options.top_k)
        self.examples = examples or ExamplePicker(load_examples())
        self.max_repairs = settings.max_repair_attempts if options.max_repairs is None else options.max_repairs
        self.values = ValueIndex.build(model, executor.db_path) if options.column_linking else None
        self.columns = ColumnRetriever(model, options.retrieval) if options.column_linking else None
        self.keys = key_columns(model)

    def answer(self, question: str, decision: RouteDecision, evidence: str | None = None) -> L3Result:
        out = L3Result(ok=False)
        query = f"{question} {evidence or ''}"
        t = time.perf_counter()
        hits = self.values.match(query) if self.values else []
        dimensions = self._dimensions(question) if self.options.column_linking else []
        if self.options.schema == "full":
            tables, context = list(self.model.tables), "全量表结构"
        else:
            linked = [table for h in hits for table in h.tables] + [c.split(".")[0] for _, c in dimensions]
            retrieval = self.retriever.retrieve(query, linked)
            linked = f"，链接命中 {retrieval.linked}" if retrieval.linked else ""
            tables, context = retrieval.all, f"表 {retrieval.tables}{linked}，连接补全 {retrieval.bridged}"
        examples = self.examples.pick(question, self.options.examples) if self.options.examples else []
        out.trace.append(TraceStep("retrieve", f"{context}，示例 {[e.id for e in examples]}", _ms(t)))
        ranked = []
        if self.columns:
            t = time.perf_counter()
            ranked = self.columns.rank(query, tables)[:COLUMNS_TRACED]
            found = [f"{h.surface} → {'、'.join(h.columns)}" + ("（模糊）" if h.fuzzy else "") for h in hits]
            named = [f"{surface} → {column}" for surface, column in dimensions]
            out.trace.append(TraceStep("link", f"取值 {found}，维度 {named}，字段前 {COLUMNS_TRACED} {ranked}", _ms(t)))
        if self.options.table_selection:
            tables = self._select_tables(out, question, evidence, tables, self._hints(question, hits, dimensions, ranked))
        columns = None  # every column of `tables`
        if self.options.column_pruning:
            hints = self._hints(question, hits, dimensions, [c for c in ranked if c.split(".")[0] in tables])
            forced = self._forced(question, hits) | {c for _, c in dimensions}
            columns = self._prune_columns(out, question, evidence, tables, hints, forced)
        out.context_tables, out.context_columns = list(tables), sorted(columns) if columns is not None else None
        request = self._request(question, decision, evidence, examples, hits)
        feedback, last = "", None
        for attempt_no in range(1, self.max_repairs + 2):
            out.attempts = attempt_no
            t = time.perf_counter()
            reply = self.llm.complete(SYSTEM, f"## 可用的表\n{schema_text(self.model, tables, columns)}\n\n{request}{feedback}",
                                      schema=SQL_SCHEMA, purpose="generate")
            out.usage.add(reply.usage)
            if reply.refused:
                return _failed(out, "模型拒绝回答该问题")
            generated = _validated(GeneratedSQL, reply.data)
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
            if attempt.error_kind in WIDEN_ON and (columns is not None or len(tables) < len(self.model.tables)):
                note = "，取消字段裁剪" if columns is not None else ""
                tables, columns = list(self.model.tables), None
                out.trace.append(TraceStep("widen", f"字段相关错误，修复改用全量表结构{note}", 0))
            feedback = _feedback(generated.sql, attempt.reason)
        reason = f"修复 {self.max_repairs} 次后仍未成功" + (f"：{last.reason}" if last else "")
        return _failed(out, reason, last.sql if last else None)

    def _select_tables(self, out: L3Result, question: str, evidence: str | None, tables: list[str],
                       hints: str) -> list[str]:
        """The candidate tables the model keeps, in candidate order, plus the join bridge over them."""
        t = time.perf_counter()
        ask = f"问题：{question}" + (f"\n外部知识：{evidence}" if evidence else "")
        user = f"## 候选表\n{table_cards(self.model, tables)}\n\n{hints}## 问题\n{ask}"
        try:
            reply = self.llm.complete(SELECT_SYSTEM, user, schema=choice_schema("tables", tables), purpose="select")
        except LLMError as exc:
            out.trace.append(TraceStep("select", f"选表调用失败，沿用候选表：{exc}", _ms(t)))
            return tables
        out.usage.add(reply.usage)
        selection = None if reply.refused else _validated(TableSelection, reply.data)
        named = {n.lower() for n in selection.tables} if selection else set()
        kept = [table for table in tables if table.lower() in named]
        if not kept:
            out.trace.append(TraceStep("select", "选表结果为空或不合格，沿用候选表", _ms(t)))
            return tables
        bridged = bridge_tables(kept, self.retriever.graph)
        dropped = [table for table in tables if table not in kept and table not in bridged]
        out.trace.append(TraceStep("select", f"保留 {kept}，连接补全 {bridged}，删去 {dropped}；理由：{selection.reason}",
                                   _ms(t)))
        return kept + bridged

    def _prune_columns(self, out: L3Result, question: str, evidence: str | None, tables: list[str], hints: str,
                       forced: set[str]) -> set[str] | None:
        """The columns the model keeps plus the forced ones, as "table.column"; None keeps every column."""
        t = time.perf_counter()
        names = {f"{table}.{c}".lower(): f"{table}.{c}" for table in tables for c in self.model.tables[table].columns}
        ask = f"问题：{question}" + (f"\n外部知识：{evidence}" if evidence else "")
        user = f"## 可用的表\n{schema_text(self.model, tables)}\n\n{hints}## 问题\n{ask}"
        try:
            reply = self.llm.complete(PRUNE_SYSTEM, user, schema=choice_schema("columns", list(names.values())),
                                      purpose="prune")
        except LLMError as exc:
            out.trace.append(TraceStep("prune", f"裁剪调用失败，保留全部字段：{exc}", _ms(t)))
            return None
        out.usage.add(reply.usage)
        selection = None if reply.refused else _validated(ColumnSelection, reply.data)
        chosen = {names[c.lower()] for c in selection.columns if c.lower() in names} if selection else set()
        if not chosen:
            out.trace.append(TraceStep("prune", "裁剪结果为空或不合格，保留全部字段", _ms(t)))
            return None
        forced = forced & set(names.values())
        kept = chosen | forced
        out.trace.append(TraceStep("prune", f"保留 {len(kept)}/{len(names)} 个字段：模型选择 {sorted(chosen - forced)}，"
                                            f"强制保留键字段与取值字段 {len(forced)} 个；理由：{selection.reason}", _ms(t)))
        return kept

    def _dimensions(self, question: str) -> list[tuple[str, str]]:
        """(surface, "table.column") for each dimension the question names, on the model's tables."""
        out = []
        for m in find_dimension_mentions(question, self.model):
            alias, column = self.model.dimensions[m.key].column.split(".")
            out.append((m.surface, f"{self.model.physical_table(alias)}.{column}"))
        return list(dict.fromkeys(out))

    def _forced(self, question: str, hits: list[ValueHit]) -> set[str]:
        """Keys and the columns holding a linked value: pruning always keeps them."""
        linked = {c for c, _ in self._linked_values(question)} if self.options.knowledge else set()
        return self.keys | {c for h in hits for c in h.columns} | linked

    def _hints(self, question: str, hits: list[ValueHit], dimensions: list[tuple[str, str]], ranked: list[str]) -> str:
        """Linking results shown to table selection and column pruning, each section ending with a blank line."""
        parts = []
        values = self._values(question) if self.options.knowledge else ""
        if values:
            parts.append("## 问题中识别到的业务取值\n" + values)
        if hits:
            parts.append("## 问题中出现的数据库取值\n" + _stored_values(hits))
        if dimensions:
            parts.append("## 问题中提到的维度（分组或筛选所在的字段）\n"
                         + "\n".join(f"- {surface} → {column}" for surface, column in dimensions))
        if ranked:
            parts.append("## 与问题最相关的字段（检索排序）\n" + "、".join(ranked))
        return "".join(p + "\n\n" for p in parts)

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
                 examples: list[SQLExample], hits: list[ValueHit] = ()) -> str:
        """Everything after the tables: definitions, values, examples and the question itself."""
        parts = []
        definitions = self._definitions(question, decision) if self.options.knowledge else ""
        if definitions:
            parts.append("## 官方口径（问题涉及这些指标时按此计算）\n" + definitions)
        values = self._values(question) if self.options.knowledge else ""
        if values:
            parts.append("## 问题中识别到的业务取值（写筛选条件时使用这些字段与编码）\n" + values)
        if hits:
            parts.append("## 问题中出现的数据库取值（筛选时使用所在字段，按存储写法）\n" + _stored_values(hits))
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
        return "\n".join(dict.fromkeys(line for _, line in self._linked_values(question)))

    def _linked_values(self, question: str) -> list[tuple[str, str]]:
        """("table.column", prompt line) for each business value the semantic layer's linker finds."""
        out = []
        for v in find_value_mentions(question, self.model, list(self.model.dimensions)):
            dim = self.model.dimensions[v.key]
            alias, column = dim.column.split(".")
            where = f"{self.model.physical_table(alias)}.{column}"
            out.append((where, f"- {v.surface} → {where} {_condition(v.codes)}（{dim.name}）"))
        return out


def _stored_values(hits: list[ValueHit]) -> str:
    lines = []
    for h in hits:
        where = "、".join(h.columns)
        note = "（模糊匹配，请核对问题原文）" if h.fuzzy else ""
        lines.append(f"- {h.surface} → {where} 中的取值 {h.value!r}{note}")
    return "\n".join(lines)


def _condition(codes: tuple[Code, ...]) -> str:
    """SQL predicate text for a value's codes: = 'x', IN ('x', 'y'), with IS NULL for a missing code."""
    literals = [repr(c) if isinstance(c, str) else str(c) for c in codes if c is not None]
    parts = [f"= {literals[0]}" if len(literals) == 1 else f"IN ({', '.join(literals)})"] if literals else []
    if None in codes:
        parts.append("IS NULL")
    return " 或 ".join(parts)


def table_cards(model: SemanticModel, tables: list[str]) -> str:
    """One line per table for the table selection: names, description, columns with Chinese names."""
    lines = []
    for t in tables:
        table = model.tables[t]
        columns = "、".join(f"{c} {col.name_zh}" for c, col in table.columns.items())
        lines.append(f"- {t}（{table.name_zh}）：{table.description} 字段：{columns}")
    return "\n".join(lines)


def choice_schema(name: str, options: list[str]) -> dict:
    """Output of a selection: the chosen `options` under `name` (constrained where the model supports it), and why."""
    return {
        "type": "object",
        "properties": {name: {"type": "array", "items": {"type": "string", "enum": options}},
                       "reason": {"type": "string"}},
        "required": [name, "reason"],
    }


def schema_text(model: SemanticModel, tables: list[str], columns: Collection[str] | None = None) -> str:
    """Table header, then one line per column: type, Chinese name, description, value codes.

    With `columns` ("table.column"), only those columns are listed.
    """
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
            if columns is not None and f"{t}.{c}" not in columns:
                continue
            values = f"；取值 {codes[(t, c)]}" if (t, c) in codes else ""
            lines.append(f"- {c} {col.type} {col.name_zh}：{col.description}{values}")
    return "\n".join(lines)


def _validated(cls: type[M], data: dict | None) -> M | None:
    try:
        return cls.model_validate(data) if data is not None else None
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
