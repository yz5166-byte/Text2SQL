"""The outcome of routing one question, shared by the rules, the parser and normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from text2sql.config import DATA_END_DATE, DATA_START_DATE
from text2sql.llm.base import Usage
from text2sql.pipeline.parser import ParseResult
from text2sql.semantic.ir import QueryIR
from text2sql.semantic.okr import l1_ineligibility
from text2sql.semantic.registry import SemanticModel

Route = Literal["L1", "L2", "L3", "CLARIFY", "OOD", "META"]
NO_TIME_NOTE = f"未指定时间，统计全部历史数据（{DATA_START_DATE} 至 {DATA_END_DATE}）"


@dataclass
class RouteDecision:
    route: Route
    source: Literal["rule", "fast_path", "llm"]
    reason: str
    ir: QueryIR | None = None
    parse: ParseResult | None = None
    clarify_kind: Literal["metric", "time"] | None = None  # choose among metrics, or supply a usable time
    clarify_options: list[str] = field(default_factory=list)  # candidate metrics of a metric clarification
    assumptions: list[str] = field(default_factory=list)  # every default or repair, disclosed in the answer
    time_hint: tuple[date, date] | None = None  # resolved time range handed to layer 3
    parse_problems: list[str] = field(default_factory=list)  # why parser replies were rejected, for the trace
    usage: Usage = field(default_factory=Usage)


def semantic_layer(ir: QueryIR, model: SemanticModel) -> Route:
    """Layer 1 when the OKR table can answer the IR, otherwise layer 2."""
    return "L1" if l1_ineligibility(ir, model) is None else "L2"
