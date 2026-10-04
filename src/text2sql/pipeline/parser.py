"""LLM structured parsing: one call returns intent and IR slots, validated against the semantic layer.

An invalid reply is retried once with the problems found; a second invalid reply returns None.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator, model_validator

from text2sql.llm.base import LLMClient, LLMError, Usage
from text2sql.pipeline.prompts import parse_schema, parse_system_prompt
from text2sql.semantic.registry import SemanticModel


def _require_known(codes: list[str], known: dict, kind: str) -> list[str]:
    unknown = [c for c in codes if c not in known]
    if unknown:
        raise ValueError(f"{unknown} 不在可用{kind}中；问题需要语义层以外的指标或维度时，status 应为 uncovered")
    return codes


class ParsedFilter(BaseModel):
    dimension: str
    values: list[str]

    @field_validator("dimension")
    @classmethod
    def dimension_in_semantic_layer(cls, code: str, info: ValidationInfo) -> str:
        _require_known([code], info.context["model"].dimensions, "维度")
        return code


class ParseResult(BaseModel):
    """One parser reply, fields in output order. Validate with context={"model": SemanticModel}."""

    intent: Literal["data_query", "metadata", "out_of_domain"]
    metrics: list[str] = []
    time_text: str | None = None
    dimensions: list[str] = []
    filters: list[ParsedFilter] = []
    query_type: Literal["point", "breakdown", "trend", "compare"] | None = None
    grain: Literal["day", "month", "quarter", "year"] | None = None
    compare: Literal["yoy", "mom"] | None = None
    order: Literal["desc", "asc"] | None = None
    limit: int | None = Field(default=None, ge=1)
    status: Literal["covered", "uncovered", "clarify"] | None = None
    reason: str | None = None

    @field_validator("metrics")
    @classmethod
    def metrics_in_semantic_layer(cls, codes: list[str], info: ValidationInfo) -> list[str]:
        return _require_known(codes, info.context["model"].metrics, "指标")

    @field_validator("dimensions")
    @classmethod
    def dimensions_in_semantic_layer(cls, codes: list[str], info: ValidationInfo) -> list[str]:
        return _require_known(codes, info.context["model"].dimensions, "维度")

    @model_validator(mode="after")
    def judgment_is_complete(self) -> "ParseResult":
        if self.intent == "data_query" and self.status is None:
            raise ValueError("intent 为 data_query 时必须给出 status")
        if self.status == "covered" and self.query_type is None:  # always determinable when covered
            raise ValueError("status 为 covered 时必须给出 query_type")
        return self


def describe_errors(exc: ValidationError) -> str:
    """Numbered, readable validation errors, sent back to the model on retry."""
    lines = []
    for i, err in enumerate(exc.errors(), 1):
        field = ".".join(str(part) for part in err["loc"]) or "整体"
        lines.append(f"{i}. 字段 {field}：{err['msg'].removeprefix('Value error, ')}")
    return "\n".join(lines)


def parse_question(llm: LLMClient, model: SemanticModel, question: str) -> tuple[ParseResult | None, Usage]:
    """One call, plus one retry that carries the problems found; None after two invalid replies."""
    usage = Usage()
    user = f"问题：{question}"
    for _ in range(2):
        reply = llm.complete(parse_system_prompt(model), user, schema=parse_schema(model), purpose="parse")
        usage.add(reply.usage)
        if reply.refused:
            raise LLMError("模型拒绝回答该问题")
        if reply.data is None:
            problems = "1. 输出不是合法的 JSON 对象"
        else:
            try:
                return ParseResult.model_validate(reply.data, context={"model": model}), usage
            except ValidationError as exc:
                problems = describe_errors(exc)
        user = f"问题：{question}\n\n上一次输出不符合要求：\n{problems}\n请修正后重新输出。"
    return None, usage
