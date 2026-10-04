"""Routing: rules first, then one LLM parse, then deterministic normalization.

1. rules.py      no model call: metadata, out-of-domain, ambiguous terms, vague time, fast path
2. parser.py     one structured LLM call, validated against the semantic layer, retried once
3. normalize.py  parse -> route and QueryIR; every IR is then fitted to the data range
"""

from __future__ import annotations

from text2sql.llm.base import LLMClient, LLMError, Usage
from text2sql.pipeline.decision import RouteDecision
from text2sql.pipeline.normalize import fit_to_data_range, normalize_parse
from text2sql.pipeline.parser import parse_question
from text2sql.pipeline.rules import Rules
from text2sql.semantic.registry import SemanticModel


class Router:
    def __init__(self, model: SemanticModel, llm: LLMClient | None, use_fast_path: bool = True):
        self.model = model
        self.llm = llm
        self.rules = Rules(model, use_fast_path)

    def route(self, question: str) -> RouteDecision:
        decision = self.rules.decide(question)
        usage = Usage()
        if decision is None:
            if self.llm is None:
                raise LLMError("规则未能判定路由，且未配置 LLM（在 .env 中设置 API Key）")
            parse, usage = parse_question(self.llm, self.model, question)
            decision = normalize_parse(question, parse, self.model) if parse else RouteDecision(
                "L3", "llm", "语义解析两次输出不合格，降级生成式 SQL", assumptions=["语义解析未成功，改用生成式 SQL 回答"])
        decision = fit_to_data_range(decision, self.model)
        decision.usage = usage
        return decision
