"""Routing: rules first, then one LLM parse, then deterministic normalization.

1. rules.py      no model call: metadata, out-of-domain, ambiguous terms, vague time, fast path
2. parser.py     one structured LLM call, validated against the semantic layer, retried once;
                 two invalid replies mean the question was not understood, so it is clarified
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
            parse, usage, problems = parse_question(self.llm, self.model, question)
            decision = normalize_parse(question, parse, self.model) if parse else RouteDecision(
                "CLARIFY", "llm", "未能理解问题，请换一种说法，或从常用指标中选择", clarify_kind="metric",
                clarify_options=[m.key for m in self.model.okr_metrics()])
            decision.parse_problems = problems
        decision = fit_to_data_range(decision, self.model)
        decision.usage = usage
        return decision
