"""LLM providers behind one interface; select with LLM_PROVIDER (deepseek | claude)."""

from __future__ import annotations

from text2sql.config import settings
from text2sql.llm.base import LLMClient, LLMError, LLMReply, Usage


def get_llm(provider: str | None = None, think_for_generate: bool | None = None, model: str | None = None) -> LLMClient:
    """`think_for_generate` switches DeepSeek's thinking mode for SQL generation (experiment X3);
    Claude always thinks adaptively, so it ignores the switch."""
    provider = provider or settings.provider
    if provider == "deepseek":
        from text2sql.llm.deepseek import DeepSeekClient

        return DeepSeekClient(model, think_for_generate=think_for_generate)
    if provider == "claude":
        from text2sql.llm.claude import ClaudeClient

        return ClaudeClient(model)
    raise LLMError(f"unknown LLM provider: {provider}")


__all__ = ["LLMClient", "LLMError", "LLMReply", "Usage", "get_llm"]
