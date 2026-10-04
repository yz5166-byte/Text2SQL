"""LLM providers behind one interface; select with LLM_PROVIDER (deepseek | claude)."""

from __future__ import annotations

from text2sql.config import settings
from text2sql.llm.base import LLMClient, LLMError, LLMReply, Usage


def get_llm(provider: str | None = None) -> LLMClient:
    provider = provider or settings.provider
    if provider == "deepseek":
        from text2sql.llm.deepseek import DeepSeekClient

        return DeepSeekClient()
    if provider == "claude":
        from text2sql.llm.claude import ClaudeClient

        return ClaudeClient()
    raise LLMError(f"unknown LLM provider: {provider}")


__all__ = ["LLMClient", "LLMError", "LLMReply", "Usage", "get_llm"]
