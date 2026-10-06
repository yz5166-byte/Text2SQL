"""Claude through the Anthropic Python SDK (Messages API).

Structured replies use output_config.format (JSON schema); reasoning depth is set with
output_config.effort because current Claude models (Opus 5, Sonnet 5.5) think adaptively and reject temperature.
A refusal is recorded as such instead of silently falling back to another model, so the
model-comparison experiment always measures the model it names.
"""

from __future__ import annotations

import json

import anthropic

from text2sql.config import settings
from text2sql.llm.base import LLMError, LLMReply, Usage, extract_json

EFFORT = {"parse": "low", "summarize": "low", "generate": "medium"}


class ClaudeClient:
    def __init__(self, model: str | None = None):
        self.model = model or settings.claude_model
        self.name = f"claude:{self.model}"
        self.client = anthropic.Anthropic(timeout=300.0, max_retries=2)

    def complete(self, system: str, user: str, *, schema: dict | None = None, purpose: str = "parse") -> LLMReply:
        output_config: dict = {"effort": EFFORT.get(purpose, "medium")}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema_for_claude(schema)}
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config=output_config,
            )
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Claude connection error: {exc}") from exc
        u = resp.usage
        usage = Usage(1, u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
                      u.output_tokens, u.cache_read_input_tokens or 0)
        if resp.stop_reason == "refusal":
            return LLMReply(text="", usage=usage, refused=True)
        if resp.stop_reason == "max_tokens":
            raise LLMError("Claude reply truncated at max_tokens")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        data = extract_json(text) if schema is not None else None
        return LLMReply(text=text, data=data, usage=usage)


def schema_for_claude(schema: dict) -> dict:
    """Structured outputs require closed objects; add additionalProperties: false recursively."""
    schema = json.loads(json.dumps(schema))

    def close(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
            for value in node.values():
                close(value)
        elif isinstance(node, list):
            for value in node:
                close(value)

    close(schema)
    return schema
