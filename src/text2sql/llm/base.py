"""Provider-neutral LLM interface used by the router (parsing) and layer 3 (SQL generation)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens


@dataclass
class LLMReply:
    text: str
    data: dict[str, Any] | None = None  # parsed JSON when a schema was requested; None if the text held none
    usage: Usage = field(default_factory=Usage)
    refused: bool = False


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str, *, schema: dict | None = None, purpose: str = "parse") -> LLMReply:
        """One chat completion. With `schema`, the reply is a JSON object matching it.

        `purpose` selects per-task settings: "parse" (routing and slot extraction, fast),
        "summarize" (one-sentence answer summary, fast) or "generate" (layer-3 SQL generation,
        more reasoning).
        """
        ...


def extract_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model text, tolerating a surrounding code fence; None if there is none.

    A malformed reply is a format problem for the caller to retry, not a service failure.
    """
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None
