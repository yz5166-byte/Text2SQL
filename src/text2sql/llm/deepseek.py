"""DeepSeek through its OpenAI-compatible API.

Parsing runs in non-thinking mode with temperature 0 and JSON output; generation can enable
thinking mode (DEEPSEEK_THINKING_FOR_GENERATE=1), which ignores temperature by design.
"""

from __future__ import annotations

import json
import os

import openai
from openai import OpenAI

from text2sql.config import settings
from text2sql.llm.base import LLMError, LLMReply, Usage, extract_json


class DeepSeekClient:
    def __init__(self, model: str | None = None, think_for_generate: bool | None = None):
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise LLMError("DEEPSEEK_API_KEY is not set (copy .env.example to .env)")
        self.model = model or settings.deepseek_model
        self.name = f"deepseek:{self.model}"
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=120, max_retries=2)
        env = os.getenv("DEEPSEEK_THINKING_FOR_GENERATE", "0") == "1"
        self.think_for_generate = env if think_for_generate is None else think_for_generate

    def complete(self, system: str, user: str, *, schema: dict | None = None, purpose: str = "parse") -> LLMReply:
        thinking = purpose == "generate" and self.think_for_generate
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 4096,
            "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
        }
        if not thinking:
            kwargs["temperature"] = 0
        if schema is not None:
            # JSON mode needs the word "json" and an example of the format in the prompt.
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["messages"][0]["content"] += (
                "\n\n以 json 格式输出，结构必须符合以下 JSON Schema：\n" + json.dumps(schema, ensure_ascii=False)
            )
        usage = Usage()
        for _ in range(2):  # JSON mode may occasionally return empty content
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except openai.APIStatusError as exc:
                raise LLMError(f"DeepSeek API error {exc.status_code}: {exc.message}") from exc
            except openai.APIConnectionError as exc:
                raise LLMError(f"DeepSeek connection error: {exc}") from exc
            u = resp.usage
            usage.add(Usage(1, u.prompt_tokens, u.completion_tokens, getattr(u, "prompt_cache_hit_tokens", 0) or 0))
            text = resp.choices[0].message.content or ""
            if text.strip():
                data = extract_json(text) if schema is not None else None
                return LLMReply(text=text, data=data, usage=usage)
        raise LLMError("empty response from DeepSeek")
