"""A stand-in model for tests: replies with the given outputs in order and records what it was sent."""

from text2sql.llm.base import LLMReply, Usage


class ScriptedLLM:
    """A dict is a structured (JSON) reply, a str a text reply, None a reply that holds no JSON."""

    name = "scripted"

    def __init__(self, *replies: dict | str | None):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str, *, schema: dict | None = None, purpose: str = "parse") -> LLMReply:
        self.prompts.append(user)
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            return LLMReply(text=reply, usage=Usage(calls=1))
        return LLMReply(text="", data=reply, usage=Usage(calls=1))
