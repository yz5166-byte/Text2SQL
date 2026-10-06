"""Model prices and the cost of one request, in USD per million tokens.

DeepSeek: api-docs.deepseek.com/quick_start/pricing, checked 2026-10-04. Peak rates are used
throughout as an upper bound (off-peak is half), so a cost does not depend on when it ran.
Claude: platform.claude.com/docs/en/about-claude/pricing, checked 2026-10-05. Cache writes
(1.25x input) are not separated in Usage, so Claude costs are slightly understated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float  # input not served from the provider's prefix cache
    cached_input: float  # input served from that cache
    output: float


PRICES = {
    "deepseek-flash": Price(input=0.30, cached_input=0.006, output=1.20),
    "claude-opus-5": Price(input=5.00, cached_input=0.50, output=25.00),
    "claude-opus-5-5": Price(input=4.00, cached_input=0.20, output=20.00),
    "claude-sonnet-5": Price(input=2.00, cached_input=0.20, output=10.00),
    "claude-sonnet-5-5": Price(input=2.00, cached_input=0.20, output=10.00),
}


def price_for(llm_name: str | None) -> Price | None:
    """Price of a client named "provider:model" (deepseek:deepseek-flash); None when unknown or no model."""
    return PRICES.get(llm_name.split(":", 1)[-1]) if llm_name else None


def request_cost(price: Price, input_tokens: int, cached_input_tokens: int, output_tokens: int) -> float:
    """`input_tokens` includes the cached ones, as Usage records them."""
    return ((input_tokens - cached_input_tokens) * price.input + cached_input_tokens * price.cached_input
            + output_tokens * price.output) / 1_000_000
