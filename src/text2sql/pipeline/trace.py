"""Per-request trace: one step per decision or call, with its duration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TraceStep:
    name: str
    detail: str
    elapsed_ms: float
