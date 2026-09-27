"""Intermediate representation (IR) of a data question in business terms.

The LLM never emits table names, joins or partitions. It emits business slots that are
normalized into a QueryIR: metric codes, dimension codes, a resolved date range and
filter values already linked to database codes. The compiler turns a QueryIR into SQL.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Code = str | int | None
QueryType = Literal["point", "breakdown", "trend", "compare"]
Grain = Literal["day", "month", "quarter", "year"]


class QueryIR(BaseModel):
    metrics: list[str] = Field(min_length=1)
    query_type: QueryType
    time_range: tuple[date, date] | None = None  # inclusive; None means no time restriction
    grain: Grain | None = None  # required for trend
    dimensions: list[str] = []  # group-by dimensions (breakdown, or split series in a trend)
    filters: dict[str, list[Code]] = {}  # dimension -> linked database codes
    compare: Literal["yoy", "mom"] | None = None  # required for compare
    order: Literal["desc", "asc"] | None = None
    limit: int | None = None

    def slot_view(self) -> dict:
        """Order-insensitive view used for slot-level accuracy."""
        return {
            "metrics": sorted(self.metrics),
            "query_type": self.query_type,
            "time_range": self.time_range,
            "grain": self.grain,
            "dimensions": sorted(self.dimensions),
            "filters": {k: sorted(v, key=lambda c: (c is None, str(c))) for k, v in sorted(self.filters.items())},
            "compare": self.compare,
            "order_limit": (self.order, self.limit),
        }
