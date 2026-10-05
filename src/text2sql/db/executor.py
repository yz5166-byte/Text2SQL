"""Read-only SQL execution: dry run, timeout, row cap and error classification.

EXPLAIN QUERY PLAN plays the role of the production dry run (the statement is compiled,
not executed); a progress handler enforces the timeout; every failure is mapped to an
ErrorKind that the layer-3 repair loop uses to decide whether a rewrite can help.

Decisions (see the progress board): read-only at the database level, with a statement
guard added in layer 3; EXPLAIN is used only to catch errors; 1,000 rows for display and
a larger cap for evaluation; a missing table is not sent back for repair.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from text2sql.config import settings

Params = Sequence[Any] | Mapping[str, Any]


class ErrorKind(str, Enum):
    SYNTAX = "syntax"
    NO_SUCH_COLUMN = "no_such_column"
    AMBIGUOUS_COLUMN = "ambiguous_column"
    TYPE = "type"
    TIMEOUT = "timeout"
    NO_SUCH_TABLE = "no_such_table"
    READ_ONLY = "read_only"
    GUARD = "guard"  # set by the layer-3 guard, before execution: a fixable generation error
    NOT_QUERY = "not_query"  # set by the layer-3 guard: a write or other non-query statement
    OTHER = "other"


# A rewrite cannot fix a table that the catalog declares but the database lacks
# (infrastructure), nor a write or other non-query statement (policy: the request itself
# asks for something the system does not do). Every other failure gets another try.
NOT_REPAIRABLE = {ErrorKind.NO_SUCH_TABLE, ErrorKind.READ_ONLY, ErrorKind.NOT_QUERY}

# (fragments of the SQLite message, kind); the first matching row wins.
_SIGNATURES = [
    (("syntax error", "incomplete input", "unrecognized token"), ErrorKind.SYNTAX),
    (("no such column",), ErrorKind.NO_SUCH_COLUMN),
    (("ambiguous column",), ErrorKind.AMBIGUOUS_COLUMN),
    (("no such table",), ErrorKind.NO_SUCH_TABLE),
    (("interrupted",), ErrorKind.TIMEOUT),
    (("readonly", "read-only", "attempt to write"), ErrorKind.READ_ONLY),
    (("datatype mismatch", "wrong number of arguments", "no such function"), ErrorKind.TYPE),
]


def classify_error(message: str) -> ErrorKind:
    msg = message.lower()
    return next((kind for fragments, kind in _SIGNATURES if any(f in msg for f in fragments)), ErrorKind.OTHER)


def is_repairable(kind: ErrorKind) -> bool:
    return kind not in NOT_REPAIRABLE


@dataclass
class ExecResult:
    ok: bool
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False  # the query had more rows than max_rows
    elapsed_s: float = 0.0
    error: str | None = None
    error_kind: ErrorKind | None = None


class SQLiteExecutor:
    """Runs statements on a read-only connection, so any write fails inside SQLite itself."""

    def __init__(self, db_path: Path, timeout_s: float | None = None, max_rows: int | None = None):
        self.db_path = Path(db_path)
        self.timeout_s = settings.sql_timeout_s if timeout_s is None else timeout_s
        self.max_rows = settings.max_rows if max_rows is None else max_rows

    def explain(self, sql: str, params: Params = ()) -> ExecResult:
        """Dry run: compile the statement and return its plan without executing it."""
        return self.execute(f"EXPLAIN QUERY PLAN {sql}", params)

    def execute(self, sql: str, params: Params = ()) -> ExecResult:
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
        con.execute("PRAGMA query_only = ON")
        deadline = time.perf_counter() + self.timeout_s
        con.set_progress_handler(lambda: time.perf_counter() > deadline, 10_000)  # True aborts the query
        t0 = time.perf_counter()
        try:
            cur = con.execute(sql, params)
            rows = cur.fetchmany(self.max_rows + 1)  # one extra row reveals truncation
            return ExecResult(ok=True, columns=[d[0] for d in cur.description or []],
                              rows=rows[: self.max_rows], truncated=len(rows) > self.max_rows,
                              elapsed_s=time.perf_counter() - t0)
        except (sqlite3.Error, ValueError) as exc:  # ValueError: e.g. a NUL character in the statement
            return ExecResult(ok=False, elapsed_s=time.perf_counter() - t0,
                              error=str(exc), error_kind=classify_error(str(exc)))
        finally:
            con.close()
