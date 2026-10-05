"""Statement guard for generated SQL (layer 3), run before the dry run.

The read-only connection stops writes inside SQLite, but not everything: ATTACH creates a
file, VACUUM INTO creates its target before failing, and sqlite_master exposes the whole
schema. So the guard admits only what is known to be safe instead of banning what is known
to be dangerous: exactly one query (SELECT or a set operation, optionally with WITH), no
function in FROM, and only tables of the semantic catalog. It also caps the outermost
LIMIT at the executor's row limit, which lets SQLite keep only the top rows when sorting.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from text2sql.db.executor import ErrorKind


@dataclass
class GuardResult:
    ok: bool
    sql: str  # the statement to run (row cap applied), or the rejected statement as given
    reason: str | None = None  # why it was rejected, phrased for the repair prompt
    error_kind: ErrorKind | None = None  # NOT_QUERY is final; GUARD and SYNTAX may be rewritten


def guard_sql(sql: str, allowed_tables: Collection[str], max_rows: int) -> GuardResult:
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as exc:  # the message itself carries terminal colour codes
        err = exc.errors[0]
        return GuardResult(False, sql, f"无法解析 SQL：{err['description']}（第 {err['line']} 行第 {err['col']} 列）",
                           ErrorKind.SYNTAX)
    if len(statements) != 1:
        return GuardResult(False, sql, f"只允许一条语句，实际为 {len(statements)} 条", ErrorKind.GUARD)
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        verb = tree.this if isinstance(tree, exp.Command) else tree.key  # VACUUM parses as a generic command
        return GuardResult(False, sql, f"只允许查询语句，不能执行 {verb.upper()}", ErrorKind.NOT_QUERY)
    if any(not isinstance(t.this, exp.Identifier) for t in tree.find_all(exp.Table)):
        return GuardResult(False, sql, "FROM 中不允许使用函数", ErrorKind.GUARD)
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables}
    unknown = sorted({t.name.lower() for t in tree.find_all(exp.Table)} - ctes - allowed)
    if unknown:
        return GuardResult(False, sql, f"表 {unknown} 不在数据字典中", ErrorKind.GUARD)

    cap = max_rows + 1  # one extra row lets the executor detect truncation
    if _exceeds(tree.args.get("limit"), cap):
        return GuardResult(True, tree.limit(cap).sql(dialect="sqlite"))
    return GuardResult(True, sql)  # unchanged text when no rewrite is needed


def _exceeds(limit: exp.Limit | None, cap: int) -> bool:
    """No LIMIT, or a literal one above the cap; a negative LIMIT means no limit in SQLite."""
    if limit is None:
        return True
    value = limit.expression
    if isinstance(value, exp.Neg):
        return True
    if isinstance(value, exp.Literal) and not value.is_string:
        return int(value.this) > cap
    return False  # a computed LIMIT is left alone; the executor's row cap still applies
