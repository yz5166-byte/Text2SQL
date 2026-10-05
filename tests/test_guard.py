"""Statement guard: what it admits, what it rejects and how, and the LIMIT cap."""

import sqlite3
from collections import Counter

import pytest

from text2sql.config import BIRD_DB_PATH, RAW_DIR, RISK_DB_PATH
from text2sql.db.executor import ErrorKind, is_repairable
from text2sql.db.guard import guard_sql
from text2sql.eval.dataset import load_bird_financial, load_domain_eval
from text2sql.semantic.registry import load_semantic_model

TABLES = set(load_semantic_model().tables)
CAP = 1000


def check(sql: str, max_rows: int = CAP):
    return guard_sql(sql, TABLES, max_rows)


@pytest.mark.parametrize("sql, kind", [
    ("DELETE FROM trans", ErrorKind.NOT_QUERY),
    ("WITH x AS (SELECT 1) UPDATE loan SET amount = 0", ErrorKind.NOT_QUERY),
    ("PRAGMA query_only = OFF", ErrorKind.NOT_QUERY),
    ("ATTACH DATABASE '/tmp/a.db' AS x", ErrorKind.NOT_QUERY),
    ("VACUUM INTO '/tmp/c.db'", ErrorKind.NOT_QUERY),
    ("SELECT 1; DELETE FROM loan", ErrorKind.GUARD),
    ("SELECT name, sql FROM sqlite_master", ErrorKind.GUARD),
    ("SELECT account_id FROM account UNION SELECT sql FROM sqlite_master", ErrorKind.GUARD),
    ("WITH loan AS (SELECT * FROM sqlite_master) SELECT * FROM loan", ErrorKind.GUARD),
    ("SELECT * FROM pragma_table_info('loan')", ErrorKind.GUARD),
])
def test_attacks_are_rejected_with_the_right_kind(sql, kind):
    r = check(sql)
    assert not r.ok and r.error_kind == kind and r.reason
    assert is_repairable(r.error_kind) == (kind == ErrorKind.GUARD)  # a write request is never rewritten


def test_the_reason_names_the_problem():
    assert check("VACUUM INTO '/tmp/c.db'").reason == "只允许查询语句，不能执行 VACUUM"
    assert check("SELECT * FROM loan JOIN secrets USING (id)").reason == "表 ['secrets'] 不在数据字典中"


@pytest.mark.parametrize("sql, expected", [
    ("SELECT * FROM loan", "SELECT * FROM loan LIMIT 1001"),  # no LIMIT: capped
    ("SELECT * FROM loan LIMIT 50000", "SELECT * FROM loan LIMIT 1001"),  # above the cap: lowered
    ("SELECT * FROM loan LIMIT -1", "SELECT * FROM loan LIMIT 1001"),  # -1 means no limit in SQLite
    ("SELECT * FROM loan LIMIT 99999 OFFSET 10", "SELECT * FROM loan LIMIT 1001 OFFSET 10"),
    ("SELECT account_id FROM loan UNION SELECT account_id FROM card_owner_free LIMIT 5", None),  # rejected
])
def test_outer_limit_is_capped(sql, expected):
    r = check(sql)
    assert (r.sql if r.ok else None) == expected


def test_queries_within_the_cap_are_returned_verbatim():
    for sql in ["SeLeCt account_id FROM loan /* top */ ORDER BY amount DESC LIMIT 10",
                "WITH big AS (SELECT * FROM loan WHERE amount > 1e5) SELECT COUNT(*) FROM big LIMIT 1"]:
        assert check(sql).sql == sql


def test_cap_follows_the_executor_row_limit():
    assert check("SELECT * FROM trans", max_rows=50_000).sql == "SELECT * FROM trans LIMIT 50001"


# ---------------------------------------------------------------- gold SQL: no false rejections, same results
def gold_sql() -> list[tuple[str, str]]:
    items = [(RISK_DB_PATH, i.gold_sql) for i in load_domain_eval() if i.gold_sql]
    if (RAW_DIR / "bird_dev_20251106.json").exists():
        items += [(BIRD_DB_PATH, i.gold_sql) for i in load_bird_financial()]
    return items


def test_gold_sql_is_never_rejected():
    rejected = [(sql, r.reason) for _, sql in gold_sql() if not (r := guard_sql(sql, TABLES, 50_000)).ok]
    assert rejected == []


@pytest.mark.slow
@pytest.mark.skipif(not RISK_DB_PATH.exists(), reason="databases missing")
def test_rewritten_gold_sql_keeps_its_result():
    """About 30 s: runs each rewritten gold statement and its original (uv run --group dev pytest -m slow)."""
    rewritten = 0
    for db, sql in gold_sql():
        r = guard_sql(sql, TABLES, max_rows=50_000)  # the evaluation cap
        if r.sql != sql:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            assert Counter(con.execute(r.sql).fetchall()) == Counter(con.execute(sql).fetchall()), sql
            con.close()
            rewritten += 1
    assert rewritten > 0
