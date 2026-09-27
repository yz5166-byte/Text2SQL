"""Validate the evaluation sets: structure, IR references, and execution of every gold SQL.

Usage: uv run python scripts/check_eval_set.py [--show]
"""

from __future__ import annotations

import sqlite3
import sys
import time
from collections import Counter

from text2sql.config import BIRD_DB_PATH, RISK_DB_PATH
from text2sql.eval.dataset import DATA_ROUTES, load_bird_financial, load_domain_eval, validate_items
from text2sql.semantic.registry import load_semantic_model


def run(con: sqlite3.Connection, sql: str) -> tuple[list[tuple], float]:
    t0 = time.perf_counter()
    rows = con.execute(sql).fetchall()
    return rows, time.perf_counter() - t0


def main(show: bool) -> int:
    model = load_semantic_model()
    items = load_domain_eval()
    problems = validate_items(items, model)

    print("domain eval:", len(items), "items")
    print("  by route:", dict(Counter(i.route for i in items)))
    print("  by split:", dict(Counter((i.route, i.split) for i in items)))

    con = sqlite3.connect(f"file:{RISK_DB_PATH}?mode=ro", uri=True)
    slow = []
    for item in items:
        if item.route not in DATA_ROUTES:
            continue
        try:
            rows, secs = run(con, item.gold_sql)
        except sqlite3.Error as exc:
            problems.append(f"{item.id}: gold SQL error: {exc}")
            continue
        if not rows or all(v is None for r in rows for v in r):
            problems.append(f"{item.id}: gold SQL returned no data")
        if secs > 2:
            slow.append((item.id, round(secs, 2)))
        if show:
            preview = rows[:4] if len(rows) > 4 else rows
            print(f"{item.id:<7} rows={len(rows):<3} {secs:5.2f}s {preview}")
    con.close()

    bird = load_bird_financial()
    bcon = sqlite3.connect(f"file:{BIRD_DB_PATH}?mode=ro", uri=True)
    bird_errors, bird_empty = 0, 0
    for item in bird:
        try:
            rows, _ = run(bcon, item.gold_sql)
            bird_empty += not rows
        except sqlite3.Error as exc:
            bird_errors += 1
            problems.append(f"{item.id}: BIRD gold SQL error: {exc}")
    bcon.close()
    print("BIRD financial:", len(bird), "items;", dict(Counter(i.difficulty for i in bird)),
          f"errors={bird_errors} empty_results={bird_empty}")

    if slow:
        print("slow gold SQL (>2s):", slow)
    for p in problems:
        print("PROBLEM", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(show="--show" in sys.argv))
