"""Execution of compiled semantic-layer queries (layers 1 and 2)."""

from __future__ import annotations

from text2sql.db.executor import ExecResult, SQLiteExecutor
from text2sql.semantic.compiler import CompiledQuery


def run_queries(executor: SQLiteExecutor, queries: list[CompiledQuery]) -> ExecResult:
    """Execute one query per base table and outer-merge their rows on the shared key columns."""
    results = [executor.execute(q.sql, q.params) for q in queries]
    failed = next((r for r in results if not r.ok), None)
    if failed or len(results) == 1:
        return failed or results[0]
    keys = queries[0].keys
    merged: dict[tuple, list] = {}
    columns = list(keys)
    for query, res in zip(queries, results):
        value_cols = [c for c in res.columns if c not in keys]
        offset = len(columns) - len(keys)
        columns += value_cols
        for row in res.rows:
            record = dict(zip(res.columns, row))
            key = tuple(record[k] for k in keys)
            slot = merged.setdefault(key, [None] * offset)
            slot.extend([None] * (offset - len(slot)))
            slot.extend(record[c] for c in value_cols)
    width = len(columns) - len(keys)
    rows = [tuple(k) + tuple(v + [None] * (width - len(v))) for k, v in merged.items()]
    return ExecResult(ok=True, columns=columns, rows=rows, elapsed_s=sum(r.elapsed_s for r in results))
