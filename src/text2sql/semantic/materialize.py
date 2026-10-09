"""Materialize the OKR metrics into ads_okr_monthly with the layer-2 compiler.

Every value comes from an ordinary layer-2 query, so the OKR table and layer 2 share one
definition of each metric. Ratios keep numerator and denominator, so any coarser grain is
recomputed as sum(numerator) / sum(denominator).

The table is rebuilt in full on every run inside one transaction, so a failed build leaves the
previous table in place. Only cells with data are written, as layer 2 lists only groups that
have data; the test suite checks that both layers return the same numbers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

from text2sql.semantic.compiler import compile_l2
from text2sql.semantic.ir import QueryIR
from text2sql.semantic.okr import ADS_TABLE
from text2sql.semantic.registry import Metric, SemanticModel

ADS_DDL = f"""
CREATE TABLE {ADS_TABLE} (
    stat_month   TEXT NOT NULL,
    region       TEXT NOT NULL,
    metric_code  TEXT NOT NULL,
    metric_value REAL,
    numerator    REAL,
    denominator  REAL,
    PRIMARY KEY (stat_month, region, metric_code)
)
"""


def materialize_okr(db_path: Path, model: SemanticModel) -> int:
    """Rebuild the OKR table atomically: on any failure the previous table is kept."""
    con = sqlite3.connect(db_path, isolation_level=None)  # transactions are opened explicitly
    try:
        con.execute("BEGIN")
        con.execute(f"DROP TABLE IF EXISTS {ADS_TABLE}")
        con.execute(ADS_DDL)
        rows = [row for metric in model.okr_metrics() for row in _metric_rows(con, model, metric)]
        con.executemany(f"INSERT INTO {ADS_TABLE} VALUES (?, ?, ?, ?, ?, ?)", rows)
        con.execute("COMMIT")
        return len(rows)
    except Exception:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def _metric_rows(con: sqlite3.Connection, model: SemanticModel, metric: Metric) -> Iterator[tuple]:
    """Rows of one metric: per month for all regions ('ALL') and per month and region; only cells with data."""
    parts = [metric.numerator, metric.denominator] if metric.agg == "ratio" else [metric.key]
    for dims in ([], ["region"]):
        (query,) = compile_l2(QueryIR(metrics=parts, query_type="trend", grain="month", dimensions=dims), model)
        for row in con.execute(query.sql, query.params):
            rec = dict(zip(query.columns, row))
            month, region = rec["period"], rec.get("region", "ALL")
            if metric.agg == "ratio":
                num, den = rec[metric.numerator], rec[metric.denominator]
                yield month, region, metric.key, num / den if den else None, num, den
            else:
                yield month, region, metric.key, rec[metric.key], None, None
