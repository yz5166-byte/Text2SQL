"""OKR materialization: reproducible output and an atomic rebuild (run on a temporary copy)."""

import shutil
import sqlite3

import pytest

from text2sql.config import RISK_DB_PATH
from text2sql.semantic import materialize as mat
from text2sql.semantic.registry import load_semantic_model

pytestmark = pytest.mark.skipif(not RISK_DB_PATH.exists(), reason="data/bank_risk.sqlite missing")
model = load_semantic_model()


@pytest.fixture
def db(tmp_path):
    copy = tmp_path / "bank_risk.sqlite"
    shutil.copyfile(RISK_DB_PATH, copy)
    return copy


def table(path):
    return sorted(sqlite3.connect(path).execute("SELECT * FROM ads_okr_monthly"))


def test_rebuild_is_reproducible(db):
    before = table(db)
    assert mat.materialize_okr(db, model) == len(before) == 3848
    assert table(db) == before


def test_failed_rebuild_keeps_previous_table(db, monkeypatch):
    before = table(db)
    original = mat._metric_rows

    def broken(con, model, metric):
        if metric.key == "risk_amt":
            raise RuntimeError("simulated failure")
        yield from original(con, model, metric)

    monkeypatch.setattr(mat, "_metric_rows", broken)
    with pytest.raises(RuntimeError):
        mat.materialize_okr(db, model)
    assert table(db) == before
