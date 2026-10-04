import pytest

from text2sql.config import BIRD_DB_PATH, RISK_DB_PATH, settings
from text2sql.db.executor import ErrorKind, SQLiteExecutor, classify_error, is_repairable

pytestmark = pytest.mark.skipif(not RISK_DB_PATH.exists(), reason="data/bank_risk.sqlite missing")


@pytest.fixture(scope="module")
def ex():
    return SQLiteExecutor(RISK_DB_PATH, timeout_s=5, max_rows=10)


def test_execute_and_row_cap(ex):
    res = ex.execute("SELECT account_id FROM account ORDER BY account_id")
    assert res.ok and len(res.rows) == 10 and res.truncated and res.columns == ["account_id"]


def test_parameters(ex):
    res = ex.execute("SELECT COUNT(*) FROM loan WHERE status IN (?, ?)", ("B", "D"))
    assert res.rows == [(76,)]


def test_dry_run_catches_errors_without_executing(ex):
    assert ex.explain("SELECT COUNT(*) FROM trans").ok
    bad = ex.explain("SELECT amout FROM trans")
    assert not bad.ok and bad.error_kind is ErrorKind.NO_SUCH_COLUMN
    assert ex.explain("SELEC 1").error_kind is ErrorKind.SYNTAX
    assert ex.explain("SELECT * FROM transactions").error_kind is ErrorKind.NO_SUCH_TABLE


def test_read_only(ex):
    res = ex.execute("DELETE FROM loan")
    assert not res.ok and res.error_kind is ErrorKind.READ_ONLY


def test_timeout():
    slow = SQLiteExecutor(RISK_DB_PATH, timeout_s=0.2)
    res = slow.execute("SELECT COUNT(*) FROM trans a JOIN trans b ON a.amount = b.amount")
    assert not res.ok and res.error_kind is ErrorKind.TIMEOUT


def test_repairability_and_unknown_errors():
    assert not is_repairable(ErrorKind.NO_SUCH_TABLE) and not is_repairable(ErrorKind.READ_ONLY)
    assert is_repairable(ErrorKind.NO_SUCH_COLUMN) and is_repairable(ErrorKind.TIMEOUT)
    assert classify_error("disk I/O error") is ErrorKind.OTHER


@pytest.mark.skipif(not BIRD_DB_PATH.exists(), reason="data/financial.sqlite missing")
def test_evaluation_cap_avoids_truncating_large_gold_results():
    from text2sql.eval.dataset import load_bird_financial
    gold = next(i for i in load_bird_financial() if i.id == "BIRD-165").gold_sql
    display = SQLiteExecutor(BIRD_DB_PATH).execute(gold)
    evaluation = SQLiteExecutor(BIRD_DB_PATH, max_rows=settings.eval_max_rows).execute(gold)
    assert display.truncated and len(display.rows) == settings.max_rows
    assert not evaluation.truncated and len(evaluation.rows) == 15140
