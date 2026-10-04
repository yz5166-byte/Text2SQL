"""Build data/bank_risk.sqlite: the BIRD financial tables plus a DWD risk-event layer.

The ODS layer is the untouched BIRD data. The DWD layer `dwd_risk_event` is derived
deterministically from real transactions with three explainable monitoring rules.
Thresholds come from the data profile (docs/DATA_PROFILE.md):

LARGE_CASH         cash withdrawal (operation = 'VYBER') with amount >= 50,000
                   (about the 99.5th percentile of cash withdrawals)
RAPID_IN_OUT       credit >= 30,000 followed, within the same day and the next 2 days,
                   by debits on the same account totalling >= 90% of the credit
OVERDRAFT_PENALTY  sanction interest posted because the balance was negative
                   (k_symbol = 'SANKC. UROK')

The ADS layer (ads_okr_monthly) is then materialized from the semantic layer with the
layer-2 compiler (text2sql.semantic.materialize), so its numbers share one definition with layer 2.
"""

from __future__ import annotations

import shutil
import sqlite3

from text2sql.config import BIRD_DB_PATH, RISK_DB_PATH
from text2sql.semantic.materialize import materialize_okr
from text2sql.semantic.registry import load_semantic_model

LARGE_CASH_MIN = 50_000
RAPID_IN_MIN = 30_000
RAPID_WINDOW_DAYS = 2
RAPID_OUT_RATIO = 0.9

INDEXES = [
    "CREATE INDEX ix_trans_account_date ON trans(account_id, date)",
    "CREATE INDEX ix_trans_date ON trans(date)",
    "CREATE INDEX ix_account_district ON account(district_id)",
    "CREATE INDEX ix_loan_account ON loan(account_id)",
    "CREATE INDEX ix_disp_account ON disp(account_id)",
    "CREATE INDEX ix_disp_client ON disp(client_id)",
    "CREATE INDEX ix_card_disp ON card(disp_id)",
    "CREATE INDEX ix_order_account ON \"order\"(account_id)",
]

DWD_DDL = """
CREATE TABLE dwd_risk_event (
    event_id      INTEGER PRIMARY KEY,
    rule_code     TEXT    NOT NULL,
    rule_name     TEXT    NOT NULL,
    risk_category TEXT    NOT NULL,
    account_id    INTEGER NOT NULL REFERENCES account (account_id),
    district_id   INTEGER NOT NULL REFERENCES district (district_id),
    trans_id      INTEGER NOT NULL REFERENCES trans (trans_id),
    event_date    DATE    NOT NULL,
    amount        INTEGER NOT NULL
)
"""

DWD_INSERT = f"""
INSERT INTO dwd_risk_event
    (rule_code, rule_name, risk_category, account_id, district_id, trans_id, event_date, amount)
SELECT rule_code, rule_name, risk_category, account_id, district_id, trans_id, event_date, amount
FROM (
    SELECT 'LARGE_CASH' AS rule_code, '大额现金取现' AS rule_name, 'AML' AS risk_category,
           t.account_id, a.district_id, t.trans_id, t.date AS event_date, t.amount
    FROM trans t
    JOIN account a ON a.account_id = t.account_id
    WHERE t.operation = 'VYBER' AND t.amount >= {LARGE_CASH_MIN}

    UNION ALL

    SELECT 'RAPID_IN_OUT', '快进快出', 'AML',
           t.account_id, a.district_id, t.trans_id, t.date, t.amount
    FROM trans t
    JOIN account a ON a.account_id = t.account_id
    WHERE t.type = 'PRIJEM'
      AND t.amount >= {RAPID_IN_MIN}
      AND (SELECT COALESCE(SUM(o.amount), 0)
           FROM trans o
           WHERE o.account_id = t.account_id
             AND o.type IN ('VYDAJ', 'VYBER')
             AND o.date BETWEEN t.date AND date(t.date, '+{RAPID_WINDOW_DAYS} day')
          ) >= {RAPID_OUT_RATIO} * t.amount

    UNION ALL

    SELECT 'OVERDRAFT_PENALTY', '透支罚息', 'CREDIT',
           t.account_id, a.district_id, t.trans_id, t.date, t.amount
    FROM trans t
    JOIN account a ON a.account_id = t.account_id
    WHERE t.k_symbol = 'SANKC. UROK'
)
ORDER BY event_date, trans_id, rule_code
"""


def build() -> None:
    if not BIRD_DB_PATH.exists():
        raise SystemExit("data/financial.sqlite missing: run scripts/download_data.py first")
    RISK_DB_PATH.unlink(missing_ok=True)
    shutil.copyfile(BIRD_DB_PATH, RISK_DB_PATH)
    con = sqlite3.connect(RISK_DB_PATH)
    try:
        for ddl in INDEXES:
            con.execute(ddl)
        con.execute(DWD_DDL)
        con.execute(DWD_INSERT)
        con.execute("CREATE INDEX ix_risk_event_date ON dwd_risk_event(event_date)")
        con.execute("CREATE INDEX ix_risk_event_account ON dwd_risk_event(account_id)")
        con.commit()
        con.execute("ANALYZE")
        rows = con.execute(
            "SELECT rule_code, COUNT(*), COUNT(DISTINCT account_id), SUM(amount) "
            "FROM dwd_risk_event GROUP BY rule_code ORDER BY rule_code"
        ).fetchall()
    finally:
        con.close()
    print(f"built {RISK_DB_PATH}")
    for rule_code, events, accounts, amount in rows:
        print(f"  {rule_code:<18} events={events:>6} accounts={accounts:>5} amount={amount:>12,}")
    n = materialize_okr(RISK_DB_PATH, load_semantic_model())
    print(f"  ads_okr_monthly    rows={n}")


if __name__ == "__main__":
    build()
