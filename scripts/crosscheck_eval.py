"""Cross-check gold answers through independent computation paths (勾稽校验).

A gold SQL can execute and still encode the wrong interpretation. These checks recompute
the same quantity along a different path (a breakdown must sum to its total, a baseline
period must equal the direct query of that period, a ratio must equal its components) and
fail loudly when two paths disagree.
"""

from __future__ import annotations

import sqlite3
import sys

from text2sql.config import RISK_DB_PATH
from text2sql.eval.dataset import load_domain_eval

items = {i.id: i for i in load_domain_eval()}
con = sqlite3.connect(f"file:{RISK_DB_PATH}?mode=ro", uri=True)


def gold(item_id: str) -> list[tuple]:
    return con.execute(items[item_id].gold_sql).fetchall()


def scalar(sql: str):
    return con.execute(sql).fetchone()[0]


def col_sum(rows: list[tuple], idx: int = -1):
    return sum(r[idx] for r in rows)


CHECKS = [
    ("L1-04 各地区不良金额之和 = L1-03 全国不良金额",
     lambda: col_sum(gold("L1-04")), lambda: gold("L1-03")[0][0]),
    ("L1-05 1998 各月事件数之和 = 1998 全年事件数（直接计数）",
     lambda: col_sum(gold("L1-05")),
     lambda: scalar("SELECT COUNT(*) FROM dwd_risk_event WHERE substr(event_date,1,4) = '1998'")),
    ("L2-05 1998 各规则事件数之和 = 1998 全年事件数",
     lambda: col_sum(gold("L2-05")),
     lambda: scalar("SELECT COUNT(*) FROM dwd_risk_event WHERE substr(event_date,1,4) = '1998'")),
    ("L2-19 1998 月份 × 风险大类之和 = 1998 全年事件数",
     lambda: col_sum(gold("L2-19")),
     lambda: scalar("SELECT COUNT(*) FROM dwd_risk_event WHERE substr(event_date,1,4) = '1998'")),
    ("L1-11 环比的本期值 = L1-01 1998 年 11 月事件数",
     lambda: gold("L1-11")[0][0], lambda: gold("L1-01")[0][0]),
    ("L1-14 近三个月 = L1-05 中 10 至 12 月之和",
     lambda: gold("L1-14")[0][0], lambda: sum(n for m, n in gold("L1-05") if m >= "1998-10")),
    ("L1-15 同比基期值 = 1996 全年事件数（直接计数）",
     lambda: gold("L1-15")[0][1],
     lambda: scalar("SELECT COUNT(*) FROM dwd_risk_event WHERE substr(event_date,1,4) = '1996'")),
    ("L1-07 1996 不良率 = 不良笔数 / 放款笔数",
     lambda: round(gold("L1-07")[0][0], 10),
     lambda: round(scalar("SELECT COUNT(*) FROM loan WHERE status IN ('B','D') AND substr(date,1,4)='1996'")
                   / scalar("SELECT COUNT(*) FROM loan WHERE substr(date,1,4)='1996'"), 10)),
    ("L1-13 各地区不良率按放款笔数加权 = 1997 全国不良率",
     lambda: round(sum(rate * scalar(
         "SELECT COUNT(*) FROM loan l JOIN account a ON a.account_id=l.account_id "
         f"JOIN district d ON d.district_id=a.district_id WHERE d.A3='{region}' AND substr(l.date,1,4)='1997'")
         for region, rate in gold("L1-13")) / scalar("SELECT COUNT(*) FROM loan WHERE substr(date,1,4)='1997'"), 10),
     lambda: round(scalar("SELECT AVG(status IN ('B','D')) FROM loan WHERE substr(date,1,4)='1997'"), 10)),
    ("L2-03 入账 + 出账 = 1998 全部交易金额（出账含 VYBER）",
     lambda: sum(gold("L2-03")[0]),
     lambda: scalar("SELECT SUM(amount) FROM trans WHERE substr(date,1,4) = '1998'")),
    ("L2-02 1997 各交易方式笔数之和（含利息 NULL 组）= 1997 全部笔数",
     lambda: col_sum(gold("L2-02")),
     lambda: scalar("SELECT COUNT(*) FROM trans WHERE substr(date,1,4) = '1997'")),
    ("L2-17 各状态贷款笔数之和 = 贷款总数",
     lambda: col_sum(gold("L2-17")), lambda: scalar("SELECT COUNT(*) FROM loan")),
    ("L2-10 男女不良笔数之和 = 不良贷款总数（经 OWNER 关联无重复）",
     lambda: col_sum(gold("L2-10")), lambda: scalar("SELECT COUNT(*) FROM loan WHERE status IN ('B','D')")),
    ("L2-08 各卡等级之和 = 卡总数", lambda: col_sum(gold("L2-08")), lambda: scalar("SELECT COUNT(*) FROM card")),
    ("L2-25 各账单频率之和 = 账户总数", lambda: col_sum(gold("L2-25")), lambda: scalar("SELECT COUNT(*) FROM account")),
    ("L2-06 反洗钱类金额 = 大额现金取现 + 快进快出金额（1998）",
     lambda: gold("L2-06")[0][0],
     lambda: scalar("SELECT SUM(amount) FROM dwd_risk_event WHERE rule_code IN ('LARGE_CASH','RAPID_IN_OUT') "
                    "AND substr(event_date,1,4) = '1998'")),
    ("L2-12 各季度活跃账户数之和 > 全年去重数（验证不可加性）",
     lambda: col_sum(gold("L2-12")) > scalar(
         "SELECT COUNT(DISTINCT account_id) FROM trans WHERE substr(date,1,4) = '1996'"), lambda: True),
    ("L3-16 无事件账户数 = 账户总数 - 有事件账户数",
     lambda: gold("L3-16")[0][0],
     lambda: scalar("SELECT COUNT(*) FROM account") - scalar("SELECT COUNT(DISTINCT account_id) FROM dwd_risk_event")),
    ("L3-20 有 UVER 代扣且不良的账户数 = 不良贷款总数（不良贷款账户均有还款代扣）",
     lambda: gold("L3-20")[0][0], lambda: scalar("SELECT COUNT(*) FROM loan WHERE status IN ('B','D')")),
    ("L3-24 1998 大额取现账户出账额 ≤ 1998 全部出账额",
     lambda: gold("L3-24")[0][0] <= gold("L2-03")[0][1], lambda: True),
]


def main() -> int:
    failed = 0
    for name, left, right in CHECKS:
        lv, rv = left(), right()
        ok = lv == rv
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  [{lv} vs {rv}]")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
