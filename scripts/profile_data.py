"""Profile data/bank_risk.sqlite and write docs/DATA_PROFILE.md.

Every number in the report is computed from the database, so the report can be
regenerated after any change to the warehouse build.
"""

from __future__ import annotations

import sqlite3

from text2sql.config import RISK_DB_PATH, ROOT

OUT = ROOT / "docs" / "DATA_PROFILE.md"


def rows(con, sql):
    return con.execute(sql).fetchall()


def one(con, sql):
    return con.execute(sql).fetchone()[0]


def table(headers, data):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in data:
        lines.append("| " + " | ".join("NULL" if v is None else (f"{v:,}" if isinstance(v, int) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def main() -> None:
    con = sqlite3.connect(f"file:{RISK_DB_PATH}?mode=ro", uri=True)
    out = ["# 数据探查报告", "", "由 `scripts/profile_data.py` 基于 `data/bank_risk.sqlite` 生成。", ""]

    out += ["## 1. 表规模与时间范围", ""]
    spec = [
        ("account", "账户", "date"), ("client", "客户", "birth_date"), ("disp", "账户权限关系", None),
        ("card", "信用卡", "issued"), ("loan", "贷款", "date"), ("order", "代扣指令", None),
        ("trans", "交易流水", "date"), ("district", "区县", None), ("dwd_risk_event", "风险事件（DWD）", "event_date"),
    ]
    data = []
    for t, zh, dcol in spec:
        n = one(con, f'SELECT COUNT(*) FROM "{t}"')
        rng = rows(con, f'SELECT MIN("{dcol}"), MAX("{dcol}") FROM "{t}"')[0] if dcol else ("", "")
        data.append((t, zh, n, f"{rng[0]} 至 {rng[1]}" if dcol else ""))
    out += [table(["表", "含义", "行数", "日期范围"], data), ""]

    out += ["## 2. 关键枚举分布", "", "### 2.1 交易方向与交易方式", ""]
    out += [table(["type", "operation", "笔数", "金额"], rows(con,
        "SELECT type, operation, COUNT(*), SUM(amount) FROM trans GROUP BY 1, 2 ORDER BY 3 DESC")), ""]
    out += ["### 2.2 交易用途（k_symbol）", ""]
    out += [table(["k_symbol", "笔数"], rows(con,
        "SELECT quote(k_symbol), COUNT(*) FROM trans GROUP BY k_symbol ORDER BY 2 DESC")), ""]
    out += ["### 2.3 贷款状态", ""]
    out += [table(["status", "笔数", "金额"], rows(con,
        "SELECT status, COUNT(*), SUM(amount) FROM loan GROUP BY 1 ORDER BY 1")), ""]
    out += ["### 2.4 按放款年份的不良率（vintage）", ""]
    out += [table(["放款年份", "放款笔数", "不良笔数", "不良率"], rows(con,
        "SELECT substr(date,1,4), COUNT(*), SUM(status IN ('B','D')), "
        "printf('%.2f%%', 100.0 * SUM(status IN ('B','D')) / COUNT(*)) FROM loan GROUP BY 1")), ""]
    out += ["### 2.5 账户、卡、权限关系", ""]
    out += [table(["维度", "取值", "数量"], rows(con,
        "SELECT 'account.frequency', frequency, COUNT(*) FROM account GROUP BY 2 "
        "UNION ALL SELECT 'card.type', type, COUNT(*) FROM card GROUP BY 2 "
        "UNION ALL SELECT 'disp.type', type, COUNT(*) FROM disp GROUP BY 2")), ""]

    out += ["## 3. 数据质量发现", ""]
    findings = [
        ("trans.type 存在未文档化编码 VYBER", one(con, "SELECT COUNT(*) FROM trans WHERE type = 'VYBER'"),
         "出账口径须包含 VYDAJ 与 VYBER"),
        ("trans.operation 为 NULL", one(con, "SELECT COUNT(*) FROM trans WHERE operation IS NULL"),
         "全部为利息入账（k_symbol = 'UROK'）"),
        ("trans.k_symbol 为 NULL", one(con, "SELECT COUNT(*) FROM trans WHERE k_symbol IS NULL"), "归为未标注"),
        ("trans.k_symbol 为单个空格", one(con, "SELECT COUNT(*) FROM trans WHERE k_symbol = ' '"), "归为未标注"),
        ("order.k_symbol 为空串", one(con, "SELECT COUNT(*) FROM \"order\" WHERE k_symbol = ''"), "归为未标注"),
        ("trans.balance 为负（透支）", one(con, "SELECT COUNT(*) FROM trans WHERE balance < 0"), "余额可为负"),
        ("district.A4 至 A7 为文本型数字", one(con, "SELECT COUNT(*) FROM district WHERE typeof(A4) = 'text'"),
         "数值比较前需 CAST"),
        ("district.A12 / A15 空值", one(con, "SELECT COUNT(*) FROM district WHERE A12 IS NULL OR A15 IS NULL"),
         "Jesenik 区县"),
        ("金额单位描述冲突", "全部金额字段", "原始文档为捷克克朗，BIRD 标注为 USD"),
    ]
    out += [table(["问题", "影响行数", "处理"], findings), ""]

    out += ["## 4. 风险事件层（DWD）", ""]
    out += [table(["规则", "名称", "风险大类", "事件数", "账户数", "涉及金额"], rows(con,
        "SELECT rule_code, rule_name, risk_category, COUNT(*), COUNT(DISTINCT account_id), SUM(amount) "
        "FROM dwd_risk_event GROUP BY 1, 2, 3 ORDER BY 1")), ""]
    out += [table(["年份", "事件数", "风险账户数"], rows(con,
        "SELECT substr(event_date,1,4), COUNT(*), COUNT(DISTINCT account_id) FROM dwd_risk_event GROUP BY 1")), ""]
    con.close()
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
