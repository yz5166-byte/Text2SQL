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
    out = ["# 数据探查报告", "",
           "本报告由 `scripts/profile_data.py` 从 `data/bank_risk.sqlite` 计算生成，所有数字都来自查询，"
           "数仓重建后可以重新生成。每节后面的解读说明这些数据对系统设计的影响。", ""]

    out += ["## 表规模与时间范围", ""]
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
    first, last = rows(con, "SELECT MIN(date), MAX(date) FROM trans")[0]
    out += [f"交易流水占了绝大部分数据量，时间覆盖 {first} 至 {last}。客户表的日期是出生日期，账户表的日期是开户日期，"
            f"都不是业务发生的时间。系统把\"上个月\"这类相对时间换算到交易与事件的时间轴上，参考日期取交易的最后一天 {last}。", ""]

    out += ["## 关键枚举分布", "", "### 交易方向与交易方式", ""]
    out += [table(["type", "operation", "笔数", "金额"], rows(con,
        "SELECT type, operation, COUNT(*), SUM(amount) FROM trans GROUP BY 1, 2 ORDER BY 3 DESC")), ""]
    vyber_type = one(con, "SELECT COUNT(*) FROM trans WHERE type = 'VYBER'")
    out += [f"资金方向（type）除了文档中的 PRIJEM（入账）和 VYDAJ（出账），还有 {vyber_type:,} 笔记为 VYBER；"
            "同一个编码 VYBER 又出现在交易方式（operation）中，表示现金取现。所以出账类指标要同时包含 VYDAJ 与 VYBER，"
            "而\"取现\"要映射到交易方式这一列，不能映射到资金方向。", ""]
    out += ["### 交易用途（k_symbol）", ""]
    out += [table(["k_symbol", "笔数"], rows(con,
        "SELECT quote(k_symbol), COUNT(*) FROM trans GROUP BY k_symbol ORDER BY 2 DESC")), ""]
    out += ["交易用途的缺失有 NULL 和单个空格两种写法，语义层把两者都归为\"未标注\"，按用途筛选时两种写法要一起处理。", ""]
    out += ["### 贷款状态", ""]
    out += [table(["status", "笔数", "金额"], rows(con,
        "SELECT status, COUNT(*), SUM(amount) FROM loan GROUP BY 1 ORDER BY 1")), ""]
    out += ["按数据集的说明，A 为合同已结束且无问题，B 为合同已结束但贷款未还清，C 为仍在履约且正常，D 为仍在履约但客户欠款。"
            "语义层把 B 与 D 定义为不良。", ""]
    out += ["### 按放款年份的不良率", ""]
    out += [table(["放款年份", "放款笔数", "不良笔数", "不良率"], rows(con,
        "SELECT substr(date,1,4), COUNT(*), SUM(status IN ('B','D')), "
        "printf('%.2f%%', 100.0 * SUM(status IN ('B','D')) / COUNT(*)) FROM loan GROUP BY 1")), ""]
    vintage = rows(con, "SELECT substr(date,1,4), 100.0 * SUM(status IN ('B','D')) / COUNT(*) FROM loan GROUP BY 1 ORDER BY 1")
    (last_year, last_rate), earlier = vintage[-1], [r for _, r in vintage[1:-1]]
    assert last_rate < min(earlier)
    out += [f"不良率按放款年份归属。{last_year} 年放款的不良率只有 {last_rate:.2f}%，而 {vintage[1][0]} 至 {vintage[-2][0]} 年在 "
            f"{min(earlier):.2f}% 至 {max(earlier):.2f}% 之间（{vintage[0][0]} 年放款很少，不作比较）。这不是风险变低，而是这些贷款发放时间短，"
            "违约还没来得及发生。按放款日期统计的不良率天然有这种账龄效应，比较不同年份时需要注意。", ""]
    out += ["### 账户、卡、权限关系", ""]
    out += [table(["维度", "取值", "数量"], rows(con,
        "SELECT 'account.frequency', frequency, COUNT(*) FROM account GROUP BY 2 "
        "UNION ALL SELECT 'card.type', type, COUNT(*) FROM card GROUP BY 2 "
        "UNION ALL SELECT 'disp.type', type, COUNT(*) FROM disp GROUP BY 2")), ""]
    owners_per_account = rows(con, "SELECT MIN(n), MAX(n) FROM (SELECT COUNT(*) AS n FROM disp WHERE type = 'OWNER' GROUP BY account_id)")[0]
    accounts_with_owner = one(con, "SELECT COUNT(DISTINCT account_id) FROM disp WHERE type = 'OWNER'")
    assert owners_per_account == (1, 1) and accounts_with_owner == one(con, "SELECT COUNT(*) FROM account")
    out += ["每个账户恰好有一个所有人（OWNER），其余为授权使用人（DISPONENT）。按账户统计客户属性时必须限定 OWNER，"
            "否则有授权使用人的账户会被重复计数；语义层通过一个限定了 OWNER 的别名来关联客户表。", ""]

    out += ["## 数据质量问题", ""]
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
    out += ["除金额单位外，这些处理都写进了语义层的字段说明和取值定义，第三层生成 SQL 时看到的表结构说明也包含它们；"
            "金额单位按原始文档取捷克克朗，说明在 `semantic/catalog.yaml` 的开头。", ""]

    out += ["## 风险事件表", ""]
    out += [table(["规则", "名称", "风险大类", "事件数", "账户数", "涉及金额"], rows(con,
        "SELECT rule_code, rule_name, risk_category, COUNT(*), COUNT(DISTINCT account_id), SUM(amount) "
        "FROM dwd_risk_event GROUP BY 1, 2, 3 ORDER BY 1")), ""]
    out += [table(["年份", "事件数", "风险账户数"], rows(con,
        "SELECT substr(event_date,1,4), COUNT(*), COUNT(DISTINCT account_id) FROM dwd_risk_event GROUP BY 1")), ""]
    events, accounts = rows(con, "SELECT COUNT(*), COUNT(DISTINCT account_id) FROM dwd_risk_event")[0]
    per_year = rows(con, "SELECT substr(event_date,1,4), COUNT(*) FROM dwd_risk_event GROUP BY 1 ORDER BY 1")
    out += [f"三条规则共识别出 {events:,} 条风险事件，涉及 {accounts:,} 个账户；同一个账户可能命中多条规则，所以各规则的账户数之和大于总数。"
            f"事件数从 {per_year[0][0]} 年的 {per_year[0][1]:,} 条增长到 {per_year[-1][0]} 年的 {per_year[-1][1]:,} 条。"
            "规则的定义与阈值见 `scripts/build_warehouse.py`。", ""]
    con.close()
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
