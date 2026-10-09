"""Write eval/review_checklist.md: a slot-by-slot Chinese decomposition of sampled gold answers.

Semantic-layer items are rendered from their gold IR so the checklist cannot drift from the
data; generative and interaction items carry a hand-written description of the gold logic.
"""

from __future__ import annotations

import sqlite3

from text2sql.config import EVAL_DIR, RISK_DB_PATH
from text2sql.eval.dataset import load_domain_eval
from text2sql.semantic.registry import load_semantic_model

SAMPLE = ["L1-02", "L1-07", "L1-08", "L1-13", "L1-15", "L2-03", "L2-04", "L2-07", "L2-10", "L2-14",
          "L2-18", "L2-32", "L3-06", "L3-12", "L3-15", "L3-18", "L3-24", "C-01", "C-06", "M-05"]

MANUAL = {
    "L3-06": ("不良贷款金额 ÷ 放款金额（按金额，而非笔数）", "不限", "无", "按地区分组"),
    "L3-12": ("1997 年单笔金额最大的交易", "1997 全年", "无", "输出该交易账户的开户区县"),
    "L3-15": ("风险事件数 ÷ 地区人口 × 10000；人口为该地区各区县 A4 之和（文本转整数）",
              "事件取 1998 全年；人口为静态值", "无", "按地区分组"),
    "L3-18": ("（本月现金取现金额 − 上月）÷ 上月", "1997-02 至 1997-12；1997-01 仅作基期",
              "交易方式 = 现金取现", "按月输出 11 行"),
    "L3-24": ("出账金额（VYDAJ 与 VYBER）", "1998 全年", "账户在 1998 年命中过大额现金取现规则", "单值"),
    "C-01": ("应澄清：损失可指不良贷款金额或风险涉及金额", "", "", "不取数"),
    "C-06": ("应澄清：未给出任何指标", "", "", "不取数"),
    "M-05": ("应直接回答出账金额口径：含 VYDAJ 与 VYBER 两种编码", "", "", "不取数"),
}

GRAIN_ZH = {"day": "日", "month": "月", "quarter": "季度", "year": "年"}


def describe(item, model):
    ir = item.gold_ir
    metric = "、".join(model.metrics[m].name for m in ir.metrics)
    time = f"{ir.time_range[0]} 至 {ir.time_range[1]}" if ir.time_range else "不限"
    filters = []
    for dim_key, codes in ir.filters.items():
        dim = model.dimensions[dim_key]
        labels = next((v.label for v in dim.values if sorted(map(str, v.codes)) == sorted(map(str, codes))), None)
        filters.append(f"{dim.name} = {labels or '、'.join(dim.label_for(c) for c in codes)}")
    dims = "、".join(model.dimensions[d].name for d in ir.dimensions)
    if ir.query_type == "point":
        shape = "单值"
    elif ir.query_type == "breakdown":
        shape = f"按{dims}分组"
    elif ir.query_type == "trend":
        shape = f"按{GRAIN_ZH[ir.grain]}趋势" + (f"，并按{dims}拆分" if dims else "")
    else:
        shape = {"yoy": "同比：本期与去年同期", "mom": "环比：本期与上一周期"}[ir.compare]
    if ir.limit:
        shape += f"，取{'最高' if ir.order == 'desc' else '最低'} {ir.limit} 个"
    elif ir.order:
        shape += "，从高到低排序" if ir.order == "desc" else "，从低到高排序"
    return metric, time, "；".join(filters) or "无", shape


def main() -> None:
    model = load_semantic_model()
    items = {i.id: i for i in load_domain_eval()}
    con = sqlite3.connect(f"file:{RISK_DB_PATH}?mode=ro", uri=True)
    lines = [
        "# 评估集口径审阅记录（抽样 20 题）",
        "",
        "开发前从评估集中抽取 20 题，把每题的金标拆成指标、时间范围、筛选条件、分组与输出四项，"
        "由人工对照问题逐项核对口径是否与问题的本意一致；这里只核对口径，数值由金标 SQL 的执行结果给出。",
        "",
        "| 编号 | 问题 | 指标 | 时间范围 | 筛选条件 | 分组与输出 | 金标结果 |",
        "|---|---|---|---|---|---|---|",
    ]
    for item_id in SAMPLE:
        item = items[item_id]
        metric, time, filters, shape = MANUAL.get(item_id) or describe(item, model)
        if item.gold_sql:
            rows = con.execute(item.gold_sql).fetchall()
            result = str(rows[0]) if len(rows) == 1 else f"{len(rows)} 行，首行 {rows[0]}"
        else:
            result = ""
        lines.append(f"| {item_id} | {item.question} | {metric} | {time} | {filters} | {shape} | {result} |")
    con.close()
    out = EVAL_DIR / "review_checklist.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
