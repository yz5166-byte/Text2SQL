"""Prompt, output schema and examples for the semantic parser (intent + IR slots in one call)."""

from __future__ import annotations

import json

from text2sql.config import REFERENCE_DATE
from text2sql.semantic.registry import SemanticModel


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


def parse_schema(model: SemanticModel) -> dict:
    """Slots come before the judgment (status), so the model judges with its extraction already written."""
    dims = list(model.dimensions)
    properties = {
        "intent": {"type": "string", "enum": ["data_query", "metadata", "out_of_domain"]},
        "metrics": {"type": "array", "items": {"type": "string", "enum": list(model.metrics)}},
        "time_text": _nullable({"type": "string"}),
        "dimensions": {"type": "array", "items": {"type": "string", "enum": dims}},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": dims},
                    "values": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["dimension", "values"],
            },
        },
        "query_type": _nullable({"type": "string", "enum": ["point", "breakdown", "trend", "compare"]}),
        "grain": _nullable({"type": "string", "enum": ["day", "month", "quarter", "year"]}),
        "compare": _nullable({"type": "string", "enum": ["yoy", "mom"]}),
        "order": _nullable({"type": "string", "enum": ["desc", "asc"]}),
        "limit": _nullable({"type": "integer"}),
        "status": _nullable({"type": "string", "enum": ["covered", "uncovered", "clarify"]}),
        "reason": _nullable({"type": "string"}),
    }
    return {"type": "object", "properties": properties, "required": list(properties)}


# Hand-written; none reuses or closely resembles an evaluation question (checked in tests/test_parser.py).
EXAMPLES: list[tuple[str, dict]] = [
    ("1996年各区县的放款金额，取前3",
     dict(metrics=["loan_amt"], time_text="1996年", dimensions=["district_name"], query_type="breakdown",
          order="desc", limit=3, status="covered")),
    ("近两个月入账金额环比变化",
     dict(metrics=["credit_amt"], time_text="近两个月", query_type="compare", compare="mom", status="covered")),
    ("1994年东波希米亚每个季度现金存入的交易笔数",
     dict(metrics=["trans_cnt"], time_text="1994年",
          filters=[{"dimension": "region", "values": ["东波希米亚"]},
                   {"dimension": "trans_operation", "values": ["现金存入"]}],
          query_type="trend", grain="quarter", status="covered")),
    ("1994年男性户主账户的交易笔数和笔均交易金额",
     dict(metrics=["trans_cnt", "avg_trans_amt"], time_text="1994年",
          filters=[{"dimension": "owner_gender", "values": ["男性"]}], query_type="point", status="covered")),
    ("每个客户平均持有几张信用卡",
     dict(status="uncovered", reason="需要人均计算，语义层没有对应指标")),
    ("1996年本金在30万以上的贷款共放了几笔",
     dict(metrics=["loan_cnt"], time_text="1996年", query_type="point", status="uncovered",
          reason="本金金额阈值不是语义层维度")),
    ("哪3个客户的交易总额最高",
     dict(metrics=["trans_amt"], query_type="breakdown", order="desc", limit=3, status="uncovered",
          reason="按单个客户排名，客户不是语义层维度")),
    ("损失是多少",
     dict(query_type="point", status="clarify", reason="损失可指不良贷款金额或风险涉及金额")),
]

_EMPTY_OUTPUT = dict(intent="data_query", metrics=[], time_text=None, dimensions=[], filters=[], query_type=None,
                     grain=None, compare=None, order=None, limit=None, status=None, reason=None)


def example_output(slots: dict) -> dict:
    """A complete parser output in schema field order, with unspecified fields empty."""
    return {**_EMPTY_OUTPUT, **slots}


def _dimension_line(model: SemanticModel, key: str) -> str:
    dim = model.dimensions[key]
    values = dim.values
    if len(values) > 10:  # long lists (77 districts): show the ones with Chinese names
        values = [v for v in values if v.label != str(v.codes[0])][:12]
    shown = "、".join(v.label for v in values)
    more = f" 等 {len(dim.values)} 个" if len(dim.values) > 10 else ""
    alias = f"（别名：{'、'.join(dim.aliases)}）" if dim.aliases else ""
    return f"- {key} {dim.name}{alias}：{dim.description}。取值：{shown}{more}"


_PROMPTS: dict[int, str] = {}


def parse_system_prompt(model: SemanticModel) -> str:
    """Stable system prompt (built once per model), which keeps provider-side prefix caching effective."""
    if id(model) not in _PROMPTS:
        _PROMPTS[id(model)] = _build_parse_prompt(model)
    return _PROMPTS[id(model)]


def _build_parse_prompt(model: SemanticModel) -> str:
    metric_lines = []
    for key, m in model.metrics.items():
        alias = f"（别名：{'、'.join(m.aliases)}）" if m.aliases else ""
        metric_lines.append(f"- {key} {m.name}{alias}：{m.description}")
    dim_lines = [_dimension_line(model, key) for key in model.dimensions]
    ambiguous = "；".join(
        f"{term} 可指 {'、'.join(model.metrics[c].name for c in cands)}" for term, cands in model.ambiguous_terms.items()
    )
    examples = "\n".join(
        f"问题：{q}\n输出：{json.dumps(example_output(slots), ensure_ascii=False)}" for q, slots in EXAMPLES
    )
    return f"""你是银行风控取数系统的语义解析器，负责把用户问题解析为结构化查询意图。

## 可用指标（metrics 只能从中选择）
{chr(10).join(metric_lines)}

## 可用维度（dimensions 与 filters.dimension 只能从中选择）
{chr(10).join(dim_lines)}

## 解析规则（按输出字段顺序）
1. intent：取数问题为 data_query，包括区县的人口、工资、失业率、犯罪数，以及客户、账户、卡、贷款、交易、代扣指令、账户权限关系等任何与本库数据有关的统计或查询，即使没有对应的已定义指标；询问指标口径、计算方法、字段含义、数据在哪张表为 metadata；与本库数据无关（如天气、写作、编程、业务办理咨询）为 out_of_domain。intent 不是 data_query 时，其余字段取空值。
2. metrics：问题所求且已定义的指标。优先使用已定义的指标，不要用筛选条件拼出已有指标：问"不良贷款金额"选 bad_loan_amt，而不是 loan_amt 加状态筛选。
3. time_text：原样摘录问题中的时间表达（如"上个月""1997年第四季度""1995年到1997年"），不要改写或换算；问题没有时间表达时为 null。
4. dimensions 与 filters：分组维度放 dimensions；筛选条件放 filters，values 原样摘录问题中的筛选值（如"南摩拉维亚""取现""不良""金卡"）。
5. query_type：单个数值为 point；按维度分组为 breakdown；按时间周期展示为 trend，grain 为周期粒度；同比或环比为 compare，compare 取 yoy 或 mom。status 为 covered 时必填。
6. 排名类问题（最多、最高、前 N 个）：query_type 为 breakdown，order 为 desc，limit 为正整数 N；"最多的是哪个"取 limit 为 1。
7. status：依据上面已摘录的槽位判断，intent 为 data_query 时必填。
   covered：问题能完整表达为"指标 × 维度 × 时间 × 维度筛选"。
   uncovered：需要上述指标以外的计算（自定义比率、人均、占比）、逐期计算的同比或环比（如每月相对上月的增长率）、账户或客户等明细级排名、指标以外字段的条件（如金额阈值、余额、年龄、人口、工资）、多步推理；reason 说明哪一部分无法表达。
   clarify：指标含义不明确或问题未给出任何可识别的指标；reason 说明需要用户明确什么。多义词：{ambiguous}。
8. 参考日期：今天是 {REFERENCE_DATE.isoformat()}，数据范围为 1993-01-01 至 1998-12-31。

## 示例
{examples}
"""
