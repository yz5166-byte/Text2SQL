"""Deterministic normalization of a parse into a route and a QueryIR (no model call).

1. Intent and status: metadata, out-of-domain, clarify and uncovered questions leave first.
2. Time: in a covered question every time expression is the period, so the question text is
   authoritative and the model's extraction only chooses among several. In an uncovered
   question a date may be a condition (a birth year), so only the model's extraction is used.
3. Reachability and value linking, repairing a value filed under the wrong dimension.
4. Contradictions: a metric's own condition that excludes every value the user filters the
   same column on (现金取现笔数 filtered to 信用卡取现) can only return 0, so it is clarified;
   a comparison asked period by period (每月相对上月) is beyond one-period IR, so it goes to layer 3.
5. Defaults: each one is disclosed in the answer.
6. Data range: applied to every IR, from the fast path as well, before anything is queried.
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

from text2sql.config import DATA_END_DATE, DATA_START_DATE
from text2sql.pipeline.decision import NO_TIME_NOTE, RouteDecision, semantic_layer
from text2sql.pipeline.meta import table_mentions
from text2sql.pipeline.parser import ParseResult
from text2sql.semantic.ir import Grain, QueryIR
from text2sql.semantic.linker import link_value, link_value_any
from text2sql.semantic.registry import SemanticModel
from text2sql.semantic.timeparse import Range, baseline_range, find_times, normalize, parse_time

GRAIN_WORDS: dict[Grain, re.Pattern] = {
    "year": re.compile(r"每年|各年|逐年|分年"),
    "quarter": re.compile(r"每个?季度|各季度|逐季|分季度"),
    "month": re.compile(r"每个?月|各月|逐月|分月"),  # not 按月: it is a value of 账单频率 (按月出账单)
    "day": re.compile(r"每天|每日|逐日|各天"),
}


def normalize_parse(question: str, p: ParseResult, model: SemanticModel) -> RouteDecision:
    if p.intent == "out_of_domain":
        return RouteDecision("OOD", "llm", "模型判定与取数无关", parse=p)
    if p.intent == "metadata":
        return RouteDecision("META", "llm", "模型判定为口径与数据资产问答", parse=p)
    if p.status == "clarify" or (p.status == "covered" and not p.metrics):
        return RouteDecision("CLARIFY", "llm", p.reason or "未识别到明确指标", parse=p, clarify_kind="metric",
                             clarify_options=[m.key for m in model.okr_metrics()])
    outside = [t for t in table_mentions(question, model) if t in model.uncovered_tables()]
    if p.status == "uncovered" or outside:  # a table the semantic layer does not cover cannot be expressed
        hint = parse_time(p.time_text) if p.time_text else None
        why = f"涉及语义层未覆盖的{'、'.join(model.tables[t].name_zh for t in outside)}" if outside else p.reason
        reason = "语义层无法表达，转生成式 SQL" + (f"：{why}" if why else "")
        return RouteDecision("L3", "llm", reason, parse=p, time_hint=(hint.start, hint.end) if hint else None)

    time_range, unclear = _covered_period(question, p.time_text)
    if unclear:
        return RouteDecision("CLARIFY", "llm", unclear, parse=p, clarify_kind="time")

    reachable = set.intersection(*(set(model.reachable_dimensions(m)) for m in p.metrics))
    bad_dims = [d for d in p.dimensions if d not in reachable]
    if bad_dims:
        return RouteDecision("L3", "llm", f"维度不可达：{bad_dims}", parse=p, time_hint=time_range)

    filters: dict[str, list] = {}
    assumptions: list[str] = []
    for f in p.filters:
        for value in f.values:
            dim = f.dimension
            link = link_value(model, dim, value) if dim in reachable else None
            if link is None:  # not in the model's dimension: look in every reachable one
                hits = link_value_any(model, sorted(reachable), value)
                if len(hits) != 1:
                    return RouteDecision("L3", "llm", f"筛选值无法映射：{value}", parse=p, time_hint=time_range)
                dim, link = hits[0]
                assumptions.append(f"筛选值“{value}”归入维度“{model.dimensions[dim].name}”")
            if link.fuzzy:
                assumptions.append(f"筛选值“{value}”按近似匹配为“{link.term}”")
            filters.setdefault(dim, [])
            filters[dim] += [c for c in link.codes if c not in filters[dim]]

    conflicting = _conflicting_metrics(model, p.metrics, filters)
    if conflicting:
        names = "、".join(model.metrics[m].name for m in conflicting)
        return RouteDecision("CLARIFY", "llm", f"指标“{names}”自带的条件与筛选条件矛盾，按字面查询结果必然为 0",
                             parse=p, clarify_kind="metric",
                             clarify_options=list(dict.fromkeys(model.base_metric(m) for m in conflicting)))

    qt = "breakdown" if p.query_type == "point" and p.dimensions else p.query_type
    if qt == "compare" and _grain_word(question):
        return RouteDecision("L3", "llm", "逐期对比（每期相对上一期）语义层不能表达，转生成式 SQL", parse=p,
                             time_hint=time_range)
    grain = p.grain if qt == "trend" else None
    if qt == "trend" and grain is None:
        grain = _grain_word(question)
        if grain is None:
            grain = "month"
            assumptions.append("未指定趋势粒度，按月展示")
    compare = None
    if qt == "compare":
        if time_range is None:
            return RouteDecision("CLARIFY", "llm", "同比或环比需要明确的统计期间", parse=p, clarify_kind="time")
        compare = p.compare or ("mom" if "环比" in question else "yoy")
    if time_range is None:
        assumptions.append(NO_TIME_NOTE)
    order = p.order or ("desc" if p.limit else None)
    ir = QueryIR(metrics=p.metrics, query_type=qt, time_range=time_range, grain=grain,
                 dimensions=p.dimensions, filters=filters, compare=compare, order=order, limit=p.limit)
    return RouteDecision(semantic_layer(ir, model), "llm", "语义层可表达", ir=ir, parse=p,
                         assumptions=assumptions, time_hint=time_range)


def _covered_period(question: str, time_text: str | None) -> tuple[Range | None, str | None]:
    """(period, None), or (None, why to clarify). The question is authoritative; the model only chooses."""
    spans = find_times(question)
    if len(spans) == 1:
        return (spans[0].start, spans[0].end), None
    if not spans:  # a time the rules cannot read in the question was reworded or invented by the model
        return None, f"无法解析时间表达“{time_text}”" if time_text else None
    wanted = normalize(time_text or "")
    own = parse_time(wanted)
    chosen = [s for s in spans if wanted and (wanted in s.text or (own and (own.start, own.end) == (s.start, s.end)))]
    if len(chosen) == 1:
        return (chosen[0].start, chosen[0].end), None
    return None, "问题中有多个时间表达，无法确定统计哪一段"


def _conflicting_metrics(model: SemanticModel, metrics: list[str], filters: dict[str, list]) -> list[str]:
    """Metrics whose own condition on a column admits none of the codes the user filters that column on."""
    wanted = {}
    for dim, codes in filters.items():
        alias, column = model.dimensions[dim].column.split(".")
        wanted[f"{model.physical_table(alias)}.{column}"] = {str(c) for c in codes}
    out = []
    for key in metrics:
        for condition in model.metrics[key].filters:
            column, allowed = _condition_values(condition)
            if column in wanted and not wanted[column] & allowed:
                out.append(key)
    return out


def _condition_values(condition: str) -> tuple[str | None, set[str]]:
    """Column and admitted values of "t.c = 'x'" or "t.c IN ('x', 'y')"; (None, empty) for anything else."""
    tree = sqlglot.parse_one(condition, read="sqlite")
    if isinstance(tree, exp.EQ) and isinstance(tree.left, exp.Column) and isinstance(tree.right, exp.Literal):
        return f"{tree.left.table}.{tree.left.name}", {tree.right.this}
    if isinstance(tree, exp.In) and isinstance(tree.this, exp.Column):
        return f"{tree.this.table}.{tree.this.name}", {v.this for v in tree.expressions if isinstance(v, exp.Literal)}
    return None, set()


def _grain_word(question: str) -> Grain | None:
    """The trend granularity the question names, when exactly one kind of granularity word appears."""
    text = normalize(question)
    hits = [grain for grain, words in GRAIN_WORDS.items() if words.search(text)]
    return hits[0] if len(hits) == 1 else None


def fit_to_data_range(decision: RouteDecision, model: SemanticModel) -> RouteDecision:
    """Never query a period outside the data; disclose a clipped period or an incomplete baseline."""
    ir = decision.ir
    if ir is None or ir.time_range is None:
        return decision
    start, end = ir.time_range
    if end < DATA_START_DATE or start > DATA_END_DATE:
        return RouteDecision("CLARIFY", decision.source, f"所问时间 {start} 至 {end} 不在数据范围内",
                             parse=decision.parse, clarify_kind="time")
    clipped = (max(start, DATA_START_DATE), min(end, DATA_END_DATE))
    if clipped != ir.time_range:
        ir = decision.ir = ir.model_copy(update={"time_range": clipped})
        decision.route, decision.time_hint = semantic_layer(ir, model), clipped
        decision.assumptions.append(f"数据范围为 {DATA_START_DATE} 至 {DATA_END_DATE}，实际统计 {clipped[0]} 至 {clipped[1]}")
    if ir.compare:
        base_start, base_end = baseline_range(*clipped, ir.compare)
        if base_end < DATA_START_DATE:
            decision.assumptions.append(f"基期 {base_start} 至 {base_end} 不在数据范围内，无法计算变化率")
        elif base_start < DATA_START_DATE:
            decision.assumptions.append(f"基期 {base_start} 至 {base_end} 部分不在数据范围内，变化率仅供参考")
    return decision
