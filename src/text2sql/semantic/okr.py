"""Layer 1: answer from the monthly OKR table when that is mathematically valid.

ads_okr_monthly holds one row per (month, region, metric): the value, plus numerator and
denominator for ratios. It is built by the layer-2 compiler (materialize.py), so both layers
share one definition of every metric. Whether a question may use the table is decided by
rules in code; the model never reads any number.

Decisions (see the progress board): the table is aggregated by region only; a distinct count
is served month by month only (a quarter goes back to the base tables); a distinct count
filtered on several regions also goes back to the base tables.
"""

from __future__ import annotations

from datetime import date

from text2sql.semantic.compiler import (
    CompileError, CompiledQuery, Params, between, compare_columns, compare_outer, compile_l2, conjoin,
    in_list, order_keys, render,
)
from text2sql.semantic.ir import QueryIR
from text2sql.semantic.registry import SemanticModel
from text2sql.semantic.timeparse import baseline_range, is_month_aligned

ADS_TABLE = "ads_okr_monthly"
PERIOD = {
    "month": "stat_month",
    "quarter": "substr(stat_month, 1, 4) || '-Q' || ((CAST(substr(stat_month, 6, 2) AS INTEGER) + 2) / 3)",
    "year": "substr(stat_month, 1, 4)",
}


def _month(d: date) -> str:
    return d.strftime("%Y-%m")


def _months(rng: tuple[date, date]) -> int:
    start, end = rng
    return (end.year - start.year) * 12 + end.month - start.month + 1


def l1_ineligibility(ir: QueryIR, model: SemanticModel) -> str | None:
    """Why the OKR table cannot answer the IR, or None when it can."""
    if not all(model.metrics[m].okr for m in ir.metrics):
        return "包含未物化的非 OKR 指标"
    if set(ir.dimensions) - {"region"} or set(ir.filters) - {"region"}:
        return "维度或筛选超出物化表的地区维度"
    if ir.grain == "day":
        return "物化表粒度为月，不支持按日"
    ranges = []
    if ir.time_range:
        if not is_month_aligned(*ir.time_range):
            return "时间范围未按整月对齐"
        ranges.append(ir.time_range)
        if ir.query_type == "compare":
            ranges.append(baseline_range(*ir.time_range, ir.compare))
    if any(model.metrics[m].additivity == "non_additive" for m in ir.metrics):
        per_month = ir.query_type == "trend" and ir.grain == "month"
        single_month = bool(ranges) and all(_months(r) == 1 for r in ranges)
        if not (per_month or single_month):
            return "去重计数类指标不可由月值累加"
        if len(ir.filters.get("region", [])) > 1:
            return "去重计数类指标不可跨地区累加"
    return None


def compile_l1(ir: QueryIR, model: SemanticModel) -> CompiledQuery:
    reason = l1_ineligibility(ir, model)
    if reason:
        raise CompileError(reason)
    p = Params()
    names = list(ir.metrics)
    where = [f"metric_code IN ({', '.join(p.add(m) for m in names)})"]
    if "region" in ir.dimensions:
        where.append("region <> 'ALL'")
    elif "region" in ir.filters:
        where.append(in_list("region", ir.filters["region"], p))
    else:
        where.append("region = 'ALL'")

    keys = [(PERIOD[ir.grain], "period")] if ir.query_type == "trend" else []
    keys += [("region", "region")] if "region" in ir.dimensions else []
    select = [f"{expr} AS {alias}" for expr, alias in keys]
    group_by = [alias for _, alias in keys]
    parts = dict(base=ADS_TABLE, joins=[], where=where, group_by=group_by, limit=ir.limit)

    def value(m: str, period: str | None = None) -> str:
        """Additive metrics sum the monthly values; a ratio divides the summed numerator by the summed denominator."""
        c = conjoin(f"metric_code = {p.add(m)}", period)
        if model.metrics[m].agg == "ratio":
            return f"SUM(CASE WHEN {c} THEN numerator END) * 1.0 / NULLIF(SUM(CASE WHEN {c} THEN denominator END), 0)"
        return f"COALESCE(SUM(CASE WHEN {c} THEN metric_value END), 0)"

    if ir.query_type == "compare":
        base_rng = baseline_range(*ir.time_range, ir.compare)
        current = between("stat_month", (_month(ir.time_range[0]), _month(ir.time_range[1])), p)
        baseline = between("stat_month", (_month(base_rng[0]), _month(base_rng[1])), p)
        where.append(f"({current} OR {baseline})")
        select += [f"{value(m, current)} AS {m}_current" for m in names]
        select += [f"{value(m, baseline)} AS {m}_baseline" for m in names]
        sql = render("compare", outer=compare_outer(group_by, names), select=select,
                     order_by=order_keys(ir, group_by, f"{names[0]}_current"), **parts)
        return CompiledQuery(sql, p.values, "L1", names, group_by, compare_columns(group_by, names))

    if ir.time_range:
        where.append(between("stat_month", (_month(ir.time_range[0]), _month(ir.time_range[1])), p))
    select += [f"{value(m)} AS {m}" for m in names]
    sql = render(ir.query_type, select=select, order_by=order_keys(ir, group_by, names[0]), **parts)
    return CompiledQuery(sql, p.values, "L1", names, group_by, group_by + names)


def compile_ir(ir: QueryIR, model: SemanticModel) -> list[CompiledQuery]:
    """Cache-aside: the OKR table when it can answer, otherwise the base tables (layer 2)."""
    if l1_ineligibility(ir, model) is None:
        return [compile_l1(ir, model)]
    return compile_l2(ir, model)
