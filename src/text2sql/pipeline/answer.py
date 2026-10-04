"""Answer assembly: formatted table, one-line conclusion, caliber, data range and SQL."""

from __future__ import annotations

from datetime import date
from typing import Any

from text2sql.config import DATA_END_DATE, DATA_START_DATE
from text2sql.semantic.ir import QueryIR
from text2sql.semantic.registry import SemanticModel
from text2sql.semantic.timeparse import baseline_range

LAYER_ZH = {"L1": "第一层：OKR 物化表查询", "L2": "第二层：语义层编译 SQL", "L3": "第三层：生成式 SQL"}
SUFFIX_ZH = {"current": "本期", "baseline": "基期", "change": "变化量", "change_rate": "变化率"}
NO_BASELINE = {"baseline": "无数据", "change": "无数据", "change_rate": "无法计算"}  # baseline before the data


def _metric_of(column: str, model: SemanticModel) -> tuple[str | None, str | None]:
    if column in model.metrics:
        return column, None
    for suffix in SUFFIX_ZH:
        if column.endswith("_" + suffix) and column[: -len(suffix) - 1] in model.metrics:
            return column[: -len(suffix) - 1], suffix
    return None, None


def header(column: str, model: SemanticModel) -> str:
    if column == "period":
        return "周期"
    if column in model.dimensions:
        return model.dimensions[column].name
    metric, suffix = _metric_of(column, model)
    if metric:
        name = model.metrics[metric].name
        return f"{name}（{SUFFIX_ZH[suffix]}）" if suffix else name
    return column


def fmt(value: Any, column: str, model: SemanticModel) -> str:
    if value is None:
        return "空"
    if column in model.dimensions:
        return model.dimensions[column].label_for(value) or str(value)
    metric, suffix = _metric_of(column, model)
    is_rate = suffix == "change_rate" or (metric and model.metrics[metric].agg == "ratio" and suffix != "change")
    if isinstance(value, (int, float)):
        if is_rate:
            return f"{value * 100:.2f}%"
        if isinstance(value, float) and not value.is_integer():
            return f"{value:,.2f}"
        return f"{int(value):,}"
    return str(value)


def baseline_missing(ir: QueryIR | None) -> bool:
    """A comparison whose baseline period lies wholly before the data starts: its 0 means no data."""
    if not (ir and ir.compare and ir.time_range):
        return False
    return baseline_range(*ir.time_range, ir.compare)[1] < DATA_START_DATE


def display_table(columns: list[str], rows: list[tuple], model: SemanticModel,
                  ir: QueryIR | None = None) -> tuple[list[str], list[list[str]]]:
    missing = baseline_missing(ir)

    def cell(value: Any, column: str) -> str:
        suffix = _metric_of(column, model)[1]
        return NO_BASELINE[suffix] if missing and suffix in NO_BASELINE else fmt(value, column, model)

    return [header(c, model) for c in columns], [[cell(v, c) for v, c in zip(r, columns)] for r in rows]


def time_scope(ir: QueryIR | None) -> str:
    if ir and ir.time_range:
        return f"{ir.time_range[0]} 至 {ir.time_range[1]}"
    return "全部历史数据"


def conclusion(ir: QueryIR, columns: list[str], rows: list[tuple], model: SemanticModel) -> str:
    if not rows:
        return "查询结果为空：所选范围内没有符合条件的数据。"
    if ir.query_type == "point" and len(rows) == 1:
        items = [f"{header(c, model)} {fmt(v, c, model)}{model.metrics[c].unit if model.metrics[c].agg != 'ratio' else ''}"
                 for c, v in zip(columns, rows[0]) if c in model.metrics]
        return f"{time_scope(ir)}：" + "，".join(items) + "。"
    if ir.query_type == "compare" and len(rows) == 1:
        m = ir.metrics[0]
        rec = dict(zip(columns, rows[0]))
        if baseline_missing(ir):
            base, rate = NO_BASELINE["baseline"], NO_BASELINE["change_rate"]
        else:
            base, rate = fmt(rec[m + "_baseline"], m, model), fmt(rec[m + "_change_rate"], m + "_change_rate", model)
        return (f"{model.metrics[m].name}：本期 {fmt(rec[m + '_current'], m, model)}，基期 {base}，变化率 {rate}"
                f"（{'同比' if ir.compare == 'yoy' else '环比'}）。")
    return f"共 {len(rows)} 行结果，见下表。"


def caliber_lines(ir: QueryIR, model: SemanticModel) -> list[str]:
    lines = [f"{model.metrics[m].name}：{model.metrics[m].description}" for m in ir.metrics]
    for dim, codes in ir.filters.items():
        labels = "、".join(model.dimensions[dim].label_for(c) or "空" for c in codes)
        lines.append(f"筛选：{model.dimensions[dim].name} = {labels}")
    return lines


def data_note(asof: date = DATA_END_DATE) -> str:
    return f"数据截至 {asof}"
