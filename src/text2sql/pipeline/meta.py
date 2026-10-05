"""Metadata answers (口径与数据资产问答) served directly from the semantic layer and catalog.

Recognised in order: metrics, ambiguous terms, dimension values; then, when no metric is
named, dimensions themselves (described with their values); when the question asks where
data lives, tables by name. Column search is the last resort and keeps only the columns that
score at least half of the best one, so a shared word such as 交易 does not drag in every
column of the transaction table.
"""

from __future__ import annotations

import re

from text2sql.semantic.linker import find_ambiguous_terms, find_metric_mentions, find_value_mentions
from text2sql.semantic.registry import Metric, SemanticModel

ADDITIVITY_ZH = {"additive": "可加", "non_additive": "不可加（去重计数）", "ratio": "比率（由分子分母汇总后相除）"}
STOP_BIGRAMS = {"哪张", "张表", "哪个", "个表", "哪些", "些表", "记录", "录了", "信息", "的信", "什么", "么意", "意思"}
LOCATION_WORDS = re.compile(r"哪张表|哪个表|哪些表|哪个字段|哪些字段|记录了|记录在|在哪")
MAX_VALUES_SHOWN = 20


def formula(model: SemanticModel, metric: Metric) -> str:
    if metric.agg == "ratio":
        return f"{model.metrics[metric.numerator].name} ÷ {model.metrics[metric.denominator].name}"
    base = {"count": "COUNT(*)", "sum": f"SUM({metric.column})",
            "count_distinct": f"COUNT(DISTINCT {metric.column})"}[metric.agg]
    where = f"，条件 {' AND '.join(metric.filters)}" if metric.filters else ""
    return f"{base}，数据表 {metric.table}{where}，时间归属 {metric.time_column}"


def describe_metric(model: SemanticModel, metric: Metric) -> str:
    okr = "；已物化到月度 OKR 表" if metric.okr else ""
    return (f"【{metric.name}】{metric.description}\n计算方式：{formula(model, metric)}\n"
            f"单位：{metric.unit}；可加性：{ADDITIVITY_ZH[metric.additivity]}{okr}")


def _bigrams(text: str) -> set[str]:
    text = re.sub(r"[^\w]", "", text)
    return {text[i:i + 2] for i in range(len(text) - 1)} - STOP_BIGRAMS


def describe_dimension(model: SemanticModel, key: str) -> str:
    dim = model.dimensions[key]
    labels = [v.label for v in dim.values]
    shown = "、".join(labels[:MAX_VALUES_SHOWN]) + (f" 等 {len(labels)} 个" if len(labels) > MAX_VALUES_SHOWN else "")
    return f"【{dim.name}】{dim.description}（字段 {dim.column}）。取值：{shown}"


def describe_table(model: SemanticModel, key: str) -> str:
    table = model.tables[key]
    columns = "、".join(f"{c}（{col.name_zh}）" for c, col in table.columns.items())
    notes = f"\n补充说明：{table.notes}" if table.notes else ""
    return f"【{table.name_zh}】{key}：{table.description}{table.grain}。\n字段：{columns}{notes}"


def dimension_mentions(question: str, model: SemanticModel) -> list[str]:
    return [k for k, d in model.dimensions.items() if any(term in question for term in (d.name, *d.aliases))]


def table_mentions(question: str, model: SemanticModel) -> list[str]:
    """Tables named in the question (账户表 is found as 账户), outside any metric name."""
    metric_spans = [(m.start, m.end) for m in find_metric_mentions(question, model)]
    found = []
    for key, table in model.tables.items():
        pos = question.find(table.name_zh.removesuffix("表"))
        if pos >= 0 and not any(s <= pos < e for s, e in metric_spans):
            found.append(key)
    return found


def search_columns(model: SemanticModel, question: str, k: int = 3) -> list[tuple[str, str, float]]:
    """Up to k columns by shared bigrams, keeping only those scoring at least half of the best."""
    q = _bigrams(question)
    scored = []
    for t_key, table in model.tables.items():
        for c_key, col in table.columns.items():
            score = len(q & _bigrams(col.name_zh)) * 2 + len(q & _bigrams(col.description))
            if score:
                scored.append((t_key, c_key, score))
    scored.sort(key=lambda x: -x[2])
    return [hit for hit in scored[:k] if hit[2] * 2 >= scored[0][2]]


def answer_meta(question: str, model: SemanticModel) -> str:
    parts: list[str] = []
    metrics = find_metric_mentions(question, model)
    for m in metrics:
        parts.append(describe_metric(model, model.metrics[m.key]))
    for amb in find_ambiguous_terms(question, model):
        names = "、".join(model.metrics[c].name for c in model.ambiguous_terms[amb.key])
        parts.append(f"【{amb.key}】为多义词，可能指：{names}。")
    values = find_value_mentions(question, model, list(model.dimensions))
    for v in values:
        dim = model.dimensions[v.key]
        codes = "、".join("NULL" if c is None else repr(c) for c in v.codes)
        text = f"【{v.surface}】属于维度“{dim.name}”（{dim.column}），对应编码 {codes}。{dim.description}"
        table = model.tables[model.physical_table(dim.alias)]
        if table.notes:
            text += f"\n补充说明：{table.notes}"
        parts.append(text)
    if not metrics:  # a dimension asked about by name, unless one of its values was already explained
        valued = {v.key for v in values}
        parts += [describe_dimension(model, d) for d in dimension_mentions(question, model) if d not in valued]
    asks_location = bool(LOCATION_WORDS.search(question))
    tables = table_mentions(question, model) if asks_location else []
    parts += [describe_table(model, t) for t in tables]
    if (not parts or asks_location) and not tables:
        hits = search_columns(model, question)
        if hits:
            lines = [f"{t}.{c}（{model.tables[t].columns[c].name_zh}）：{model.tables[t].columns[c].description}"
                     for t, c, _ in hits]
            tables = "、".join(dict.fromkeys(f"{t}（{model.tables[t].name_zh}）" for t, _, _ in hits))
            parts.append(f"相关数据表：{tables}\n" + "\n".join(lines))
    if not parts:
        names = "、".join(m.name for m in model.metrics.values())
        return f"未在指标与数据字典中找到相关条目。当前支持的指标：{names}。"
    return "\n\n".join(dict.fromkeys(parts))
