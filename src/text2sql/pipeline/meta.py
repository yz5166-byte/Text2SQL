"""Metadata answers (口径与数据资产问答) served directly from the semantic layer and catalog."""

from __future__ import annotations

import re

from text2sql.semantic.linker import find_ambiguous_terms, find_metric_mentions, find_value_mentions
from text2sql.semantic.registry import Metric, SemanticModel

ADDITIVITY_ZH = {"additive": "可加", "non_additive": "不可加（去重计数）", "ratio": "比率（由分子分母汇总后相除）"}
STOP_BIGRAMS = {"哪张", "张表", "哪个", "个表", "哪些", "些表", "记录", "录了", "信息", "的信", "什么", "么意", "意思"}


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


def search_columns(model: SemanticModel, question: str, k: int = 3) -> list[tuple[str, str, float]]:
    q = _bigrams(question)
    scored = []
    for t_key, table in model.tables.items():
        for c_key, col in table.columns.items():
            score = len(q & _bigrams(col.name_zh)) * 2 + len(q & _bigrams(col.description))
            if score:
                scored.append((t_key, c_key, score))
    return sorted(scored, key=lambda x: -x[2])[:k]


def answer_meta(question: str, model: SemanticModel) -> str:
    parts: list[str] = []
    for m in find_metric_mentions(question, model):
        parts.append(describe_metric(model, model.metrics[m.key]))
    for amb in find_ambiguous_terms(question, model):
        names = "、".join(model.metrics[c].name for c in model.ambiguous_terms[amb.key])
        parts.append(f"【{amb.key}】为多义词，可能指：{names}。")
    for v in find_value_mentions(question, model, list(model.dimensions)):
        dim = model.dimensions[v.key]
        codes = "、".join("NULL" if c is None else repr(c) for c in v.codes)
        text = f"【{v.surface}】属于维度“{dim.name}”（{dim.column}），对应编码 {codes}。{dim.description}"
        table = model.tables[model.physical_table(dim.alias)]
        if table.notes:
            text += f"\n补充说明：{table.notes}"
        parts.append(text)
    if not parts or re.search(r"哪张表|哪个表|哪些表|哪个字段|哪些字段|记录了", question):
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
