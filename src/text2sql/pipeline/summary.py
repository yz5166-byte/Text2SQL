"""One-sentence summary of a breakdown or trend result, written by the model and checked by code.

Every number in the summary must appear verbatim in the result table (or in the question
itself); the model may select and phrase, but never compute. A summary that fails the check,
or no model at all, leaves the plain "共 N 行结果，见下表" conclusion in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from text2sql.llm.base import LLMClient, LLMError, Usage

SUMMARY_MAX_ROWS = 100  # larger tables are left to the reader
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
SYSTEM = (
    "你是银行风控数据分析助手。根据用户问题和查询结果表，用一句中文（不超过 60 字）指出最值得关注的要点，"
    "例如最高与最低、首末变化方向或峰值所在。"
    "数字一律使用阿拉伯数字，且只能原样引用表中出现的数值，不要自行计算差值、比例、合计或行数。只输出这一句话。"
)


@dataclass
class Summary:
    text: str | None  # None: keep the plain conclusion
    note: str  # outcome, for the trace
    usage: Usage = field(default_factory=Usage)


def summarize(llm: LLMClient | None, question: str, columns: list[str], rows: list[list[str]]) -> Summary:
    """`columns` and `rows` are the display table, so numbers are checked in the form the user sees."""
    if llm is None:
        return Summary(None, "未配置模型，不生成总结")
    if len(rows) > SUMMARY_MAX_ROWS:
        return Summary(None, f"结果超过 {SUMMARY_MAX_ROWS} 行，不生成总结")
    table = "\n".join(" | ".join(r) for r in [columns, *rows])
    try:
        reply = llm.complete(SYSTEM, f"问题：{question}\n\n查询结果：\n{table}", purpose="summarize")
    except LLMError as exc:
        return Summary(None, f"模型调用失败：{exc}")
    text = reply.text.strip()
    if reply.refused or not text:
        return Summary(None, "模型未给出总结", reply.usage)
    unsupported = unsupported_numbers(text, question, rows)
    if unsupported:
        return Summary(None, f"数字 {unsupported} 不在结果表中，退回", reply.usage)
    return Summary(text, "通过数字校验", reply.usage)


def unsupported_numbers(text: str, question: str, rows: list[list[str]]) -> list[str]:
    """Numbers in `text` found neither in the table cells nor in the question (thousands separators ignored)."""
    allowed = {n.replace(",", "") for cell in [question, *(c for r in rows for c in r)] for n in NUMBER.findall(cell)}
    return [n for n in NUMBER.findall(text) if n.replace(",", "") not in allowed]
