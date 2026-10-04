"""Rule prejudgment, no model call: metadata and out-of-domain questions, declared ambiguous terms,
vague time words, and the fast path for simple single-metric questions.

A rule must never guess: None hands the question to the model.
"""

from __future__ import annotations

import re

from text2sql.pipeline.decision import NO_TIME_NOTE, RouteDecision, semantic_layer
from text2sql.semantic.ir import QueryIR
from text2sql.semantic.linker import find_ambiguous_terms, find_metric_mentions, find_value_mentions
from text2sql.semantic.registry import SemanticModel
from text2sql.semantic.timeparse import normalize, parse_time, vague_time_word

# Words asking how a figure is defined or where data lives (metadata questions).
META_WORDS = re.compile(
    r"怎么算|如何计算|怎么计算|计算方法|计算公式|计算逻辑|口径|定义|什么意思|指什么|含义|包含哪些|包括哪些"
    r"|哪张表|哪个表|哪些表|哪个字段|哪些字段|在哪张表|记录了")
# Words asking for a figure, whatever the domain (data requests).
DATA_WORDS = re.compile(
    r"多少|几|统计|趋势|对比|同比|环比|排名|最多|最高|最低|占比|比例|合计|总数|分布|增长|下降|变化|查询|查一下")
# Words that may remain once the metric, time and region of a fast-path question are removed.
FILLER_WORDS = re.compile(
    r"请问|我想知道|告诉我|帮我|查一下|查看|看一下|看看|一下|查询|统计|总共|一共|合计|多少"
    r"|期间|以来|至今|分别|的|是|有|为|共|几|吗|呢|啊|呀|吧|请|[？?，,。.！!、：:\s]")
# Business words not covered by the semantic layer's names, aliases and values.
GENERIC_DOMAIN_TERMS = [
    "数据", "指标", "交易", "账户", "贷款", "放款", "风险", "金额", "笔数", "客户", "卡", "地区", "区县", "余额",
    "代扣", "取现", "存入", "转账", "不良", "违约", "事件", "开户", "流水", "罚息", "透支", "利息", "银行",
    "口径", "字段", "表", "损失", "规模", "预警", "案件", "户主", "养老金", "保险",
]


def domain_terms(model: SemanticModel) -> set[str]:
    """Every name, alias and value label of the semantic layer, plus generic business words."""
    terms = set(GENERIC_DOMAIN_TERMS) | set(model.ambiguous_terms)
    for m in model.metrics.values():
        terms |= {m.name, *m.aliases}
    for d in model.dimensions.values():
        terms |= {d.name, *d.aliases}
        for v in d.values:
            terms |= {t for t in (v.label, *v.aliases) if len(t) >= 2}
    return terms


class Rules:
    def __init__(self, model: SemanticModel, use_fast_path: bool = True):
        self.model = model
        self.use_fast_path = use_fast_path
        self._domain_terms = domain_terms(model)

    def decide(self, question: str) -> RouteDecision | None:
        """Decisions that need no model; None hands the question to the model."""
        if META_WORDS.search(question) and not DATA_WORDS.search(question):
            return RouteDecision("META", "rule", "命中口径与数据资产问答模式")
        if not any(term in question for term in self._domain_terms) and not DATA_WORDS.search(question):
            return RouteDecision("OOD", "rule", "既无业务领域词，也无取数词")
        ambiguous = find_ambiguous_terms(question, self.model)
        if ambiguous:
            options = [c for m in ambiguous for c in self.model.ambiguous_terms[m.key]]
            terms = "、".join(dict.fromkeys(m.key for m in ambiguous))
            return RouteDecision("CLARIFY", "rule", f"多义词：{terms}", clarify_kind="metric",
                                 clarify_options=list(dict.fromkeys(options)))
        vague = vague_time_word(question)
        if vague:
            return RouteDecision("CLARIFY", "rule", f"时间表达“{vague}”不明确", clarify_kind="time")
        if self.use_fast_path:
            return self._fast_path(question)
        return None

    def _fast_path(self, question: str) -> RouteDecision | None:
        """Single metric, optional time and region values, and nothing else but filler words."""
        text = normalize(question)
        metrics = find_metric_mentions(text, self.model)
        if len(metrics) != 1:
            return None
        spans = [(metrics[0].start, metrics[0].end)]
        span = parse_time(text)
        if span:
            pos = text.find(span.text)
            spans.append((pos, pos + len(span.text)))
        regions = find_value_mentions(text, self.model, ["region"])
        spans += [(r.start, r.end) for r in regions]
        residual = "".join(ch for i, ch in enumerate(text) if not any(s <= i < e for s, e in spans))
        if FILLER_WORDS.sub("", residual):
            return None
        filters = {"region": [c for r in regions for c in r.codes]} if regions else {}
        ir = QueryIR(metrics=[metrics[0].key], query_type="point",
                     time_range=(span.start, span.end) if span else None, filters=filters)
        return RouteDecision(semantic_layer(ir, self.model), "fast_path", "单指标简单问题，规则完成解析", ir=ir,
                             assumptions=[] if span else [NO_TIME_NOTE])
