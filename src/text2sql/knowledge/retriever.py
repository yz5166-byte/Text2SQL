"""Schema retrieval for layer 3: rank tables, keep the top k, then add the tables that join them.

Each table is one document (see units.py), scored by BM25, by embeddings, or by both fused
with reciprocal rank fusion (RRF), which combines ranks rather than raw scores that live on
different scales. The join bridge follows the foreign keys declared in the catalog
("关联 table.column") and adds every table on a shortest path between two retrieved tables,
such as disp and account between client and loan: tables a question never names but its
JOIN cannot do without. Tables found another way, such as those holding a value the question
names (values.py), join the candidates before the bridge is computed.

Columns are ranked the same way, one document per column, optionally within given tables.

With a Chinese-only embedding model (bge-small-zh), the schema retrievers' hybrid ranking falls
back to BM25 alone for a query without Chinese characters: on 75 tables that model's rankings of
English BIRD questions left no question with every needed table among the top 5, and fused with
BM25 they pulled it from 36 to 9 of 106. The multilingual default (bge-m3) needs no such rule:
fused, it finds 56 (scripts/eval_retrieval.py --distractors). The example library (examples.py)
holds Chinese questions only and always keeps the fusion.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import numpy as np

from text2sql.knowledge import embedding
from text2sql.knowledge.bm25 import BM25
from text2sql.knowledge.units import column_units, table_units
from text2sql.semantic.registry import SemanticModel

Method = Literal["bm25", "vector", "hybrid"]
RRF_K = 60  # the usual constant: damps the weight of the very top ranks
CHINESE = re.compile(r"[一-鿿]")
FOREIGN_KEY = re.compile(r"关联 (\w+)\.\w+")


@dataclass
class Retrieval:
    tables: list[str]  # the top k, best first
    bridged: list[str]  # added to connect them
    linked: list[str] = field(default_factory=list)  # added from outside the ranking, such as value hits

    @property
    def all(self) -> list[str]:
        return self.tables + self.linked + self.bridged


class HybridIndex:
    """Ranks documents for a query by BM25, by embeddings, or by both fused with RRF."""

    def __init__(self, docs: Mapping[str, str], method: Method = "hybrid", bm25_unless_chinese: bool = False):
        self.docs, self.method, self.bm25_unless_chinese = dict(docs), method, bm25_unless_chinese
        self.keys = list(self.docs)
        self.bm25 = BM25(self.docs)
        self._vectors: np.ndarray | None = None  # document embeddings, computed on first vector query

    def rank(self, text: str) -> list[str]:
        scores = self._scores(text)
        return sorted(scores, key=lambda k: -scores[k])

    def _scores(self, text: str) -> Mapping[str, float]:
        if self.method == "bm25" or (self.method == "hybrid" and self.bm25_unless_chinese and not CHINESE.search(text)):
            return self.bm25.scores(text)
        if self.method == "vector":
            return self._vector_scores(text)
        return rrf(self.bm25.scores(text), self._vector_scores(text))

    def _vector_scores(self, text: str) -> dict[str, float]:
        if self._vectors is None:
            self._vectors = embedding.embed([self.docs[k] for k in self.keys])
        return dict(zip(self.keys, (self._vectors @ embedding.embed([text])[0]).tolist()))


class TableRetriever:
    def __init__(self, model: SemanticModel, method: Method = "hybrid", k: int = 5, bridge: bool = True):
        self.k, self.bridge = k, bridge
        self.index = HybridIndex(table_units(model), method, bm25_unless_chinese=not embedding.multilingual())
        self.graph = foreign_key_graph(model)

    def retrieve(self, text: str, linked: list[str] = ()) -> Retrieval:
        """The top k tables, then `linked` tables not among them, then the bridge over both."""
        top = self.index.rank(text)[: self.k]
        extra = [t for t in dict.fromkeys(linked) if t not in top and t in self.graph]
        return Retrieval(top, bridge_tables(top + extra, self.graph) if self.bridge else [], extra)


class ColumnRetriever:
    """Detailed column documents by default: they recall more of the non-key columns (scripts/eval_columns.py)."""

    def __init__(self, model: SemanticModel, method: Method = "hybrid", detailed: bool = True):
        self.index = HybridIndex(column_units(model, detailed), method, bm25_unless_chinese=not embedding.multilingual())

    def rank(self, text: str, tables: list[str] | None = None) -> list[str]:
        """Columns as "table.column", best first; only those of `tables` when given."""
        ranked = self.index.rank(text)
        if tables is None:
            return ranked
        keep = set(tables)
        return [c for c in ranked if c.split(".")[0] in keep]


def rrf(*rankings: Mapping[str, float]) -> dict[str, float]:
    """Sum of 1 / (RRF_K + rank) over the score maps, rank counted from 0; ties keep catalog order."""
    fused: dict[str, float] = {}
    for scores in rankings:
        for rank, key in enumerate(sorted(scores, key=lambda k: -scores[k])):
            fused[key] = fused.get(key, 0.0) + 1 / (RRF_K + rank)
    return fused


def foreign_key_graph(model: SemanticModel) -> dict[str, set[str]]:
    """Undirected table graph from the catalog's "关联 table.column" notes."""
    graph: dict[str, set[str]] = {t: set() for t in model.tables}
    for t, table in model.tables.items():
        for col in table.columns.values():
            m = FOREIGN_KEY.search(col.description)
            if m and m.group(1) in graph and m.group(1) != t:
                graph[t].add(m.group(1))
                graph[m.group(1)].add(t)
    return graph


def key_columns(model: SemanticModel) -> set[str]:
    """Primary and foreign keys as "table.column", from the catalog's 主键 and 关联 notes."""
    return {f"{t}.{c}" for t, table in model.tables.items() for c, col in table.columns.items()
            if "主键" in col.description or "关联" in col.description}


def bridge_tables(tables: list[str], graph: dict[str, set[str]]) -> list[str]:
    """Tables on any shortest path between two of `tables`, excluding `tables`, in catalog order."""
    dist = {t: _distances(graph, t) for t in tables}
    on_path: set[str] = set()
    for a, b in combinations(tables, 2):
        if b in dist[a]:
            on_path |= {x for x in graph if x in dist[a] and x in dist[b] and dist[a][x] + dist[b][x] == dist[a][b]}
    return [t for t in graph if t in on_path and t not in tables]


def _distances(graph: dict[str, set[str]], start: str) -> dict[str, int]:
    dist, queue = {start: 0}, deque([start])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    return dist
