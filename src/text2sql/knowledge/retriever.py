"""Table retrieval for layer 3: rank tables, keep the top k, then add the tables that join them.

Each table is one document (see units.py), scored by BM25, by embeddings, or by both fused
with reciprocal rank fusion (RRF), which combines ranks rather than raw scores that live on
different scales. The join bridge follows the foreign keys declared in the catalog
("关联 table.column") and adds every table on a shortest path between two retrieved tables,
such as disp and account between client and loan: tables a question never names but its
JOIN cannot do without.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np

from text2sql.knowledge import embedding
from text2sql.knowledge.bm25 import BM25
from text2sql.knowledge.units import table_units
from text2sql.semantic.registry import SemanticModel

Method = Literal["bm25", "vector", "hybrid"]
RRF_K = 60  # the usual constant: damps the weight of the very top ranks
FOREIGN_KEY = re.compile(r"关联 (\w+)\.\w+")


@dataclass
class Retrieval:
    tables: list[str]  # the top k, best first
    bridged: list[str]  # added to connect them

    @property
    def all(self) -> list[str]:
        return self.tables + self.bridged


class HybridIndex:
    """Ranks documents for a query by BM25, by embeddings, or by both fused with RRF."""

    def __init__(self, docs: Mapping[str, str], method: Method = "hybrid"):
        self.docs, self.method = dict(docs), method
        self.keys = list(self.docs)
        self.bm25 = BM25(self.docs)
        self._vectors: np.ndarray | None = None  # document embeddings, computed on first vector query

    def rank(self, text: str) -> list[str]:
        scores = self._scores(text)
        return sorted(scores, key=lambda k: -scores[k])

    def _scores(self, text: str) -> Mapping[str, float]:
        if self.method == "bm25":
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
        self.index = HybridIndex(table_units(model), method)
        self.graph = foreign_key_graph(model)

    def retrieve(self, text: str) -> Retrieval:
        top = self.index.rank(text)[: self.k]
        return Retrieval(top, bridge_tables(top, self.graph) if self.bridge else [])


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
