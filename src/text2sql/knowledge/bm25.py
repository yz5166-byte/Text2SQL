"""BM25 over mixed Chinese and English text: Chinese as character bigrams, English as words."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Hashable, Mapping

CHINESE_RUN = re.compile(r"[一-鿿]+")
WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """English words (snake_case split) plus the character bigrams of each Chinese run."""
    text = text.lower()
    tokens = WORD.findall(text.replace("_", " "))
    for run in CHINESE_RUN.findall(text):
        tokens += [run[i:i + 2] for i in range(len(run) - 1)] or [run]
    return tokens


class BM25:
    def __init__(self, docs: Mapping[Hashable, str], k1: float = 1.5, b: float = 0.75):
        self.keys = list(docs)
        self.tf = [Counter(tokenize(docs[k])) for k in self.keys]
        self.length = [sum(tf.values()) for tf in self.tf]
        self.avg_length = sum(self.length) / len(self.length)
        df = Counter(term for tf in self.tf for term in tf)
        n = len(self.keys)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> dict[Hashable, float]:
        terms = tokenize(query)
        out = {}
        for key, tf, length in zip(self.keys, self.tf, self.length):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_length)
            out[key] = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
        return out
