"""BM25 over mixed Chinese and English text: Chinese as character bigrams, English as words.

English words lose a plural ending (accounts, districts, salaries), on documents and questions
alike, so that a question about accounts finds the account table. Measured on 75 tables (BIRD
financial plus the distractor tables), this raised the BIRD questions with every needed table
among the top 5 from 35 to 42 of 106 (scripts/eval_retrieval.py --distractors).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Hashable, Mapping

CHINESE_RUN = re.compile(r"[一-鿿]+")
WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """English words (snake_case split, plural ending removed) plus the character bigrams of each Chinese run."""
    text = text.lower()
    tokens = [singular(w) for w in WORD.findall(text.replace("_", " "))]
    for run in CHINESE_RUN.findall(text):
        tokens += [run[i:i + 2] for i in range(len(run) - 1)] or [run]
    return tokens


def singular(word: str) -> str:
    """A crude plural strip, the same on both sides of the match: salaries -> salary, accounts -> account."""
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


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
