"""Value index: the distinct values of the database's text columns, found in a question.

Questions often name a value as it is stored ("gold" cards, "east Bohemia", "POPLATEK
MESICNE"), and the value tells which column a filter belongs on and which table the query
needs. Only text columns with at most MAX_DISTINCT values are indexed, and a value without a
letter is left out, since numbers and dates stored as text (district.A4 to A7) are not names.

A value matches when its normalized form occurs in the question as whole words. Values
shorter than MIN_LENGTH characters (gender F, loan status A) and a few common words that
happen to be stored values (ALL) are skipped, since they occur in questions by accident. A
proper name (a value that starts with a capital and is not all capitals, such as the district
Most) must also start with a capital in the question, so that "the most accounts" does not
match it, and a code of at most SHORT_CODE capitals (BAD, PER) must appear exactly as stored,
so that "bad loans" and "per month" do not match it. A misspelt value can still match by character-trigram similarity, when one value is
the unique best for a span of the question at FUZZY_THRESHOLD or above; such a hit is marked
fuzzy.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from text2sql.semantic.registry import SemanticModel

MAX_DISTINCT = 1000
MIN_LENGTH = 3
FUZZY_THRESHOLD = 0.8
SHORT_CODE = 3
COMMON_WORDS = {"all", "and", "any", "none", "null", "other", "the"}
LETTER = re.compile(r"[^\W\d_]")
TOKEN = re.compile(r"\S+")
PUNCTUATION = ".,;:!?()\"'"


@dataclass(frozen=True)
class ValueHit:
    value: str  # as stored
    columns: tuple[str, ...]  # "table.column", every column that stores it
    surface: str  # as written in the question
    fuzzy: bool = False

    @property
    def tables(self) -> list[str]:
        return list(dict.fromkeys(c.split(".")[0] for c in self.columns))


class ValueIndex:
    def __init__(self, values: dict[str, list[str]]):
        """`values` maps "table.column" to its distinct stored values."""
        self.entries: dict[str, tuple[str, list[str]]] = {}  # normalized -> (stored value, columns)
        for column, stored in values.items():
            for value in stored:
                key = normalize(value)
                if len(key) >= MIN_LENGTH and key not in COMMON_WORDS and LETTER.search(key):
                    self.entries.setdefault(key, (value, []))[1].append(column)
        self.patterns = {k: re.compile(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])") for k in self.entries}
        self.by_words: dict[int, list[str]] = defaultdict(list)  # fuzzy candidates by word count
        for k in self.entries:
            self.by_words[len(k.split())].append(k)
        self.grams = {k: trigrams(k) for k in self.entries}

    @classmethod
    def build(cls, model: SemanticModel, db_path: Path) -> ValueIndex:
        """Index the TEXT columns of the model's tables that have at most MAX_DISTINCT values.

        A column of a sampled table is judged by its distinct count in the full table (Column.distinct).
        """
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        values = {}
        try:
            for t, table in model.tables.items():
                for c, col in table.columns.items():
                    if col.type != "TEXT":
                        continue
                    distinct = col.distinct
                    if distinct is None:
                        distinct = con.execute(f'SELECT COUNT(DISTINCT "{c}") FROM "{t}"').fetchone()[0]
                    if distinct > MAX_DISTINCT:
                        continue
                    rows = con.execute(f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL')
                    values[f"{t}.{c}"] = [r[0] for r in rows if isinstance(r[0], str)]
        finally:
            con.close()
        return cls(values)

    def match(self, question: str) -> list[ValueHit]:
        """Whole-word matches, longest first so that VYBER inside VYBER KARTOU is not counted, then fuzzy ones."""
        cased = fold(question)
        text = cased.lower()
        if len(text) != len(cased):  # a character whose lower case is longer: compare without case
            cased = text
        found = [(m.start(), m.end(), k) for k, p in self.patterns.items() for m in p.finditer(text)
                 if self._cased(k, cased, m.start())]
        taken: list[tuple[int, int]] = []
        hits: dict[str, ValueHit] = {}
        for start, end, key in sorted(found, key=lambda f: f[0] - f[1]):
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            if key not in hits:
                value, columns = self.entries[key]
                hits[key] = ValueHit(value, tuple(columns), cased[start:end])
        for start, end, key in self._fuzzy(text, taken):
            if key not in hits and self._cased(key, cased, start):
                value, columns = self.entries[key]
                hits[key] = ValueHit(value, tuple(columns), cased[start:end], fuzzy=True)
        return list(hits.values())

    def _cased(self, key: str, cased: str, start: int) -> bool:
        """A proper name must start with a capital in the question too; a short code must match exactly."""
        value = self.entries[key][0]
        if value.isupper() and len(value) <= SHORT_CODE:
            return cased[start:start + len(value)] == value
        return not (value[0].isupper() and not value.isupper()) or cased[start].isupper()

    def _fuzzy(self, text: str, taken: list[tuple[int, int]]):
        """Spans of n words, longest n first, whose unique best value of n words is similar enough."""
        tokens = [(m.start(), m.end()) for m in TOKEN.finditer(text)]
        for n in sorted(self.by_words, reverse=True):
            keys = self.by_words[n]
            for i in range(len(tokens) - n + 1):
                raw = text[tokens[i][0]:tokens[i + n - 1][1]]
                span = raw.strip(PUNCTUATION)
                start = tokens[i][0] + len(raw) - len(raw.lstrip(PUNCTUATION))
                end = start + len(span)
                if len(span) < MIN_LENGTH or any(start < e and s < end for s, e in taken):
                    continue
                grams = trigrams(span)
                scored = sorted(((dice(grams, self.grams[k]), k) for k in keys), reverse=True)
                if scored[0][0] >= FUZZY_THRESHOLD and (len(scored) == 1 or scored[1][0] < scored[0][0]):
                    taken.append((start, end))
                    yield start, end, scored[0][1]


def normalize(text: str) -> str:
    """Lower case, accents removed, whitespace collapsed."""
    return fold(text).lower()


def fold(text: str) -> str:
    """Accents removed, whitespace collapsed, case kept."""
    text = unicodedata.normalize("NFKD", text)
    return " ".join("".join(ch for ch in text if not unicodedata.combining(ch)).split())


def trigrams(text: str) -> set[str]:
    padded = f"  {text} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def dice(a: set[str], b: set[str]) -> float:
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0
