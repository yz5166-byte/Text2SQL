"""Dictionary-based linking between user wording and the semantic layer.

The dictionaries are the governed vocabularies in semantic/*.yaml. Three jobs:
1. find_metric_mentions   which metrics a question names (longest match over names and aliases)
2. find_ambiguous_terms   declared ambiguous words (损失 ...) not covered by a longer metric name
3. link_value             one value phrase (取现, 南摩拉维亚, 不良) mapped to the codes of one dimension

Metrics are found by longest match. A value links by exact key first, and by similarity only
when one value is the unique best at or above 0.85; a link made by similarity is disclosed in
the answer.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Iterator

from text2sql.semantic.registry import Code, SemanticModel

FUZZY_THRESHOLD = 0.85


def _key(text: Code) -> str:
    """Comparison key: case-insensitive, without spaces, hyphens, underscores or dots."""
    return re.sub(r"[\s\-_.·]", "", str(text)).casefold()


@dataclass(frozen=True)
class Mention:
    key: str  # metric key, ambiguous term, or dimension key
    surface: str  # the matched words in the question
    start: int
    end: int
    codes: tuple[Code, ...] = ()  # database codes, for value mentions


@dataclass(frozen=True)
class ValueLink:
    codes: tuple[Code, ...]
    term: str  # the dictionary term that matched
    fuzzy: bool  # matched by similarity rather than equality


# ---------------------------------------------------------------- scanning a whole question
def _occurrences(text: str, term: str) -> Iterator[tuple[int, int]]:
    start = text.find(term)
    while start != -1:
        yield start, start + len(term)
        start = text.find(term, start + 1)


def longest_match(candidates: list[Mention]) -> list[Mention]:
    """Keep candidates from the longest down, dropping any that overlaps one already kept."""
    kept: list[Mention] = []
    for c in sorted(candidates, key=lambda c: (c.start - c.end, c.start)):
        if all(c.end <= k.start or c.start >= k.end for k in kept):
            kept.append(c)
    return sorted(kept, key=lambda c: c.start)


def find_metric_mentions(text: str, model: SemanticModel) -> list[Mention]:
    return longest_match([
        Mention(key, term, s, e)
        for key, metric in model.metrics.items()
        for term in dict.fromkeys([metric.name, *metric.aliases])
        for s, e in _occurrences(text, term)
    ])


def find_dimension_mentions(text: str, model: SemanticModel) -> list[Mention]:
    """Dimensions named in free text by name or alias (地区, 卡等级), keyed by dimension."""
    return longest_match([
        Mention(key, term, s, e)
        for key, dim in model.dimensions.items()
        for term in dict.fromkeys([dim.name, *dim.aliases])
        for s, e in _occurrences(text, term)
    ])


def find_ambiguous_terms(text: str, model: SemanticModel) -> list[Mention]:
    """Ambiguous words that are not part of a longer metric name or alias."""
    metrics = find_metric_mentions(text, model)
    return [
        Mention(term, term, s, e)
        for term in model.ambiguous_terms
        for s, e in _occurrences(text, term)
        if all(e <= m.start or s >= m.end for m in metrics)
    ]


# ---------------------------------------------------------------- linking one value phrase
def _value_terms(model: SemanticModel, dim_key: str) -> Iterator[tuple[str, tuple[Code, ...]]]:
    """(term, codes) for every label, alias and single code of a dimension's values.

    Terms come in a fixed order (label, aliases, code) so that ties resolve the same way in every run.
    """
    for value in model.dimensions[dim_key].values:
        single_code = [str(value.codes[0])] if len(value.codes) == 1 and value.codes[0] is not None else []
        for term in dict.fromkeys([value.label, *value.aliases, *single_code]):
            yield term, tuple(value.codes)


def link_value(model: SemanticModel, dim_key: str, phrase: Code) -> ValueLink | None:
    """Exact key match; otherwise the unique most similar value at or above the threshold."""
    if phrase is None:
        return None
    target = _key(phrase)
    terms = list(_value_terms(model, dim_key))
    for term, codes in terms:
        if _key(term) == target:
            return ValueLink(codes, term, fuzzy=False)

    best: dict[tuple[Code, ...], tuple[float, str]] = {}  # best score per distinct value
    for term, codes in terms:
        score = difflib.SequenceMatcher(None, _key(term), target).ratio()
        if score > best.get(codes, (0.0, ""))[0]:
            best[codes] = (score, term)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])
    if not ranked or ranked[0][1][0] < FUZZY_THRESHOLD:
        return None
    if len(ranked) > 1 and ranked[1][1][0] == ranked[0][1][0]:
        return None  # two different values tie: not unique, do not guess
    codes, (_, term) = ranked[0]
    return ValueLink(codes, term, fuzzy=True)


def link_value_any(model: SemanticModel, dims: list[str], phrase: Code) -> list[tuple[str, ValueLink]]:
    """Every dimension among `dims` in which `phrase` links; used to repair a misplaced filter."""
    return [(d, link) for d in dims if (link := link_value(model, d, phrase)) is not None]


def find_value_mentions(text: str, model: SemanticModel, dims: list[str]) -> list[Mention]:
    """Values of the given dimensions named in free text (terms of two or more characters)."""
    return longest_match([
        Mention(dim, term, s, e, codes)
        for dim in dims
        for term, codes in _value_terms(model, dim)
        if len(term) >= 2
        for s, e in _occurrences(text, term)
    ])
