"""Execution accuracy (EX): compare a predicted result set with the gold result set.

Two judgments are provided.

bird_ex     The official BIRD rule: set(pred_rows) == set(gold_rows) on raw values.
lenient_ex  Used for the domain set. The prediction may carry extra explanatory columns:
            it is correct when every gold column maps to a distinct predicted column and the
            projected rows equal the gold rows as a multiset (as a sequence when order
            matters). Numbers are equal up to 1e-6 relative error, or when the less precise
            side is a rounding (1 to 4 decimals) of the other. Strings ignore case and
            surrounding spaces.
"""

from __future__ import annotations

from typing import Any, Sequence

Row = Sequence[Any]


def bird_ex(pred: list[Row], gold: list[Row]) -> bool:
    return set(map(tuple, pred)) == set(map(tuple, gold))


def _decimals(x: float | int) -> int:
    if isinstance(x, int):
        return 0
    text = repr(float(x))
    if "e" in text or "E" in text:
        return 17
    return len(text.split(".", 1)[1]) if "." in text else 0


def _norm(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    text = str(v).strip()
    try:
        return int(text) if text.lstrip("-").isdigit() else float(text)
    except ValueError:
        return text.casefold()


def values_equal(a: Any, b: Any) -> bool:
    a, b = _norm(a), _norm(b)
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b or abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b)):
            return True
        p = min(_decimals(a), _decimals(b))
        return 1 <= p <= 4 and abs(round(a, p) - round(b, p)) < 10 ** -(p + 3)
    return a == b


def _sort_key(v: Any):
    v = _norm(v)
    if v is None:
        return (0, 0, "")
    if isinstance(v, (int, float)):
        return (1, float(v), "")
    return (2, 0, str(v))


def _column(rows: list[Row], j: int) -> list[Any]:
    return sorted((r[j] for r in rows), key=_sort_key)


def _columns_match(pred: list[Row], k: int, gold: list[Row], j: int) -> bool:
    return all(values_equal(a, b) for a, b in zip(_column(pred, k), _column(gold, j)))


def _rows_match(pred: list[Row], gold: list[Row], ordered: bool) -> bool:
    if not ordered:
        key = lambda r: tuple(_sort_key(v) for v in r)  # noqa: E731
        pred, gold = sorted(pred, key=key), sorted(gold, key=key)
    return all(all(values_equal(a, b) for a, b in zip(pr, gr)) for pr, gr in zip(pred, gold))


def lenient_ex(pred: list[Row], gold: list[Row], order_matters: bool = False) -> bool:
    if len(pred) != len(gold):
        return False
    if not gold:
        return True
    n_gold, n_pred = len(gold[0]), len(pred[0])
    if n_pred < n_gold:
        return False
    candidates = [
        [k for k in range(n_pred) if _columns_match(pred, k, gold, j)] for j in range(n_gold)
    ]

    def search(j: int, used: tuple[int, ...]) -> bool:
        if j == n_gold:
            projected = [tuple(r[k] for k in used) for r in pred]
            return _rows_match(projected, gold, order_matters)
        return any(search(j + 1, used + (k,)) for k in candidates[j] if k not in used)

    return search(0, ())
