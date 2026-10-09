"""Deterministic parser for Chinese time expressions.

The model only copies the time phrase out of a question; this module turns it into an
inclusive date range. Rules are tried in a fixed order and the first match wins.

Conventions
  "today" is the reference date: 1998-12-31, the last day with data
  上个月 / 上季度 / 去年 are calendar periods, not rolling windows
  近 N 个月 includes the current month
  a two-digit year resolves to the century nearest the reference year (97 -> 1997)
  mom baseline: whole months shift by months, any other range shifts by its length in days
  yoy baseline: the same range one year earlier (29 February -> 28 February)
  vague words such as 最近 are never guessed; the caller asks the user instead
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

from text2sql.config import REFERENCE_DATE

Range = tuple[date, date]


@dataclass(frozen=True)
class TimeSpan:
    start: date
    end: date
    text: str  # the matched phrase inside normalize(question)


# ---------------------------------------------------------------- calendar helpers
def month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def month_range(y: int, m: int) -> Range:
    return date(y, m, 1), month_end(y, m)


def quarter_range(y: int, q: int) -> Range:
    return date(y, 3 * q - 2, 1), month_end(y, 3 * q)


def half_range(y: int, first: bool) -> Range:
    return (date(y, 1, 1), date(y, 6, 30)) if first else (date(y, 7, 1), date(y, 12, 31))


def year_range(y: int) -> Range:
    return date(y, 1, 1), date(y, 12, 31)


def shift_month(y: int, m: int, k: int) -> tuple[int, int]:
    """(year, month) moved by k months."""
    i = y * 12 + (m - 1) + k
    return i // 12, i % 12 + 1


def is_month_aligned(start: date, end: date) -> bool:
    return start.day == 1 and end == month_end(end.year, end.month)


# ---------------------------------------------------------------- normalization
_FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")
_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def cn_to_int(s: str) -> int:
    """Chinese numerals up to 99: 三 -> 3, 十二 -> 12, 二十 -> 20, 九八 -> 98."""
    if "十" not in s:
        return int("".join(str(_CN_DIGIT[c]) for c in s))
    tens, _, ones = s.partition("十")
    return _CN_DIGIT.get(tens, 1) * 10 + _CN_DIGIT.get(ones, 0)


def normalize(text: str) -> str:
    """ASCII digits, no spaces, and Chinese numerals before a time unit turned into digits."""
    text = text.translate(_FULLWIDTH).replace(" ", "")
    text = re.sub(r"[qｑ](?=[1-4])", "Q", text)
    return re.sub(r"[零一二两三四五六七八九十]+(?=个月|月|日|号|季度|季|天|周|年)",
                  lambda m: str(cn_to_int(m.group())), text)


# A year followed by 年: four or two digits, or a relative word (今年, 去年, 上一年, 前年 ...).
YEAR = r"(\d{4}|\d{2}|今|本|去|上一|上|前)年"
_RELATIVE_YEAR = {"今": 0, "本": 0, "去": -1, "上一": -1, "上": -1, "前": -2}


def to_year(token: str, ref: date) -> int:
    if token in _RELATIVE_YEAR:
        return ref.year + _RELATIVE_YEAR[token]
    y = int(token)
    if y >= 100:
        return y
    century = ref.year // 100
    return min(((century + k) * 100 + y for k in (-1, 0, 1)), key=lambda c: abs(c - ref.year))


# ---------------------------------------------------------------- rule table
Resolver = Callable[[re.Match, date], Range]
SEP = r"(?:到|至|~|～|—|-)"


def _day_range(m: re.Match, ref: date) -> Range:
    y1, m1 = to_year(m[1], ref), int(m[2])
    y2 = to_year(m[4], ref) if m[4] else y1
    m2 = int(m[5]) if m[5] else m1
    return date(y1, m1, int(m[3])), date(y2, m2, int(m[6]))


def _month_span(m: re.Match, ref: date) -> Range:
    y1 = to_year(m[1], ref)
    y2 = to_year(m[3], ref) if m[3] else y1
    return date(y1, int(m[2]), 1), month_end(y2, int(m[4]))


def _quarter_span(m: re.Match, ref: date) -> Range:
    y1 = to_year(m[1], ref)
    y2 = to_year(m[3], ref) if m[3] else y1
    return quarter_range(y1, int(m[2]))[0], quarter_range(y2, int(m[4]))[1]


def _previous_quarter(ref: date) -> Range:
    q = (ref.month - 1) // 3 + 1
    return quarter_range(ref.year, q - 1) if q > 1 else quarter_range(ref.year - 1, 4)


RULES: list[tuple[re.Pattern, Resolver]] = [(re.compile(p), f) for p, f in [
    # 1. year to date and "last N units": checked first so that 近12年 is never read as the year 2012
    (r"今年以来|本年以来|年初至今|年初以来|今年至今", lambda m, r: (date(r.year, 1, 1), r)),
    (r"(?:近|最近|过去)(\d+)个?月", lambda m, r: (date(*shift_month(r.year, r.month, 1 - int(m[1])), 1), r)),
    (r"(?:近|最近|过去)半年", lambda m, r: (date(*shift_month(r.year, r.month, -5), 1), r)),
    (r"(?:近|最近|过去)(\d+)年", lambda m, r: (date(r.year - int(m[1]) + 1, 1, 1), r)),
    (r"(?:近|最近|过去)(\d+)(?:天|日)", lambda m, r: (r - timedelta(days=int(m[1]) - 1), r)),
    (r"(?:近|最近|过去)(\d+)周", lambda m, r: (r - timedelta(days=7 * int(m[1]) - 1), r)),
    # 2. explicit ranges, before single units so that 1995年到1997年 is not read as 1995 alone
    (rf"(\d{{4}})-(\d{{1,2}})-(\d{{1,2}}){SEP}(\d{{4}})-(\d{{1,2}})-(\d{{1,2}})",
     lambda m, r: (date(int(m[1]), int(m[2]), int(m[3])), date(int(m[4]), int(m[5]), int(m[6])))),
    (rf"{YEAR}(\d{{1,2}})月(\d{{1,2}})[日号](?:到|至|~|～)(?:{YEAR})?(?:(\d{{1,2}})月)?(\d{{1,2}})[日号]", _day_range),
    (rf"{YEAR}(\d{{1,2}})月?{SEP}(?:{YEAR})?(\d{{1,2}})月", _month_span),
    (rf"{YEAR}第?([1-4])季度?(?:到|至)(?:{YEAR})?第?([1-4])季度", _quarter_span),
    (rf"(\d{{4}}|\d{{2}}|今|本|去|上一|上|前)年?{SEP}{YEAR}",
     lambda m, r: (year_range(to_year(m[1], r))[0], year_range(to_year(m[2], r))[1])),
    # 3. single units, from the finest to the coarsest
    (r"(\d{4})-(\d{1,2})-(\d{1,2})", lambda m, r: (date(int(m[1]), int(m[2]), int(m[3])),) * 2),
    (rf"{YEAR}(\d{{1,2}})月(\d{{1,2}})[日号]", lambda m, r: (date(to_year(m[1], r), int(m[2]), int(m[3])),) * 2),
    (rf"{YEAR}(上|下)半年", lambda m, r: half_range(to_year(m[1], r), first=m[2] == "上")),
    (rf"{YEAR}第?([1-4])季度?", lambda m, r: quarter_range(to_year(m[1], r), int(m[2]))),
    (r"(\d{4})年?Q([1-4])", lambda m, r: quarter_range(int(m[1]), int(m[2]))),
    (rf"{YEAR}(\d{{1,2}})月", lambda m, r: month_range(to_year(m[1], r), int(m[2]))),
    (r"(\d{4})-(\d{1,2})(?![\d-])", lambda m, r: month_range(int(m[1]), int(m[2]))),
    (YEAR, lambda m, r: year_range(to_year(m[1], r))),
    # 4. calendar periods relative to today (after step 3, so 去年12月 is read as a month)
    (r"上个?季度|上一季度|上季", lambda m, r: _previous_quarter(r)),
    (r"本季度|这个季度|本季|当季", lambda m, r: quarter_range(r.year, (r.month - 1) // 3 + 1)),
    (r"上个?月|上一个?月", lambda m, r: month_range(*shift_month(r.year, r.month, -1))),
    (r"本月|这个月|当月", lambda m, r: month_range(r.year, r.month)),
    (r"上半年", lambda m, r: half_range(r.year, first=True)),
    (r"下半年", lambda m, r: half_range(r.year, first=False)),
    (r"前天", lambda m, r: (r - timedelta(days=2),) * 2),
    (r"昨天|昨日", lambda m, r: (r - timedelta(days=1),) * 2),
    (r"今天|今日", lambda m, r: (r, r)),
]]

VAGUE = re.compile(r"最近|近期|近来|目前|当前|现在|前段时间|这段时间|这阵子")


# ---------------------------------------------------------------- public API
def parse_time(text: str | None, ref: date = REFERENCE_DATE) -> TimeSpan | None:
    """The first time expression in `text` as an inclusive range; None when there is none or it is invalid."""
    norm = normalize(text or "")
    for pattern, resolve in RULES:
        m = pattern.search(norm)
        if m:
            try:
                start, end = resolve(m, ref)
            except ValueError:  # an impossible date such as 2月30日
                return None
            return TimeSpan(start, end, m.group()) if start <= end else None
    return None


def find_times(text: str | None, ref: date = REFERENCE_DATE) -> list[TimeSpan]:
    """Every time expression in `text`, one per distinct range: mask each match and search again."""
    rest = normalize(text or "")
    found: dict[Range, TimeSpan] = {}
    while (span := parse_time(rest, ref)) is not None and span.text in rest:
        found.setdefault((span.start, span.end), span)
        rest = rest.replace(span.text, "#", 1)
    return list(found.values())


def vague_time_word(text: str) -> str | None:
    """A vague time word (最近, 目前 ...) when the text holds no parseable time expression."""
    if parse_time(text) is not None:
        return None
    m = VAGUE.search(normalize(text))
    return m.group() if m else None


def baseline_range(start: date, end: date, compare: str) -> Range:
    """Baseline period of a comparison: yoy = one year earlier, mom = the adjacent previous period."""
    if compare == "yoy":
        return _minus_year(start), _minus_year(end)
    if is_month_aligned(start, end):
        n = (end.year - start.year) * 12 + end.month - start.month + 1
        return date(*shift_month(start.year, start.month, -n), 1), month_end(*shift_month(end.year, end.month, -n))
    days = (end - start).days + 1
    return start - timedelta(days=days), start - timedelta(days=1)


def _minus_year(d: date) -> date:
    return d.replace(year=d.year - 1, day=min(d.day, calendar.monthrange(d.year - 1, d.month)[1]))
