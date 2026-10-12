"""Repeated evaluation runs: mean and range of each metric, and how stable each item is across runs.

The Markdown it renders goes into docs/EXPERIMENTS.md between section markers, so no number in
the report is copied by hand.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

from text2sql.config import RUNS_DIR

# (label, path into metrics.json, unit): "%" for rates, "s" for seconds, "$" for US dollars
ROWS = [
    ("执行准确率 EX", ("ex",), "%"),
    ("第一层 EX", ("ex_by_route", "L1"), "%"),
    ("第二层 EX", ("ex_by_route", "L2"), "%"),
    ("第三层 EX", ("ex_by_route", "L3"), "%"),
    ("BIRD 严格口径 EX", ("ex_strict",), "%"),
    ("可执行率", ("executable_rate",), "%"),
    ("路由准确率", ("route_accuracy",), "%"),
    ("IR 完全一致", ("ir_exact_match",), "%"),
    ("澄清精确率", ("clarify_precision",), "%"),
    ("澄清召回率", ("clarify_recall",), "%"),
    ("延迟 P50", ("efficiency", "latency_p50"), "s"),
    ("延迟 P95", ("efficiency", "latency_p95"), "s"),
    ("每题平均调用次数", ("efficiency", "llm_calls_mean"), ""),
    ("每题费用（高峰价上限）", ("efficiency", "cost_usd_mean"), "$"),
    ("每轮总费用（高峰价上限）", ("efficiency", "cost_usd_total"), "$"),
]
GROUP_NAMES = {"L1": "第一层题", "L2": "第二层题", "L3": "第三层题",
               "simple": "简单题", "moderate": "中等题", "challenging": "困难题",
               "recalled": "所需表全部召回的题", "missed": "检索漏表的题"}
SOURCES = {"rule": "规则", "fast_path": "快速通道", "llm": "模型解析", "none": "未判定（路由失败或解耦评估）"}


def load_runs(names: list[str], runs_dir: Path = RUNS_DIR) -> list[dict]:
    return [json.loads((runs_dir / n / "metrics.json").read_text(encoding="utf-8")) for n in names]


def _get(metrics: dict, path: tuple[str, ...]):
    for key in path:
        metrics = metrics.get(key) if isinstance(metrics, dict) else None
    return metrics


def _fmt(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value * 100:.1f}%"
    if unit == "s":
        return f"{value:.2f} 秒"
    if unit == "$":
        return f"{value:.4f}"
    return f"{value:.2f}"


def metric_table(runs: list[dict]) -> str:
    lines = ["| 指标 | 均值 | 最小 | 最大 |", "|---|---|---|---|"]
    for label, path, unit in ROWS:
        values = [v for v in (_get(r, path) for r in runs) if v is not None]
        if values:
            lines.append(f"| {label} | {_fmt(statistics.mean(values), unit)} | {_fmt(min(values), unit)} "
                         f"| {_fmt(max(values), unit)} |")
    return "\n".join(lines)


def traffic_table(runs: list[dict]) -> str:
    """Mean share of questions per deciding source and per predicted route."""
    n = runs[0]["n_items"]
    sources = Counter()
    routes = Counter()
    for r in runs:
        sources.update(r["source_share"])
        routes.update({route: v["n"] for route, v in r["by_pred_route"].items()})
    lines = ["| 维度 | 取值 | 平均题数 | 占比 |", "|---|---|---|---|"]
    for key, count in sources.most_common():
        lines.append(f"| 判定来源 | {SOURCES.get(key, key)} | {count / len(runs):.1f} | {count / len(runs) / n:.1%} |")
    for key, count in sorted(routes.items()):
        lines.append(f"| 最终路由 | {key} | {count / len(runs):.1f} | {count / len(runs) / n:.1%} |")
    return "\n".join(lines)


def stability(runs: list[dict]) -> dict[str, list[str]]:
    """Items correct in every run, in none, and in some (unstable)."""
    out: dict[str, list[str]] = {"always": [], "never": [], "sometimes": []}
    for item in runs[0]["per_item"]:
        hits = sum(r["per_item"][item] for r in runs)
        out["always" if hits == len(runs) else "never" if hits == 0 else "sometimes"].append(item)
    return out


def stability_line(runs: list[dict]) -> str:
    s = stability(runs)
    detail = "；".join(f"{label} {', '.join(s[key])}" for key, label in (("never", "每轮都错："), ("sometimes", "时对时错："))
                      if s[key])
    return (f"{len(runs)} 轮中每轮都对 {len(s['always'])} 题，每轮都错 {len(s['never'])} 题，"
            f"时对时错 {len(s['sometimes'])} 题。" + (f"\n\n{detail}" if detail else ""))


def render(runs: list[dict], names: list[str]) -> str:
    meta = runs[0]["meta"]
    head = (f"运行：{', '.join(names)}；模型 {meta.get('llm')}；数据 {meta.get('split')}，"
            f"{runs[0]['n_items']} 题 × {len(runs)} 轮。")
    return "\n\n".join([head, metric_table(runs), traffic_table(runs), stability_line(runs)])


def _predictions(runs_dir: Path, name: str, ids) -> dict[str, dict]:
    with open(runs_dir / name / "predictions.jsonl", encoding="utf-8") as f:
        return {p["item_id"]: p for p in map(json.loads, f) if p["item_id"] in ids}


def comparison_table(groups: dict[str, list[str]], routes: dict[str, str], format_only: frozenset[str] = frozenset(),
                     key: str = "per_item", runs_dir: Path = RUNS_DIR) -> str:
    """One row per configuration over the same items (`routes`: item id -> group, a gold route or a difficulty).

    EX as mean and range over runs (`key`: per_item, or per_item_strict for BIRD's own rule); EX counting
    the listed format-only items as correct, when any are listed; EX per group; how many items got the
    same SQL and the same result in every run; latency, calls, cost. A single run has no range and no
    run-to-run agreement, and says so instead.
    """
    route_names = sorted(set(routes.values()), key=lambda r: list(GROUP_NAMES).index(r) if r in GROUP_NAMES else 99)
    head = ["方案", "执行准确率 EX", *(["格式差异视为正确"] if format_only else []),
            *[GROUP_NAMES.get(r, r) for r in route_names], "SQL 各轮一致", "结果各轮一致", "延迟 P50", "每题调用", "每题费用"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, names in groups.items():
        ex, adjusted, by_route, latency, calls, cost = [], [], {r: [] for r in route_names}, [], [], []
        runs = [_predictions(runs_dir, n, routes) for n in names]
        for name, preds in zip(names, runs):
            per_item = json.loads((runs_dir / name / "metrics.json").read_text(encoding="utf-8"))[key]
            ex.append(statistics.mean(per_item[i] for i in routes))
            adjusted.append(statistics.mean(per_item[i] or i in format_only for i in routes))
            for r in route_names:
                by_route[r].append(statistics.mean(per_item[i] for i, g in routes.items() if g == r))
            latency.append(statistics.median(p["latency_s"] for p in preds.values()))
            calls += [p["llm_calls"] for p in preds.values()]
            cost += [p["cost_usd"] for p in preds.values()]
        same_sql = sum(len({run[i]["sql"] for run in runs}) == 1 for i in routes)
        same_rows = sum(len({json.dumps(run[i]["rows"]) for run in runs}) == 1 for i in routes)
        single = len(names) == 1
        cells = [label, f"{ex[0]:.1%}（单轮）" if single else f"{statistics.mean(ex):.1%}（{min(ex):.1%} 至 {max(ex):.1%}）",
                 *([f"{statistics.mean(adjusted):.1%}"] if format_only else []),
                 *[f"{statistics.mean(by_route[r]):.1%}" for r in route_names],
                 *(["单轮不适用"] * 2 if single else [f"{same_sql}/{len(routes)}", f"{same_rows}/{len(routes)}"]),
                 f"{statistics.mean(latency):.2f} 秒", f"{statistics.mean(calls):.2f}", f"{statistics.mean(cost):.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def linking_table(groups: dict[str, list[str]], gold: dict[str, tuple[set[str], set[str]]],
                  columns_of: dict[str, list[str]], runs_dir: Path = RUNS_DIR) -> str:
    """What layer 3 handed to generation, against what the gold SQL needs (experiment X6).

    `gold` maps an item to (tables, non-key columns as "table.column") of its gold SQL; `columns_of`
    lists each table's columns, for a run that did not prune. Averaged over the group's runs.
    """
    lines = ["| 方案 | 所需表齐全 | 非键金标字段保留 | 平均表数 | 平均字段数 |", "|---|---|---|---|---|"]
    for label, names in groups.items():
        complete, kept, needed, tables, columns = [], 0, 0, [], []
        for name in names:
            preds = _predictions(runs_dir, name, gold)
            for iid, (need_tables, need_columns) in gold.items():
                p = preds[iid]
                ctx_tables = set(p.get("context_tables") or [])
                ctx_columns = set(p["context_columns"]) if p.get("context_columns") is not None else \
                    {c for t in ctx_tables for c in columns_of.get(t, [])}
                complete.append(need_tables <= ctx_tables)
                kept, needed = kept + len(need_columns & ctx_columns), needed + len(need_columns)
                tables.append(len(ctx_tables))
                columns.append(len(ctx_columns))
        lines.append(f"| {label} | {sum(complete) / len(complete):.1%} | {kept / needed:.1%} | "
                     f"{statistics.mean(tables):.1f} | {statistics.mean(columns):.1f} |")
    return "\n".join(lines)


def repair_table(groups: dict[str, list[str]], ids: set[str], key: str = "per_item",
                 runs_dir: Path = RUNS_DIR) -> str:
    """How the repair loop behaved: attempts per question, how often a repair was needed and paid off."""
    lines = ["| 方案 | 平均生成次数 | 需要修复的题次 | 修复后答对 | 改用全量表结构 |", "|---|---|---|---|---|"]
    for label, names in groups.items():
        attempts, repaired, recovered, widened = [], 0, 0, 0
        for name in names:
            correct = json.loads((runs_dir / name / "metrics.json").read_text(encoding="utf-8"))[key]
            for p in _predictions(runs_dir, name, ids).values():
                attempts.append(p["attempts"])
                if p["attempts"] > 1:
                    repaired += 1
                    recovered += correct[p["item_id"]]
                widened += p["attempts"] > 1 and any(s["name"] == "widen" for s in p["trace"])
        lines.append(f"| {label} | {statistics.mean(attempts):.2f} | {repaired}/{len(attempts)}（{repaired / len(attempts):.1%}） "
                     f"| {recovered} | {widened} |")
    return "\n".join(lines)


FAILURES = ["正确", "执行失败", "结果为空", "列数与金标不同", "列数相同，行数不同", "行列数相同，数值不同"]


def failure_table(groups: dict[str, list[str]], gold: dict[str, list], key: str = "per_item",
                  runs_dir: Path = RUNS_DIR) -> str:
    """Where answers go wrong, averaged over runs: not run, empty, other columns, other rows, other values."""
    lines = ["| 方案 | " + " | ".join(FAILURES) + " |", "|" + "---|" * (len(FAILURES) + 1)]
    for label, names in groups.items():
        counts = Counter()
        for name in names:
            correct = json.loads((runs_dir / name / "metrics.json").read_text(encoding="utf-8"))[key]
            for iid, p in _predictions(runs_dir, name, gold).items():
                rows, expected = p["rows"], gold[iid]
                counts[
                    "正确" if correct[iid] else "执行失败" if rows is None else "结果为空" if not rows
                    else "列数与金标不同" if expected and len(rows[0]) != len(expected[0])
                    else "列数相同，行数不同" if len(rows) != len(expected) else "行列数相同，数值不同"
                ] += 1
        lines.append(f"| {label} | " + " | ".join(f"{counts[c] / len(names):.1f}" for c in FAILURES) + " |")
    return "\n".join(lines)


def answered_table(groups: dict[str, list[str]], ids: set[str], runs_dir: Path = RUNS_DIR) -> str:
    """For questions that should be clarified: how often each configuration answered with data instead."""
    lines = ["| 方案 | 直接给出答案的比例 | 题数 × 轮数 |", "|---|---|---|"]
    for label, names in groups.items():
        outcomes = [p["rows"] is not None for n in names for p in _predictions(runs_dir, n, ids).values()]
        lines.append(f"| {label} | {statistics.mean(outcomes):.1%} | {len(ids)} × {len(names)} |")
    return "\n".join(lines)


def write_section(doc: Path, section: str, body: str) -> None:
    """Replace the text between <!-- SECTION:BEGIN --> and <!-- SECTION:END --> in `doc`."""
    text = doc.read_text(encoding="utf-8")
    pattern = re.compile(rf"(<!-- {section}:BEGIN -->\n).*?(\n<!-- {section}:END -->)", re.S)
    if not pattern.search(text):
        raise ValueError(f"markers for {section} not found in {doc}")
    doc.write_text(pattern.sub(lambda m: m.group(1) + body + m.group(2), text), encoding="utf-8")
