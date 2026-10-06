"""Compare configurations over the same items, optionally writing the table into docs/EXPERIMENTS.md.

  uv run python scripts/compare_runs.py --group "分层链路=x1_test_r*" --group "朴素基线=x2_naive_r?" \
      --format-only eval/x2_format_only.yaml --section X2
  uv run python scripts/compare_runs.py --clarify --group "朴素基线=x2_naive_clarify_r*" --section X2-CLARIFY
  uv run python scripts/compare_runs.py --dataset bird --by-recall --group "Claude=x5_bird" --section X5-RECALL
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch

import yaml

from text2sql.config import BIRD_DB_PATH, RISK_DB_PATH, ROOT, RUNS_DIR
from text2sql.eval.dataset import DATA_ROUTES, bird_tables, load_bird_financial, load_domain_eval
from text2sql.eval.repeats import answered_table, comparison_table, failure_table, repair_table, write_section
from text2sql.eval.retrieval import coverage
from text2sql.eval.runner import gold_result_sets
from text2sql.knowledge.retriever import TableRetriever
from text2sql.semantic.registry import load_semantic_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, help='"label=run pattern", repeatable')
    ap.add_argument("--dataset", choices=["domain", "bird"], default="domain")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--repairs", action="store_true", help="repair-loop table instead of the comparison")
    ap.add_argument("--failures", action="store_true", help="where answers go wrong, per configuration")
    ap.add_argument("--section", help="report section to write, such as X2")
    ap.add_argument("--format-only", help="YAML of item ids whose differences are only in output form")
    ap.add_argument("--clarify", action="store_true", help="questions that should be clarified: rate answered")
    ap.add_argument("--by-recall", action="store_true",
                    help="BIRD only: group by whether the default retrieval finds every table the gold SQL reads")
    args = ap.parse_args()

    wanted = ("CLARIFY",) if args.clarify else DATA_ROUTES
    if args.dataset == "bird":  # grouped by difficulty, scored by BIRD's own rule
        items, key = load_bird_financial(), "per_item_strict"
        groups_of = {i.id: i.difficulty for i in items}
        if args.by_recall:  # the same retrieval as layer 3's default, on the same query text
            retriever = TableRetriever(load_semantic_model().restricted_to(bird_tables()))
            found = coverage(retriever, {i.id: (f"{i.question} {i.evidence or ''}", i.gold_sql) for i in items})
            groups_of = {iid: "recalled" if ok else "missed" for iid, ok in found.items()}
    else:
        items, key = [i for i in load_domain_eval() if i.route in wanted and args.split in ("all", i.split)], "per_item"
        groups_of = {i.id: i.route for i in items}
    runs = sorted(d.name for d in RUNS_DIR.iterdir())
    groups = {}
    for spec in args.group:
        label, pattern = spec.split("=", 1)
        groups[label] = [n for n in runs if fnmatch(n, pattern)]
        if not groups[label]:
            raise SystemExit(f"no run matches {pattern}")
    if args.clarify:
        table = answered_table(groups, set(groups_of))
    elif args.repairs:
        table = repair_table(groups, set(groups_of), key)
    elif args.failures:
        gold = gold_result_sets(items, BIRD_DB_PATH if args.dataset == "bird" else RISK_DB_PATH)
        table = failure_table(groups, {i.id: gold[i.id] for i in items}, key)
    else:
        listed = yaml.safe_load(open(args.format_only, encoding="utf-8"))["items"] if args.format_only else []
        table = comparison_table(groups, groups_of, frozenset(e["id"] for e in listed), key)
    print(table)
    if args.section:
        write_section(ROOT / "docs" / "EXPERIMENTS.md", args.section, table)


if __name__ == "__main__":
    main()
