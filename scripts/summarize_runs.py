"""Summarise repeated evaluation runs, optionally writing the tables into docs/EXPERIMENTS.md.

  uv run python scripts/summarize_runs.py "x1_test_r*"                 # print
  uv run python scripts/summarize_runs.py "x1_test_r*" --section X1    # print and write into the report
"""

from __future__ import annotations

import argparse
from fnmatch import fnmatch

from text2sql.config import ROOT, RUNS_DIR
from text2sql.eval.repeats import load_runs, render, write_section


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="+", help="run names under runs/, shell-style wildcards allowed")
    ap.add_argument("--section", help="report section to write, such as X1")
    args = ap.parse_args()

    names = sorted(d.name for d in RUNS_DIR.iterdir() if any(fnmatch(d.name, p) for p in args.patterns))
    if not names:
        raise SystemExit(f"no run matches {args.patterns}")
    body = render(load_runs(names), names)
    print(body)
    if args.section:
        write_section(ROOT / "docs" / "EXPERIMENTS.md", args.section, body)


if __name__ == "__main__":
    main()
