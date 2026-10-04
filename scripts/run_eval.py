"""Evaluation entry point.

Examples
  uv run python scripts/run_eval.py --mode oracle                       # decoupled check of layers 1 and 2
  uv run python scripts/run_eval.py --split dev --provider deepseek     # end-to-end on the dev split
  uv run python scripts/run_eval.py --split test --provider claude --name x5_claude_test
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from text2sql.config import RISK_DB_PATH, settings
from text2sql.eval.dataset import load_domain_eval
from text2sql.eval.runner import agent_predictor, oracle_ir_predictor, run
from text2sql.llm import get_llm
from text2sql.pipeline.workflow import Text2SQLAgent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    ap.add_argument("--provider", choices=["deepseek", "claude", "none"], default="deepseek")
    ap.add_argument("--mode", choices=["full", "oracle"], default="full")
    ap.add_argument("--no-fast-path", action="store_true")
    ap.add_argument("--name")
    args = ap.parse_args()

    items = load_domain_eval()
    if args.split != "all":
        items = [i for i in items if i.split == args.split]
    llm = None if args.provider == "none" or args.mode == "oracle" else get_llm(args.provider)
    agent = Text2SQLAgent(llm=llm, use_fast_path=not args.no_fast_path, max_rows=settings.eval_max_rows)
    if args.mode == "oracle":
        items = [i for i in items if i.route in ("L1", "L2")]
        predictor = oracle_ir_predictor(agent)
    else:
        predictor = agent_predictor(agent)
    provider = "none" if llm is None else args.provider  # the oracle mode never calls a model
    name = args.name or f"{args.mode}_{provider}_{args.split}_{datetime.now():%Y%m%d_%H%M%S}"
    meta = {"split": args.split, "provider": provider, "mode": args.mode,
            "fast_path": not args.no_fast_path, "llm": getattr(llm, "name", None), "n": len(items)}
    metrics = run(items, predictor, RISK_DB_PATH, name, meta)
    print(json.dumps({k: v for k, v in metrics.items() if k != "by_pred_route"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
