"""Evaluation entry point.

Examples
  uv run python scripts/run_eval.py --mode oracle                       # decoupled check of layers 1 and 2
  uv run python scripts/run_eval.py --split dev --provider deepseek     # end-to-end on the dev split
  uv run python scripts/run_eval.py --split test --provider claude --name x5_claude_test
  uv run python scripts/run_eval.py --split test --repeats 3 --name x1_test      # runs x1_test_r1 ... _r3
  uv run python scripts/run_eval.py --split test --mode naive --repeats 3        # X2: plain text-to-SQL baseline
  uv run python scripts/run_eval.py --dataset bird --mode all-l3 --variant no-repair --repeats 3   # X3 ablation
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from text2sql.config import BIRD_DB_PATH, RISK_DB_PATH, settings
from text2sql.db.executor import SQLiteExecutor
from text2sql.eval.dataset import DATA_ROUTES, bird_tables, load_bird_financial, load_domain_eval
from text2sql.eval.distractors import distractor_db, with_distractors
from text2sql.eval.runner import agent_predictor, generation_predictor, oracle_ir_predictor, run
from text2sql.llm import get_llm
from text2sql.pipeline.layer3 import NAIVE, GenerativeSQL, Layer3Options
from text2sql.pipeline.workflow import Text2SQLAgent
from text2sql.semantic.registry import load_semantic_model


VARIANTS = {  # layer-3 configurations for --mode all-l3 (experiment X3)
    "full": Layer3Options(),
    "no-examples": Layer3Options(examples=0),
    "no-knowledge": Layer3Options(knowledge=False),
    "no-repair": Layer3Options(max_repairs=0),
    "thinking": Layer3Options(),  # the same layer 3 on a client that thinks before writing SQL
    "full-schema": Layer3Options(schema="full"),  # X4: every table instead of the retrieved ones
    "bm25": Layer3Options(retrieval="bm25"),
    "vector": Layer3Options(retrieval="vector"),
    "column-linking": Layer3Options(column_linking=True),  # X6: value index and column ranking
    "table-selection": Layer3Options(table_selection=True),  # X6: the model filters the candidate tables
    "column-pruning": Layer3Options(column_pruning=True),  # X6: the model prunes the columns of the final tables
    # X6, cumulative: column-level linking, then table selection, then column pruning; pool of 5 or 20 tables
    "linking-selection": Layer3Options(column_linking=True, table_selection=True),
    "all-linking": Layer3Options(column_linking=True, table_selection=True, column_pruning=True),
    "all-linking-k20": Layer3Options(column_linking=True, table_selection=True, column_pruning=True, top_k=20),
    "full-k20": Layer3Options(top_k=20),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["domain", "bird"], default="domain")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev", help="domain set only")
    ap.add_argument("--provider", choices=["deepseek", "claude", "none"], default="deepseek")
    ap.add_argument("--model", help="model id for the provider, such as claude-sonnet-5-5 (default: from .env)")
    ap.add_argument("--mode", choices=["full", "oracle", "all-l3", "naive"], default="full",
                    help="full pipeline; gold IR; every data question to layer 3 (X2, X3); plain text-to-SQL (X2)")
    ap.add_argument("--variant", choices=list(VARIANTS), default="full", help="layer-3 configuration for all-l3")
    ap.add_argument("--distractors", action="store_true",
                    help="layer 3 on the large schema: BIRD's other dev tables mixed in (scripts/build_distractors.py)")
    ap.add_argument("--no-fast-path", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--repeats", type=int, default=1, help="independent runs, named <name>_r1, <name>_r2 ...")
    ap.add_argument("--limit", type=int, help="first N questions only, for a smoke run")
    ap.add_argument("--routes", help="comma-separated gold routes to keep, such as CLARIFY (default: all; "
                                     "data routes for all-l3 and naive)")
    args = ap.parse_args()

    no_model = args.provider == "none" or args.mode == "oracle"
    llm = None if no_model else get_llm(args.provider, think_for_generate=args.variant == "thinking", model=args.model)
    if args.dataset == "bird":  # straight to layer 3, BIRD's own 8 tables, on BIRD's own database
        if args.mode not in ("all-l3", "naive"):
            raise SystemExit("BIRD questions go straight to layer 3: use --mode all-l3 or naive")
        items, gold_db = load_bird_financial(), BIRD_DB_PATH
        model = load_semantic_model().restricted_to(bird_tables())
        executor = SQLiteExecutor(BIRD_DB_PATH, max_rows=settings.eval_max_rows)
    else:
        items, gold_db = load_domain_eval(), RISK_DB_PATH
        if args.split != "all":
            items = [i for i in items if i.split == args.split]
        agent = Text2SQLAgent(llm=llm, use_fast_path=not args.no_fast_path, max_rows=settings.eval_max_rows)
        model, executor = agent.model, agent.executor
    if args.distractors:
        if args.mode not in ("all-l3", "naive"):
            raise SystemExit("--distractors applies to layer 3: use --mode all-l3 or naive")
        model = with_distractors(model)
        executor = SQLiteExecutor(distractor_db(gold_db), max_rows=settings.eval_max_rows)
    if args.mode == "oracle":
        items = [i for i in items if i.route in ("L1", "L2")]
        predictor = oracle_ir_predictor(agent)
    elif args.mode in ("all-l3", "naive"):
        items = [i for i in items if i.route in (args.routes.split(",") if args.routes else DATA_ROUTES)]
        options = VARIANTS[args.variant] if args.mode == "all-l3" else NAIVE
        predictor = generation_predictor(GenerativeSQL(llm, model, executor, options))
    else:
        items = [i for i in items if not args.routes or i.route in args.routes.split(",")]
        predictor = agent_predictor(agent)
    items = items[: args.limit] if args.limit else items
    provider = "none" if llm is None else args.provider  # the oracle mode never calls a model
    split = "bird" if args.dataset == "bird" else args.split
    base = args.name or f"{args.mode}_{provider}_{split}_{datetime.now():%Y%m%d_%H%M%S}"
    meta = {"dataset": args.dataset, "split": split, "provider": provider, "mode": args.mode, "variant": args.variant,
            "fast_path": not args.no_fast_path, "distractors": args.distractors, "llm": getattr(llm, "name", None), "n": len(items)}
    for k in range(1, args.repeats + 1):
        name = base if args.repeats == 1 else f"{base}_r{k}"
        metrics = run(items, predictor, gold_db, name, meta)
        shown = {k: v for k, v in metrics.items() if k not in ("by_pred_route", "per_item", "per_item_strict")}
        print(name, json.dumps(shown, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()
