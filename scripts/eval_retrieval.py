"""Table retrieval on the layer-3 questions: BM25, embeddings and RRF, with and without the join bridge.

  uv run python scripts/eval_retrieval.py                              # domain layer-3 questions and BIRD
  uv run python scripts/eval_retrieval.py --dataset bird --bridged-only --section X4-RECALL
"""

from __future__ import annotations

import argparse

from text2sql.config import ROOT
from text2sql.eval.dataset import bird_tables, load_bird_financial, load_domain_eval
from text2sql.eval.repeats import write_section
from text2sql.eval.retrieval import evaluate_retrieval, gold_tables
from text2sql.knowledge.retriever import TableRetriever
from text2sql.semantic.registry import load_semantic_model

NAMES = {"bm25": "BM25", "vector": "向量", "hybrid": "混合"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dataset", choices=["both", "domain", "bird"], default="both",
                    help="bird uses BIRD's own 8 tables, as in experiments X3 and X4")
    ap.add_argument("--bridged-only", action="store_true", help="only with the join bridge, plus the full schema")
    ap.add_argument("--section", help="report section to write the table into, such as X4-RECALL")
    args = ap.parse_args()

    model = load_semantic_model()
    suites = {}
    if args.dataset in ("both", "domain"):
        suites["自建第三层"] = (model, [(i.question, i.gold_sql) for i in load_domain_eval() if i.route == "L3"])
    if args.dataset in ("both", "bird"):
        bird = model.restricted_to(bird_tables()) if args.dataset == "bird" else model
        suites["BIRD"] = (bird, [(f"{i.question} {i.evidence or ''}", i.gold_sql) for i in load_bird_financial()])

    lines = ["| 题目 | 表结构 | 表召回 | 所需表全部召回 | 平均交给模型的表数 |", "|---|---|---|---|---|"]
    for name, (m, cases) in suites.items():
        if args.bridged_only:  # the full schema recalls everything by construction
            covered = sum(gold_tables(sql, m.tables) <= set(m.tables) for _, sql in cases)
            lines.append(f"| {name} | 全量 | 100.0% | {covered}/{len(cases)} | {len(m.tables):.1f} |")
        for method in ("bm25", "vector", "hybrid"):
            for bridge in ((True,) if args.bridged_only else (False, True)):
                r = evaluate_retrieval(TableRetriever(m, method, args.k, bridge), cases)
                label = f"{NAMES[method]}前 {args.k} 张" + ("加连接补全" if bridge else "")
                lines.append(f"| {name} | {label} | {r.table_recall:.1%} | {r.complete}/{r.questions} | {r.mean_tables:.1f} |")
    table = "\n".join(lines)
    print(table)
    if args.section:
        write_section(ROOT / "docs" / "EXPERIMENTS.md", args.section, table)


if __name__ == "__main__":
    main()
