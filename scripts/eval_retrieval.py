"""Table retrieval on the layer-3 questions: BM25, embeddings and RRF, with and without the join bridge.

  uv run python scripts/eval_retrieval.py            # top 5 tables
  uv run python scripts/eval_retrieval.py --k 3
"""

from __future__ import annotations

import argparse

from text2sql.eval.dataset import load_bird_financial, load_domain_eval
from text2sql.eval.retrieval import evaluate_retrieval
from text2sql.knowledge.retriever import TableRetriever
from text2sql.semantic.registry import load_semantic_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    model = load_semantic_model()
    suites = {
        "自建第三层": [(i.question, i.gold_sql) for i in load_domain_eval() if i.route == "L3"],
        "BIRD": [(f"{i.question} {i.evidence or ''}", i.gold_sql) for i in load_bird_financial()],
    }
    print(f"{'题目':<8}{'方法':<8}{'连接补全':<8}{'表召回':>8}{'全部召回':>10}{'平均表数':>8}")
    for name, cases in suites.items():
        for method in ("bm25", "vector", "hybrid"):
            for bridge in (False, True):
                r = evaluate_retrieval(TableRetriever(model, method, args.k, bridge), cases)
                print(f"{name:<8}{method:<10}{'是' if bridge else '否':<10}{r.table_recall:>8.1%}"
                      f"{f'{r.complete}/{r.questions}':>10}{r.mean_tables:>10.1f}")


if __name__ == "__main__":
    main()
