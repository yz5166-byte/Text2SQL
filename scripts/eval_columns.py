"""Column-level linking on the layer-3 questions: column recall and value-index hits. No model calls.

  uv run python scripts/eval_columns.py
  uv run python scripts/eval_columns.py --section X6-COLUMNS

Column recall@K is the share of the non-key columns a gold statement reads that are among the
top K non-key columns of the ranking over the whole schema, for the short and the detailed
column documents. Keys (主键, 关联) are left out on both sides: they carry the joins, which the
join bridge and the column pruning keep anyway, and a question rarely names them. A
value hit is correct when the gold statement compares one of the hit's columns with that value;
value recall counts the gold (column, value) pairs whose value the index holds. The last table
shows what the value hits add to table retrieval (hybrid top 5 plus the join bridge).
"""

from __future__ import annotations

import argparse

from text2sql.config import BIRD_DB_PATH, RISK_DB_PATH, ROOT
from text2sql.eval.dataset import bird_tables, load_bird_financial, load_domain_eval
from text2sql.eval.distractors import distractor_db, with_distractors
from text2sql.eval.repeats import write_section
from text2sql.eval.retrieval import gold_columns, gold_tables, gold_values
from text2sql.knowledge.retriever import ColumnRetriever, TableRetriever, key_columns
from text2sql.knowledge.values import ValueIndex, normalize
from text2sql.semantic.registry import load_semantic_model

KS = (5, 10, 15, 20)
NAMES = {"bm25": "BM25", "vector": "向量", "hybrid": "混合"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distractors", action="store_true", help="BIRD's other dev tables mixed in (large schema)")
    ap.add_argument("--section", help="report section to write the tables into, such as X6-COLUMNS")
    args = ap.parse_args()

    model = load_semantic_model()
    bird = model.restricted_to(bird_tables())
    domain_db, bird_db = RISK_DB_PATH, BIRD_DB_PATH
    if args.distractors:
        model, bird, domain_db, bird_db = (with_distractors(model), with_distractors(bird),
                                           distractor_db(RISK_DB_PATH), distractor_db(BIRD_DB_PATH))
    suites = {
        "自建第三层": (model, domain_db, [(i.question, i.gold_sql) for i in load_domain_eval() if i.route == "L3"]),
        "BIRD": (bird, bird_db, [(f"{i.question} {i.evidence or ''}", i.gold_sql) for i in load_bird_financial()]),
    }

    recall = ["| 题目 | 字段文档 | 检索 | " + " | ".join(f"召回@{k}" for k in KS) + f" | 全部召回@{KS[-1]} |",
              "|---|---|---|" + "---|" * (len(KS) + 1)]
    values = ["| 题目 | 索引取值数 | 命中 | 其中模糊 | 命中正确率 | 金标取值对 | 可索引的 | 取值召回 |",
              "|---|---|---|---|---|---|---|---|"]
    tables = ["| 题目 | 候选表 | 表召回 | 所需表全部召回 | 平均候选表数 |", "|---|---|---|---|---|"]
    for name, (m, db, cases) in suites.items():
        keys = key_columns(m)
        gold = [gold_columns(sql, m) - keys for _, sql in cases]
        for detailed in (False, True):
            for method in ("bm25", "vector", "hybrid"):
                retriever = ColumnRetriever(m, method, detailed)
                ranks = [[c for c in retriever.rank(text) if c not in keys] for text, _ in cases]
                cells = [f"{_recall(gold, ranks, k):.1%}" for k in KS]
                complete = sum(g <= set(r[:KS[-1]]) for g, r in zip(gold, ranks))
                recall.append(f"| {name} | {'详细' if detailed else '简短'} | {NAMES[method]} | "
                              + " | ".join(cells) + f" | {complete}/{len(cases)} |")

        index = ValueIndex.build(m, db)
        hits = [index.match(text) for text, _ in cases]
        pairs = [gold_values(sql, m) for _, sql in cases]
        flat = [h for hs in hits for h in hs]
        correct = sum(_correct(h, p) for hs, p in zip(hits, pairs) for h in hs)
        indexable = [(c, v) for p in pairs for c, v in p if normalize(v) in index.entries]
        found = sum(any(c in h.columns and normalize(h.value) == normalize(v) for h in hs)
                    for hs, p in zip(hits, pairs) for c, v in p if normalize(v) in index.entries)
        values.append(f"| {name} | {len(index.entries)} | {len(flat)} | {sum(h.fuzzy for h in flat)} | "
                      f"{_share(correct, len(flat))} | {sum(map(len, pairs))} | {len(indexable)} | "
                      f"{_share(found, len(indexable))} |")

        retriever, need = TableRetriever(m), [gold_tables(sql, m.tables) for _, sql in cases]
        for label, use_hits in (("混合前 5 张加连接补全", False), ("加取值命中的表", True)):
            got = [set(retriever.retrieve(text, [t for h in hs for t in h.tables] if use_hits else []).all)
                   for (text, _), hs in zip(cases, hits)]
            share = sum(len(n & g) for n, g in zip(need, got)) / sum(map(len, need))
            complete = sum(n <= g for n, g in zip(need, got))
            tables.append(f"| {name} | {label} | {share:.1%} | {complete}/{len(cases)} | "
                          f"{sum(map(len, got)) / len(cases):.1f} |")

    report = "\n\n".join("\n".join(t) for t in (recall, values, tables))
    print(report)
    if args.section:
        write_section(ROOT / "docs" / "EXPERIMENTS.md", args.section, report)


def _recall(gold: list[set[str]], ranks: list[list[str]], k: int) -> float:
    return sum(len(g & set(r[:k])) for g, r in zip(gold, ranks)) / sum(map(len, gold))


def _correct(hit, pairs: set[tuple[str, str]]) -> bool:
    return any(c in hit.columns and normalize(v) == normalize(hit.value) for c, v in pairs)


def _share(n: int, d: int) -> str:
    return f"{n / d:.1%}" if d else "无"


if __name__ == "__main__":
    main()
