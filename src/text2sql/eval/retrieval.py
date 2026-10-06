"""Retrieval evaluation: are the tables a gold SQL statement reads among the retrieved ones?

A question whose tables are not all retrieved cannot be answered correctly, so besides the
recall over tables the report counts the questions with complete coverage.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from text2sql.knowledge.retriever import TableRetriever


def gold_tables(sql: str, catalog_tables: Collection[str]) -> set[str]:
    """Catalog tables the statement reads; CTE names are not tables."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    read = {t.name.lower() for t in tree.find_all(exp.Table)} - ctes
    return read & {t.lower() for t in catalog_tables}


@dataclass
class RetrievalReport:
    questions: int
    table_recall: float  # share of all gold tables that were retrieved
    complete: int  # questions with every gold table retrieved
    mean_tables: float  # tables handed to the model per question


def coverage(retriever: TableRetriever, cases: dict[str, tuple[str, str]]) -> dict[str, bool]:
    """Per case id (query text, gold SQL): were all the tables the gold SQL reads retrieved?"""
    return {cid: gold_tables(sql, retriever.graph) <= set(retriever.retrieve(text).all)
            for cid, (text, sql) in cases.items()}


def evaluate_retrieval(retriever: TableRetriever, cases: list[tuple[str, str]]) -> RetrievalReport:
    """`cases` are (query text, gold SQL) pairs."""
    hit = total = complete = handed = 0
    for text, sql in cases:
        gold = gold_tables(sql, retriever.graph)
        got = set(retriever.retrieve(text).all)
        hit, total = hit + len(gold & got), total + len(gold)
        complete += gold <= got
        handed += len(got)
    return RetrievalReport(len(cases), hit / total, complete, handed / len(cases))
