"""Retrieval evaluation: are the tables and columns a gold SQL statement reads among the retrieved ones?

A question whose tables are not all retrieved cannot be answered correctly, so besides the
recall over tables the report counts the questions with complete coverage. Columns are
resolved to their tables through aliases and subqueries (sqlglot's qualify and scopes); the
literals a gold statement compares a column with are the values the value index should find.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from text2sql.knowledge.retriever import TableRetriever
from text2sql.semantic.registry import SemanticModel


def gold_tables(sql: str, catalog_tables: Collection[str]) -> set[str]:
    """Catalog tables the statement reads; CTE names are not tables."""
    tree = sqlglot.parse_one(sql, read="sqlite")
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    read = {t.name.lower() for t in tree.find_all(exp.Table)} - ctes
    return read & {t.lower() for t in catalog_tables}


def gold_columns(sql: str, model: SemanticModel) -> set[str]:
    """Catalog columns the statement reads, as "table.column"."""
    return {column for column, _ in _resolved(sql, model)}


def gold_values(sql: str, model: SemanticModel) -> set[tuple[str, str]]:
    """("table.column", literal) for every column compared with a string literal by =, IN or LIKE."""
    out = set()
    for column, node in _resolved(sql, model):
        parent = node.parent
        if isinstance(parent, (exp.EQ, exp.Like)):
            literals = [parent.left, parent.right]
        elif isinstance(parent, exp.In) and parent.this is node:
            literals = parent.expressions
        else:
            continue
        out |= {(column, lit.name) for lit in literals if isinstance(lit, exp.Literal) and lit.is_string}
    return out


def _resolved(sql: str, model: SemanticModel) -> list[tuple[str, exp.Column]]:
    """Each column reference with the catalog column it reads; references to CTEs and subqueries are skipped."""
    schema = {t: {c: col.type for c, col in table.columns.items()} for t, table in model.tables.items()}
    names = {(t.lower(), c.lower()): f"{t}.{c}" for t, table in model.tables.items() for c in table.columns}
    tree = qualify(sqlglot.parse_one(sql, read="sqlite"), schema=schema, dialect="sqlite",
                   validate_qualify_columns=False, quote_identifiers=False)
    out = []
    for scope in traverse_scope(tree):
        for column in scope.columns:
            source = scope.sources.get(column.table)
            if isinstance(source, exp.Table) and (source.name.lower(), column.name.lower()) in names:
                out.append((names[(source.name.lower(), column.name.lower())], column))
    return out


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
