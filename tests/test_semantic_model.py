"""Consistency checks between the semantic YAML files and the physical database."""

import sqlite3
from collections import Counter

import pytest

from text2sql.config import RISK_DB_PATH
from text2sql.semantic.registry import load_semantic_model

model = load_semantic_model()


@pytest.fixture(scope="module")
def con():
    if not RISK_DB_PATH.exists():
        pytest.skip("data/bank_risk.sqlite missing: run scripts/build_warehouse.py")
    c = sqlite3.connect(f"file:{RISK_DB_PATH}?mode=ro", uri=True)
    yield c
    c.close()


def test_no_broken_references():
    assert model.validate_references() == []


def test_catalog_matches_database(con):
    for name, table in model.tables.items():
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
        if not cols and table.layer == "ADS":
            continue  # materialized later from the semantic layer
        assert set(cols) == set(table.columns), f"{name}: catalog {set(table.columns)} vs db {set(cols)}"


def test_dimension_values_cover_database(con):
    for key, dim in model.dimensions.items():
        table = model.physical_table(dim.alias)
        col = dim.column.split(".", 1)[1]
        db_values = {r[0] for r in con.execute(f'SELECT DISTINCT "{col}" FROM "{table}"')}
        declared = {code for v in dim.values for code in v.codes}
        assert db_values <= declared, f"{key}: undeclared values {db_values - declared}"
        assert declared <= db_values, f"{key}: declared values absent in data {declared - db_values}"


def test_every_metric_reaches_region_and_time():
    for key in model.metrics:
        assert "region" in model.reachable_dimensions(key), key


def test_owner_path_is_many_to_one_chain():
    path = model.join_path("trans", "owner_client")
    assert [(e.from_, e.to) for e in path] == [
        ("trans", "account"), ("account", "owner_disp"), ("owner_disp", "owner_client")
    ]
    assert model.join_path("loan", "dwd_risk_event") is None  # never walk one-to-many


def test_metric_names_and_aliases_are_unique():
    terms = Counter()
    for metric in model.metrics.values():
        for term in {metric.name, *metric.aliases}:
            terms[term] += 1
    duplicated = [t for t, n in terms.items() if n > 1]
    assert duplicated == []
    assert not set(terms) & set(model.ambiguous_terms), "ambiguous terms must not be plain aliases"
