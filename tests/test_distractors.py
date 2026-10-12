"""Distractor tables for the large-schema evaluation: copying, renaming, keys, sampling and merging."""

import csv
import io
import json
import sqlite3

import pytest

from text2sql.eval import distractors
from text2sql.eval.distractors import column_type, copy_database, merge, read_notes, with_distractors
from text2sql.knowledge import units
from text2sql.knowledge.retriever import foreign_key_graph, key_columns
from text2sql.knowledge.values import ValueIndex
from text2sql.semantic.registry import load_semantic_model

NOTES = b"original_column_name,column_name,column_description,data_format,value_description\r\n" \
        b"id,loan id,the id of the loan,integer,\r\nkind,loan kind,kind of loan,text,\"A: auto\nB: boat\"\r\n"


@pytest.fixture
def toy(tmp_path):
    """A source database whose table loan collides with the catalog and references branch."""
    path = tmp_path / "toy.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE branch (id INTEGER PRIMARY KEY, city VARCHAR(20))")
    con.execute("CREATE TABLE loan (id INTEGER PRIMARY KEY, branch_id INTEGER REFERENCES branch, kind TEXT, note)")
    con.executemany("INSERT INTO branch VALUES (?, ?)", [(1, "Brno"), (2, "Praha")])
    con.executemany("INSERT INTO loan VALUES (?, ?, ?, ?)", [(i, 1 + i % 2, ("auto", "boat")[i % 2], f"n{i}")
                                                        for i in range(1500)])
    con.commit()
    con.close()
    out = sqlite3.connect(tmp_path / "distractors.sqlite")
    entries, sources = copy_database(path, "toy", {"loan": read_notes(io.BytesIO(NOTES))}, out,
                                     tmp_path / "desc", {"loan", "account"})
    out.commit()
    return tmp_path, out, entries, sources


def test_column_types_follow_sqlite_affinity():
    assert [column_type(t) for t in ("INTEGER", "varchar(20)", "DOUBLE", "DATETIME", "")] == \
           ["INTEGER", "TEXT", "REAL", "DATE", "TEXT"]


def test_a_taken_name_is_prefixed_and_keys_become_notes(toy):
    _, _, entries, sources = toy
    assert sources == {"branch": "toy.branch", "toy_loan": "toy.loan"}
    loan = entries["toy_loan"]["columns"]
    assert loan["id"] == {"type": "INTEGER", "name_zh": "loan id", "description": "the id of the loan; 主键"}
    assert loan["branch_id"]["description"] == "关联 branch.id"  # REFERENCES branch: its primary key
    assert loan["kind"]["description"] == "kind of loan; A: auto B: boat"
    assert loan["note"]["type"] == "TEXT" and loan["note"]["distinct"] == 1500  # counted on the full table


def test_rows_are_capped_and_notes_drop_the_column_name(toy):
    tmp_path, out, _, _ = toy
    assert out.execute('SELECT COUNT(*) FROM "toy_loan"').fetchone() == (1000,)
    rows = list(csv.reader(open(tmp_path / "desc" / "toy_loan.csv", encoding="utf-8")))
    assert rows[1] == ["id", "", "the id of the loan", ""]


def test_the_model_gains_tables_keys_and_joins(toy, monkeypatch):
    tmp_path, _, entries, sources = toy
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"tables": entries, "sources": sources}), encoding="utf-8")
    base = load_semantic_model()
    model = with_distractors(base, catalog)
    assert list(model.tables)[: len(base.tables)] == list(base.tables) and {"branch", "toy_loan"} <= set(model.tables)
    assert foreign_key_graph(model)["toy_loan"] == {"branch"}
    assert {"toy_loan.id", "toy_loan.branch_id", "branch.id"} <= key_columns(model)
    with pytest.raises(ValueError, match="clash"):
        with_distractors(model, catalog)
    monkeypatch.setattr(units, "DISTRACTOR_DESC_DIR", tmp_path / "desc")
    assert "kind of loan" in units.table_units(model)["toy_loan"]


def test_the_value_index_judges_a_sample_by_the_full_table(toy):
    tmp_path, out, entries, sources = toy
    catalog = tmp_path / "catalog.json"

    def indexed():
        catalog.write_text(json.dumps({"tables": entries, "sources": sources}), encoding="utf-8")
        model = with_distractors(load_semantic_model().restricted_to(set()), catalog)
        index = ValueIndex.build(model, tmp_path / "distractors.sqlite")
        return {c for _, cols in index.entries.values() for c in cols}

    assert indexed() == {"branch.city", "toy_loan.kind"}  # note: 1,500 distinct in full, 1,000 in the sample
    entries["toy_loan"]["columns"]["kind"]["distinct"] = 5000  # as if the full table had many kinds
    assert indexed() == {"branch.city"}


def test_merge_adds_the_distractor_tables_to_a_copy(toy, tmp_path):
    _, out, _, _ = toy
    out.close()
    base = tmp_path / "base.sqlite"
    sqlite3.connect(base).execute("CREATE TABLE account (account_id INTEGER)").connection.commit()
    merge(base, tmp_path / "merged.sqlite", ["branch", "toy_loan"], tmp_path / "distractors.sqlite")
    con = sqlite3.connect(tmp_path / "merged.sqlite")
    assert {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")} == \
           {"account", "branch", "toy_loan"}
    assert con.execute('SELECT COUNT(*) FROM "toy_loan"').fetchone() == (1000,)
    assert distractors.distractor_db(base).name == "base_distractors.sqlite"
