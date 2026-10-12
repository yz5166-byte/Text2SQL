"""Large-schema evaluation: BIRD's other dev databases mixed in as distractor tables.

A distractor table carries structure only: its columns with BIRD's English names and notes,
primary and foreign keys written as 主键 and 关联 notes (so that the join bridge and the key
columns work on it as on the catalog's tables), and no description, dimensions or metrics.
Its database holds at most ROW_LIMIT rows; the distinct count of each text column in the full
table is kept, so the value index treats the column as it would on full data.

scripts/build_distractors.py builds the tables; with_distractors() adds them to a semantic
model, and distractor_db() names the database that holds a base database plus them.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import sqlite3
from pathlib import Path
from typing import IO

from text2sql.config import DISTRACTOR_CATALOG, DISTRACTOR_DB_PATH
from text2sql.semantic.registry import SemanticModel, Table

ROW_LIMIT = 1000
NOTE_FIELDS = ("original_column_name", "column_name", "column_description", "value_description")


def available() -> bool:
    return DISTRACTOR_CATALOG.exists() and DISTRACTOR_DB_PATH.exists()


def with_distractors(model: SemanticModel, catalog: Path = DISTRACTOR_CATALOG) -> SemanticModel:
    """The model plus the distractor tables."""
    extra = {k: Table(key=k, **v) for k, v in json.loads(catalog.read_text(encoding="utf-8"))["tables"].items()}
    if clash := set(extra) & set(model.tables):
        raise ValueError(f"distractor tables clash with the model: {sorted(clash)}")
    return model.model_copy(update={"tables": {**model.tables, **extra}})


def distractor_db(base: Path) -> Path:
    """The database holding `base` and the distractor tables (financial.sqlite -> financial_distractors.sqlite)."""
    return base.with_name(f"{base.stem}_distractors.sqlite")


def read_notes(f: IO[bytes]) -> dict[str, tuple[str, str, str]]:
    """BIRD's (column name, description, value notes) by lower-cased column, from a description CSV."""
    text = io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace")  # a few BIRD files are not clean UTF-8
    return {row["original_column_name"].strip().lower(): tuple(_clean(row.get(k)) for k in NOTE_FIELDS[1:])
            for row in csv.DictReader(text) if row.get("original_column_name")}


def copy_database(path: Path, name: str, notes: dict[str, dict[str, tuple[str, str, str]]], out: sqlite3.Connection,
                  desc_dir: Path, taken: set[str]) -> tuple[dict[str, dict], dict[str, str]]:
    """Copy every table of database `name` into `out` with at most ROW_LIMIT rows.

    `notes` holds BIRD's column notes by lower-cased table; `taken` the lower-cased table names
    already in use, to which this database's tables are added. A taken name becomes
    <name>_<table>. Returns the catalog entries and each table's source ("database.table").
    """
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        renamed = {}
        for t in tables:
            renamed[t.lower()] = f"{name}_{t}" if t.lower() in taken else t
            taken.add(renamed[t.lower()].lower())
        entries, sources = {}, {}
        for t in tables:
            new = renamed[t.lower()]
            info = src.execute(f'PRAGMA table_info("{t}")').fetchall()  # cid, name, type, notnull, default, pk
            columns = {}
            for _, c, declared, _, _, pk in info:
                column_name, description, values = notes.get(t.lower(), {}).get(c.lower(), ("", "", ""))
                keys = ["主键"] if pk else []
                keys += [f"关联 {target}" for target in _references(src, t, c, renamed)]
                entry = {"type": column_type(declared), "name_zh": column_name or c,
                         "description": "; ".join(p for p in (description, values, *keys) if p)}
                if entry["type"] == "TEXT":
                    entry["distinct"] = src.execute(f'SELECT COUNT(DISTINCT {_quote(c)}) FROM "{t}"').fetchone()[0]
                columns[c] = entry
            out.execute(f'CREATE TABLE "{new}" ({", ".join(_quote(c) + " " + (d or "") for _, c, d, *_ in info)})')
            rows = src.execute(f'SELECT * FROM "{t}" LIMIT {ROW_LIMIT}').fetchall()
            out.executemany(f'INSERT INTO "{new}" VALUES ({", ".join("?" * len(info))})', rows)
            _write_notes(desc_dir / f"{new}.csv", notes.get(t.lower(), {}), [c for _, c, *_ in info])
            entries[new] = {"name_zh": new, "layer": "BIRD", "grain": "", "description": "", "columns": columns}
            sources[new] = f"{name}.{t}"
        return entries, sources
    finally:
        src.close()


def merge(base: Path, dest: Path, tables: list[str], distractors: Path = DISTRACTOR_DB_PATH) -> None:
    """Write a copy of `base` with the distractor `tables` added to `dest`."""
    shutil.copyfile(base, dest)
    con = sqlite3.connect(dest)
    try:
        con.execute("ATTACH DATABASE ? AS d", (str(distractors),))
        for t in tables:
            con.execute(con.execute("SELECT sql FROM d.sqlite_master WHERE type = 'table' AND name = ?", (t,)).fetchone()[0])
            con.execute(f'INSERT INTO main."{t}" SELECT * FROM d."{t}"')
        con.commit()
        con.execute("DETACH DATABASE d")
    finally:
        con.close()


def column_type(declared: str) -> str:
    """The catalog's type for a declared SQLite type, after SQLite's affinity rules."""
    d = (declared or "").upper()
    if "INT" in d:
        return "INTEGER"
    if any(k in d for k in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if any(k in d for k in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return "REAL"
    if "DATE" in d or "TIME" in d:
        return "DATE"
    return "TEXT"  # no declared type


def _references(src: sqlite3.Connection, table: str, column: str, renamed: dict[str, str]) -> list[str]:
    """"table.column" each foreign key on `column` points to, under the distractor names."""
    out = []
    for _, _, target, frm, to, *_ in src.execute(f'PRAGMA foreign_key_list("{table}")'):
        if frm != column or target.lower() not in renamed:
            continue
        if to is None:  # the target's primary key
            to = next((r[1] for r in src.execute(f'PRAGMA table_info("{target}")') if r[5]), None)
        if to:
            out.append(f"{renamed[target.lower()]}.{to}")
    return out


def _write_notes(path: Path, notes: dict[str, tuple[str, str, str]], columns: list[str]) -> None:
    """BIRD's notes without the column name, which the catalog already holds as the column's name_zh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(NOTE_FIELDS)
        for c in columns:
            _, description, values = notes.get(c.lower(), ("", "", ""))
            writer.writerow((c, "", description, values))


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())
