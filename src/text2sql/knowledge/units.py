"""Knowledge units for retrieval: one document per catalog table, and one per column.

A table document holds the table's names, description, grain and notes, every column's name
and Chinese name, and BIRD's English column notes, so that Chinese questions and English
(BIRD) questions both find it. Measured on the layer-3 questions (hybrid, top 5, join
bridge), this beats one document per column and stays short: column descriptions and
dimension values were tried and diluted the table's distinctive words.

BIRD's notes for a distractor table (scripts/build_distractors.py) are read from its own
directory when the table is not one of financial's.

A column document comes in two forms. The short one names the column (table, column, Chinese
and English names); the detailed one adds the descriptions and the meaning of its codes.
Which ranks better is measured by scripts/eval_columns.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

from text2sql.config import BIRD_DESC_DIR, DISTRACTOR_DESC_DIR
from text2sql.semantic.registry import SemanticModel


def table_units(model: SemanticModel, bird_desc_dir: Path = BIRD_DESC_DIR) -> dict[str, str]:
    units = {}
    for t, table in model.tables.items():
        english = _bird_notes(_bird_csv(bird_desc_dir, t))
        columns = " ".join(f"{c} {col.name_zh} {english.get(c, '')}" for c, col in table.columns.items())
        units[t] = " ".join(p for p in (t, table.name_zh, table.description, table.grain, table.notes, columns) if p)
    return units


def column_units(model: SemanticModel, detailed: bool = False,
                 bird_desc_dir: Path = BIRD_DESC_DIR) -> dict[str, str]:
    """One document per column, keyed "table.column"."""
    codes = _value_labels(model) if detailed else {}
    units = {}
    for t, table in model.tables.items():
        bird = _bird_columns(_bird_csv(bird_desc_dir, t))
        for c, col in table.columns.items():
            name, description, values = bird.get(c, ("", "", ""))
            parts = [f"{t}.{c}", table.name_zh, col.name_zh, name]
            if detailed:
                parts += [col.description, description, values, codes.get((t, c), "")]
            units[f"{t}.{c}"] = " ".join(p for p in parts if p)
    return units


def _value_labels(model: SemanticModel) -> dict[tuple[str, str], str]:
    """Labels and codes of each dimension's values, by physical (table, column)."""
    out = {}
    for dim in model.dimensions.values():
        alias, column = dim.column.split(".")
        out[(model.physical_table(alias), column)] = " ".join(
            f"{v.label} {' '.join(str(c) for c in v.codes if c is not None)}" for v in dim.values[:20])
    return out


def _bird_csv(bird_desc_dir: Path, table: str) -> Path:
    path = bird_desc_dir / f"{table}.csv"
    return path if path.exists() else DISTRACTOR_DESC_DIR / f"{table}.csv"


def _bird_notes(path: Path) -> dict[str, str]:
    """BIRD's English column name, description and value notes in one string, by column."""
    return {c: " ".join(n for n in notes if n) for c, notes in _bird_columns(path).items()}


def _bird_columns(path: Path) -> dict[str, tuple[str, str, str]]:
    """BIRD's (column name, description, value notes) by column; empty if the file is absent."""
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8-sig", errors="replace") as f:  # a few BIRD files are not clean UTF-8
        for row in csv.DictReader(f):
            notes = (row.get("column_name"), row.get("column_description"), row.get("value_description"))
            out[row["original_column_name"].strip()] = tuple((n or "").strip() for n in notes)
    return out
