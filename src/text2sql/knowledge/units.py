"""Knowledge units for retrieval: one document per catalog table, in Chinese and English.

A table document holds the table's names, description, grain and notes, every column's name
and Chinese name, and BIRD's English column notes, so that Chinese questions and English
(BIRD) questions both find it. Measured on the layer-3 questions (hybrid, top 5, join
bridge), this beats one document per column and stays short: column descriptions and
dimension values were tried and diluted the table's distinctive words.
"""

from __future__ import annotations

import csv
from pathlib import Path

from text2sql.config import BIRD_DESC_DIR
from text2sql.semantic.registry import SemanticModel


def table_units(model: SemanticModel, bird_desc_dir: Path = BIRD_DESC_DIR) -> dict[str, str]:
    units = {}
    for t, table in model.tables.items():
        english = _bird_notes(bird_desc_dir / f"{t}.csv")
        columns = " ".join(f"{c} {col.name_zh} {english.get(c, '')}" for c, col in table.columns.items())
        units[t] = " ".join(p for p in (t, table.name_zh, table.description, table.grain, table.notes, columns) if p)
    return units


def _bird_notes(path: Path) -> dict[str, str]:
    """BIRD's English column name, description and value notes, by column (empty if the file is absent)."""
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8-sig", errors="replace") as f:  # a few BIRD files are not clean UTF-8
        for row in csv.DictReader(f):
            notes = (row.get("column_name"), row.get("column_description"), row.get("value_description"))
            out[row["original_column_name"].strip()] = " ".join(n for n in notes if n)
    return out
