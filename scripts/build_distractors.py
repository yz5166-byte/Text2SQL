"""Build the distractor tables for the large-schema evaluation from BIRD's other dev databases.

  uv run python scripts/build_distractors.py               # download dev.zip if missing, then build
  uv run python scripts/build_distractors.py --merge-only  # only the merged databases (after build_warehouse.py)

Every table of the 10 dev databases other than financial is copied with at most 1,000 rows into
data/bird_distractors.sqlite, its BIRD column notes into data/bird_distractor_description/ and
its catalog entry into data/bird_distractors.json (see text2sql.eval.distractors). The merged
databases, financial and the bank risk warehouse each with the distractor tables, are what the
large-schema evaluation runs on. The archive and the extracted databases are deleted afterwards.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import zipfile

from download_data import DEV_ZIP_URL, _download

from text2sql.config import (BIRD_DB_PATH, DISTRACTOR_CATALOG, DISTRACTOR_DB_PATH, DISTRACTOR_DESC_DIR, RAW_DIR,
                             RISK_DB_PATH)
from text2sql.eval.distractors import ROW_LIMIT, copy_database, distractor_db, merge, read_notes
from text2sql.semantic.registry import load_semantic_model

SKIP = {"financial"}  # the benchmark's own database


def build() -> None:
    dev_zip, inner_zip = RAW_DIR / "dev.zip", RAW_DIR / "dev_databases.zip"
    if not dev_zip.exists():
        _download(DEV_ZIP_URL, dev_zip)
    with zipfile.ZipFile(dev_zip) as outer, open(inner_zip, "wb") as out:
        shutil.copyfileobj(outer.open("dev_20240627/dev_databases.zip"), out)
    DISTRACTOR_DB_PATH.unlink(missing_ok=True)
    shutil.rmtree(DISTRACTOR_DESC_DIR, ignore_errors=True)
    taken = {t.lower() for t in load_semantic_model().tables}
    catalog = {"source": "BIRD dev (dev_20240627)", "row_limit": ROW_LIMIT, "tables": {}, "sources": {}}
    out = sqlite3.connect(DISTRACTOR_DB_PATH)
    try:
        with zipfile.ZipFile(inner_zip) as inner:
            files = [n for n in inner.namelist() if n.startswith("dev_databases/") and n.endswith(".sqlite")]
            for member in sorted(files):
                name = member.split("/")[1]
                if name in SKIP:
                    continue
                prefix = f"dev_databases/{name}/database_description/"
                notes = {n[len(prefix):-4].lower(): read_notes(inner.open(n))
                         for n in inner.namelist() if n.startswith(prefix) and n.endswith(".csv")}
                db = RAW_DIR / f"{name}.sqlite"
                with inner.open(member) as src, open(db, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                tables, sources = copy_database(db, name, notes, out, DISTRACTOR_DESC_DIR, taken)
                out.commit()
                for leftover in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
                    leftover.unlink(missing_ok=True)  # a database in WAL mode leaves these after a read
                catalog["tables"] |= tables
                catalog["sources"] |= sources
                print(f"{name}: {len(tables)} tables")
    finally:
        out.close()
    DISTRACTOR_CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    inner_zip.unlink()
    dev_zip.unlink()
    print(f"ok: {len(catalog['tables'])} tables in {DISTRACTOR_DB_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge-only", action="store_true", help="rebuild only the merged databases")
    args = ap.parse_args()
    if not args.merge_only:
        build()
    tables = list(json.loads(DISTRACTOR_CATALOG.read_text(encoding="utf-8"))["tables"])
    for base in (BIRD_DB_PATH, RISK_DB_PATH):
        merge(base, distractor_db(base), tables)
        print(f"ok: {distractor_db(base)}")


if __name__ == "__main__":
    main()
