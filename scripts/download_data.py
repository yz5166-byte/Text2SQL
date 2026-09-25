"""Download the BIRD financial database and the cleaned BIRD dev questions.

Sources
1. BIRD dev set (official OSS mirror): contains financial.sqlite and column descriptions.
2. BIRD dev 2025-11-06 cleaned questions (Hugging Face, CC BY-SA 4.0).

Only the financial database and its description CSVs are kept; the rest of the
346 MB archive is deleted after extraction.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile

from text2sql.config import BIRD_DB_PATH, BIRD_DESC_DIR, RAW_DIR

DEV_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
CLEAN_DEV_URL = (
    "https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106/resolve/main/"
    "data/dev_20251106-00000-of-00001.json"
)
CLEAN_DEV_PATH = RAW_DIR / "bird_dev_20251106.json"
FINANCIAL_SHA256 = "d15d89cdb068a202b6f2b99342af44dffc1d52545b39ceaf62efdc0ba570101e"


def _download(url: str, dest) -> None:
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_financial_db() -> None:
    if BIRD_DB_PATH.exists() and _sha256(BIRD_DB_PATH) == FINANCIAL_SHA256:
        print(f"ok: {BIRD_DB_PATH}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dev_zip = RAW_DIR / "dev.zip"
    _download(DEV_ZIP_URL, dev_zip)
    inner_zip = RAW_DIR / "dev_databases.zip"
    with zipfile.ZipFile(dev_zip) as outer, open(inner_zip, "wb") as out:
        shutil.copyfileobj(outer.open("dev_20240627/dev_databases.zip"), out)
    prefix = "dev_databases/financial/"
    BIRD_DESC_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(inner_zip) as inner:
        with inner.open(prefix + "financial.sqlite") as src, open(BIRD_DB_PATH, "wb") as dst:
            shutil.copyfileobj(src, dst)
        for name in inner.namelist():
            if name.startswith(prefix + "database_description/") and name.endswith(".csv"):
                with inner.open(name) as src, open(BIRD_DESC_DIR / name.rsplit("/", 1)[-1], "wb") as dst:
                    shutil.copyfileobj(src, dst)
    dev_zip.unlink()
    inner_zip.unlink()
    digest = _sha256(BIRD_DB_PATH)
    if digest != FINANCIAL_SHA256:
        raise RuntimeError(f"financial.sqlite checksum mismatch: {digest}")
    print(f"ok: {BIRD_DB_PATH}")


def fetch_clean_dev_questions() -> None:
    if CLEAN_DEV_PATH.exists():
        print(f"ok: {CLEAN_DEV_PATH}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    _download(CLEAN_DEV_URL, CLEAN_DEV_PATH)
    print(f"ok: {CLEAN_DEV_PATH}")


if __name__ == "__main__":
    fetch_financial_db()
    fetch_clean_dev_questions()
