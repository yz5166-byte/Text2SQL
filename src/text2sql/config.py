"""Central configuration: paths, dataset reference date, model and runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BIRD_DB_PATH = DATA_DIR / "financial.sqlite"  # pristine BIRD database, used for the BIRD benchmark
RISK_DB_PATH = DATA_DIR / "bank_risk.sqlite"  # BIRD tables plus the derived DWD / ADS risk layer
BIRD_DESC_DIR = DATA_DIR / "bird_financial_description"
# Distractor tables from BIRD's other dev databases (scripts/build_distractors.py), for a large-schema evaluation
DISTRACTOR_DB_PATH = DATA_DIR / "bird_distractors.sqlite"  # at most 1,000 rows per table
DISTRACTOR_CATALOG = DATA_DIR / "bird_distractors.json"
DISTRACTOR_DESC_DIR = DATA_DIR / "bird_distractor_description"
SEMANTIC_DIR = ROOT / "semantic"
EVAL_DIR = ROOT / "eval"
RUNS_DIR = ROOT / "runs"
CACHE_DIR = ROOT / ".cache"

# The dataset covers 1993-01-01 to 1998-12-31. "Today" is pinned so that relative
# time expressions ("上个月", "今年以来") resolve reproducibly.
REFERENCE_DATE = date(1998, 12, 31)
DATA_START_DATE = date(1993, 1, 1)
DATA_END_DATE = date(1998, 12, 31)


@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("LLM_PROVIDER", "deepseek")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
    sql_timeout_s: float = float(os.getenv("SQL_TIMEOUT_S", "10"))
    max_rows: int = int(os.getenv("MAX_ROWS", "1000"))  # rows kept for display
    eval_max_rows: int = int(os.getenv("EVAL_MAX_ROWS", "50000"))  # BIRD has gold results of 15,140 rows
    max_repair_attempts: int = int(os.getenv("MAX_REPAIR_ATTEMPTS", "3"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "bge-m3")  # or bge-small-zh (knowledge/embedding.py)


settings = Settings()
