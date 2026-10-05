"""Sentence embeddings from a local model: BAAI/bge-small-zh-v1.5 through fastembed (about 90 MB).

The model is downloaded into .cache/fastembed on first use and loaded once per process.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from text2sql.config import CACHE_DIR

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIR = CACHE_DIR / "fastembed"


def model_cached() -> bool:
    """Whether the model is already on disk (tests skip rather than download)."""
    return any(EMBEDDING_DIR.glob("models--*bge-small-zh-v1.5"))


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding  # imported lazily: BM25-only use never loads ONNX

    return TextEmbedding(MODEL_NAME, cache_dir=str(EMBEDDING_DIR))


def embed(texts: list[str]) -> np.ndarray:
    """One unit-length row per text, so a dot product is the cosine similarity."""
    vectors = np.array(list(_model().embed(texts)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
