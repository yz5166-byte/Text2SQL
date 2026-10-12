"""Sentence embeddings from a local model through fastembed, chosen by EMBEDDING_MODEL.

bge-m3 (BAAI, MIT) is multilingual: Chinese and English text land in one space, so an English
question can be matched with Chinese table notes and Chinese examples. Its official ONNX export
(about 2.3 GB) is registered with fastembed as a custom model, so no PyTorch is needed; the
dense vector is the normalized [CLS] state. Its files are hard-linked side by side into one
directory, because ONNX Runtime refuses weights that the download cache's symlinks place in
another directory. Text is cut at MAX_TOKENS, as bge-small-zh does: bge-m3 would read 8,192
tokens, and a batch padded to the longest table note takes minutes on a CPU. bge-small-zh-v1.5 (about 90 MB) is the Chinese model
the project started with, kept for comparison (experiment X6).

The model is downloaded into .cache/fastembed on first use and loaded once per process. Vectors
are cached on disk by model and text (.cache/vectors), so a document is embedded once, not in
every evaluation process: bge-m3 takes minutes over the large schema's long table notes.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np

from text2sql.config import CACHE_DIR, settings

EMBEDDING_DIR = CACHE_DIR / "fastembed"
VECTOR_DIR = CACHE_DIR / "vectors"
MAX_TOKENS = 512
BATCH = 16
MODELS = {  # setting -> (fastembed model name, Hugging Face repo whose files are cached, multilingual)
    "bge-m3": ("BAAI/bge-m3", "BAAI/bge-m3", True),
    "bge-small-zh": ("BAAI/bge-small-zh-v1.5", "Qdrant/bge-small-zh-v1.5", False),
}
BGE_M3_DIR = EMBEDDING_DIR / "bge-m3-onnx"
BGE_M3_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                "onnx/model.onnx", "onnx/model.onnx_data", "onnx/Constant_7_attr__value"]


def model_name() -> str:
    return MODELS[settings.embedding_model][0]


def multilingual() -> bool:
    """Whether the model places English and Chinese text in one space."""
    return MODELS[settings.embedding_model][2]


def model_cached() -> bool:
    """Whether the model is already on disk (tests skip rather than download)."""
    if settings.embedding_model == "bge-m3":
        return all((BGE_M3_DIR / name).exists() for name in BGE_M3_FILES)
    repo = MODELS[settings.embedding_model][1]
    return any(EMBEDDING_DIR.glob(f"models--{repo.replace('/', '--')}"))


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding  # imported lazily: BM25-only use never loads ONNX
    from fastembed.common.model_description import ModelSource, PoolingType

    if settings.embedding_model != "bge-m3":
        return TextEmbedding(model_name(), cache_dir=str(EMBEDDING_DIR))
    if "BAAI/bge-m3" not in {m["model"] for m in TextEmbedding.list_supported_models()}:
        TextEmbedding.add_custom_model(
            model="BAAI/bge-m3", pooling=PoolingType.CLS, normalization=True, sources=ModelSource(hf="BAAI/bge-m3"),
            dim=1024, model_file="onnx/model.onnx", additional_files=BGE_M3_FILES[5:], license="mit", size_in_gb=2.29)
    model = TextEmbedding("BAAI/bge-m3", cache_dir=str(EMBEDDING_DIR), specific_model_path=str(_bge_m3_dir()))
    model.model.tokenizer.enable_truncation(max_length=MAX_TOKENS)
    return model


def _bge_m3_dir() -> Path:
    """The ONNX export with its files side by side, downloaded on first use (about 2.3 GB)."""
    if model_cached():
        return BGE_M3_DIR
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download("BAAI/bge-m3", allow_patterns=BGE_M3_FILES, cache_dir=str(EMBEDDING_DIR)))
    for name in BGE_M3_FILES:
        source, target = (snapshot / name).resolve(), BGE_M3_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(source, target)  # the same bytes, no second copy
            except OSError:
                shutil.copyfile(source, target)
    return BGE_M3_DIR


def embed(texts: list[str]) -> np.ndarray:
    """One unit-length row per text, so a dot product is the cosine similarity."""
    folder = VECTOR_DIR / settings.embedding_model
    paths = [folder / f"{hashlib.sha1(t.encode('utf-8')).hexdigest()}.npy" for t in texts]
    missing = [i for i, path in enumerate(paths) if not path.exists()]
    if missing:
        folder.mkdir(parents=True, exist_ok=True)
        for i, vector in zip(missing, _model().embed([texts[i] for i in missing], batch_size=BATCH)):
            np.save(paths[i], np.asarray(vector, dtype=np.float32))
    vectors = np.array([np.load(path) for path in paths])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
