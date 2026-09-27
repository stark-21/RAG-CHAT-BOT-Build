"""CPU sentence embeddings with a content-hash SQLite cache.

One `SentenceTransformer` instance is shared by ingest and query (architecture 6.4). The
model is ~90 MB, so it is loaded lazily through `lru_cache` and never at import time.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from src import paths
from src.config import get_settings, project_path

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from sentence_transformers import SentenceTransformer

EMBED_DIM = 384
BATCH_SIZE = 32
_MODEL_ID = ""


def _model_id() -> str:
    global _MODEL_ID
    if not _MODEL_ID:
        _MODEL_ID = get_settings().embedding_model
    return _MODEL_ID


def _configure_hf_home() -> None:
    """Point HuggingFace at the project cache before the library is imported."""
    settings = get_settings()
    target = project_path(settings.hf_home)
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(target))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache(maxsize=1)
def get_model() -> "SentenceTransformer":
    """Return the shared, CPU-pinned SentenceTransformer instance."""
    _configure_hf_home()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_model_id(), device="cpu")


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_path():
    return paths.EMBED_CACHE_PATH


def _connect() -> sqlite3.Connection:
    target = cache_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            hash      TEXT PRIMARY KEY,
            text_hash TEXT,
            vector    BLOB,
            model     TEXT
        )
        """
    )
    return connection


def _lookup(keys: list[str], model: str) -> dict[str, np.ndarray]:
    if not keys:
        return {}
    placeholders = ",".join("?" * len(keys))
    with _connect() as connection:
        rows = connection.execute(
            f"SELECT hash, vector FROM embeddings WHERE model = ? AND hash IN ({placeholders})",
            (model, *keys),
        ).fetchall()
    found: dict[str, np.ndarray] = {}
    for key, blob in rows:
        if blob is None:
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.size == EMBED_DIM:
            found[key] = vector
    return found


def _store(model: str, vectors: dict[str, list[float]], texts: dict[str, str]) -> None:
    if not vectors:
        return
    rows = [
        (key, _text_hash(texts[key]), np.asarray(vector, dtype=np.float32).tobytes(), model)
        for key, vector in vectors.items()
    ]
    with _connect() as connection:
        connection.executemany(
            "INSERT OR REPLACE INTO embeddings (hash, text_hash, vector, model) VALUES (?, ?, ?, ?)",
            rows,
        )


def _check_dim(vectors: list[list[float]]) -> None:
    for vector in vectors:
        if len(vector) != EMBED_DIM:
            raise ValueError(
                f"expected {EMBED_DIM}-dim embeddings, got {len(vector)}. "
                f"Check that {_model_id()} is the configured EMBEDDING_MODEL."
            )


def embed_documents(texts: list[str], batch_size: int = BATCH_SIZE) -> list[list[float]]:
    """Embed `texts`, reusing the SQLite cache for anything already seen.

    Duplicate texts are encoded once. Vectors are returned in the input order.
    """
    if not texts:
        return []

    model_id = _model_id()
    ordered_keys: list[str] = []
    key_to_text: dict[str, str] = {}
    for text in texts:
        key = _cache_key(text)
        key_to_text.setdefault(key, text)
        if key not in ordered_keys:
            ordered_keys.append(key)

    cached = _lookup(ordered_keys, model_id)
    missing = [key for key in ordered_keys if key not in cached]
    if missing:
        model = get_model()
        fresh = model.encode(
            [key_to_text[key] for key in missing],
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors = {key: [float(value) for value in row] for key, row in zip(missing, fresh)}
        _check_dim(list(vectors.values()))
        _store(model_id, vectors, {key: key_to_text[key] for key in missing})
        cached.update(vectors)

    return [[float(value) for value in cached[_cache_key(text)]] for text in texts]


def embed_query(text: str) -> list[float]:
    """Embed a single query string with the same model and normalisation."""
    return embed_documents([text])[0]


def cache_stats() -> dict:
    target = cache_path()
    if not target.exists():
        return {"path": str(target), "rows": 0, "model": _model_id()}
    with _connect() as connection:
        rows = connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE model = ?", (_model_id(),)
        ).fetchone()[0]
    return {"path": str(target), "rows": int(rows), "model": _model_id()}


def clear_cache() -> None:
    target = cache_path()
    if target.exists():
        target.unlink()
