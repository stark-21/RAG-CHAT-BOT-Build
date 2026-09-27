"""Project path constants. Nothing here touches the filesystem at import time."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
DOCS_DIR = PROJECT_ROOT / "docs"
LOGS_DIR = PROJECT_ROOT / "logs"

EMBED_CACHE_PATH = PROCESSED_DIR / "embeddings.sqlite"
ANSWER_CACHE_PATH = PROCESSED_DIR / "answer_cache.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

INGEST_LOG = LOGS_DIR / "ingest.jsonl"
INGEST_ERRORS = LOGS_DIR / "ingest_errors.json"
QUERY_LOG = LOGS_DIR / "queries.jsonl"

SOURCES_YAML = Path(__file__).resolve().parent / "ingest" / "sources.yaml"
META_SUFFIX = ".meta.json"

_ALL_DIRS = (DATA_DIR, RAW_DIR, PROCESSED_DIR, CHROMA_DIR, DOCS_DIR, LOGS_DIR)


def ensure_dirs() -> None:
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def scheme_raw_dir(scheme_id: str) -> Path:
    return RAW_DIR / scheme_id
