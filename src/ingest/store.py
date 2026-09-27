"""ChromaDB persistence for embedded chunks.

Two behaviours here are load-bearing and easy to get wrong:

* **Metadata must be scalar.** Chroma 1.5.9 accepts a list-valued metadata field without
  any error, but a later `where={"scheme_id": "S1"}` filter then matches *nothing* and
  returns an empty result. A scheme filter that silently finds no rows looks exactly like
  "no context in the sources", so P4 would report insufficient context instead of
  crashing. `flatten_metadata()` therefore rejects anything that is not a scalar, and
  stores only the eleven fields listed in architecture 6.5.
* **Vectors are L2-normalised at write time** (`normalize_embeddings=True` in the
  embedder), so the dot product equals cosine similarity. The collection is still created
  with `hnsw:space=cosine` so the distance reported by a query is a true cosine distance
  rather than an implicit L2 conversion.
* **`query()` only returns `distances` when it is asked for.** Leaving `distances` out of
  `include` gives `result["distances"] is None`, not an empty list, so a retriever that
  forgets it fails with a `TypeError` deep inside score handling. Pass
  `include=["documents", "metadatas", "distances"]`. With `hnsw:space=cosine` the returned
  value is a cosine *distance*, so similarity is `1 - distance`.
"""

from __future__ import annotations

import shutil
import sqlite3
from functools import lru_cache
from pathlib import Path

from src.config import get_settings, project_path
from src.ingest.embedder import EMBED_DIM, _model_id
from src.types import Chunk

# architecture.md 6.5 - exactly these fields, nothing else.
METADATA_FIELDS = (
    "scheme_id",
    "scheme_name",
    "category",
    "doc_type",
    "section",
    "source_url",
    "publisher",
    "retrieved_at",
    "chunk_index",
    "splitter",
    "content_hash",
)

_SCALARS = (str, int, float, bool)


class MetadataError(ValueError):
    """Raised when a metadata value cannot be stored in Chroma."""


def flatten_metadata(chunk: Chunk) -> dict:
    """Reduce a Chunk to the scalar-only metadata dict Chroma accepts."""
    source = {
        "scheme_id": chunk.scheme_id,
        "scheme_name": chunk.scheme_name,
        "category": chunk.category,
        "doc_type": chunk.doc_type,
        "section": chunk.section,
        "source_url": chunk.source_url,
        "publisher": chunk.publisher,
        "retrieved_at": chunk.retrieved_at,
        "chunk_index": chunk.chunk_index,
        "splitter": chunk.splitter,
        "content_hash": chunk.content_hash,
    }
    metadata: dict = {}
    for field in METADATA_FIELDS:
        value = source[field]
        if value is None:
            raise MetadataError(f"{chunk.chunk_id}: metadata field {field!r} is None")
        if isinstance(value, bool) or not isinstance(value, _SCALARS):
            raise MetadataError(
                f"{chunk.chunk_id}: metadata field {field!r} must be a scalar, "
                f"got {type(value).__name__}. Chroma would silently break `where` filters."
            )
        if isinstance(value, str) and not value.strip():
            raise MetadataError(f"{chunk.chunk_id}: metadata field {field!r} is empty")
        metadata[field] = value
    return metadata


@lru_cache(maxsize=1)
def get_client():
    import chromadb

    target = project_path(get_settings().chroma_dir)
    target.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(target))


def _collection_metadata() -> dict:
    return {
        "hnsw:space": "cosine",
        "embedding_model": _model_id(),
        "dim": EMBED_DIM,
    }


@lru_cache(maxsize=1)
def get_collection():
    """Return the `mf_faq` collection, creating it on first use."""
    client = get_client()
    name = get_settings().collection_name
    try:
        return client.get_collection(name=name)
    except Exception:
        # Not created yet (or deleted by reset_collection).
        return client.create_collection(name=name, metadata=_collection_metadata())


def upsert_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> int:
    """Write chunks and their vectors. Returns the number of records upserted."""
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError(f"got {len(chunks)} chunks but {len(vectors)} vectors")

    metadatas = [flatten_metadata(chunk) for chunk in chunks]
    for vector in vectors:
        if len(vector) != EMBED_DIM:
            raise ValueError(f"expected {EMBED_DIM}-dim vector, got {len(vector)}")

    collection = get_collection()
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        embeddings=[list(map(float, vector)) for vector in vectors],
        metadatas=metadatas,
    )
    return len(chunks)


def _live_segment_ids(chroma_dir) -> set[str]:
    """Segment ids the index still tracks. Empty set means "could not tell"."""
    database = Path(chroma_dir) / "chroma.sqlite3"
    if not database.exists():
        return set()
    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT id FROM segments").fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return set()
    return {str(row[0]) for row in rows}


def _segment_dirs(chroma_dir) -> set[str]:
    root = Path(chroma_dir)
    if not root.is_dir():
        return set()
    return {entry.name for entry in root.iterdir() if entry.is_dir()}


def _prune_orphan_segments(chroma_dir, candidates: set[str]) -> int:
    """Delete segment directories that pre-date the reset and are no longer referenced.

    `delete_collection()` drops the segment row but leaves the directory behind, so every
    `--rebuild` would otherwise strand ~170 KB. Only directories captured *before* the
    delete are considered, so a segment created by the recreate can never be removed.
    Failures are ignored: leaking a directory must never fail a build.
    """
    live = _live_segment_ids(chroma_dir)
    if not live:
        return 0

    removed = 0
    for name in candidates - live:
        target = Path(chroma_dir) / name
        try:
            shutil.rmtree(target)
            removed += 1
        except OSError:
            continue
    return removed


def reset_collection() -> None:
    """Delete and recreate the collection so stale chunks never mix with new ones.

    `get_or_create_collection` ignores a changed `hnsw:space`, so an existing collection
    keeps whatever distance metric it was built with. Deleting is the only reliable reset.
    """
    settings = get_settings()
    chroma_dir = project_path(settings.chroma_dir)
    name = settings.collection_name
    stale = _segment_dirs(chroma_dir)

    try:
        get_client().delete_collection(name=name)
    except Exception:
        pass
    get_collection.cache_clear()
    get_collection()

    removed = _prune_orphan_segments(chroma_dir, stale)
    if removed:
        print(f"pruned {removed} orphaned vector segment(s) from {settings.chroma_dir}")


def search(
    query: str,
    n_results: int = 5,
    where: dict | None = None,
) -> dict:
    """Embed `query` with our own model and run the vector search.

    Phase P4 must use this rather than `collection.query(query_texts=...)`. Chroma's
    text path lazily downloads its own 79 MB ONNX copy of the model on first use (measured:
    175 s) and bypasses the SQLite embedding cache. The two embedders were measured to be
    numerically identical (cosine 1.0000 on identical input), so the results would still be
    correct, but the download and the cache bypass are pure waste.

    Returns ids, `distances` (cosine distance, so similarity is `1 - distance`),
    `metadatas` and `documents`.
    """
    from src.ingest.embedder import embed_query

    vector = embed_query(query)
    kwargs = {"where": where} if where else {}
    return get_collection().query(
        query_embeddings=[vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
        **kwargs,
    )


def collection_stats() -> dict:
    """Summarise the stored index for the CLI and for tests."""
    collection = get_collection()
    stats = {"name": collection.name, "count": collection.count()}
    stats.update({f"meta_{key}": value for key, value in (collection.metadata or {}).items()})
    try:
        peek = collection.peek(limit=1)
        embeddings = peek.get("embeddings")
        if embeddings is not None and len(embeddings):
            stats["peek_dim"] = len(embeddings[0])
    except Exception as exc:  # pragma: no cover - diagnostics only
        stats["peek_error"] = f"{type(exc).__name__}: {exc}"
    return stats


def reset_caches() -> None:
    """Test helper: forget the memoised client and collection."""
    get_collection.cache_clear()
    get_client.cache_clear()
