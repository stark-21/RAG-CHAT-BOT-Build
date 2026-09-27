"""Phase P4 - Retriever.

Turns a user question into a ranked list of scored chunks, or into an abstention when
nothing clears the similarity threshold. Every lookup goes through the cached
`get_model()` / `get_collection()` singletons so the pipeline never re-loads the
embedding model or re-opens the persistent client.

The abstain path is a first-class outcome, not an error: a facts-only assistant that
answers a question it has no evidence for is worse than one that declines. When
`passed` is False the caller (P7) must refuse rather than generate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.config import get_settings
from src.ingest.chunker import count_tokens
from src.ingest.embedder import embed_query
from src.ingest.store import METADATA_FIELDS, get_collection
from src.ingest.sources import resolve_scheme
from src.types import Chunk, RetrievedChunk

__all__ = ["RetrievalResult", "dedupe", "retrieve", "resolve_scheme"]

# Fetch extra candidates so that filtering and dedup can still fill `top_k`. A single
# verbose section can otherwise account for every slot we asked Chroma for and leave
# the answer with one citation when five were requested.
_OVERFETCH_FACTOR = 4
_MIN_OVERFETCH = 10


@dataclass
class RetrievalResult:
    hits: list[RetrievedChunk]
    threshold: float
    passed: bool
    resolved_scheme_id: str | None
    latency_ms: int
    query: str = ""
    filtered: bool = field(default=False)

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def citations(self) -> list[Chunk]:
        return [hit.chunk for hit in self.hits]

    def __bool__(self) -> bool:
        return self.passed


def _chunk_from_result(chunk_id: str, text: str, meta: dict) -> Chunk:
    """Rebuild a `Chunk` from a Chroma row.

    `token_count` is not stored as metadata (it is derivable, and keeping it out of the
    index avoids a second source of truth that can drift from the text), so it is
    recomputed from the same estimator the chunker used.
    """
    missing = [f for f in METADATA_FIELDS if f not in meta]
    if missing:
        raise ValueError(f"incomplete metadata for {chunk_id}: missing {missing}")
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        token_count=count_tokens(text),
        scheme_id=str(meta["scheme_id"]),
        scheme_name=str(meta["scheme_name"]),
        category=str(meta["category"]),
        doc_type=str(meta["doc_type"]),
        section=str(meta["section"]),
        source_url=str(meta["source_url"]),
        publisher=str(meta["publisher"]),
        retrieved_at=str(meta["retrieved_at"]),
        chunk_index=int(meta["chunk_index"]),
        splitter=str(meta["splitter"]),
        content_hash=str(meta["content_hash"]),
    )


def _evidence_key(hit: RetrievedChunk) -> tuple[str, str, str]:
    """Identity of the section a hit came from, for grouping evidence.

    Normalised because section titles reach us straight from the source HTML, and the
    same logical section can be spelled differently by different parts of a page. The
    real corpus contains all five "Exit load" sections twice, as `Exit Load` and
    `Exit load`. Comparing raw strings let that one section occupy four of five slots
    and defeat the very cap `dedupe()` exists to enforce.
    """
    def norm(value: str) -> str:
        return " ".join((value or "").lower().split())

    return (norm(hit.chunk.scheme_id), norm(hit.chunk.doc_type), norm(hit.chunk.section))


def dedupe(hits: list[RetrievedChunk], per_section: int = 2) -> list[RetrievedChunk]:
    """Cap chunks per `(scheme_id, doc_type, section)` so one verbose section cannot
    crowd out every other piece of evidence, then restore score order.

    Ordering is by score descending, with `chunk_id` as a tie-break so the result is
    deterministic when two chunks share a score.
    """
    if per_section < 1:
        raise ValueError("per_section must be >= 1")
    kept: list[RetrievedChunk] = []
    seen: dict[tuple[str, str, str], int] = {}
    for hit in sorted(hits, key=lambda h: (-h.score, h.chunk.chunk_id)):
        key = _evidence_key(hit)
        used = seen.get(key, 0)
        if used >= per_section:
            continue
        seen[key] = used + 1
        kept.append(hit)
    return kept


def retrieve(
    query: str,
    top_k: int | None = None,
    min_similarity: float | None = None,
    scheme_id: str | None = None,
    *,
    per_section: int = 2,
) -> RetrievalResult:
    """Search the index and return ranked hits plus an abstention verdict.

    `scheme_id` overrides alias detection, which is what P7 wants once P6's
    `resolve_scheme()` has already run as a guardrail.

    A `where` filter is applied only when exactly one scheme is resolved. An ambiguous
    or absent scheme deliberately searches the whole corpus, so `GEN` reference material
    stays reachable for questions that name no scheme.
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.top_k
    threshold = min_similarity if min_similarity is not None else settings.min_similarity
    text = (query or "").strip()
    started = time.perf_counter()

    if not text:
        return RetrievalResult(
            hits=[],
            threshold=threshold,
            passed=False,
            resolved_scheme_id=scheme_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            query=query,
        )

    resolved = scheme_id or resolve_scheme(text)
    n_results = max(k * _OVERFETCH_FACTOR, k + _MIN_OVERFETCH)
    where = {"scheme_id": resolved} if resolved else None

    raw = get_collection().query(
        query_embeddings=[embed_query(text)],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = (raw.get("ids") or [[]])[0]
    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    hits: list[RetrievedChunk] = []
    for chunk_id, doc, meta, distance in zip(ids, documents, metadatas, distances):
        # `hnsw:space` is cosine, so chroma returns cosine *distance* and similarity is
        # its complement. Clamping guards against a hair outside [0, 1] from fp drift.
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        hits.append(
            RetrievedChunk(chunk=_chunk_from_result(chunk_id, doc, dict(meta)), score=score)
        )

    hits = dedupe(hits, per_section=per_section)[:k]
    passed = any(hit.score >= threshold for hit in hits)

    return RetrievalResult(
        hits=hits,
        threshold=threshold,
        passed=passed,
        resolved_scheme_id=resolved,
        latency_ms=int((time.perf_counter() - started) * 1000),
        query=query,
        filtered=resolved is not None,
    )
