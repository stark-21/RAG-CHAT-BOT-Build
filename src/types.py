"""Canonical data contracts shared by every phase (implementation.md section 4).

These names are the integration surface for all later phases; do not rename them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["answered", "refused", "insufficient_context", "pii_rejected", "error"]


@dataclass
class SourceSpec:
    scheme_id: str
    scheme_name: str
    category: str
    entry_url: str
    aliases: list[str]
    documents: list[dict]


@dataclass
class SourceDoc:
    doc_id: str
    scheme_id: str
    scheme_name: str
    category: str
    doc_type: str
    title: str
    source_url: str
    publisher: str
    retrieved_at: str
    content_hash: str
    raw_path: str
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    status: str = "ok"
    reason: str = ""
    stats: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    token_count: int
    scheme_id: str
    scheme_name: str
    category: str
    doc_type: str
    section: str
    source_url: str
    publisher: str
    retrieved_at: str
    chunk_index: int
    splitter: str
    content_hash: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class Citation:
    label: str
    url: str
    doc_type: str
    retrieved_at: str


@dataclass
class QueryTrace:
    query_hash: str
    guards: dict
    hits: list[dict]
    threshold: float
    threshold_passed: bool
    cited_doc_ids: list[str]
    latency_ms: dict
    error: str | None = None


@dataclass
class Answer:
    status: Status
    text: str
    citations: list[Citation]
    last_updated: str | None
    trace: QueryTrace
