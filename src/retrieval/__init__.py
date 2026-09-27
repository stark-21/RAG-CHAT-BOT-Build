"""Phase P4: query -> ranked, scored, scheme-filtered chunks."""

from src.retrieval.retriever import (
    RetrievalResult,
    dedupe,
    retrieve,
    resolve_scheme,
)

__all__ = ["RetrievalResult", "dedupe", "retrieve", "resolve_scheme"]
