"""Deterministic input guardrails: PII rejection, intent classification, refusals.

These run before retrieval and before any LLM call, and they cost nothing. Nothing in
this package imports the embedder, the vector store, or an LLM provider.
"""

from src.guardrails.intent import (
    Intent,
    classify_intent,
    mentions_out_of_corpus,
)
from src.guardrails.pii import PATTERNS, detect_pii, redact
from src.guardrails.refusal import DISCLAIMER, EDUCATIONAL_LINKS, build_refusal

__all__ = [
    "DISCLAIMER",
    "EDUCATIONAL_LINKS",
    "Intent",
    "PATTERNS",
    "build_refusal",
    "classify_intent",
    "detect_pii",
    "mentions_out_of_corpus",
    "redact",
]
