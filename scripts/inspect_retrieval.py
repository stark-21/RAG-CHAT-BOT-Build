"""Inspect what retrieval actually returns, without generating an answer.

    python -m scripts.inspect_retrieval "what is the exit load"
    python -m scripts.inspect_retrieval --scheme S3 "lock in period"
    python -m scripts.inspect_retrieval --min-similarity 0.5 --top-k 3 "expense ratio"
    python -m scripts.inspect_retrieval --interactive

Shows, per query: how the scheme was resolved, whether a filter was applied, the similarity
gate, and every hit with its score, chunk id, document type, and section. That is everything
retrieval decides - the answer text is a separate stage, so this isolates the part you are
testing.
"""

from __future__ import annotations

import argparse
import sys

from src.config import get_settings
from src.guardrails.intent import resolve_scheme
from src.retrieval.retriever import retrieve

PREVIEW_CHARS = 150


def show(query: str, *, top_k: int | None, min_similarity: float | None, scheme_id: str | None) -> None:
    result = retrieve(query, top_k=top_k, min_similarity=min_similarity, scheme_id=scheme_id)

    resolved = result.resolved_scheme_id or "-"
    print(f"\nquery            : {query!r}")
    print(f"resolved scheme  : {resolved}" + (f"  (filtered to {resolved})" if result.filtered else "  (whole corpus)"))
    print(f"gate             : min_similarity={result.threshold}  top_k={top_k or get_settings().top_k}")
    print(f"verdict          : {'PASS' if result.passed else 'ABSTAIN'}  ({len(result.hits)} hits, top score "
          f"{result.hits[0].score:.4f})" if result.hits else f"verdict          : ABSTAIN (0 hits)")
    print(f"latency          : {result.latency_ms} ms")

    if not result.hits:
        print("\n  no chunks passed the gate - the bot would answer 'I don't have that' here.")
        return

    for rank, hit in enumerate(result.hits, 1):
        chunk = hit.chunk
        text = " ".join(chunk.text.split())
        if len(text) > PREVIEW_CHARS:
            text = text[:PREVIEW_CHARS] + "..."
        print(f"\n  [{rank}] score {hit.score:.4f}  {chunk.chunk_id}")
        print(f"      {chunk.doc_type}  |  scheme {chunk.scheme_id}  |  tokens {chunk.token_count}")
        print(f"      section: {chunk.section}")
        print(f"      {text}")


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Inspect retrieval results without generating an answer.")
    parser.add_argument("query", nargs="*", help="one or more queries to inspect")
    parser.add_argument("--scheme", default=None, help="force a scheme id instead of detecting one")
    parser.add_argument("--top-k", type=int, default=None, help=f"override TOP_K (default {settings.top_k})")
    parser.add_argument(
        "--min-similarity", type=float, default=None, help=f"override the gate (default {settings.min_similarity})"
    )
    parser.add_argument("--interactive", action="store_true", help="keep asking until you type 'quit'")
    args = parser.parse_args()

    if args.interactive:
        print(f"TOP_K={settings.top_k}  MIN_SIMILARITY={settings.min_similarity}   (type 'quit' to exit)")
        while True:
            try:
                query = input("\nquery> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if query.lower() in {"quit", "exit", "q", ""}:
                return 0
            show(query, top_k=args.top_k, min_similarity=args.min_similarity, scheme_id=args.scheme)

    if not args.query:
        parser.print_help()
        return 1

    for query in args.query:
        show(query, top_k=args.top_k, min_similarity=args.min_similarity, scheme_id=args.scheme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
