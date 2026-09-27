"""Build-phase CLI: load -> chunk -> embed -> store.

    python -m src.ingest.build_index --rebuild           # full pipeline, fresh index
    python -m src.ingest.build_index --stage chunk        # chunks.jsonl from snapshots
    python -m src.ingest.build_index --stage load         # refresh data/raw from the registry
    python -m src.ingest.build_index --stage embed        # warm the embedding cache only
    python -m src.ingest.build_index --stage store        # upsert existing chunks.jsonl
    python -m src.ingest.build_index --stage all          # every stage, reusing snapshots

`--rebuild` resets the vector collection first, so chunks from an older chunker can never
survive a rebuild.
"""

from __future__ import annotations

import argparse
import json
import time

from src import paths
from src.paths import PROJECT_ROOT
from src.ingest.chunker import MAX_TOKENS, OVERLAP, chunk_document, count_tokens
from src.ingest.loader import load_all, load_snapshots
from src.observability import get_logger
from src.types import Chunk

logger = get_logger("build_index")


def write_chunks(chunks: list[Chunk], target=None) -> None:
    destination = target or paths.CHUNKS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")


def read_chunks(target=None) -> list[Chunk]:
    source = target or paths.CHUNKS_PATH
    if not source.exists():
        return []
    chunks: list[Chunk] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        chunks.append(Chunk(**payload))
    return chunks


def build(*, force: bool = False, only_schemes: list[str] | None = None) -> list[Chunk]:
    load_all(force=force, only_schemes=only_schemes)
    return chunk_snapshots(only_schemes=only_schemes)


def chunk_snapshots(*, only_schemes: list[str] | None = None) -> list[Chunk]:
    docs = load_snapshots(only_schemes=only_schemes)
    if not docs:
        logger.warning("no snapshots found in %s - run --stage load first", paths.RAW_DIR)
        return []

    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, max_tokens=MAX_TOKENS, overlap=OVERLAP))
    write_chunks(chunks)
    _print_summary(chunks, docs)
    return chunks


def embed_chunks(chunks: list[Chunk] | None = None) -> tuple[list[Chunk], list[list[float]]]:
    from src.ingest import embedder

    chunks = chunks if chunks is not None else read_chunks()
    if not chunks:
        logger.warning("no chunks to embed - run --stage chunk first")
        return [], []

    started = time.perf_counter()
    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    elapsed = time.perf_counter() - started
    print(
        f"\nembedded {len(chunks)} chunks in {elapsed:.1f}s "
        f"({len(vectors)} vectors, dim={len(vectors[0])}) cache={embedder.cache_stats()}"
    )
    return chunks, vectors


def store_chunks(chunks: list[Chunk] | None = None, vectors: list[list[float]] | None = None) -> int:
    from src.ingest import store

    if chunks is None or vectors is None:
        chunks, vectors = embed_chunks(chunks)
    if not chunks:
        return 0

    written = store.upsert_chunks(chunks, vectors)
    print(f"stored {written} chunks -> {paths.CHROMA_DIR}")
    for key, value in sorted(store.collection_stats().items()):
        print(f"  {key:22} {value}")
    return written


def _print_summary(chunks: list[Chunk], docs: list) -> None:
    per_doc: dict[str, int] = {}
    per_splitter: dict[str, int] = {}
    per_scheme: dict[str, int] = {}
    for chunk in chunks:
        per_doc[chunk.chunk_id.rsplit("__", 1)[0]] = per_doc.get(chunk.chunk_id.rsplit("__", 1)[0], 0) + 1
        per_splitter[chunk.splitter] = per_splitter.get(chunk.splitter, 0) + 1
        per_scheme[chunk.scheme_id] = per_scheme.get(chunk.scheme_id, 0) + 1

    tokens = [chunk.token_count for chunk in chunks]
    print(f"\ndocuments={len(docs)}  chunks={len(chunks)}  out={paths.CHUNKS_PATH}")
    if tokens:
        print(
            f"tokens: total={sum(tokens)} min={min(tokens)} median={sorted(tokens)[len(tokens) // 2]} "
            f"max={max(tokens)} (cap={MAX_TOKENS})"
        )
    print("by doc_type/splitter:", dict(sorted(per_splitter.items())))
    print("by scheme:          ", dict(sorted(per_scheme.items())))
    for doc_id, count in sorted(per_doc.items()):
        print(f"  {doc_id:<28} {count:>3} chunks")
    over = [c.chunk_id for c in chunks if c.token_count > MAX_TOKENS]
    if over:
        print(f"WARNING: {len(over)} chunks exceed the cap: {over[:5]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the ingestion artifacts.")
    parser.add_argument(
        "--stage",
        choices=["load", "chunk", "embed", "store", "export", "all"],
        default="all",
        help="load | chunk | embed | store | export | all (default: all)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="force-refresh snapshots and reset the vector collection first",
    )
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these scheme ids")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    stages = ["load", "chunk", "embed", "store", "export"] if args.stage == "all" else [args.stage]

    if args.rebuild:
        from src.ingest import store

        store.reset_collection()
        print("collection reset")

    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for stage in stages:
        if stage == "load":
            load_all(force=args.rebuild, only_schemes=args.only)
        elif stage == "chunk":
            chunks = chunk_snapshots(only_schemes=args.only)
        elif stage == "embed":
            chunks, vectors = embed_chunks(chunks or None)
        elif stage == "store":
            if vectors:
                store_chunks(chunks, vectors)
            else:
                store_chunks()
        elif stage == "export":
            from src.ingest import exporter

            # Always exports from the persisted corpus, never from a partial in-memory run,
            # so a single-stage invocation cannot write a manifest describing fewer chunks
            # than the index actually holds.
            csv_path, md_path = exporter.export()
            print(f"wrote {csv_path.relative_to(PROJECT_ROOT)}")
            print(f"wrote {md_path.relative_to(PROJECT_ROOT)}")

    if stages == ["load", "chunk"]:
        chunks = read_chunks()

    over_cap = [c for c in chunks if c.token_count > args.max_tokens]
    if over_cap:
        print(f"ERROR: {len(over_cap)} chunks exceed --max-tokens={args.max_tokens}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
