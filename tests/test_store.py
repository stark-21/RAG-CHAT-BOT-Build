"""Phase P3 tests: embedder, embedding cache, and the ChromaDB store.

The store tests run against a temporary Chroma directory so they never touch
`data/chroma`. The one integration test reads the real index read-only.
"""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore", category=UserWarning)

from src.config import reset_settings_cache
from src.ingest import embedder, store
from src.types import Chunk

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def make_chunk(chunk_id: str = "S1__overview__0000", **kwargs) -> Chunk:
    defaults = dict(
        chunk_id=chunk_id,
        text="HDFC Large Cap Fund - Direct - Growth — Expense ratio\nExpense ratio: 1.03%",
        token_count=18,
        scheme_id="S1",
        scheme_name="HDFC Large Cap Fund - Direct - Growth",
        category="Large Cap",
        doc_type="scheme_page",
        section="Expense ratio",
        source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        publisher="HDFC AMC (via Groww)",
        retrieved_at="2026-09-27",
        chunk_index=0,
        splitter="LabelValueSplitter",
        content_hash="abc123",
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


@pytest.fixture()
def temp_collection(tmp_path, monkeypatch):
    """A throwaway Chroma collection, isolated from data/chroma."""
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    reset_settings_cache()
    store.reset_caches()
    store.reset_collection()
    yield store.get_collection()
    store.reset_caches()
    reset_settings_cache()


@pytest.fixture(scope="module")
def model_loaded():
    """Load the model once for the whole module; skip if it cannot be fetched."""
    try:
        return embedder.get_model()
    except Exception as exc:  # pragma: no cover - offline environment
        pytest.skip(f"embedding model unavailable: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------- embedder


def test_embed_query_has_384_dimensions(model_loaded):
    """Acceptance criterion: len(embed_query("test")) == 384."""
    vector = embedder.embed_query("test")

    assert len(vector) == 384
    assert len(vector) == embedder.EMBED_DIM


def test_embeddings_are_l2_normalised(model_loaded):
    vector = embedder.embed_query("What is the exit load?")

    norm = sum(value * value for value in vector) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_embed_documents_preserves_order_and_deduplicates(model_loaded):
    texts = ["exit load is one percent", "expense ratio is 1.03%", "exit load is one percent"]

    vectors = embedder.embed_documents(texts)

    assert len(vectors) == 3
    assert len({len(v) for v in vectors}) == 1
    assert vectors[0] == vectors[2]  # duplicate text, same vector
    assert vectors[0] != vectors[1]


def test_embed_documents_rejects_empty_input():
    assert embedder.embed_documents([]) == []


def test_cache_key_is_sha256_of_the_text():
    import hashlib

    text = "Expense ratio: 1.03%"
    assert embedder._cache_key(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_second_call_is_served_from_the_sqlite_cache(model_loaded, monkeypatch):
    texts = ["a unique cache probe sentence for p3"]
    first = embedder.embed_documents(texts)

    def explode():
        raise AssertionError("model.encode ran despite a full cache hit")

    monkeypatch.setattr(embedder, "get_model", explode)
    second = embedder.embed_documents(texts)

    assert second == first
    assert embedder.cache_stats()["rows"] >= 1


def test_cache_is_not_reused_for_a_different_model(model_loaded, tmp_path, monkeypatch):
    """A row written by another model must be ignored, not silently reused."""
    texts = ["cache model isolation probe"]
    embedder.embed_documents(texts)
    before = embedder.cache_stats()["rows"]

    monkeypatch.setattr(embedder, "_MODEL_ID", "some/other-model")
    try:
        assert embedder.cache_stats()["rows"] == 0
    finally:
        monkeypatch.setattr(embedder, "_MODEL_ID", MODEL_ID)
    assert embedder.cache_stats()["rows"] == before


def test_check_dim_rejects_a_wrong_width_vector():
    with pytest.raises(ValueError, match="384-dim"):
        embedder._check_dim([[0.0] * 128])


# ------------------------------------------------------------------- metadata


def test_metadata_has_exactly_the_architecture_fields():
    metadata = store.flatten_metadata(make_chunk())

    assert tuple(sorted(metadata)) == tuple(sorted(store.METADATA_FIELDS))
    assert set(store.METADATA_FIELDS) == {
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
    }


def test_metadata_values_are_all_scalars():
    metadata = store.flatten_metadata(make_chunk())

    for key, value in metadata.items():
        assert isinstance(value, (str, int, float, bool)), f"{key} is {type(value).__name__}"


def test_metadata_drops_non_schema_fields():
    """Chunk carries more fields than Chroma should store; nothing extra may leak in."""
    metadata = store.flatten_metadata(make_chunk(token_count=18, text="some text"))

    assert "text" not in metadata
    assert "token_count" not in metadata
    assert "chunk_id" not in metadata


@pytest.mark.parametrize("bad", [["S1", "S2"], {"a": 1}, None, ("S1",)])
def test_flatten_metadata_rejects_non_scalars(bad):
    """Regression: Chroma 1.5.9 accepts a list-valued metadata field without error, but a
    later `where` filter then matches nothing, which looks like "no context found"."""
    chunk = make_chunk()
    object.__setattr__(chunk, "scheme_id", bad)

    with pytest.raises(store.MetadataError):
        store.flatten_metadata(chunk)


def test_flatten_metadata_rejects_empty_strings():
    with pytest.raises(store.MetadataError, match="empty"):
        store.flatten_metadata(make_chunk(section="   "))


# ---------------------------------------------------------------------- store


def test_collection_declares_cosine_space_and_model(temp_collection):
    metadata = dict(temp_collection.metadata)

    assert metadata["hnsw:space"] == "cosine"
    assert metadata["embedding_model"] == MODEL_ID
    assert metadata["dim"] == 384


def test_upsert_chunks_writes_every_record(temp_collection):
    chunks = [make_chunk(f"S1__overview__{i:04d}", chunk_index=i) for i in range(3)]
    vectors = [[0.01 * (i + 1)] * 384 for i in range(3)]

    written = store.upsert_chunks(chunks, vectors)

    assert written == 3
    assert temp_collection.count() == 3


def test_upsert_is_idempotent_on_the_same_ids(temp_collection):
    chunks = [make_chunk()]
    vectors = [[0.02] * 384]

    store.upsert_chunks(chunks, vectors)
    store.upsert_chunks(chunks, [[0.03] * 384])

    assert temp_collection.count() == 1


def test_upsert_rejects_a_vector_count_mismatch(temp_collection):
    with pytest.raises(ValueError, match="2 chunks but 1 vectors"):
        store.upsert_chunks([make_chunk("a"), make_chunk("b")], [[0.0] * 384])


def test_upsert_rejects_a_wrong_dimension(temp_collection):
    with pytest.raises(ValueError, match="384-dim"):
        store.upsert_chunks([make_chunk()], [[0.0] * 128])


def test_upsert_of_nothing_is_a_no_op(temp_collection):
    assert store.upsert_chunks([], []) == 0


def test_peek_reports_384_dimensions_and_the_metadata_keys(temp_collection):
    store.upsert_chunks([make_chunk()], [[0.04] * 384])
    peek = temp_collection.peek(limit=1)

    assert len(peek["embeddings"][0]) == 384
    assert tuple(sorted(peek["metadatas"][0])) == tuple(sorted(store.METADATA_FIELDS))


def test_scheme_filter_matches_stored_records(temp_collection):
    """The guarantee P4 depends on: scalar metadata makes `where` filtering work."""
    expected = {
        "S1": ["S1__overview__0000"],
        "S2": ["S2__overview__0000"],
        "GEN": ["GEN__sebi_cas__0000"],
    }
    store.upsert_chunks(
        [
            make_chunk("S1__overview__0000", scheme_id="S1"),
            make_chunk("S2__overview__0000", scheme_id="S2"),
            make_chunk("GEN__sebi_cas__0000", scheme_id="GEN"),
        ],
        [[0.05] * 384, [0.06] * 384, [0.07] * 384],
    )

    for scheme_id, ids in expected.items():
        found = temp_collection.get(where={"scheme_id": scheme_id})
        assert found["ids"] == ids


def test_reset_collection_empties_the_index(temp_collection):
    store.upsert_chunks([make_chunk()], [[0.08] * 384])
    assert temp_collection.count() == 1

    store.reset_collection()

    assert store.get_collection().count() == 0
    assert store.get_collection().metadata["hnsw:space"] == "cosine"


def test_collection_stats_reports_count_and_metadata(temp_collection):
    store.upsert_chunks([make_chunk()], [[0.09] * 384])
    stats = store.collection_stats()

    assert stats["count"] == 1
    assert stats["meta_hnsw:space"] == "cosine"
    assert stats["peek_dim"] == 384
    assert stats["name"] == "mf_faq"


def test_query_returns_cosine_distances_only_when_requested(temp_collection):
    """Regression for the P4 trap: omitting `distances` from `include` yields None.

    Documents, metadatas and distances are all optional in chromadb 1.5.9, and a missing
    one comes back as None rather than an empty list.
    """
    store.upsert_chunks(
        [make_chunk("S1__overview__0000", text="Exit load of 1% if redeemed within 1 year")],
        [[0.01] * 384],
    )
    vector = [[0.01] * 384]

    without = temp_collection.query(query_embeddings=vector, n_results=1, include=["documents"])
    with_scores = temp_collection.query(
        query_embeddings=vector, n_results=1, include=["documents", "metadatas", "distances"]
    )

    assert without["distances"] is None
    assert with_scores["distances"] == [[pytest.approx(0.0, abs=1e-5)]]
    assert with_scores["metadatas"][0][0]["scheme_id"] == "S1"


def test_where_filter_actually_filters_after_upsert(temp_collection):
    store.upsert_chunks(
        [
            make_chunk("S1__overview__0000", scheme_id="S1", text="expense ratio one point zero three"),
            make_chunk("S3__overview__0000", scheme_id="S3", text="a different scheme entirely"),
        ],
        [[0.11] * 384, [0.22] * 384],
    )
    result = temp_collection.query(
        query_embeddings=[[0.11] * 384],
        n_results=5,
        where={"scheme_id": "S1"},
        include=["documents", "metadatas", "distances"],
    )

    assert result["ids"] == [["S1__overview__0000"]]


def test_reset_collection_prunes_orphaned_segment_directories(temp_collection, tmp_path):
    """Regression: `delete_collection()` leaves the HNSW segment directory on disk, so
    every `--rebuild` stranded ~170 KB. Directories that pre-date the reset and are no
    longer tracked must be removed."""
    chroma_dir = tmp_path / "chroma"
    store.upsert_chunks([make_chunk()], [[0.1] * 384])

    orphan = chroma_dir / "00000000-0000-0000-0000-000000000001"
    orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "data_level0.bin").write_bytes(b"stale")

    store.reset_collection()

    assert not orphan.exists()
    # The recreated collection must not have been harmed by the prune.
    assert store.get_collection().count() == 0
    assert store.get_collection().metadata["hnsw:space"] == "cosine"


def test_prune_never_deletes_a_live_segment(temp_collection, tmp_path):
    chroma_dir = tmp_path / "chroma"
    store.upsert_chunks([make_chunk()], [[0.12] * 384])
    live_dirs = {
        entry.name
        for entry in chroma_dir.iterdir()
        if entry.is_dir() and entry.name in store._live_segment_ids(chroma_dir)
    }
    assert live_dirs, "expected at least one live segment directory"

    store._prune_orphan_segments(chroma_dir, live_dirs)

    for name in live_dirs:
        assert (chroma_dir / name).exists()


def test_search_uses_our_embedder_not_chromas_default(temp_collection, monkeypatch):
    """Regression: `collection.query(query_texts=...)` makes Chroma download its own
    79 MB ONNX model and bypasses the SQLite cache. `search()` must always embed with
    `src.ingest.embedder` and pass `query_embeddings`."""
    store.upsert_chunks(
        [make_chunk("S1__overview__0000", text="Exit load of 1% if redeemed within 1 year")],
        [[0.01] * 384],
    )
    seen: dict = {}

    def spy(text):
        seen["text"] = text
        return [0.01] * 384

    monkeypatch.setattr(embedder, "embed_query", spy)
    result = store.search("what is the exit load", n_results=1)

    assert seen["text"] == "what is the exit load"
    assert result["ids"] == [["S1__overview__0000"]]
    assert result["distances"][0][0] == pytest.approx(0.0, abs=1e-5)
    assert result["metadatas"][0][0]["scheme_id"] == "S1"


def test_search_applies_a_where_filter(temp_collection, monkeypatch):
    store.upsert_chunks(
        [
            make_chunk("S1__overview__0000", scheme_id="S1", text="an s1 chunk"),
            make_chunk("S3__overview__0000", scheme_id="S3", text="an s3 chunk"),
        ],
        [[0.01] * 384, [0.02] * 384],
    )
    monkeypatch.setattr(embedder, "embed_query", lambda text: [0.01] * 384)

    result = store.search("anything", n_results=5, where={"scheme_id": "S3"})

    assert result["ids"] == [["S3__overview__0000"]]


# ------------------------------------------------------------------ integration


def test_real_collection_count_matches_chunks_jsonl():
    """Acceptance criterion: collection.count() == number of lines in chunks.jsonl."""
    from src import paths
    from src.ingest.build_index import read_chunks

    if not paths.CHUNKS_PATH.exists():
        pytest.skip("chunks.jsonl not built yet")
    chunks = read_chunks()
    if not chunks:
        pytest.skip("chunks.jsonl is empty")

    stats = store.collection_stats()
    if stats["count"] == 0:
        pytest.skip("vector collection not built yet - run build_index --rebuild")

    assert stats["count"] == len(chunks)
