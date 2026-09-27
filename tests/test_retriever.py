"""P4 - Retriever tests.

Most cases run against the real 188-chunk index in `data/chroma` because the acceptance
criteria are about real corpus behaviour, not mocks. The collection is only reset when
it is genuinely empty, so a populated index survives a test run.
"""

from __future__ import annotations

import pytest
from collections import Counter

from src.config import reset_settings_cache
from src.ingest.chunker import count_tokens
from src.ingest.store import collection_stats
from src.retrieval import retriever
from src.retrieval.retriever import RetrievalResult, dedupe, retrieve, resolve_scheme
from src.types import Chunk, RetrievedChunk

EMBED = [0.05] * 384


def _real_index_available() -> bool:
    try:
        return collection_stats().get("count", 0) > 0
    except Exception:
        return False


needs_real_index = pytest.mark.skipif(
    not _real_index_available(),
    reason="requires a built index; run `python -m src.ingest.build_index --rebuild`",
)


# ---------------------------------------------------------------- resolve_scheme


def test_resolve_scheme_matches_the_specified_aliases():
    assert resolve_scheme("hdfc elss tax saver") == "S3"
    assert resolve_scheme("tell me about quant funds") is None


def test_resolve_scheme_is_case_and_word_boundary_aware():
    assert resolve_scheme("What is the EXIT LOAD on HDFC LaRgE cAp?") == "S1"
    # A longer word merely containing an alias must not match.
    assert resolve_scheme("balancedly spoken, nothing specific") is None
    assert resolve_scheme("") is None


def test_resolve_scheme_returns_none_when_two_schemes_are_named():
    # Guards the retriever from narrowing the corpus to a scheme the user did not ask
    # about, which would silently drop the other scheme's answer.
    assert resolve_scheme("compare large cap and flexi cap") is None


# ------------------------------------------------------------------------- dedupe


def _hit(chunk_id: str, score: float, section: str, *, scheme_id="S1", doc_type="scheme_page") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=f"body of {chunk_id}",
            token_count=3,
            scheme_id=scheme_id,
            scheme_name="HDFC Large Cap Fund",
            category="Large cap",
            doc_type=doc_type,
            section=section,
            source_url="https://example.invalid",
            publisher="Groww",
            retrieved_at="2026-01-01",
            chunk_index=0,
            splitter="SectionSplitter",
            content_hash="deadbeef",
        ),
        score=score,
    )


def test_dedupe_caps_hits_per_scheme_doc_type_and_section():
    hits = [
        _hit("a", 0.90, "Exit Load"),
        _hit("b", 0.89, "Exit Load"),
        _hit("c", 0.88, "Exit Load"),
        _hit("d", 0.70, "Portfolio"),
    ]
    kept = dedupe(hits, per_section=2)
    kept_ids = [h.chunk.chunk_id for h in kept]

    assert kept_ids == ["a", "b", "d"]
    assert sum(1 for h in kept if h.chunk.section == "Exit Load") <= 2


def test_dedupe_keeps_distinct_sections_and_doc_types():
    hits = [
        _hit("a", 0.90, "Exit Load"),
        _hit("b", 0.89, "Exit Load", doc_type="reference"),
        _hit("c", 0.88, "Exit Load", scheme_id="S3"),
        _hit("d", 0.87, "Portfolio"),
    ]
    assert len(dedupe(hits, per_section=2)) == 4


def test_dedupe_preserves_score_order_and_is_deterministic():
    hits = [_hit("b", 0.50, "X"), _hit("a", 0.50, "X")]
    first = [h.chunk.chunk_id for h in dedupe(hits)]
    second = [h.chunk.chunk_id for h in dedupe(list(reversed(hits)))]
    assert first == second == ["a", "b"]

    ordered = [_hit("low", 0.10, "A"), _hit("high", 0.90, "B")]
    assert [h.chunk.chunk_id for h in dedupe(ordered)] == ["high", "low"]


def test_dedupe_rejects_a_useless_per_section():
    with pytest.raises(ValueError):
        dedupe([_hit("a", 0.5, "X")], per_section=0)


# ------------------------------------------------- contract shape (no live index)


def test_retrieve_abstains_on_an_empty_query_without_touching_the_index():
    result = retrieve("   ")
    assert isinstance(result, RetrievalResult)
    assert result.passed is False
    assert result.hits == []
    assert bool(result) is False


def test_retrieve_defaults_come_from_settings(monkeypatch):
    monkeypatch.setattr(retriever, "get_collection", lambda: _StubCollection([]))
    monkeypatch.setattr(retriever, "embed_query", lambda text: EMBED)
    result = retrieve("hdfc large cap exit load")

    assert result.threshold == 0.30
    assert result.resolved_scheme_id == "S1"
    assert result.filtered is True
    assert result.passed is False
    assert result.latency_ms >= 0


def test_explicit_scheme_id_overrides_alias_detection(monkeypatch):
    stub = _StubCollection([])
    monkeypatch.setattr(retriever, "get_collection", lambda: stub)
    monkeypatch.setattr(retriever, "embed_query", lambda text: EMBED)

    result = retrieve("tell me about quant funds", scheme_id="S2")

    assert result.resolved_scheme_id == "S2"
    assert stub.last_where == {"scheme_id": "S2"}


def test_ambiguous_query_searches_the_whole_corpus(monkeypatch):
    stub = _StubCollection([])
    monkeypatch.setattr(retriever, "get_collection", lambda: stub)
    monkeypatch.setattr(retriever, "embed_query", lambda text: EMBED)

    result = retrieve("compare large cap and flexi cap")

    assert result.resolved_scheme_id is None
    assert result.filtered is False
    assert stub.last_where is None


def test_no_where_filter_when_the_query_names_no_scheme(monkeypatch):
    stub = _StubCollection([])
    monkeypatch.setattr(retriever, "get_collection", lambda: stub)
    monkeypatch.setattr(retriever, "embed_query", lambda text: EMBED)

    result = retrieve("what documents are available")

    assert result.resolved_scheme_id is None
    assert stub.last_where is None


def test_passed_is_true_when_any_hit_clears_the_threshold(monkeypatch):
    stub = _StubCollection([("S1__exit__0000", "Exit load\n1% within 1 year", 0.80)])
    monkeypatch.setattr(retriever, "get_collection", lambda: stub)
    monkeypatch.setattr(retriever, "embed_query", lambda text: EMBED)

    result = retrieve("hdfc large cap exit load")

    assert result.passed is True
    assert result.hits[0].score == pytest.approx(0.80, abs=1e-6)
    assert result.top_score == pytest.approx(0.80, abs=1e-6)
    assert result.citations[0].chunk_id == "S1__exit__0000"
    assert result.citations[0].token_count == count_tokens("Exit load\n1% within 1 year")


def test_passed_is_false_when_every_hit_is_below_the_threshold(monkeypatch):
    stub = _StubCollection([("S1__exit__0000", "Exit load\n1% within 1 year", 0.29)])
    monkeypatch.setattr(retriever, "get_collection", lambda: stub)
    monkeypatch.setattr(retriever, "embed_query", lambda text: EMBED)

    assert retrieve("hdfc large cap exit load").passed is False


def test_top_k_truncates_the_hit_list(monkeypatch):
    # Distinct sections, otherwise the per-section cap (correctly) trims first.
    rows = [(f"S1__s__{i:04d}", "body", 0.05 * i) for i in range(12)]
    stub = _StubCollection(rows, section_for=lambda cid: f"Section {cid}")
    monkeypatch.setattr(retriever, "get_collection", lambda: stub)
    monkeypatch.setattr(retriever, "embed_query", lambda text: EMBED)

    assert len(retrieve("hdfc large cap", top_k=3).hits) == 3


def test_dedupe_treats_differently_cased_sections_as_one_section():
    """Regression: every "Exit load" section exists in the real corpus under both
    `Exit Load` and `Exit load`. Raw-string comparison let that single section fill four
    of five slots, which is exactly the crowding-out the cap exists to prevent."""
    hits = [
        _hit("a", 0.90, "HDFC Large Cap Fund — Exit Load"),
        _hit("b", 0.89, "HDFC Large Cap Fund — Exit load"),
        _hit("c", 0.88, "HDFC Large Cap Fund — EXIT LOAD"),
        _hit("d", 0.70, "HDFC Large Cap Fund —  Exit   load "),
    ]
    kept = [h.chunk.chunk_id for h in dedupe(hits, per_section=2)]

    assert kept == ["a", "b"]


def test_dedupe_cap_still_leaves_room_for_other_sections(monkeypatch):
    """The point of the cap: a verbose section must not crowd out other evidence."""
    rows = [
        ("S1__a", "Exit load body", 0.90),
        ("S1__b", "Exit load body", 0.88),
        ("S1__c", "Exit load body", 0.86),
        ("S1__d", "Expense ratio body", 0.50),
        ("S1__e", "Benchmark body", 0.40),
    ]
    stub = _StubCollection(rows, section_for=lambda cid: "Exit load" if cid in {"S1__a", "S1__b", "S1__c"} else cid[-1])
    monkeypatch.setattr(retriever, "get_collection", lambda: stub)
    monkeypatch.setattr(retriever, "embed_query", lambda text: EMBED)

    ids = [h.chunk.chunk_id for h in retrieve("hdfc large cap", top_k=3).hits]

    assert len(ids) == 3
    assert "S1__d" in ids, "a lower-scoring distinct section should still get a slot"


def test_incomplete_metadata_is_reported_clearly(monkeypatch):
    stub = _StubCollection([("S1__exit__0000", "text", 0.9)], drop=("section",))
    monkeypatch.setattr(retriever, "get_collection", lambda: stub)
    monkeypatch.setattr(retriever, "embed_query", lambda text: EMBED)

    with pytest.raises(ValueError, match="missing"):
        retrieve("hdfc large cap")


# ------------------------------------------------------------ real-index behaviour


@needs_real_index
def test_known_fact_ranks_its_chunk_in_the_top_3():
    """Acceptance: a known fact returns its chunk in the top 3 (exit-load question)."""
    result = retrieve("What is the exit load on HDFC Large Cap Fund?", top_k=3)

    assert result.passed is True
    top3 = [hit.chunk.chunk_id for hit in result.hits]
    assert any("exit" in hit.chunk.section.lower() or "1%" in hit.chunk.text for hit in result.hits)
    assert any("1%" in hit.chunk.text for hit in result.hits), top3


@needs_real_index
def test_scheme_restricted_search_excludes_other_schemes():
    """Acceptance: a where-filter test proves other schemes' chunks are excluded."""
    result = retrieve("HDFC Small Cap Fund exit load", top_k=5)

    assert result.resolved_scheme_id == "S4"
    assert result.hits, "expected at least one S4 hit"
    assert {hit.chunk.scheme_id for hit in result.hits} == {"S4"}


@needs_real_index
def test_nonsense_query_abstains():
    """Acceptance: a nonsense query yields passed is False, exercising the abstain path."""
    result = retrieve("kjhgfdsa qwerty zxcvbn zzzz nonsense gibberish")

    assert result.passed is False
    assert result.hits == [] or all(h.score < result.threshold for h in result.hits)


@needs_real_index
def test_real_hits_respect_the_dedupe_cap():
    result = retrieve("HDFC Large Cap Fund exit load and portfolio holdings", top_k=5)

    # The invariant is about *normalised* sections: the corpus legitimately stores one
    # logical section under two spellings, so raw strings can never be compared for
    # uniqueness here.
    keys = [retriever._evidence_key(h) for h in result.hits]
    counts = Counter(keys)

    assert max(counts.values()) <= 2, keys
    # And the exit-load section specifically must not take over the result.
    exit_slots = sum(1 for k in counts if k[2].endswith("exit load"))
    assert exit_slots <= 2, keys


@needs_real_index
def test_latency_is_measured():
    assert retrieve("what is the exit load on HDFC Large Cap Fund?").latency_ms >= 0


class _StubCollection:
    """Minimal stand-in recording the `where` filter the retriever chose."""

    def __init__(self, rows, drop: tuple[str, ...] = (), section_for=None):
        self.rows = rows
        self.drop = drop
        self.section_for = section_for
        self.last_where: dict | None = None
        self.last_n: int | None = None

    def query(self, *, query_embeddings, n_results, where=None, include=None):
        self.last_where = where
        self.last_n = n_results
        rows = self.rows[:n_results]
        ids, docs, metas, dists = [], [], [], []
        for chunk_id, doc, score in rows:
            meta = {f: f"v-{f}" for f in _META_FIELDS}
            meta["chunk_index"] = 0
            meta["section"] = self.section_for(chunk_id) if self.section_for else "v-section"
            for field in self.drop:
                meta.pop(field, None)
            ids.append(chunk_id)
            docs.append(doc)
            metas.append(meta)
            dists.append(1.0 - score)
        return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}


_META_FIELDS = retriever.METADATA_FIELDS
