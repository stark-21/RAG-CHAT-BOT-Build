"""P7 - Pipeline orchestrator tests.

The load-bearing tests here use mocks to prove *negative* claims: that a refusal performs no
Chroma call and no LLM call, and that a PAN never reaches the log. A pipeline that merely
returns the right `status` could still be quietly calling the LLM on a refusal path, which
is the failure mode worth defending against.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import pipeline
from src.config import get_settings, reset_settings_cache
from src.llm.provider import ProviderError
from src.pipeline import answer_question
from src.types import Answer

FACTUAL = "What is the exit load on HDFC Large Cap Fund?"
ADVICE = "Should I buy the small cap fund?"
OUT_OF_SCOPE = "What is the expense ratio of Parag Parag Flexi Cap?"
PAN_QUERY = "My PAN is ABCDE1234F, what is the exit load?"
NONSENSE = "kjhgfdsa qwerty zxcvbn gibberish zzzz"


@pytest.fixture
def no_side_effects(monkeypatch, tmp_path):
    """Redirect the query log and make any Chroma/LLM call an immediate failure."""
    log = tmp_path / "queries.jsonl"
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", log)

    def forbidden(name):
        def boom(*args, **kwargs):
            raise AssertionError(f"{name} must not be called on this path")

        return boom

    monkeypatch.setattr(pipeline, "retrieve", forbidden("retrieve"))
    monkeypatch.setattr(pipeline, "llm_generate", forbidden("llm_generate"))
    return log


def read_log(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------- refusal fast paths


def test_advice_is_refused_with_no_retrieval_and_no_llm(no_side_effects):
    answer = answer_question(ADVICE)

    assert isinstance(answer, Answer)
    assert answer.status == "refused"
    assert answer.trace.guards["intent"] == "advice"
    assert answer.trace.latency_ms == {"retrieve": 0, "generate": 0, "total": answer.trace.latency_ms["total"]}


def test_performance_is_refused_with_no_llm_call(no_side_effects):
    answer = answer_question("Which fund has the best 5-year returns?")
    assert answer.status == "refused"
    assert answer.trace.guards["intent"] == "performance"


def test_out_of_scope_is_refused_with_the_scope_message(no_side_effects):
    answer = answer_question(OUT_OF_SCOPE)

    assert answer.status == "refused"
    assert answer.trace.guards["intent"] == "out_of_scope"
    assert "not covered" in answer.text.lower() or "not one of" in answer.text.lower()


def test_out_of_scope_never_links_the_similar_hdfc_scheme(no_side_effects):
    answer = answer_question(OUT_OF_SCOPE)

    for citation in answer.citations:
        assert "hdfc" not in citation.url.lower(), citation.url


def test_pan_is_pii_rejected_and_never_logged(no_side_effects):
    answer = answer_question(PAN_QUERY)

    assert answer.status == "pii_rejected"
    assert "ABCDE1234F" not in answer.text

    written = no_side_effects.read_text(encoding="utf-8")
    assert "ABCDE1234F" not in written
    assert "My PAN" not in written
    assert "ABCDE1234F" not in json.dumps(read_log(no_side_effects))


def test_pii_refusal_still_logs_one_line(no_side_effects):
    answer_question(PAN_QUERY)
    records = read_log(no_side_effects)

    assert len(records) == 1
    assert records[0]["status"] == "pii_rejected"
    assert records[0]["query_redacted"] is True


def test_a_refusal_log_carries_guards_and_no_query_text(no_side_effects):
    answer_question(ADVICE)
    record = read_log(no_side_effects)[0]

    assert record["guards"]["intent"] == "advice"
    assert record["query_hash"]
    assert "should i buy" not in json.dumps(record).lower()


# ---------------------------------------------------------------- answered path


def test_factual_question_is_answered_with_citation_and_date(monkeypatch):
    monkeypatch.setattr(
        pipeline, "llm_generate",
        lambda prompt: f"The exit load is 1% if redeemed within 1 year.\nsource: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    )
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", Path("/tmp/p7-answered.jsonl"))

    answer = answer_question(FACTUAL)

    assert answer.status == "answered", answer.trace.error
    assert answer.citations
    assert answer.last_updated
    assert "Last updated from sources:" in answer.text
    assert answer.trace.threshold_passed is True


def test_answered_text_is_capped_at_three_sentences(monkeypatch):
    url = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
    monkeypatch.setattr(
        pipeline, "llm_generate",
        lambda prompt: f"A one. B two. C three. D four. E five. F six. source: {url}",
    )
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", Path("/tmp/p7-cap.jsonl"))

    answer = answer_question(FACTUAL)
    body = answer.text.split("Last updated from sources:")[0]

    assert answer.status == "answered"
    assert "D four" not in body and "F six" not in body


def test_nonsense_query_abstains_with_no_llm_call(no_side_effects):
    """A nonsense query does reach the retriever, but must not reach the LLM."""
    monkeypatch_hits = None

    def fake_retrieve(query, **kwargs):
        from src.retrieval.retriever import RetrievalResult

        return RetrievalResult(
            hits=[],
            threshold=get_settings().min_similarity,
            passed=False,
            resolved_scheme_id=None,
            latency_ms=1,
        )

    import src.pipeline as p

    original = p.retrieve
    p.retrieve = fake_retrieve
    try:
        answer = answer_question(NONSENSE)
    finally:
        p.retrieve = original

    assert answer.status == "insufficient_context"
    assert answer.trace.error == "below_similarity_threshold"
    assert "will not answer it from memory" in answer.text


def test_insufficient_context_offers_the_official_page_link(monkeypatch):
    def fake_retrieve(query, **kwargs):
        from src.retrieval.retriever import RetrievalResult

        return RetrievalResult(
            hits=[],
            threshold=get_settings().min_similarity,
            passed=False,
            resolved_scheme_id=None,
            latency_ms=1,
        )

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", Path("/tmp/p7-insuff.jsonl"))

    answer = answer_question(NONSENSE)
    assert "Learn more: http" in answer.text


def test_empty_query_is_insufficient_and_logged(monkeypatch):
    log = Path("/tmp/p7-empty.jsonl")
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", log)

    answer = answer_question("   ")

    assert answer.status == "insufficient_context"
    assert read_log(log)


# ------------------------------------------------------------- L6 demotion in situ


def test_a_fabricated_percentage_is_never_answered(monkeypatch):
    """Acceptance: an injected provider returning a fabricated % must not be 'answered'."""
    calls = {"n": 0}

    def fabricating(prompt):
        calls["n"] += 1
        return (
            "This fund has delivered 18.5% annualised returns. "
            "source: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
        )

    monkeypatch.setattr(pipeline, "llm_generate", fabricating)
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", Path("/tmp/p7-fab.jsonl"))

    answer = answer_question(FACTUAL)

    assert answer.status != "answered"
    assert answer.status == "insufficient_context"
    assert "18.5" in answer.text or "18.5" in (answer.trace.error or "")
    assert calls["n"] == 2, "exactly one regeneration attempt"


def test_an_invented_citation_is_never_answered(monkeypatch):
    monkeypatch.setattr(
        pipeline, "llm_generate",
        lambda prompt: "Exit load is 1%. See https://hdfc.example.com/fees for details.",
    )
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", Path("/tmp/p7-inv.jsonl"))

    answer = answer_question(FACTUAL)

    assert answer.status == "insufficient_context"
    assert answer.trace.error.startswith("citation_not_in_context")


def test_a_second_good_attempt_is_accepted(monkeypatch):
    """The regeneration path must actually be able to succeed."""
    url = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
    responses = iter(
        [
            f"This fund returned 18.5% last year. source: {url}",
            f"The exit load is 1% if redeemed within 1 year. source: {url}",
        ]
    )
    monkeypatch.setattr(pipeline, "llm_generate", lambda prompt: next(responses))
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", Path("/tmp/p7-retry.jsonl"))

    answer = answer_question(FACTUAL)

    assert answer.status == "answered", answer.trace.error
    assert answer.trace.guards.get("regenerated") is True


# ------------------------------------------------------------- provider resilience


def test_provider_exception_retries_then_falls_back_to_the_stub(monkeypatch):
    calls = {"n": 0}

    def broken(prompt):
        calls["n"] += 1
        raise ProviderError("hosted provider returned HTTP 500")

    monkeypatch.setattr(pipeline, "llm_generate", broken)
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", Path("/tmp/p7-broken.jsonl"))

    answer = answer_question(FACTUAL)

    assert calls["n"] == 2, "one retry"
    assert answer.status == "answered"
    assert answer.trace.guards.get("fell_back_to_stub") is True
    assert "no LLM available" in answer.text


def test_a_retrieval_failure_becomes_an_error_answer_not_a_crash(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(pipeline, "retrieve", boom)
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", Path("/tmp/p7-boom.jsonl"))

    answer = answer_question(FACTUAL)

    assert answer.status == "error"
    assert "chroma exploded" not in answer.text
    assert "chroma exploded" in answer.trace.error
    assert "Traceback" not in answer.text


def test_a_broken_guard_does_not_leak_a_stack_trace(monkeypatch):
    monkeypatch.setattr(
        pipeline, "classify_intent",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("guard blew up")),
    )
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", Path("/tmp/p7-guard.jsonl"))

    answer = answer_question(FACTUAL)

    assert answer.status == "error"
    assert "Traceback" not in answer.text
    assert "guard blew up" in answer.trace.error


# ------------------------------------------------------------------------ logging


def test_exactly_one_log_line_per_call_with_total_latency(monkeypatch, tmp_path):
    log = tmp_path / "queries.jsonl"
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", log)
    monkeypatch.setattr(
        pipeline, "llm_generate",
        lambda prompt: "The exit load is 1%. source: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    )

    before = len(read_log(log))
    answer_question(FACTUAL)
    answer_question(ADVICE)
    answer_question(OUT_OF_SCOPE)
    after = read_log(log)

    assert len(after) - before == 3
    for record in after[before:]:
        assert record["kind"] == "query"
        assert "total" in record["latency_ms"]
        assert record["query_hash"]
        assert record["status"] in {"answered", "refused", "insufficient_context", "error", "pii_rejected"}


def test_query_hash_is_stable_and_normalised(monkeypatch, tmp_path):
    log = tmp_path / "queries.jsonl"
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", log)
    monkeypatch.setattr(pipeline, "retrieve", lambda *a, **k: pytest.fail("should abstain early"))

    answer_question(ADVICE)
    answer_question("  SHOULD   i buy the small cap fund?  ")

    records = read_log(log)
    assert records[0]["query_hash"] == records[1]["query_hash"], "case and spacing must not matter"


def test_telemetry_failure_never_breaks_an_answer(monkeypatch, tmp_path):
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", tmp_path / "nope" / "queries.jsonl")
    monkeypatch.setattr(
        pipeline, "llm_generate",
        lambda prompt: "The exit load is 1%. source: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
    )

    answer = answer_question(FACTUAL)
    assert answer.status == "answered"


# --------------------------------------------------------------------- packaging


def test_pipeline_imports_without_streamlit():
    import sys

    assert "streamlit" not in sys.modules
    assert callable(answer_question)


def test_answer_question_accepts_an_explicit_settings_object():
    from dataclasses import replace

    answer = answer_question(ADVICE, settings=replace(get_settings(), top_k=1))
    assert answer.status == "refused"


def test_the_trace_lists_every_retrieved_chunk_not_only_the_cited_ones(monkeypatch):
    """`hits` is what retrieval returned; `cited_doc_ids` is what the answer used. The
    generator's partial trace listed only cited chunks, and because the pipeline preferred
    it, a question that cited 1 of 5 retrieved chunks reported a single hit - so the "Why
    this answer?" panel and the novelty check could not see the other four."""
    from src.pipeline import answer_question
    from src.retrieval.retriever import RetrievalResult
    from src.types import Answer, Chunk, Citation, QueryTrace, RetrievedChunk

    def make(i: int, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"S1__overview__{i:04d}",
                text=f"HDFC Large Cap Fund - Direct - Growth — Section {i}\nValue {i}%",
                token_count=20,
                scheme_id="S1",
                scheme_name="HDFC Large Cap Fund - Direct - Growth",
                category="Large Cap",
                doc_type="scheme_page",
                section=f"Section {i}",
                source_url="https://example.invalid/a",
                publisher="p",
                retrieved_at="2026-09-27",
                chunk_index=i,
                splitter="LabelValueSplitter",
                content_hash=f"h{i}",
            ),
            score=score,
        )

    hits = [make(i, 0.9 - i / 100) for i in range(5)]
    result = RetrievalResult(
        hits=hits,
        threshold=0.3,
        passed=True,
        resolved_scheme_id="S1",
        latency_ms={"retrieve": 10},
        query="q",
        filtered=True,
    )
    monkeypatch.setattr("src.pipeline.retrieve", lambda *a, **k: result)

    # A provider that cites only the first chunk, and whose answer is otherwise valid.
    def generate(query, *, settings=None, **kwargs):
        return Answer(
            status="answered",
            text="Value 0%.",
            citations=[Citation(label="c", url="https://example.invalid/a", doc_type="scheme_page", retrieved_at="2026-09-27")],
            last_updated="2026-09-27",
            trace=QueryTrace(
                query_hash="h",
                guards={},
                hits=[{"chunk_id": "S1__overview__0000", "score": 0.9}],
                threshold=0.3,
                threshold_passed=True,
                cited_doc_ids=["S1__overview__0000"],
                latency_ms={},
            ),
        )

    monkeypatch.setattr("src.pipeline.llm_generate", generate)

    answer = answer_question("What is the value?")

    # All five retrieved chunks are listed, in rank order, with their scores.
    assert [h["chunk_id"] for h in answer.trace.hits] == [f"S1__overview__{i:04d}" for i in range(5)]
    assert [h["score"] for h in answer.trace.hits] == [pytest.approx(0.9 - i / 100) for i in range(5)]
    # Each row carries the provenance the "Why this answer?" panel renders.
    assert answer.trace.hits[0]["section"] == "Section 0"
    assert answer.trace.hits[0]["doc_type"] == "scheme_page"
