"""P7 - Pipeline orchestrator.

One entrypoint, `answer_question(query) -> Answer`, owning every branch in
architecture.md 7.1:

    0. ingress guards   PII -> intent -> scope      (no Chroma, no LLM)
    1. retrieve         embed, filter, threshold    (abstain without an LLM call)
    2. augment          immutable rules + numbered context blocks
    3. generate         temperature 0, one regeneration
    4. validate+render  the three L6 checks, then a render-ready Answer
    5. log              one JSONL line, every query, refusals included

The layering is the product. A refusal costs no embedding, no vector search and no LLM
call, which is asserted in the tests with mocks rather than assumed. It also means an
advice question cannot be "helped along" by a weak threshold or a verbose retriever,
because it never reaches either.

Importing this module must not require Streamlit; P8 imports it, not the other way round.
"""

from __future__ import annotations

import time

from src.config import Settings, get_settings
from src.guardrails.intent import classify_intent, mentions_out_of_corpus, resolve_scheme
from src.guardrails.pii import detect_pii
from src.guardrails.refusal import build_refusal
from src.llm.generator import generate as llm_generate
from src.llm.generator import parse_answer
from src.llm.prompt_builder import build_prompt
from src.llm.provider import get_provider
from src.observability import hash_query, query_record
from src.retrieval.retriever import retrieve
from src.retrieval.memory import TurnBuffer, expand_with_scheme
from src.types import Answer, QueryTrace, RetrievedChunk

__all__ = ["answer_question"]

# Intents that must be refused before any retrieval. `out_of_scope` is handled by its own
# step below so the documented guard order (intent, then scope) is visible in the code.
_REFUSE_INTENTS = ("personal_data", "advice", "performance")
_RETRYABLE = ("novel_number", "citation_not_in_context", "no_citation_in_context", "empty_response")


def _latency_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _hit_rows(hits: list[RetrievedChunk] | None) -> list[dict]:
    """Flatten retrieved chunks into the trace/log row shape.

    `doc_type` and `section` are carried as well as the id and score so the UI's
    "Why this answer?" panel and `logs/queries.jsonl` both show which part of which
    document was used, matching the canonical trace in `implementation.md` §6.1.
    """
    return [
        {
            "chunk_id": hit.chunk.chunk_id,
            "score": round(hit.score, 4),
            "doc_type": hit.chunk.doc_type,
            "section": hit.chunk.section,
        }
        for hit in (hits or [])
    ]


def _trace(
    query: str,
    *,
    guards: dict,
    hits: list[RetrievedChunk] | None = None,
    threshold: float | None = None,
    threshold_passed: bool = False,
    cited: list[str] | None = None,
    latency: dict | None = None,
    error: str | None = None,
) -> QueryTrace:
    return QueryTrace(
        query_hash=hash_query(query),
        guards=guards,
        hits=_hit_rows(hits),
        threshold=threshold if threshold is not None else get_settings().min_similarity,
        threshold_passed=threshold_passed,
        cited_doc_ids=list(cited or []),
        latency_ms=dict(latency or {}),
        error=error,
    )


# ------------------------------------------------------------------ terminal branches


def _error(exc: Exception, trace: QueryTrace) -> Answer:
    """A complete, renderable failure. The message sits behind the trace, never in `text`."""
    text = (
        "Something went wrong while answering that question. "
        "Please try rephrasing it, or ask about a specific published fact."
    )
    answer = Answer(
        status="error",
        text=text,
        citations=[],
        last_updated=None,
        trace=QueryTrace(
            query_hash=trace.query_hash,
            guards=trace.guards,
            hits=trace.hits,
            threshold=trace.threshold,
            threshold_passed=False,
            cited_doc_ids=[],
            latency_ms=trace.latency_ms,
            error=f"{type(exc).__name__}: {exc}",
        ),
    )
    return answer


def _insufficient(query: str, trace: QueryTrace, scheme_id: str | None = None) -> Answer:
    """Nothing in the corpus clears the bar. No LLM call is made to produce this."""
    from src.guardrails.refusal import _link_for

    link = _link_for("unknown", scheme_id)
    settings = get_settings()
    text = (
        "I could not find that in the published material for the schemes in this demo's "
        f"scope, so I will not answer it from memory.\n\n{settings.disclaimer}\n\nLearn more: {link}"
    )
    answer = Answer(
        status="insufficient_context",
        text=text,
        citations=[],
        last_updated=None,
        trace=QueryTrace(
            query_hash=hash_query(query),
            guards=trace.guards,
            hits=trace.hits,
            threshold=trace.threshold,
            threshold_passed=False,
            cited_doc_ids=[],
            latency_ms=trace.latency_ms,
            error=trace.error,
        ),
    )
    return answer


def _guard(query: str, settings: Settings | None = None) -> Answer | None:
    """Run the ingress guards. Returns a finished `Answer` if one fired, else None.

    Step 0 of architecture.md 7.1. Nothing here touches Chroma or an LLM.
    """
    cfg = settings or get_settings()

    pii = detect_pii(query)
    if pii:
        # Never echo or store the query itself once an identifier is present.
        return build_refusal(
            "personal_data",
            resolve_scheme(query),
            query=query,
            evidence={"rule": "pii_detected"},
            pii_types=pii,
        )

    intent, evidence = classify_intent(query)

    if intent in _REFUSE_INTENTS:
        return build_refusal(
            intent,
            resolve_scheme(query),
            query=query,
            evidence=evidence,
            pii_types=pii,
        )

    if intent == "out_of_scope" or mentions_out_of_corpus(query):
        # A resolved scheme is deliberately not passed through: "expense ratio of Parag
        # Parag Flexi Cap" resolves S2 via the "flexi cap" alias, and linking that HDFC page
        # would be the substitution this guard exists to prevent.
        return build_refusal(
            "out_of_scope",
            query=query,
            evidence=evidence if intent == "out_of_scope" else {"rule": "out_of_corpus_token"},
            pii_types=pii,
        )

    return None


# ----------------------------------------------------------------------- generation


def _generate_with_fallback(prompt: str) -> tuple[str, dict]:
    """Complete the prompt, degrading rather than failing.

    On an LLM exception: retry once, then fall back to the offline `stub` extract, and only
    then give up. Returns the raw text and a note about how it was produced.
    """
    notes: dict = {"llm_calls": 0, "fell_back_to_stub": False}

    try:
        notes["llm_calls"] += 1
        return llm_generate(prompt), notes
    except Exception as first:
        # Any backend can fail differently (HTTPError, socket timeout, JSON shape), and a
        # facts-only demo should degrade to a cited extract rather than show a stack trace.
        notes["first_error"] = f"{type(first).__name__}: {first}"

    try:
        notes["llm_calls"] += 1
        return llm_generate(prompt), notes
    except Exception as second:
        notes["second_error"] = f"{type(second).__name__}: {second}"

    from src.llm.provider import StubProvider

    notes["fell_back_to_stub"] = True
    return StubProvider().generate(prompt, temperature=0.0), notes


# ------------------------------------------------------------------------- entrypoint


def answer_question(
    query: str,
    settings: Settings | None = None,
    *,
    use_cache: bool = True,
    history: TurnBuffer | None = None,
) -> Answer:
    """Answer one question. Never raises: every path returns a render-ready `Answer`.

    `use_cache=False` forces the live path. The cache is consulted only *after* the ingress
    guards, so a cached answer can never bypass a refusal: a question that must be refused is
    refused on every run, cached or not.

    `history` is an optional bounded window of prior questions, used only to scope a
    deictic follow-up to a scheme. It never reaches generation and never alters the guards,
    the cache key, or the hashed query that gets logged; with `history=None` this function
    behaves exactly as it did before.
    """
    started = time.perf_counter()
    cfg = settings or get_settings()
    text = (query or "").strip()
    latency: dict = {}
    guards: dict = {}

    if not text:
        answer = _insufficient(query, _trace(query, guards={"empty_query": True}))
        answer.trace.latency_ms = {"retrieve": 0, "generate": 0, "total": _latency_ms(started)}
        _log(answer, guards, latency)
        return answer

    # ---- 0. ingress guards (no Chroma, no LLM) -------------------------------------
    try:
        refused = _guard(text, cfg)
    except Exception as exc:  # a broken guard must not leak a stack trace to the UI
        answer = _error(exc, _trace(text, guards={"guard_error": True}))
        answer.trace.latency_ms = {"retrieve": 0, "generate": 0, "total": _latency_ms(started)}
        _log(answer, guards, latency)
        return answer

    if refused is not None:
        answer = refused
        answer.trace.latency_ms = {"retrieve": 0, "generate": 0, "total": _latency_ms(started)}
        _log(answer, dict(answer.trace.guards), latency, pii_blocked=bool(answer.trace.guards.get("pii_types")))
        return answer

    # ---- 1. retrieve ---------------------------------------------------------------
    # A follow-up that names no scheme of its own inherits the most recent one from the
    # conversation, so "and its exit load?" is scoped to the fund under discussion instead
    # of searching all five. Resolved here, before the cache lookup, so a cached answer
    # establishes the same referent a retrieved one would.
    carried = history.carried_scheme_id(text) if history is not None else None
    # Only the embedded query is expanded. The guards, the cache key, and the logged hash
    # all keep using `text`, so a carried scheme can never change which guard applies, what
    # the cache is keyed on, or what is written to the query log.
    search_text = expand_with_scheme(text, carried)

    if use_cache:
        # Only reached once the guards have passed, so a refusal is never replayed.
        try:
            from src.llm.cache import AnswerCache

            cached = AnswerCache().get(text)
        except Exception:
            cached = None
        if cached is not None:
            # A cache hit returns before retrieval, so the referent is recorded here.
            # Without it the three cached demo questions would leave nothing to carry
            # forward, and the demo's own follow-ups would lose their scheme.
            if history is not None:
                history.remember(text)
            cached.trace = _trace(
                text,
                guards={"cache": "hit"},
                hits=[],
                cited=[],
                threshold=cfg.min_similarity,
            )
            cached.trace.latency_ms = {"retrieve": 0, "generate": 0, "total": _latency_ms(started)}
            _log(cached, cached.trace.guards, latency)
            return cached

    retrieve_started = time.perf_counter()
    try:
        result = retrieve(
            search_text, top_k=cfg.top_k, min_similarity=cfg.min_similarity, scheme_id=carried
        )
    except Exception as exc:
        latency["retrieve"] = _latency_ms(retrieve_started)
        answer = _error(exc, _trace(text, guards=guards, latency=latency))
        answer.trace.latency_ms = {**latency, "generate": 0, "total": _latency_ms(started)}
        _log(answer, guards, latency)
        return answer
    latency["retrieve"] = _latency_ms(retrieve_started)
    if history is not None:
        history.remember(text, result.resolved_scheme_id)

    guards = {
        "intent": "factual_or_unknown",
        "scheme_id": result.resolved_scheme_id,
        "filtered": result.filtered,
        "hits": len(result.hits),
        "top_score": round(result.top_score, 4),
    }
    if carried:
        guards["carried_scheme_id"] = carried
        guards["query_expanded"] = search_text != text

    # ---- threshold abstain: still no LLM call --------------------------------------
    if not result.passed:
        trace = _trace(
            text,
            guards=guards,
            hits=result.hits,
            threshold=result.threshold,
            threshold_passed=False,
            latency=latency,
            error="below_similarity_threshold",
        )
        answer = _insufficient(text, trace, result.resolved_scheme_id)
        answer.trace.latency_ms = {**latency, "generate": 0, "total": _latency_ms(started)}
        _log(answer, guards, latency)
        return answer

    # ---- 2-4. augment, generate, validate ------------------------------------------
    generate_started = time.perf_counter()
    try:
        prompt = build_prompt(
            result.hits,
            text,
            max(h.chunk.retrieved_at for h in result.hits) if result.hits else "",
        )
        raw, notes = _generate_with_fallback(prompt)
        answer = parse_answer(raw, result.hits)

        # One regeneration attempt, then demote. A validation failure is the model's fault,
        # so it is worth one more try; a provider exception is handled inside the helper.
        if answer.status != "answered" and (answer.trace.error or "").startswith(_RETRYABLE):
            raw, retry_notes = _generate_with_fallback(prompt)
            notes["llm_calls"] += retry_notes["llm_calls"]
            notes["regenerated"] = True
            answer = parse_answer(raw, result.hits)
    except Exception as exc:
        latency["generate"] = _latency_ms(generate_started)
        trace = _trace(
            text,
            guards=guards,
            hits=result.hits,
            threshold=result.threshold,
            threshold_passed=True,
            latency=latency,
        )
        answer = _error(exc, trace)
        answer.trace.latency_ms = {**latency, "generate": latency["generate"], "total": _latency_ms(started)}
        _log(answer, guards, latency)
        return answer

    latency["generate"] = _latency_ms(generate_started)
    guards["llm_calls"] = notes.get("llm_calls", 0)
    if notes.get("regenerated"):
        guards["regenerated"] = True
    if notes.get("fell_back_to_stub"):
        guards["fell_back_to_stub"] = True

    # ---- 5. attach the pipeline's trace and log -------------------------------------
    # `hits` is what retrieval *returned*; `cited_doc_ids` is what the answer actually used.
    # The generator's partial trace listed only the chunks it cited, and preferring it here
    # (`answer.trace.hits or ...`) made the trace and the "Why this answer?" panel show one
    # chunk where five were retrieved and weighed - hiding the evidence the model rejected.
    answer.trace = QueryTrace(
        query_hash=hash_query(text),
        guards=guards,
        hits=_hit_rows(result.hits),
        threshold=result.threshold,
        threshold_passed=answer.status == "answered",
        cited_doc_ids=answer.trace.cited_doc_ids,
        latency_ms={**latency, "generate": latency["generate"], "total": _latency_ms(started)},
        error=answer.trace.error,
    )
    if use_cache:
        _store_in_cache(text, answer)
    _log(answer, guards, latency)
    return answer


def _store_in_cache(text: str, answer: Answer) -> None:
    """Write-through for the demo cache. `put` refuses anything that is not `answered`, so
    refusals, PII rejections, insufficient-context demotions, and errors are never stored."""
    try:
        from src.llm.cache import AnswerCache

        AnswerCache().put(text, answer)
    except Exception:
        pass  # caching is a convenience; it must never change what the user is told


def _log(answer: Answer, guards: dict, latency: dict, pii_blocked: bool = False) -> None:
    """Append exactly one JSONL line per query, refusals included (FR-13).

    When an identifier was detected the query text is omitted entirely; the normalised
    hash is enough to correlate lines and is what makes that safe.
    """
    record = {
        "query_hash": answer.trace.query_hash,
        "status": answer.status,
        "guards": guards,
        "threshold": answer.trace.threshold,
        "threshold_passed": answer.trace.threshold_passed,
        "cited_doc_ids": answer.trace.cited_doc_ids,
        "latency_ms": answer.trace.latency_ms,
        "error": answer.trace.error,
    }
    if pii_blocked:
        record["query_redacted"] = True
    try:
        query_record(**record)
    except Exception:
        # Telemetry must never be the reason a user gets no answer.
        pass
