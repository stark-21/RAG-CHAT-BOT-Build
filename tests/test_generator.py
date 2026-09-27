"""P5 - Generator and L6 output-validation tests.

These are the tests that matter most for trust: they assert the assistant cannot emit a
number or a URL that is not in the retrieved corpus.
"""

from __future__ import annotations

import pytest

from src.config import get_settings, reset_settings_cache
from src.llm.generator import (
    LAST_UPDATED_PREFIX,
    cap_sentences,
    extract_urls,
    generate,
    numbers_in,
    parse_answer,
)
from src.llm.prompt_builder import build_prompt
from src.llm.provider import ProviderError, StubProvider, get_provider
from src.types import Chunk, RetrievedChunk

URL_S1 = "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"
URL_S3 = "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth"

CONTEXT_TEXT = (
    "HDFC Large Cap Fund - Direct - Growth - Exit load\n"
    "Exit load of 1% if redeemed within 1 year.\n"
    "Exit load of 0.5% if redeemed after 1 year but before 18 months."
)


def hit(
    chunk_id: str = "S1__overview__0012",
    *,
    scheme_id: str = "S1",
    text: str = CONTEXT_TEXT,
    url: str = URL_S1,
    section: str = "Exit load",
    retrieved_at: str = "2026-09-27",
    score: float = 0.83,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=text,
            token_count=42,
            scheme_id=scheme_id,
            scheme_name="HDFC Large Cap Fund - Direct - Growth",
            category="Large cap",
            doc_type="scheme_page",
            section=section,
            source_url=url,
            publisher="HDFC AMC (via Groww)",
            retrieved_at=retrieved_at,
            chunk_index=12,
            splitter="LabelValueSplitter",
            content_hash="abc123",
        ),
        score=score,
    )


# --------------------------------------------------------------------- extract_urls


def test_extract_urls_finds_and_dedupes():
    text = f"See {URL_S1} and also {URL_S1}. Details at {URL_S3}."
    assert extract_urls(text) == [URL_S1, URL_S3]


def test_extract_urls_strips_trailing_punctuation():
    assert extract_urls(f"source: {URL_S1},") == [URL_S1]
    assert extract_urls(f"({URL_S1})") == [URL_S1]


def test_extract_urls_on_text_without_urls():
    assert extract_urls("no links here") == []
    assert extract_urls("") == []


# ----------------------------------------------------------------------- numbers_in


def test_numbers_in_normalises_currency_and_separators():
    assert numbers_in("Rs. 1,00,000") == {"100000"}
    assert numbers_in("INR 100000") == {"100000"}
    assert numbers_in("₹1,00,000") == {"100000"}
    assert numbers_in("1,00,000") == {"100000"}


def test_numbers_in_strips_a_trailing_percent():
    assert numbers_in("18.5%") == {"18.5"}
    assert numbers_in("0.005 %") == {"0.005"}


def test_numbers_in_splits_a_date_into_its_parts():
    assert numbers_in("2026-09-27") == {"2026", "09", "27"}


def test_numbers_in_keeps_decimals_intact():
    assert numbers_in("expense ratio is 1.05 percent") == {"1.05"}


def test_numbers_in_on_text_without_numbers():
    assert numbers_in("no digits here") == set()


# ------------------------------------------------------------------- cap_sentences


def test_cap_sentences_keeps_the_first_n():
    text = "One. Two. Three. Four. Five. Six."
    assert cap_sentences(text, 3) == "One. Two. Three."


def test_cap_sentences_leaves_short_text_alone():
    assert cap_sentences("Only one sentence.", 3) == "Only one sentence."


def test_cap_sentences_does_not_split_on_decimals():
    text = "The charge is 0.005% of the investment. That is small. Third sentence here."
    capped = cap_sentences(text, 2)
    assert "0.005%" in capped
    assert capped.count(".") >= 2


def test_cap_sentences_does_not_split_on_currency_abbreviations():
    text = "The minimum is Rs. 500 per month. It can be doubled. Third one."
    capped = cap_sentences(text, 2)
    assert "Rs. 500" in capped


def test_cap_sentences_rejects_a_useless_limit():
    with pytest.raises(ValueError):
        cap_sentences("One. Two.", 0)


# ------------------------------------------------------------------ L6: citations


def test_an_invented_url_is_flagged_and_demoted():
    raw = f"The exit load is 1%. See https://invented.example.com/fake for details."
    answer = parse_answer(raw, [hit()])

    assert answer.status == "insufficient_context"
    assert answer.trace.error.startswith("citation_not_in_context")
    assert "invented.example.com" in answer.trace.error


def test_a_context_url_is_accepted():
    raw = f"The exit load is 1% if redeemed within 1 year. source: {URL_S1}"
    answer = parse_answer(raw, [hit()])

    assert answer.status == "answered"
    assert answer.trace.error is None


def test_an_answer_with_no_citation_is_demoted():
    answer = parse_answer("The exit load is 1%.", [hit()])

    assert answer.status == "insufficient_context"
    assert answer.trace.error == "no_citation_in_context"


def test_an_empty_response_is_demoted():
    answer = parse_answer("   ", [hit()])
    assert answer.status == "insufficient_context"
    assert answer.trace.error == "empty_response"


# -------------------------------------------------------------------- L6: numbers


def test_a_novel_number_is_demoted_to_insufficient_context():
    """Acceptance: "18.5%" with no such number in context -> insufficient_context."""
    raw = f"The fund returned 18.5% last year. source: {URL_S1}"
    answer = parse_answer(raw, [hit()])

    assert answer.status == "insufficient_context"
    assert answer.trace.error.startswith("novel_number")
    assert "18.5" in answer.trace.error


def test_an_invented_expense_ratio_is_demoted():
    raw = f"The expense ratio is 0.42%. source: {URL_S1}"
    answer = parse_answer(raw, [hit()])

    assert answer.status == "insufficient_context"
    assert answer.trace.error.startswith("novel_number")


def test_numbers_from_the_context_are_allowed():
    raw = f"Exit load is 1% within 1 year, and 0.5% between 1 and 18 months. source: {URL_S1}"
    assert parse_answer(raw, [hit()]).status == "answered"


def test_a_reformatted_context_number_is_allowed():
    # 0.005 in the context, written as 0.005% in the answer.
    hits = [hit(text="Stamp duty on investment: 0.005% (from July 1st, 2020).")]
    raw = f"Stamp duty is 0.005%. source: {URL_S1}"
    assert parse_answer(raw, hits).status == "answered"


# ---------------------------------------------------------------- L6: sentence cap


def test_a_six_sentence_answer_is_capped_and_keeps_its_footer():
    raw = (
        f"One fact. Two facts. Three facts. Four facts. Five facts. Six facts. source: {URL_S1}"
    )
    answer = parse_answer(raw, [hit()])

    assert answer.status == "answered"
    body = answer.text.split(LAST_UPDATED_PREFIX)[0]
    assert body.count(".") <= 4  # three sentences, each ending in a period
    assert "Four facts" not in body
    assert "Six facts" not in body
    assert LAST_UPDATED_PREFIX in answer.text
    assert "2026-09-27" in answer.text
    assert URL_S1 in answer.text


def test_trimming_never_strips_the_citation():
    # The URL sits in the last sentence, which the cap removes.
    raw = f"One. Two. Three. Four. Five. source: {URL_S1}"
    answer = parse_answer(raw, [hit()])

    assert answer.status == "answered"
    assert URL_S1 in answer.text
    assert LAST_UPDATED_PREFIX in answer.text


def test_max_sentences_comes_from_settings():
    reset_settings_cache()
    assert get_settings().max_sentences == 3


# --------------------------------------------------------------- last_updated rule


def test_last_updated_is_the_newest_cited_source_not_the_newest_hit():
    old = hit("old", retrieved_at="2026-01-01", score=0.9, url=URL_S1)
    new = hit("new", retrieved_at="2026-09-27", score=0.5, url=URL_S3)

    raw = f"Exit load is 1%. source: {URL_S1}"
    answer = parse_answer(raw, [old, new])

    assert answer.last_updated == "2026-01-01", "must follow the cited doc, not the top hit"


def test_last_updated_picks_the_newest_among_several_cited_docs():
    old = hit("old", retrieved_at="2026-01-01", url=URL_S1)
    new = hit("new", retrieved_at="2026-09-27", url=URL_S3)

    raw = f"Exit load is 1%. sources: {URL_S1} and {URL_S3}"
    answer = parse_answer(raw, [old, new])

    assert answer.last_updated == "2026-09-27"


def test_citations_are_deduplicated_and_carry_provenance():
    old = hit("old", retrieved_at="2026-01-01", url=URL_S1)
    answer = parse_answer(f"Exit load is 1%. source: {URL_S1}", [old, hit("dupe", url=URL_S1)])

    # One citation, because the URL is the citation. Every chunk from a cited document is
    # still listed as supporting evidence, in the score order the caller supplied.
    assert len(answer.citations) == 1
    assert answer.citations[0].url == URL_S1
    assert answer.citations[0].retrieved_at == "2026-01-01"
    assert answer.trace.cited_doc_ids == ["old", "dupe"]


# ---------------------------------------------------------------------- providers


def test_get_provider_stub_needs_no_key_and_no_network():
    reset_settings_cache()
    provider = get_provider()

    assert isinstance(provider, StubProvider)
    assert provider.needs_network is False


def test_stub_quotes_the_fact_not_the_repeated_heading():
    """Chunk text repeats its heading on line 1; quoting that answers nothing."""
    prompt = build_prompt([hit()], "What is the exit load?", "2026-09-27")
    out = StubProvider().generate(prompt, temperature=0.0, max_tokens=300)

    assert "Exit load of 1% if redeemed within 1 year." in out
    assert out.count("HDFC Large Cap Fund - Direct - Growth - Exit load") == 0


def test_stub_output_is_labelled_as_an_extract():
    prompt = build_prompt([hit()], "What is the exit load?", "2026-09-27")
    out = StubProvider().generate(prompt, temperature=0.0, max_tokens=300)

    assert out.startswith(StubProvider.LABEL)
    assert "no LLM available" in out
    assert URL_S1 in out


def test_stub_survives_prompt_with_no_context():
    prompt = build_prompt([], "What is the exit load?", "2026-09-27")
    out = StubProvider().generate(prompt, temperature=0.0, max_tokens=300)

    assert out.startswith(StubProvider.LABEL)
    assert "no context" in out


def test_stub_classify_does_not_invent_a_verdict():
    assert StubProvider().classify("Should I buy this?") == "unknown"


def test_generate_uses_the_injected_provider_at_zero_temperature():
    seen: dict = {}

    class Recorder:
        def generate(self, prompt, *, temperature, max_tokens):
            seen.update(temperature=temperature, max_tokens=max_tokens)
            return f"Exit load is 1%. source: {URL_S1}"

        def classify(self, prompt):
            return "unknown"

    prompt = build_prompt([hit()], "What is the exit load?", "2026-09-27")
    out = generate(prompt, provider=Recorder())

    assert seen["temperature"] == get_settings().temperature
    assert seen["max_tokens"] == 300
    assert parse_answer(out, [hit()]).status == "answered"


def test_get_provider_rejects_an_unknown_provider():
    from src.config import get_settings as gs
    from dataclasses import replace

    reset_settings_cache()
    bogus = replace(gs(), llm_provider="banana")
    with pytest.raises(ProviderError):
        get_provider(bogus)


def test_hosted_provider_requires_a_key():
    from src.llm.provider import HostedProvider

    with pytest.raises(ProviderError):
        HostedProvider("", "gpt-4o-mini")


def test_hosted_provider_never_reveals_its_key():
    from src.llm.provider import HostedProvider

    provider = HostedProvider("sk-super-secret", "gpt-4o-mini")
    assert "sk-super-secret" not in repr(provider)
    assert "***" in repr(provider)


def test_unknown_llm_provider_is_rejected_by_config_validation(monkeypatch):
    reset_settings_cache()
    monkeypatch.setenv("LLM_PROVIDER", "banana")
    reset_settings_cache()
    try:
        with pytest.raises(Exception):
            get_settings()
    finally:
        reset_settings_cache()


class TestCaptionLineStaysOnItsOwnLine:
    """A provider citation line is provenance, not prose. Joining it with a space produced
    "... from the date of investment. source: https://..." - a mangled sentence with a raw
    URL stranded in the middle of the answer."""

    def test_the_caption_keeps_its_own_line(self):
        out = cap_sentences("Exit load is 1%.\nsource: https://example.invalid/a", 3)
        assert out == "Exit load is 1%.\nsource: https://example.invalid/a"

    def test_ordinary_prose_is_still_joined_into_one_paragraph(self):
        out = cap_sentences("Exit load is 1%. It applies within one year.", 3)
        assert out == "Exit load is 1%. It applies within one year."

    def test_the_cap_still_counts_the_caption_as_one_sentence(self):
        assert len(cap_sentences("Exit load is 1%.\nsource: https://example.invalid/a", 3).split()) > 0
        assert cap_sentences("Exit load is 1%.\nsource: https://example.invalid/a", 1) == "Exit load is 1%."
