"""P6 - Refusal builder tests, plus the PII acceptance criteria and the anti-drift guard
that keeps `observability` redaction aligned with the guardrail patterns."""

from __future__ import annotations

import re

import pytest

from src import observability
from src.config import get_settings, reset_settings_cache
from src.guardrails import pii
from src.guardrails.pii import detect_pii, pattern_signature, redact
from src.guardrails.refusal import DISCLAIMER, EDUCATIONAL_LINKS, build_refusal
from src.guardrails.intent import Intent
from src.ingest.sources import load_registry
from src.types import Answer

ALL_INTENTS = ("factual", "advice", "performance", "out_of_scope", "personal_data", "unknown")


# ------------------------------------------------------------------ PII (FR-9)


def test_detect_pii_returns_the_pan_type_and_never_the_value():
    found = detect_pii("My PAN is ABCDE1234F")

    assert found == ["pan"]
    assert "ABCDE1234F" not in repr(found)


def test_detect_pii_finds_each_type():
    assert "pan" in detect_pii("PAN ABCDE1234F")
    assert "aadhaar" in detect_pii("aadhaar 2345 6789 0123")
    assert "email" in detect_pii("write to me at ramesh.kumar@example.com")
    assert "phone" in detect_pii("call me on 9876543210")
    assert "phone" in detect_pii("call me on +91 98765 43210")
    assert "account" in detect_pii("account number 123456789012")
    assert "otp" in detect_pii("my otp is 482913")


def test_detect_pii_reports_types_only_and_is_deduplicated():
    found = detect_pii("PAN ABCDE1234F and again ABCDE1234F")
    assert found == ["pan"]


def test_aadhaar_and_phone_are_not_also_reported_as_account():
    # A 10-digit phone and a 12-digit Aadhaar are both long digit runs; reporting the
    # vague `account` type alongside them would be noise.
    assert detect_pii("9876543210") == ["phone"]
    assert detect_pii("234567890123") == ["aadhaar"]


def test_detect_pii_on_clean_text():
    assert detect_pii("What is the exit load on HDFC Large Cap Fund?") == []
    assert detect_pii("") == []


def test_redact_masks_every_type_in_one_string():
    text = (
        "PAN ABCDE1234F, aadhaar 2345 6789 0123, email me at ramesh@example.com, "
        "phone 9876543210, account 123456789012, otp 482913"
    )
    out = redact(text)

    for secret in (
        "ABCDE1234F",
        "2345 6789 0123",
        "ramesh@example.com",
        "9876543210",
        "123456789012",
        "482913",
    ):
        assert secret not in out, secret
    assert out.count("[REDACTED]") >= 6


def test_spaced_and_prefixed_phone_forms_are_detected_and_masked():
    for text in (
        "call me on 9876543210",
        "call me on +91 98765 43210",
        "call me on 98765-43210",
        "+919876543210",
    ):
        assert "phone" in detect_pii(text), text
        assert "987654321" not in redact(text), text


def test_a_twelve_digit_run_is_never_mistaken_for_a_phone():
    # Aadhaar numbers starting 6-9 are 12 digits; the phone pattern must not match a
    # 10-digit window inside one, or every Aadhaar would also report as `phone`.
    for number in ("6789 0123 4567", "987654321098", "612345678901"):
        found = detect_pii(number)
        assert "phone" not in found, (number, found)


def test_date_like_numbers_are_not_pii():
    assert detect_pii("as on 08 May 2015") == []
    assert detect_pii("NAV of 12.3456 on 01 Jan 2026") == []


def test_redact_handles_empty_input():
    assert redact("") == ""


# --------------------------------------------- observability / pii anti-drift guard


def test_guardrail_modules_import_cleanly_in_any_order():
    """`observability` imports `redact` from `guardrails.pii`, so populating
    `guardrails/__init__.py` can easily close the cycle
    `observability -> guardrails -> guardrails.refusal -> observability`.
    Import cycles are order-dependent, so each order is checked in a fresh interpreter."""
    import subprocess
    import sys

    orders = [
        "import src.observability, src.guardrails, src.guardrails.refusal",
        "import src.guardrails.refusal, src.observability",
        "import src.guardrails, src.observability, src.config",
        "import src.retrieval.retriever, src.guardrails, src.observability",
    ]
    for statement in orders:
        result = subprocess.run(
            [sys.executable, "-c", statement],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{statement}\n{result.stderr}"


def test_observability_uses_the_guardrail_patterns():
    """P0 created a duplicated copy of these patterns in observability; it now imports
    `redact` from here. This asserts the two cannot drift apart again."""
    assert observability.redact is pii.redact
    assert pattern_signature(), "pattern signature must be non-empty"
    assert all(":" in sig for sig in pattern_signature())


def test_observability_redacts_nested_log_records(tmp_path):
    record = {"query": "my pan is ABCDE1234F", "hits": [{"text": "call 9876543210"}]}
    path = tmp_path / "log.jsonl"

    observability.log_jsonl(path, record)
    written = path.read_text(encoding="utf-8")

    assert "ABCDE1234F" not in written
    assert "9876543210" not in written
    assert "[REDACTED]" in written


# --------------------------------------------------------------- build_refusal


def test_build_refusal_returns_a_complete_answer():
    answer = build_refusal("advice", query="Should I buy this fund?")

    assert isinstance(answer, Answer)
    assert answer.status == "refused"
    assert answer.text
    assert answer.citations, "a refusal must still cite an educational link"
    assert answer.last_updated is None, "a refusal is backed by no indexed document"
    assert answer.trace.guards["intent"] == "advice"
    assert answer.trace.hits == []
    assert answer.trace.threshold_passed is False


@pytest.mark.parametrize("intent", ALL_INTENTS)
def test_every_intent_has_a_disclaimer_and_a_non_empty_link(intent):
    answer = build_refusal(intent)

    assert DISCLAIMER in answer.text
    assert answer.text.strip()
    assert len(answer.citations) == 1
    assert answer.citations[0].url.startswith("http")
    assert answer.citations[0].label
    assert EDUCATIONAL_LINKS[intent].startswith("http")


def test_personal_data_refusal_uses_the_pii_rejected_status():
    answer = build_refusal("personal_data", query="My PAN is ABCDE1234F", pii_types=["pan"])

    assert answer.status == "pii_rejected"
    assert answer.trace.guards["pii_types"] == ["pan"]


def test_personal_data_refusal_never_echoes_the_identifier():
    answer = build_refusal("personal_data", query="My PAN is ABCDE1234F", pii_types=["pan"])

    assert "ABCDE1234F" not in answer.text
    assert "ABCDE1234F" not in repr(answer.citations)


def test_performance_refusal_quotes_no_figures():
    answer = build_refusal("performance", scheme_id="S1", query="best returns?")

    # No percentages, decimals or multi-digit figures. A lone digit is fine: the
    # disclaimer legitimately says "5 HDFC schemes".
    assert "%" not in answer.text
    assert not re.search(r"\d+\.\d+", answer.text), answer.text
    assert not re.search(r"\b\d{2,}\b", answer.text), answer.text


def test_a_resolved_scheme_upgrades_the_link_to_its_own_page():
    spec = load_registry().scheme("S1")
    answer = build_refusal("performance", scheme_id="S1", query="returns?")

    assert answer.citations[0].url == spec.entry_url
    assert answer.trace.guards["scheme_id"] == "S1"


def test_out_of_scope_never_links_a_scheme_page_even_when_one_resolves():
    """"Expense ratio of Parag Parag Flexi Cap?" resolves S2 via the "flexi cap" alias.
    Linking the HDFC Flexi Cap page would be the exact substitution this guard prevents."""
    from src.guardrails.intent import classify_intent, resolve_scheme

    query = "Expense ratio of Parag Parag Flexi Cap?"
    assert classify_intent(query)[0] == "out_of_scope"
    resolved = resolve_scheme(query)
    assert resolved == "S2"

    answer = build_refusal("out_of_scope", scheme_id=resolved, query=query)

    assert answer.citations[0].url == EDUCATIONAL_LINKS["out_of_scope"]
    assert "hdfc" not in answer.citations[0].url.lower()


def test_personal_data_ignores_a_resolved_scheme_and_links_the_statement_source():
    answer = build_refusal("personal_data", scheme_id="S1", query="my holdings")

    assert answer.citations[0].url.endswith("consolidated_account_statement.html")


def test_an_unknown_scheme_falls_back_to_the_default_link():
    answer = build_refusal("performance", scheme_id="S99", query="returns?")

    assert answer.citations[0].url == EDUCATIONAL_LINKS["performance"]


def test_refusal_links_are_reachable_urls_present_in_the_corpus_sources():
    """Every link must be a source we actually ingested, so no refusal sends a user to a
    URL this project invented."""
    from src.ingest.build_index import read_chunks

    known = {c.source_url for c in read_chunks()} | {
        s.entry_url for s in load_registry().schemes
    }
    for intent, url in EDUCATIONAL_LINKS.items():
        assert url in known, f"{intent} -> {url} is not a verified source URL"


def test_disclaimer_setting_is_surfaced_alongside_the_canonical_disclaimer(monkeypatch):
    reset_settings_cache()
    monkeypatch.setenv("DISCLAIMER", "Demo build. Not investment advice.")
    reset_settings_cache()
    try:
        text = build_refusal("advice").text
        assert DISCLAIMER in text
        assert "Demo build. Not investment advice." in text
    finally:
        reset_settings_cache()


def test_a_refusal_never_claims_a_fresh_verification():
    """`last_updated` is the retrieval date of the document behind the answer. A refusal has
    none, so it must be None rather than the current time."""
    answer = build_refusal("advice", query="Should I buy this?")

    assert answer.last_updated is None
    assert not answer.text.startswith("Last updated from sources:")


def test_evidence_is_carried_into_the_trace():
    answer = build_refusal("advice", evidence={"rule": "advice_term", "matched": "should_i"})
    assert answer.trace.guards["rule"] == "advice_term"
