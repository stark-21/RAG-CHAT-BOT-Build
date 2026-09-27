"""P6 - Intent classification tests.

The six PRD 10.1 refusal phrasings are pinned individually, because that table is the
contract a reviewer checks by hand.
"""

from __future__ import annotations

import pytest

from src.guardrails import intent as intent_mod
from src.guardrails.intent import (
    OUT_OF_CORPUS_TOKENS,
    classify_intent,
    mentions_out_of_corpus,
    resolve_scheme,
)


def intent_of(query: str, **kw) -> str:
    return classify_intent(query, **kw)[0]


# ------------------------------------------------------------- PRD 10.1 phrasings


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Should I buy the small cap fund?", "advice"),
        ("Move HDFC Large Cap to Flexi Cap?", "advice"),
        ("Best fund for my child's education in 10 years?", "advice"),
        ("Which fund has the highest 5-yr return?", "performance"),
        ("What will this fund return next year?", "performance"),
        ("Expense ratio of Parag Parag Flexi Cap?", "out_of_scope"),
    ],
)
def test_prd_10_1_refusal_phrasings(query, expected):
    assert intent_of(query) == expected


# -------------------------------------------------------- acceptance: rule ordering


def test_performance_is_checked_before_advice():
    # Genuinely both. Performance must win so the refusal links the factsheet.
    assert intent_of("Should I buy the fund with the best returns?") == "performance"


def test_factual_and_out_of_scope_acceptance_cases():
    assert intent_of("What is the expense ratio of the HDFC Large Cap Fund?") == "factual"
    assert intent_of("What is the expense ratio of Parag Parag Flexi Cap?") == "out_of_scope"


def test_exit_load_question_is_factual_not_advice():
    # Guards a real trap: the corpus says "redeemed within 1 year", and treating
    # "redeem" as a sell verb would misclassify a factual question as advice.
    assert intent_of("What is the exit load if redeemed within 1 year?") == "factual"


def test_advice_examples():
    assert intent_of("Should I buy the small cap fund?") == "advice"
    assert intent_of("Can I buy the flexi cap fund?") == "advice"
    assert intent_of("Which fund is best fund for a 10 year goal?") == "advice"
    assert intent_of("Should I switch to balanced advantage?") == "advice"
    assert intent_of("Is it good for a long term goal?") == "advice"
    assert intent_of("Is SIP vs lump sum better?") == "advice"


def test_performance_examples():
    assert intent_of("Which fund has the best 5-year returns?") == "performance"
    assert intent_of("What is the CAGR of HDFC Large Cap?") == "performance"
    assert intent_of("Show me the ranking of these funds") == "performance"
    assert intent_of("What is the projected return?") == "performance"
    assert intent_of("Which fund did well last year?") == "performance"


def test_factual_examples():
    assert intent_of("What is the exit load?") == "factual"
    assert intent_of("What is the lock-in period for ELSS?") == "factual"
    assert intent_of("What is the minimum SIP amount?") == "factual"
    assert intent_of("What is the benchmark of HDFC Large Cap?") == "factual"
    assert intent_of("What is the riskometer category?") == "factual"
    assert intent_of("How do I download the CAS?") == "factual"
    assert intent_of("What is the AUM of HDFC Small Cap?") == "factual"
    assert intent_of("Is there a tax benefit?") == "factual"


def test_personal_data_beats_every_other_rule():
    assert intent_of("My PAN is ABCDE1234F") == "personal_data"
    assert intent_of("my holdings please") == "personal_data"
    assert intent_of("Show me my SIP details") == "personal_data"
    assert intent_of("what is the exit load on my account?") == "personal_data"


def test_pii_in_a_performance_question_still_triggers_personal_data():
    # Personal data is checked first precisely so identifiers never reach the log.
    assert intent_of("What were the returns? My PAN is ABCDE1234F") == "personal_data"


# ------------------------------------------------------------------ unknown / llm


def test_empty_query_is_unknown():
    assert intent_of("") == "unknown"
    assert intent_of("   ") == "unknown"


def test_bare_statement_with_no_vocabulary_is_unknown():
    verdict, evidence = classify_intent("hdfc")
    assert verdict == "unknown"
    assert evidence["rule"] == "no_rule_matched"


def test_use_llm_falls_back_safely_when_provider_is_absent(monkeypatch):
    monkeypatch.setattr(intent_mod, "_llm_fallback", lambda q: ("unknown", {"rule": "stubbed"}))
    assert intent_of("hdfc", use_llm=True) == "unknown"


def test_llm_fallback_never_overturns_a_matched_rule(monkeypatch):
    def explode(query):
        raise AssertionError("LLM must not be consulted when a rule already fired")

    monkeypatch.setattr(intent_mod, "_llm_fallback", explode)
    assert intent_of("What is the exit load?", use_llm=True) == "factual"
    assert intent_of("Should I buy this fund?", use_llm=True) == "advice"


def test_llm_fallback_defaults_to_unknown_without_p5(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "src.llm.provider":
            raise ImportError("P5 not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    verdict, evidence = intent_mod._llm_fallback("something unclassifiable")
    assert verdict == "unknown"
    assert evidence["rule"] == "llm_unavailable"


# ----------------------------------------------------------- out-of-corpus tokens


@pytest.mark.parametrize("token", OUT_OF_CORPUS_TOKENS)
def test_every_out_of_corpus_token_is_detected(token):
    assert mentions_out_of_corpus(f"What is the expense ratio of {token} fund?") is True


def test_out_of_corpus_needs_word_boundaries():
    assert mentions_out_of_corpus("What is the SBI rate?") is True
    assert mentions_out_of_corpus("principally speaking, no") is False
    assert mentions_out_of_corpus("What is the HDFC Large Cap expense ratio?") is False


# ---------------------------------------------------------------------- evidence


def test_evidence_names_the_rule_that_fired():
    verdict, evidence = classify_intent("Which fund has the best 5-yr return?")
    assert verdict == "performance"
    assert evidence["rule"] == "performance_term"
    assert evidence["matched"] == "return"


def test_evidence_never_echoes_the_query():
    _, evidence = classify_intent("My PAN is ABCDE1234F and my holdings are X")
    serialised = repr(evidence)
    assert "ABCDE1234F" not in serialised
    assert evidence["types"] == ["pan"]


@pytest.mark.parametrize("verdict", ["factual", "advice", "FACTUAL", " advice "])
def test_llm_may_only_choose_between_factual_and_advice(monkeypatch, verdict):
    class Provider:
        def classify(self, prompt):
            return verdict

        def generate(self, prompt, *, temperature, max_tokens):  # pragma: no cover
            return ""

    monkeypatch.setattr("src.llm.provider.get_provider", lambda *a, **k: Provider())
    got, evidence = intent_mod._llm_fallback("something unclassifiable")

    assert got == verdict.strip().lower()
    assert evidence["rule"] == "llm_classifier"


@pytest.mark.parametrize(
    "verdict",
    ["out_of_scope", "personal_data", "performance", "answered", "", None, "I am a helpful assistant"],
)
def test_llm_cannot_unilaterally_widen_a_refusal_decision(monkeypatch, verdict):
    """A model must not be able to talk its way past a safety decision."""
    class Provider:
        def classify(self, prompt):
            return verdict

        def generate(self, prompt, *, temperature, max_tokens):  # pragma: no cover
            return ""

    monkeypatch.setattr("src.llm.provider.get_provider", lambda *a, **k: Provider())
    got, _evidence = intent_mod._llm_fallback("something unclassifiable")

    assert got == "unknown"


def test_llm_fallback_works_against_the_real_stub_provider():
    """P5 now exists, so the fallback must reach it and get an honest "unknown"."""
    got, evidence = intent_mod._llm_fallback("hdfc")

    assert got == "unknown"
    assert evidence["rule"] == "llm_classifier"


def test_resolve_scheme_is_re_exported_for_the_pipeline():
    assert resolve_scheme("hdfc elss tax saver") == "S3"
    assert resolve_scheme("tell me about quant funds") is None


# --- Personal-data rule: document obtain vs. the user's own holdings ---------------------


def test_asking_how_to_download_a_statement_is_not_a_personal_data_request():
    """PRD section 12 Q6. "How do I download my capital-gains statement?" asks where a
    document type comes from, and the corpus answers it with the CAMS/KFintech links. The
    `my ... statement` rule used to swallow it and refuse a question the PRD requires to be
    answered."""
    intent, _evidence = classify_intent("How do I download my capital-gains statement?")

    assert intent == "factual"


@pytest.mark.parametrize(
    "query",
    [
        "How do I download my capital-gains statement?",
        "How do I get my statement from CAMS?",
        "Where can I download my statement?",
    ],
)
def test_document_obtain_phrasings_stay_factual(query: str):
    assert classify_intent(query)[0] == "factual"


@pytest.mark.parametrize(
    "query",
    [
        "Show me my holdings",
        "What is my account balance?",
        "What is my NAV?",
        "What is in my portfolio?",
        "List my folios",
        "What are my investments?",
        "Show me my SIP details",
    ],
)
def test_asking_about_my_own_data_still_refuses(query: str):
    """The document carve-out must not weaken the rule it was carved out of."""
    assert classify_intent(query)[0] == "personal_data"


def test_an_explicit_identifier_still_refuses_inside_a_download_phrasing():
    """`my_account_data` is skipped for document questions, but an identifier is not a
    document type, so "download my PAN" must still be treated as personal data."""
    intent, evidence = classify_intent("Where can I download my PAN card?")

    assert intent == "personal_data"
    assert evidence.get("matched") == "my_identifier"


def test_plural_account_nouns_are_matched():
    """`transaction` without an optional plural could never match "transactions", because
    `\\b` does not fall between "transaction" and the trailing "s"."""
    intent, _evidence = classify_intent("Can you check my transactions?")

    assert intent == "personal_data"
