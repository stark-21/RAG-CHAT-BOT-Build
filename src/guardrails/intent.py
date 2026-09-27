"""P6 - Intent classification and out-of-corpus detection (PRD FR-7, FR-9).

Everything here is deterministic regex vocabulary. That is a deliberate choice for a
facts-only assistant: the guard that decides whether to refuse must be reproducible and
inspectable, and must cost nothing. An LLM is only consulted for the residue
(`use_llm=True`), never to overturn a rule that already fired.

Layering matters and is not arbitrary:

    personal_data -> performance -> advice -> out_of_scope -> factual -> unknown

`performance` is checked *before* `advice` because "should I buy the fund with the best
returns" is genuinely both, and the correct answer is a performance refusal that links
the factsheet rather than generic advice-speak.
"""

from __future__ import annotations

import re
from typing import Literal

from src.guardrails.pii import detect_pii
from src.ingest.sources import resolve_scheme

Intent = Literal[
    "factual",
    "advice",
    "performance",
    "out_of_scope",
    "personal_data",
    "unknown",
]

__all__ = [
    "Intent",
    "ADVICE_TERMS",
    "FACTUAL_TERMS",
    "OUT_OF_CORPUS_TOKENS",
    "PERFORMANCE_TERMS",
    "classify_intent",
    "mentions_out_of_corpus",
    "resolve_scheme",
]

# Return performance or a projection of it. No digits or percentages are ever produced
# here; a refusal points at the official factsheet instead.
PERFORMANCE_TERMS: list[tuple[str, re.Pattern[str]]] = [
    ("return", re.compile(r"\breturns?\b", re.IGNORECASE)),
    ("cagr", re.compile(r"\bcagr\b", re.IGNORECASE)),
    ("best_performing", re.compile(r"\bbest[\s-]performing\b", re.IGNORECASE)),
    ("top_performing", re.compile(r"\btop[\s-]performing\b", re.IGNORECASE)),
    ("ranking", re.compile(r"\brank(?:ing|s)?\b", re.IGNORECASE)),
    ("alpha", re.compile(r"\balpha\b", re.IGNORECASE)),
    ("which_fund_did_well", re.compile(r"\bwhich fund did well\b", re.IGNORECASE)),
    ("highest", re.compile(r"\bhighest\b|\blowest\b", re.IGNORECASE)),
    (
        "will_produce",
        re.compile(
            r"\bwill\b[^?]{0,40}?\b(?:give|produce|deliver|return|earn|grow)\b",
            re.IGNORECASE,
        ),
    ),
    ("projected", re.compile(r"\bproject(?:ed|ion|ions)\b", re.IGNORECASE)),
]

# Asking for a recommendation, or for an action on a portfolio.
ADVICE_TERMS: list[tuple[str, re.Pattern[str]]] = [
    ("should_i", re.compile(r"\bshould (?:i|we|you)\b", re.IGNORECASE)),
    ("recommend", re.compile(r"\brecommend(?:ed|ation)?\b", re.IGNORECASE)),
    ("best_fund_for", re.compile(r"\bbest fund for\b", re.IGNORECASE)),
    ("which_should_i", re.compile(r"\bwhich should i\b", re.IGNORECASE)),
    ("switch_to", re.compile(r"\bswitch(?:ing)? to\b", re.IGNORECASE)),
    ("allocate", re.compile(r"\ballocat(?:e|ion|ing)\b", re.IGNORECASE)),
    ("is_it_good_for", re.compile(r"\bis it good for\b", re.IGNORECASE)),
    ("worth_buying", re.compile(r"\bworth (?:buying|investing)\b", re.IGNORECASE)),
    ("sip_vs_lump_sum", re.compile(r"\blump[\s-]?sum\b", re.IGNORECASE)),
    ("buy_sell", re.compile(r"\b(?:buy|purchase|sell)\b", re.IGNORECASE)),
    # "Move HDFC Large Cap to Flexi Cap?" - PRD 10.1 switch/portfolio phrasing. Kept
    # narrow: a move verb followed by "to" within the same clause.
    (
        "move_to",
        re.compile(r"\b(?:move|switch|transfer|shift|rotate)\b[^?]{0,40}?\bto\b", re.IGNORECASE),
    ),
]

# "my <account thing>" - the assistant has no access to any user's holdings. A bare
# "show me" is deliberately NOT a signal: "show me the ranking of these funds" is a
# performance question, and the possessive form already covers "show me my SIP".
PERSONAL_DATA_TERMS: list[tuple[str, re.Pattern[str]]] = [
    (
        "my_account_data",
        re.compile(
            r"\bmy\b[^?]{0,40}?\b(?:sips?|holdings?|accounts?|statements?|portfolios?|folios?|"
            r"nav|transactions?|investments?)\b",
            re.IGNORECASE,
        ),
    ),
    ("my_identifier", re.compile(r"\bmy\b[^?]{0,20}?\b(?:pan|aadhaar|email|phone|number)\b", re.IGNORECASE)),
]

# Vocabulary that marks a question this corpus can actually answer.
FACTUAL_TERMS: list[tuple[str, re.Pattern[str]]] = [
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in [
        ("expense_ratio", r"\bexpense ratio\b"),
        ("exit_load", r"\bexit load\b"),
        ("lock_in", r"\block[\s-]?in\b"),
        ("minimum_sip", r"\bminimum sip\b"),
        ("benchmark", r"\bbenchmark\b"),
        ("riskometer", r"\briskometer\b"),
        ("statement", r"\bstatement\b"),
        ("tax", r"\btax\b"),
        ("download", r"\bdownload\b"),
        ("nav", r"\bnav\b"),
        ("aum", r"\baum\b"),
        ("charges", r"\bcharges\b"),
        ("fact_sheet", r"\bfact[\s-]?sheet\b"),
    ]
]

# AMCs and fund families outside the 5-scheme corpus. Asking about one of these must be
# refused as out of scope rather than silently answered from the nearest HDFC scheme,
# which is the failure mode that makes a scoped assistant dangerous.
OUT_OF_CORPUS_TOKENS: tuple[str, ...] = (
    "parag parag",
    "parag",
    "axis",
    "kotak",
    "kotak mahindra",
    "motilal",
    "nippon",
    "icici",
    "sbi",
    "mirae",
    "quant",
    "canara",
    "principal",
)

_OUT_OF_CORPUS_RE = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(re.escape(t) for t in OUT_OF_CORPUS_TOKENS) + r")(?![a-z0-9])",
    re.IGNORECASE,
)

_QUESTION_MARK = re.compile(r"\?")


def mentions_out_of_corpus(query: str) -> bool:
    """True when the query names an AMC or fund family this demo does not cover."""
    return bool(_OUT_OF_CORPUS_RE.search(query or ""))


def _first_match(query: str, terms: list[tuple[str, re.Pattern[str]]]) -> str | None:
    for name, pattern in terms:
        if pattern.search(query):
            return name
    return None


# "How do I download my capital-gains statement?" asks where a document type comes from,
# which the corpus answers with the CAMS/KFintech links. It is not a request for the
# user's own holdings, so the `my ... statement` rule must not swallow it. PRD section 12
# Q6 is exactly this question and is required to be answered.
_DOCUMENT_OBTAIN = re.compile(
    r"\b(?:download|obtain|fetch|generate|request|get|where\s+(?:do|can)\s+i)\b", re.IGNORECASE
)
_DOCUMENT_NOUN = re.compile(r"\b(?:statement|statement[s]?|report|passbook)\b", re.IGNORECASE)


def _is_document_obtain_question(query: str) -> bool:
    return bool(_DOCUMENT_OBTAIN.search(query) and _DOCUMENT_NOUN.search(query))


def _personal_data_rule(query: str) -> str | None:
    """`my_account_data`, unless the question is about obtaining a document.

    An explicit identifier ("my PAN", "my Aadhaar number") always wins, so a genuine
    "download my PAN" phrasing still refuses.
    """
    identifier = next((n for n, p in PERSONAL_DATA_TERMS if n == "my_identifier"), None)
    for name, pattern in PERSONAL_DATA_TERMS:
        if name == "my_account_data" and _is_document_obtain_question(query):
            continue
        if pattern.search(query):
            return name
    return identifier if identifier and any(
        p.search(query) for n, p in PERSONAL_DATA_TERMS if n == "my_identifier"
    ) else None


def _llm_fallback(query: str) -> tuple[Intent, dict]:
    """Ask the configured provider to split `unknown` into factual vs advice.

    P5 owns the provider, so this imports lazily and degrades to `unknown` rather than
    failing the request when P5 is absent or the provider is a stub. Refusing to guess is
    the right failure mode: a wrong `factual` verdict would let a recommendation through.
    """
    try:
        from src.llm.provider import get_provider  # noqa: PLC0415 - optional, P5 dependency
    except Exception:
        return "unknown", {"rule": "llm_unavailable", "note": "provider not installed"}

    try:
        verdict = get_provider().classify(query)
    except Exception as exc:  # pragma: no cover - provider-specific
        return "unknown", {"rule": "llm_error", "error": type(exc).__name__}

    verdict = (verdict or "").strip().lower()
    # Only `factual` and `advice` are delegable. `performance`, `out_of_scope` and
    # `personal_data` are safety decisions that must stay with the deterministic rules: a
    # model must not be able to talk its way past a refusal.
    if verdict in ("factual", "advice"):
        return verdict, {"rule": "llm_classifier", "verdict": verdict}
    return "unknown", {"rule": "llm_classifier", "verdict": verdict or "unparseable"}


def classify_intent(query: str, use_llm: bool = False) -> tuple[Intent, dict]:
    """Classify `query` and report which rule decided it.

    The evidence dict names the rule and the vocabulary entry that fired. It never
    echoes the query text, because the query may contain the PII that triggered the
    `personal_data` branch in the first place (NFR-8).
    """
    text = (query or "").strip()
    if not text:
        return "unknown", {"rule": "empty_query"}

    pii = detect_pii(text)
    if pii:
        return "personal_data", {"rule": "pii_detected", "types": pii}

    personal = _personal_data_rule(text)
    if personal:
        return "personal_data", {"rule": "personal_data_term", "matched": personal}

    performance = _first_match(text, PERFORMANCE_TERMS)
    if performance:
        return "performance", {"rule": "performance_term", "matched": performance}

    advice = _first_match(text, ADVICE_TERMS)
    if advice:
        return "advice", {"rule": "advice_term", "matched": advice}

    if mentions_out_of_corpus(text):
        return "out_of_scope", {"rule": "out_of_corpus_token"}

    factual = _first_match(text, FACTUAL_TERMS)
    if factual:
        return "factual", {"rule": "factual_term", "matched": factual}
    if _QUESTION_MARK.search(text):
        return "factual", {"rule": "interrogative"}

    if use_llm:
        return _llm_fallback(text)

    return "unknown", {"rule": "no_rule_matched"}
