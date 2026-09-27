"""P6 - Refusal builder (PRD FR-7, NFR-8).

A refusal is a complete, useful `Answer`, not an exception. It states the boundary, points
at an authoritative public source, and never contains a number the corpus cannot cite.

Every link below is a URL already present in `data/raw` (SEBI/AMFI pages fetched during
ingestion) or in `sources.yaml`. Nothing here invents an endpoint, and no link resolves to
an HDFC factsheet PDF: those returned HTTP 403 during P1 ingestion, so pointing users at
them would send them to a dead link.
"""

from __future__ import annotations

from src.config import get_settings
from src.ingest.sources import load_registry
from src.types import Answer, Citation, QueryTrace

__all__ = ["DISCLAIMER", "EDUCATIONAL_LINKS", "build_refusal"]

DISCLAIMER = (
    "I can only share published facts about the 5 HDFC schemes in this demo's scope - "
    "facts-only, no investment advice."
)

# Verified during ingestion; see data/raw/*.meta.json and sources.yaml.
_SEBI_CAS = "https://investor.sebi.gov.in/consolidated_account_statement.html"
_AMFI_DOWNLOAD_CAS = "https://www.amfiindia.com/online-center/download-cas"
_AMFI_EDUCATION = (
    "https://www.amfiindia.com/investor/knowledge-center-info"
    "?zoneName=CategorizationOfMutualFundSchemes"
)

# Per-intent defaults. `build_refusal` upgrades `performance` and `advice` to the
# resolved scheme's own page when it knows which scheme was asked about.
EDUCATIONAL_LINKS: dict[str, str] = {
    "performance": _AMFI_EDUCATION,
    "advice": _AMFI_EDUCATION,
    "out_of_scope": _AMFI_EDUCATION,
    "personal_data": _SEBI_CAS,
    "factual": _AMFI_EDUCATION,
    "unknown": _AMFI_EDUCATION,
}

_BODY: dict[str, str] = {
    "performance": (
        "Performance figures and rankings are outside what this demo will answer, and I "
        "will not quote a return from memory. The official factsheet is the authoritative "
        "source for past performance."
    ),
    "advice": (
        "I cannot recommend a scheme, tell you whether to buy or switch, or plan an "
        "allocation. I can share published facts - expense ratio, exit load, benchmark, "
        "minimum SIP - for the schemes in scope so you can decide for yourself."
    ),
    "out_of_scope": (
        "That fund is not one of the 5 HDFC schemes in this demo's corpus, so I have no "
        "retrieved source for it and I will not answer from memory or substitute a "
        "similar-looking HDFC scheme."
    ),
    "personal_data": (
        "I cannot access or discuss any individual account, and please do not share "
        "personal identifiers such as a PAN, Aadhaar number, account number or OTP. "
        "Your own statements are available directly: the AMFI CAS download guide is at "
        f"{_AMFI_DOWNLOAD_CAS}"
    ),
    "factual": "That question is outside what this demo can answer from its corpus.",
    "unknown": "I could not map that question to the schemes in this demo's scope.",
}


def _link_for(intent: str, scheme_id: str | None) -> str:
    """Pick the most specific authoritative link available for `intent`.

    Only `performance` and `advice` are upgraded to the resolved scheme's own page.
    `out_of_scope` deliberately is not: "expense ratio of Parag Parag Flexi Cap" resolves
    a scheme id via the "flexi cap" alias, and linking that HDFC page would be exactly the
    substitution this guard exists to prevent.
    """
    if scheme_id and intent in ("performance", "advice"):
        spec = load_registry().scheme(scheme_id)
        if spec is not None:
            return spec.entry_url
    return EDUCATIONAL_LINKS.get(intent, EDUCATIONAL_LINKS["unknown"])


def build_refusal(
    intent: str,
    scheme_id: str | None = None,
    *,
    query: str = "",
    evidence: dict | None = None,
    pii_types: list[str] | None = None,
) -> Answer:
    """Return a fully-formed refusal `Answer` for `intent`.

    The `personal_data` branch uses `status="pii_rejected"` to distinguish "you sent me an
    identifier" from a policy refusal; the exact detected types go in the trace, never in
    the text and never in the log.
    """
    body = _BODY.get(intent, _BODY["unknown"])
    link = _link_for(intent, scheme_id)
    settings = get_settings()

    # Imported here, not at module scope: `observability` imports `redact` from
    # `guardrails.pii`, so importing it at the top of this module closes the cycle
    # `observability -> guardrails -> guardrails.refusal -> observability`. The hash is
    # deliberately shared rather than reimplemented, so a trace's `query_hash` still
    # matches the corresponding line in the query log.
    from src.observability import hash_query, now_iso

    stamp = now_iso()

    if settings.disclaimer and settings.disclaimer not in body and settings.disclaimer != DISCLAIMER:
        body = f"{body}\n\n{settings.disclaimer}"

    text = f"{body}\n\n{DISCLAIMER}\n\nLearn more: {link}"
    if intent == "personal_data":
        text = (
            "Please remove any personal identifiers from your question and try again. "
            "I cannot access individual account data.\n\n"
            f"{text}"
        )

    trace = QueryTrace(
        query_hash=hash_query(query),
        guards={
            "intent": intent,
            "refused": True,
            "rule": (evidence or {}).get("rule", "refusal_builder"),
            "pii_types": list(pii_types or []),
            "scheme_id": scheme_id,
        },
        hits=[],
        threshold=settings.min_similarity,
        threshold_passed=False,
        cited_doc_ids=[],
        latency_ms={},
    )

    from src.observability import hash_query

    # `last_updated` means "when the document backing this answer was retrieved". A refusal
    # is backed by no indexed document, so it is None rather than `now_iso()`: telling a
    # user a refusal was verified moments ago would imply the facts were freshly checked.
    return Answer(
        status="pii_rejected" if intent == "personal_data" else "refused",
        text=text,
        citations=[
            Citation(
                label=f"Educational reference ({intent})",
                url=link,
                doc_type="educational",
                retrieved_at=stamp,
            )
        ],
        last_updated=None,
        trace=trace,
    )
