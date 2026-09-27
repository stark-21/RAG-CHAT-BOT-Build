"""P5 - Generation and output validation (the L6 validations).

A language model is the least trustworthy component in this pipeline, so its output is
treated as untrusted input and re-validated against the retrieved context before a user
ever sees it:

1. **Citation check** - every URL in the answer must be one we retrieved. A model that
   invents a plausible-looking link is the most damaging failure mode here, because the
   whole product promise is "cited".
2. **Number check** - every numeric token in the answer must already exist in the
   retrieved context. This is what makes the assistant facts-only: it cannot state a
   return, a ratio or a fee that the corpus does not contain, so it cannot hallucinate one.
3. **Sentence cap** - keep the first `max_sentences` sentences, then re-append the citation
   line and the "Last updated from sources" line, so trimming can never strip the
   provenance the user needs.

Violations demote the answer to `insufficient_context` rather than raising. The machine
-readable reason is left in `trace.error` so the P7 orchestrator can decide whether one
regeneration is worth attempting before giving up.
"""

from __future__ import annotations

import re

from src.config import get_settings
from src.llm.provider import MAX_TOKENS, LLMProvider, get_provider
from src.observability import now_iso
from src.types import Answer, Citation, QueryTrace, RetrievedChunk

__all__ = [
    "cap_sentences",
    "extract_urls",
    "generate",
    "numbers_in",
    "parse_answer",
]

LAST_UPDATED_PREFIX = "Last updated from sources:"

_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_TRAILING_PUNCT = ".,;:)]}"

# A number, optionally carrying a currency prefix or a percent sign. Deliberately does not
# accept a leading sign: "2026-09-27" must yield 2026, 09 and 27 rather than one signed
# token, or every mandated date footer would read as a novel number.
_NUMBER_RE = re.compile(
    r"(?:rs\.?|inr|₹)?\s*\d[\d,]*(?:\.\d+)?\s*%?",
    re.IGNORECASE,
)

# Tokens that end in "." without ending a sentence.
_ABBREVIATIONS = ("Rs.", "No.", "Dr.", "Mr.", "Ms.", "vs.", "etc.", "e.g.", "i.e.", "approx.")
_SENTINEL = "\x00"

_CITATION_ERROR = "citation_not_in_context"
_NO_CITATION_ERROR = "no_citation_in_context"
_NUMBER_ERROR = "novel_number"


def extract_urls(text: str) -> list[str]:
    """Every http(s) URL in `text`, de-duplicated, in order of appearance."""
    seen: list[str] = []
    for match in _URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(_TRAILING_PUNCT)
        if url and url not in seen:
            seen.append(url)
    return seen


def numbers_in(text: str) -> set[str]:
    """Normalised numeric tokens in `text`.

    Normalisation strips thousands separators (including the Indian `1,00,000`), currency
    prefixes and a trailing `%`, so `Rs. 1,00,000`, `INR 100000` and `100000` compare equal.
    """
    found: set[str] = set()
    for match in _NUMBER_RE.finditer(text or ""):
        token = match.group(0)
        # `\brs\.?\b` cannot work here: there is no word boundary between the "." of "Rs."
        # and the following space, so the trailing \b would reject the most common form.
        token = re.sub(r"(?i)(?:\brs\.?|\binr\b)", "", token)
        token = token.replace("₹", "").replace(",", "").replace("%", "").strip()
        if token and token[0].isdigit():
            found.add(token)
    return found


def cap_sentences(text: str, max_sentences: int) -> str:
    """Keep at most `max_sentences` sentences from `text`.

    Decimal points and common abbreviations are masked first so `0.005%` and `Rs. 500`
    are not mistaken for sentence ends.
    """
    if max_sentences < 1:
        raise ValueError("max_sentences must be >= 1")
    body = (text or "").strip()
    if not body:
        return ""

    masked = body
    for i, abbreviation in enumerate(_ABBREVIATIONS):
        masked = re.sub(re.escape(abbreviation), f"{_SENTINEL}{i}{_SENTINEL}", masked, flags=re.IGNORECASE)
    masked = re.sub(r"(?<=\d)\.(?=\d)", f"{_SENTINEL}d{_SENTINEL}", masked)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", masked) if s.strip()]
    kept = sentences[:max_sentences]

    # A provider's trailing citation line is provenance, not prose. Joining it with a space
    # read as "... from the date of investment. source: https://..." - one mangled sentence
    # and a raw URL in the middle of the answer - so it keeps its own line.
    out = kept[0] if kept else ""
    for sentence in kept[1:]:
        out = f"{out}\n{sentence}" if sentence.lower().startswith("source:") else f"{out} {sentence}"
    for i in range(len(_ABBREVIATIONS)):
        out = out.replace(f"{_SENTINEL}{i}{_SENTINEL}", _ABBREVIATIONS[i])
    return out.replace(f"{_SENTINEL}d{_SENTINEL}", ".").strip()


def generate(prompt: str, provider: LLMProvider | None = None) -> str:
    """Single completion attempt at temperature 0. Retries are P7's decision, not ours."""
    settings = get_settings()
    llm = provider or get_provider(settings)
    return llm.generate(prompt, temperature=settings.temperature, max_tokens=MAX_TOKENS)


def _context_urls(hits: list[RetrievedChunk]) -> set[str]:
    return {hit.chunk.source_url for hit in hits if hit.chunk.source_url}


def _allowed_numbers(hits: list[RetrievedChunk]) -> set[str]:
    """Numbers the model is permitted to state: anything in the retrieved text, plus the
    dates and URLs it must echo in the mandated footer."""
    allowed: set[str] = set()
    for hit in hits:
        allowed |= numbers_in(hit.chunk.text)
        allowed |= numbers_in(hit.chunk.retrieved_at)
        allowed |= numbers_in(hit.chunk.source_url)
        allowed |= numbers_in(hit.chunk.section)
    return allowed


def _cited_hits(text: str, hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    urls = set(extract_urls(text))
    cited = [hit for hit in hits if hit.chunk.source_url in urls]
    if cited:
        return cited
    # No usable citation: fall back to the best hit, since the footer will cite it anyway.
    return hits[:1]


def _answer(
    status: str,
    text: str,
    hits: list[RetrievedChunk],
    *,
    error: str | None,
) -> Answer:
    cited = _cited_hits(text, hits)
    last_updated = max((h.chunk.retrieved_at for h in cited), default="") or None

    seen: set[str] = set()
    citations: list[Citation] = []
    for hit in cited:
        url = hit.chunk.source_url
        if not url or url in seen:
            continue
        seen.add(url)
        citations.append(
            Citation(
                label=f"{hit.chunk.scheme_id} - {hit.chunk.section}"[:120],
                url=url,
                doc_type=hit.chunk.doc_type,
                retrieved_at=hit.chunk.retrieved_at,
            )
        )

    return Answer(
        status=status,
        text=text,
        citations=citations,
        last_updated=last_updated,
        trace=QueryTrace(
            query_hash="",
            guards={},
            hits=[
                {
                    "chunk_id": h.chunk.chunk_id,
                    "score": round(h.score, 4),
                    "doc_type": h.chunk.doc_type,
                    "section": h.chunk.section,
                }
                for h in cited
            ],
            threshold=get_settings().min_similarity,
            threshold_passed=status == "answered",
            cited_doc_ids=[h.chunk.chunk_id for h in cited],
            latency_ms={},
            error=error,
        ),
    )


def parse_answer(raw: str, hits: list[RetrievedChunk]) -> Answer:
    """Validate a raw completion against the retrieved context and build an `Answer`."""
    settings = get_settings()
    body = (raw or "").strip()
    if not body:
        return _answer("insufficient_context", "", hits, error="empty_response")

    context_urls = _context_urls(hits)
    urls = extract_urls(body)
    invented = [u for u in urls if u not in context_urls]
    if invented:
        return _answer(
            "insufficient_context",
            "",
            hits,
            error=f"{_CITATION_ERROR}:{invented[0]}",
        )

    if not urls:
        return _answer("insufficient_context", "", hits, error=_NO_CITATION_ERROR)

    # Validate numbers on the model's own words, before we append our footer.
    novel = sorted(numbers_in(body) - _allowed_numbers(hits))
    if novel:
        return _answer(
            "insufficient_context",
            "",
            hits,
            error=f"{_NUMBER_ERROR}:{novel[0]}",
        )

    trimmed = cap_sentences(body, settings.max_sentences)
    if not extract_urls(trimmed):
        # Trimming removed the citation; re-append the best context URL so rule 3 holds.
        trimmed = f"{trimmed}\nsource: {hits[0].chunk.source_url}" if hits else trimmed

    cited = _cited_hits(trimmed, hits)
    last_updated = max((h.chunk.retrieved_at for h in cited), default="") or now_iso()[:10]

    footer = f"{LAST_UPDATED_PREFIX} {last_updated}"
    text = f"{trimmed}\n\n{footer}" if trimmed else footer

    return _answer("answered", text, hits, error=None)
