"""PII detection and redaction (PRD FR-9, NFR-8).

`detect_pii` returns finding *types* only and never the matched value, so no caller
can accidentally log or echo a personal identifier.
"""

from __future__ import annotations

import re

PAN = "pan"
AADHAAR = "aadhaar"
ACCOUNT = "account"
EMAIL = "email"
PHONE = "phone"
OTP = "otp"

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (OTP, re.compile(r"\b(?:otp|one[\s-]?time\s+password)\b\D{0,10}(\d{4,6})\b", re.IGNORECASE)),
    (PAN, re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)),
    (AADHAAR, re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")),
    (EMAIL, re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    # `+91 98765 43210` and `9876543210` are both common, and the country code must be
    # consumed as a unit: `\b` cannot sit between "91" and the number, so the prefix is
    # an explicit alternative. `(?![0-9])` rather than `\b` so a grouped number can end
    # on a space, and so a 12-digit Aadhaar is never read as a 10-digit phone window.
    (PHONE, re.compile(r"(?:\+\s?91[\s-]?|\b)[6-9](?:[\s-]?\d){9}(?![0-9])")),
    (ACCOUNT, re.compile(r"\b\d{8,18}\b")),
]

# An Aadhaar number and a 10-digit phone number are also long digit runs, so report
# the more specific type only.
_MORE_SPECIFIC = (AADHAAR, PHONE)


def detect_pii(text: str) -> list[str]:
    """Return the sorted, de-duplicated names of PII types present in `text`."""
    found: set[str] = set()
    for name, pattern in PATTERNS:
        if pattern.search(text or ""):
            found.add(name)
    if any(name in found for name in _MORE_SPECIFIC):
        found.discard(ACCOUNT)
    return sorted(found)


def redact(text: str) -> str:
    """Replace every PII match in `text` with `[REDACTED]`."""
    if not text:
        return text
    out = text
    for _name, pattern in PATTERNS:
        out = pattern.sub("[REDACTED]", out)
    return out


def pattern_signature() -> list[str]:
    """Stable signature of PATTERNS, used to assert loggers and guards stay in sync."""
    return [f"{name}:{pattern.pattern}" for name, pattern in PATTERNS]
