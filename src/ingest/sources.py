"""Source registry: parsing, host policy, and scheme alias resolution.

The registry is declarative YAML so the required `docs/sources.*` deliverable can be
generated from the same data the index is built from (PRD FR-11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import yaml

from src import paths
from src.types import SourceSpec

GENERAL_SCHEME_ID = "GEN"
GENERAL_SCHEME_NAME = "General - AMFI/SEBI investor education"

VALID_DOC_TYPES = {
    "scheme_page",
    "factsheet",
    "kim_sid",
    "faq",
    "fees",
    "riskometer",
    "guide",
    "education",
    "other",
}


class RegistryError(ValueError):
    """Raised when sources.yaml is internally inconsistent."""


@dataclass(frozen=True)
class Registry:
    amc: str
    amc_short: str
    amc_domain: str
    allowed_hosts: tuple[str, ...]
    include_performance: bool
    schemes: tuple[SourceSpec, ...]
    reference_documents: tuple[dict, ...]

    def scheme(self, scheme_id: str) -> SourceSpec | None:
        return next((s for s in self.schemes if s.scheme_id == scheme_id), None)

    def alias_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for spec in self.schemes:
            for alias in spec.aliases:
                mapping[alias.lower()] = spec.scheme_id
        return mapping


def _read_yaml(path: Path | None = None) -> dict:
    target = path or paths.SOURCES_YAML
    if not target.exists():
        raise RegistryError(f"Source registry not found: {target}")
    with target.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise RegistryError(f"{target} must contain a mapping at the top level")
    return data


def _parse_scheme(raw: dict) -> SourceSpec:
    missing = [key for key in ("scheme_id", "scheme_name", "category", "entry_url") if not raw.get(key)]
    if missing:
        raise RegistryError(f"Scheme entry missing {missing}: {raw.get('scheme_id') or raw}")
    documents = list(raw.get("documents") or [])
    if not documents:
        raise RegistryError(f"Scheme {raw['scheme_id']} declares no documents")
    return SourceSpec(
        scheme_id=str(raw["scheme_id"]),
        scheme_name=str(raw["scheme_name"]),
        category=str(raw["category"]),
        entry_url=str(raw["entry_url"]),
        aliases=[str(a).lower() for a in (raw.get("aliases") or [])],
        documents=documents,
    )


def load_registry(path: Path | None = None) -> Registry:
    data = _read_yaml(path)
    schemes = tuple(_parse_scheme(item) for item in (data.get("schemes") or []))

    seen: set[str] = set()
    for spec in schemes:
        if spec.scheme_id in seen:
            raise RegistryError(f"Duplicate scheme_id: {spec.scheme_id}")
        seen.add(spec.scheme_id)

    policy = data.get("content_policy") or {}
    return Registry(
        amc=str(data.get("amc", "")),
        amc_short=str(data.get("amc_short", "")),
        amc_domain=str(data.get("amc_domain", "")),
        allowed_hosts=tuple(str(h).lower() for h in (data.get("allowed_hosts") or [])),
        include_performance=bool(policy.get("include_performance", False)),
        schemes=schemes,
        reference_documents=tuple(data.get("reference_documents") or []),
    )


def load_sources(path: Path | None = None) -> list[SourceSpec]:
    """Contract name used by later phases: the five in-scope schemes."""
    return list(load_registry(path).schemes)


def is_allowed_url(url: str, allowed_hosts: tuple[str, ...] | list[str] | None = None) -> bool:
    """True only for https URLs whose host is on the allowlist (PRD 5.3 rule 2)."""
    if not url:
        return False
    hosts = tuple(allowed_hosts) if allowed_hosts is not None else load_registry().allowed_hosts
    if not hosts:
        return False
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    return any(host == allowed or host.endswith("." + allowed) for allowed in hosts)


def iter_documents(registry: Registry | None = None):
    """Yield `(target_spec, document, scheme_id_for_storage)` for every fetchable slot.

    A document with `applies_to: ["*"]` is stored once under the GEN pseudo-scheme so
    it is not duplicated five times; a document that names specific schemes is stored
    under each of them, which keeps scheme-filtered retrieval honest.
    """
    reg = registry or load_registry()
    for spec in reg.schemes:
        for document in spec.documents:
            yield spec, document, spec.scheme_id

    for document in reg.reference_documents:
        applies_to = document.get("applies_to") or ["*"]
        if "*" in applies_to:
            general = SourceSpec(
                scheme_id=GENERAL_SCHEME_ID,
                scheme_name=GENERAL_SCHEME_NAME,
                category="General",
                entry_url=str(document.get("url") or ""),
                aliases=[],
                documents=[],
            )
            yield general, document, GENERAL_SCHEME_ID
            continue
        for scheme_id in applies_to:
            target = reg.scheme(str(scheme_id))
            if target is None:
                raise RegistryError(f"Reference document {document.get('slug')!r} targets unknown scheme {scheme_id!r}")
            yield target, document, target.scheme_id


def make_doc_id(scheme_id: str, document: dict) -> str:
    return f"{scheme_id}__{document.get('slug') or document.get('doc_type') or 'doc'}"


def resolve_scheme(text: str, registry: Registry | None = None) -> str | None:
    """Return the single scheme named in `text`, else None.

    Case-insensitive alias match on word boundaries. Alias length is deliberately
    irrelevant: once we demand exactly one distinct scheme, "hdfc balanced advantage
    fund" resolves to S5 without a longest-alias rule, because "balanced" and
    "balanced advantage" both belong to S5.

    Naming two or more schemes returns None. That keeps a caller from narrowing the
    corpus to a scheme the user did not actually ask about, which matters because the
    retriever turns a resolved id straight into a `where={"scheme_id": ...}` filter.
    """
    reg = registry or load_registry()
    haystack = (text or "").lower()
    matched: set[str] = set()
    for alias, scheme_id in reg.alias_map().items():
        if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", haystack):
            matched.add(scheme_id)
    return matched.pop() if len(matched) == 1 else None


def registry_stats(registry: Registry | None = None) -> dict:
    reg = registry or load_registry()
    documents = list(iter_documents(reg))
    return {
        "amc": reg.amc,
        "schemes": len(reg.schemes),
        "document_slots": len(documents),
        "urls_configured": sum(1 for _s, d, _t in documents if d.get("url")),
        "unverified": sum(1 for _s, d, _t in documents if d.get("verify")),
        "include_performance": reg.include_performance,
    }
