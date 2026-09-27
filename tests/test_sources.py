"""Phase P1 tests: registry integrity, host policy, alias resolution."""

from __future__ import annotations

import pytest

from src.ingest.sources import (
    GENERAL_SCHEME_ID,
    RegistryError,
    is_allowed_url,
    iter_documents,
    load_registry,
    load_sources,
    make_doc_id,
    registry_stats,
    resolve_scheme,
)

EXPECTED_SCHEMES = {
    "S1": "Large Cap",
    "S2": "Flexi Cap",
    "S3": "ELSS",
    "S4": "Small Cap",
    "S5": "Hybrid",
}


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_five_schemes_are_registered(registry):
    specs = load_sources()
    assert len(specs) == 5
    assert {spec.scheme_id: spec.category for spec in specs} == EXPECTED_SCHEMES


def test_scheme_ids_are_unique(registry):
    ids = [spec.scheme_id for spec in registry.schemes]
    assert len(ids) == len(set(ids))


def test_every_scheme_has_aliases_and_documents(registry):
    for spec in registry.schemes:
        assert spec.aliases, f"{spec.scheme_id} has no aliases"
        assert spec.documents, f"{spec.scheme_id} has no documents"


def test_every_configured_url_is_on_the_allowlist(registry):
    for _spec, document, _target in iter_documents(registry):
        url = document.get("url")
        if not url:
            continue
        assert is_allowed_url(url, registry.allowed_hosts), f"not allowlisted: {url}"


def test_brief_urls_are_the_entry_urls(registry):
    assert registry.scheme("S3").entry_url == (
        "https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth"
    )
    assert registry.scheme("S1").entry_url == "https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth"


def test_third_party_blog_is_rejected():
    assert is_allowed_url("https://randomblog.example.com/mutual-funds") is False
    assert is_allowed_url("https://groww.in.evil.com/x") is False
    assert is_allowed_url("http://groww.in/mutual-funds/x") is False
    assert is_allowed_url("not-a-url") is False
    assert is_allowed_url("https://groww.in/mutual-funds/x") is True


def test_performance_is_excluded_by_policy(registry):
    assert registry.include_performance is False


def test_reference_documents_expand_to_gen_and_specific_schemes(registry):
    targets: dict[str, set[str]] = {}
    for _spec, document, target in iter_documents(registry):
        targets.setdefault(str(document.get("slug")), set()).add(target)
    assert targets["download_cas"] == {GENERAL_SCHEME_ID}
    assert targets["tax_regime"] == {GENERAL_SCHEME_ID}
    assert targets["sebi_elss"] == {"S3"}
    assert targets["overview"] == {"S1", "S2", "S3", "S4", "S5"}


def test_doc_ids_are_unique_and_prefixed_by_scheme(registry):
    doc_ids = [make_doc_id(target, document) for _s, document, target in iter_documents(registry)]
    assert len(doc_ids) == len(set(doc_ids))
    assert all(doc_id.split("__")[0] for doc_id in doc_ids)


def test_resolve_scheme_picks_the_right_scheme(registry):
    assert resolve_scheme("What is the expense ratio of the HDFC Large Cap Fund?", registry) == "S1"
    assert resolve_scheme("hdfc elss tax saver lock-in", registry) == "S3"
    assert resolve_scheme("tell me about the flexi cap fund", registry) == "S2"
    assert resolve_scheme("HDFC Balanced Advantage Fund benchmark", registry) == "S5"


def test_resolve_scheme_returns_none_when_ambiguous_or_unknown(registry):
    assert resolve_scheme("tell me about quant and parag funds", registry) is None
    assert resolve_scheme("what is weather today", registry) is None
    assert resolve_scheme("", registry) is None


def test_longest_alias_wins(registry):
    assert resolve_scheme("hdfc balanced advantage fund", registry) == "S5"


def test_registry_stats_are_sane(registry):
    stats = registry_stats(registry)
    assert stats["schemes"] == 5
    assert stats["urls_configured"] > 0
    assert stats["unverified"] >= 1
    assert stats["include_performance"] is False


def test_duplicate_scheme_id_is_rejected(tmp_path):
    path = tmp_path / "dupes.yaml"
    path.write_text(
        "allowed_hosts: [groww.in]\n"
        "schemes:\n"
        "  - {scheme_id: S1, scheme_name: A, category: Large Cap, entry_url: 'https://groww.in/a',"
        " documents: [{doc_type: fees, url: 'https://groww.in/a'}]}\n"
        "  - {scheme_id: S1, scheme_name: B, category: Large Cap, entry_url: 'https://groww.in/b',"
        " documents: [{doc_type: fees, url: 'https://groww.in/b'}]}\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="Duplicate scheme_id"):
        load_registry(path)
