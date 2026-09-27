"""Phase P0 tests: config loading, validation, and the PII/redaction primitives."""

from __future__ import annotations

import pytest

from src import config, observability
from src.config import ConfigError, get_settings
from src.guardrails.pii import detect_pii, pattern_signature, redact


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_KEY",
        "TOP_K",
        "MIN_SIMILARITY",
        "MAX_SENTENCES",
        "COLLECTION_NAME",
        "REQUEST_TIMEOUT",
        "REQUEST_DELAY_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    config.reset_settings_cache()
    yield
    config.reset_settings_cache()


def test_defaults_apply_without_env_file():
    settings = get_settings()
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert settings.collection_name == "mf_faq"
    assert settings.top_k == 10
    assert settings.min_similarity == 0.30
    assert settings.temperature == 0.0
    assert settings.max_sentences == 3
    assert settings.disclaimer == "Facts-only. No investment advice."
    assert settings.llm_provider == "stub"


def test_env_overrides_are_coerced(monkeypatch):
    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("MIN_SIMILARITY", "0.45")
    config.reset_settings_cache()
    settings = get_settings()
    assert settings.top_k == 8
    assert isinstance(settings.min_similarity, float)
    assert settings.min_similarity == 0.45


def test_hosted_provider_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "hosted")
    config.reset_settings_cache()
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        get_settings()


def test_hosted_provider_with_key_is_accepted(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "hosted")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    config.reset_settings_cache()
    assert get_settings().llm_api_key == "test-key"


def test_all_problems_are_reported_at_once(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setenv("MIN_SIMILARITY", "1.5")
    config.reset_settings_cache()
    with pytest.raises(ConfigError) as excinfo:
        get_settings()
    message = str(excinfo.value)
    assert "LLM_PROVIDER" in message
    assert "TOP_K" in message
    assert "MIN_SIMILARITY" in message


def test_non_numeric_value_is_reported(monkeypatch):
    monkeypatch.setenv("TOP_K", "many")
    config.reset_settings_cache()
    with pytest.raises(ConfigError, match="TOP_K must be an integer"):
        get_settings()


def test_pii_detection_returns_types_only():
    findings = detect_pii("My PAN is ABCDE1234F")
    assert findings == ["pan"]
    assert "ABCDE1234F" not in str(findings)


def test_pii_detects_every_required_type():
    assert detect_pii("aadhaar 2345 6789 0123") == ["aadhaar"]
    assert detect_pii("account 123456789012") == ["account"]
    assert detect_pii("mail me at ravi@example.com") == ["email"]
    assert detect_pii("call 9876543210") == ["phone"]
    assert detect_pii("your otp is 482913") == ["otp"]


def test_clean_query_has_no_findings():
    assert detect_pii("What is the exit load on the HDFC Large Cap Fund?") == []


def test_redact_masks_all_types_in_one_string():
    masked = redact("PAN ABCDE1234F, aadhaar 2345 6789 0123, ravi@example.com, 9876543210")
    for secret in ("ABCDE1234F", "2345 6789 0123", "ravi@example.com", "9876543210"):
        assert secret not in masked
    assert masked.count("[REDACTED]") == 4


def test_observability_uses_the_same_patterns():
    masked = observability.redact("PAN ABCDE1234F")
    assert masked == "PAN [REDACTED]"
    assert pattern_signature() == pattern_signature()


def test_query_hash_is_stable_and_case_insensitive():
    assert observability.hash_query("Exit Load?") == observability.hash_query("exit   load?")
    assert len(observability.hash_query("exit load")) == 12


def test_log_jsonl_redacts_records(tmp_path):
    target = tmp_path / "queries.jsonl"
    observability.log_jsonl(target, {"query_hash": "abc", "note": "PAN ABCDE1234F"})
    body = target.read_text(encoding="utf-8")
    assert "ABCDE1234F" not in body
    assert "[REDACTED]" in body
