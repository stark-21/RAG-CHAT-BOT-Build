"""Application configuration. Secrets are read from the environment only."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from src import paths

VALID_PROVIDERS = ("hosted", "local", "stub")
VALID_CATEGORIES = ("Large Cap", "Flexi Cap", "ELSS", "Small Cap", "Hybrid")


class ConfigError(RuntimeError):
    """Raised with every configuration problem listed at once."""


@dataclass(frozen=True)
class Settings:
    embedding_model: str
    chroma_dir: str
    collection_name: str
    top_k: int
    min_similarity: float
    llm_provider: str
    llm_model: str
    llm_api_key: str | None
    temperature: float
    max_sentences: int
    disclaimer: str
    hf_home: str
    data_dir: str
    docs_dir: str
    logs_dir: str
    request_timeout: int
    request_delay_seconds: float
    user_agent: str


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(paths.PROJECT_ROOT / ".env", override=False)

    if "HF_HOME" in os.environ:
        os.environ["HF_HOME"] = str((paths.PROJECT_ROOT / _env("HF_HOME")).resolve())

    settings = Settings(
        embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        chroma_dir=_env("CHROMA_DIR", str(paths.CHROMA_DIR)),
        collection_name=_env("COLLECTION_NAME", "mf_faq"),
    # TOP_K=10 and MIN_SIMILARITY=0.30 come from `python eval/run_eval.py`. The sweep found
    # fact recall is entirely TOP_K-driven: TOP_K=3 recovered 3/6 expected facts, TOP_K=5
    # recovered 5/6, and TOP_K=10 recovered 6/6. MIN_SIMILARITY had no effect on recall at
    # any of the nine values tested (0.20-0.60) and every nonsense probe abstained at every
    # one, so the probe set cannot identify it and 0.30 was left at the middle of the range
    # rather than tuned to an arbitrary grid edge.
    top_k=_env_int("TOP_K", 10),
    min_similarity=_env_float("MIN_SIMILARITY", 0.30),
        llm_provider=_env("LLM_PROVIDER", "stub"),
        llm_model=_env("LLM_MODEL", ""),
        llm_api_key=os.environ.get("LLM_API_KEY") or None,
        temperature=_env_float("LLM_TEMPERATURE", 0.0),
        max_sentences=_env_int("MAX_SENTENCES", 3),
        disclaimer=_env("DISCLAIMER", "Facts-only. No investment advice."),
        hf_home=_env("HF_HOME", "./.cache/hf"),
        data_dir=_env("DATA_DIR", str(paths.DATA_DIR)),
        docs_dir=_env("DOCS_DIR", str(paths.DOCS_DIR)),
        logs_dir=_env("LOGS_DIR", str(paths.LOGS_DIR)),
        request_timeout=_env_int("REQUEST_TIMEOUT", 30),
        request_delay_seconds=_env_float("REQUEST_DELAY_SECONDS", 1.0),
        user_agent=_env(
            "USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        ),
    )
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    problems: list[str] = []

    if settings.llm_provider not in VALID_PROVIDERS:
        problems.append(
            f"LLM_PROVIDER must be one of {VALID_PROVIDERS}, got {settings.llm_provider!r}"
        )
    if settings.llm_provider == "hosted" and not settings.llm_api_key:
        problems.append("LLM_PROVIDER=hosted requires LLM_API_KEY to be set in .env")
    if not 0.0 <= settings.min_similarity <= 1.0:
        problems.append(f"MIN_SIMILARITY must be within 0.0-1.0, got {settings.min_similarity}")
    if settings.top_k < 1:
        problems.append(f"TOP_K must be >= 1, got {settings.top_k}")
    if settings.max_sentences < 1:
        problems.append(f"MAX_SENTENCES must be >= 1, got {settings.max_sentences}")
    if not settings.collection_name.strip():
        problems.append("COLLECTION_NAME must not be empty")
    if settings.request_timeout < 1:
        problems.append(f"REQUEST_TIMEOUT must be >= 1, got {settings.request_timeout}")
    if settings.request_delay_seconds < 0:
        problems.append("REQUEST_DELAY_SECONDS must be >= 0")

    if problems:
        raise ConfigError(
            "Invalid configuration:\n  - " + "\n  - ".join(problems) + "\nCopy .env.example to .env and fix the values above."
        )


def reset_settings_cache() -> None:
    """Test helper: drop the memoised settings so env changes take effect."""
    get_settings.cache_clear()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else paths.PROJECT_ROOT / path
