"""Structured logging with PII redaction applied to every record (NFR-8, FR-13)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import paths
from src.guardrails.pii import redact

_LOGGER_NAME = "ragfaq"
_WS = re.compile(r"\s+")


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME if name is None else f"{_LOGGER_NAME}.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def normalise_query(text: str) -> str:
    return _WS.sub(" ", (text or "").strip().lower())


def hash_query(text: str) -> str:
    return hashlib.sha256(normalise_query(text).encode("utf-8")).hexdigest()[:12]


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


def log_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_redact_value(record), ensure_ascii=False, default=str)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def ingest_record(**fields: Any) -> None:
    log_jsonl(paths.INGEST_LOG, {"ts": now_iso(), "kind": "ingest", **fields})


def query_record(**fields: Any) -> None:
    log_jsonl(paths.QUERY_LOG, {"ts": now_iso(), "kind": "query", **fields})
