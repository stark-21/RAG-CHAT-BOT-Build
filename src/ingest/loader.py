"""Phase P1 loading: polite fetching, extraction, and immutable raw snapshots.

Rules enforced here (PRD FR-1):
- host allowlist is checked before any network call;
- one bad URL never aborts the run;
- re-running with an unchanged content hash rewrites nothing (idempotence);
- the snapshot on disk is the cleaned text that was actually retrieved, and it is
  never edited afterwards, so `retrieved_at` stays auditable.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src import paths
from src.config import Settings, get_settings
from src.ingest import cleaner
from src.ingest.sources import Registry, iter_documents, load_registry, make_doc_id
from src.observability import content_hash, get_logger, ingest_record, now_iso, today_iso
from src.types import SourceDoc, SourceSpec

MIN_TEXT_CHARS = 200
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

logger = get_logger("ingest")


@dataclass
class FetchResult:
    ok: bool
    status_code: int | None
    final_url: str
    content: bytes = b""
    content_type: str = ""
    encoding: str = ""
    error: str = ""
    elapsed_ms: int = 0
    attempts: int = 1


def _charset_of(content_type: str) -> str:
    for part in (content_type or "").split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            return part.split("=", 1)[1].strip().strip("\"'") or "utf-8"
    return "utf-8"


def decode_html(content: bytes, content_type: str) -> str:
    encoding = _charset_of(content_type)
    try:
        return content.decode(encoding, errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def _session(settings: Settings) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": settings.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9",
        }
    )
    return session


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(requests.RequestException),
)
def _get(session: requests.Session, url: str, settings: Settings) -> requests.Response:
    return session.get(url, timeout=settings.request_timeout, allow_redirects=True)


def fetch_url(url: str, *, session: requests.Session | None = None, settings: Settings | None = None) -> FetchResult:
    cfg = settings or get_settings()
    http = session or _session(cfg)
    started = time.perf_counter()
    try:
        response = _get(http, url, cfg)
    except requests.RequestException as exc:
        return FetchResult(
            ok=False,
            status_code=None,
            final_url=url,
            error=f"{type(exc).__name__}: {exc}"[:200],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    elapsed = int((time.perf_counter() - started) * 1000)
    retry_after = response.headers.get("Retry-After")
    if response.status_code in RETRYABLE_STATUS:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 10.0))
            except ValueError:
                pass
    if not response.ok:
        return FetchResult(
            ok=False,
            status_code=response.status_code,
            final_url=response.url,
            content_type=response.headers.get("Content-Type", ""),
            error=f"http_{response.status_code}",
            elapsed_ms=elapsed,
        )
    return FetchResult(
        ok=True,
        status_code=response.status_code,
        final_url=response.url,
        content=response.content,
        content_type=response.headers.get("Content-Type", ""),
        encoding=_charset_of(response.headers.get("Content-Type", "")),
        elapsed_ms=elapsed,
    )


def _is_pdf(document: dict, fetched: FetchResult | None) -> bool:
    if (document.get("format") or "").lower() == "pdf":
        return True
    if fetched and "application/pdf" in fetched.content_type.lower():
        return True
    return str((document or {}).get("url") or "").lower().split("?")[0].endswith(".pdf")


def _failed_doc(spec: SourceSpec, document: dict, doc_id: str, url: str, reason: str) -> SourceDoc:
    return SourceDoc(
        doc_id=doc_id,
        scheme_id=spec.scheme_id,
        scheme_name=spec.scheme_name,
        category=spec.category,
        doc_type=str(document.get("doc_type") or "other"),
        title=str(document.get("title") or ""),
        source_url=url,
        publisher=str(document.get("publisher") or ""),
        retrieved_at=today_iso(),
        content_hash="",
        raw_path="",
        text="",
        status="failed",
        reason=reason,
    )


def fetch_doc(
    spec: SourceSpec,
    document: dict,
    *,
    registry: Registry | None = None,
    session: requests.Session | None = None,
    settings: Settings | None = None,
) -> SourceDoc:
    cfg = settings or get_settings()
    reg = registry or load_registry()
    doc_id = make_doc_id(spec.scheme_id, document)
    url = str(document.get("url") or "")

    if not url:
        return _failed_doc(spec, document, doc_id, "", "url_not_configured")
    if not _url_allowed(url, reg):
        return _failed_doc(spec, document, doc_id, url, "domain_not_allowed")

    fetched = fetch_url(url, session=session, settings=cfg)
    if not fetched.ok:
        return _failed_doc(spec, document, doc_id, url, fetched.error or "fetch_failed")

    if _is_pdf(document, fetched):
        try:
            cleaned = cleaner.extract_pdf(fetched.content)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the run
            logger.warning("pdf extraction failed for %s: %s", url, exc)
            return _failed_doc(spec, document, doc_id, url, f"pdf_error:{type(exc).__name__}")
    else:
        cleaned = cleaner.clean_html(
            decode_html(fetched.content, fetched.content_type),
            include_performance=reg.include_performance,
        )

    if len(cleaned.text) < MIN_TEXT_CHARS:
        return _failed_doc(spec, document, doc_id, url, "empty_extraction")

    return SourceDoc(
        doc_id=doc_id,
        scheme_id=spec.scheme_id,
        scheme_name=spec.scheme_name,
        category=spec.category,
        doc_type=str(document.get("doc_type") or "other"),
        title=document.get("title") or cleaned.title or spec.scheme_name,
        source_url=fetched.final_url or url,
        publisher=str(document.get("publisher") or reg.amc),
        retrieved_at=today_iso(),
        content_hash=content_hash(cleaned.text),
        raw_path="",
        text=cleaned.text,
        tables=cleaned.tables,
        status="ok",
        reason="",
        stats=dict(cleaned.stats),
    )


def _url_allowed(url: str, registry: Registry) -> bool:
    from src.ingest.sources import is_allowed_url

    return is_allowed_url(url, registry.allowed_hosts)


def write_doc(doc: SourceDoc, *, force: bool = False) -> tuple[Path | None, Path | None, bool]:
    """Persist a snapshot. Returns `(txt_path, meta_path, rewritten)`."""
    if doc.status != "ok":
        return None, None, False
    directory = paths.scheme_raw_dir(doc.scheme_id)
    directory.mkdir(parents=True, exist_ok=True)
    txt_path = directory / f"{doc.doc_id}.txt"
    meta_path = directory / f"{doc.doc_id}{paths.META_SUFFIX}"

    if meta_path.exists() and not force:
        try:
            previous = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        if previous.get("content_hash") == doc.content_hash:
            doc.raw_path = str(txt_path.relative_to(paths.PROJECT_ROOT))
            return txt_path, meta_path, False

    doc.raw_path = str(txt_path.relative_to(paths.PROJECT_ROOT))
    payload = {key: value for key, value in asdict(doc).items() if key not in ("text", "tables")}
    payload["note"] = ""
    txt_path.write_text(doc.text, encoding="utf-8")
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return txt_path, meta_path, True


def load_all(
    *,
    registry: Registry | None = None,
    settings: Settings | None = None,
    force: bool = False,
    only_schemes: list[str] | None = None,
) -> list[SourceDoc]:
    cfg = settings or get_settings()
    reg = registry or load_registry()
    paths.ensure_dirs()

    session = _session(cfg)
    docs: list[SourceDoc] = []
    first = True

    for spec, document, _target in iter_documents(reg):
        if only_schemes and spec.scheme_id not in only_schemes:
            continue
        if not first and cfg.request_delay_seconds > 0:
            time.sleep(cfg.request_delay_seconds)
        first = False

        doc = fetch_doc(spec, document, registry=reg, session=session, settings=cfg)
        _txt, _meta, rewritten = write_doc(doc, force=force)
        docs.append(doc)

        ingest_record(
            doc_id=doc.doc_id,
            scheme_id=doc.scheme_id,
            doc_type=doc.doc_type,
            status=doc.status,
            reason=doc.reason,
            url=doc.source_url,
            content_hash=doc.content_hash[:12],
            chars=len(doc.text),
            tables=len(doc.tables),
            rewritten=rewritten,
        )

    failures = [d for d in docs if d.status != "ok"]
    paths.INGEST_ERRORS.write_text(
        json.dumps(
            [
                {
                    "doc_id": d.doc_id,
                    "scheme_id": d.scheme_id,
                    "url": d.source_url,
                    "reason": d.reason,
                }
                for d in failures
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _print_summary(docs, reg, force)
    return docs


def _print_summary(docs: list[SourceDoc], registry: Registry, force: bool) -> None:
    ok = [d for d in docs if d.status == "ok"]
    print(f"AMC: {registry.amc}   slots={len(docs)}   ok={len(ok)}   failed={len(docs) - len(ok)}   force={force}")
    print(f"{'doc_id':<26} {'type':<12} {'status':<7} {'chars':>7} {'tbl':>4}  reason")
    for doc in docs:
        print(
            f"{doc.doc_id:<26} {doc.doc_type:<12} {doc.status:<7} {len(doc.text):>7} "
            f"{len(doc.tables):>4}  {doc.reason}"
        )
    print(f"\nsnapshots: {paths.RAW_DIR}")
    print(f"failures:  {paths.INGEST_ERRORS}")


def load_snapshots(*, only_schemes: list[str] | None = None) -> list[SourceDoc]:
    """Read every successful snapshot back off disk (input for phase P2)."""
    docs: list[SourceDoc] = []
    if not paths.RAW_DIR.exists():
        return docs
    for meta_path in sorted(paths.RAW_DIR.glob(f"*/*{paths.META_SUFFIX}")):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if payload.get("status") != "ok":
            continue
        if only_schemes and payload.get("scheme_id") not in only_schemes:
            continue
        txt_path = meta_path.with_name(meta_path.name[: -len(paths.META_SUFFIX)] + ".txt")
        if not txt_path.exists():
            continue
        docs.append(
            SourceDoc(
                doc_id=payload["doc_id"],
                scheme_id=payload["scheme_id"],
                scheme_name=payload["scheme_name"],
                category=payload["category"],
                doc_type=payload["doc_type"],
                title=payload.get("title", ""),
                source_url=payload["source_url"],
                publisher=payload.get("publisher", ""),
                retrieved_at=payload["retrieved_at"],
                content_hash=payload["content_hash"],
                raw_path=str(txt_path.relative_to(paths.PROJECT_ROOT)),
                text=txt_path.read_text(encoding="utf-8"),
                stats=payload.get("stats") or {},
            )
        )
    return docs


def main() -> int:
    docs = load_all()
    ok = sum(1 for d in docs if d.status == "ok")
    logger.info("ingest finished at %s: %s/%s ok", now_iso(), ok, len(docs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
