"""HTML/PDF cleaning for the facts-only corpus.

Two jobs beyond ordinary boilerplate stripping:

1. Keep tables *inline and re-parsable* by wrapping each kept table in
   `<TABLE n> ... </TABLE>` sentinels with a pipe-delimited rendering, so phase P2 can
   rebuild the header + row structure instead of re-deriving it from flat text.
2. Enforce the facts-only product rule by dropping performance and ranking blocks
   (PRD 2.3 non-goals, PRD 10.1). Return figures must never reach retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO

from bs4 import BeautifulSoup, NavigableString

BOILERPLATE_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "iframe",
    "svg",
    "button",
    "select",
)

HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")

# Distributor pages bury the scheme content in a wrapper div and leave the rest of the
# page as site chrome. Observed on the live pages: `div.pw14MainWrapper` around
# `div.layout-main`. Preferring the largest such container cut one scheme page from
# 724 extracted lines to 237.
CONTENT_ROOT_SELECTORS = ("main", '[role="main"]')
CONTENT_ROOT_CLASS_HINTS = (
    "mainwrapper",
    "layout-main",
    "main-wrapper",
    "maincontent",
    "content-wrapper",
    "page-content-wrapper",
    "article-body",
    "post-content",
    "site-main",
)
MIN_ROOT_CHARS = 800

BOILERPLATE_LINE_PATTERNS = (
    re.compile(
        r"^\s*(download the app|about us|pricing|blog|media & press|careers|"
        r"help & support|trust & safety|investor relations|contact us|disclaimer)\b",
        re.I,
    ),
    re.compile(r"cookie", re.I),
    re.compile(r"^\s*(accept all|accept cookies|we use cookies|consent)\b", re.I),
    re.compile(r"^\s*(log in|login|sign in|sign up)\s*$", re.I),
    re.compile(r"^\s*©.*\d{4}.*$", re.I),
    # Breadcrumb trails, e.g. "Home / Personal Finance and Investment / ... / Content".
    # These are navigation, and leaving them in produced a chunk whose entire body was a
    # breadcrumb on the SEBI pages.
    re.compile(r"^\s*home(\s*/\s*[^/]{2,60})+\s*$", re.I),
    re.compile(r"^\s*(home|content|topic of interest|you are here)\s*$", re.I),
)

# Markers that identify a performance / ranking table. Portfolio-holding tables do not
# match any of these, so factual holdings data is kept.
PERFORMANCE_TABLE_MARKERS = (
    "historic returns",
    "would've become",
    "total investment",
    "category average",
    "fund returns",
    "fund size(cr)",
    "rank (",
)

# Markers for performance / ranking prose lines.
PERFORMANCE_LINE_PATTERNS = (
    re.compile(r"historic returns", re.I),
    re.compile(r"return calculator", re.I),
    re.compile(r"category average", re.I),
    re.compile(r"\bfund returns\b", re.I),
    re.compile(r"\b(?:1|3|5|7|10)y returns\b", re.I),
    re.compile(r"^\s*rank\b", re.I),
    re.compile(r"would've become", re.I),
    re.compile(r"^\s*returns\s*$", re.I),
)

EXCLUSION_SENTINEL = "[performance data excluded - facts-only policy]"
TABLE_OPEN = "<TABLE {n}>"
TABLE_CLOSE = "</TABLE {n}>"

# The scheme pages render a return ticker as many tiny nodes, so the label and the
# figure end up on different lines. These patterns mark the start of such a block and
# the standalone figures that follow it are removed with it.
PERF_CONTEXT_PATTERNS = (
    re.compile(r"annualised|annualized|\bcagr\b|since inception", re.I),
    re.compile(r"returns on your|return calculator|historic returns", re.I),
    re.compile(r"^\d+[DWMY]$"),
    re.compile(r"^All$", re.I),
)
_STANDALONE_NUMBER = re.compile(r"^[+-]?[\d,]+(?:\.\d+)?$")
# A return ticker renders as a signed figure and its percent sign on separate lines.
# A *negative holding weight* also carries a sign, but always inside a table row, so
# only whole lines are treated as ticker values here.
_SIGNED_PCT_ALONE = re.compile(r"^[+-]\d[\d.,]*\s*%$")
_SIGNED_NUMBER_ALONE = re.compile(r"^[+-]\d[\d.,]*$")

_WS = re.compile(r"[^\S\n]+")
_WS_NL = re.compile(r"[^\S\n]*\n[^\S\n]*")
_BLANK = re.compile(r"\n{2,}")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


@dataclass
class CleanResult:
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    title: str = ""
    stats: dict = field(default_factory=dict)


def normalize_ws(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_NL.sub("\n", text)
    text = _WS.sub(" ", text)
    return _BLANK.sub("\n", text).strip()


def table_to_rows(table) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [normalize_ws(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    return rows


def rows_to_text(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell for cell in row) for row in rows)


def is_performance_table(rows: list[list[str]]) -> bool:
    flat = " ".join(" ".join(row) for row in rows).lower()
    return any(marker in flat for marker in PERFORMANCE_TABLE_MARKERS)


def is_performance_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in PERFORMANCE_LINE_PATTERNS)


def is_boilerplate_line(line: str) -> bool:
    return any(pattern.search(line) for pattern in BOILERPLATE_LINE_PATTERNS)


def pick_content_root(soup) -> tuple[object, str]:
    """Return `(root, how)` for the element most likely to hold the real content.

    Falls back to `<body>` and then to the soup itself, so fragments used in tests and
    pages without landmarks still work.
    """
    for selector in CONTENT_ROOT_SELECTORS:
        element = soup.select_one(selector)
        if element is not None and len(element.get_text(strip=True)) > MIN_ROOT_CHARS:
            return element, selector

    best = None
    for element in soup.find_all(["div", "section", "article", "main"]):
        classes = " ".join(element.get("class") or []).lower()
        if any(hint in classes for hint in CONTENT_ROOT_CLASS_HINTS):
            if len(element.get_text(strip=True)) > MIN_ROOT_CHARS:
                if best is None or len(element.get_text(strip=True)) > len(best.get_text(strip=True)):
                    best = element
    if best is not None:
        return best, "class:" + ".".join((best.get("class") or [])[:3])

    body = soup.find("body")
    if body is not None:
        return body, "body"
    return soup, "soup"


def mark_headings(root) -> int:
    """Replace h1-h6 with `#`-prefixed markers so P2 has real structure to split on."""
    marked = 0
    for tag in root.find_all(HEADING_TAGS):
        text = normalize_ws(tag.get_text(" ", strip=True))
        if not text:
            continue
        tag.replace_with(NavigableString(f"\n{'#' * int(tag.name[1])} {text}\n"))
        marked += 1
    return marked


def sentinelise_tables(root, *, include_performance: bool) -> tuple[list[list[list[str]]], int, int]:
    kept: list[list[list[str]]] = []
    found = 0
    dropped = 0
    for index, table in enumerate(root.find_all("table")):
        found += 1
        rows = table_to_rows(table)
        if not include_performance and is_performance_table(rows):
            table.replace_with(NavigableString(EXCLUSION_SENTINEL))
            dropped += 1
            continue
        kept.append(rows)
        table.replace_with(
            NavigableString(
                f"\n{TABLE_OPEN.format(n=index)}\n{rows_to_text(rows)}\n{TABLE_CLOSE.format(n=index)}\n"
            )
        )
    return kept, found, dropped


def finalise_lines(lines: list[str], *, include_performance: bool = False) -> tuple[list[str], int]:
    """Drop boilerplate, performance prose, and return-ticker fragments.

    Returns `(kept_lines, dropped_count)`.
    """
    items = [line for line in lines if line]
    kept: list[str] = []
    dropped = 0
    in_return_block = False
    index = 0

    while index < len(items):
        line = items[index]
        following = items[index + 1] if index + 1 < len(items) else ""

        if not include_performance:
            if is_performance_line(line) or any(p.search(line) for p in PERF_CONTEXT_PATTERNS):
                dropped += 1
                in_return_block = True
                index += 1
                continue
            if _SIGNED_PCT_ALONE.match(line):
                dropped += 1
                in_return_block = True
                index += 1
                continue
            if _SIGNED_NUMBER_ALONE.match(line) and following.strip() == "%":
                dropped += 2
                in_return_block = True
                index += 2
                continue
            if in_return_block and (line == "%" or _STANDALONE_NUMBER.match(line)):
                dropped += 1
                index += 1
                continue

        if is_boilerplate_line(line):
            dropped += 1
            index += 1
            continue
        if kept and kept[-1] == line:
            index += 1
            continue

        kept.append(line)
        in_return_block = False
        index += 1

    return kept, dropped


def clean_html(html: str, *, include_performance: bool = False) -> CleanResult:
    soup = BeautifulSoup(html or "", "lxml")
    title = normalize_ws(soup.title.get_text()) if soup.title else ""

    for tag in soup(list(BOILERPLATE_TAGS)):
        tag.decompose()

    root, root_kind = pick_content_root(soup)
    headings_marked = mark_headings(root)
    kept_tables, tables_found, tables_dropped = sentinelise_tables(
        root, include_performance=include_performance
    )

    raw_lines = root.get_text("\n").split("\n")
    lines, lines_dropped = finalise_lines(
        [normalize_ws(raw) for raw in raw_lines],
        include_performance=include_performance,
    )

    text = "\n".join(lines)
    return CleanResult(
        text=text,
        tables=kept_tables,
        title=title,
        stats={
            "chars": len(text),
            "lines": len(lines),
            "headings_marked": headings_marked,
            "content_root": root_kind,
            "tables_found": tables_found,
            "tables_kept": len(kept_tables),
            "tables_dropped": tables_dropped,
            "lines_dropped": lines_dropped,
        },
    )


def clean_pdf_text(text: str, *, include_performance: bool = False) -> CleanResult:
    joined = _HYPHEN_BREAK.sub(r"\1\2", (text or "").replace("\r\n", "\n").replace("\r", "\n"))
    lines, lines_dropped = finalise_lines(
        [normalize_ws(raw) for raw in joined.split("\n")],
        include_performance=include_performance,
    )
    body = "\n".join(lines)
    return CleanResult(
        text=body,
        tables=[],
        title="",
        stats={
            "chars": len(body),
            "lines": len(lines),
            "tables_found": 0,
            "tables_kept": 0,
            "tables_dropped": 0,
            "lines_dropped": lines_dropped,
        },
    )


def extract_pdf(content: bytes) -> CleanResult:
    """Best-effort pdfplumber extraction; returns an empty result on failure."""
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("pdfplumber is required for PDF sources") from exc

    pages: list[str] = []
    tables: list[list[list[str]]] = []
    with pdfplumber.open(BytesIO(content)) as pdf:
        info = getattr(pdf, "metadata", None) or {}
        title = normalize_ws(str(info.get("Title") or ""))
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
            for raw in page.extract_tables() or []:
                rows = [[normalize_ws(str(cell)) if cell is not None else "" for cell in row] for row in raw]
                rows = [row for row in rows if any(cell for cell in row)]
                if rows:
                    tables.append(rows)

    result = clean_pdf_text("\n".join(pages))
    result.title = title
    result.tables = tables
    result.stats["pages"] = len(pages)
    result.stats["tables_found"] = len(tables)
    return result
