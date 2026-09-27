"""Phase P2 chunking: structure-aware splitters chosen from the real corpus.

Design decisions and the evidence behind them are recorded in
`docs/chunking_decision.md`. Summary of what the data forced:

* The scheme pages are **heading-delimited sections of label/value pairs**
  (`Expense ratio` then `1.03%`), not tables, so a `LabelValueSplitter` matters more
  than a table-aware one. Fee and exit-load figures are plain label/value lines.
* Only one table per scheme page survives the facts-only filter (holdings), and it must
  never be split mid-row with its header lost, hence `TableAwareSplitter`.
* The AMFI / SEBI pages are heading-delimited prose, hence `SectionSplitter`.
* Every chunk is prefixed with `{scheme_name} â€” {heading path}` so a retrieved fact is
  never ambiguous about which of the five schemes it came from. The prefix names the leaf
  heading, extended with as many ancestors as it takes to make that leaf unique inside its
  document: these pages restate the same term in several sections, and a bare leaf left the
  citation and the section-keyed dedupe cap unable to tell `Understand terms > Tax` from
  `Tax implication > Tax`.
"""

from __future__ import annotations

import math
import re

from src.ingest.cleaner import EXCLUSION_SENTINEL, TABLE_CLOSE, TABLE_OPEN
from src.types import Chunk, SourceDoc

MAX_TOKENS = 500
MIN_TOKENS = 24
OVERLAP = 0.12
CHARS_PER_TOKEN = 4.0
WORDS_PER_TOKEN = 1 / 1.35
QUALIFY_SEP = " > "

# `TABLE_OPEN` is "<TABLE {n}>", so the literal prefixes have to be written out rather
# than derived by formatting with an empty number, which would yield "<TABLE >".
TABLE_OPEN_PREFIX = "<TABLE "
TABLE_CLOSE_PREFIX = "</TABLE "

# A heading marker, but never a data line that merely starts with a hash: groww renders
# a fund rank as "#2 in India", which is a fact, not a section.
_HEADING = re.compile(r"^(#{1,6})\s+([A-Za-z].*)$")
_QUESTION = re.compile(r"^(.{6,180}?\?)\s*$")
# A short, non-sentence line that introduces a value: "Min. for SIP", "Expense ratio".
_LABEL = re.compile(r"^[A-Z][A-Za-z0-9 .()/&'%-]{2,60}$")
_SENTENCE_END = re.compile(r"[.!?]\s*$")

DISPATCH: dict[str, str] = {
    "scheme_page": "LabelValueSplitter",
    "education": "SectionSplitter",
    "guide": "SectionSplitter",
    "faq": "QASplitter",
    "kim_sid": "SectionSplitter",
    "factsheet": "SectionSplitter",
    "fees": "TableAwareSplitter",
    "riskometer": "SectionSplitter",
    "other": "RecursiveSplitter",
}


def count_tokens(text: str) -> int:
    """Conservative token estimate for `all-MiniLM-L6-v2` (512-token limit)."""
    if not text:
        return 0
    by_words = len(text.split()) / WORDS_PER_TOKEN
    by_chars = len(text) / CHARS_PER_TOKEN
    return int(math.ceil(max(by_words, by_chars)))


def parse_blocks(text: str) -> list[dict]:
    """Split cleaned text into ordered heading / table / prose blocks.

    Table rows are pulled back out of the `<TABLE n>` sentinel block so the splitter
    sees real rows rather than a flat string.
    """
    lines = [line.strip() for line in (text or "").splitlines()]
    blocks: list[dict] = []
    path: list[str] = []
    levels: list[int] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue

        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            label = heading.group(2).strip()
            # Keep only ancestors that are genuinely shallower. The source pages skip
            # heading levels (h5 straight under h2), so slicing by level alone would
            # leave a stale ancestor such as "Holdings ( 50 )" on every later section.
            keep = [i for i, existing in enumerate(levels) if existing < level]
            path = [path[i] for i in keep] + [label]
            levels = [levels[i] for i in keep] + [level]
            blocks.append({"kind": "heading", "text": label, "level": level, "path": list(path)})
            index += 1
            continue

        if line.startswith(EXCLUSION_SENTINEL):
            blocks.append({"kind": "excluded", "text": "", "path": list(path)})
            index += 1
            continue

        if line.startswith(TABLE_OPEN_PREFIX):
            table_path = list(path)
            rows: list[list[str]] = []
            index += 1
            while index < len(lines) and not lines[index].startswith(TABLE_CLOSE_PREFIX):
                row = [cell.strip() for cell in lines[index].split("|")]
                if any(row):
                    rows.append(row)
                index += 1
            index += 1
            blocks.append({"kind": "table", "rows": rows, "path": table_path})
            continue

        blocks.append({"kind": "line", "text": line, "path": list(path)})
        index += 1

    return blocks


def split_table(rows: list[list[str]], max_tokens: int = MAX_TOKENS) -> list[list[list[str]]]:
    """Group table rows into chunks, never splitting a row.

    The header row is repeated in every group so a rate is never retrieved without the
    slab label it belongs to.
    """
    if not rows:
        return []
    header, body = rows[0], rows[1:]
    groups: list[list[list[str]]] = []
    current: list[list[str]] = []
    for row in body:
        candidate = [*current, row]
        rendered = "\n".join(" | ".join(item) for item in [header, *candidate])
        if current and count_tokens(rendered) > max_tokens:
            groups.append([header, *current])
            current = [row]
            continue
        current = candidate
    if current:
        groups.append([header, *current])
    return groups or [[header]]


def render_table(rows: list[list[str]]) -> str:
    return "\n".join(" | ".join(cell for cell in row) for row in rows)


def merge_label_value_pairs(lines: list[str]) -> list[str]:
    """Join `Label` + `value` line pairs into one `Label: value` line.

    Scheme pages state every fact as two consecutive lines. Merging them keeps the fact
    inside a single chunk and gives the embedder a natural query-shaped string.
    """
    merged: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index]
        following = lines[index + 1] if index + 1 < len(lines) else ""

        is_label = bool(_LABEL.match(current)) and not _SENTENCE_END.search(current)
        value_like = bool(following) and (
            following.startswith(("â‚¹", "Rs", "INR"))
            or bool(re.match(r"^[+\-]?[\d.,]+", following))
            or len(following) <= 60
        )
        if is_label and value_like and not following.startswith("#"):
            merged.append(f"{current}: {following}")
            index += 2
            continue
        merged.append(current)
        index += 1
    return merged


def split_section(
    body_lines: list[str],
    heading_path: list[str],
    max_tokens: int = MAX_TOKENS,
    overlap: float = OVERLAP,
) -> list[tuple[str, list[str]]]:
    """Pack a section's lines into token-bounded overlapping units."""
    units: list[tuple[str, list[str]]] = []
    current: list[str] = []

    for line in body_lines:
        if count_tokens("\n".join(current + [line])) > max_tokens and current:
            units.append((heading_path[-1] if heading_path else "", current))
            carry = _carry_overlap(current, overlap, max_tokens)
            current = [*carry, line]
            continue
        current.append(line)

    if current:
        units.append((heading_path[-1] if heading_path else "", current))

    if not units and heading_path:
        units.append((heading_path[-1], []))
    return units


def _carry_overlap(lines: list[str], overlap: float, max_tokens: int) -> list[str]:
    if overlap <= 0:
        return []
    target = int(max_tokens * overlap)
    carry: list[str] = []
    for line in reversed(lines):
        if count_tokens("\n".join([line, *carry])) > target:
            break
        carry.insert(0, line)
    return carry


def split_qa(lines: list[str], max_tokens: int = MAX_TOKENS) -> list[list[str]]:
    """One unit per question and its answer, so the question text is embedded."""
    units: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if _QUESTION.match(line) and current:
            units.append(current)
            current = [line]
            continue
        current.append(line)
        if count_tokens("\n".join(current)) > max_tokens:
            units.append(current)
            current = []
    if current:
        units.append(current)
    return units


def split_recursive(text: str, max_tokens: int = MAX_TOKENS, overlap: float = OVERLAP) -> list[str]:
    """Dependency-free recursive character splitter for narrative prose."""
    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    return _recursive(text, max_tokens, overlap, separators)


def _recursive(text: str, max_tokens: int, overlap: float, separators: list[str]) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]

    for index, separator in enumerate(separators):
        if separator not in text:
            continue
        parts = [part for part in text.split(separator) if part.strip()]
        if len(parts) <= 1:
            continue
        if index < len(separators) - 1:
            parts = [part + separator for part in parts[:-1]] + [parts[-1]]

        chunks: list[str] = []
        buffer = ""
        for part in parts:
            if buffer and count_tokens(buffer + part) > max_tokens:
                chunks.append(buffer.strip())
                tail = _carry_overlap(buffer.split("\n"), overlap, max_tokens)
                buffer = "\n".join(tail) + part
                continue
            buffer += part
        if buffer.strip():
            chunks.append(buffer.strip())
        return chunks
    return [text]


def _is_table_row(line: str) -> bool:
    return " | " in line


def _section_units(
    doc: SourceDoc, merge_pairs: bool, max_tokens: int = MAX_TOKENS
) -> list[tuple[list[str], list[str]]]:
    """Return `(heading_path, lines)` units for a document.

    A table is the unit that must not be split naively, so both sentinelised HTML tables
    and any run of pipe-delimited lines are routed through `split_table`. Without the
    second case a long holdings table emitted continuation chunks that started mid-table
    with no header, losing the column meaning of every value in them.
    """
    blocks = parse_blocks(doc.text)
    units: list[tuple[list[str], list[str]]] = []
    path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            units.append((list(path), list(buffer)))
            buffer.clear()

    def add_table(table_path: list[str], rows: list[list[str]]) -> None:
        # Leave room for the `{scheme} â€” {section}` prefix so a group rarely has to be
        # re-split after its header is attached.
        leaf = table_path[-1] if table_path else (doc.title or doc.scheme_name)
        reserve = count_tokens(f"{doc.scheme_name} — {leaf}\n") + 8
        budget = max(24, max_tokens - reserve)
        for group in split_table(rows, max_tokens=budget):
            units.append((list(table_path), [render_table(group)]))

    index = 0
    while index < len(blocks):
        block = blocks[index]
        kind = block["kind"]

        if kind == "heading":
            flush()
            path = list(block["path"])
            index += 1
            continue

        if kind == "excluded":
            index += 1
            continue

        if kind == "table":
            flush()
            add_table(block["path"], block["rows"])
            index += 1
            continue

        if kind == "line" and _is_table_row(block["text"]):
            flush()
            table_path = list(block["path"])
            rows: list[list[str]] = []
            while (
                index < len(blocks)
                and blocks[index]["kind"] == "line"
                and _is_table_row(blocks[index]["text"])
            ):
                rows.append([cell.strip() for cell in blocks[index]["text"].split("|")])
                index += 1
            add_table(table_path, rows)
            continue

        buffer.append(block["text"])
        index += 1

    flush()
    if merge_pairs:
        units = [(path, merge_label_value_pairs(lines)) for path, lines in units]
    return units


def _section_labels(paths: list[list[str]]) -> dict[tuple[str, ...], str]:
    """Map each heading path to the label used in the chunk prefix.

    The leaf heading is the label whenever it is unique inside its document, because the
    full outline path on these distributor pages nests unrelated sections several levels
    deep and reads as `Holdings ( 50 ) > Understand terms > Tax`.

    Where a leaf repeats within one document, the shortest ancestor prefix that makes the
    name unique is prepended instead. Those pages restate the same term in several places
    (`Tax` in the glossary, in the fee table, and under tax implications), and a bare leaf
    made both the user-facing citation and the section-keyed dedupe cap ambiguous about
    which section a chunk came from.
    """
    groups: dict[str, set[tuple[str, ...]]] = {}
    for path in paths:
        if path:
            # Keyed case-insensitively so `Exit Load` and `Exit load` are qualified as the
            # collision they are, rather than passing as distinct because the pages differ
            # in capitalisation. The section-keyed dedupe cap folds case too, so a
            # case-only difference would otherwise still group unrelated sections.
            groups.setdefault(path[-1].casefold(), set()).add(tuple(path))

    labels: dict[tuple[str, ...], str] = {}
    for leaf, members in groups.items():
        if len(members) == 1:
            labels[next(iter(members))] = next(iter(members))[-1]
            continue
        for path in members:
            # Start past the leaf: every member of the group shares the leaf, so depth 1
            # never separates them. Take the shallowest depth at which all members yield
            # distinct suffixes, which falls back to the full path on a genuine tie.
            depth = 2
            while depth <= len(path) and len(
                {QUALIFY_SEP.join(p[-depth:]) for p in members}
            ) != len(members):
                depth += 1
            labels[path] = QUALIFY_SEP.join(path[-depth:])
    return labels


def _is_contiguous_run(haystack: list[str], needle: list[str]) -> bool:
    """Whether `needle` appears inside `haystack` as an unbroken ordered run of lines."""
    if not needle or len(needle) > len(haystack):
        return False
    span = len(needle)
    return any(haystack[start : start + span] == needle for start in range(len(haystack) - span + 1))


def _drop_contained(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks whose body only restates lines already carried by an earlier chunk.

    The distributor pages repeat a figure in more than one place, so a section such as
    `Exit load, stamp duty and tax > Exit load` can emit a chunk that is a verbatim
    restatement of a line already in the exit-load table chunk. Order-sensitive run matching
    is used so a table with the same lines in another order is never mistaken for a
    duplicate, and the earliest container is the one kept.
    """
    kept: list[Chunk] = []
    kept_bodies: list[list[str]] = []
    for chunk in chunks:
        body = [line.strip() for line in chunk.text.splitlines()[1:] if line.strip()]
        if body and any(_is_contiguous_run(other, body) for other in kept_bodies):
            continue
        kept.append(chunk)
        kept_bodies.append(body)
    return kept


def chunk_document(doc: SourceDoc, max_tokens: int = MAX_TOKENS, overlap: float = OVERLAP) -> list[Chunk]:
    """Split one document into retrieval-sized, fully-stamped chunks."""
    doc_type = doc.doc_type or "other"
    splitter_name = DISPATCH.get(doc_type, DISPATCH["other"])
    chunks: list[Chunk] = []
    seen: set[str] = set()

    def emit(section: list[str], body_lines: list[str], splitter: str) -> None:
        body = "\n".join(line for line in body_lines if line).strip()
        if not body:
            return
        # The label is the leaf heading, qualified with its ancestors only where the leaf
        # repeats inside this document. See `_section_labels`.
        leaf = labels.get(tuple(section), section[-1] if section else "")
        if leaf:
            header = f"{doc.scheme_name} — {leaf}"
        else:
            header = doc.scheme_name or doc.title
        # The context prefix is mandatory, so it is shortened rather than allowed to
        # breach the cap when a caller asks for a very small budget.
        if count_tokens(f"{header}\n") + 2 >= max_tokens:
            header = header[: max(8, int(CHARS_PER_TOKEN * (max_tokens - 2)))].rstrip()
        text = f"{header}\n{body}"
        if count_tokens(text) > max_tokens:
            # A table group that still overflows must keep its header on every piece.
            keep = body_lines[0] if body_lines and _is_table_row(body_lines[0]) else ""
            budget = max(1, max_tokens - count_tokens(f"{header}\n{keep}\n" if keep else f"{header}\n"))
            for piece in split_recursive(body, budget, overlap):
                _append(f"{header}\n{keep}\n{piece}" if keep else f"{header}\n{piece}", header, splitter)
            return
        _append(text, header, splitter)

    def _append(text: str, header: str, splitter: str) -> None:
        fingerprint = f"{header}::{text}"
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}__{index:04d}",
                text=text,
                token_count=count_tokens(text),
                scheme_id=doc.scheme_id,
                scheme_name=doc.scheme_name,
                category=doc.category,
                doc_type=doc_type,
                section=header or doc.title,
                source_url=doc.source_url,
                publisher=doc.publisher,
                retrieved_at=doc.retrieved_at,
                chunk_index=index,
                splitter=splitter,
                content_hash=doc.content_hash,
            )
        )

    if splitter_name == "TableAwareSplitter":
        units = [
            (path, body, splitter_name)
            for path, lines in _section_units(doc, merge_pairs=False, max_tokens=max_tokens)
            for _name, body in split_section(lines, path, max_tokens, overlap)
        ]
    elif splitter_name == "QASplitter":
        units = [
            (path, unit, splitter_name)
            for path, lines in _section_units(doc, merge_pairs=False, max_tokens=max_tokens)
            for unit in split_qa(lines, max_tokens)
        ]
    elif splitter_name == "RecursiveSplitter":
        units = [([], piece.splitlines(), splitter_name) for piece in split_recursive(doc.text, max_tokens, overlap)]
    elif splitter_name == "LabelValueSplitter":
        units = [
            (path, body, splitter_name)
            for path, lines in _section_units(doc, merge_pairs=True, max_tokens=max_tokens)
            for _name, body in split_section(lines, path, max_tokens, overlap)
        ]
    else:
        units = [
            (path, body, splitter_name)
            for path, lines in _section_units(doc, merge_pairs=False, max_tokens=max_tokens)
            for _name, body in split_section(lines, path, max_tokens, overlap)
        ]

    # Collected before the first emit so a repeated leaf is known to be repeated.
    labels = _section_labels([path for path, _body, _splitter in units])
    for path, body, splitter in units:
        emit(path, body, splitter)

    return _drop_contained([chunk for chunk in chunks if chunk.token_count >= 1])

