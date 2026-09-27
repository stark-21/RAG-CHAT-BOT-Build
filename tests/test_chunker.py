"""Phase P2 tests: structure-aware splitting, token caps, and chunk metadata.

Several tests are regression tests for defects found by running the splitter over the
real corpus in `data/raw/`; each one names the failure it prevents.
"""

from __future__ import annotations

import json

import pytest

from src.ingest import chunker
from src.ingest.chunker import (
    DISPATCH,
    _drop_contained,
    _section_labels,
    chunk_document,
    count_tokens,
    merge_label_value_pairs,
    parse_blocks,
    split_qa,
    split_recursive,
    split_section,
    split_table,
)
from src.types import SourceDoc

SCHEME_NAME = "HDFC Large Cap Fund - Direct - Growth"

SCHEME_PAGE = """NAV: 25 Sep '26
1,189.08
Min. for SIP
₹100
Fund size (AUM)
₹39,933.37 Cr
Expense ratio
1.03%
Rating
4
## Holdings ( 50 )
<TABLE 1>
Name | Sector | Instruments | Assets
ICICI Bank Ltd | Financial | Equity | 10.05%
HDFC Bank Ltd | Financial | Equity | 6.88%
Net Payables | Unspecified | Net Payables | -0.10%
</TABLE 1>
See All
### Minimum investments
Min. for 1st investment
₹100
Min. for SIP
₹100
## Understand terms
##### Expense ratio
A fee payable to a mutual fund house for managing your mutual fund investments.
##### Exit load
A fee payable to a mutual fund house for exiting a fund before the completion of the exit load period.
### Exit Load
Exit load of 1% if redeemed within 1 year
### Fund house
#2 in India
HDFC Mutual Fund
"""


def make_doc(text: str = SCHEME_PAGE, doc_type: str = "scheme_page", **kwargs) -> SourceDoc:
    defaults = dict(
        doc_id="S1__overview",
        scheme_id="S1",
        scheme_name=SCHEME_NAME,
        category="Large cap",
        doc_type=doc_type,
        title=SCHEME_NAME,
        source_url="https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth",
        publisher="Groww (brief-supplied distributor)",
        retrieved_at="2026-09-27T00:00:00+00:00",
        content_hash="abc123",
        raw_path="data/raw/S1/S1__overview.txt",
        text=text,
    )
    defaults.update(kwargs)
    return SourceDoc(**defaults)


# --------------------------------------------------------------------------- tokens


def test_count_tokens_is_conservative_and_monotonic():
    assert count_tokens("") == 0
    short = count_tokens("Expense ratio: 1.03%")
    long = count_tokens("Expense ratio: 1.03%. " * 40)
    assert 0 < short < long
    # Never under-count relative to the character heuristic.
    text = "a" * 400
    assert count_tokens(text) >= len(text) / 4


# -------------------------------------------------------------------------- tables


def test_split_table_repeats_header_on_every_group():
    rows = [["Name", "Sector", "Assets"]] + [
        [f"Holding {i}", "Equity", f"{i}.0%"] for i in range(60)
    ]
    groups = split_table(rows, max_tokens=80)

    assert len(groups) > 1
    for group in groups:
        assert group[0] == ["Name", "Sector", "Assets"]


def test_split_table_never_splits_a_row():
    rows = [["Name", "Assets"]] + [[f"Fund {i}", "1.0%"] for i in range(40)]
    for group in split_table(rows, max_tokens=40):
        for row in group:
            assert len(row) == 2


def test_split_table_handles_header_only_and_empty():
    assert split_table([]) == []
    assert split_table([["Name", "Assets"]]) == [[["Name", "Assets"]]]


# --------------------------------------------------------------------- parse_blocks


def test_parse_blocks_reads_the_table_sentinel():
    """Regression: sentinel matching used `TABLE_OPEN.format(n="")`, yielding
    `"<TABLE >"`, so `<TABLE 1>` was parsed as a prose line and its rows were split
    arbitrarily with no header."""
    blocks = parse_blocks(SCHEME_PAGE)
    tables = [b for b in blocks if b["kind"] == "table"]

    assert len(tables) == 1
    assert tables[0]["rows"][0] == ["Name", "Sector", "Instruments", "Assets"]
    assert "".join(b["text"] for b in blocks if b["kind"] == "line").count("<TABLE") == 0


def test_parse_blocks_treats_hash_rank_as_data_not_heading():
    """Regression: `#2 in India` is a fund rank, and was being read as a heading,
    which reset the section path and produced a section label of `#2 in India`."""
    blocks = parse_blocks(SCHEME_PAGE)

    assert not any(b["kind"] == "heading" and b["text"].startswith("#") for b in blocks)
    assert "Fund house" in [b["text"] for b in blocks if b["kind"] == "heading"]


def test_parse_blocks_path_ignores_stale_ancestors_when_levels_are_skipped():
    """Regression: the pages nest an h5 under an h2, so slicing the path by level left
    `Holdings ( 50 )` attached to every later section."""
    blocks = parse_blocks(SCHEME_PAGE)
    expense = next(
        b for b in blocks if b["kind"] == "heading" and b["text"] == "Expense ratio"
    )

    assert expense["path"] == ["Understand terms", "Expense ratio"]


# ------------------------------------------------------------ label / value merging


def test_merge_label_value_pairs_joins_facts():
    merged = merge_label_value_pairs(
        ["Min. for SIP", "₹100", "Expense ratio", "1.03%", "Fund size (AUM)", "₹39,933.37 Cr"]
    )

    assert merged == [
        "Min. for SIP: ₹100",
        "Expense ratio: 1.03%",
        "Fund size (AUM): ₹39,933.37 Cr",
    ]


def test_merge_label_value_pairs_leaves_sentences_alone():
    lines = [
        "A fee payable to a mutual fund house for managing investments.",
        "Very High risk.",
    ]

    assert merge_label_value_pairs(lines) == lines


# ------------------------------------------------------------------- section / qa


def test_split_section_respects_the_token_cap():
    lines = [f"Line {i} of the fund facts section." for i in range(200)]
    units = split_section(lines, ["Facts"], max_tokens=60, overlap=0.1)

    assert len(units) > 1
    for _name, body in units:
        assert count_tokens("\n".join(body)) <= 60


def test_split_qa_isolates_each_question():
    units = split_qa(
        [
            "What is the exit load?",
            "It is 1% within 1 year.",
            "Is there a lock-in?",
            "ELSS has a 3 year lock-in.",
        ]
    )

    assert len(units) == 2
    assert units[0][0].endswith("exit load?")
    assert units[1][0].endswith("lock-in?")


def test_split_recursive_does_not_glue_lines_together():
    """Regression: the overlap tail was joined with `""`, producing lines such as
    `... 1.03%SRF Ltd | Materials ...`."""
    text = "\n".join(f"Row {i} | Equity | {i}.00%" for i in range(200))
    pieces = split_recursive(text, max_tokens=40, overlap=0.15)

    assert len(pieces) > 1
    for piece in pieces:
        for line in piece.splitlines():
            assert re_row(line), f"glued row: {line!r}"


def re_row(line: str) -> bool:
    return line.startswith("Row ") or line.count("|") == 2


def test_split_recursive_is_a_noop_under_the_cap():
    assert split_recursive("Expense ratio: 1.03%", max_tokens=500) == ["Expense ratio: 1.03%"]
    assert split_recursive("", max_tokens=500) == []


# ----------------------------------------------------------------- chunk_document


def test_chunk_document_contract():
    chunks = chunk_document(make_doc())

    assert chunks
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"S1__overview__{index:04d}"
        assert chunk.text.startswith(SCHEME_NAME)
        assert chunk.scheme_id == "S1"
        assert chunk.scheme_name == SCHEME_NAME
        assert chunk.source_url.endswith("hdfc-large-cap-fund-direct-growth")
        assert chunk.publisher and chunk.retrieved_at and chunk.content_hash
        assert chunk.doc_type == "scheme_page"
        assert chunk.section
        assert chunk.splitter == "LabelValueSplitter"
        assert chunk.token_count <= 500
        assert chunk.token_count == count_tokens(chunk.text)


def test_chunk_document_keeps_key_facts_self_contained():
    blob = "\n".join(chunk.text for chunk in chunk_document(make_doc()))

    for fact in ("Expense ratio: 1.03%", "Min. for SIP: ₹100", "Fund size (AUM): ₹39,933.37 Cr"):
        assert fact in blob


def test_chunk_document_repeats_holdings_header_on_every_table_chunk():
    chunks = [c for c in chunk_document(make_doc()) if " | " in c.text]
    table_chunks = [c for c in chunks if "ICICI" in c.text or "HDFC Bank" in c.text or "Net Payables" in c.text]

    assert table_chunks
    for chunk in table_chunks:
        assert "Name | Sector | Instruments | Assets" in chunk.text


def test_chunk_document_preserves_negative_holdings():
    blob = "\n".join(c.text for c in chunk_document(make_doc()))

    assert "Net Payables | Unspecified | Net Payables | -0.10%" in blob


def test_chunk_document_never_exceeds_a_tight_cap():
    for cap in (60, 120, 500):
        chunks = chunk_document(make_doc(), max_tokens=cap)
        assert chunks
        for chunk in chunks:
            assert chunk.token_count <= cap
            assert chunk.text.startswith(SCHEME_NAME)


def test_chunk_document_drops_duplicate_text():
    doc = make_doc(text="Expense ratio\n1.03%\n" * 40)
    texts = [c.text for c in chunk_document(doc)]

    assert len(texts) == len(set(texts))


def test_chunk_document_survives_empty_document():
    assert chunk_document(make_doc(text="")) == []


@pytest.mark.parametrize(
    ("doc_type", "splitter"),
    [
        ("scheme_page", "LabelValueSplitter"),
        ("education", "SectionSplitter"),
        ("guide", "SectionSplitter"),
        ("faq", "QASplitter"),
        ("kim_sid", "SectionSplitter"),
        ("fees", "TableAwareSplitter"),
    ],
)
def test_dispatch_matches_the_splitter_used(doc_type, splitter):
    assert DISPATCH[doc_type] == splitter
    text = "## Fees\nWhat is the exit load?\nIt is 1% within 1 year."
    chunks = chunk_document(make_doc(text=text, doc_type=doc_type))

    assert chunks
    assert chunks[0].splitter == splitter


def test_dispatch_has_a_fallback():
    assert DISPATCH["other"] == "RecursiveSplitter"
    chunks = chunk_document(
        make_doc(text="A plain paragraph with no headings at all.", doc_type="something-new")
    )

    assert chunks[0].splitter == "RecursiveSplitter"


def test_fee_table_document_keeps_slab_header():
    """The real corpus has no fee table, so the `fees` doc type is covered by a
    synthetic KIM-style slab table."""
    doc = make_doc(
        text=(
            "## Fee slabs\n"
            "<TABLE 0>\n"
            "Purchase amount | Exit load\n"
            "0-1 year | 1%\n"
            "1-2 years | 0.5%\n"
            "Above 2 years | 0%\n"
            "</TABLE 0>"
        ),
        doc_type="fees",
    )
    chunks = [c for c in chunk_document(doc) if " | " in c.text]

    assert chunks
    for chunk in chunks:
        assert "Purchase amount | Exit load" in chunk.text


# --------------------------------------------------------------- real corpus gate


@pytest.mark.skipif(
    not (chunker.__file__ and __import__("src.paths", fromlist=["RAW_DIR"]).RAW_DIR.exists()),
    reason="no snapshots in data/raw",
)
def test_real_snapshots_chunk_without_navigation_bleed():
    from src.ingest.loader import load_snapshots

    docs = load_snapshots()
    if not docs:
        pytest.skip("no snapshots present")

    nav_terms = ("Intraday", "ETF Screener", "Buy now, pay later", "Demat Account", "F&O")
    for doc in docs:
        chunks = chunk_document(doc)
        assert chunks, doc.doc_id
        for chunk in chunks:
            assert chunk.token_count <= 500
            assert chunk.text.startswith(chunk.scheme_name)
            assert "<TABLE" not in chunk.text
            assert not any(term in chunk.text for term in nav_terms), (
                f"{chunk.chunk_id} leaked site navigation"
            )


def test_chunks_jsonl_is_valid_when_present():
    from src import paths

    if not paths.CHUNKS_PATH.exists():
        pytest.skip("chunks.jsonl not built yet")

    lines = [line for line in paths.CHUNKS_PATH.read_text(encoding="utf-8").splitlines() if line]
    assert lines
    seen = set()
    for line in lines:
        payload = json.loads(line)
        assert payload["chunk_id"] not in seen
        seen.add(payload["chunk_id"])
        for field in (
            "text",
            "token_count",
            "scheme_id",
            "scheme_name",
            "doc_type",
            "section",
            "source_url",
            "publisher",
            "retrieved_at",
            "splitter",
            "content_hash",
        ):
            assert payload.get(field) not in (None, ""), field


# --- Repeated leaf headings: disambiguated labels, and restatements removed -----------


def test_a_unique_leaf_keeps_its_bare_name():
    """Short labels are the norm; the parent is only added to resolve a real collision."""
    labels = _section_labels([["Holdings ( 50 )"], ["Fund house"]])

    assert labels[("Holdings ( 50 )",)] == "Holdings ( 50 )"
    assert labels[("Fund house",)] == "Fund house"


def test_a_repeated_leaf_is_qualified_by_its_parent():
    """`Tax` in the glossary and `Tax` under tax implications must not share a label."""
    labels = _section_labels(
        [
            ["Understand terms", "Tax"],
            ["Exit load, stamp duty and tax", "Tax"],
        ]
    )

    assert labels[("Understand terms", "Tax")] == "Understand terms > Tax"
    assert labels[("Exit load, stamp duty and tax", "Tax")] == (
        "Exit load, stamp duty and tax > Tax"
    )


def test_a_leaf_differing_only_by_capitalisation_still_collides():
    """`Exit Load` and `Exit load` are the same name to a reader and to the case-folded
    section key used by the dedupe cap, so capitalisation alone must not pass them as
    distinct sections."""
    labels = _section_labels(
        [
            ["Understand terms", "Exit load"],
            ["Understand terms", "Exit Load"],
        ]
    )

    qualified = set(labels.values())

    assert len(qualified) == 2
    assert "Understand terms > Exit load" in qualified
    assert "Understand terms > Exit Load" in qualified


def test_qualification_stops_at_the_shallowest_depth_that_separates_the_group():
    """Over-qualifying would reintroduce the `Holdings ( 50 ) > Understand terms > Tax`
    labels the leaf-only rule was chosen to avoid."""
    labels = _section_labels(
        [
            ["A", "B", "Tax"],
            ["C", "D", "Tax"],
        ]
    )

    assert labels[("A", "B", "Tax")] == "B > Tax"
    assert labels[("C", "D", "Tax")] == "D > Tax"


def test_a_leaf_with_no_distinguishing_ancestor_falls_back_to_the_full_path():
    """Two sections can share every ancestor but the leaf at different depths; the label
    degrades to the whole outline rather than colliding again."""
    labels = _section_labels(
        [
            ["Holdings", "Details", "Tax"],
            ["Details", "Tax"],
        ]
    )

    qualified = set(labels.values())

    assert len(qualified) == 2
    assert "Holdings > Details > Tax" in qualified
    assert "Details > Tax" in qualified


def test_an_empty_path_is_not_qualified():
    """A document with no headings yields a single unnamed unit that must stay unnamed."""
    assert _section_labels([[]]) == {}


def test_a_chunk_that_merely_restates_an_earlier_chunk_is_dropped():
    """The pages repeat a figure under more than one heading; the second mention of a line
    already carried in full adds no retrievable fact."""
    doc = make_doc(
        SCHEME_PAGE
        + "## Exit load, stamp duty and tax\n"
        + "#### Exit load\n"
        + "Exit load of 1% if redeemed within 1 year\n"
    )

    chunks = chunk_document(doc)
    bodies = [chunk.text for chunk in chunks]

    assert sum("Exit load of 1% if redeemed within 1 year" in text for text in bodies) == 1


def test_a_chunk_is_kept_when_its_body_is_not_already_present():
    """Containment must be all-or-nothing: a chunk adding any new line is kept even if it
    shares its opening lines with another."""
    doc = make_doc(SCHEME_PAGE + "## Exit load detail\nExit load of 1% if redeemed within 1 year\nand waived thereafter\n")

    bodies = [chunk.text for chunk in chunk_document(doc)]

    assert any("and waived thereafter" in text for text in bodies)


def test_reordered_table_lines_are_not_mistaken_for_a_duplicate():
    """Containment is order-sensitive, so a table listing the same rows in another order
    is real content and must survive."""
    first = chunker.Chunk(
        chunk_id="S1__overview__0000",
        text="Scheme - Growth - A\nName | Sector\nICICI | Financial\nHDFC | Financial",
        token_count=20,
        scheme_id="S1",
        scheme_name="Scheme - Growth - A",
        category="Large Cap",
        doc_type="scheme_page",
        section="Scheme - Growth - A",
        source_url="https://example.invalid",
        publisher="p",
        retrieved_at="2026-09-27",
        chunk_index=0,
        splitter="TableAwareSplitter",
        content_hash="h1",
    )
    second = chunker.Chunk(
        chunk_id="S1__overview__0001",
        text="Scheme - Growth - B\nName | Sector\nHDFC | Financial\nICICI | Financial",
        token_count=20,
        scheme_id="S1",
        scheme_name="Scheme - Growth - B",
        category="Large Cap",
        doc_type="scheme_page",
        section="Scheme - Growth - B",
        source_url="https://example.invalid",
        publisher="p",
        retrieved_at="2026-09-27",
        chunk_index=1,
        splitter="TableAwareSplitter",
        content_hash="h2",
    )

    kept = _drop_contained([first, second])

    assert [chunk.chunk_id for chunk in kept] == ["S1__overview__0000", "S1__overview__0001"]


def test_chunks_carry_no_empty_label_after_qualification():
    """The prefix is the citation label shown to a user, so it must never collapse to just
    the scheme name once a path is resolved."""
    chunks = chunk_document(make_doc(SCHEME_PAGE))

    assert chunks
    for chunk in chunks:
        assert chunk.section.strip()
        assert chunk.section.startswith(chunk.scheme_name)
