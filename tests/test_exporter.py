"""Tests for the P10 source manifests.

The manifests are generated from the indexed corpus, so the aggregation is the part that
has to be right: one row per document, not per chunk, with the document-level fields and the
token statistics carried through.
"""

from src.ingest.exporter import SOURCES_CSV_HEADER, aggregate, render_markdown
from src.types import Chunk

import pytest


def make_chunk(chunk_id: str, **overrides) -> Chunk:
    base = dict(
        chunk_id=chunk_id,
        text=f"Body of {chunk_id}",
        token_count=100,
        scheme_id="S1",
        scheme_name="HDFC Large Cap Fund - Direct - Growth",
        category="Large Cap",
        doc_type="scheme_page",
        section="Fees",
        source_url="https://groww.in/a",
        publisher="Groww",
        retrieved_at="2026-09-27",
        chunk_index=0,
        splitter="LabelValueSplitter",
        content_hash="h",
    )
    base.update(overrides)
    return Chunk(**base)


class TestAggregate:
    def test_groups_chunks_into_one_row_per_document(self):
        chunks = [
            make_chunk("S1__overview__0000"),
            make_chunk("S1__overview__0001"),
            make_chunk("S1__overview__0002"),
            make_chunk("S3__sebi_elss__0000"),
        ]
        rows = aggregate(chunks)
        assert [r.source_id for r in rows] == ["S1__overview", "S3__sebi_elss"]

    def test_carries_chunk_count_and_token_statistics(self):
        chunks = [
            make_chunk("S1__overview__0000", token_count=50),
            make_chunk("S1__overview__0001", token_count=150),
            make_chunk("S1__overview__0002", token_count=100),
        ]
        row = aggregate(chunks)[0]
        assert row.chunk_count == 3
        assert row.token_total == 300
        assert row.token_min == 50
        assert row.token_max == 150

    def test_keeps_document_level_fields_from_the_first_chunk(self):
        row = aggregate([make_chunk("S1__overview__0000")])[0]
        assert row.publisher == "Groww"
        assert row.doc_type == "scheme_page"
        assert row.source_url == "https://groww.in/a"

    def test_the_grouping_key_survives_renumbering(self):
        """`__NNNN` is the chunk index, so two documents cannot collide and a re-chunk that
        renumbers chunks still lands in the same row."""
        assert aggregate([make_chunk("S1__overview__0042")])[0].source_id == "S1__overview"
        assert len(aggregate([make_chunk("S1__a__0000"), make_chunk("S1__b__0000")])) == 2


class TestRenderMarkdown:
    def test_reports_corpus_totals_and_every_host(self):
        chunks = [
            make_chunk("S1__overview__0000", source_url="https://groww.in/a"),
            make_chunk("S1__overview__0001", source_url="https://groww.in/a"),
            make_chunk("S3__sebi_elss__0000", source_url="https://www.amfiindia.com/b", scheme_id="S3"),
        ]
        rows = aggregate(chunks)
        md = render_markdown(rows, chunks)
        assert "- Documents: **2**" in md
        assert "- Chunks: **3**" in md
        assert "`groww.in` | 2" in md
        assert "`www.amfiindia.com` | 1" in md

    def test_escapes_pipes_so_a_url_cannot_break_the_table(self):
        chunk = make_chunk("S1__overview__0000", scheme_name="Flexi | Large Cap")
        md = render_markdown(aggregate([chunk]), [chunk])
        # The raw pipe must not survive into the table cell unescaped.
        assert "Flexi \\| Large Cap" in md

    def test_every_document_row_links_to_its_url(self):
        chunk = make_chunk("S1__overview__0000", source_url="https://groww.in/a")
        assert "](https://groww.in/a)" in render_markdown(aggregate([chunk]), [chunk])


class TestCsvHeader:
    def test_header_matches_the_dataclass_fields(self):
        import csv
        import io

        from src.ingest.exporter import SourceRow

        fields = list(SOURCES_CSV_HEADER)
        row = SourceRow("s", "p", "S1", "n", "c", "scheme_page", "u", "2026-09-27", 1, 10, 10, 10)
        assert list(row.as_csv().keys()) == fields
        # Round-trips through a real DictWriter without an extra or missing column.
        buffer = io.StringIO()
        csv.DictWriter(buffer, fieldnames=fields).writeheader()
        assert len(buffer.getvalue().strip().split(",")) == len(fields)


class TestExportMissingCorpus:
    def test_export_explains_how_to_build_when_there_is_no_chunks_file(self, monkeypatch, tmp_path):
        from src.ingest import exporter

        monkeypatch.setattr(exporter, "CHUNKS_PATH", tmp_path / "nope.jsonl")
        with pytest.raises(FileNotFoundError, match="--stage chunk"):
            exporter.export()
