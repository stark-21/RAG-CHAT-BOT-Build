"""Tests for the P9 evaluation harness.

The harness is the only thing standing between a threshold change and a silently worse bot,
so the tuning logic gets tested directly rather than only through a full corpus run.
"""

import sys

import pytest

from eval.run_eval import answer_body, host_of, is_factual_case, numbers_in, recommend


def row(top_k: int, threshold: float, fact_found: int, *, total_facts=6, abstained=4, total_nonsense=4):
    return {
        "top_k": top_k,
        "threshold": threshold,
        "fact_found": fact_found,
        "factual_total": total_facts,
        "nonsense_abstained": abstained,
        "nonsense_total": total_nonsense,
    }


class TestNumbers:
    def test_extracts_amounts_and_ratios(self):
        assert "1.03" in numbers_in("Expense ratio is 1.03% of assets")
        assert "100" in numbers_in("minimum SIP is INR 100 per month")

    def test_ignores_bare_text(self):
        assert not numbers_in("Exit load applies within one year")
        assert numbers_in("no digits here") == set()


class TestAnswerBody:
    def test_strips_the_pipeline_footer(self):
        """The pipeline appends `Last updated from sources: <date>`; that is provenance and
        must not be counted as answer prose."""
        text = "Exit load is 1%.\n\nLast updated from sources: 2026-09-27"
        assert answer_body(text) == "Exit load is 1%."

    def test_leaves_a_real_answer_untouched(self):
        assert answer_body("Exit load is 1%.") == "Exit load is 1%."

    def test_keeps_the_answer_when_the_stub_labels_its_own_extract(self):
        """The stub used to put its label and the extract on one line, which made the whole
        response look like a footer and hid the answer from this filter."""
        body = answer_body("From the official source (no LLM available):\nExit load is 1%.")
        assert "Exit load is 1%." in body


class TestIsFactualCase:
    @pytest.mark.parametrize(
        "case,expected",
        [
            ({"expect_status": "answered"}, True),
            ({"expect_status": "refused"}, False),
            ({"expect_status": "pii_rejected"}, False),
            ({"expect_status": "insufficient_context"}, False),
        ],
    )
    def test_only_answered_cases_are_factual(self, case, expected):
        assert is_factual_case(case) is expected


class TestHostOf:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.amfiindia.com/a", "www.amfiindia.com"),
            ("https://groww.in/b", "groww.in"),
            ("https://investor.sebi.gov.in/c", "investor.sebi.gov.in"),
        ],
    )
    def test_strips_scheme_and_www_untouched(self, url, expected):
        assert host_of(url) == expected


class TestRecommend:
    def test_maximises_fact_recall_first(self):
        rows = [row(3, t, 3) for t in (0.20, 0.30, 0.60)] + [row(10, t, 6) for t in (0.20, 0.60)]
        assert recommend(rows)["top_k"] == 10

    def test_never_recommends_a_looser_gate_for_equal_recall(self):
        """The original rule picked the lowest threshold that abstained, which cannot add
        recall but does admit more irrelevant chunks. On this corpus that would have moved
        the default from 0.30 to 0.20 for nothing."""
        rows = [row(5, 0.20, 5), row(5, 0.30, 5), row(5, 0.60, 5)]
        assert recommend(rows)["threshold"] == 0.30

    def test_keeps_the_default_when_recall_is_flat_in_the_threshold(self):
        from src.config import get_settings

        current = get_settings()
        rows = [row(5, t, 5) for t in (0.20, 0.30, 0.40, 0.60)]
        result = recommend(rows)
        assert result["threshold"] == current.min_similarity
        assert "not identifiable" in result["reason"]

    def test_falls_back_to_the_default_when_a_probe_is_retrieved(self):
        from src.config import get_settings

        current = get_settings()
        rows = [row(5, 0.30, 5, abstained=2)]
        result = recommend(rows)
        assert result["top_k"] == current.top_k
        assert result["threshold"] == current.min_similarity


class TestEvalBypassesTheDemoCache:
    def test_the_harness_never_measures_the_cache(self, monkeypatch):
        """Three eval questions are also demo-cache questions. A cache hit returns an empty
        trace with no retrieved context, so without `use_cache=False` the harness silently
        measures the cache instead of the pipeline - which is what dropped retrieval-layer
        recovery from 6/6 to 4/6 the moment the cache was first seeded."""
        import eval.run_eval as harness

        seen: list = []

        def fake_answer_question(query, **kwargs):
            seen.append(kwargs.get("use_cache"))
            raise SystemExit  # stop at the first call; the signature is what matters

        monkeypatch.setattr(harness, "answer_question", fake_answer_question)
        # `main()` parses sys.argv, which under pytest holds pytest's own arguments and
        # would make argparse exit before the harness ever asks a question.
        monkeypatch.setattr(sys, "argv", ["run_eval"])
        with pytest.raises(SystemExit):
            harness.main()

        assert seen, "the harness never called answer_question"
        assert all(flag is False for flag in seen), f"expected use_cache=False on every call, got {seen}"
