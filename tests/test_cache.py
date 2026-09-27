"""Tests for the P10 demo answer cache.

The whole point of this component is the rule it must never break: only `answered` results
are stored or served. A cached refusal would mean the guardrails could be changed without any
visible effect, so that rule is tested from every direction - at write time, at read time, and
through the pipeline.
"""

from src.llm.cache import CACHE_VERSION, DEMO_QUESTIONS, AnswerCache
from src.types import Answer, Citation, QueryTrace

import json
import pytest

Q = "What is the expense ratio of the HDFC Large Cap Fund - Direct - Growth?"


def make_answer(status: str, text: str) -> Answer:
    return Answer(
        status=status,
        text=text,
        citations=[Citation(label="c", url="https://groww.in/a", doc_type="scheme_page", retrieved_at="2026-09-27")]
        if status == "answered"
        else [],
        last_updated="2026-09-27" if status == "answered" else None,
        trace=QueryTrace(
            query_hash="h",
            guards={},
            hits=[],
            threshold=0.3,
            threshold_passed=status == "answered",
            cited_doc_ids=[],
            latency_ms={},
        ),
    )


def answered(text: str = "Expense ratio is 1.03%.") -> Answer:
    return make_answer("answered", text)


@pytest.fixture
def cache(tmp_path):
    return AnswerCache(tmp_path / "answer_cache.json")


@pytest.fixture
def no_side_effects(monkeypatch, tmp_path):
    """Keep the query log out of the real logs directory."""
    monkeypatch.setattr("src.observability.paths.QUERY_LOG", tmp_path / "queries.jsonl")
    return tmp_path / "queries.jsonl"


class TestOnlyAnsweredIsCacheable:
    @pytest.mark.parametrize("status", ["refused", "pii_rejected", "insufficient_context", "error"])
    def test_a_non_answered_result_is_never_stored(self, cache, status):
        assert cache.put(Q, make_answer(status, "no")) is False
        cache.save()
        assert len(cache) == 0

    def test_a_refusal_is_never_served_even_if_the_file_contains_one(self, cache, tmp_path):
        """Defence in depth: a hand-edited or stale file must not make the bot serve a
        refusal, so the status is re-checked on read and not only on write."""
        cache.path.write_text(
            json.dumps(
                {
                    "version": CACHE_VERSION,
                    "entries": {
                        AnswerCache.key_for(Q): {
                            "status": "refused",
                            "text": "I cannot help with that.",
                            "citations": [],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        assert AnswerCache(cache.path).get(Q) is None


class TestRoundTrip:
    def test_an_answered_result_survives_a_save_and_load(self, cache, tmp_path):
        cache.put(Q, answered())
        cache.save()
        got = AnswerCache(cache.path).get(Q)
        assert got is not None
        assert got.status == "answered"
        assert got.text == "Expense ratio is 1.03%."
        assert got.citations[0].url == "https://groww.in/a"
        assert got.last_updated == "2026-09-27"

    def test_a_miss_returns_none(self, cache):
        cache.put(Q, answered())
        assert cache.get("a completely different question") is None

    def test_the_key_is_a_hash_so_the_file_cannot_leak_query_text(self, cache):
        cache.put(Q, answered())
        cache.save()
        body = cache.path.read_text(encoding="utf-8")
        # The question itself is never stored, only its hash - the same privacy property the
        # query log relies on. (The answer text naturally repeats words from the question;
        # that is the answer, not a leak of the query.)
        assert Q not in body
        reloaded = AnswerCache(cache.path)
        assert len(reloaded) == 1
        assert reloaded.get(Q) is not None
        key = AnswerCache.key_for(Q)
        assert len(key) == 12 and all(c in "0123456789abcdef" for c in key)


class TestRobustness:
    def test_a_missing_file_is_an_empty_cache_not_an_error(self, tmp_path):
        assert len(AnswerCache(tmp_path / "absent.json")) == 0

    def test_a_corrupt_file_is_an_empty_cache_not_an_error(self, cache):
        cache.path.write_text("{not json", encoding="utf-8")
        assert len(AnswerCache(cache.path)) == 0

    def test_a_wrong_version_file_is_ignored(self, cache):
        cache.put(Q, answered())
        cache.path.write_text(json.dumps({"version": 999, "entries": {}}), encoding="utf-8")
        assert AnswerCache(cache.path).get(Q) is None

    def test_clear_empties_the_cache(self, cache):
        cache.put(Q, answered())
        cache.clear()
        assert cache.get(Q) is None


class TestPipelineIntegration:
    def test_a_cached_answer_is_replayed_and_labelled_as_a_hit(self, monkeypatch, tmp_path, no_side_effects):
        from src.pipeline import answer_question

        cache = AnswerCache(tmp_path / "answer_cache.json")
        cache.put(Q, answered("Expense ratio is 1.03%."))
        cache.save()
        monkeypatch.setattr("src.llm.cache.ANSWER_CACHE_PATH", cache.path)

        answer = answer_question(Q)
        assert answer.status == "answered"
        assert answer.trace.guards.get("cache") == "hit"

    def test_use_cache_false_ignores_a_populated_cache(self, monkeypatch, tmp_path, no_side_effects):
        from src.pipeline import answer_question

        cache = AnswerCache(tmp_path / "answer_cache.json")
        cache.put(Q, answered("Expense ratio is 1.03%."))
        cache.save()
        monkeypatch.setattr("src.llm.cache.ANSWER_CACHE_PATH", cache.path)

        answer = answer_question(Q, use_cache=False)
        assert answer.trace.guards.get("cache") != "hit"

    def test_a_refusal_is_never_replayed_even_when_the_cache_is_on(self, monkeypatch, tmp_path, no_side_effects):
        """The cache is consulted only after the ingress guards, so a question that must be
        refused is refused on every run."""
        from src.pipeline import answer_question

        monkeypatch.setattr("src.llm.cache.ANSWER_CACHE_PATH", tmp_path / "answer_cache.json")
        answer = answer_question("Should I buy the HDFC Balanced Advantage Fund?")
        assert answer.status == "refused"
        assert answer.trace.guards.get("cache") != "hit"

    def test_a_broken_cache_file_does_not_break_the_answer(self, monkeypatch, tmp_path, no_side_effects):
        from src.pipeline import answer_question

        broken = tmp_path / "answer_cache.json"
        broken.write_text("{not json", encoding="utf-8")
        monkeypatch.setattr("src.llm.cache.ANSWER_CACHE_PATH", broken)
        answer = answer_question("What is the exit load on HDFC Small Cap Fund?")
        assert answer.status == "answered"


class TestDemoQuestions:
    def test_there_are_exactly_three_and_they_are_distinct(self):
        assert len(DEMO_QUESTIONS) == 3
        assert len(set(DEMO_QUESTIONS)) == 3

    def test_every_demo_question_is_answerable_so_the_cache_can_fill(self):
        """An advice question must be refused and a refusal is never cached, so including one
        would leave the cache permanently short by one."""
        from src.guardrails.intent import classify_intent

        for question in DEMO_QUESTIONS:
            assert classify_intent(question)[0] == "factual", question

    def test_the_refusal_demo_question_is_not_one_of_the_cached_ones(self):
        from src.llm.cache import DEMO_REFUSAL_QUESTION

        assert DEMO_REFUSAL_QUESTION not in DEMO_QUESTIONS

    def test_questions_reports_coverage_in_demo_order(self, cache):
        cache.put(DEMO_QUESTIONS[2], answered())
        cache.put(DEMO_QUESTIONS[0], answered())
        assert cache.questions() == [DEMO_QUESTIONS[0], DEMO_QUESTIONS[2]]
