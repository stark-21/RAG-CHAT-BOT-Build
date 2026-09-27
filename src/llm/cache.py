"""Replay cache for a small, fixed set of verified answers.

This exists for one reason: a three-minute demo must not depend on a network call, a paid
API key, or a model that has not finished loading. It is deliberately tiny and deliberately
dumb.

Two rules keep it from becoming a way to show something untrue:

* Only `answered` results are ever written or served. A refusal, a PII rejection, an
  insufficient-context demotion, or an error is a live safety decision - caching one would
  mean a later change to the guardrails silently did nothing, because the old verdict kept
  being replayed. That is the failure mode this design exists to prevent.
* A cache hit is labelled in the trace (`guards["cache"] = "hit"`), so a reviewer can see
  from the answer itself that it was replayed rather than generated.

Seeding requires a real provider. Run it with `LLM_API_KEY` set:

    python -m src.llm.cache --seed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.observability import hash_query
from src.paths import ANSWER_CACHE_PATH
from src.types import Answer

#: The only status a demo may replay. Everything else is a safety decision, not content.
CACHEABLE_STATUSES = frozenset({"answered"})

#: The three questions the demo script asks, in order. All three are factual on purpose:
#: an advice question must be refused, and a refusal is never cached, so including one here
#: would leave the cache permanently incomplete.
DEMO_QUESTIONS = (
    "What is the expense ratio of the HDFC Large Cap Fund - Direct - Growth?",
    "What is the exit load on HDFC Small Cap Fund - Direct - Growth?",
    "What is the minimum SIP amount for the HDFC Flexi Cap Fund?",
)

#: Asked live during the demo to show the advice guardrail. Deliberately *not* in the cache:
#: it must be refused on every run, and caching a refusal would hide any later change to the
#: guardrails.
DEMO_REFUSAL_QUESTION = "Should I buy the HDFC Balanced Advantage Fund?"

#: Cache format version, so a stale or hand-edited file is rejected rather than trusted.
CACHE_VERSION = 1


class AnswerCache:
    """A JSON file of verified answers, keyed by the same query hash the logs use.

    The key is the hash and never the query text, so the cache file - like
    `logs/queries.jsonl` - cannot leak what a user typed.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        # Resolved at call time, not at import time, so a test - or a deliberate
        # relocation of the cache - can redirect it.
        self.path = Path(path if path is not None else ANSWER_CACHE_PATH)
        self._entries: dict[str, dict] = {}
        self._loaded = False

    def load(self) -> "AnswerCache":
        """Read the file if it exists. A missing, unreadable, or wrong-version file is an
        empty cache, never an exception: a bad cache must not break the app."""
        if self._loaded:
            return self
        self._loaded = True
        if not self.path.exists():
            return self
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return self
        if not isinstance(payload, dict) or payload.get("version") != CACHE_VERSION:
            return self
        entries = payload.get("entries")
        if isinstance(entries, dict):
            # Keep only entries shaped like an entry this module wrote. The filter has to
            # match `_entry_from_answer` exactly: an earlier version tested for an "answer"
            # key that `save` never writes, which silently discarded every entry on load and
            # made the cache a permanent no-op.
            self._entries = {
                str(k): v
                for k, v in entries.items()
                if isinstance(v, dict) and "status" in v and "text" in v
            }
        return self

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"version": CACHE_VERSION, "entries": self._entries}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.path

    def __len__(self) -> int:
        self.load()
        return len(self._entries)

    @staticmethod
    def key_for(query: str) -> str:
        return hash_query(query)

    def get(self, query: str) -> Answer | None:
        """Return a replayable answer, or None. A cached entry whose status is no longer
        cacheable is ignored even if it is in the file, so a hand-edited or stale file cannot
        make the bot serve a refusal."""
        self.load()
        entry = self._entries.get(self.key_for(query))
        if not entry:
            return None
        if entry.get("status") not in CACHEABLE_STATUSES:
            return None
        return _answer_from_entry(entry)

    def put(self, query: str, answer: Answer) -> bool:
        """Store an answer. Returns False when it was not stored, so callers can report it."""
        if answer.status not in CACHEABLE_STATUSES:
            return False
        self.load()
        self._entries[self.key_for(query)] = _entry_from_answer(answer)
        return True

    def clear(self) -> None:
        self.load()
        self._entries.clear()

    def questions(self) -> list[str]:
        """The demo questions, in order, skipping any that are not currently cached."""
        self.load()
        return [q for q in DEMO_QUESTIONS if self.key_for(q) in self._entries]


def _entry_from_answer(answer: Answer) -> dict:
    return {
        "status": answer.status,
        "text": answer.text,
        "last_updated": answer.last_updated,
        "citations": [
            {
                "label": c.label,
                "url": c.url,
                "doc_type": c.doc_type,
                "retrieved_at": c.retrieved_at,
            }
            for c in answer.citations
        ],
    }


def _answer_from_entry(entry: dict) -> Answer:
    from src.types import Citation

    trace = None
    if entry.get("trace"):
        from src.types import QueryTrace

        trace = QueryTrace(**entry["trace"])
    return Answer(
        status=entry["status"],
        text=entry.get("text", ""),
        citations=[Citation(**c) for c in entry.get("citations", [])],
        last_updated=entry.get("last_updated"),
        trace=trace,
    )


def seed(questions: tuple[str, ...] = DEMO_QUESTIONS, *, use_cache: bool = False) -> tuple[int, int]:
    """Run the live pipeline for each question and store the answers.

    `use_cache` defaults to False so seeding always regenerates: re-seeding from the cache
    would just copy the old file back over itself and look like a successful fresh run.
    """
    from src.pipeline import answer_question

    cache = AnswerCache()
    stored = 0
    for question in questions:
        answer = answer_question(question, use_cache=use_cache)
        if cache.put(question, answer):
            stored += 1
            print(f"cached [{answer.status}] {question}")
        else:
            print(f"NOT cached [{answer.status}] {question}")
    cache.save()
    print(f"{stored}/{len(questions)} cached -> {cache.path}")
    return stored, len(questions)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the verified demo answer cache.")
    parser.add_argument("--seed", action="store_true", help="generate and store the demo answers")
    parser.add_argument("--list", action="store_true", help="show which demo questions are cached")
    parser.add_argument("--clear", action="store_true", help="empty the cache")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="with --seed, bypass the cache while generating so the run is genuinely live",
    )
    parser.add_argument("--query", default=None, help="print the cached answer for one question")
    args = parser.parse_args(argv)

    cache = AnswerCache()
    if args.clear:
        cache.clear()
        cache.save()
        print(f"cleared {cache.path}")
        return 0
    if args.seed:
        seed(use_cache=args.no_cache is False)
        return 0
    if args.query:
        answer = cache.get(args.query)
        if answer is None:
            print("not cached")
            return 1
        print(answer.text)
        return 0
    # Default, and --list: report coverage so a demo gap is visible before it is live.
    print(f"{len(cache)}/{len(DEMO_QUESTIONS)} demo answers cached ({cache.path})")
    for question in DEMO_QUESTIONS:
        mark = "yes" if question in cache.questions() else "NO "
        print(f"  [{mark}] {question}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
