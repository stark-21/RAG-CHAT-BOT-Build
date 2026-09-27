"""Time the P10 demo beats end to end so `docs/demo_script.md` carries measured numbers.

This is a rehearsal harness, not a test: it asserts nothing, it just prints the wall time of
each beat plus the total, twice, so the script's 180-second budget is checked against reality
instead of against a guess. Run it with the cache warm (see the demo script) - it exercises
the same three questions plus the guardrail beats.

    python -m scripts.rehearse_demo
"""

from __future__ import annotations

import time

from src.pipeline import answer_question

# (beat label, question, what the presenter is doing while it runs)
# Note: expanding "Why this answer?" is a UI interaction, not a query, so it costs no system
# time and is deliberately not a beat here.
BEATS: tuple[tuple[str, str, str], ...] = (
    ("beat2", "What is the expense ratio of the HDFC Large Cap Fund - Direct - Growth?", "factual answer + citation"),
    ("beat3", "Should I buy the HDFC Balanced Advantage Fund?", "advice refusal"),
    ("beat3", "What is the exit load on HDFC Small Cap Fund?", "same fund, factual -> answered"),
    ("beat4", "My PAN is ABCDE1234F, what is the exit load?", "PII rejection"),
    ("beat4", "What is the expense ratio of Parag Parag Flexi Cap?", "out-of-scope refusal"),
    ("beat5", "kjhgfdsa qwerty zxcvbn", "nonsense abstains"),
)

#: Deliberately *not* one of the three cached demo questions. A cached question returns in
#: about a millisecond without ever loading the embedding model, so warming up with one looks
#: instant and then hands the whole cost to the first uncached retrieval mid-demo.
WARMUP_QUESTION = "Is there a lock-in period on the HDFC ELSS Tax Saver Fund?"

BUDGET_SECONDS = 180.0


def run_once(label: str) -> float:
    print(f"\n--- {label} ---")
    started = time.perf_counter()
    slowest = ("", 0.0)
    for beat, question, note in BEATS:
        beat_started = time.perf_counter()
        answer = answer_question(question)
        seconds = time.perf_counter() - beat_started
        if seconds > slowest[1]:
            slowest = (beat, seconds)
        print(f"  {beat:<6} {answer.status:<21} {seconds * 1000:7.0f} ms  ({note})")
    total = time.perf_counter() - started

    # Speech is the real cost; the retrieval is the part being measured here.
    spoken = 3.0 * len(BEATS)
    print(f"  total system time : {total:.1f} s")
    print(f"  slowest beat      : {slowest[0]} at {slowest[1] * 1000:.0f} ms")
    print(f"  within {BUDGET_SECONDS:.0f} s budget for system time: {total <= BUDGET_SECONDS}")
    return total


def main() -> int:
    # Warm the embedding model first, exactly as the demo script instructs, so the first
    # beat is not paying a one-off load that has been observed at up to 13 s. The warm-up
    # question must not be a cached one, or it returns instantly and loads nothing.
    print("warming the embedding model with an uncached question...")
    warm_started = time.perf_counter()
    warm = answer_question(WARMUP_QUESTION)
    warm_seconds = time.perf_counter() - warm_started
    print(f"  {WARMUP_QUESTION}")
    print(
        f"  warm-up took {warm_seconds:.1f} s -> [{warm.status}], "
        f"cache={warm.trace.guards.get('cache')}  (not part of the demo)"
    )
    if warm.trace.guards.get("cache") == "hit":
        print("  WARNING: the warm-up was served from cache, so the model is still cold.")

    first = run_once("rehearsal 1")
    second = run_once("rehearsal 2")
    print(f"\nrehearsal 1: {first:.1f} s | rehearsal 2: {second:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
