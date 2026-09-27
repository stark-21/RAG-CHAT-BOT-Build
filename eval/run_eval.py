"""P9 evaluation harness: runs the PRD section 12 set end to end, reports metrics, sweeps
the retrieval thresholds, and writes `docs/sample_qa.md` from the real run.

Every question goes through `src.pipeline.answer_question` with the guardrails active -
there is no bypass path here, because an evaluation that can skip the guardrails does not
measure the system that ships.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.ingest.build_index import read_chunks  # noqa: E402
from src.llm.generator import numbers_in  # noqa: E402
from src.pipeline import answer_question  # noqa: E402
from src.retrieval.retriever import retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEST_SET = ROOT / "eval" / "test_set.yaml"
SAMPLE_QA = ROOT / "docs" / "sample_qa.md"

THRESHOLD_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
TOP_K_GRID = (3, 5, 8, 10)

# Queries that must never be answered. A threshold is only acceptable if it still turns
# these away, which is why they are separate from the eval set: they measure abstention,
# not accuracy.
NONSENSE_PROBES = (
    "asdkjh qwerty zxcvb",
    "what is the meaning of life",
    "tomorrow's weather on mars",
    "explain the wobble of a quantum teaspoon",
)

USER_AGENT = get_settings().user_agent

# Sentences are counted on the answer body only; the citation and "last updated" lines are
# provenance rather than answer text.
_DATE_LINE = re.compile(r"^\s*(last updated|source:|from the official source)", re.I)
_SOURCE_LINE = re.compile(r"^\s*(source|citation|https?://)", re.I)


def answer_body(text: str) -> str:
    """The answer proper, with provenance lines removed.

    The generator appends `Last updated from sources: 2026-09-27` and a `source:` line.
    Those carry a date, and a date is a number: counting it as answer content produced
    novelty violations for a footer the system is required to print, and inflated the
    sentence count on every response.
    """
    return " ".join(
        line
        for line in (text or "").splitlines()
        if line.strip() and not _DATE_LINE.match(line) and not _SOURCE_LINE.match(line)
    ).strip()


def sentence_count(text: str) -> int:
    joined = answer_body(text)
    if not joined:
        return 0
    return len([part for part in re.split(r"(?<=[.!?])\s+", joined) if part.strip()])


def is_factual_case(case: dict) -> bool:
    return str(case.get("expect_status", "")) == "answered"


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def citation_resolves(url: str, timeout: float = 6.0) -> bool | None:
    """Best-effort reachability. `None` means the check could not run, which is reported
    separately from a definite failure so an offline run is not scored as broken links."""
    if not url:
        return None
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 400
    except HTTPError as exc:
        # 403 from a bot-blocked AMC page is a live server answering, not a dead link.
        return exc.code in (401, 403, 405, 429)
    except (URLError, TimeoutError, OSError):
        return False
    except Exception:  # noqa: BLE001 - never let a link check abort the eval
        return None


@dataclass
class Result:
    case: dict
    status: str
    text: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    citation_hosts: list[str] = field(default_factory=list)
    link_ok: list[bool | None] = field(default_factory=list)
    sentences: int = 0
    latency_ms: int = 0
    novelty: list[str] = field(default_factory=list)
    guards: dict = field(default_factory=dict)
    hits: list[dict] = field(default_factory=list)
    threshold: float = 0.0
    fact_in_context: bool | None = None

    @property
    def expected_status(self) -> str:
        return str(self.case.get("expect_status", ""))

    @property
    def is_factual(self) -> bool:
        return self.expected_status == "answered"


def evaluate_case(case: dict, check_links: bool) -> Result:
    query = str(case["query"])
    started = time.perf_counter()
    # `use_cache=False` is essential: the demo cache holds three of these questions, and a
    # cached answer returns an empty trace with no retrieved context, so the harness would
    # measure the cache instead of the pipeline it exists to evaluate.
    answer = answer_question(query, use_cache=False)
    latency = int((time.perf_counter() - started) * 1000)

    failures: list[str] = []
    expected_status = str(case.get("expect_status", ""))
    if answer.status != expected_status:
        failures.append(f"status {answer.status!r} != expected {expected_status!r}")

    lowered = (answer.text or "").lower()
    for needle in case.get("expect_contains", []) or []:
        if str(needle).lower() not in lowered:
            failures.append(f"missing phrase {needle!r}")
    for needle in case.get("expect_numbers", []) or []:
        if str(needle).lower() not in lowered:
            failures.append(f"missing number {needle!r}")

    hosts = [host_of(c.url) for c in answer.citations]
    want_host = case.get("expect_citation_host")
    if want_host and not any(want_host in host for host in hosts):
        failures.append(f"no citation from {want_host} (got {hosts or 'none'})")

    # Novelty: every number in the answer must already exist in a retrieved chunk. This is
    # the anti-fabrication check and it is independent of the expected values above.
    novelty: list[str] = []
    context = ""
    if answer.trace and answer.trace.hits:
        wanted = {h.get("chunk_id") for h in answer.trace.hits}
        context = "\n".join(chunk.text for chunk in read_chunks() if chunk.chunk_id in wanted)
    if answer.status == "answered" and context:
        allowed = set(numbers_in(context))
        novelty = sorted(set(numbers_in(answer_body(answer.text or ""))) - allowed)

    # Was the expected fact even in the retrieved context? Separating this from the
    # generated answer is what tells a retrieval miss apart from a generation miss, which
    # matters because the offline stub cannot be blamed on the retriever.
    fact_in_context: bool | None = None
    if is_factual_case(case):
        haystack = context.lower()
        needles = [str(n).lower() for n in (case.get("expect_numbers") or [])]
        phrases = [str(n).lower() for n in (case.get("expect_contains") or [])]
        fact_in_context = bool(
            haystack
            and all(n in haystack for n in needles)
            and all(p in haystack for p in phrases)
        )

    links = [citation_resolves(c.url) for c in answer.citations] if check_links else []
    if check_links and links and not any(links):
        failures.append("no citation URL resolved")

    return Result(
        case=case,
        status=answer.status,
        text=answer.text or "",
        passed=not failures,
        failures=failures,
        citations=[{"label": c.label, "url": c.url} for c in answer.citations],
        citation_hosts=hosts,
        link_ok=links,
        sentences=sentence_count(answer.text or ""),
        latency_ms=latency,
        novelty=novelty,
        guards=dict(answer.trace.guards) if answer.trace and answer.trace.guards else {},
        hits=list(answer.trace.hits) if answer.trace and answer.trace.hits else [],
        threshold=float(answer.trace.threshold) if answer.trace else 0.0,
        fact_in_context=fact_in_context,
    )


def sweep(cases: list[dict]) -> list[dict]:
    """Retrieval-only sweep over MIN_SIMILARITY x TOP_K.

    Deliberately measures retrieval, not generation: with the offline stub the generated
    text says nothing about the threshold, and re-generating 18 x 10 answers would only
    measure the stub. `fact_found` asks whether the expected value is present in the
    retrieved context at all, which is the property the threshold actually controls.
    """
    factual = [c for c in cases if c.get("expect_status") == "answered"]
    rows: list[dict] = []
    for top_k in TOP_K_GRID:
        for threshold in THRESHOLD_GRID:
            found = 0
            for case in factual:
                result = retrieve(str(case["query"]), top_k=top_k, min_similarity=threshold)
                context = "\n".join(hit.chunk.text for hit in result.hits).lower()
                needles = [str(n).lower() for n in (case.get("expect_numbers") or [])]
                phrases = [str(n).lower() for n in (case.get("expect_contains") or [])]
                if all(n in context for n in needles) and all(p in context for p in phrases):
                    found += 1
            abstained = sum(
                1 for probe in NONSENSE_PROBES if not retrieve(probe, top_k=top_k, min_similarity=threshold).passed
            )
            rows.append(
                {
                    "top_k": top_k,
                    "threshold": threshold,
                    "fact_found": found,
                    "factual_total": len(factual),
                    "nonsense_abstained": abstained,
                    "nonsense_total": len(NONSENSE_PROBES),
                }
            )
    return rows


def recommend(rows: list[dict]) -> dict:
    """Pick the config that recovers the most expected facts, and is the tightest it can be
    while doing so.

    The earlier version recommended the *lowest* MIN_SIMILARITY that abstained on every
    nonsense probe, tie-broken toward the loosest gate. That is backwards: lowering the gate
    cannot add recall that a higher gate did not already have, and it can only admit more
    irrelevant chunks, which is what lets a wrong answer slip through. On this corpus every
    nonsense probe abstains at every threshold, so the old rule picked 0.20 for no gain at
    all while weakening the abstention margin.

    So: maximise fact recall, then take the tightest gate and the smallest context that
    achieve it. If recall turns out to be flat in the threshold, the probe set cannot
    identify the value and the honest answer is to keep the current default rather than
    quote a number the data does not support.
    """
    current = get_settings()
    total = rows[0]["factual_total"] if rows else 0
    safe = [r for r in rows if r["nonsense_abstained"] == r["nonsense_total"]]
    if not safe:
        return {
            "top_k": current.top_k,
            "threshold": current.min_similarity,
            "reason": "no config abstained on every nonsense probe; keeping the current default",
        }

    max_found = max(r["fact_found"] for r in safe)
    # Smallest context that achieves the best recall - fewer chunks means less prompt noise.
    best_k = min(r["top_k"] for r in safe if r["fact_found"] == max_found)
    at_k = [r for r in safe if r["fact_found"] == max_found and r["top_k"] == best_k]

    if len({r["fact_found"] for r in safe if r["top_k"] == best_k}) == 1:
        # Recall is identical at every threshold tested, so the probe set has no power to
        # choose one. Say so instead of inventing a preference for the grid's edge.
        threshold = current.min_similarity
        note = (
            f"MIN_SIMILARITY is not identifiable here - all {len(THRESHOLD_GRID)} tested values "
            f"recover {max_found}/{total} facts and abstain on {safe[0]['nonsense_total']} probes - "
            f"so keep the current {current.min_similarity}"
        )
    else:
        threshold = max(r["threshold"] for r in at_k)
        note = (
            f"tightest gate that still recovers {max_found}/{total} expected facts and abstains on "
            f"all {safe[0]['nonsense_total']} nonsense probes"
        )

    return {
        "top_k": best_k,
        "threshold": threshold,
        "reason": f"TOP_K {best_k} maximises fact recall ({max_found}/{total}); {note}",
    }


def write_sample_qa(results: list[Result], choice: dict, sweep_rows: list[dict]) -> None:
    passed = sum(1 for r in results if r.passed)
    factual = [r for r in results if r.is_factual]
    lines = [
        "# Sample Q&A",
        "",
        "Generated by `python eval/run_eval.py` from a real run against the persisted index.",
        "Do not hand-edit: regenerate it.",
        "",
        f"- Date of run: {time.strftime('%Y-%m-%d')}",
        f"- Result: **{passed}/{len(results)} cases passed**",
        f"- LLM provider in this run: `{get_settings().llm_provider or 'stub'}`",
        f"- `MIN_SEMILARITY` = {choice['threshold']}, `TOP_K` = {choice['top_k']} ({choice['reason']})",
        "",
    ]

    if get_settings().llm_provider in ("", "stub", None):
        lines += [
            "> **Answers below were produced by the offline `StubProvider`, not a language model.**",
            "> No `LLM_API_KEY` was configured. The stub lifts a sentence from the top-ranked",
            "> chunk, so it is correct only when the expected value happens to be the first line",
            "> of that chunk. This measures retrieval and the guardrails, not generation quality.",
            "",
        ]

    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        lines += [f"## {r.case['id']} — {mark}", "", f"**Q.** {r.case['query']}", ""]
        lines += [f"**A.** {r.text.strip()}", ""]
        if r.citations:
            lines.append("**Links**")
            for citation in r.citations:
                lines.append(f"- [{citation['label']}]({citation['url']})")
            lines.append("")
        if r.is_factual:
            stamp = "n/a"
            lines.append(f"- Status: `{r.status}` (expected `{r.expected_status}`)")
            lines.append(f"- Last updated from sources: {stamp if stamp != 'n/a' else 'see answer'}")
        lines += [
            f"- Sentences (answer body only): {r.sentences} (cap 3)",
            f"- Latency: {r.latency_ms} ms",
            f"- Guards: `{json.dumps(r.guards, ensure_ascii=False)}`",
        ]
        if r.hits:
            top = r.hits[0]
            lines.append(f"- Top chunk: `{top.get('chunk_id')}` score {top.get('score')} — {top.get('section')}")
        if r.fact_in_context is not None:
            lines.append(
                f"- Expected fact present in retrieved context: "
                f"{'yes' if r.fact_in_context else 'NO — retrieval miss'}"
            )
        if r.link_ok:
            verdict = ", ".join("ok" if ok else ("unreachable" if ok is False else "unchecked") for ok in r.link_ok)
            lines.append(f"- Citation link check: {verdict}")
        if r.novelty:
            lines.append(f"- **Novel numbers not in context: {r.novelty}**")
        if r.failures:
            lines.append("- Failures: " + "; ".join(r.failures))
        lines.append("")

    lines += [
        "## Threshold sweep",
        "",
        "Retrieval-only: does the retrieved context contain the expected value, and do the",
        "nonsense probes abstain? Generation is excluded because the offline stub would not",
        "inform the result.",
        "",
        "| TOP_K | MIN_SIMILARITY | expected facts found | nonsense abstained |",
        "| --- | --- | --- | --- |",
    ]
    for row in sweep_rows:
        lines.append(
            f"| {row['top_k']} | {row['threshold']:.2f} | "
            f"{row['fact_found']}/{row['factual_total']} | "
            f"{row['nonsense_abstained']}/{row['nonsense_total']} |"
        )
    lines.append("")
    SAMPLE_QA.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the P9 evaluation set.")
    parser.add_argument("--no-links", action="store_true", help="skip HTTP citation checks")
    parser.add_argument("--no-sweep", action="store_true", help="skip the threshold sweep")
    parser.add_argument("--show-sources", action="store_true", help="print where each expected value came from")
    parser.add_argument("--quiet", action="store_true", help="suppress the per-question table")
    args = parser.parse_args()

    cases = yaml.safe_load(TEST_SET.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        print("eval/test_set.yaml is empty or malformed", file=sys.stderr)
        return 1

    if args.show_sources:
        by_id = {c.chunk_id: c for c in read_chunks()}
        for case in cases:
            chunk_id = case.get("source_chunk")
            chunk = by_id.get(chunk_id) if chunk_id else None
            print(f"{case['id']}  {case['query']}")
            print(f"   expect_numbers={case.get('expect_numbers') or []}")
            if chunk:
                print(f"   from {chunk_id}: {' | '.join(chunk.text.splitlines()[1:4])[:150]}")
            elif chunk_id:
                print(f"   MISSING source chunk {chunk_id} - the corpus changed, re-verify this case")
            print()
        return 0

    results = [evaluate_case(case, check_links=not args.no_links) for case in cases]

    if not args.quiet:
        print(f"{'id':5} {'status':20} {'sent':4} {'ms':>6}  result")
        print("-" * 78)
        for r in results:
            mark = "PASS" if r.passed else "FAIL"
            print(f"{r.case['id']:5} {r.status:20} {r.sentences:<4} {r.latency_ms:>6}  {mark}")
            for failure in r.failures:
                print(f"{'':5} {'':20} {'':4} {'':>6}    - {failure}")

    factual = [r for r in results if r.is_factual]
    declining = [r for r in results if r.expected_status in ("refused", "pii_rejected")]
    answered = [r for r in results if r.status == "answered"]
    cited = [r for r in results if r.citations]
    link_checked = [r for r in results if r.link_ok]
    latencies = [r.latency_ms for r in results]
    novelty_hits = [r for r in results if r.novelty]
    # The 3-sentence cap is a generation constraint (`MAX_SENTENCES`, enforced by L6 in
    # `parse_answer`). A refusal is fixed compliance text chosen at P6, not generated, so
    # scoring its length against the cap would report a violation the system never made.
    over_cap = [r for r in results if r.status == "answered" and r.sentences > get_settings().max_sentences]

    print()
    print("=" * 78)
    print("P9 metrics")
    print("=" * 78)
    print(f"cases passed                 : {sum(1 for r in results if r.passed)}/{len(results)}")
    print(f"factual accuracy (answer)    : {sum(1 for r in factual if r.passed)}/{len(factual)}")
    in_context = [r for r in factual if r.fact_in_context is not None]
    print(f"expected fact in context     : {sum(1 for r in in_context if r.fact_in_context)}/{len(in_context)}"
          "   (retrieval layer, provider-independent)")
    print(f"citation present (all)       : {len(cited)}/{len(results)}")
    print(f"refusal rate on guard set    : {sum(1 for r in declining if r.status == r.expected_status)}/{len(declining)}")
    print(f"novel numbers in answers     : {sum(len(r.novelty) for r in novelty_hits)}")
    print(f"answers over {get_settings().max_sentences} sentences          : {len(over_cap)}")
    warm = latencies[1:] or latencies
    print(f"latency p50 / max (ms)       : {int(statistics.median(warm))} / {max(warm)}  (warm; first call "
          f"excluded, it pays the {latencies[0]} ms model load)")
    if link_checked:
        ok = sum(1 for r in link_checked for v in r.link_ok if v is True)
        unknown = sum(1 for r in link_checked for v in r.link_ok if v is None)
        print(f"citation links reachable     : {ok}/{sum(len(r.link_ok) for r in link_checked)}"
              + (f" ({unknown} unchecked)" if unknown else ""))
    print(f"llm provider                 : {get_settings().llm_provider or 'stub (offline)'}")

    if novelty_hits:
        print()
        print("NOVELTY VIOLATIONS (numbers in an answer absent from its context):")
        for r in novelty_hits:
            print(f"  {r.case['id']}: {r.novelty}")

    sweep_rows: list[dict] = []
    choice = {"top_k": get_settings().top_k, "threshold": get_settings().min_similarity, "reason": "sweep skipped; config defaults kept"}
    if not args.no_sweep:
        print()
        print("sweeping MIN_SIMILARITY x TOP_K (retrieval only)…")
        sweep_rows = sweep(cases)
        choice = recommend(sweep_rows)
        print(f"recommended: MIN_SIMILARITY={choice['threshold']}  TOP_K={choice['top_k']}")
        print(f"reason: {choice['reason']}")

    write_sample_qa(results, choice, sweep_rows)
    print()
    print(f"wrote {SAMPLE_QA.relative_to(ROOT)}")

    # Exit code reflects the guardrail and fabrication guarantees, which must hold on any
    # provider. Factual wording depends on the provider, so it is reported, not enforced.
    blocking = sum(len(r.novelty) for r in novelty_hits) + (
        len(declining) - sum(1 for r in declining if r.status == r.expected_status)
    )
    if blocking:
        print(f"\nFAILED: {blocking} guardrail or novelty violation(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
