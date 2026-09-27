# Demo script - three minutes

Target: **180 seconds**. Two rehearsals are recorded at the bottom; both must finish inside
the limit with the timings below as the worst case.

## Before you start (not part of the three minutes)

```powershell
python -m src.llm.cache --list      # must print 3/3
python -m src.ingest.build_index --stage export
streamlit run src/app.py
```

Then **warm the model before you start the timer.** Ask one question **that is not one of the
three cached demo questions** - use the second example button, "Is there a lock-in period on
the HDFC ELSS Tax Saver Fund?" - and wait for the reply.

This detail matters. A *cached* question returns in about a millisecond without ever loading
the embedding model, so warming up with one looks instant and then dumps the entire cost on
the first uncached retrieval mid-demo. That is not hypothetical: warming up with a cached
question left a **12.8 s** retrieval sitting in beat 2, and the same warm-up done properly
takes 1.0 s outside the timer. Cold loads have been observed anywhere from 0.9 s to 15 s on
this machine. Every later retrieval is milliseconds. (The UI labels a slow first retrieval as
a cold model load rather than reporting it as answer latency, so it will not *look* like a
hang - but you should not make anyone wait through it.)

To check the warm-up worked, `python -m scripts.rehearse_demo` prints the cache status of its
own warm-up question and warns if it was a hit.

If `--list` shows anything other than `3/3`, do not present. Re-seed with `LLM_API_KEY` set
first - otherwise the cached answers are stub extracts, not generated answers, and you should
say so when you demo.

---

## Beat 1 - what this is (0:00 - 0:25)

> "This bot answers facts about five HDFC mutual fund schemes, and nothing else. It reads a
> fixed corpus of ten official pages, and every sentence it produces is checked against those
> pages before you see it. It will not tell you what to buy."

Point at the disclaimer under the title: **Facts-only. No investment advice.**

> "The corpus is small and fixed - 184 chunks from 10 documents. That is deliberate. The point
> is not a large corpus, it is that every answer is traceable."

## Beat 2 - a factual answer with its citation (0:25 - 1:00)

Click the first example button, **"What is the expense ratio of the HDFC Large Cap Fund -
Direct - Growth?"**

> "It answers, and it cites. The citation is clickable and it goes to the exact page the
> number came from. Note the retrieval time in the corner - milliseconds, because the
> index is local."

Expand **"Why this answer?"**

> "This one came from the demo cache, and the trace says so - `cache: hit`, with no retrieved
> chunks listed, because nothing was retrieved. A replayed answer tells you it was replayed."

Now click the second example button, **"Is there a lock-in period on the HDFC ELSS Tax Saver
Fund?"** - this one is not cached, so it runs live - and expand its trace.

> "Same panel, but this time it is the live answer. This is the part most RAG demos skip: the
> chunks that were retrieved, their similarity scores, and which of them the answer actually
> cited. When an answer is wrong, this is where you can see whether retrieval missed or the
> model did."

## Beat 3 - it refuses to advise (1:00 - 1:25)

Type: **"Should I buy the HDFC Balanced Advantage Fund?"**

> "It refuses. Not because I muted it - because advice questions are a guardrail that runs
> before retrieval, so no model is even called."

Then type: **"What is the exit load on HDFC Small Cap Fund?"**

> "But a fact about the same fund answers normally. The guardrail is on the question type, not
> the fund."

## Beat 4 - the PII and scope walls (1:25 - 1:50)

Type: **"My PAN is ABCDE1234F, what is the exit load?"**

> "A PAN is rejected before anything is retrieved, and it is never written to the logs. The
> query log stores a hash, not the question."

Type: **"What is the expense ratio of Parag Parag Flexi Cap?"**

> "A fund outside the five is out of scope. It would rather say 'I do not cover that' than
> answer from general knowledge."

## Beat 5 - what it does not know (1:50 - 2:15)

Type gibberish: **"kjhgfdsa qwerty zxcvbn"**

> "Nonsense abstains. There is a similarity floor, and below it the bot says it does not know
> rather than answering from the nearest unrelated paragraph."

## Beat 6 - how it is verified (2:15 - 2:45)

> "There is an evaluation harness with ten cases. It measures refusals, citation presence,
> whether every number in the answer existed in the source, sentence count, and link
> reachability - and it separates retrieval quality from generation quality, so a bad answer
> can be attributed to the right layer."

Open `docs/sample_qa.md`.

> "This is a real run. And here is the honest part: factual accuracy is zero out of six,
> because this run used the offline stub provider with no API key. Retrieval is six out of
> six - the right evidence is in the context every time. The failure is entirely the
> provider, which echoes a single line instead of writing an answer. With a key configured
> that row is the one that still has to be proven."

## Close (2:45 - 3:00)

> "So: fixed corpus, cited answers, refuses advice, no PII in the logs, and an eval harness
> that reports its own failures instead of hiding them."

---

## If you are asked

**"Is the answer correct?"** - Every number is validated against the retrieved text before you
see it, and the citation is the page it came from. Whether the *phrasing* is right needs a
real model; see beat 6.

**"Why not use a bigger corpus?"** - Then the eval set and the guardrails stop being
meaningful. The small fixed corpus is what makes a wrong answer attributable.

**"Where does the data come from?"** - Groww scheme pages plus AMFI and SEBI investor
education pages. HDFC's own site returns HTTP 403 to automated requests, so the issuer's own
pages are *not* in the corpus. That is the most important known limit, and it is written down
in the README.

**"Do you log my questions?"** - No. `logs/queries.jsonl` stores a 12-character hash of the
normalised query, the guard decisions, the hit count, the top score, latency, and the status.

---

## Rehearsal log

System times are measured by `python -m scripts.rehearse_demo`, which drives the real
pipeline through every beat above. They are the *system* cost only - the 180 s budget is
dominated by speaking, and each beat is budgeted at roughly 3 s of talk.

| Rehearsal | Date | Warm-up | Total system time | Slowest beat | Result |
| --- | --- | --- | ---: | ---: | --- |
| 1 | 2026-09-27 | 1.0 s, uncached, `[answered]` | **0.1 s** | 26 ms | pass |
| 2 | 2026-09-27 | (already warm) | **0.1 s** | 24 ms | pass |

A first attempt that warmed up with a *cached* question recorded 12.9 s total, with the
12.8 s landing in beat 2. That run is the reason for the warm-up rule above; it is not
counted as a pass.

Both rehearsals come in far inside 180 s, so the budget is safe even if the cold load is not
avoided - but you should still warm up, because a 13 s pause reads as a crash.

If beat 6 runs long, cut the `--show-sources` walkthrough - it is the only part that can be
shortened without losing a guarantee.
