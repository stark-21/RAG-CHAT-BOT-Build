# HDFC Mutual Fund Facts Bot

A retrieval-augmented chatbot that answers **facts only** about five HDFC mutual fund schemes,
from a fixed corpus of official pages. Every answer is grounded in a retrieved chunk and
carries a clickable citation.

> **Disclaimer: Facts-only. No investment advice.**

The bot refuses buy/sell/should-I questions outright. It does not rank schemes, predict
returns, or recommend anything.

---

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. Build the index: scrape -> chunk -> embed -> store -> write source manifests
python -m src.ingest.build_index --stage all

# 2. Launch the UI
streamlit run src/app.py

# 3. Run the evaluation suite (writes docs/sample_qa.md)
python eval/run_eval.py
```

### Running fully offline

The default provider is `stub`, which needs no API key and no network. It echoes the
highest-scoring line from the retrieved context and labels the output as an extract, so a
stubbed run can never be mistaken for a generated answer. Every pipeline stage, guardrail,
and test runs against it.

To use a real model, set the key in `.env`:

```ini
LLM_PROVIDER=hosted
LLM_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

`LLM_PROVIDER` accepts `stub`, `hosted`, or `local`. Anything else raises at startup rather
than silently falling back. For a local model (Ollama by default):

```ini
LLM_PROVIDER=local
LLM_MODEL=llama3.1
LLM_BASE_URL=http://localhost:11434
```

If a hosted call fails, the pipeline falls back to `stub` and records
`guards["fell_back_to_stub"] = True`, so a degraded run is visible in the trace rather than
looking like a normal answer.

---

## How a question is answered

```
question
  |
  v
[0] ingress guards      intent -> advice / PII / out-of-corpus refusal. No retrieval, no LLM.
  |
  v
[1] answer cache        optional replay of a verified answer (see "Demo cache")
  |
  v
[2] retrieval           embed query -> Chroma -> scheme filter -> per-section dedupe
  |                     -> MIN_SIMILARITY gate. Below the gate -> insufficient_context.
  v
[3] generation          build a prompt from the retrieved chunks, call the provider
  |
  v
[4] validation          every number in the answer must exist in the retrieved context,
  |                     and every URL must belong to a retrieved source. One retry, then
  |                     demote to insufficient_context. The model is never shown the error.
  v
[5] trace + log         attach the trace, write logs/queries.jsonl (no raw query text)
```

| Stage | Module |
| --- | --- |
| Guards | `src/guardrails/` (`intent`, `pii`, `refusal`) |
| Retrieval | `src/retrieval/retriever.py` |
| Prompt + validation | `src/llm/prompt_builder.py`, `src/llm/generator.py` |
| Providers | `src/llm/provider.py` (`StubProvider`, plus hosted providers) |
| Orchestration | `src/pipeline.py` |
| Observability | `src/observability.py` |
| UI | `src/app.py` |

---

## The corpus

10 documents, 184 chunks, 29,466 tokens (min 14, median 72, max 500).

| Host | Chunks |
| --- | ---: |
| `groww.in` | 124 |
| `www.amfiindia.com` | 52 |
| `investor.sebi.gov.in` | 8 |

| Scheme | Name | Chunks |
| --- | --- | ---: |
| `S1` | HDFC Large Cap Fund - Direct - Growth | 21 |
| `S2` | HDFC Equity (Flexi Cap) Fund - Direct - Growth | 22 |
| `S3` | HDFC ELSS Tax Saver Fund - Direct - Growth | 56 |
| `S4` | HDFC Small Cap Fund - Direct - Growth | 22 |
| `S5` | HDFC Balanced Advantage Fund - Direct - Growth | 37 |
| `GEN` | General - AMFI/SEBI investor education | 26 |

Regenerate the manifests after any rebuild:

```powershell
python -m src.ingest.build_index --stage export   # -> docs/sources.csv, docs/sources.md
```

The manifests are built from `data/processed/chunks.jsonl`, never from the live scraper, so
they always describe the exact corpus the bot answers from.

---

## Configuration

All settings are environment variables with working defaults (`.env` is optional).

| Variable | Default | Meaning |
| --- | --- | --- |
| `TOP_K` | `10` | chunks retrieved per question |
| `MIN_SIMILARITY` | `0.30` | below this cosine score the bot abstains |
| `MAX_SENTENCES` | `3` | hard cap on answer sentences |
| `TEMPERATURE` | `0.0` | generation temperature |
| `LLM_PROVIDER` | `stub` | `stub`, `hosted`, or `local` |
| `LLM_API_KEY` | unset | required for `hosted` |
| `LLM_BASE_URL` | `http://localhost:11434` | endpoint for `local` |
| `REQUEST_DELAY_SECONDS` | `1.0` | politeness delay between scrapes |
| `COLLECTION_NAME` | `mf_faq` | Chroma collection |

### Where the thresholds came from

`python eval/run_eval.py` runs a retrieval-only sweep over `MIN_SIMILARITY x TOP_K`. The
tuner maximises fact recall and then takes the **tightest** gate that still achieves it - it
never recommends loosening the gate, because lowering it cannot add recall but can admit
irrelevant chunks.

Measured on this corpus:

| `TOP_K` | Facts recovered (of 6) |
| ---: | ---: |
| 3 | 3 |
| 5 | 5 |
| 10 | **6** |

`MIN_SIMILARITY` had **no effect** on recall at any of the nine values tested (0.20-0.60),
and all four out-of-domain probes abstained at every one. The probe set therefore cannot
identify it, so `0.30` was left in the middle of the range rather than tuned to an arbitrary
grid edge. Revisit this if the corpus or the question set changes.

---

## Guardrails

| Guard | Behaviour |
| --- | --- |
| Advice | "Should I buy...", "which is better" -> refused, with a redirect to facts |
| PII | PAN, folio, account, email, phone -> rejected before retrieval |
| Out of corpus | A fund outside the five schemes -> refused as out of scope |
| Unknown | No confident scheme match -> `insufficient_context` rather than a guess |
| Similarity | Top score below `MIN_SIMILARITY` -> `insufficient_context` |
| Fabrication | A number not present in the retrieved context -> demote and retry once |
| Citation | A URL not belonging to a retrieved source -> demote |

Every terminal path returns a renderable `Answer`. `answer_question()` does not raise, so the
UI can never show a stack trace.

---

## Evaluation

```powershell
python eval/run_eval.py              # metrics + docs/sample_qa.md
python eval/run_eval.py --show-sources
```

Ten cases: six factual, three refusals, one PII rejection. Each factual case carries expected
phrases, expected numbers, an expected citation host, and the corpus chunk the answer should
come from - all extracted from the indexed corpus rather than written from memory.

The harness always calls the pipeline with `use_cache=False`. Three of the ten questions are
also demo-cache questions, and a cache hit returns no retrieved context, so without that flag
the harness would grade the cache instead of the system. A test enforces it.

Latest measured run (**offline `stub` provider, no API key**):

| Metric | Result |
| --- | --- |
| Cases passed | 4/10 |
| Required refusals / rejections | **4/4** |
| Citation present | **10/10** |
| Citation links reachable | **10/10** |
| Expected fact present in retrieved context | **6/6** |
| Novel numbers in answers | **0** |
| Answers over 3 sentences | **0** |
| Latency, warm (p50 / max) | 15 ms / 24 ms |
| First call in a process | 0.9 s typical, up to ~15 s observed |
| **Factual accuracy of the generated text** | **0/6** |

**Read those two accuracy rows together.** Retrieval is verified and good: all six expected
facts are in the context handed to the model. The generated text scores 0/6 because `StubProvider`
echoes a single line from the top chunk and has no ability to compose an answer - asked for the
exit load it returns the glossary definition of "exit load". That is the provider, not the
retriever, and the harness reports the two layers separately precisely so this is not mistaken
for a retrieval failure.

The project target of >= 9/10 correct factual answers **has not been demonstrated.** It
requires `LLM_API_KEY` and a re-run.

---

## Demo cache

`data/processed/answer_cache.json` replays verified answers for three fixed questions so a
demo does not depend on a network call or a cold model.

```powershell
python -m src.llm.cache --list          # coverage
python -m src.llm.cache --seed          # regenerate from the live pipeline
python -m src.llm.cache --clear
```

```python
answer_question(q)                 # uses the cache
answer_question(q, use_cache=False)  # forces the live path
```

Two rules make the cache safe:

- **Only `answered` results are stored or served.** A refusal, PII rejection,
  insufficient-context demotion, or error is a live safety decision. Caching one would mean a
  later change to the guardrails silently did nothing, because the old verdict kept replaying.
  The status is re-checked on read as well as on write, so a hand-edited file cannot make the
  bot serve a refusal.
- **The cache is consulted after the ingress guards**, so a cached answer can never bypass a
  refusal.

Keys are `hash_query()` digests, never query text, so the file cannot leak what was asked.
A cache hit is labelled `guards["cache"] = "hit"` in the trace, so a replay is visible in the
answer's own "Why this answer?" panel.

> The currently cached text was generated by the **offline stub**. Re-run `--seed` with
> `LLM_API_KEY` set before using the cache in a real presentation.

---

## Observability and privacy

- `logs/queries.jsonl` records `query_hash`, guards, hit count, top score, latency, and
  status. **Raw query text is never written**, matching the cache's keying.
- `logs/ingest.jsonl` and `logs/ingest_errors.json` record the scrape, including failures.
- Every answer carries a trace: the retrieved chunks with scores, the cited chunk ids, the
  thresholds applied, and the latency breakdown. The UI shows it under "Why this answer?".

---

## Tests

```powershell
python -m pytest              # 376 tests
```

The suite runs entirely offline against the stub and a real Chroma collection. It covers
chunking invariants, guardrail precedence, retrieval thresholds, the L6 validation rules, the
UI status mapping, the eval tuner's objective, the exporter's aggregation, and the cache's
safety rules.

---

## Known limits

1. **No factual accuracy has been demonstrated.** Needs `LLM_API_KEY`; see Evaluation.
2. **HDFC-hosted pages are unreachable.** All `hdfcmutualfund.com` requests return HTTP 403,
   so the corpus is distributor and regulator pages only. Five HDFC KIM/factsheet URLs are
   listed in `src/ingest/sources.yaml` but have never been fetched.
3. **Citations point to distributor pages, not the issuer.** A Groww scheme page is a
   secondary source. Every answer is still traceable to a specific retrieved chunk.
4. **Five schemes only.** Anything else is refused as out of scope by design.
5. **Snapshot data.** Facts are as of the `retrieved_at` date shown with each answer and are
   never silently refreshed.
6. **The embedding model is small** (`all-MiniLM-L6-v2`, 384-dim, CPU). Recall is good on the
   six factual questions but is not a benchmark.
7. **Stamp duty appears on both the investment and redemption pages** of a scheme page, so the
   0.005% figure is retrievable from two chunks.
8. **A stray ONNX model cache** (~79 MB) sits at `C:\Users\gteja\.cache\chroma\onnx_models\`.
   The project sets `HF_HOME=./.cache/hf`, but Chroma's ONNX runtime caches to the user
   profile. Harmless, and safe to delete.
