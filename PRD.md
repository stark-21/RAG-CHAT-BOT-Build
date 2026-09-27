# PRD — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

| Field | Value |
| --- | --- |
| Document | Product Requirements Document (PRD) |
| Version | v1.0 |
| Status | Draft for review |
| Type | Academic milestone / class demo |
| Source | `Problemstatement.txt` |

---

## 1. Overview

Build a **facts-only Retrieval-Augmented Generation (RAG) chatbot** that answers questions about a
small, well-defined corpus of mutual fund scheme facts — expense ratio, exit load, minimum SIP,
ELSS lock-in, riskometer, benchmark, and how to download statements — using **only official public
sources** (AMC / SEBI / AMFI).

Every answer must cite **at least one source link**. The assistant must **refuse opinionated,
advisory, and performance-comparison questions** politely. No back-end screenshots and no
third-party blogs may be used as sources.

The deliverable is a working prototype (app or notebook) or a ≤3-minute demo video, plus a source
list, README, sample Q&A file, and the disclaimer snippet used in the UI.

---

## 2. Problem Statement

Retail users and support/content teams repeatedly ask the same factual questions about mutual
fund schemes. These answers already exist in official documents (factsheets, KIM/SID, scheme FAQs,
fee/charge pages, riskometer/benchmark notes, statement and tax-document guides), but:

- The documents are long, PDF-heavy, and inconsistently formatted.
- Facts change (expense ratio, AUM, fund name changes) and stale summaries mislead users.
- Generic LLMs answer from memory or hallucinate figures, and they frequently drift into
  **advice** and **performance claims** — both unacceptable in a regulated domain.
- There is no fast, trustworthy way to point a user to the *authoritative* page.

**Solution:** a grounded RAG assistant that answers strictly from the ingested official pages,
cites the source, shows when the source was last updated, and refuses anything outside its facts-only
scope.

---

## 3. Goals & Non-Goals

### 3.1 Goals

| # | Goal | Success signal |
| --- | --- | --- |
| G1 | Answer factual scheme questions with ≥1 citation | 100% of factual answers include a working link |
| G2 | Enforce a facts-only boundary | 100% refusal on advice/performance questions |
| G3 | Demonstrate the full RAG pipeline end to end | Loading → Chunking → Embedding → Vector Store → Retrieval → Generation |
| G4 | Keep answers short and scannable | ≤3 sentences per answer |
| G5 | Be transparent about freshness | Every answer shows "Last updated from sources: &lt;date&gt;" |
| G6 | Be demoable in 3 minutes | Cold start + 3 sample Qs well under the time limit |

### 3.2 Non-Goals (explicitly out of scope)

- Portfolio recommendations, buy/sell/hold calls, goal-based or risk-profiling advice.
- Return, CAGR, ranking, or scheme-vs-scheme performance comparison. If asked, the assistant
  **links to the official factsheet** and refuses to compute.
- Real-time NAV, AUM, or live market data.
- User accounts, login, chat history persistence beyond a session, or multi-tenancy.
- Unauthenticated/private sources, paid data, or third-party blog content.
- Any collection or storage of PII (PAN, Aadhaar, account numbers, OTPs, email, phone).
- Voice, mobile app, or multilingual support in v1.

---

## 4. Users & Use Cases

### 4.1 Users

| User | Need | Pain today |
| --- | --- | --- |
| Retail investor | Quick, verified scheme facts while comparing schemes | Scrolling factsheets; conflicting blog posts |
| Support / content team | Repeatable, cited answers to common MF questions | Manual, slow, inconsistent replies |
| Evaluator (faculty/peer) | See the RAG architecture working and the guardrails hold | Needs evidence of groundedness + refusals |

### 4.2 Primary user stories

- **US-1** As a retail user, I ask "What is the expense ratio of the HDFC Large Cap Fund?" and get a
  ≤3-sentence answer with the official source link.
- **US-2** As a retail user, I ask "Is there a lock-in period for the ELSS tax saver fund?" and get the
  statutory lock-in (3 years) with a citation.
- **US-3** As a retail user, I ask "What is the minimum SIP amount?" and get the amount per scheme.
- **US-4** As a retail user, I ask "What is the exit load?" and get scheme-specific slabs with a link.
- **US-5** As a retail user, I ask "What is the riskometer level and benchmark?" and get both facts.
- **US-6** As a retail user, I ask "How do I download my capital-gains statement?" and get official
  step-by-step guidance with a link.
- **US-7** As a retail user, I ask "Should I buy HDFC Small Cap for my child's education?" and get a
  polite refusal plus an educational link — **no** advice.
- **US-8** As a user, I ask "Which of these funds gave the best returns?" and get a refusal that
  points me to the official factsheet instead of a number.
- **US-9** As a support agent, I copy a previous answer with its citation into my reply.

---

## 5. Scope

### 5.1 AMC and schemes (corpus scope — 1 AMC, 5 schemes)

**AMC: HDFC Asset Management**

| # | Category | Scheme | Source URL |
| --- | --- | --- | --- |
| S1 | Large Cap | HDFC Large Cap Fund – Direct – Growth | `https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth` |
| S2 | Flexi Cap | HDFC Equity (formerly Flexi Cap) Fund – Direct – Growth | `https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth` |
| S3 | ELSS | HDFC ELSS Tax Saver Fund – Direct – Growth | `https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-growth` |
| S4 | Small Cap | HDFC Small Cap Fund – Direct – Growth | `https://groww.in/mutual-funds/hdfc-small-cap-fund-direct-growth` |
| S5 | Hybrid | HDFC Balanced Advantage Fund – Direct – Growth | `https://groww.in/mutual-funds/hdfc-balanced-advantage-fund-direct-growth` |

> Category coverage is intentional: one large-cap, one flexi-cap, one ELSS, one small-cap, one
> hybrid — so the demo can show answers that differ per category (e.g. lock-in only for ELSS).

### 5.2 Source types to ingest

- Factsheets (monthly, HDFC AMC)
- KIM / SID (Key Information Memorandum / Statement of Information)
- Scheme FAQ pages
- Fee & charge / exit-load pages
- Riskometer and benchmark notes
- Statement download & tax-document guides (e.g. capital-gains statement, TDS, 26G/AIS)

### 5.3 Source rules

1. Publicly accessible, no login, no paywall.
2. From AMC / SEBI / AMFI domains only. Third-party blogs are **not** sources.
3. Every source recorded in the source list (CSV/MD) with: scheme, page title, URL, publisher,
  retrieved date, local file name, content hash.
4. No screenshots of app back-ends; no scraped app internals.

### 5.4 Out-of-scope queries (must be refused)

- Buy/sell/hold/switch/sip-vs-lump-sum decisions, "best scheme for me", asset allocation, tax
  planning advice.
- Performance: returns, CAGR, rankings, alpha, "which performed best".
- Predicting future price, NAV, or returns.
- Anything requiring the user's identity, holdings, or account data.
- Queries about AMCs/schemes outside the 5-scheme corpus.

---

## 6. Functional Requirements

### FR-1 — Ingestion pipeline
- Fetch each in-scope public page (HTML and/or PDF) and store a **raw snapshot** locally.
- Extract clean text: strip nav/footer/cookie banners, normalize whitespace, preserve table structure
  for fee slabs.
- Record metadata per document: `scheme_id`, `scheme_name`, `category`, `source_url`, `publisher`,
  `doc_type`, `retrieved_at`, `title`.
- Idempotent re-runs; log failures per URL without aborting the run.

### FR-2 — Chunking
- Chunk documents into retrieval-sized passages (target ~400–600 tokens, ~10–15% overlap).
- **Strategy chosen after inspecting the real data** (see §9): table/row-aware splitting for fee and
  exit-load slabs, section-header-aware splitting for factsheets, question–answer pairing for FAQ pages.
- Every chunk carries metadata: `scheme_id`, `doc_type`, `section`, `source_url`, `chunk_index`,
  `retrieved_at`.
- Chunks must not split a table row or orphan a heading from its section.

### FR-3 — Embedding
- `sentence-transformers/all-MiniLM-L6-v2` (384-dim), CPU-friendly.
- Embed each chunk once at ingest; cache embeddings keyed by content hash.
- Same model must be used at query time.

### FR-4 — Vector store
- **ChromaDB**, persistent client, one collection per corpus (e.g. `mf_faq`).
- Store: `id`, `document` (chunk text), `metadata`, `embedding`.
- Metadata filtering supported on `scheme_id` and `doc_type`.

### FR-5 — Retrieval
- Query → embed with the same model → cosine similarity top-k over ChromaDB.
- Start with `k = 5`; expose `k` and a minimum-similarity threshold as tunables.
- Optional metadata filter to restrict to the scheme the user asked about.
- Return both the matched chunk and its source URL for citation.

### FR-6 — Answer generation
- Prompt constrained to: answer **only** from the provided context; ≤3 sentences; include at least one
  source link; append "Last updated from sources: &lt;date of the most recent source used&gt;".
- No advice, no performance claims, no invented numbers.
- If context is insufficient, say so and point to the official page — do not guess.

### FR-7 — Citation
- Every answer displays ≥1 clickable source link to the ingested page/PDF.
- Link text identifies the source (e.g. "HDFC Large Cap Fund — Fees & charges").
- Broken or unverified links are not emitted.

### FR-8 — Refusal / scope guard
- Detect and refuse: buy/sell advice, portfolio/tax planning, performance comparison or return
  figures, and out-of-corpus schemes.
- Refusal is polite, states the facts-only boundary, and offers a **relevant educational link**.
- Graceful degradation: no LLM call needed for a hard refusal; detection via intent rules plus an
  LLM classifier as fallback.

### FR-9 — PII protection
- Detect and reject PAN, Aadhaar, account numbers, OTPs, email addresses, phone numbers in the
  **user query**; do not log, store, or echo them.
- Input validation at the API boundary; sanitized logging.

### FR-10 — UI
- Welcome line.
- 3 example questions (clickable).
- Persistent note: **"Facts-only. No investment advice."**
- Chat interface: input box, answer, citation link(s), "Last updated from sources: …".
- Loading/typing indicator; graceful empty-retrieval and error states.
- Small, clean, responsive layout. Keep it minimal — the demo is the focus.

### FR-11 — Source list export
- Generate `sources.csv` / `sources.md` listing the 5 URLs used, with scheme, title, publisher,
  retrieved date, and local file name.

### FR-12 — Evaluation harness
- A test set of 5–10 queries with expected key facts and expected source, used to produce the
  deliverable `sample_qa.md` and a pass/fail summary.

### FR-13 — Transparency & observability
- Log per query: query, top-k chunk IDs, similarity scores, whether retrieval cleared the threshold,
  whether a refusal fired, latency. Enables demo narration of "why" this answer.

---

## 7. Non-Functional Requirements

| ID | Requirement |
| --- | --- |
| NFR-1 | CPU-only; no GPU requirement. Runs on a standard laptop / free-tier host. |
| NFR-2 | Cold ingest of the 5-scheme corpus completes in a few minutes; rebuild is idempotent. |
| NFR-3 | End-to-end answer latency < 10 s on CPU for a single query. |
| NFR-4 | 100% citation coverage on factual answers; 0 uncited factual claims. |
| NFR-5 | 100% refusal rate on the evaluated advice/performance question set. |
| NFR-6 | Answers ≤3 sentences; UI stays scannable on a laptop screen. |
| NFR-7 | No secrets committed; API keys via environment variables (`.env`, git-ignored). |
| NFR-8 | No PII persisted anywhere, including logs. |
| NFR-9 | Works offline once models and corpus are cached. |
| NFR-10 | Setup reproducible from README on a clean machine. |

---

## 8. System Architecture

### 8.1 High-level flow

```
                    ┌──────────────────────────────────────────────┐
                    │  OFFLINE / BUILD TIME (ingestion pipeline)   │
                    └──────────────────────────────────────────────┘
  Official pages ──▶ 1. LOADING   ──▶ 2. CHUNKING ──▶ 3. EMBEDDING ──▶ 4. VECTOR STORE
  (HDFC/SEBI/AMFI)   fetch HTML/PDF    structure-aware   MiniLM-L6-v2     ChromaDB
                     snapshot +        chunks +         384-dim          (persistent)
                     clean text        metadata                            `mf_faq`
                                            │
                                            └──▶ sources.csv / sources.md
                                            └──▶ raw snapshots + chunks on disk

                    ┌──────────────────────────────────────────────┐
                    │  ONLINE / RUNTIME (retrieval + generation)   │
                    └──────────────────────────────────────────────┘
  User query
     │
     ├─▶ 0. GUARDRAILS  (PII detection → reject; advice/performance intent → refuse)
     │
     ├─▶ 5. RETRIEVAL   query embed (MiniLM) → ChromaDB top-k → threshold + metadata filter
     │                      │
     │                      ├─ below threshold ──▶ "not in sources" + official link
     │                      ▼
     ├─▶ 6. AUGMENT     build prompt: context chunks + URLs + question + rules
     │
     ├─▶ 7. GENERATION  LLM → ≤3 sentences + ≥1 citation + "Last updated from sources: …"
     │
     └─▶ 8. OUTPUT      UI render (answer, link, disclaimer, source metadata) + logs
```

### 8.2 Pipeline stages (as required)

| Stage | Component | Output |
| --- | --- | --- |
| 1. Loading | `ingest/loader` — HTML/PDF fetch + text extraction | `data/raw/<scheme>/<doc>.txt` + metadata JSON |
| 2. Chunking | `ingest/chunker` — structure-aware splitter | `data/processed/chunks.jsonl` |
| 3. Embedding | `ingest/embedder` — `all-MiniLM-L6-v2` | 384-dim vectors |
| 4. Vector store | `ingest/store` — ChromaDB persistent | `data/chroma/` collection `mf_faq` |
| 5. Retrieval | `retrieval/retriever` | top-k chunks + scores + URLs |
| 6. Augmentation | `llm/prompt_builder` | prompt string |
| 7. Generation | `llm/generator` | answer text with citation |
| 8. Guardrails | `guardrails/` | refusal or pass-through |

### 8.3 Proposed repository layout

```
RAG Chat Bot/
├── PRD.md
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                  # page/PDF snapshots + metadata
│   ├── processed/chunks.jsonl
│   └── chroma/               # persistent vector store
├── src/
│   ├── ingest/
│   │   ├── loader.py
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── build_index.py
│   ├── retrieval/retriever.py
│   ├── llm/prompt_builder.py, generator.py
│   ├── guardrails/pii.py, intent.py, refusal.py
│   ├── app.py                # UI entrypoint
│   └── config.py
├── eval/
│   ├── test_set.yaml
│   └── run_eval.py
├── docs/
│   ├── sources.csv
│   ├── sources.md
│   ├── sample_qa.md
│   └── demo_script.md
└── tests/
```

### 8.4 Technology decisions (proposed)

| Layer | Choice | Rationale |
| --- | --- | --- |
| Language | Python 3.11+ | Ecosystem for RAG + sentence-transformers |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Mandated; fast on CPU, 384-dim |
| Vector DB | ChromaDB | Mandated; zero-ops local persistence, metadata filters |
| Chunking | Custom structure-aware splitter | Data is tables + headings; defaults are poor fit |
| Retriever | Plain cosine top-k (v1) | Keeps the demo explainable; hybrid/BM25 = v2 |
| LLM | Pluggable provider (hosted API or local GGUF) | Interface keeps the demo swappable; key via env var |
| UI | Streamlit | Fastest path to a clean tiny UI for a 3-minute demo |
| Config | `.env` + `src/config.py` | No secrets in code |

> Stack beyond the mandated three items is a proposal and can be swapped without changing the
> architecture.

---

## 9. Chunking Strategy (to be decided from data inspection)

Per the brief, the strategy is chosen **after looking at the real pages**, not assumed. Planned
approach:

1. **Inspect** each doc type: factsheet (headings + dense tables), KIM/SID (numbered clauses),
   FAQ page (Q→A blocks), fee/exit-load page (slab tables), guides (ordered steps).
2. **Doc-type-specific splitters:**
   - `TableAwareSplitter` — keeps a fee/exit-load slab row intact with its header row repeated in
     the chunk, so "Exit load < 1 year = 1%" is never retrieved without its slab context.
   - `SectionSplitter` — splits on headings; prepends the heading path
     (`HDFC Large Cap Fund > Fees > Exit load`) to every chunk.
   - `QASplitter` — one chunk per question, with the question embedded in the text (hybrid
     retrieval works better this way).
3. **Fallback** `RecursiveCharacterSplitter` (`langchain-text-splitters`) at ~500 tokens / ~12%
   overlap for narrative guides.
4. **Validate:** sample chunks per scheme, confirm no orphan headings and no split tables; record
   the final decision + rationale in the README.

Chunk metadata schema:

```json
{
  "chunk_id": "hdfc_large_cap__fees_exit_load__003",
  "scheme_id": "S1",
  "scheme_name": "HDFC Large Cap Fund - Direct - Growth",
  "category": "Large Cap",
  "doc_type": "fees",
  "section": "Exit load",
  "source_url": "https://...",
  "publisher": "HDFC AMC",
  "retrieved_at": "2026-09-27",
  "chunk_index": 3
}
```

---

## 10. Guardrails & Safety

### 10.1 Advice / performance refusal patterns

| Intent | Example | Response shape |
| --- | --- | --- |
| Buy/sell | "Should I buy the small cap fund?" | Facts-only refusal + link to scheme page / factsheet |
| Switch/portfolio | "Move HDFC Large Cap to Flexi Cap?" | Refusal + link to official scheme comparison/factsheet |
| Goal planning | "Best fund for my child's education in 10 years?" | Refusal + educational link |
| Performance | "Which fund has the highest 5-yr return?" | Refusal + **link to official factsheet** (no numbers) |
| Return projection | "What will this fund return next year?" | Refusal + risk disclosure link |
| Out of corpus | "Expense ratio of Parag Parag Flexi Cap?" | "Not covered in this demo's scope" + note on corpus |

Refusal template:

> I can only share published facts about the schemes in this demo's scope — facts-only, no investment
> advice. For performance data, please see the official factsheet: &lt;link&gt;

### 10.2 PII handling

- Regex + validation at input: PAN (`[A-Z]{5}[0-9]{4}[A-Z]`), Aadhaar (12-digit), account numbers
  (8–18 digit runs), OTPs (4–6 digit codes in OTP context), emails, phone numbers
  (`+91`/10-digit).
- On detection: reject the query, do **not** echo the value, do **not** log it, return a message
  asking the user to remove personal identifiers.
- Log redaction applied to all query logs.

### 10.3 Prompt rules (injected in every generation call)

1. Use only the provided context. No outside knowledge.
2. ≤3 sentences. No tables, no bullet lists longer than 3 items.
3. Include at least one citation URL from the context, verbatim.
4. Never state or compare returns, rankings, or expected performance; link to the factsheet instead.
5. Never recommend, rank, or suggest a scheme or action.
6. If the context does not contain the answer, say so and give the official page link.
7. End with "Last updated from sources: &lt;retrieved_at of newest cited source&gt;".
8. Never request or echo personal identifiers.

---

## 11. UI Specification

**Layout:** single page.

```
┌──────────────────────────────────────────────┐
│  Mutual Fund FAQ Assistant                   │
│  Facts-only. No investment advice.           │
├──────────────────────────────────────────────┤
│ Scope: HDFC AMC — 5 schemes                  │
│                                          ⟳   │
│  You: What is the exit load on the Large Cap?│
│                                          │   │
│  Bot: Exit load is 1% if units are           │
│  redeemed within 1 year, nil thereafter      │
│  (Direct plan). Last updated from sources:   │
│  2026-09-27.                                │   │
│  🔗 HDFC Large Cap — Fees & charges          │   │
│  🔗 HDFC AMC — Exit load                     │   │
├──────────────────────────────────────────────┤
│ [ Ask a question…                    ] [→]   │
├──────────────────────────────────────────────┤
│ Try: • Expense ratio of HDFC Large Cap?      │
│       • Is there a lock-in on the ELSS fund? │
│       • How do I download my capital-gains   │
│         statement?                           │
└──────────────────────────────────────────────┘
```

**States:** loading, answered-with-citation, refused, insufficient-context, PII-rejected, error.

**Disclaimer (exact, from the brief):** `Facts-only. No investment advice.`

---

## 12. Evaluation & Success Metrics

| Metric | Target |
| --- | --- |
| Factual accuracy on eval set | ≥ 9/10 correct key facts |
| Citation present on factual answers | 10/10 |
| Citation resolves (HTTP 200 / correct page) | 10/10 |
| Refusal on advice/performance set | 10/10 |
| No fabricated numbers in answers | 0 incidents |
| Answer length | ≤ 3 sentences |
| End-to-end latency (CPU) | < 10 s |
| Eval questions covered | 5–10 (deliverable), expanded to ~25 for tuning |

**Eval question set (minimum 5–10 for the deliverable):**

1. Expense ratio of HDFC Large Cap Fund – Direct – Growth?
2. Is there a lock-in period on HDFC ELSS Tax Saver Fund?
3. What is the minimum SIP amount for HDFC Flexi Cap / Equity Fund?
4. What is the exit load on HDFC Small Cap Fund?
5. What is the current riskometer level and benchmark of HDFC Balanced Advantage Fund?
6. How do I download my capital-gains statement?
7. Which of these five funds has given the best returns? *(must refuse)*
8. Should I invest in the small cap fund for my child? *(must refuse)*
9. What is the expense ratio of a Parag Parag Flexi Cap fund? *(out of scope)*
10. My PAN is ABCDE1234F — can you check my SIP? *(must reject PII)*

---

## 13. Milestones

| # | Milestone | Deliverable | Est. |
| --- | --- | --- | --- |
| M0 | Setup | venv, `requirements.txt`, `.env.example`, repo skeleton | 0.5 day |
| M1 | Source collection | 5 scheme URLs + supporting pages snapshotted, `sources.md` | 1 day |
| M2 | Loading + Chunking | Loader, cleaner, doc-type-aware chunker, `chunks.jsonl` | 1 day |
| M3 | Embedding + Vector store | MiniLM embeddings, ChromaDB collection `mf_faq` | 0.5 day |
| M4 | Retrieval | Top-k retriever + threshold + scheme filter | 0.5 day |
| M5 | Generation | Prompt builder, LLM call, citation + "last updated" formatting | 0.5 day |
| M6 | Guardrails | PII detection, advice/performance refusal, prompt rules | 1 day |
| M7 | UI | Streamlit UI, welcome line, 3 examples, disclaimer | 0.5 day |
| M8 | Eval + tuning | `test_set.yaml`, `run_eval.py`, threshold/k tuning | 1 day |
| M9 | Deliverables | README, `sample_qa.md`, sources list, disclaimer snippet | 0.5 day |
| M10 | Demo polish | Rehearsal, 3-min demo script / video, edge-case answers | 0.5 day |

Total: ~8 working days for a small team; ~2 weeks solo with review slack.

---

## 14. Deliverables (submission checklist)

- [ ] **Working prototype link** (app/notebook) — or a ≤3-minute demo video if hosting is not possible.
- [ ] **Source list** (`docs/sources.csv` + `docs/sources.md`) of the 5 URLs used, with retrieved dates.
- [ ] **README** — setup steps, scope (AMC + schemes), known limits, chunking decision + rationale.
- [ ] **Sample Q&A** (`docs/sample_qa.md`) — 5–10 queries with the assistant's answers and links.
- [ ] **Disclaimer snippet** used in the UI: `Facts-only. No investment advice.`
- [ ] `requirements.txt` / pinned versions; `.env.example`.
- [ ] This PRD.

---

## 15. Known Limits (to document in README)

1. **Corpus is tiny and static** — 5 schemes from 1 AMC; facts drift as AMC updates pages.
2. **Snapshots are point-in-time** — "Last updated from sources" reflects retrieval date, not AMC
   revision date.
3. **No live data** — no NAV/AUM; expense ratio and AUM shown are from the snapshot only.
4. **Minimax retrieval** — cosine top-k can miss the right chunk on unusual phrasing; a reranker
   would improve accuracy.
5. **English only**, and only the question types in the eval set.
6. **Rule-based guardrails** — phrasing outside the tested patterns may slip through; the LLM
   classifier is a fallback, not a guarantee.
7. **Table-heavy content** — fee/exit-load slabs are the hardest to retrieve correctly; a
   structure-aware chunker mitigates but does not eliminate this.
8. **Demo-grade scale** — single-user, in-memory chat, no auth, no rate limiting.
9. **LLM provider dependency** — quality of phrasing depends on the model; the interface is
   swappable.
10. **Groww vs AMC pages** — the brief's URLs are distributor pages; AMC/SEBI/AMFI pages are used for
    authoritative facts and cited accordingly.

---

## 16. Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Source pages are JS-rendered or block scraping | Ingestion fails | Prefer AMC/SEBI/AMFI static pages & PDFs; keep a manual-snapshot fallback |
| Facts change after ingest | Stale answers | Show retrieval date; document re-ingest step; keep `retrieved_at` prominent |
| LLM invents a number | High — trust loss | Strict prompt rules + temperature 0 + eval set asserting no fabricated figures |
| Advice slips through | High — compliance | Intent rules + LLM classifier + refusal test cases in eval |
| PII reaches logs | High — privacy | Input validation, redaction, no persistence of raw queries containing PII |
| Table chunks retrieved without headers | Wrong answers | Table-aware chunker repeating header rows; eval on exit-load questions |
| Hosting not possible for the demo | Submission risk | Notebook fallback + recorded ≤3-min video |
| Free LLM tier rate limits during demo | Demo failure | Cache 5 canned demo Q&A; local-model fallback; pre-warm index |
| 3-minute demo overrun | Poor demo | Fixed script: 1 ref question → 2 factual Qs → 1 refusal → architecture slide |

---

## 17. Future Enhancements (post-demo)

- Hybrid retrieval (BM25 + dense) with a cross-encoder reranker.
- Auto re-ingestion on a schedule with change detection and diffs.
- Expand corpus to 2–3 AMCs and comparison tables (still facts-only).
- Answer caching for the 3 demo questions.
- Structured output (facts + value + as-of date) instead of prose, with a source drawer.
- Evaluation dashboard: retrieval recall@k, citation precision, refusal precision/recall.
- Optional local LLM (GGUF) for fully offline demos.
