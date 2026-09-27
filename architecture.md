# Architecture — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

| Field | Value |
| --- | --- |
| Document | Software Architecture Document |
| Version | v1.0 |
| Status | Draft for review |
| Derived from | `PRD.md` v1.0 |
| Scope | Build-time ingestion pipeline + runtime retrieval/generation service |

---

## 1. Purpose

`PRD.md` defines **what** the system must do. This document defines **how** it is built: component
boundaries, module responsibilities, public interfaces, data schemas, data flow, failure handling,
configuration, and the traceability from every PRD requirement to the code that satisfies it.

**Audience:** the team building the demo, and any evaluator reading the code alongside this doc.

---

## 2. Architecture Drivers

Design choices are driven by these constraints, in priority order:

| # | Driver | Architectural consequence |
| --- | --- | --- |
| D1 | **Facts-only, no advice, no performance claims** (FR-8, NFR-5) | Guardrails are a *layered, first-class subsystem*, not prompt text alone. Refusal short-circuits before retrieval and before the LLM. |
| D2 | **Every answer cited** (FR-7, NFR-4) | Source URL is carried as structured metadata end to end; the UI renders links from metadata, never from free text parsing alone. |
| D3 | **No PII** (FR-9, NFR-8) | Input validation at the single ingress point; a redaction utility used by every logger. |
| D4 | **Mandated stack**: MiniLM-L6-v2 + ChromaDB | One embedding model instance shared by ingest and query; Chroma persistent client is the only store. |
| D5 | **CPU-only, offline-capable** (NFR-1, NFR-9) | Small model, batch embedding at ingest, no reranker in v1, artifacts cached to disk. |
| D6 | **Tiny, static corpus** (5 schemes) | Simplicity beats scale: no queues, no service split, no distributed store. One process. |
| D7 | **Demoable in 3 minutes** (G6) | Pre-warmed index, answer cache for the 3 demo questions, single command to start. |
| D8 | **Explainability for evaluation** (FR-13) | Every answer returns a trace object (chunks, scores, decisions) — this is what the demo narrates. |

**Explicitly rejected:** a framework-first design (LangChain/LlamaIndex graph orchestration). The
pipeline is small and mandated; hand-written stages are easier to explain in a demo and easier to
debug. `langchain-text-splitters` is used only as a splitter utility, not as an orchestrator.

---

## 3. System Context

```
┌────────────┐        ┌───────────────────────────────────────────────┐        ┌──────────────────┐
│  Retail    │        │      Mutual Fund FAQ Assistant (this system)   │        │  Official public │
│  investor  │───────▶│                                               │───────▶│  sources         │
│            │◀───────│   guardrails → retrieval → generation → UI     │        │  HDFC AMC        │
│  Support   │  text +│                                               │◀───────│  SEBI / AMFI     │
│  agent     │  cites │   (no accounts, no stored PII, no user data)  │  fetch │  (read-only)     │
└────────────┘        └───────────────────────┬───────────────────────┘        └──────────────────┘
                                              │
                                              ▼
                                   ┌─────────────────────┐
                                   │  LLM provider       │
                                   │  (pluggable, temp 0)│
                                   └─────────────────────┘
```

Trust boundaries:

- **B1** User → app: untrusted input (PII, prompt injection, out-of-scope intent).
- **B2** App → sources: outbound read-only HTTP; no credentials sent.
- **B3** App → LLM: only retrieved corpus text is sent; user PII never is (blocked at B1).

---

## 4. Component Architecture

### 4.1 Two time-scopes

The system has an **offline build phase** and an **online query phase**. They share the embedding
model and the store, but have separate entrypoints.

```
                          ┌──────── BUILD PHASE (offline, run once per corpus refresh) ────────┐

  sources.yaml            ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  (source registry)  ───▶ │  LOADER  │──▶│ CHUNKER  │──▶│ EMBEDDER │──▶│  STORE   │──▶│ EXPORTER │
                         │ fetch +  │   │ structure│   │ MiniLM   │   │ ChromaDB │   │ sources  │
                         │ clean    │   │ aware    │   │ 384-dim  │   │ persist  │   │ .csv/.md │
                         └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                               │              │              │              │
                               ▼              ▼              ▼              ▼
                          data/raw/     chunks.jsonl    embed cache     data/chroma/
                                                                       docs/sources.*

                          └───────────────────────────────────────────────────────────────────────┘

                          ┌──────── QUERY PHASE (online, per question) ──────────────────────────┐

  user query ─▶ ┌───────────────┐   ┌────────────┐   ┌───────────┐   ┌──────────┐   ┌────────┐
                 │ INGRESS       │──▶│ RETRIEVER  │──▶│ PROMPT    │──▶│  LLM     │──▶│ RENDER │
                 │ pii + intent  │   │ top-k      │   │ BUILDER   │   │ generate │   │ + cite │
                 │ + scope check │   │ + filter   │   │ + rules   │   │ temp 0   │   │ + trace│
                 └───────┬───────┘   └─────┬──────┘   └───────────┘   └──────────┘   └────────┘
                    refuse│                │below threshold                                ▲
                         ▼                ▼                                             │
                    REFUSAL ──────▶ INSUFFICIENT_CONTEXT ─────────────────────────────┘
```

### 4.2 Module inventory

| Module | Path | Responsibility | Key functions |
| --- | --- | --- | --- |
| Config | `src/config.py` | Load env + constants; single source of tunables | `get_settings()`, `Settings` |
| Source registry | `src/ingest/sources.yaml` → `sources.py` | Declarative list of in-scope URLs with scheme/doc-type tags | `load_sources()` |
| Loader | `src/ingest/loader.py` | HTTP fetch (HTML/PDF), extract text, save raw snapshot + doc metadata | `fetch_doc()`, `load_all()` |
| Cleaner | `src/ingest/cleaner.py` | Strip nav/footer/cookies, normalize whitespace, detect tables | `clean_html()`, `clean_pdf_text()`, `extract_tables()` |
| Chunker | `src/ingest/chunker.py` | Doc-type dispatch to structure-aware splitters | `chunk_document()`, `split_table()`, `split_section()`, `split_qa()`, `split_recursive()` |
| Embedder | `src/ingest/embedder.py` | MiniLM wrapper; batch embed; content-hash cache | `get_model()`, `embed_documents()`, `embed_query()` |
| Store | `src/ingest/store.py` | ChromaDB persistent client; upsert/clear/query | `get_collection()`, `upsert_chunks()`, `reset_collection()` |
| Ingest orchestrator | `src/ingest/build_index.py` | Wires Loading→Chunking→Embedding→Store; idempotent | `main()`, `build()` |
| Exporter | `src/ingest/exporter.py` | Write `docs/sources.csv` + `sources.md` from doc metadata | `export_sources()` |
| Retriever | `src/retrieval/retriever.py` | Embed query, top-k cosine, threshold, scheme filter, trace | `retrieve()` |
| Prompt builder | `src/llm/prompt_builder.py` | System rules + context blocks + question + date | `build_prompt()` |
| Generator | `src/llm/generator.py` | Provider-agnostic LLM call, temp 0, sentence cap, parse output | `generate()`, `parse_answer()` |
| LLM provider | `src/llm/provider.py` | Interface + implementations (hosted API / local GGUF / echo-stub) | `LLMProvider.generate()` |
| PII guard | `src/guardrails/pii.py` | Detect PAN/Aadhaar/account/OTP/email/phone; redact for logs | `detect_pii()`, `redact()` |
| Intent guard | `src/guardrails/intent.py` | Rule layer + optional LLM classifier for advice/performance/scope | `classify_intent()` |
| Refusal | `src/guardrails/refusal.py` | Polite refusal text + relevant educational link | `build_refusal()` |
| Orchestrator | `src/pipeline.py` | The single `answer_question()` control flow (all branches) | `answer_question()` |
| UI | `src/app.py` | Streamlit chat UI, welcome, 3 examples, disclaimer, states | — |
| Eval harness | `eval/run_eval.py` | Run `test_set.yaml`, assert facts/citations/refusals, emit `sample_qa.md` | `main()` |
| Observability | `src/observability.py` | Structured logs, per-query trace, latencies | `log_query()`, `get_trace()` |

### 4.3 Repository layout (final)

```
RAG Chat Bot/
├── PRD.md
├── architecture.md            # this document
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/<scheme_id>/<doc_id>.{txt,pdf,html}
│   ├── raw/<scheme_id>/<doc_id>.meta.json
│   ├── processed/chunks.jsonl
│   ├── processed/embeddings.sqlite        # content-hash cache
│   └── chroma/                            # persistent ChromaDB dir
├── src/
│   ├── config.py
│   ├── pipeline.py
│   ├── observability.py
│   ├── app.py
│   ├── ingest/{sources.yaml,sources.py,loader.py,cleaner.py,chunker.py,embedder.py,store.py,exporter.py,build_index.py}
│   ├── retrieval/retriever.py
│   ├── llm/{provider.py,prompt_builder.py,generator.py}
│   └── guardrails/{pii.py,intent.py,refusal.py}
├── eval/{test_set.yaml,run_eval.py}
├── docs/{sources.csv,sources.md,sample_qa.md,demo_script.md}
└── tests/{test_chunker.py,test_pii.py,test_intent.py,test_retriever.py,test_pipeline.py}
```

---

## 5. Public Interfaces & Contracts

All cross-module calls use plain dataclasses (no framework types leak between layers).

```python
# src/config.py
@dataclass(frozen=True)
class Settings:
    embedding_model: str        # "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str             # "data/chroma"
    collection_name: str        # "mf_faq"
    top_k: int                  # 5
    min_similarity: float       # 0.30  (tuned in M8)
    llm_provider: str           # "hosted" | "local" | "stub"
    llm_model: str
    llm_api_key_env: str        # "LLM_API_KEY"
    temperature: float          # 0.0
    max_sentences: int          # 3
    disclaimer: str             # "Facts-only. No investment advice."

# shared contracts
@dataclass
class SourceDoc:      # output of loader
    doc_id: str; scheme_id: str; scheme_name: str; category: str
    doc_type: str;    # factsheet|kim_sid|faq|fees|riskometer|guide
    title: str; source_url: str; publisher: str
    retrieved_at: str;   # ISO date
    content_hash: str
    raw_path: str; text: str; tables: list[list[list[str]]]

@dataclass
class Chunk:
    chunk_id: str; text: str; token_count: int
    scheme_id: str; scheme_name: str; category: str
    doc_type: str; section: str; source_url: str
    publisher: str; retrieved_at: str; chunk_index: int; splitter: str

@dataclass
class RetrievedChunk:
    chunk: Chunk; score: float

@dataclass
class Answer:
    status: Literal["answered","refused","insufficient_context","pii_rejected","error"]
    text: str
    citations: list[Citation]     # {label, url, retrieved_at, doc_type}
    last_updated: str | None       # newest retrieved_at among cited docs
    trace: QueryTrace              # chunks, scores, decisions, latency
```

**Contract rules**

1. `llm/provider.py` is the only module that knows which LLM vendor is in use.
2. `guardrails/` never imports `llm/` except the optional classifier; retrieval/generation never
   import UI code.
3. `ingest/` is importable without an LLM key (build phase is LLM-free).
4. The UI renders `Answer` only — it contains no retrieval or prompt logic.

---

## 6. Build Phase — Data Flow

### 6.1 Source registry (declarative, reviewable)

```yaml
# src/ingest/sources.yaml  (excerpt)
amc: HDFC Asset Management
schemes:
  - scheme_id: S1
    scheme_name: HDFC Large Cap Fund - Direct - Growth
    category: Large Cap
    entry_url: https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth
    aliases: ["large cap", "hdfc large cap", "hdfc large cap fund"]
    documents:
      - {doc_type: fees,        publisher: HDFC AMC, url: "https://www.hdfcassetmanagement.com/.../fees"}
      - {doc_type: factsheet,   publisher: HDFC AMC, url: "https://www.hdfcassetmanagement.com/.../factsheet.pdf", format: pdf}
      - {doc_type: riskometer,  publisher: HDFC AMC, url: "https://www.hdfcassetmanagement.com/.../riskometer"}
  - scheme_id: S3   # ELSS ...
  # S2 flexi-cap, S4 small-cap, S5 hybrid
```

Rationale: the corpus is small and human-curated; a YAML registry is auditable and lets the
exporter generate the required `sources.csv` deliverable mechanically (FR-11). Domain policy
(`allowed_publishers`, `blocked_domains`) is validated at load time.

### 6.2 Stage 1 — Loading

```
fetch (requests, timeout=30, retry 3x, UA set, rate-limit 1 req/sec)
   ├─ HTML → BeautifulSoup text + <table> extraction
   └─ PDF  → pdfplumber text + table extraction
        │
        ▼
cleaner: remove script/style/nav/footer/cookie-banner/aside; collapse whitespace;
         detect table blocks and mark them inline with <TABLE n> markers
        │
        ▼
save data/raw/<scheme_id>/<doc_id>.txt (+ .meta.json) and record content_hash = sha256(text)
```

- **Idempotence (FR-1):** if `content_hash` matches the stored meta, skip re-embedding (re-embed
  only new/changed docs). Failures are collected into `logs/ingest_errors.json`; the run continues.
- **Snapshot integrity:** raw files are never edited; cleaning is applied at chunk time so the
  snapshot remains auditable evidence of what was retrieved on `retrieved_at`.
- **Domain policy check:** reject any URL whose host is not in the allowlist (AMC / SEBI / AMFI /
  the brief's distributor URLs) — enforces PRD §5.3 rule 2 mechanically.

### 6.3 Stage 2 — Chunking (dispatch by `doc_type`)

| `doc_type` | Splitter | Behavior |
| --- | --- | --- |
| `fees` | `TableAwareSplitter` | For each table, emit one chunk per logical row group with the **header row repeated**; then prose. Never split a row. |
| `factsheet` | `SectionSplitter` | Split on headings; prepend heading path (`Scheme > Fees > Exit load`) to every chunk; pack sections to ~500 tokens with 12% overlap. |
| `faq` | `QASplitter` | One chunk per Q→A pair; text begins with the question so embeddings match question phrasing. |
| `kim_sid` | `SectionSplitter` | Split on numbered clauses; keep clause number in `section`. |
| `riskometer` | `SectionSplitter` | Small doc — usually a single chunk. |
| `guide` | `RecursiveSplitter` | `RecursiveCharacterSplitter` 500 tokens / 12% overlap (ordered steps preserved). |
| other | `RecursiveSplitter` | Fallback. |

```
chunk_document(doc) ->
    splitter = DISPATCH[doc.doc_type]
    header_path = doc.title
    for unit in splitter(doc):
        text = f"{scheme_name} — {header_path}\n{unit.text}"   # context prefix
        if tokens(text) > MAX_TOKENS: split_recursive(text)
        emit Chunk(chunk_id=f"{doc.doc_id}__{i:04d}", splitter=splitter.name, ...)
```

**Design notes**

- **Context prefix** on every chunk is the cheapest fix for "which scheme?" ambiguity: a chunk
  retrieved for an expense-ratio question always names the scheme and section.
- **Header repetition** in table chunks prevents the classic failure: retrieving "1%" without
  knowing it is "within 1 year" (PRD risk R6).
- **Token counting** uses the same tokenizer family as the embedder's max sequence length (512) to
  avoid silent truncation by MiniLM.
- **Final strategy is confirmed after inspecting real pages** (PRD §9 step 4); the dispatch table
  above is the starting hypothesis and the decision + rationale get recorded in the README.

Output: `data/processed/chunks.jsonl` — one JSON object per line with the §6.4 schema.

### 6.4 Stage 3 — Embedding

```
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
normalize_embeddings = True                      # so dot == cosine
for batch in batches(chunks, 32):
    keys   = [sha256(chunk.text) for chunk in batch]
    cached = sqlite SELECT hash, vector WHERE hash IN keys
    todo   = [chunk for chunk in batch if hash not in cached]
    vectors = model.encode(todo.text, batch_size=32, normalize_embeddings=True)
    INSERT missing (hash, vector) into cache
```

- `normalize_embeddings=True` makes Chroma's L2/IP behaviour equivalent to cosine (documented in
  `store.py` docstring) — avoids a class of ranking bugs.
- One model instance, loaded lazily and cached at module scope, so ingest and query share it
  (single ~90 MB model in RAM).
- Embedding cache keyed by content hash makes re-ingest of an unchanged corpus near-instant.

### 6.5 Stage 4 — Vector Store

```
client = chromadb.PersistentClient(path="data/chroma")
coll   = client.get_or_create_collection(
             name="mf_faq",
             metadata={"hnsw:space": "cosine", "embedding_model": MODEL_ID, "dim": 384})
coll.upsert(ids=[c.chunk_id], documents=[c.text], embeddings=[...], metadatas=[...])
```

- **Metadata stored per chunk:** `scheme_id`, `scheme_name`, `category`, `doc_type`, `section`,
  `source_url`, `publisher`, `retrieved_at`, `chunk_index`, `splitter`, `content_hash`.
  Chroma metadata values must be scalars → no nested dicts/lists (enforced in `store.py`).
- `reset_collection()` wipes and rebuilds; used when the chunker changes so old chunks never mix
  with new ones.
- Rebuild command: `python -m src.ingest.build_index --rebuild`.

### 6.6 Stage 5 — Source export (FR-11)

From the collected doc metadata, write `docs/sources.csv` (columns: `scheme_id, scheme_name,
category, doc_type, title, publisher, source_url, retrieved_at, local_file, content_hash`) and a
human-readable `docs/sources.md` with the 5 entry URLs first.

---

## 7. Query Phase — Data Flow

### 7.1 Single entrypoint: `pipeline.answer_question()`

```
answer_question(query) -> Answer
│
├─ 0. INGRESS GUARDS                          (guardrails/, no LLM, no retrieval)
│     ├─ pii.detect_pii(query)  ─────────────▶ status="pii_rejected"   (return, nothing logged raw)
│     ├─ intent.classify_intent(query) ───────▶ status="refused"       (return with educational link)
│     └─ scope.check_scheme(query)  ──────────▶ status="refused"       (out-of-corpus AMC/scheme)
│
├─ 1. RETRIEVE                                (retrieval/retriever.py)
│     ├─ q = embedder.embed_query(query)                       [same MiniLM instance]
│     ├─ filter = {"scheme_id": [...] } if a scheme was resolved from the query
│     ├─ hits = coll.query(q, n=top_k, where=filter)
│     └─ if max(hits.score) < min_similarity ─▶ status="insufficient_context"
│                                             (reply + official page link, no LLM)
│
├─ 2. AUGMENT                                 (llm/prompt_builder.py)
│     └─ system rules (immutable) + numbered context blocks
│        [S1·fees·Exit load] text … source: URL
│        + question + retrieved_at (newest) + disclaimer reminder
│
├─ 3. GENERATE                                (llm/generator.py, temperature 0)
│     └─ answer_text = provider.generate(prompt)
│        enforce: ≤3 sentences, ≥1 context URL present, no digits absent from context
│
├─ 4. POST-VALIDATE & RENDER                   (llm/generator.parse_answer)
│     ├─ citations = URLs in the answer ∩ context URLs  → if empty, fall back to all cited docs' URLs
│     ├─ last_updated = max(retrieved_at over cited docs)
│     ├─ hard-check: if answer contains a number not present in context → mark insufficient_context
│     └─ emit Answer(status, text, citations, last_updated, trace)
│
└─ 5. LOG TRACE                               (observability.log_query)
      query_hash, top-k chunk_ids + scores, guards fired, threshold pass/fail, latencies
```

Branch coverage: every `return` in `answer_question` produces a complete `Answer` object, so the UI
has exactly one rendering path with a `status` switch. This is what makes the demo's "states" real
rather than aspirational.

### 7.2 Sequence — factual question

```
User        UI         Pipeline     PII     Intent    Retriever  Chroma   Prompt     LLM
 │  submit   │            │          │        │          │          │        │        │
 │──────────▶│            │          │        │          │          │        │        │
 │           │───────────▶│          │        │          │          │        │        │
 │           │            │─detect──▶│        │          │          │        │        │
 │           │            │◀──clean──│        │          │          │        │        │
 │           │            │─classify──────────▶│         │          │        │        │
 │           │            │◀──"factual"───────│         │          │        │        │
 │           │            │─retrieve───────────────────▶│─query───▶│        │        │
 │           │            │◀──top-k chunks + scores───────────────────────────  │        │
 │           │            │─build_prompt─────────────────────────────────────▶│        │
 │           │            │─generate────────────────────────────────────────────────────▶│
 │           │            │◀──answer text────────────────────────────────────────────────  │
 │           │            │─validate citations / numbers / sentence cap                    │
 │           │◀──Answer(status=answered, citations=[…], last_updated=…)                   │
 │◀──render──│                                                                     │
```

### 7.3 Sequence — advice question (refusal short-circuit)

```
User: "Should I buy the small cap fund for my child?"
 │  UI ──▶ Pipeline ──▶ PII (clean) ──▶ Intent.classify_intent
 │                                  ├─ rule layer: /\b(should|which|best)\b/ + action verb
 │                                  │             + scheme token → intent = "advice"
 │                                  └─ (if ambiguous) LLM classifier → "advice"
 │  ◀── Answer(status="refused", refusal text + HDFC scheme/factsheet educational link)
 │  No retrieval. No LLM generation. Logged with guard=intent.advice.
```

### 7.4 Retrieval design details

- **Metric:** cosine (Chroma collection created with `hnsw:space=cosine`; embeddings L2-normalized).
- **k = 5, threshold = 0.30** — starting values; M8 tunes them against `eval/test_set.yaml`
  (≥0.9 accuracy target). Threshold is the main "abstain instead of guess" lever.
- **Scheme filter:** `intent.resolve_scheme(query)` matches aliases from `sources.yaml`
  (e.g. "large cap" → S1). If exactly one scheme resolves, apply `where={"scheme_id": id}` to
  sharpen ranking; if none or several, search all schemes and let the prompt disambiguate.
- **Deduplication:** collapse hits sharing the same `(scheme_id, section)` to at most 2 chunks so
  one verbose section cannot crowd out the other evidence.
- **Context budget:** top-k texts truncated to ~1,800 tokens total to stay well inside a small
  model's context while keeping ≥2 distinct sources when available.
- **Trace:** every returned `Answer` carries `QueryTrace{guards, hits[(chunk_id, score)], threshold,
  cited_doc_ids, latency_ms{retrieve, generate, total}}` — the UI shows it behind a "Why this answer?"
  expander (FR-13).

### 7.5 Prompt architecture

Three immutable parts, assembled by `build_prompt(context, question, retrieved_at)`:

1. **System rules** (never user-overridable; user text is only ever the `QUESTION` field):
   answer only from CONTEXT; ≤3 sentences; include ≥1 context URL verbatim; never state/compare
   returns; never recommend; if CONTEXT lacks the answer, say so and give the official link; end with
   "Last updated from sources: <date>"; never request personal identifiers.
2. **CONTEXT** — numbered, labelled blocks with scheme, doc type, section, URL, date:
   ```
   [1] (S1 · HDFC Large Cap Fund · fees · Exit load · retrieved 2026-09-27)
       Exit load: < 1 year — 1.00%; 1–12 months ...; > 12 months — Nil.
       source: https://...
   ```
3. **QUESTION** — the user query, PII-screened.

Post-generation validation (defence in depth against hallucinated figures — PRD risk R3):

- **Citation check:** every URL in the output must exist in CONTEXT URLs; otherwise the answer is
  regenerated once with a stricter reminder, else demoted to `insufficient_context`.
- **Number check:** extract numeric tokens from the answer; every one must appear in the retrieved
  context (allowing for formatting variants). A violation ⇒ `insufficient_context`. This is a cheap,
  high-value check for a domain full of percentages and rupee amounts.
- **Sentence check:** count sentences; if > `max_sentences`, keep the first N and re-append the
  citation + date line.

### 7.6 Guardrails architecture (layered, D1)

| Layer | Where | Detects | Cost | On hit |
| --- | --- | --- | --- | --- |
| L1 PII patterns | ingress | PAN, Aadhaar, 8–18-digit runs, OTP context, email, phone | ~0 ms | reject; redact in logs; no LLM, no retrieval |
| L2 Intent rules | ingress | advice verbs (`should`, `best`, `recommend`, `switch`, `allocate`, `is it good`), performance terms (`return`, `CAGR`, `best performing`, `ranking`, `alpha`, `will ... return`) | ~0 ms | refusal + educational link |
| L3 Scope check | ingress | AMC/scheme tokens not in `sources.yaml` | ~0 ms | "not in this demo's scope" |
| L4 LLM classifier (optional) | ingress | anything L1–L3 marked ambiguous | ~300 ms | refusal or pass-through |
| L5 Prompt rules | generation | residual advice/performance phrasing | in prompt | LLM self-refusal |
| L6 Output validation | post-gen | uncited URL, number not in context, >3 sentences | ~0 ms | regenerate once, else demote to `insufficient_context` |

Design intent: L1–L3 make the demo's refusals **deterministic and free**, which is both
compliance-safer and better for a live demo. L4–L6 are the safety net. L6's number check is what
makes "no fabricated figures" a testable property rather than a hope.

`pii.py` pattern set (compiled once, case-insensitive where appropriate):
PAN `[A-Z]{5}[0-9]{4}[A-Z]`, Aadhaar `\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b`, account `\b\d{8,18}\b`,
email, phone `(\+91[- ]?)?[6-9]\d{9}`, OTP `(otp|one time password)\D{0,10}\d{4,6}`. Values are
replaced with `[REDACTED]` in every log line via `observability.redact()`.

Refusal text template (`guardrails/refusal.py`):

```
I can only share published facts about the 5 HDFC schemes in this demo's scope — facts-only,
no investment advice. For performance figures, please see the official factsheet: <link>
```

`build_refusal(intent)` picks the link: advice → scheme page / investor-education page;
performance → official factsheet; out-of-scope → the corpus scope note.

---

## 8. UI Architecture (FR-10)

Single Streamlit page, `src/app.py`, stateless per query (no DB, no history persistence).

```
st.set_page_config(page_title="Mutual Fund FAQ Assistant", layout="centered")
st.title("Mutual Fund FAQ Assistant")
st.caption("Facts-only. No investment advice.")            # exact disclaimer string from config
st.caption("Scope: HDFC AMC — Large Cap, Flexi Cap, ELSS, Small Cap, Balanced Advantage")
for q in EXAMPLE_QUESTIONS: st.button(q) → seeds st.session_state["query"]
chat = st.chat_input("Ask a factual question about expense ratio, exit load, SIP, lock-in, …")
answer = pipeline.answer_question(chat)                   # single call
render(answer)   # switch on answer.status
```

Render rules (one function, status-driven):

| `status` | Rendered |
| --- | --- |
| `answered` | answer text; `🔗` citation links as markdown; `Last updated from sources: <date>`; "Why this answer?" expander with the trace |
| `refused` | refusal text; educational link; disclaimer |
| `insufficient_context` | "I couldn't find that in the official sources I have." + official page link |
| `pii_rejected` | "Please remove personal identifiers (PAN, Aadhaar, account number, OTP, email, phone). I don't store personal data." |
| `error` | friendly error + `trace.error` behind the expander; never a raw stack trace in the chat area |

Example questions (the 3 demo prompts, from PRD §12):
1. "What is the expense ratio of the HDFC Large Cap Fund?"
2. "Is there a lock-in period on the HDFC ELSS Tax Saver Fund?"
3. "How do I download my capital-gains statement?"

Startup: `@st.cache_resource` builds the Chroma client + embedder once. If `data/chroma` is
missing, the app shows a one-line instruction (`python -m src.ingest.build_index`) instead of
crashing — demos survive a fresh clone.

---

## 9. Configuration & Secrets (NFR-7)

`.env` (git-ignored) + `.env.example` (committed):

```
LLM_PROVIDER=hosted          # hosted | local | stub
LLM_MODEL=                   # provider-specific model id
LLM_API_KEY=                 # never committed
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
MIN_SIMILARITY=0.30
CHROMA_DIR=data/chroma
COLLECTION_NAME=mf_faq
LLM_TEMPERATURE=0.0
MAX_SENTENCES=3
DISCLAIMER=Facts-only. No investment advice.
HF_HOME=./.cache/hf          # keep model cache inside the project for offline reuse
```

`src/config.py` validates at import: required keys present, numeric ranges sane, collection name
valid, and raises a single actionable error listing what is missing. Secrets are read only from
env; nothing is echoed to logs.

---

## 10. Failure Modes & Resilience

| Failure | Detection | Handling | User sees |
| --- | --- | --- | --- |
| Source page 404/blocked/JS-rendered | fetch status, empty text | log to `logs/ingest_errors.json`, continue, mark doc `status=failed`; manual-snapshot fallback documented | — (build time) |
| PDF text extraction garbled | <200 chars extracted | flag for manual review; excluded from index | — (build time) |
| Model download fails (offline) | exception at load | `HF_HUB_OFFLINE=1` + pre-populated `HF_HOME`; README documents pre-warm | Startup error with instructions |
| Chroma dir missing | collection count == 0 | UI shows build instruction | "Run `build_index` first." |
| LLM API error / rate limit | exception + 429 | one retry with backoff; then `stub` provider returns the top context extract with citations (clearly labelled) | Cited extract, or friendly error |
| LLM returns uncited answer | L6 validation | regenerate once → demote to `insufficient_context` | "Couldn't find that in the sources." |
| LLM returns a number not in context | L6 number check | demote to `insufficient_context` | "Couldn't find that in the sources." |
| Retrieval below threshold | score < `min_similarity` | abstain (no LLM call) | "Not in the official sources I have." + page link |
| PII in query | L1 | reject before any downstream call | Removal request |
| Malicious prompt ("ignore instructions") | L5 + L6 | instructions are system-level; answer must be context-cited and number-checked | Normal cited answer or refusal |

**Demo resilience:** pre-build the index, pre-warm the embedder, cache the 3 demo answers
(`data/processed/answer_cache.json`, keyed by normalized query hash) so the live demo cannot be
derailed by a network hiccup. The cache is bypassed for the refusal demo questions so guardrails
are demonstrated live.

---

## 11. Observability (FR-13)

Structured JSONL to `logs/queries.jsonl`, one record per query:

```json
{"ts":"2026-09-27T10:15:03Z","query_hash":"a1b2c3","guards":{"pii":"clean","intent":"factual","scope":"S1"},
 "hits":[{"chunk_id":"hdfc_large_cap__0007__0002","score":0.71,"doc_type":"fees"}],
 "threshold":0.30,"threshold_passed":true,"cited":["hdfc_large_cap__0007"],
 "status":"answered","latency_ms":{"retrieve":210,"generate":2400,"total":2680}}
```

- Queries are stored **hashed**, with PII-redacted text only when needed for debugging — raw PII
  never reaches disk (NFR-8).
- The trace is returned in-process to the UI ("Why this answer?") so the demo can show *why* a
  chunk was retrieved — this doubles as evaluation evidence.
- Ingest logs: `logs/ingest.jsonl` (per-doc: fetch status, chars, tables, chunks, hash, duration).

---

## 12. Testing Strategy

| Level | Scope | Examples |
| --- | --- | --- |
| Unit | pure functions | `split_table` never splits a row & repeats headers; `detect_pii` catches all 5 PII types incl. the PAN eval case; `classify_intent` flags all 6 refusal phrasings; `parse_answer` sentence cap; number-check detects an injected figure |
| Integration | index build on a 2-doc fixture | ingest → chunks → embed → store → retrieve top-1 contains expected fact |
| Contract | eval harness | `eval/test_set.yaml` runs the 10 PRD §12 questions end to end; asserts fact, citation, refusal, PII rejection |
| Manual | UI | the 6 states render; 3-minute demo script rehearsed |

CI-lite: `pytest tests/ -q` and `python eval/run_eval.py` are the pre-demo gate. Both must be green
before recording the demo video.

---

## 13. Deployment Options (pick one for submission)

| Option | How | Trade-off |
| --- | --- | --- |
| **A. Local Streamlit** (default) | `streamlit run src/app.py` | Zero hosting; demo on own laptop; video fallback needed for the "link" deliverable |
| **B. Streamlit Community Cloud / HF Space** | push repo, set `LLM_API_KEY` in secrets | Gives the required "working prototype link"; free tier cold start, rate limits |
| **C. Notebook** | `notebooks/demo.ipynb` walking ingestion → retrieval → answers | Best for showing the pipeline stages; weakest as a chat UI |

Recommendation: build for A, ship B if the account is available, keep C as the artifact that proves
the pipeline. Architecture is identical in all three; only the entrypoint changes.

---

## 14. Key Design Decisions (ADR summary)

| # | Decision | Alternatives rejected | Why |
| --- | --- | --- | --- |
| A1 | Plain cosine top-k, no reranker | Cross-encoder rerank, hybrid BM25+dense | CPU budget, 3-min demo, explainability; revisit if accuracy < 9/10 |
| A2 | Structure-aware custom chunker | Naive `RecursiveCharacterSplitter` everywhere, LangChain splitter chains | Fee/exit-load tables and FAQ Q→A pairs are the top failure modes (PRD risk R6) |
| A3 | Guardrails pre-LLM + output validation | Prompt-only guardrails | Deterministic refusals, free, and testable; prompt-only cannot be verified |
| A4 | Number-in-context check | Trust temperature 0 | Temperature 0 does not prevent plausible fabricated percentages |
| A5 | Hand-written stages, no orchestration framework | LangChain / LlamaIndex | Mandated pipeline is small; explicit code is demo-explainable and debuggable |
| A6 | Pluggable LLM provider interface | Vendor SDK in the pipeline | Demo can fail over to local/stub on rate limits; keeps keys out of logic |
| A7 | Citations as structured metadata | Trust the LLM to cite | Guarantees NFR-4 independently of model behaviour |
| A8 | Persistent ChromaDB dir, idempotent rebuild | In-memory Chroma | NFR-2/NFR-9: rebuild is cheap, index survives restarts, works offline |
| A9 | Snapshot raw text, clean at chunk time | Clean in place | Auditable evidence of what was retrieved on `retrieved_at` |
| A10 | Answer cache for the 3 demo questions only | Cache everything / no cache | De-risks the live demo without hiding guardrails |

---

## 15. Requirement Traceability (PRD → architecture)

| PRD req | Satisfied by | Verified by |
| --- | --- | --- |
| FR-1 Ingestion | §6.1 registry, §6.2 loader/cleaner, idempotence by content hash | `logs/ingest.jsonl`, manual snapshot review |
| FR-2 Chunking | §6.3 doc-type dispatch, §6.3 context prefix, header repetition | `tests/test_chunker.py`, chunk samples in README |
| FR-3 Embedding | §6.4 single MiniLM instance, normalized, hash-cached | dimension assert (384) in `test_retriever.py` |
| FR-4 Vector store | §6.5 Chroma persistent, scalar metadata only | `coll.count()` after build |
| FR-5 Retrieval | §7.4 cosine top-k, threshold, scheme filter, dedup | `tests/test_retriever.py` |
| FR-6 Generation | §7.5 prompt parts + §7.5 post-validation | `tests/test_pipeline.py` |
| FR-7 Citation | §5 `Answer.citations`, §7.5 citation check, §8 render | eval: citation present 10/10 |
| FR-8 Refusal | §7.6 L1–L6, §7.3 short-circuit | eval: refusal 10/10 |
| FR-9 PII | §7.6 L1 patterns, ingress position, redaction | `tests/test_pii.py` + eval Q10 |
| FR-10 UI | §8 single Streamlit page, 6 states, disclaimer | manual walkthrough |
| FR-11 Source list | §6.6 exporter → `docs/sources.csv` / `.md` | file diff vs registry |
| FR-12 Eval harness | `eval/run_eval.py` + `eval/test_set.yaml` | run output → `docs/sample_qa.md` |
| FR-13 Observability | §11 JSONL logs + in-process trace | inspect `logs/queries.jsonl` |
| NFR-1 CPU | §6.4 `device="cpu"`, no reranker | laptop run |
| NFR-2 Fast ingest | §6.4 embedding cache, idempotent rebuild | rebuild timing |
| NFR-3 <10 s | §7.4 k=5, ~1.8k-token context, temp 0 | latency in trace |
| NFR-4 100% citations | A7 structured citations + §7.5 check | eval metric |
| NFR-5 100% refusals | A3 layered guardrails | eval metric |
| NFR-6 ≤3 sentences | §5 `max_sentences`, §7.5 sentence check | eval metric |
| NFR-7 No secrets | §9 env-only config, `.env.example` | repo scan |
| NFR-8 No PII stored | hashed logs, redaction | log inspection |
| NFR-9 Offline | persistent index + `HF_HOME` cache, local/stub provider | airplane-mode test |
| NFR-10 Reproducible | §13 + README setup | clean-machine run |

---

## 16. Build Order (maps to PRD §13 milestones)

```
M0  config.py, requirements.txt, .env.example, repo skeleton
M1  sources.yaml (5 schemes + docs) → loader → data/raw/  → docs/sources.md
M2  cleaner.py + chunker.py (dispatch) → chunks.jsonl     [+ chunk decision note in README]
M3  embedder.py + store.py → data/chroma/                 [384-dim assert]
M4  retriever.py (k, threshold, scheme filter, trace)
M5  prompt_builder.py + provider.py + generator.py (+ L6 validations)
M6  guardrails/{pii,intent,refusal}.py
M7  app.py (Streamlit, 6 states, disclaimer, 3 examples)
M8  eval/{test_set.yaml,run_eval.py} → tune threshold/k → docs/sample_qa.md
M9  README.md, docs/sources.csv, disclaimer snippet
M10 demo_script.md, answer cache warm, 3-min rehearsal / video
```

Each milestone ends green on `pytest tests/ -q`. The build phase (M1–M3) is fully LLM-free, so
the index can be built and verified before any API key exists — useful when keys are late.
