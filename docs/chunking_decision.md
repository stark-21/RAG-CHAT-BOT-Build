# P2 Chunking Decision

Records what the real corpus looked like and why each splitter was chosen. Every
observation below comes from the 10 snapshots in `data/raw/` (2026-09-27), not from
assumptions.

## 1. What the data actually is

| doc | lines | tables found/kept | headings marked | dominant shape |
| --- | --- | --- | --- | --- |
| `S1__overview` | 202 | 4 / 1 | 26 | heading-delimited label/value facts + one holdings table |
| `S2__overview` | 233 | 4 / 1 | 26 | same |
| `S3__overview` | 210 | 3 / 1 | 25 | same |
| `S4__overview` | 244 | 4 / 1 | 26 | same |
| `S5__overview` | 593 | 4 / 1 | 30 | same, longest page |
| `S3__amfi_categorisation` | 219 | 0 / 0 | 72 | heading-delimited prose |
| `S3__sebi_elss` | 41 | 0 / 0 | 6 | heading-delimited prose |
| `GEN__download_cas` | 53 | 0 / 0 | 34 | link list |
| `GEN__tax_regime` | 102 | 0 / 0 | 34 | heading-delimited prose |
| `GEN__sebi_cas` | 7 | 0 / 0 | 3 | single short section |

Line counts, table counts and heading counts are the `stats` block now persisted in
each `data/raw/**/*.meta.json`, so these figures can be re-verified without re-running
the fetch.

Three findings drove the design:

1. **Fee and exit-load facts are label/value pairs, not tables.** Every scheme page
   states them as two consecutive lines:

   ```
   Expense ratio
   1.03%
   Min. for SIP
   ₹100
   Exit load of 1% if redeemed within 1 year
   ```

   `implementation.md` P2 anticipated a table-aware splitter for fees. In this corpus
   that is the wrong tool: a table splitter would leave `Expense ratio` and `1.03%` as
   two independent chunks, and the second one has no meaning on its own. The P2
   acceptance criterion "fee/exit-load table per scheme with header repeated" is
   therefore met by a *label/value* splitter for the real data, and by
   `TableAwareSplitter` only for the `fees` doc type (covered by a synthetic fixture
   because no fee table exists in the corpus).

2. **Only one table survives the facts-only filter per scheme page** — the holdings
   table. Each scheme page contains 3-4 `<table>` elements; the performance and
   peer-ranking ones are replaced by `[performance data excluded - facts-only policy]`
   in P1. The holdings table is real retrieval material, and at 50 rows it is ~1.2k
   tokens, so it must be split *with the header repeated* or a retrieved `0.66%` has no
   column meaning.

3. **No KIM, SID, or factsheet documents exist.** `https://www.hdfcfund.com/mutual-funds/fund-documents/kim`
   returns HTTP 403 to automated clients, so all five `kim_sid` slots failed. The
   splitters for those doc types are therefore implemented and unit-tested against
   fixtures, not against live data.

## 2. Cleaner changes required first (P2 is "cleaner integration")

The pages initially extracted as 724-1131 lines, of which roughly lines 0-60 were site
navigation (`Stocks`, `Intraday`, `ETF Screener`, `IPO`, `MTFs`, `Demat Account`,
`F&O`, …) and everything past ~250 was a footer containing a list of *other HDFC
schemes* and an A-Z index. That is both retrieval noise and a cross-scheme
contamination risk.

| change | where | effect |
| --- | --- | --- |
| `pick_content_root()` prefers `<main>`, `[role="main"]`, then the largest container whose class matches a main/content hint, else `<body>` | `src/ingest/cleaner.py` | Groww pages: content is in `div.pw14MainWrapper`, which yields 237 lines before filtering and 202-593 in the stored snapshot, down from 724-1131. SEBI ELSS resolves to `div.page-content-wrapper`; AMFI has no landmark and falls back to `<body>`. The `> 800` char guard keeps small fragments and tests working. |
| `mark_headings()` rewrites `h1`-`h6` as `#`-prefixed markers | `src/ingest/cleaner.py` | 25-30 real headings per scheme page, 72 on the AMFI categorisation page; gives the section splitter actual structure instead of guesses. |
| breadcrumb-trail line patterns | `src/ingest/cleaner.py` | Removed `Home / Personal Finance and Investment / … / Content`, which had produced a chunk whose entire body was navigation. |

A generic "repeated line across documents" heuristic was considered and rejected: the
fact `Expense ratio` appears on all five scheme pages, so cross-document repetition
cannot distinguish chrome from a fact that happens to be common.

## 3. Splitters chosen

| doc_type | splitter | reason from the data |
| --- | --- | --- |
| `scheme_page` | `LabelValueSplitter` | label/value facts are the bulk of the content; merges `Min. for SIP` + `₹100` into one line so a fact cannot be split across chunks |
| `education`, `guide`, `kim_sid`, `factsheet`, `riskometer` | `SectionSplitter` | heading-delimited prose; AMFI pages contribute 72 and 34 headings respectively |
| `fees` | `TableAwareSplitter` | no live example; covered by a synthetic KIM-style slab table fixture |
| `faq` | `QASplitter` | no live example; embeds the question text with its answer |
| anything else | `RecursiveSplitter` | dependency-free fallback |

`RecursiveSplitter` is implemented locally rather than pulling in
`langchain-text-splitters`, which would drag in `langchain-core` for ~30 lines of
logic. ADR A5 already avoids heavy frameworks; this keeps that choice at the chunker
too.

## 4. Defects found by running the splitter on real data

All four were found by inspecting `data/processed/chunks.jsonl`, and each has a
regression test named after the failure in `tests/test_chunker.py`.

1. **Sentinel tables were never parsed.** `TABLE_OPEN.format(n="")` produces
   `"<TABLE >"`, which does not match `"<TABLE 1>"`, so `<TABLE 1>` was treated as an
   ordinary text line and the holdings rows were packed as prose. Holdings arrived in
   3 chunks, one of which began mid-table with no header.
2. **Header repetition was lost on continuation chunks** even after the fix, because a
   50-row table that overflowed the cap was re-split by the recursive fallback with no
   awareness of the header. Tables now reserve prefix room, and the fallback repeats
   the first row when it is a table row.
3. **`#2 in India` parsed as a heading.** Groww renders a fund rank starting with `#`,
   which reset the heading path and produced a section labelled `#2 in India`. The
   heading pattern now requires a letter after the hashes.
4. **Lines were glued together in the recursive fallback.** The overlap tail was joined
   with `""` instead of `"\n"`, producing lines such as
   `Eicher Motors Ltd | Consumer Discretionary | Equity | 1.03%SRF Ltd | Materials`.

Also fixed: the section label is the **leaf** heading. The distributor pages nest
unrelated sections several levels deep (an `h5` sits directly under an `h2`), so
retaining the full outline path produced misleading citation labels like
`Holdings ( 50 ) > Understand terms > Tax`.

## 4a. Leaf headings that repeat within a document

The leaf-only rule above has a failure mode, and the real corpus hits it hard. A scheme
page states the same term in several unrelated places:

```
## Understand terms
##### Tax                     -> glossary definition
##### Exit load               -> glossary definition
### Exit Load                -> the actual fee schedule
### Exit load, stamp duty and tax
#### Exit load               -> the same figure restated
#### Tax implication         -> capital-gains treatment
```

`Exit Load` and `Exit load` are the same name to a reader, and they were the same name to
the P4 section-keyed dedupe cap, which folds case. Measuring the built corpus: **18 groups
covering 64 of 188 chunks** shared a section label with another chunk in the same
document. Two consequences, the second the more serious:

1. a citation to `Tax` did not say *which* `Tax`;
2. the dedupe cap of 2 per section grouped **unrelated** sections together, so it could
   suppress genuinely distinct evidence. That is a latent correctness risk in the
   retrieval hot path, not a cosmetic one.

`Exit Load` and `Exit load` passing as distinct because of capitalisation alone is the
clearest evidence the rule needed the case-folded comparison.

**Fix** (`_section_labels` in `src/ingest/chunker.py`): keep the bare leaf whenever it is
unique inside its document, and otherwise prepend the shortest run of ancestors that
separates every member of the group. Grouping is case-insensitive to match the dedupe
key. The full path remains the fallback when no ancestor run separates them, so a genuine
tie degrades to the whole outline rather than colliding again. Most chunks keep a short
label, which is what the leaf-only rule was for:

```
Tax
Understand terms > Exit load
Understand terms > Exit Load
Stamp duty
Tax implication
```

**Also dropped**: 4 chunks whose body only restated lines already carried in full by an
earlier chunk (`S1__overview__0013`, `S2__overview__0014`, `S4__overview__0014`,
`S5__overview__0025` before renumbering). Containment is order-sensitive contiguous line
matching, so a table listing the same rows in a different order is real content and
survives.

Verified after the rebuild: across all 10 documents, **0 cases** where two distinct
heading paths resolve to the same label. A label shared by several chunks is now always
one section split into pieces, which is correct for a 326-row holdings table.

Cost: the qualified prefix is part of the embedded text, so it slightly dilutes the
vector for the affected chunks — the exit-load query's top score moved 0.835 → 0.815, well
clear of the 0.30 threshold, with the ranking unchanged.

## 5. Result

```
documents=10  chunks=184  data/processed/chunks.jsonl
tokens: total=29466 min=14 median=72 max=500 (cap=500)
by splitter: LabelValueSplitter=124  SectionSplitter=60
by scheme:   GEN=26  S1=21  S2=22  S3=56  S4=22  S5=37
```

184 rather than the original 188: four chunks were restatements of a line already present,
removed by the containment pass in 4a.

Verified on the built artifact:

- every chunk is `<= 500` estimated tokens (`all-MiniLM-L6-v2` limit is 512);
- every chunk begins with `{scheme_name} — {section}`, except the 6 chunks that precede a
  document's first heading and therefore have no section of their own — their `section` is
  the scheme name alone, which still identifies the source;
- within each document, no two distinct heading paths share a section label (see 4a);
- `chunk_id` values are unique and sequential;
- holdings tables repeat `Name | Sector | Instruments | Assets` on every piece;
- `Net Payables | Unspecified | Net Payables | -0.10%` survives;
- no chunk contains site navigation or a `<TABLE>` sentinel;
- no performance figures or rankings are present.

## 6. Known limitations carried into P3

- **KIM/SID/factsheet are unpopulated.** Their splitters are fixture-tested only. If the
  HDFC 403 is ever resolved, re-run `python -m src.ingest.build_index --rebuild`; no
  chunker change is required.
- **`S5__overview` produces 37 chunks**, roughly 1.8x `S1`. The page is twice as long
  and its "Fund house" section lists other schemes managed by HDFC. That list is
  correctly labelled but is cross-scheme text inside a scheme page; a strict reading of
  "one answer per scheme" may want it filtered. Left in place rather than special-cased,
  because hand-filtering scheme lists is exactly the kind of rule that silently drops
  real facts.
- **Token counts are estimates**, `max(words / 1.35, chars / 4)`, chosen to over-count
  rather than under-count. P3 replaces this with the real MiniLM tokenizer, at which
  point `count_tokens()` becomes authoritative.
- **`See All` and `View details`** survive as short lines. They are source artefacts,
  not facts, but they are harmless next to a prefixed fact and removing them would mean
  maintaining a distributor-specific blocklist.
