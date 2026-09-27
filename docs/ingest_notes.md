# Ingestion Notes (Phase P1)

Written during P1 so later phases and the README are based on the corpus that actually
exists, not on an assumed one.

## What is in `data/raw/`

10 successful snapshots across 6 publisher pages, all HTML, no PDFs:

| doc_id | Scheme | doc_type | Publisher | Chars | Tables |
| --- | --- | --- | --- | --- | --- |
| `S1__overview` | S1 Large Cap | `scheme_page` | HDFC AMC (via Groww) | ~17.6k | 1 |
| `S2__overview` | S2 Flexi Cap | `scheme_page` | HDFC AMC (via Groww) | ~19.7k | 1 |
| `S3__overview` | S3 ELSS | `scheme_page` | HDFC AMC (via Groww) | ~18.2k | 1 |
| `S4__overview` | S4 Small Cap | `scheme_page` | HDFC AMC (via Groww) | ~20.0k | 1 |
| `S5__overview` | S5 Balanced Advantage | `scheme_page` | HDFC AMC (via Groww) | ~45.4k | 1 |
| `S3__amfi_categorisation` | S3 ELSS | `education` | AMFI | ~16.1k | 0 |
| `S3__sebi_elss` | S3 ELSS | `education` | SEBI | ~3.0k | 0 |
| `GEN__download_cas` | GEN (all schemes) | `guide` | AMFI | ~2.4k | 0 |
| `GEN__tax_regime` | GEN (all schemes) | `guide` | AMFI | ~14.8k | 0 |
| `GEN__sebi_cas` | GEN (all schemes) | `education` | SEBI | ~0.6k | 0 |

## Facts the corpus can answer today

| Question type | Grounded in | Note |
| --- | --- | --- |
| Expense ratio | `S*__overview` | present as a label/value pair, e.g. `Expense ratio` / `1.03%` |
| Exit load | `S*__overview` | label/value pair, e.g. `1% if redeemed within 1 year` |
| Minimum SIP / lumpsum | `S*__overview` | `Min. for SIP` / value |
| Benchmark | `S*__overview` | label/value pair |
| Risk band | `S*__overview` | `Very High risk` prose, not the SEBI riskometer wording |
| ELSS 3-year lock-in | `S3__sebi_elss`, `S3__amfi_categorisation` | statutory fact, Section 80C |
| Download a statement | `GEN__download_cas`, `GEN__sebi_cas` | CAMS / KFintech / MF Central |
| Capital-gains tax | `GEN__tax_regime` | official tax reference, **not** personal tax advice |

**Gap:** the scheme pages do not state the ELSS lock-in, so the lock-in answer is grounded in
the AMFI/SEBI education pages and cited to them, not to the scheme page.

## Failed slots (recorded in `logs/ingest_errors.json`, run never aborts)

| Slot | URL | Reason | Note |
| --- | --- | --- | --- |
| `S1..S5__kim` | `https://www.hdfcfund.com/mutual-funds/fund-documents/kim` | `http_403` | Real AMC page, but its WAF refuses automated clients from this network. HDFC's own domain is `hdfcfund.com` (not `hdfcassetmanagement.com`, which does not resolve). |

`investor.sebi.gov.in` returned `http_502` on the first run and succeeded on the second, so SEBI
reachability is **flaky** rather than blocked. Both SEBI pages are now captured; if a future run
reports `http_502` for them, re-run before treating them as gone.

## Consequences for later phases

1. **No PDFs and no KIM/SID in the corpus.** `kim_sid` and `factsheet` splitters in P2 will have
   no input yet. They stay implemented and unit-tested against fixtures, and activate the moment a
   PDF source is added (manual browser save into `data/raw/<scheme_id>/` is the documented fallback,
   and `load_snapshots()` will pick it up).
2. **`scheme_page` is the dominant doc type.** P2 must add a `scheme_page` splitter. Observed
   structure: `get_text("\n")` yields a clean label/value line pair per fact
   (`Expense ratio` then `1.03%`), plus one `<TABLE n>` block per holdings table. A
   **label/value-aware** splitter matters more here than a table-aware one — the fee and exit-load
   figures are not inside `<table>` elements.
3. **Benchmark and risk band are single label/value pairs**, so they will land in whatever chunk
   covers that region. The context prefix (`{scheme_name} — {section}`) is what keeps them
   unambiguous across the five schemes.
4. **`GEN` is a pseudo-scheme.** Scheme-filtered retrieval in P4 must include `GEN` chunks
   (`where={"$or": [{"scheme_id": "S3"}, {"scheme_id": "GEN"}]}`), otherwise the statement and
   tax questions lose their evidence the moment a scheme filter is applied.
5. **Performance data is physically absent** (see below), which makes the P5 number-in-context
   check meaningful instead of noisy.

## Facts-only enforcement at ingestion

`content_policy.include_performance: false` in `src/ingest/sources.yaml` makes the cleaner drop,
before chunking:

- the historic-returns table, the category-average/rank table, and the peer-comparison table
  (measured after the P2 content-root change: 3-4 tables found per scheme page, exactly 1
  kept — the holdings table);
- performance prose lines (`Historic returns`, `Return calculator`, `Category average`,
  `Fund returns`, `1Y Returns`, `Rank ...`);
- return-ticker fragments, which the pages render as `+8.71` on one line and `%` on the next.

Verified after each run: **0 performance markers and 0 lone signed figures in any snapshot.**

A negative holding weight such as `Net Payables | Unspecified | Net Payables | -0.10%` is
**kept** — a sign inside a table row is portfolio data, not a return. This is why the filter
matches whole lines and ticker shapes rather than "any signed number".
