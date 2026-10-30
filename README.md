# docqa-rag

A small, production-minded **Retrieval-Augmented Generation** service for question answering over
internal documentation: structure-aware chunking, hybrid retrieval (BM25 + dense, fused with RRF),
citations, abstention, a FastAPI service, and — the part most demos skip — an **evaluation harness
with a CI quality gate**.

It runs fully **offline** by default (no API keys, no model downloads), so you can clone it and run
everything in under a minute. An optional Anthropic-backed generator and a sentence-transformers
embedder are available as adapters.

> **Data note.** The corpus in `data/corpus` is **synthetic** (a fictional company, "Northwind"),
> written for this repo. Nothing here comes from any employer or customer.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

docqa ingest                                   # build the index into var/index
docqa ask "How long are database backups retained?"
docqa search "rotate service account credentials" -k 3
docqa eval --out docs/eval-report.md           # retrieval + answer metrics
docqa serve                                    # http://127.0.0.1:8000/docs

pytest -q && ruff check .
```

```bash
curl -s localhost:8000/v1/ask -H 'content-type: application/json' \
  -d '{"question": "How often must API tokens be rotated?"}'
```

## Architecture

```
 .md/.txt/.pdf ─► loaders ─► structure-aware chunker ─► index (atomic, versioned on disk)
                                                         ├─ BM25 inverted index
                                                         └─ dense vectors (LSA or sentence-transformers)
 query ─► BM25 ─┐
                ├─► Reciprocal Rank Fusion ─► feature reranker (optional) ─► per-doc cap ─► hits
 query ─► dense ┘
 hits ─► support gate ─► generator (extractive | Anthropic) ─► answer + citations, or abstain
```

Details and trade-offs: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

Key design points:

- **Chunking respects structure.** Chunks never cross a heading, code fences/tables/lists are kept
  whole, every chunk carries its heading path as a contextual header, overlap is whole-block, and a
  hard guarantee holds that no chunk exceeds `max_tokens`.
- **Hybrid retrieval** with RRF avoids having to normalise incomparable BM25 and cosine scores.
- **Abstention.** If the evidence is weak the system says so instead of answering (also saves LLM
  cost, since the gate runs before the call).
- **LLM generator hygiene.** Retrieved text is fenced as untrusted data, citations are validated
  against the chunks actually supplied, an answer without a valid citation is rejected, retries use
  exponential backoff, and the HTTP transport is injectable so it is unit-tested offline.
- **Index is a versioned artifact.** Written to a temp dir and swapped in atomically; loading checks
  schema version and integrity; vector state is saved as `npz` + JSON (no pickle).
- **Service basics.** `/healthz`, `/readyz`, optional API-key auth (constant-time compare),
  request IDs, JSON logs, bounded inputs.

## Evaluation (read this before trusting any number)

46 hand-written questions (34 answerable, 12 unanswerable) over 53 chunks, split dev/test.
Relevance is "right document + right section", so results do not depend on the chunker.

| Mode | Recall@1 | Recall@5 | MRR@10 | nDCG@10 | p50 latency |
|---|---|---|---|---|---|
| bm25 | 0.779 | 0.941 | 0.858 | 0.889 | 0.11 ms |
| dense (LSA) | 0.779 | 0.941 | 0.858 | 0.889 | 0.27 ms |
| **hybrid** (default) | 0.809 | 0.941 | 0.875 | 0.902 | 0.46 ms |
| hybrid + rerank | 0.779 | 0.941 | 0.858 | 0.889 | 1.51 ms |

Honest reading:

- The corpus is tiny and synthetic, and **I wrote both the documents and the questions**. These
  numbers are a *regression signal*, not a benchmark. Differences between modes are about one
  question and are **not statistically meaningful**.
- Dense equals BM25 here because LSA on 53 chunks runs at near-full rank and behaves like TF-IDF.
  A real embedding model is the obvious next experiment (adapter included, **not tested here**).
- The reranker did **not** help on this data. It is kept as an ablation, and not enabled by default.
- Two failures (q29, q34) are genuine vocabulary mismatches; I did not patch the pipeline to fix
  them.
- **Abstention is the weak spot.** The lexical-coverage gate at `min_support=0.5` (chosen on dev)
  gives 8% false abstention on dev but **41% on test**, and only 67% correct abstention. It does not
  generalise, and the test split is not pristine (I saw aggregate results before splitting). The
  threshold sweep is in the report. A production system needs a stronger groundedness signal
  (cross-encoder / NLI / LLM judge) and a larger labelled set.

Full methodology: [docs/EVALUATION.md](docs/EVALUATION.md) · generated report:
[docs/eval-report.md](docs/eval-report.md).

CI fails if Recall@5 < 0.85 or MRR@10 < 0.75 (thresholds sit below observed values, so they catch
regressions rather than noise).

## Configuration

All via `DOCQA_*` environment variables (see `src/docqa/config.py`).

| Variable | Default | Meaning |
|---|---|---|
| `DOCQA_CORPUS_DIR` / `DOCQA_INDEX_DIR` | `data/corpus` / `var/index` | input / index location |
| `DOCQA_CHUNK_MAX_TOKENS` / `_OVERLAP_TOKENS` | 120 / 25 | chunk size and overlap |
| `DOCQA_EMBEDDER` | `lsa` | `lsa` (offline) or `st` (sentence-transformers) |
| `DOCQA_DEFAULT_MODE` | `hybrid` | `bm25`, `dense`, `hybrid`, `hybrid_rerank` |
| `DOCQA_MAX_PER_DOC` | 3 | diversity cap per document |
| `DOCQA_GENERATOR` | `extractive` | `extractive` or `anthropic` |
| `DOCQA_MIN_SUPPORT` | 0.5 | abstention threshold |
| `DOCQA_LLM_MODEL` | `claude-sonnet-5-5` | model for the `anthropic` generator |
| `ANTHROPIC_API_KEY` | – | required for the `anthropic` generator |
| `DOCQA_API_KEY` | – | if set, `/v1/*` requires `X-API-Key` |

## Limitations

- Synthetic, tiny corpus; no claim of real-world accuracy.
- Exact (brute-force) vector search: fine up to ~100k chunks, then use an ANN index.
- The Anthropic generator and sentence-transformers adapter are covered by unit tests with fakes but
  were **not run against the live API/model** in this repo.
- Dockerfile is provided but was not built in the authoring environment.
- Single-process index in memory; no incremental updates (rebuild is cheap and atomic).
- Markdown-first; PDF support is text extraction only (no OCR, no layout/table recovery).

## Roadmap

Real embedding model + larger public corpus (e.g. an open docs set) with more labelled questions;
cross-encoder reranking; groundedness judge for abstention; incremental indexing; ANN index.

## License

MIT
