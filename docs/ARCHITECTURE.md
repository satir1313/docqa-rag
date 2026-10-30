# Architecture

## Modules (`src/docqa`)

| Module | Responsibility |
|---|---|
| `ingestion/loaders.py` | Read `.md/.txt/.pdf`, parse front matter, stable slug `doc_id`s, skip-and-report bad files, fail on duplicate ids |
| `ingestion/chunking.py` | Block parser (headings, code fences, tables, lists, paragraphs) → packer with heading path, whole-block overlap, oversize splitting |
| `embeddings/` | `Embedder` protocol; offline LSA (TF-IDF → SVD in NumPy); optional sentence-transformers adapter |
| `indexing/` | `BM25Index`, exact cosine `VectorIndex`, `IndexStore` (atomic, versioned) |
| `retrieval/` | RRF fusion, metadata filters, feature reranker, `Retriever` |
| `generation/` | Support gate, extractive generator, Anthropic generator |
| `evaluation/` | Metrics and harness (ablation, splits, thresholds, gate) |
| `api/` | FastAPI app |
| `pipeline.py`, `cli.py`, `config.py` | Wiring |

## Decisions and trade-offs

**Chunk by structure, not by fixed window.** A fixed window splits a procedure mid-step and mixes
unrelated sections. Cost: chunk sizes vary; the packer enforces an upper bound only.

**Heading path as contextual header.** A chunk saying "30 days" is ambiguous; "Data retention >
Backups" is not. It is added to the *indexed* text, not the displayed text.

**RRF instead of score blending.** BM25 and cosine scores are on different scales; rank-based fusion
needs no calibration. It discards score magnitude, which is the trade-off.

**Feature reranker.** Cheap, explainable (coverage, bigram match, proximity, heading match), blended
with the fused score. Measured gain on this corpus: none. A cross-encoder is the realistic upgrade.

**Offline LSA embedder.** Makes the repo reproducible without downloads. At this corpus size it
adds little over BM25; that is documented rather than hidden.

**Atomic index.** Build in a temp dir, then rename, so a crashed ingest never leaves a half-written
index serving traffic. `schema_version` guards against loading incompatible layouts.

**Support gate before generation.** Cheap lexical check; saves LLM cost and prevents obviously
unsupported answers. Known weak (see EVALUATION.md).

**FastAPI note.** `api/app.py` intentionally does not use `from __future__ import annotations`:
dependency aliases defined inside `create_app` would otherwise be treated as query parameters.
