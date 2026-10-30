"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion (RRF), then an
optional rerank stage and a per-document diversity cap.

RRF (Cormack et al., 2009) fuses *ranks*, not scores, so it needs no score calibration between
a lexical retriever (unbounded BM25 scores) and a dense one (cosine in [-1, 1]):
    rrf(d) = sum over retrievers r of  w_r / (k + rank_r(d))
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from docqa.config import Settings
from docqa.embeddings import Embedder
from docqa.indexing import BM25Index, VectorIndex
from docqa.retrieval.filters import Filters, matches_filters
from docqa.retrieval.rerank import FeatureReranker, Reranker
from docqa.schemas import Chunk, Hit, RetrievalMode, RetrievalResult
from docqa.text import tokenize


def reciprocal_rank_fusion(
    rankings: dict[str, Sequence[int]], *, k: int = 60, weights: dict[str, float] | None = None
) -> dict[int, tuple[float, dict[str, int]]]:
    """Fuse named best-first rankings of item ids.

    Returns {item: (fused_score, {retriever_name: 1-based rank})}.
    """
    fused: dict[int, tuple[float, dict[str, int]]] = {}
    for name, ranking in rankings.items():
        weight = (weights or {}).get(name, 1.0)
        for position, item in enumerate(ranking, start=1):
            score, ranks = fused.get(item, (0.0, {}))
            ranks = {**ranks, name: position}
            fused[item] = (score + weight / (k + position), ranks)
    return fused


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        vectors: np.ndarray,
        embedder: Embedder,
        settings: Settings | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.chunks = chunks
        self.embedder = embedder
        self.bm25 = BM25Index([tokenize(c.index_text) for c in chunks])
        self.vectors = VectorIndex(vectors)
        self.reranker: Reranker = reranker or FeatureReranker(self.settings.rerank_fusion_weight)

    def _allowed(self, filters: Filters | None) -> np.ndarray | None:
        if not filters:
            return None
        return np.fromiter(
            (matches_filters(c.metadata, filters) for c in self.chunks),
            dtype=bool,
            count=len(self.chunks),
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: Filters | None = None,
        mode: RetrievalMode = RetrievalMode.HYBRID_RERANK,
    ) -> RetrievalResult:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        allowed = self._allowed(filters)
        if allowed is not None and not allowed.any():
            return RetrievalResult(query=query, mode=mode, hits=[], timings_ms={"total": 0.0})

        depth = max(self.settings.retrieval_depth, k)
        bm25_ranked: list[tuple[int, float]] = []
        dense_ranked: list[tuple[int, float]] = []

        if mode is not RetrievalMode.DENSE:
            t = time.perf_counter()
            bm25_ranked = self.bm25.search(tokenize(query), depth, allowed)
            timings["bm25"] = (time.perf_counter() - t) * 1000
        if mode is not RetrievalMode.BM25:
            t = time.perf_counter()
            dense_ranked = self.vectors.search(self.embedder.embed_query(query), depth, allowed)
            timings["dense"] = (time.perf_counter() - t) * 1000

        raw_scores = {"bm25": dict(bm25_ranked), "dense": dict(dense_ranked)}
        fused = reciprocal_rank_fusion(
            {"bm25": [i for i, _ in bm25_ranked], "dense": [i for i, _ in dense_ranked]},
            k=self.settings.rrf_k,
        )
        candidates = len(fused)

        if mode is RetrievalMode.BM25:
            ordered = [(i, s) for i, s in bm25_ranked]
        elif mode is RetrievalMode.DENSE:
            ordered = [(i, s) for i, s in dense_ranked]
        else:
            ordered = sorted(
                ((i, score) for i, (score, _) in fused.items()), key=lambda x: (-x[1], x[0])
            )

        hits: list[Hit] = []
        pool = ordered if mode is not RetrievalMode.HYBRID_RERANK else ordered[: max(
            self.settings.rerank_depth, k
        )]
        for idx, score in pool:
            ranks = fused[idx][1]
            signals = {"rrf": round(fused[idx][0], 6)}
            if "bm25" in ranks:
                signals["bm25_rank"] = float(ranks["bm25"])
                signals["bm25"] = round(raw_scores["bm25"][idx], 4)
            if "dense" in ranks:
                signals["dense_rank"] = float(ranks["dense"])
                signals["dense"] = round(raw_scores["dense"][idx], 4)
            hits.append(Hit(chunk=self.chunks[idx], score=score, rank=0, signals=signals))

        if mode is RetrievalMode.HYBRID_RERANK:
            t = time.perf_counter()
            hits = self.reranker.rerank(query, hits)
            timings["rerank"] = (time.perf_counter() - t) * 1000

        hits = self._diversify(hits, k)
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return RetrievalResult(
            query=query,
            mode=mode,
            hits=[h.model_copy(update={"rank": r}) for r, h in enumerate(hits, start=1)],
            timings_ms={name: round(v, 3) for name, v in timings.items()},
            candidates=candidates,
        )

    def _diversify(self, hits: list[Hit], k: int) -> list[Hit]:
        """Cap chunks per document so one long document cannot crowd out every other source."""
        cap = self.settings.max_per_doc
        per_doc: dict[str, int] = {}
        kept: list[Hit] = []
        for hit in hits:
            if cap and per_doc.get(hit.chunk.doc_id, 0) >= cap:
                continue
            per_doc[hit.chunk.doc_id] = per_doc.get(hit.chunk.doc_id, 0) + 1
            kept.append(hit)
            if len(kept) == k:
                break
        return kept
