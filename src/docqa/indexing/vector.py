"""Exact (brute-force) cosine search over L2-normalised vectors.

Exact search is the right default up to ~10^5 chunks: no recall loss, no tuning, trivially
correct. Swap in FAISS/HNSW/pgvector behind this same `search` signature when you outgrow it.
"""

from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, matrix: np.ndarray) -> None:
        if matrix.ndim != 2:
            raise ValueError("vector matrix must be 2-dimensional")
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def search(
        self, query: np.ndarray, k: int, allowed: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        """Top-k (index, cosine) pairs with positive similarity, best first."""
        n = len(self)
        if n == 0 or k <= 0:
            return []
        sims = self.matrix @ query.astype(np.float32)
        if allowed is not None:
            sims = np.where(allowed, sims, -np.inf)
        k = min(k, n)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.lexsort((top, -sims[top]))]  # similarity desc, index asc
        return [(int(i), float(sims[i])) for i in top if np.isfinite(sims[i]) and sims[i] > 0]
