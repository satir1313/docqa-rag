"""Okapi BM25 over an inverted index (no external search engine required).

Scoring (per query term t, document d):
    idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))     # always non-negative
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np


class BM25Index:
    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.doc_len = np.array([len(d) for d in documents], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, tokens in enumerate(documents):
            for term, tf in Counter(tokens).items():
                self.postings[term].append((doc_idx, tf))
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def search(
        self, query_tokens: Sequence[str], k: int, allowed: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        """Return up to k (doc_index, score) pairs, best first. Ties break on index (stable)."""
        if not self.n_docs or k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_idx, tf in self.postings[term]:
                if allowed is not None and not allowed[doc_idx]:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl)
                scores[doc_idx] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
