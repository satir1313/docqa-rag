"""Latent Semantic Analysis embedder: TF-IDF (unigrams + bigrams) -> truncated SVD.

Why this exists: it is deterministic, needs no model download or API key, and makes the whole
project reproducible offline (CI included). It is a *baseline* dense signal, not a substitute
for a neural embedding model. For real workloads use `SentenceTransformerEmbedder` or any
other implementation of the `Embedder` protocol.

Implementation notes:
- Pure NumPy; the SVD input is a dense (n_chunks x vocab) matrix, so this embedder is meant for
  corpora up to a few tens of thousands of chunks. Beyond that, use a pretrained model.
- State is saved as `.npz` + JSON (no pickle), so loading an index never executes code.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from docqa.embeddings.base import l2_normalize
from docqa.text import tokenize

_STATE_NPZ = "lsa_state.npz"
_STATE_VOCAB = "lsa_vocab.json"


def _terms(text: str) -> list[str]:
    tokens = tokenize(text)
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]


class LSAEmbedder:
    name = "lsa"

    def __init__(self, dim: int = 64) -> None:
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self.requested_dim = dim
        self.dim = dim
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.components: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    # ---- training -------------------------------------------------------------------------
    def fit(self, texts: list[str]) -> None:
        if len(texts) < 2:
            raise ValueError("LSA needs at least 2 texts to fit")
        term_lists = [_terms(t) for t in texts]
        df: Counter[str] = Counter()
        for terms in term_lists:
            df.update(set(terms))
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array(
            [math.log((1 + n) / (1 + df[t])) + 1.0 for t in self.vocab], dtype=np.float32
        )
        tfidf = self._tfidf_from_terms(term_lists)
        _, singular, vt = np.linalg.svd(tfidf, full_matrices=False)
        keep = int(min(self.requested_dim, int((singular > 1e-6).sum())))
        self.components = vt[:keep].astype(np.float32)
        self.dim = keep

    def _tfidf_from_terms(self, term_lists: list[list[str]]) -> np.ndarray:
        matrix = np.zeros((len(term_lists), len(self.vocab)), dtype=np.float32)
        for row, terms in enumerate(term_lists):
            for term, count in Counter(terms).items():
                col = self.vocab.get(term)
                if col is not None:
                    matrix[row, col] = 1.0 + math.log(count)  # sublinear tf
        return l2_normalize(matrix * self.idf)

    # ---- inference ------------------------------------------------------------------------
    def _embed(self, texts: list[str]) -> np.ndarray:
        if not self.vocab:
            raise RuntimeError("embedder is not fitted")
        tfidf = self._tfidf_from_terms([_terms(t) for t in texts])
        return l2_normalize(tfidf @ self.components.T)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text])[0]

    # ---- persistence ----------------------------------------------------------------------
    def save(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / _STATE_NPZ, idf=self.idf, components=self.components)
        (directory / _STATE_VOCAB).write_text(json.dumps(list(self.vocab)), encoding="utf-8")
        return {"requested_dim": self.requested_dim, "dim": self.dim}

    @classmethod
    def load(cls, directory: Path, params: dict[str, Any]) -> LSAEmbedder:
        embedder = cls(dim=int(params.get("requested_dim", 64)))
        with np.load(directory / _STATE_NPZ, allow_pickle=False) as state:
            embedder.idf = state["idf"].astype(np.float32)
            embedder.components = state["components"].astype(np.float32)
        terms = json.loads((directory / _STATE_VOCAB).read_text(encoding="utf-8"))
        embedder.vocab = {term: i for i, term in enumerate(terms)}
        embedder.dim = int(embedder.components.shape[0])
        return embedder
