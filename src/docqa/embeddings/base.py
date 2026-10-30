"""Embedder interface plus a small factory so the index records *which* embedder built it.

An index built with one embedder is meaningless under another (different vector space), so the
manifest stores the embedder name + params and `load_embedder` refuses to guess.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def fit(self, texts: list[str]) -> None:
        """Learn corpus statistics. A no-op for pretrained models."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 L2-normalised vector."""

    def save(self, directory: Path) -> dict[str, Any]:
        """Persist learned state into `directory`; return JSON-serialisable params."""


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return (matrix / np.where(norms == 0, 1.0, norms)).astype(np.float32)


def make_embedder(kind: str, *, dim: int = 64, st_model: str = "") -> Embedder:
    if kind == "lsa":
        from docqa.embeddings.lsa import LSAEmbedder

        return LSAEmbedder(dim=dim)
    if kind == "sentence-transformers":
        from docqa.embeddings.st import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(model_name=st_model)
    raise ValueError(f"unknown embedder {kind!r} (expected 'lsa' or 'sentence-transformers')")


def load_embedder(name: str, params: dict[str, Any], directory: Path) -> Embedder:
    if name == "lsa":
        from docqa.embeddings.lsa import LSAEmbedder

        return LSAEmbedder.load(directory, params)
    if name == "sentence-transformers":
        from docqa.embeddings.st import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder.load(directory, params)
    raise ValueError(f"index was built with unknown embedder {name!r}")
