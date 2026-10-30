"""Adapter for sentence-transformers models (optional dependency).

NOTE: this adapter is intentionally thin and is NOT exercised by the offline test suite or the
bundled evaluation numbers, because that would require downloading model weights. Run
`DOCQA_EMBEDDER=sentence-transformers docqa ingest && docqa eval` to evaluate it on your data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from docqa.embeddings.base import l2_normalize


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is not installed: pip install docqa-rag[embeddings]"
            ) from exc
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def fit(self, texts: list[str]) -> None:  # pretrained: nothing to learn
        return None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(texts, batch_size=32, show_progress_bar=False)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def save(self, directory: Path) -> dict[str, Any]:
        return {"model_name": self.model_name, "dim": self.dim}

    @classmethod
    def load(cls, directory: Path, params: dict[str, Any]) -> SentenceTransformerEmbedder:
        return cls(model_name=str(params["model_name"]))
