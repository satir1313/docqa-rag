"""Runtime configuration. Everything is overridable via DOCQA_* environment variables so the
same artifact runs unchanged in CI, Docker and local development (12-factor style)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"DOCQA_{name}", default)


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = Path("data/corpus")
    index_dir: Path = Path("var/index")

    # chunking
    chunk_max_tokens: int = 120
    chunk_overlap_tokens: int = 25

    # embeddings
    embedder: str = "lsa"  # "lsa" (offline baseline) | "sentence-transformers"
    embedding_dim: int = 64
    st_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # retrieval
    default_mode: str = "hybrid"  # chosen from the ablation in docs/EVALUATION.md
    retrieval_depth: int = 30  # candidates fetched from each retriever before fusion
    rrf_k: int = 60
    rerank_depth: int = 15
    rerank_fusion_weight: float = 0.35
    max_per_doc: int = 3

    # generation
    generator: str = "extractive"  # "extractive" | "anthropic"
    min_support: float = 0.5
    llm_model: str = "claude-sonnet-5-5"
    llm_timeout_s: float = 30.0
    llm_max_tokens: int = 400

    # service
    api_key: str | None = None
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            corpus_dir=Path(_env("CORPUS_DIR", str(d.corpus_dir)) or d.corpus_dir),
            index_dir=Path(_env("INDEX_DIR", str(d.index_dir)) or d.index_dir),
            chunk_max_tokens=int(_env("CHUNK_MAX_TOKENS", str(d.chunk_max_tokens)) or 0),
            chunk_overlap_tokens=int(_env("CHUNK_OVERLAP_TOKENS", str(d.chunk_overlap_tokens)) or 0),
            embedder=_env("EMBEDDER", d.embedder) or d.embedder,
            embedding_dim=int(_env("EMBEDDING_DIM", str(d.embedding_dim)) or 0),
            st_model=_env("ST_MODEL", d.st_model) or d.st_model,
            default_mode=_env("DEFAULT_MODE", d.default_mode) or d.default_mode,
            retrieval_depth=int(_env("RETRIEVAL_DEPTH", str(d.retrieval_depth)) or 0),
            rrf_k=int(_env("RRF_K", str(d.rrf_k)) or 0),
            rerank_depth=int(_env("RERANK_DEPTH", str(d.rerank_depth)) or 0),
            rerank_fusion_weight=float(
                _env("RERANK_FUSION_WEIGHT", str(d.rerank_fusion_weight)) or 0
            ),
            max_per_doc=int(_env("MAX_PER_DOC", str(d.max_per_doc)) or 0),
            generator=_env("GENERATOR", d.generator) or d.generator,
            min_support=float(_env("MIN_SUPPORT", str(d.min_support)) or 0),
            llm_model=_env("LLM_MODEL", d.llm_model) or d.llm_model,
            llm_timeout_s=float(_env("LLM_TIMEOUT_S", str(d.llm_timeout_s)) or 0),
            llm_max_tokens=int(_env("LLM_MAX_TOKENS", str(d.llm_max_tokens)) or 0),
            api_key=_env("API_KEY"),
            log_level=(_env("LOG_LEVEL", d.log_level) or d.log_level).upper(),
        )
