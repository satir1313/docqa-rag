"""Typed data contracts shared across ingestion, retrieval, generation and the API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    source_path: str
    title: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    heading_path: list[str] = Field(default_factory=list)
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str

    @property
    def section(self) -> str:
        return " > ".join(self.heading_path)

    @property
    def index_text(self) -> str:
        """Text used for indexing: the heading path is prepended as contextual header, so a
        chunk that says "30 days" is still findable by the section it belongs to."""
        return f"{self.section}\n{self.text}" if self.heading_path else self.text


class RetrievalMode(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class Hit(BaseModel):
    chunk: Chunk
    score: float
    rank: int
    signals: dict[str, float] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    mode: RetrievalMode
    hits: list[Hit]
    timings_ms: dict[str, float] = Field(default_factory=dict)
    candidates: int = 0


class Citation(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    section: str
    quote: str | None = None


class Answer(BaseModel):
    text: str
    abstained: bool
    support: float
    citations: list[Citation] = Field(default_factory=list)
    generator: str


class IngestReport(BaseModel):
    documents: int
    chunks: int
    skipped: list[str] = Field(default_factory=list)
    avg_chunk_tokens: float
    max_chunk_tokens: int
    embedder: str
    embedding_dim: int
    duration_s: float
    index_dir: str
