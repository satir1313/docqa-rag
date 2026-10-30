"""High-level orchestration: build an index from a corpus, load it, search and answer."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from docqa.config import Settings
from docqa.embeddings import Embedder, make_embedder
from docqa.generation import AnswerGenerator, AnthropicGenerator, ExtractiveGenerator
from docqa.indexing import IndexStore
from docqa.ingestion import ChunkerConfig, chunk_document, load_documents
from docqa.retrieval import Retriever
from docqa.retrieval.filters import Filters
from docqa.schemas import Answer, IngestReport, RetrievalMode, RetrievalResult

log = logging.getLogger("docqa.pipeline")


def make_generator(settings: Settings) -> AnswerGenerator:
    if settings.generator == "anthropic":
        return AnthropicGenerator(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model=settings.llm_model,
            min_support=settings.min_support,
            max_tokens=settings.llm_max_tokens,
            timeout_s=settings.llm_timeout_s,
        )
    if settings.generator == "extractive":
        return ExtractiveGenerator(min_support=settings.min_support)
    raise ValueError(f"unknown generator {settings.generator!r}")


def build_index(
    settings: Settings,
    corpus_dir: Path | None = None,
    index_dir: Path | None = None,
    embedder: Embedder | None = None,
) -> IngestReport:
    started = time.perf_counter()
    corpus_dir = Path(corpus_dir or settings.corpus_dir)
    index_dir = Path(index_dir or settings.index_dir)

    documents, skipped = load_documents(corpus_dir)
    if not documents:
        raise ValueError(f"no ingestible documents found in {corpus_dir}")
    cfg = ChunkerConfig(settings.chunk_max_tokens, settings.chunk_overlap_tokens)
    chunks = [chunk for doc in documents for chunk in chunk_document(doc, cfg)]
    if len(chunks) < 2:
        raise ValueError("need at least 2 chunks to build an index")

    embedder = embedder or make_embedder(
        settings.embedder, dim=settings.embedding_dim, st_model=settings.st_model
    )
    texts = [c.index_text for c in chunks]
    embedder.fit(texts)
    vectors = embedder.embed_documents(texts)

    snapshot = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in asdict(settings).items()
        if k != "api_key"
    }
    IndexStore.save(
        index_dir,
        chunks=chunks,
        vectors=vectors,
        embedder=embedder,
        documents=documents,
        settings_snapshot=snapshot,
    )
    sizes = [c.token_count for c in chunks]
    report = IngestReport(
        documents=len(documents),
        chunks=len(chunks),
        skipped=skipped,
        avg_chunk_tokens=round(float(np.mean(sizes)), 1),
        max_chunk_tokens=int(max(sizes)),
        embedder=embedder.name,
        embedding_dim=embedder.dim,
        duration_s=round(time.perf_counter() - started, 3),
        index_dir=str(index_dir),
    )
    log.info("index built", extra=report.model_dump())
    return report


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        settings: Settings,
        manifest: dict[str, Any],
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.settings = settings
        self.manifest = manifest

    @classmethod
    def load(
        cls,
        settings: Settings,
        index_dir: Path | None = None,
        generator: AnswerGenerator | None = None,
    ) -> RagPipeline:
        loaded = IndexStore.load(Path(index_dir or settings.index_dir))
        retriever = Retriever(loaded.chunks, loaded.vectors, loaded.embedder, settings)
        return cls(retriever, generator or make_generator(settings), settings, loaded.manifest)

    def search(
        self,
        query: str,
        k: int = 5,
        filters: Filters | None = None,
        mode: RetrievalMode | None = None,
    ) -> RetrievalResult:
        mode = mode or RetrievalMode(self.settings.default_mode)
        return self.retriever.retrieve(query, k=k, filters=filters, mode=mode)

    def ask(
        self, question: str, k: int = 5, filters: Filters | None = None
    ) -> tuple[Answer, RetrievalResult]:
        result = self.search(question, k=k, filters=filters)
        return self.generator.generate(question, result.hits), result
