"""FastAPI service.

Operational properties:
- /healthz (liveness) never touches the index; /readyz (readiness) reports 503 until an index
  is loaded, so orchestrators don't route traffic to a pod that cannot answer.
- Optional API-key auth (`DOCQA_API_KEY`) with constant-time comparison.
- Every request gets an X-Request-ID (propagated if supplied) and one structured log line.
- Input is validated and bounded (query length, k) before any work happens.
- Handlers are sync `def`, so FastAPI runs the CPU-bound retrieval in its threadpool instead of
  blocking the event loop. Retrieval is read-only, hence safe to run concurrently.
"""

# NOTE: no `from __future__ import annotations` here on purpose. FastAPI resolves the
# `Auth`/`Pipe` dependency aliases defined inside create_app(); with postponed annotations they
# are unresolvable strings and get silently treated as required query parameters (HTTP 422).
# tests/test_api.py guards this.
import hmac
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from docqa import __version__
from docqa.config import Settings
from docqa.logging_utils import configure_logging
from docqa.pipeline import RagPipeline
from docqa.schemas import Answer, Hit, RetrievalMode

log = logging.getLogger("docqa.api")
_ValueOrList = str | list[str]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    mode: RetrievalMode | None = None  # defaults to the server's configured mode
    filters: dict[str, _ValueOrList] | None = None

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, _ValueOrList] | None = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class SourceOut(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    section: str
    score: float
    text: str
    signals: dict[str, float]


class SearchResponse(BaseModel):
    query: str
    mode: RetrievalMode
    results: list[SourceOut]
    timings_ms: dict[str, float]


class AskResponse(BaseModel):
    answer: Answer
    sources: list[SourceOut]
    timings_ms: dict[str, float]


def _source(hit: Hit) -> SourceOut:
    return SourceOut(
        rank=hit.rank,
        chunk_id=hit.chunk.chunk_id,
        doc_id=hit.chunk.doc_id,
        section=hit.chunk.section,
        score=round(hit.score, 6),
        text=hit.chunk.text,
        signals=hit.signals,
    )


def create_app(settings: Settings | None = None, pipeline: RagPipeline | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is not None:
            app.state.pipeline = pipeline
        else:
            try:
                app.state.pipeline = RagPipeline.load(settings)
                log.info("index loaded", extra={"chunks": app.state.pipeline.manifest["num_chunks"]})
            except Exception as exc:  # noqa: BLE001 - stay up, report not-ready
                app.state.pipeline = None
                log.error("index not loaded", extra={"error": str(exc)})
        yield

    app = FastAPI(title="docqa-rag", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.pipeline = pipeline

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
        return response

    def require_api_key(
        request: Request, x_api_key: Annotated[str | None, Header()] = None
    ) -> None:
        expected = request.app.state.settings.api_key
        if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    def get_pipeline(request: Request) -> RagPipeline:
        current = request.app.state.pipeline
        if current is None:
            raise HTTPException(status_code=503, detail="index not loaded")
        return current

    Auth = Annotated[None, Depends(require_api_key)]
    Pipe = Annotated[RagPipeline, Depends(get_pipeline)]

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def readyz(request: Request) -> dict[str, Any]:
        current = request.app.state.pipeline
        if current is None:
            raise HTTPException(status_code=503, detail="index not loaded")
        return {"status": "ready", "chunks": current.manifest["num_chunks"]}

    @app.get("/v1/stats")
    def stats(_: Auth, pipe: Pipe) -> dict[str, Any]:
        m = pipe.manifest
        return {
            "documents": m["num_documents"],
            "chunks": m["num_chunks"],
            "embedder": m["embedder"]["name"],
            "dim": m["dim"],
            "created_at": m["created_at"],
            "generator": pipe.generator.name,
        }

    @app.post("/v1/search", response_model=SearchResponse)
    def search(body: SearchRequest, _: Auth, pipe: Pipe) -> SearchResponse:
        result = pipe.search(body.query, k=body.k, filters=body.filters, mode=body.mode)
        return SearchResponse(
            query=body.query,
            mode=result.mode,
            results=[_source(h) for h in result.hits],
            timings_ms=result.timings_ms,
        )

    @app.post("/v1/ask", response_model=AskResponse)
    def ask(body: AskRequest, _: Auth, pipe: Pipe) -> AskResponse:
        t = time.perf_counter()
        answer, result = pipe.ask(body.question, k=body.k, filters=body.filters)
        timings = {**result.timings_ms, "answer_total": round((time.perf_counter() - t) * 1000, 3)}
        return AskResponse(
            answer=answer, sources=[_source(h) for h in result.hits], timings_ms=timings
        )

    return app
