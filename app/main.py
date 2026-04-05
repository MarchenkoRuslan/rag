"""FastAPI application entrypoint."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog.contextvars
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.services.embeddings import build_embedding_provider
from app.services.vector_store import VectorStore
from app.utils.logging import configure_logging, get_logger

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    embeddings = build_embedding_provider(settings)
    store = VectorStore(
        settings.storage_dir.resolve(),
        embeddings.dimension,
        settings,
    )
    app.state.settings = settings
    app.state.embeddings = embeddings
    app.state.store = store
    log.info(
        "app_startup",
        embedding_dim=embeddings.dimension,
        embedding_provider=settings.embedding_provider.value,
        llm_provider=settings.llm_provider.value,
    )
    yield
    store.close()
    log.info("app_shutdown")


app = FastAPI(
    title="RAG Knowledge System",
    description="Production-oriented RAG with FAISS, citations, and configurable LLM/embeddings.",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_settings.cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_and_access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=rid)
    t0 = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response


app.include_router(router)


@app.get("/")
async def root() -> dict:
    return {
        "service": "rag-knowledge-system",
        "docs": "/docs",
        "health": "/health",
    }
