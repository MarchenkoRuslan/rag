"""FastAPI application entrypoint."""

# pylint: disable=no-member
# (Pydantic Settings attributes are real at runtime.)

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog.contextvars
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.auth import api_key_rejection
from app.api.routes import router
from app.config import LLMProvider, get_settings
from app.services.embeddings import build_embedding_provider
from app.services.generation import build_ollama_client, build_openai_client
from app.services.vector_store import VectorStore
from app.utils.logging import configure_logging, get_logger

log = get_logger("main")

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(application: FastAPI):
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
    llm_clients: dict = {}
    if settings.llm_provider == LLMProvider.OPENAI:
        llm_clients["openai"] = build_openai_client(settings)
    else:
        llm_clients["ollama"] = build_ollama_client(settings)

    application.state.settings = settings
    application.state.embeddings = embeddings
    application.state.store = store
    application.state.llm_clients = llm_clients

    origins = settings.cors_origins_list()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    log.info(
        "app_startup",
        embedding_dim=embeddings.dimension,
        embedding_provider=settings.embedding_provider.value,
        llm_provider=settings.llm_provider.value,
    )
    yield
    ollama_c = llm_clients.get("ollama")
    if ollama_c is not None:
        ollama_c.close()
    store.close()
    log.info("app_shutdown")


app = FastAPI(
    title="RAG Knowledge System",
    description="Production-oriented RAG with FAISS, citations, and configurable LLM/embeddings.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter


def _request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or "unknown"


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    rid = _request_id(request)
    log.exception("unhandled_error", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": rid},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    rid = _request_id(request)
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "request_id": rid,
        },
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(
    request: Request, exc: RateLimitExceeded  # pylint: disable=unused-argument
) -> JSONResponse:
    rid = _request_id(request)
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please slow down.",
            "request_id": rid,
        },
    )


@app.middleware("http")
async def request_context_and_access_log(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=rid)
    settings = getattr(request.app.state, "settings", None) or get_settings()
    reject = api_key_rejection(request, settings)
    if reject is not None:
        reject.headers["X-Request-ID"] = rid
        return reject
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
app.include_router(router, prefix="/v1")


@app.get("/")
async def root() -> dict:
    return {
        "service": "rag-knowledge-system",
        "docs": "/docs",
        "health": "/health",
        "api_v1_prefix": "/v1",
    }
