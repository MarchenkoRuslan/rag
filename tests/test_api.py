"""API tests with mocked LLM."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.requests import Request

from app.api.auth import api_key_rejection
from app.api.rate_limit import limiter
from app.api.routes import router
from app.config import get_settings
from app.services.vector_store import VectorStore
from tests.conftest import FakeEmbeddings, RagTestSettings, default_rag_test_settings


@asynccontextmanager
async def _bind_test_app_state(
    application: FastAPI,
    test_settings: RagTestSettings,
    store: VectorStore,
    emb: FakeEmbeddings,
):
    application.state.settings = test_settings
    application.state.store = store
    application.state.embeddings = emb
    application.state.llm_clients = {}
    yield


def _rag_app_bundle(
    tmp_path,
    monkeypatch,
    *,
    api_key: str | None = None,
    max_ingest_bytes: int | None = None,
    health_check_llm: bool = False,
    api_key_exempt_docs: bool | None = None,
):
    get_settings.cache_clear()
    storage = getattr(limiter, "_storage", None)
    if storage is not None and hasattr(storage, "reset"):
        storage.reset()
    root = tmp_path / "api"
    monkeypatch.setenv("DATA_DIR", str((root / "data").resolve()))
    monkeypatch.setenv("STORAGE_DIR", str((root / "storage").resolve()))
    test_settings = default_rag_test_settings()
    updates: dict = {}
    if api_key is not None:
        updates["api_key"] = api_key
    if max_ingest_bytes is not None:
        updates["max_ingest_bytes"] = max_ingest_bytes
    if health_check_llm:
        updates["health_check_llm"] = True
    if api_key_exempt_docs is not None:
        updates["api_key_exempt_docs"] = api_key_exempt_docs
    if updates:
        test_settings = test_settings.model_copy(update=updates)
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        async with _bind_test_app_state(application, test_settings, store, emb):
            yield

    fastapi_app = FastAPI(lifespan=lifespan)
    fastapi_app.state.limiter = limiter

    @fastapi_app.middleware("http")
    async def _optional_api_key(request: Request, call_next):
        st = getattr(request.app.state, "settings", None)
        if st is not None:
            rej = api_key_rejection(request, st)
            if rej is not None:
                return rej
        return await call_next(request)

    fastapi_app.include_router(router)
    fastapi_app.include_router(router, prefix="/v1")

    @fastapi_app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(
        request: Request,
        exc: RateLimitExceeded,  # pylint: disable=unused-argument
    ) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Rate limit exceeded. Please slow down.",
                "request_id": request.headers.get("x-request-id") or "unknown",
            },
        )

    fastapi_app.add_middleware(SlowAPIMiddleware)
    return fastapi_app, store, emb


@pytest.fixture(name="rag")
def _rag_http_bundle(tmp_path, monkeypatch):
    app, store, emb = _rag_app_bundle(tmp_path, monkeypatch)
    try:
        with TestClient(app) as client:
            yield client, store, emb
    finally:
        get_settings.cache_clear()


@pytest.fixture(name="rag_auth")
def _rag_http_bundle_auth(tmp_path, monkeypatch):
    app, store, emb = _rag_app_bundle(tmp_path, monkeypatch, api_key="pytest-secret")
    try:
        with TestClient(app) as client:
            yield client, store, emb
    finally:
        get_settings.cache_clear()


def test_health(rag):
    client, _, _ = rag
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["vectors"] == 0
    assert data["index_empty"] is True


def test_health_v1_matches(rag):
    client, _, _ = rag
    a = client.get("/health").json()
    b = client.get("/v1/health").json()
    assert a == b


def test_documents_empty(rag):
    client, _, _ = rag
    r = client.get("/documents")
    assert r.status_code == 200
    assert r.json()["documents"] == []


def test_documents_v1(rag):
    client, _, _ = rag
    r = client.get("/v1/documents")
    assert r.status_code == 200
    assert r.json()["documents"] == []


@patch("app.api.routes.generate_answer", return_value="Answer [1]")
def test_query_mocked_llm(mock_gen, rag):
    client, store, emb = rag
    assert store.count() == 0 and emb.dimension == 16
    ingest = client.post(
        "/ingest",
        files={"file": ("t.txt", b"hello world rag system", "text/plain")},
    )
    assert ingest.status_code == 200, ingest.text
    r = client.post("/query", json={"question": "what is rag?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "answer" in body
    assert body["answer"] == "Answer [1]"
    assert "metrics" in body
    assert body["metrics"]["num_sources_used"] >= 0
    assert body.get("index_empty") is False
    mock_gen.assert_called_once()


def test_query_index_empty_flag(rag):
    client, _, _ = rag
    r = client.post("/query", json={"question": "anything"})
    assert r.status_code == 200
    body = r.json()
    assert body.get("index_empty") is True
    assert "don't have enough information" in body.get("answer", "").lower()


def test_delete_document(rag):
    client, store, _emb = rag
    ingest = client.post(
        "/ingest",
        files={"file": ("d.txt", b"chunk one\n\nchunk two content", "text/plain")},
    )
    assert ingest.status_code == 200
    assert store.count() > 0
    r = client.delete("/documents/d.txt")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["chunks_removed"] > 0
    assert client.get("/documents").json()["documents"] == []


def test_reingest_replaces_chunks(rag):
    client, store, _emb = rag
    r1 = client.post(
        "/ingest",
        files={"file": ("same.txt", (b"v1 " * 300), "text/plain")},
    )
    assert r1.status_code == 200
    chunks1 = r1.json()["chunks_added"]
    assert store.count() == chunks1
    r2 = client.post(
        "/ingest",
        files={"file": ("same.txt", (b"v2 " * 300), "text/plain")},
    )
    assert r2.status_code == 200
    chunks2 = r2.json()["chunks_added"]
    assert store.count() == chunks2


def test_reingest_failure_keeps_existing_chunks(rag):
    client, store, _emb = rag
    first = client.post(
        "/ingest",
        files={"file": ("same.txt", (b"v1 " * 200), "text/plain")},
    )
    assert first.status_code == 200
    before = store.count()

    with patch("app.api.routes.ingest_bytes", side_effect=RuntimeError("embed failure")):
        failed = client.post(
            "/ingest",
            files={"file": ("same.txt", (b"v2 " * 200), "text/plain")},
        )
    assert failed.status_code == 502
    assert store.count() == before
    docs = client.get("/documents").json()["documents"]
    assert any(d["filename"] == "same.txt" for d in docs)


def test_ingest_rejects_oversized_payload(tmp_path, monkeypatch):
    app, _, _ = _rag_app_bundle(tmp_path, monkeypatch, max_ingest_bytes=80)
    try:
        with TestClient(app) as client:
            big = b"x" * 200
            r = client.post(
                "/ingest",
                files={"file": ("big.txt", big, "text/plain")},
            )
            assert r.status_code == 413
            assert "too large" in r.json()["detail"].lower()
    finally:
        get_settings.cache_clear()


@patch("app.api.routes.probe_llm", return_value=(True, None))
def test_health_with_llm_probe(mock_probe, tmp_path, monkeypatch):
    app, _, _ = _rag_app_bundle(tmp_path, monkeypatch, health_check_llm=True)
    try:
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            data = r.json()
            assert data["llm_ok"] is True
            mock_probe.assert_called_once()
    finally:
        get_settings.cache_clear()


def test_ingest_requires_api_key_when_configured(rag_auth):
    client, _, _ = rag_auth
    r = client.post(
        "/ingest",
        files={"file": ("a.txt", b"hi", "text/plain")},
    )
    assert r.status_code == 401
    ok = client.post(
        "/ingest",
        files={"file": ("a.txt", b"hi", "text/plain")},
        headers={"X-API-Key": "pytest-secret"},
    )
    assert ok.status_code == 200, ok.text


def test_health_exempt_from_api_key(rag_auth):
    client, _, _ = rag_auth
    r = client.get("/health")
    assert r.status_code == 200


def test_documents_require_api_key_when_configured(rag_auth):
    client, _, _ = rag_auth
    r = client.get("/documents")
    assert r.status_code == 401


def test_wrong_length_api_key_returns_401_not_500(rag_auth):
    """Mismatched key length must not trip hmac.compare_digest ValueError."""
    client, _, _ = rag_auth
    r = client.post(
        "/ingest",
        files={"file": ("a.txt", b"hi", "text/plain")},
        headers={"Authorization": "Bearer x"},
    )
    assert r.status_code == 401


def test_ingest_rate_limit_returns_429(tmp_path, monkeypatch):
    app, _, _ = _rag_app_bundle(tmp_path, monkeypatch)
    try:
        with TestClient(app) as client:
            for i in range(10):
                r = client.post(
                    "/ingest",
                    files={"file": (f"rl{i}.txt", b"paragraph one\n\n" * 5, "text/plain")},
                )
                assert r.status_code == 200, r.text
            blocked = client.post(
                "/ingest",
                files={"file": ("rl_overflow.txt", b"paragraph one\n\n" * 5, "text/plain")},
            )
            assert blocked.status_code == 429
    finally:
        get_settings.cache_clear()


# --- ingest error branches ---


def test_ingest_unsupported_extension(rag):
    client, _, _ = rag
    r = client.post(
        "/ingest",
        files={"file": ("data.json", b'{"a":1}', "application/json")},
    )
    assert r.status_code == 400
    assert "only .txt and .pdf" in r.json()["detail"].lower()


def test_ingest_empty_file(rag):
    client, _, _ = rag
    r = client.post(
        "/ingest",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_ingest_path_traversal(rag):
    client, _, _ = rag
    r = client.post(
        "/ingest",
        files={"file": ("../etc.txt", b"payload", "text/plain")},
    )
    assert r.status_code == 400
    assert "invalid filename" in r.json()["detail"].lower()


# --- DELETE 404 ---


def test_delete_nonexistent_document(rag):
    client, _, _ = rag
    r = client.delete("/documents/nonexistent.txt")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


# --- query validation (Pydantic 422) ---


def test_query_empty_question(rag):
    client, _, _ = rag
    r = client.post("/query", json={"question": ""})
    assert r.status_code == 422


def test_query_top_k_zero(rag):
    client, _, _ = rag
    r = client.post("/query", json={"question": "ok", "top_k": 0})
    assert r.status_code == 422


def test_query_relevance_threshold_too_high(rag):
    client, _, _ = rag
    r = client.post(
        "/query",
        json={"question": "ok", "relevance_threshold": 1.5},
    )
    assert r.status_code == 422


@patch("app.api.routes.generate_answer_stream", side_effect=RuntimeError("llm down"))
def test_query_stream_emits_error_event(_mock_stream, rag):
    client, _, _ = rag
    ingest = client.post(
        "/ingest",
        files={"file": ("s.txt", b"stream test content", "text/plain")},
    )
    assert ingest.status_code == 200
    r = client.post("/query/stream", json={"question": "what?"})
    assert r.status_code == 200
    assert '"type": "sources"' in r.text
    assert '"type": "error"' in r.text
    assert '"type": "done"' in r.text


@patch("app.api.routes.retrieve_chunks", side_effect=RuntimeError("retrieval down"))
def test_query_stream_retrieval_error_maps_to_502(_mock_ret, rag):
    client, _, _ = rag
    r = client.post("/query/stream", json={"question": "what?"})
    assert r.status_code == 502
    assert "retrieval failed" in r.json()["detail"].lower()


# --- auth: Bearer, wrong key, /v1/health exempt ---


def test_auth_bearer_header(rag_auth):
    client, _, _ = rag_auth
    r = client.post(
        "/ingest",
        files={"file": ("b.txt", b"hi", "text/plain")},
        headers={"Authorization": "Bearer pytest-secret"},
    )
    assert r.status_code == 200, r.text


def test_auth_wrong_key_rejected(rag_auth):
    client, _, _ = rag_auth
    r = client.post(
        "/ingest",
        files={"file": ("b.txt", b"hi", "text/plain")},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r.status_code == 401


def test_v1_health_exempt_from_api_key(rag_auth):
    client, _, _ = rag_auth
    r = client.get("/v1/health")
    assert r.status_code == 200


def test_docs_protected_when_api_key_docs_exemption_disabled(tmp_path, monkeypatch):
    app, _, _ = _rag_app_bundle(
        tmp_path,
        monkeypatch,
        api_key="pytest-secret",
        api_key_exempt_docs=False,
    )
    try:
        with TestClient(app) as client:
            denied = client.get("/openapi.json")
            assert denied.status_code == 401
            allowed = client.get("/openapi.json", headers={"X-API-Key": "pytest-secret"})
            assert allowed.status_code == 200
    finally:
        get_settings.cache_clear()


# --- health: LLM probe failure ---


@patch("app.api.routes.probe_llm", return_value=(False, "timeout"))
def test_health_llm_probe_failure(mock_probe, tmp_path, monkeypatch):
    app, _, _ = _rag_app_bundle(tmp_path, monkeypatch, health_check_llm=True)
    try:
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            data = r.json()
            assert data["llm_ok"] is False
            assert data["llm_error"] == "timeout"
            mock_probe.assert_called_once()
    finally:
        get_settings.cache_clear()
