"""API tests with mocked LLM."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from app.api.auth import api_key_rejection
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
    yield


def _rag_app_bundle(
    tmp_path,
    monkeypatch,
    *,
    api_key: str | None = None,
    max_ingest_bytes: int | None = None,
    health_check_llm: bool = False,
):
    get_settings.cache_clear()
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
    if updates:
        test_settings = test_settings.model_copy(update=updates)
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        async with _bind_test_app_state(application, test_settings, store, emb):
            yield

    fastapi_app = FastAPI(lifespan=lifespan)

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
    app, store, emb = _rag_app_bundle(
        tmp_path, monkeypatch, api_key="pytest-secret"
    )
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
