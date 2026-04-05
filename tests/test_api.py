"""API tests with mocked LLM."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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


@pytest.fixture(name="rag")
def _rag_http_bundle(tmp_path, monkeypatch):
    get_settings.cache_clear()
    root = tmp_path / "api"
    monkeypatch.setenv("DATA_DIR", str((root / "data").resolve()))
    monkeypatch.setenv("STORAGE_DIR", str((root / "storage").resolve()))
    test_settings = default_rag_test_settings()
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        async with _bind_test_app_state(application, test_settings, store, emb):
            yield

    fastapi_app = FastAPI(lifespan=lifespan)
    fastapi_app.include_router(router)
    with TestClient(fastapi_app) as client:
        yield client, store, emb
    get_settings.cache_clear()


def test_health(rag):
    client, _, _ = rag
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_documents_empty(rag):
    client, _, _ = rag
    r = client.get("/documents")
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
    mock_gen.assert_called_once()


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
