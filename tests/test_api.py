"""API tests with mocked LLM."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes import router
from app.config import EmbeddingProvider, LLMProvider, get_settings
from app.services.vector_store import VectorStore
from tests.conftest import FakeEmbeddings, RagTestSettings


@asynccontextmanager
async def _test_lifespan(app: FastAPI, test_settings, store, emb):
    app.state.settings = test_settings
    app.state.store = store
    app.state.embeddings = emb
    yield


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    get_settings.cache_clear()
    root = tmp_path / "api"
    monkeypatch.setenv("DATA_DIR", str((root / "data").resolve()))
    monkeypatch.setenv("STORAGE_DIR", str((root / "storage").resolve()))
    test_settings = RagTestSettings(
        openai_api_key="sk-test-key-for-pytest",
        embedding_provider=EmbeddingProvider.LOCAL,
        llm_provider=LLMProvider.OPENAI,
        llm_model="gpt-4o-mini",
        chunk_size=200,
        chunk_overlap=20,
        top_k=5,
        relevance_threshold=0.0,
    )
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with _test_lifespan(app, test_settings, store, emb):
            yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(router)
    with TestClient(app) as client:
        yield client, store, emb
    get_settings.cache_clear()


def test_health(api_client):
    client, _, _ = api_client
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_documents_empty(api_client):
    client, _, _ = api_client
    r = client.get("/documents")
    assert r.status_code == 200
    assert r.json()["documents"] == []


@patch("app.api.routes.generate_answer", return_value="Answer [1]")
def test_query_mocked_llm(mock_gen, api_client):
    client, store, emb = api_client
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


def test_delete_document(api_client):
    client, store, emb = api_client
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


def test_reingest_replaces_chunks(api_client):
    client, store, emb = api_client
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
