"""Tests for vector store and retrieval."""

from __future__ import annotations

import numpy as np

from app.services.ingestion import ingest_bytes
from app.services.retrieval import retrieve_chunks
from app.services.vector_store import VectorStore
from tests.conftest import FakeEmbeddings


def test_vector_store_add_and_search(test_settings):
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)
    texts = ["alpha beta gamma", "delta epsilon zeta", "foo bar baz"]
    vecs = emb.embed_texts(texts)
    store.add_chunks(vecs, "doc.txt", texts)
    q = emb.embed_texts(["alpha gamma"])[0]
    ids, scores = store.search(q, top_k=2)
    assert len(ids) >= 1
    assert scores[0] > 0.5


def test_retrieve_respects_threshold(test_settings):
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)
    ingest_bytes(
        "notes.txt",
        b"Python is a programming language.\n\nFAISS is a vector index.",
        store,
        emb,
        test_settings,
    )
    chunks = retrieve_chunks(
        "What is Python?",
        store,
        emb,
        test_settings,
        top_k=5,
        relevance_threshold=0.0,
    )
    assert len(chunks) >= 1
    mean = float(np.mean([c.relevance_score for c in chunks]))
    assert mean > 0
