"""Tests for VectorStore: thread safety, migration, dimension mismatch."""

from __future__ import annotations

import threading

import faiss
import numpy as np
import pytest

from app.services.vector_store import VectorStore
from tests.conftest import FakeEmbeddings


def test_concurrent_add_and_search(test_settings):
    """Multiple threads adding and searching should not corrupt state."""
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)

    errors: list[Exception] = []

    def writer(idx: int) -> None:
        try:
            texts = [f"document {idx} chunk {i}" for i in range(5)]
            vecs = emb.embed_texts(texts)
            store.add_chunks(vecs, f"doc_{idx}.txt", texts)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    def reader() -> None:
        try:
            q = emb.embed_texts(["document"])[0]
            store.search(q, top_k=3)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    threads = []
    for i in range(4):
        threads.append(threading.Thread(target=writer, args=(i,)))
        threads.append(threading.Thread(target=reader))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Thread errors: {errors}"
    assert store.count() == 20


def test_concurrent_delete_and_search(test_settings):
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)

    for i in range(4):
        texts = [f"file {i} chunk {j}" for j in range(3)]
        vecs = emb.embed_texts(texts)
        store.add_chunks(vecs, f"f{i}.txt", texts)

    errors: list[Exception] = []

    def deleter(filename: str) -> None:
        try:
            store.delete_by_filename(filename)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    def reader() -> None:
        try:
            q = emb.embed_texts(["file"])[0]
            store.search(q, top_k=5)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            errors.append(exc)

    threads = [
        threading.Thread(target=deleter, args=("f0.txt",)),
        threading.Thread(target=deleter, args=("f1.txt",)),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    assert store.count() == 6


def test_dimension_mismatch_raises(test_settings):
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)
    texts = ["hello"]
    vecs = emb.embed_texts(texts)
    store.add_chunks(vecs, "a.txt", texts)
    store.close()

    updated = test_settings.model_copy(update={"embedding_dimension_override": 32})
    with pytest.raises(RuntimeError, match="dimension"):
        VectorStore(updated.storage_dir, 32, updated)


def test_migrate_flat_ip_to_idmap(test_settings):
    """Write a plain IndexFlatIP, load it, and verify migration to IDMap2."""
    dim = 16
    flat = faiss.IndexFlatIP(dim)
    vecs = np.random.default_rng(42).standard_normal((3, dim)).astype(np.float32)
    faiss.normalize_L2(vecs)
    flat.add(vecs)  # pylint: disable=no-value-for-parameter

    storage = test_settings.storage_dir
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "index.faiss").write_bytes(faiss.serialize_index(flat))

    store = VectorStore(storage, dim, test_settings)
    assert store.count() == 3


def test_list_documents_pagination(test_settings):
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)

    for i in range(5):
        texts = [f"chunk for doc {i}"]
        vecs = emb.embed_texts(texts)
        store.add_chunks(vecs, f"doc_{i:02d}.txt", texts)

    assert store.document_count() == 5
    page1 = store.list_documents(offset=0, limit=2)
    assert len(page1) == 2
    page2 = store.list_documents(offset=2, limit=2)
    assert len(page2) == 2
    page3 = store.list_documents(offset=4, limit=2)
    assert len(page3) == 1


def test_persist_and_reload(test_settings):
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)
    texts = ["persist test"]
    vecs = emb.embed_texts(texts)
    store.add_chunks(vecs, "p.txt", texts)
    store.close()

    store2 = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)
    assert store2.count() == 1
    docs = store2.list_documents()
    assert docs[0]["filename"] == "p.txt"
    store2.close()
