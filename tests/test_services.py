"""Tests for service-layer edge cases: PDF ingest, Ollama, generation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import fitz
import pytest

from app.services.generation import (
    build_user_prompt,
    generate_answer,
    generate_answer_stream,
)
from app.services.ingestion import extract_text_from_bytes, ingest_bytes
from app.services.retrieval import RetrievedChunk, retrieval_relevance_stats
from app.services.vector_store import VectorStore
from tests.conftest import FakeEmbeddings, default_rag_test_settings


def _make_pdf(text: str) -> bytes:
    """Create a minimal valid PDF with given text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_pdf_valid():
    data = _make_pdf("Hello from PDF")
    text = extract_text_from_bytes("sample.pdf", data)
    assert "Hello from PDF" in text


def test_ingest_pdf_bytes(test_settings):
    emb = FakeEmbeddings(16)
    store = VectorStore(test_settings.storage_dir, emb.dimension, test_settings)
    data = _make_pdf("PDF chunk one.\n\nPDF chunk two with more text for testing.")
    result = ingest_bytes("test.pdf", data, store, emb, test_settings)
    assert result.chunks_added >= 1
    assert result.characters_extracted > 0


def test_extract_unsupported_extension():
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text_from_bytes("data.csv", b"a,b,c")


def test_generate_answer_empty_chunks():
    settings = default_rag_test_settings()
    answer = generate_answer("anything", [], settings)
    assert "don't have enough information" in answer.lower()


@patch("app.services.generation._ollama_chat")
def test_generate_answer_ollama(mock_chat):
    mock_chat.return_value = "Ollama says: answer [1]"
    settings = default_rag_test_settings().model_copy(update={"llm_provider": "ollama"})
    chunk = RetrievedChunk(
        citation_id=1,
        faiss_id=0,
        filename="doc.txt",
        chunk_index=0,
        text="context text",
        relevance_score=0.9,
    )
    result = generate_answer("question?", [chunk], settings)
    assert result == "Ollama says: answer [1]"
    mock_chat.assert_called_once()


@patch("app.services.generation.OpenAI")
def test_generate_answer_openai_client_reuse(mock_openai_cls):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "OpenAI answer"
    mock_client.chat.completions.create.return_value = mock_resp

    settings = default_rag_test_settings()
    chunk = RetrievedChunk(
        citation_id=1,
        faiss_id=0,
        filename="doc.txt",
        chunk_index=0,
        text="text",
        relevance_score=0.8,
    )
    result = generate_answer("q?", [chunk], settings, openai_client=mock_client)
    assert result == "OpenAI answer"
    mock_openai_cls.assert_not_called()


def test_build_user_prompt():
    chunks = [
        RetrievedChunk(
            citation_id=1,
            faiss_id=0,
            filename="f.txt",
            chunk_index=0,
            text="first",
            relevance_score=0.9,
        ),
        RetrievedChunk(
            citation_id=2,
            faiss_id=1,
            filename="g.txt",
            chunk_index=1,
            text="second",
            relevance_score=0.7,
        ),
    ]
    prompt = build_user_prompt("test?", chunks)
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "test?" in prompt


@patch("app.services.generation.build_openai_client")
def test_generate_answer_stream_closes_owned_openai_client(mock_builder):
    settings = default_rag_test_settings()
    chunk = RetrievedChunk(
        citation_id=1,
        faiss_id=0,
        filename="doc.txt",
        chunk_index=0,
        text="context",
        relevance_score=0.9,
    )

    class _Delta:
        def __init__(self, content: str):
            self.content = content

    class _Choice:
        def __init__(self, content: str):
            self.delta = _Delta(content)

    class _Chunk:
        def __init__(self, content: str):
            self.choices = [_Choice(content)]

    client = MagicMock()
    client.chat.completions.create.return_value = [_Chunk("A"), _Chunk("B")]
    mock_builder.return_value = client

    out = list(generate_answer_stream("q", [chunk], settings))
    assert out == ["A", "B"]
    client.close.assert_called_once()


def test_retrieval_relevance_stats_empty():
    mean, mx = retrieval_relevance_stats([])
    assert mean is None
    assert mx is None


def test_retrieval_relevance_stats_values():
    chunks = [
        RetrievedChunk(
            citation_id=1,
            faiss_id=0,
            filename="a.txt",
            chunk_index=0,
            text="t",
            relevance_score=0.6,
        ),
        RetrievedChunk(
            citation_id=2,
            faiss_id=1,
            filename="a.txt",
            chunk_index=1,
            text="t",
            relevance_score=0.8,
        ),
    ]
    mean, mx = retrieval_relevance_stats(chunks)
    assert mean == pytest.approx(0.7)
    assert mx == pytest.approx(0.8)
