"""Tests for text extraction and chunking."""

from __future__ import annotations

import pytest

from app.services.ingestion import chunk_text, extract_text_from_bytes


def test_extract_txt():
    text = extract_text_from_bytes("a.txt", b"Hello \xd0\xbc\xd0\xb8\xd1\x80")
    assert "Hello" in text


def test_extract_pdf_invalid():
    with pytest.raises(Exception):
        extract_text_from_bytes("x.pdf", b"not a pdf")


def test_chunk_text_paragraphs(test_settings):
    raw = "A" * 50 + "\n\n" + "B" * 50 + "\n\n" + "C" * 300
    chunks = chunk_text(raw, test_settings)
    assert len(chunks) >= 1
    assert all(len(c) <= test_settings.chunk_size + 20 for c in chunks)


def test_chunk_text_empty(test_settings):
    assert chunk_text("   ", test_settings) == []
