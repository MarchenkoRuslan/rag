"""Document ingestion: extract text, chunk, embed, persist to vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from app.config import Settings
from app.services.embeddings import EmbeddingProviderBase
from app.services.vector_store import VectorStore
from app.utils.logging import get_logger

log = get_logger("ingestion")


@dataclass
class IngestResult:
    filename: str
    chunks_added: int
    characters_extracted: int


def extract_text_from_bytes(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            parts: list[str] = []
            for page in doc:
                parts.append(page.get_text())
        finally:
            doc.close()
        return "\n\n".join(parts)
    raise ValueError(f"Unsupported file type: {suffix}. Allowed: .txt, .pdf")


def chunk_text(text: str, settings: Settings) -> list[str]:
    """Paragraph-aware chunking with sliding windows for long blocks."""
    text = text.strip()
    if not text:
        return []
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        sep = "\n\n" if buf else ""
        candidate = f"{buf}{sep}{p}" if buf else p
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.extend(_window_split(buf, chunk_size, overlap))
            if len(p) <= chunk_size:
                buf = p
            else:
                chunks.extend(_window_split(p, chunk_size, overlap))
                buf = ""
    if buf:
        chunks.extend(_window_split(buf, chunk_size, overlap))
    return [c for c in chunks if c]


def _window_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(text)
    step = max(1, chunk_size - overlap)
    while i < n:
        j = min(i + chunk_size, n)
        chunk = text[i:j]
        if j < n:
            nl = chunk.rfind("\n")
            if nl > chunk_size // 3:
                chunk = chunk[:nl]
                j = i + nl
        chunk = chunk.strip()
        if chunk:
            out.append(chunk)
        if j >= n:
            break
        next_i = j - overlap
        i = next_i if next_i > i else i + step
    return out


def ingest_bytes(
    filename: str,
    data: bytes,
    store: VectorStore,
    embeddings: EmbeddingProviderBase,
    settings: Settings,
) -> IngestResult:
    raw = extract_text_from_bytes(filename, data)
    if not raw.strip():
        raise ValueError("Document is empty or contains no extractable text")
    chunks = chunk_text(raw, settings)
    if not chunks:
        raise ValueError("No chunks produced from document")
    vectors = embeddings.embed_texts(chunks)
    added = store.add_chunks(vectors, filename, chunks)
    log.info(
        "ingest_complete",
        filename=filename,
        chunks=added,
        chars=len(raw),
    )
    return IngestResult(
        filename=filename,
        chunks_added=added,
        characters_extracted=len(raw),
    )
