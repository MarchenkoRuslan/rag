"""Top-k retrieval with relevance filtering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import Settings
from app.services.embeddings import EmbeddingProviderBase
from app.services.vector_store import VectorStore
from app.utils.logging import get_logger

log = get_logger("retrieval")


@dataclass
class RetrievedChunk:
    citation_id: int
    faiss_id: int
    filename: str
    chunk_index: int
    text: str
    relevance_score: float


def retrieve_chunks(
    question: str,
    store: VectorStore,
    embeddings: EmbeddingProviderBase,
    settings: Settings,
    top_k: int | None = None,
    relevance_threshold: float | None = None,
) -> list[RetrievedChunk]:
    k = top_k if top_k is not None else settings.top_k
    thr = (
        relevance_threshold
        if relevance_threshold is not None
        else settings.relevance_threshold
    )
    if store.count() == 0:
        log.warning("retrieval_empty_index")
        return []
    qvec = embeddings.embed_texts([question])
    if qvec.shape[0] == 0:
        return []
    # embed_texts returns normalized; search normalizes again — OK
    ids, scores = store.search(qvec[0], k)
    records = store.get_by_faiss_ids(ids)
    results: list[RetrievedChunk] = []
    for fid, score in zip(ids, scores):
        if score < thr:
            continue
        rec = records.get(fid)
        if rec is None:
            log.warning("missing_chunk_metadata", faiss_id=fid)
            continue
        results.append(
            RetrievedChunk(
                citation_id=len(results) + 1,
                faiss_id=fid,
                filename=rec.filename,
                chunk_index=rec.chunk_index,
                text=rec.text,
                relevance_score=float(score),
            )
        )
    log.info(
        "retrieval_done",
        candidates=len(ids),
        after_threshold=len(results),
        top_score=float(max(scores)) if scores else None,
    )
    return results


def retrieval_relevance_stats(
    chunks: list[RetrievedChunk],
) -> tuple[float | None, float | None]:
    """Mean and max similarity scores for simple 'accuracy' hint metrics."""
    if not chunks:
        return None, None
    scores = [c.relevance_score for c in chunks]
    return float(np.mean(scores)), float(max(scores))
