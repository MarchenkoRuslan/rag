"""FastAPI routes for ingest, query, and document listing."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.deps import EmbeddingsDep, SettingsDep, VectorStoreDep
from app.api.filename import require_safe_filename
from app.models.schemas import (
    DeleteDocumentResponse,
    DocumentItem,
    DocumentListResponse,
    IngestResponse,
    QueryMetricsResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from app.services.generation import generate_answer
from app.services.ingestion import ingest_bytes
from app.services.retrieval import retrieve_chunks, retrieval_relevance_stats
from app.utils.logging import get_logger
from app.utils.metrics import QueryMetrics, SegmentTimer

log = get_logger("api")
router = APIRouter()

_ALLOWED_INGEST_SUFFIXES = frozenset({"txt", "pdf"})


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    settings: SettingsDep,
    store: VectorStoreDep,
    embeddings: EmbeddingsDep,
    file: UploadFile = File(...),
) -> IngestResponse:
    filename = file.filename or "document.txt"
    safe_name = require_safe_filename(filename)
    if "." not in safe_name:
        suffix = ""
    else:
        suffix = safe_name.lower().rsplit(".", 1)[-1]
    if suffix not in _ALLOWED_INGEST_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .pdf files are supported",
        )
    try:
        data = await file.read()
    except Exception as e:
        log.exception("ingest_read_error", error=str(e))
        raise HTTPException(status_code=400, detail="Failed to read upload") from e
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / safe_name).write_bytes(data)
    store.delete_by_filename(safe_name)
    try:
        result = ingest_bytes(safe_name, data, store, embeddings, settings)
    except ValueError as e:
        log.warning("ingest_validation", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.exception("ingest_error", error=str(e))
        raise HTTPException(
            status_code=502,
            detail="Embedding or storage failed",
        ) from e
    return IngestResponse(
        filename=safe_name,
        chunks_added=result.chunks_added,
        characters_extracted=result.characters_extracted,
    )


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    settings: SettingsDep,
    store: VectorStoreDep,
    embeddings: EmbeddingsDep,
) -> QueryResponse:
    metrics = QueryMetrics()
    with SegmentTimer() as t_all:
        with SegmentTimer() as t_ret:
            chunks = retrieve_chunks(
                body.question,
                store,
                embeddings,
                settings,
                top_k=body.top_k,
                relevance_threshold=body.relevance_threshold,
            )
        metrics.retrieval_time_ms = t_ret.elapsed_s * 1000
        metrics.num_chunks_retrieved = len(chunks)
        mean_s, max_s = retrieval_relevance_stats(chunks)
        metrics.mean_relevance_score = mean_s
        metrics.max_relevance_score = max_s
        k = body.top_k or settings.top_k
        metrics.retrieval_accuracy_hint = (
            (mean_s or 0.0) * min(1.0, len(chunks) / max(k, 1))
            if chunks
            else 0.0
        )
        with SegmentTimer() as t_gen:
            try:
                answer = generate_answer(body.question, chunks, settings)
            except Exception as e:
                log.exception("generation_error", error=str(e))
                raise HTTPException(
                    status_code=502,
                    detail="LLM generation failed",
                ) from e
        metrics.generation_time_ms = t_gen.elapsed_s * 1000
    metrics.response_time_ms = t_all.elapsed_s * 1000
    metrics.num_sources_used = len(chunks)
    sources = [
        SourceItem(
            citation_id=c.citation_id,
            filename=c.filename,
            chunk_index=c.chunk_index,
            text=c.text,
            relevance_score=c.relevance_score,
        )
        for c in chunks
    ]
    log.info(
        "query_complete",
        question_len=len(body.question),
        sources=len(sources),
        response_ms=metrics.response_time_ms,
    )
    return QueryResponse(
        answer=answer,
        sources=sources,
        metrics=QueryMetricsResponse(
            **metrics.to_dict(),
        ),
    )


@router.delete("/documents/{filename}", response_model=DeleteDocumentResponse)
async def delete_document(
    filename: str,
    settings: SettingsDep,
    store: VectorStoreDep,
) -> DeleteDocumentResponse:
    safe_name = require_safe_filename(filename)
    removed = store.delete_by_filename(safe_name)
    data_path = settings.data_dir / safe_name
    data_removed = False
    if data_path.is_file():
        try:
            data_path.unlink()
            data_removed = True
        except OSError as e:
            log.warning("data_file_delete_failed", path=str(data_path), error=str(e))
    if removed == 0 and not data_removed:
        raise HTTPException(status_code=404, detail="Document not found")
    log.info("document_deleted", filename=safe_name, chunks_removed=removed)
    return DeleteDocumentResponse(
        filename=safe_name,
        chunks_removed=removed,
        data_file_removed=data_removed,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(store: VectorStoreDep) -> DocumentListResponse:
    rows = store.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentItem(
                filename=r["filename"],
                chunk_count=r["chunk_count"],
                uploaded_at=r["uploaded_at"],
            )
            for r in rows
        ]
    )


@router.get("/health")
async def health(store: VectorStoreDep) -> dict:
    return {"status": "ok", "vectors": store.count()}
