"""FastAPI routes for ingest, query, and document listing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.api.deps import EmbeddingsDep, LLMClientsDep, SettingsDep, VectorStoreDep
from app.api.filename import require_safe_filename
from app.api.rate_limit import limiter
from app.models.schemas import (
    DeleteDocumentResponse,
    DocumentItem,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QueryMetricsResponse,
    QueryRequest,
    QueryResponse,
    SourceItem,
)
from app.services.generation import generate_answer, generate_answer_stream
from app.services.ingestion import ingest_bytes
from app.services.llm_health import probe_llm
from app.services.retrieval import retrieve_chunks, retrieval_relevance_stats
from app.utils.logging import get_logger
from app.utils.metrics import QueryMetrics, SegmentTimer

log = get_logger("api")
router = APIRouter()

_ALLOWED_INGEST_SUFFIXES = frozenset({"txt", "pdf"})


@router.post("/ingest", response_model=IngestResponse)
@limiter.limit("10/minute")
def ingest(
    request: Request,  # pylint: disable=unused-argument
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
        data = file.file.read()
    except Exception as e:
        log.exception("ingest_read_error", error=str(e))
        raise HTTPException(status_code=400, detail="Failed to read upload") from e
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > settings.max_ingest_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(data)} bytes "
                f"(max {settings.max_ingest_bytes})"
            ),
        )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    data_path = settings.data_dir / safe_name
    tmp_path: Path | None = None
    with tempfile.NamedTemporaryFile(
        prefix="upload-",
        suffix=f".{suffix}" if suffix else ".tmp",
        dir=str(settings.data_dir),
        delete=False,
    ) as tmp:
        tmp.write(data)
        tmp.flush()
        tmp_path = Path(tmp.name)
    try:
        result = ingest_bytes(safe_name, data, store, embeddings, settings)
        tmp_path.replace(data_path)
    except ValueError as e:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        log.warning("ingest_validation", error=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
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
@limiter.limit("30/minute")
def query(
    request: Request,  # pylint: disable=unused-argument
    body: QueryRequest,
    settings: SettingsDep,
    store: VectorStoreDep,
    embeddings: EmbeddingsDep,
    llm_clients: LLMClientsDep,
) -> QueryResponse:
    index_empty = store.count() == 0
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
                answer = generate_answer(
                    body.question,
                    chunks,
                    settings,
                    openai_client=llm_clients.get("openai"),
                    ollama_client=llm_clients.get("ollama"),
                )
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
        index_empty=index_empty,
    )


@router.post("/query/stream")
@limiter.limit("30/minute")
def query_stream(
    request: Request,  # pylint: disable=unused-argument
    body: QueryRequest,
    settings: SettingsDep,
    store: VectorStoreDep,
    embeddings: EmbeddingsDep,
    llm_clients: LLMClientsDep,
) -> StreamingResponse:
    """SSE endpoint: streams answer tokens as `data: {token}` events."""
    try:
        chunks = retrieve_chunks(
            body.question,
            store,
            embeddings,
            settings,
            top_k=body.top_k,
            relevance_threshold=body.relevance_threshold,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        log.exception("retrieval_stream_error", error=str(e))
        raise HTTPException(status_code=502, detail="Retrieval failed") from e
    sources = [
        {
            "citation_id": c.citation_id,
            "filename": c.filename,
            "chunk_index": c.chunk_index,
            "text": c.text,
            "relevance_score": c.relevance_score,
        }
        for c in chunks
    ]

    def _event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        try:
            token_gen = generate_answer_stream(
                body.question,
                chunks,
                settings,
                openai_client=llm_clients.get("openai"),
                ollama_client=llm_clients.get("ollama"),
            )
            for token in token_gen:
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.exception("generation_stream_error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'detail': 'LLM generation failed'})}\n\n"
        finally:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/documents/{filename}", response_model=DeleteDocumentResponse)
def delete_document(
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
def list_documents(
    store: VectorStoreDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
) -> DocumentListResponse:
    rows = store.list_documents(offset=offset, limit=limit)
    total = store.document_count()
    return DocumentListResponse(
        documents=[
            DocumentItem(
                filename=r["filename"],
                chunk_count=r["chunk_count"],
                uploaded_at=r["uploaded_at"],
            )
            for r in rows
        ],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/health", response_model=HealthResponse)
def health(store: VectorStoreDep, settings: SettingsDep) -> HealthResponse:
    n = store.count()
    resp = HealthResponse(status="ok", vectors=n, index_empty=n == 0)
    if settings.health_check_llm:
        ok, err = probe_llm(settings)
        resp.llm_ok = ok
        if not ok:
            resp.llm_error = err or "unknown"
    return resp
