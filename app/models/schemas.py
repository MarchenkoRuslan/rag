"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    relevance_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class SourceItem(BaseModel):
    citation_id: int
    filename: str
    chunk_index: int
    text: str
    relevance_score: float


class QueryMetricsResponse(BaseModel):
    response_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    num_sources_used: int
    num_chunks_retrieved: int
    mean_relevance_score: float | None = None
    max_relevance_score: float | None = None
    retrieval_accuracy_hint: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    metrics: QueryMetricsResponse
    index_empty: bool = False


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    characters_extracted: int
    message: str = "ok"


class DocumentItem(BaseModel):
    filename: str
    chunk_count: int
    uploaded_at: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentItem]
    total: int = 0
    offset: int = 0
    limit: int = 50


class DeleteDocumentResponse(BaseModel):
    filename: str
    chunks_removed: int
    data_file_removed: bool


class HealthResponse(BaseModel):
    status: str
    vectors: int
    index_empty: bool
    llm_ok: bool | None = None
    llm_error: str | None = None


class ErrorDetail(BaseModel):
    detail: str
    request_id: str | None = None
