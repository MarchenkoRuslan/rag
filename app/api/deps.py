"""FastAPI dependencies: resolve services from app.state (lifespan)."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request

from app.config import Settings
from app.services.embeddings import EmbeddingProviderBase
from app.services.vector_store import VectorStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.store


def get_embeddings(request: Request) -> EmbeddingProviderBase:
    return request.app.state.embeddings


def get_llm_clients(request: Request) -> dict[str, Any]:
    return getattr(request.app.state, "llm_clients", {})


SettingsDep = Annotated[Settings, Depends(get_settings)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
EmbeddingsDep = Annotated[EmbeddingProviderBase, Depends(get_embeddings)]
LLMClientsDep = Annotated[dict[str, Any], Depends(get_llm_clients)]
