"""Embedding providers: OpenAI and local sentence-transformers."""

from __future__ import annotations

import abc
from collections.abc import Sequence

import numpy as np
from openai import OpenAI

from app.config import EmbeddingProvider, Settings


def _normalize_batch(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows for cosine similarity via inner product."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (vectors / norms).astype(np.float32)


class EmbeddingProviderBase(abc.ABC):
    """Embeds text to L2-normalized float32 vectors."""

    @property
    @abc.abstractmethod
    def dimension(self) -> int: ...

    @abc.abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return shape (n, dim) float32, row-normalized."""


class OpenAIEmbeddingProvider(EmbeddingProviderBase):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=3,
        )
        self._model = settings.openai_embedding_model
        self._dim = settings.embedding_dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        # API returns in same order as input
        data = sorted(resp.data, key=lambda x: x.index)
        arr = np.array([d.embedding for d in data], dtype=np.float32)
        return _normalize_batch(arr)


class LocalEmbeddingProvider(EmbeddingProviderBase):
    def __init__(self, settings: Settings) -> None:
        # Heavy optional dependency: import only when local embeddings are used.
        # pylint: disable-next=import-outside-toplevel
        from sentence_transformers import SentenceTransformer

        self._settings = settings
        self._model = SentenceTransformer(settings.local_embedding_model)
        raw_dim = self._model.get_sentence_embedding_dimension()
        if raw_dim is None:
            raise RuntimeError("Embedding model did not report output dimension")
        self._dim = int(raw_dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        emb = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return _normalize_batch(arr)


def build_embedding_provider(settings: Settings) -> EmbeddingProviderBase:
    if settings.embedding_provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddingProvider(settings)
    return LocalEmbeddingProvider(settings)
