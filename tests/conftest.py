"""Shared fixtures for tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from pydantic_settings import SettingsConfigDict

from app.config import EmbeddingProvider, LLMProvider, Settings, get_settings
from app.services.embeddings import EmbeddingProviderBase


class RagTestSettings(Settings):
    """No `.env` file; process env can still apply (see autouse monkeypatch)."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore")


@pytest.fixture(autouse=True)
def _isolate_env_for_tests(monkeypatch):
    """Pin provider/key; drop path overrides so explicit RagTestSettings paths win."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-pytest")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    for key in (
        "DATA_DIR",
        "STORAGE_DIR",
        "CORS_ALLOW_ORIGINS",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
        "RELEVANCE_THRESHOLD",
    ):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def test_settings(tmp_path, monkeypatch) -> RagTestSettings:
    monkeypatch.setenv("DATA_DIR", str((tmp_path / "data").resolve()))
    monkeypatch.setenv("STORAGE_DIR", str((tmp_path / "storage").resolve()))
    return RagTestSettings(
        openai_api_key="sk-test-key-for-pytest",
        embedding_provider=EmbeddingProvider.LOCAL,
        llm_provider=LLMProvider.OPENAI,
        llm_model="gpt-4o-mini",
        chunk_size=200,
        chunk_overlap=20,
        top_k=5,
        relevance_threshold=0.0,
    )


class FakeEmbeddings(EmbeddingProviderBase):
    """Deterministic pseudo-embeddings for tests (no network)."""

    def __init__(self, dimension: int = 16) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        out = []
        for t in texts:
            v = np.zeros(self._dim, dtype=np.float32)
            for i, b in enumerate(t.encode("utf-8", errors="ignore")):
                v[i % self._dim] += b / 255.0
            n = float(np.linalg.norm(v))
            if n < 1e-9:
                v[0] = 1.0
            else:
                v = v / n
            out.append(v)
        return np.stack(out, axis=0)
