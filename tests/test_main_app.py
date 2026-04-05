"""Integration smoke tests for the real app.main application."""

from __future__ import annotations

import importlib

from fastapi.testclient import TestClient

from app.config import get_settings
from tests.conftest import FakeEmbeddings


def _fake_embedding_provider(_settings):
    """Avoid loading sentence-transformers during app.main integration smoke test."""
    return FakeEmbeddings(16)


def test_main_app_health_with_lifespan(monkeypatch, tmp_path):
    """Real app startup should succeed and serve health endpoints."""
    monkeypatch.setenv("DATA_DIR", str((tmp_path / "data").resolve()))
    monkeypatch.setenv("STORAGE_DIR", str((tmp_path / "storage").resolve()))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "*")
    monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "false")
    get_settings.cache_clear()

    import app.main as main_mod  # pylint: disable=import-outside-toplevel

    main_mod = importlib.reload(main_mod)
    monkeypatch.setattr(main_mod, "build_embedding_provider", _fake_embedding_provider)
    try:
        with TestClient(main_mod.app) as client:
            health = client.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
            assert health.headers.get("X-Request-ID")
            assert health.headers.get("Deprecation") == "true"

            health_v1 = client.get("/v1/health")
            assert health_v1.status_code == 200
            assert health_v1.headers.get("Deprecation") is None
    finally:
        get_settings.cache_clear()
