# Agent instructions — RAG Knowledge System

Use this file as the primary project contract for automated assistants (Cursor, Codex, etc.).

## Language policy

- **All source code** (Python, config strings shown to users, log message keys where they are English today, Streamlit UI copy, tests): **English only**. No Cyrillic in `*.py`, `*.md` under `app/`, `tests/`, `ui/`, or in `.env.example` values meant for operators.
- **User-facing docs** in the repo root (`README.md`, this file): **English** for consistency with the codebase.

## Stack

- **API**: FastAPI (`app/main.py`), routes in `app/api/routes.py`.
- **Dependencies**: `app/api/deps.py` resolves `Settings`, `VectorStore`, and `EmbeddingProviderBase` from `request.app.state` (wired in `lifespan`).
- **Services**: `ingestion`, `retrieval`, `generation`, `embeddings`; **storage** is `app/services/vector_store.py` (FAISS + SQLite).
- **UI**: `ui/streamlit_app.py` calls the HTTP API (`RAG_API_URL`).
- **Config**: `app/config.py` + `.env` / `.env.example` via `pydantic-settings`.

## Architecture (current)

```text
HTTP  →  routes  →  services (ingest / retrieve / generate)
                         ↓
                   vector_store (FAISS + SQLite)
                         ↓
                   files under DATA_DIR, index under STORAGE_DIR
```

**Design choices worth preserving**

- **Normalized vectors + inner product** in FAISS approximates cosine similarity; dimension must match the active embedding provider.
- **Persistence**: index saved with `faiss.serialize_index` and `Path.write_bytes` so Unicode paths work on Windows; loader supports legacy `write_index` files via a temporary ASCII path.
- **Re-ingest**: same filename replaces vectors (delete-by-filename then add).
- **Citations**: retrieval assigns `citation_id`; generation prompt requires `[N]`-style references.

**Reasonable extensions** (do not implement unless the task asks)

- Versioned API prefix (`/v1/...`).
- Background jobs for large PDFs.
- Pluggable `VectorStore` protocol for Qdrant/PGVector.
- `RAGService` facade if route handlers grow beyond orchestration.

## Running and testing

- Create and activate `.venv`, then `pip install -r requirements.txt`.
- API: `uvicorn app.main:app --reload` from repo root (with `PYTHONPATH` or `pip install -e .` if the project uses a package layout — follow `README.md`).
- Tests: `pytest` from repo root; tests patch `DATA_DIR` / `STORAGE_DIR` — do not rely on the developer’s real `storage/` directory.

## Conventions

- Prefer **small, focused diffs**; match existing style and typing (`from __future__ import annotations` where used).
- Use **`SegmentTimer`** (`app/utils/metrics.py`) for nested timing in routes instead of ad-hoc `perf_counter` lists.
- Use **structlog** with event names and structured fields; include `request_id` context when touching HTTP code.
- **Safe filenames**: use `app/api/filename.py` for uploads and path segments — no `..`, `/`, or `\`.
- **Secrets**: never commit `.env`; extend `.env.example` when adding settings.

## Files to avoid editing without explicit user request

- User-owned planning docs outside this repo (e.g. personal `.cursor/plans/*.plan.md`).

## Quick checklist before finishing a change

1. `pytest` passes.
2. `pylint app tests ui/streamlit_app.py` passes (install dev deps: `pip install -r requirements-dev.txt`; `pyproject.toml` configures venv `init-hook` and pragmatic disables).
3. No new Cyrillic in code or project English docs.
4. New settings documented in `.env.example` and `app/config.py`.
