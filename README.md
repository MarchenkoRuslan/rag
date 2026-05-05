# RAG Knowledge System

![RAG Knowledge System overview](slide.png)

Production-oriented RAG stack: FastAPI, FAISS + SQLite, OpenAI or local embeddings (sentence-transformers), OpenAI or Ollama for the LLM, and a minimal Streamlit UI.

## Requirements

- Python 3.13+ (recommended baseline)

## Setup

Create and **activate** a virtual environment first so dependencies stay isolated from the system Python.

**Windows (PowerShell)**

```powershell
cd c:\Repos\rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
# Optional: pip install -e ".[local]"   # sentence-transformers for EMBEDDING_PROVIDER=local
# Optional: pip install -e ".[ui]"      # Streamlit UI
# Contributors: pip install -e ".[dev]"
```

Runtime dependencies are in **`pyproject.toml`**. Extras: **`local`** (local embeddings), **`ui`** (Streamlit). **`dev`** is for tests and tooling (ruff, mypy, pytest, pytest-cov).

If script execution is blocked by policy, for the current user once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux / macOS**

```bash
cd /path/to/rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# Optional: pip install -e ".[local]" ".[ui]"  # as needed
# Contributors: pip install -e ".[dev]"
```

Your prompt should show `(.venv)`.

## Configuration

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # Unix
```

Edit `.env`: OpenAI key (if using `EMBEDDING_PROVIDER=openai` and/or `LLM_PROVIDER=openai`), models, relevance threshold, and `DATA_DIR` / `STORAGE_DIR`.

## Run the API

With `.venv` **activated**:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Docs: http://127.0.0.1:8000/docs

## Run the Streamlit UI

Install the UI extra (`pip install -e ".[ui]"` or `pip install -e ".[dev,ui]"` for contributors). In a second terminal, activate `.venv`, then:

```bash
streamlit run ui/streamlit_app.py
```

Set `RAG_API_URL` (default `http://127.0.0.1:8000`) to match the API.
If the API enforces `RAG_API_KEY`, set the same key in the Streamlit process environment.

## Tests

Install dev tools: `pip install -e ".[dev]"`.

```bash
pytest
ruff format --check app tests ui eval/scripts
ruff check app tests ui eval/scripts
mypy app
```

Local `pytest` does not enable coverage by default. CI runs `pytest` with `--cov=app --cov-report=term-missing --cov-fail-under=65` (see `.github/workflows/ci.yml`). To match CI locally:

```bash
pytest --cov=app --cov-report=term-missing --cov-fail-under=65
```

## Evaluation

The repo ships a minimal **golden set** under [`eval/`](eval/) that fixes what we mean by a "good answer":

- **Retrieval quality** — Precision@k / Recall@k / NDCG@k against `relevant_files` and `relevant_chunks` in [`eval/golden/questions.jsonl`](eval/golden/questions.jsonl).
- **Generation quality** — a four-axis rubric (groundedness, answer relevance, citations, refusal correctness) in [`eval/golden/rubric.md`](eval/golden/rubric.md).

Reproducible corpus + ingest CLI (writes to `eval/.cache/`, never touches the live `storage/`):

```bash
python eval/scripts/ingest_fixtures.py --clear --print-chunk-map
```

Metric computation and optional LLM-as-judge automation build on top of these artifacts. Schema is enforced by `tests/test_golden_set.py`. See [eval/README.md](eval/README.md) for the full workflow and the rationale behind the hybrid (doc-level + chunk-level) markup.

## Layout

- `pyproject.toml` — dependencies, extras (`local`, `ui`, `dev`), ruff/mypy/pytest config
- `app/` — FastAPI app, ingestion / retrieval / generation services, config
- `data/` — uploaded files
- `storage/` — FAISS index and SQLite chunk metadata
- `embeddings/` — optional model cache (`HF_HOME=embeddings` in `.env`)
- `ui/streamlit_app.py` — chat, upload, and document removal
- `eval/` — golden set assets (12 labeled questions, fixture corpus, generation rubric, ingest CLI). See [eval/README.md](eval/README.md).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest` | Upload `.txt` / `.pdf` (same filename replaces existing chunks) |
| POST | `/query` | Question → answer with sources and metrics |
| POST | `/query/stream` | SSE stream with sources, token events, and terminal done/error events |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{filename}` | Remove document from index and `data/` (safe basename only) |
| GET | `/health` | Service health, vector count, and optional LLM probe |

The same methods are available under the `/v1` prefix (for example `POST /v1/query`). Legacy paths without `/v1` remain available and include deprecation headers.

## Operations notes

- **Logs**: JSON via `structlog`; each request logs `method`, `path`, `status_code`, `duration_ms`; response includes `X-Request-ID`. Clients (including Streamlit) may send `X-Request-ID` to correlate with logs.
- **Optional auth**: set `RAG_API_KEY` to require `X-API-Key` or `Authorization: Bearer` on API routes (`/health` stays open). Control docs/openapi exemption with `API_KEY_EXEMPT_DOCS`.
- **Ingest size cap**: `MAX_INGEST_BYTES` (default 20MB); oversize uploads return HTTP 413.
- **Health LLM probe**: set `HEALTH_CHECK_LLM=true` to include `llm_ok` / `llm_error` in `GET /health` (short outbound check).
- **CORS**: `CORS_ALLOW_ORIGINS` (`*` or comma-separated origins) and `CORS_ALLOW_CREDENTIALS` (must be `false` when origins contain `*`).
- **LLM / OpenAI HTTP timeout**: `OPENAI_TIMEOUT_SECONDS` (OpenAI embeddings and chat, and Ollama chat requests).
- **Rate limits**: default `60/minute` global; endpoint-specific limits include ingest `10/minute` and query/query_stream `30/minute`.
- **Legacy route policy**: non-versioned routes remain available for compatibility and include `Deprecation` + `Link` headers pointing to `/v1`.
- **Streamlit exposure**: do not expose Streamlit directly on the public internet; place it behind reverse proxy + SSO/VPN/IP allowlist.
- **Index files**: written with `faiss.serialize_index` for Unicode paths on Windows; older `write_index` files are loaded via a temporary ASCII path.

See [.env.example](.env.example) for all environment variables.

## Docker Compose

- **`api`** image installs the base package only (OpenAI embeddings by default). To run **`EMBEDDING_PROVIDER=local`** in containers, rebuild with build arg `INSTALL_EXTRAS=local` (or `local,ui` if needed).
- **`ui`** image is built with `INSTALL_EXTRAS=ui` so Streamlit is included.

## Agent / AI assistants

See [AGENTS.md](AGENTS.md) for architecture notes and editing rules for this repository.
