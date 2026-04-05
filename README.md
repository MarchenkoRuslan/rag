# RAG Knowledge System

Production-oriented RAG stack: FastAPI, FAISS + SQLite, OpenAI or local embeddings (sentence-transformers), OpenAI or Ollama for the LLM, and a minimal Streamlit UI.

## Requirements

- Python 3.11+ (recommended)

## Setup

Create and **activate** a virtual environment first so dependencies stay isolated from the system Python.

**Windows (PowerShell)**

```powershell
cd c:\Repos\rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If script execution is blocked by policy, for the current user once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Linux / macOS**

```bash
cd /path/to/rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

In a second terminal, activate `.venv`, then:

```bash
streamlit run ui/streamlit_app.py
```

Set `RAG_API_URL` (default `http://127.0.0.1:8000`) to match the API.

## Tests

```bash
pytest
```

## Layout

- `app/` — FastAPI app, ingestion / retrieval / generation services, config
- `data/` — uploaded files
- `storage/` — FAISS index and SQLite chunk metadata
- `embeddings/` — optional model cache (`HF_HOME=embeddings` in `.env`)
- `ui/streamlit_app.py` — chat, upload, and document removal

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest` | Upload `.txt` / `.pdf` (same filename replaces existing chunks) |
| POST | `/query` | Question → answer with sources and metrics |
| GET | `/documents` | List indexed documents |
| DELETE | `/documents/{filename}` | Remove document from index and `data/` (safe basename only) |

## Operations notes

- **Logs**: JSON via `structlog`; each request logs `method`, `path`, `status_code`, `duration_ms`; response includes `X-Request-ID`.
- **CORS**: `CORS_ALLOW_ORIGINS` (`*` or comma-separated origins).
- **LLM / OpenAI HTTP timeout**: `OPENAI_TIMEOUT_SECONDS` (OpenAI embeddings and chat, and Ollama chat requests).
- **Index files**: written with `faiss.serialize_index` for Unicode paths on Windows; older `write_index` files are loaded via a temporary ASCII path.

See [.env.example](.env.example) for all environment variables.

## Agent / AI assistants

See [AGENTS.md](AGENTS.md) for architecture notes and editing rules for this repository.
