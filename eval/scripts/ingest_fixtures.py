"""Ingest the golden-set fixtures into a dedicated FAISS+SQLite store.

CLI utility for the eval workflow. By default it writes to ``eval/.cache/``
so the live ``storage/`` directory is never touched. The script is a thin
wrapper around the existing ingestion service.

Typical usage::

    python eval/scripts/ingest_fixtures.py --clear --print-chunk-map

The optional ``--print-chunk-map`` flag prints
``filename -> [(chunk_index, preview)]`` so the operator can fill in
``relevant_chunks`` entries inside ``eval/golden/questions.jsonl``.

Reproducibility note: the ``chunk_index`` values committed in
``eval/golden/questions.jsonl`` were produced with chunk size 500 and
chunk overlap 50. The script enforces these defaults via CLI flags so
running it under a different ``.env`` does not silently invalidate the
golden-set markup.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Imports below depend on the repo root being on sys.path when this file
# is executed directly (for example, ``python eval/scripts/ingest_fixtures.py``).
from app.config import Settings  # noqa: E402
from app.services.embeddings import build_embedding_provider  # noqa: E402
from app.services.ingestion import ingest_bytes  # noqa: E402
from app.services.vector_store import VectorStore  # noqa: E402

DEFAULT_FIXTURES_DIR = REPO_ROOT / "eval" / "golden" / "fixtures"
DEFAULT_STORAGE_DIR = REPO_ROOT / "eval" / ".cache" / "storage"
DEFAULT_DATA_DIR = REPO_ROOT / "eval" / ".cache" / "data"
GOLDEN_CHUNK_SIZE = 500
GOLDEN_CHUNK_OVERLAP = 50


def _print_chunk_map(db_path: Path, preview_chars: int = 80) -> None:
    if not db_path.is_file():
        print(f"[chunk-map] metadata.db not found at {db_path}", file=sys.stderr)
        return
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT filename, chunk_index, text FROM chunks ORDER BY filename, chunk_index"
        ).fetchall()
    finally:
        conn.close()
    current_file: str | None = None
    for filename, chunk_index, text in rows:
        if filename != current_file:
            print(f"\n{filename}:")
            current_file = filename
        preview = text.replace("\n", " ").strip()[:preview_chars]
        print(f"  [{chunk_index:02d}] {preview}")


def _build_settings(
    data_dir: Path,
    storage_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> Settings:
    """Build Settings with explicit data/storage and chunking via env overrides.

    Mirrors the test pattern in ``tests/conftest.py`` so we do not depend on
    pydantic-settings keyword aliasing rules. Chunking is forced via env so
    that the golden-set markup stays valid regardless of what is in ``.env``.

    LLM_PROVIDER is always forced to ``ollama`` so that the openai-key
    validator does not fire: this script only ingests (embed + store) and
    never calls any LLM. Embedding provider is left to the caller's env.
    """
    os.environ["DATA_DIR"] = str(data_dir)
    os.environ["STORAGE_DIR"] = str(storage_dir)
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
    # The ingest script never calls generation, so the LLM provider is
    # irrelevant. Forcing ollama prevents validate_openai_key from requiring
    # OPENAI_API_KEY when EMBEDDING_PROVIDER=local (the offline/CI path).
    os.environ.setdefault("LLM_PROVIDER", "ollama")
    return Settings()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    parser.add_argument("--storage-dir", type=Path, default=DEFAULT_STORAGE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=GOLDEN_CHUNK_SIZE,
        help=(
            f"CHUNK_SIZE override (default {GOLDEN_CHUNK_SIZE}, "
            "matches the chunk_index values in questions.jsonl)"
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=GOLDEN_CHUNK_OVERLAP,
        help=(
            f"CHUNK_OVERLAP override (default {GOLDEN_CHUNK_OVERLAP}, "
            "matches the chunk_index values in questions.jsonl)"
        ),
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe storage-dir before ingesting so chunk_index values are deterministic",
    )
    parser.add_argument(
        "--print-chunk-map",
        action="store_true",
        help="Print filename / chunk_index / preview after ingest",
    )
    args = parser.parse_args(argv)

    fixtures_dir = args.fixtures_dir.resolve()
    storage_dir = args.storage_dir.resolve()
    data_dir = args.data_dir.resolve()

    if not fixtures_dir.is_dir():
        parser.error(f"Fixtures directory not found: {fixtures_dir}")

    if args.clear and storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be strictly less than --chunk-size")

    if args.chunk_size != GOLDEN_CHUNK_SIZE or args.chunk_overlap != GOLDEN_CHUNK_OVERLAP:
        print(
            f"[warn] using non-default chunking "
            f"(size={args.chunk_size}, overlap={args.chunk_overlap}); "
            "regenerate relevant_chunks in questions.jsonl after this run.",
            file=sys.stderr,
        )

    settings = _build_settings(
        data_dir,
        storage_dir,
        args.chunk_size,
        args.chunk_overlap,
    )
    embeddings = build_embedding_provider(settings)
    store = VectorStore(storage_dir, embeddings.dimension, settings)

    txt_files = sorted(fixtures_dir.glob("*.txt"))
    if not txt_files:
        parser.error(f"No .txt fixtures found in {fixtures_dir}")

    try:
        for path in txt_files:
            data = path.read_bytes()
            result = ingest_bytes(path.name, data, store, embeddings, settings)
            print(
                f"ingested {path.name}: "
                f"chunks={result.chunks_added} chars={result.characters_extracted}"
            )
    finally:
        store.close()
        emb_close = getattr(embeddings, "close", None)
        if callable(emb_close):
            emb_close()

    if args.print_chunk_map:
        _print_chunk_map(storage_dir / "metadata.db")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
