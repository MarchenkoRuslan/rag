"""FAISS index + SQLite metadata for chunk storage and retrieval."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config import Settings
from app.utils.logging import get_logger

log = get_logger("vector_store")


@dataclass
class ChunkRecord:
    faiss_id: int
    filename: str
    chunk_index: int
    text: str


def _is_idmap_index(index: faiss.Index) -> bool:
    return hasattr(index, "add_with_ids") and hasattr(index, "remove_ids")


class VectorStore:
    """Inner-product index on L2-normalized vectors (cosine similarity)."""

    def __init__(self, storage_dir: Path, embedding_dim: int, settings: Settings) -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._storage_dir / "index.faiss"
        self._db_path = self._storage_dir / "metadata.db"
        self._settings = settings
        self._dim = int(embedding_dim)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        self._verify_meta()
        self._index = self._load_or_create_index()

    def _init_db(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS store_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                faiss_id INTEGER PRIMARY KEY,
                filename TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_filename ON chunks(filename);
            """
        )
        self._conn.commit()

    def _meta_get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM store_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _meta_set(self, key: str, value: str, *, commit: bool = True) -> None:
        self._conn.execute(
            """
            INSERT INTO store_meta(key, value) VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        if commit:
            self._conn.commit()

    def _verify_meta(self) -> None:
        dim_s = self._meta_get("embedding_dim")
        prov_s = self._meta_get("embedding_provider")
        if dim_s is None and prov_s is None:
            return
        if dim_s is not None and int(dim_s) != self._dim:
            raise RuntimeError(
                f"Storage embedding dimension {dim_s} != current {self._dim}. "
                "Clear storage directory or match EMBEDDING_PROVIDER / models."
            )
        if prov_s is not None and prov_s != self._settings.embedding_provider.value:
            log.warning(
                "embedding_provider_mismatch",
                stored=prov_s,
                current=self._settings.embedding_provider.value,
            )

    def _ensure_meta_initialized(self) -> None:
        if self._meta_get("embedding_dim") is None:
            self._meta_set("embedding_dim", str(self._dim), commit=False)
            self._meta_set(
                "embedding_provider",
                str(self._settings.embedding_provider.value),
                commit=False,
            )

    def _empty_idmap_index(self) -> faiss.Index:
        base = faiss.IndexFlatIP(self._dim)
        return faiss.IndexIDMap2(base)

    def _migrate_flat_ip_to_idmap(self, flat: faiss.Index) -> faiss.Index:
        n = int(flat.ntotal)
        d = int(flat.d)
        new_index = self._empty_idmap_index()
        if n == 0:
            log.info("migrated_empty_flat_to_idmap")
            return new_index
        vecs = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            vecs[i] = flat.reconstruct(int(i))
        ids = np.arange(n, dtype=np.int64)
        new_index.add_with_ids(vecs, ids)  # pylint: disable=no-value-for-parameter
        log.info("migrated_flat_ip_to_idmap", vectors=n)
        self._write_index_blob(new_index)
        return new_index

    def _write_index_blob(self, index: faiss.Index) -> None:
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        blob = bytes(faiss.serialize_index(index))
        fd, temp_path = tempfile.mkstemp(
            prefix="index-", suffix=".faiss", dir=str(self._storage_dir)
        )
        try:
            with os.fdopen(fd, "wb") as tmp:
                tmp.write(blob)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(temp_path, self._index_path)
        except Exception:
            Path(temp_path).unlink(missing_ok=True)
            raise

    def _read_index_from_disk(self) -> faiss.Index:
        raw = self._index_path.read_bytes()
        try:
            return faiss.deserialize_index(raw)
        except Exception:  # pylint: disable=broad-exception-caught
            with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                return faiss.read_index(tmp_path)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    def _load_or_create_index(self) -> faiss.Index:
        if not self._index_path.exists():
            return self._empty_idmap_index()
        index = self._read_index_from_disk()
        if int(index.d) != self._dim:
            raise RuntimeError(f"FAISS index dimension {index.d} != expected {self._dim}")
        if isinstance(index, faiss.IndexFlatIP):
            return self._migrate_flat_ip_to_idmap(index)
        if _is_idmap_index(index):
            return index
        raise RuntimeError(f"Unsupported FAISS index type: {type(index)}")

    def persist(self) -> None:
        with self._lock:
            self._write_index_blob(self._index)

    def count(self) -> int:
        with self._lock:
            return int(self._index.ntotal)

    def _next_faiss_ids(self, n: int) -> np.ndarray:
        row = self._conn.execute("SELECT COALESCE(MAX(faiss_id), -1) AS m FROM chunks").fetchone()
        start = int(row["m"]) + 1
        return np.arange(start, start + n, dtype=np.int64)

    def add_chunks(
        self,
        vectors: np.ndarray,
        filename: str,
        chunk_texts: Sequence[str],
    ) -> int:
        """Append normalized vectors; returns number of chunks added."""
        if vectors.size == 0:
            return 0
        if vectors.shape[1] != self._dim:
            raise ValueError(f"Vector dim {vectors.shape[1]} != {self._dim}")
        n = len(chunk_texts)
        if vectors.shape[0] != n:
            raise ValueError("vectors and chunk_texts length mismatch")
        with self._lock:
            index_before = faiss.serialize_index(self._index)
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                ids_np = self._next_faiss_ids(n)
                if not _is_idmap_index(self._index):
                    raise RuntimeError("FAISS index is not ID-mapped")
                self._index.add_with_ids(vectors, ids_np)
                now = datetime.now(UTC).isoformat()
                rows = [
                    (int(ids_np[i]), filename, i, text, now) for i, text in enumerate(chunk_texts)
                ]
                self._conn.executemany(
                    """
                    INSERT INTO chunks(faiss_id, filename, chunk_index, text, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                self._ensure_meta_initialized()
                self._write_index_blob(self._index)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                self._index = faiss.deserialize_index(index_before)
                self._write_index_blob(self._index)
                raise
        log.info(
            "chunks_added",
            filename=filename,
            count=len(chunk_texts),
            total=self._index.ntotal,
        )
        return len(chunk_texts)

    def delete_by_filename(self, filename: str) -> int:
        """Remove all chunks for a file from SQLite and FAISS. Returns rows removed."""
        with self._lock:
            index_before = faiss.serialize_index(self._index)
            rows = self._conn.execute(
                "SELECT faiss_id FROM chunks WHERE filename = ?", (filename,)
            ).fetchall()
            if not rows:
                return 0
            ids = np.array([int(r["faiss_id"]) for r in rows], dtype=np.int64)
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute("DELETE FROM chunks WHERE filename = ?", (filename,))
                if _is_idmap_index(self._index):
                    self._index.remove_ids(ids)
                self._write_index_blob(self._index)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                self._index = faiss.deserialize_index(index_before)
                self._write_index_blob(self._index)
                raise
        log.info("chunks_deleted", filename=filename, count=len(ids))
        return len(ids)

    def replace_chunks(
        self,
        vectors: np.ndarray,
        filename: str,
        chunk_texts: Sequence[str],
    ) -> int:
        """Atomically replace all chunks for filename and append new ones."""
        if vectors.size == 0:
            return self.delete_by_filename(filename)
        if vectors.shape[1] != self._dim:
            raise ValueError(f"Vector dim {vectors.shape[1]} != {self._dim}")
        n = len(chunk_texts)
        if vectors.shape[0] != n:
            raise ValueError("vectors and chunk_texts length mismatch")
        with self._lock:
            index_before = faiss.serialize_index(self._index)
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                old = self._conn.execute(
                    "SELECT faiss_id FROM chunks WHERE filename = ?", (filename,)
                ).fetchall()
                old_ids = np.array([int(r["faiss_id"]) for r in old], dtype=np.int64)
                self._conn.execute("DELETE FROM chunks WHERE filename = ?", (filename,))
                if len(old_ids) and _is_idmap_index(self._index):
                    self._index.remove_ids(old_ids)

                ids_np = self._next_faiss_ids(n)
                if not _is_idmap_index(self._index):
                    raise RuntimeError("FAISS index is not ID-mapped")
                self._index.add_with_ids(vectors, ids_np)
                now = datetime.now(UTC).isoformat()
                rows = [
                    (int(ids_np[i]), filename, i, text, now) for i, text in enumerate(chunk_texts)
                ]
                self._conn.executemany(
                    """
                    INSERT INTO chunks(faiss_id, filename, chunk_index, text, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                self._ensure_meta_initialized()
                self._write_index_blob(self._index)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                self._index = faiss.deserialize_index(index_before)
                self._write_index_blob(self._index)
                raise
        log.info(
            "chunks_replaced",
            filename=filename,
            count=n,
            total=self._index.ntotal,
        )
        return n

    def get_by_faiss_ids(self, ids: list[int]) -> dict[int, ChunkRecord]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT faiss_id, filename, chunk_index, text FROM chunks "
                f"WHERE faiss_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {
            int(r["faiss_id"]): ChunkRecord(
                faiss_id=int(r["faiss_id"]),
                filename=r["filename"],
                chunk_index=int(r["chunk_index"]),
                text=r["text"],
            )
            for r in rows
        }

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[list[int], list[float]]:
        """Returns (ids, scores) for inner product (cosine for normalized vectors)."""
        q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        with self._lock:
            if self._index.ntotal == 0:
                return [], []
            k = min(top_k, int(self._index.ntotal))
            scores, ids = self._index.search(q, k)
        ids_list = [int(i) for i in ids[0] if i >= 0]
        scores_list = [float(s) for s, i in zip(scores[0], ids[0], strict=True) if i >= 0]
        return ids_list, scores_list

    def list_documents(
        self,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        query = """
            SELECT filename,
                   COUNT(*) AS chunk_count,
                   MIN(created_at) AS first_upload
            FROM chunks
            GROUP BY filename
            ORDER BY filename
        """
        params: list[int] = []
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            params = [limit, offset]
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "filename": r["filename"],
                "chunk_count": int(r["chunk_count"]),
                "uploaded_at": str(r["first_upload"]) if r["first_upload"] is not None else None,
            }
            for r in rows
        ]

    def document_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(DISTINCT filename) AS n FROM chunks").fetchone()
        return int(row["n"]) if row else 0

    def close(self) -> None:
        self._conn.close()
