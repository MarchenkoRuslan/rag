"""Schema validation for the golden set.

Keeps `eval/golden/questions.jsonl` consistent with the fixtures in
`eval/golden/fixtures/`. Exercises only the pure-Python `chunk_text`
function from the ingestion service; never loads embeddings or FAISS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.services.ingestion import chunk_text

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "eval" / "golden"
FIXTURES_DIR = GOLDEN_DIR / "fixtures"
QUESTIONS_PATH = GOLDEN_DIR / "questions.jsonl"

REQUIRED_FIELDS = {"id", "question", "relevant_files", "relevant_chunks", "answerable"}

# Must match GOLDEN_CHUNK_SIZE / GOLDEN_CHUNK_OVERLAP in
# eval/scripts/ingest_fixtures.py.
GOLDEN_CHUNK_SIZE = 500
GOLDEN_CHUNK_OVERLAP = 50


def _load_records() -> list[dict]:
    if not QUESTIONS_PATH.is_file():
        pytest.fail(f"Missing golden file: {QUESTIONS_PATH}")
    records: list[dict] = []
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON on line {line_no}: {e}")
    return records


@pytest.fixture(scope="module")
def golden_records() -> list[dict]:
    return _load_records()


@pytest.fixture(scope="module")
def fixture_filenames() -> set[str]:
    if not FIXTURES_DIR.is_dir():
        pytest.fail(f"Missing fixtures directory: {FIXTURES_DIR}")
    return {p.name for p in FIXTURES_DIR.glob("*.txt")}


@pytest.fixture(scope="module")
def fixture_chunk_counts() -> dict[str, int]:
    """Chunk counts per fixture under the golden-set chunking config.

    Builds Settings with explicit aliases (pydantic-settings honors aliases,
    not field names, when env-aliased fields are populated via __init__).
    Module-scoped fixtures run before the function-scoped autouse env fixture,
    so we cannot rely on the conftest env shimming here.
    """
    settings = Settings(
        CHUNK_SIZE=GOLDEN_CHUNK_SIZE,
        CHUNK_OVERLAP=GOLDEN_CHUNK_OVERLAP,
        OPENAI_API_KEY="sk-test-not-real",
        EMBEDDING_PROVIDER="local",
        LLM_PROVIDER="ollama",
    )
    counts: dict[str, int] = {}
    for path in sorted(FIXTURES_DIR.glob("*.txt")):
        counts[path.name] = len(chunk_text(path.read_text(encoding="utf-8"), settings))
    return counts


def test_required_fields_present(golden_records: list[dict]) -> None:
    assert golden_records, "questions.jsonl is empty"
    for rec in golden_records:
        missing = REQUIRED_FIELDS - rec.keys()
        assert not missing, f"{rec.get('id', '?')} missing fields: {missing}"


def test_field_types(golden_records: list[dict]) -> None:
    for rec in golden_records:
        rid = rec["id"]
        assert isinstance(rid, str) and rid, f"{rid!r}: id must be non-empty string"
        assert isinstance(rec["question"], str) and rec["question"].strip(), (
            f"{rid}: question must be non-empty string"
        )
        assert isinstance(rec["relevant_files"], list), f"{rid}: relevant_files must be list"
        assert all(isinstance(x, str) for x in rec["relevant_files"]), (
            f"{rid}: relevant_files entries must be strings"
        )
        assert isinstance(rec["relevant_chunks"], list), f"{rid}: relevant_chunks must be list"
        for ch in rec["relevant_chunks"]:
            assert isinstance(ch, dict), f"{rid}: each relevant_chunks entry must be object"
            assert set(ch.keys()) >= {"filename", "chunk_index"}, (
                f"{rid}: each relevant_chunks entry needs filename and chunk_index"
            )
            assert isinstance(ch["filename"], str), f"{rid}: chunk filename must be string"
            assert isinstance(ch["chunk_index"], int) and ch["chunk_index"] >= 0, (
                f"{rid}: chunk_index must be non-negative int"
            )
        assert isinstance(rec["answerable"], bool), f"{rid}: answerable must be bool"


def test_ids_unique(golden_records: list[dict]) -> None:
    ids = [rec["id"] for rec in golden_records]
    duplicates = {rid for rid in ids if ids.count(rid) > 1}
    assert not duplicates, f"Duplicate ids: {sorted(duplicates)}"


def test_relevant_files_exist(golden_records: list[dict], fixture_filenames: set[str]) -> None:
    for rec in golden_records:
        for filename in rec["relevant_files"]:
            assert filename in fixture_filenames, (
                f"{rec['id']}: relevant_files refers to missing fixture {filename!r}"
            )


def test_relevant_chunks_filenames_match_relevant_files(
    golden_records: list[dict], fixture_filenames: set[str]
) -> None:
    for rec in golden_records:
        relevant = set(rec["relevant_files"])
        for ch in rec["relevant_chunks"]:
            assert ch["filename"] in fixture_filenames, (
                f"{rec['id']}: chunk filename {ch['filename']!r} not in fixtures"
            )
            assert ch["filename"] in relevant, (
                f"{rec['id']}: chunk filename {ch['filename']!r} not listed in relevant_files"
            )


def test_unanswerable_have_empty_labels(golden_records: list[dict]) -> None:
    for rec in golden_records:
        if not rec["answerable"]:
            assert rec["relevant_files"] == [], (
                f"{rec['id']}: answerable=false must have empty relevant_files"
            )
            assert rec["relevant_chunks"] == [], (
                f"{rec['id']}: answerable=false must have empty relevant_chunks"
            )


def test_answerable_have_at_least_one_relevant_file(golden_records: list[dict]) -> None:
    for rec in golden_records:
        if rec["answerable"]:
            assert rec["relevant_files"], (
                f"{rec['id']}: answerable=true must list at least one relevant file"
            )


def test_unanswerable_count_in_expected_range(golden_records: list[dict]) -> None:
    unanswerable = [r for r in golden_records if not r["answerable"]]
    assert 1 <= len(unanswerable) <= 2, (
        f"Expected 1-2 unanswerable items for the minimal golden set, got {len(unanswerable)}"
    )


def test_relevant_chunk_indices_within_bounds(
    golden_records: list[dict],
    fixture_chunk_counts: dict[str, int],
) -> None:
    """Every (filename, chunk_index) must point to a real chunk under the
    golden-set chunking config (size 500, overlap 50)."""
    for rec in golden_records:
        for ch in rec["relevant_chunks"]:
            count = fixture_chunk_counts.get(ch["filename"])
            assert count is not None, (
                f"{rec['id']}: chunk references unknown fixture {ch['filename']!r}"
            )
            assert ch["chunk_index"] < count, (
                f"{rec['id']}: chunk_index {ch['chunk_index']} is out of bounds "
                f"for {ch['filename']!r} which produces {count} chunks "
                f"under chunk_size={GOLDEN_CHUNK_SIZE}, overlap={GOLDEN_CHUNK_OVERLAP}"
            )


def test_fixtures_are_non_empty(fixture_chunk_counts: dict[str, int]) -> None:
    for filename, count in fixture_chunk_counts.items():
        assert count >= 1, f"Fixture {filename!r} produced zero chunks"
