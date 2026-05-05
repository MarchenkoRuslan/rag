# RAG evaluation: golden set

This folder defines what we mean by a "good answer" in this repo and ships
the minimal data to measure it. **No code under `app/` depends on anything
in this folder.** It provides definitions and reproducible evaluation data.

## Two layers of quality

A RAG system can fail in two very different ways. The metrics below are kept
deliberately separate so a regression points at exactly one layer.

| Layer      | Question                                                | How we measure                          |
| ---------- | ------------------------------------------------------- | --------------------------------------- |
| Retrieval  | Did the top-k contain the chunks we needed?             | Precision@k, Recall@k, NDCG@k           |
| Generation | Does the answer rely on the context and answer the ask? | Rubric (and optionally LLM-as-judge)    |

Confusing the two is the most common reason RAG eval reports look fine but
the product still ships hallucinations. See [`golden/rubric.md`](golden/rubric.md)
for the rubric and the per-axis triage table.

## Granularity choice: hybrid

`questions.jsonl` carries **two** levels of relevance labels:

- `relevant_files`: list of filenames. **Always populated.** Stable under
  changes to `CHUNK_SIZE` / `CHUNK_OVERLAP`. Drives doc-level
  Precision@k / Recall@k / NDCG@k.
- `relevant_chunks`: optional list of `{filename, chunk_index}`. Tied to a
  specific chunking configuration. Drives chunk-level metrics on the subset
  of items that have it.

Why hybrid:

- Doc-level is the **stable baseline**: it survives chunker tweaks, so it is
  the metric we will track over time.
- Chunk-level is **diagnostic**: when doc-level Recall@k is fine but answers
  still suffer, chunk-level metrics show whether the right chunk is in the
  top-k.

When you change `CHUNK_SIZE` or `CHUNK_OVERLAP`, doc-level metrics keep
working unchanged. Chunk-level labels must be regenerated (see workflow
below).

## Folder layout

```text
eval/
  README.md                        # this file
  golden/
    questions.jsonl                # 12 labeled questions (hybrid)
    rubric.md                      # generation rubric
    fixtures/                      # tiny reproducible corpus
      chunking-strategies.txt
      embeddings-faiss.txt
      evaluation-metrics.txt
      llm-hallucinations.txt
      prompt-injection.txt         # distractor
      rag-overview.txt
      vector-stores.txt            # distractor
  scripts/
    ingest_fixtures.py             # CLI: build a fresh index in eval/.cache/
```

## `questions.jsonl` schema

One JSON object per line:

```json
{
  "id": "q-001",
  "question": "What does CHUNK_OVERLAP control and what is its constraint relative to CHUNK_SIZE?",
  "relevant_files": ["chunking-strategies.txt"],
  "relevant_chunks": [
    {"filename": "chunking-strategies.txt", "chunk_index": 3},
    {"filename": "chunking-strategies.txt", "chunk_index": 4}
  ],
  "answerable": true,
  "notes": "Free-form rationale for reviewers."
}
```

| Field             | Type                          | Required | Notes                                                    |
| ----------------- | ----------------------------- | -------- | -------------------------------------------------------- |
| `id`              | string                        | yes      | Unique across the file, e.g. `q-001`.                    |
| `question`        | string                        | yes      | The user-facing question.                                |
| `relevant_files`  | `list[str]`                   | yes      | Filenames present in `golden/fixtures/`. Empty list when `answerable=false`. |
| `relevant_chunks` | `list[{filename, chunk_index}]` | yes (may be empty) | Chunk-level labels. Empty when not annotated or when `answerable=false`. |
| `answerable`      | bool                          | yes      | `false` means the corpus has no evidence and the system should refuse. |
| `notes`           | string                        | no       | Optional reviewer note.                                  |

The current set has **12 questions**: 8 single-doc, 2 multi-doc, 2
unanswerable.

## Reproducing the index

The script below builds a fresh FAISS+SQLite store under `eval/.cache/` so
the live `storage/` directory is never touched.

**Prerequisite.** Choose an embedding provider before running:

- `EMBEDDING_PROVIDER=openai` (default) needs `OPENAI_API_KEY`.
- `EMBEDDING_PROVIDER=local` needs `pip install -e ".[local]"`. This is
  preferred for offline / CI runs because it is deterministic.

```bash
# from repo root, .venv activated
python eval/scripts/ingest_fixtures.py --clear --print-chunk-map
```

What the script does:

1. Wipes `eval/.cache/storage/` so chunk indices are deterministic
   (`--clear`).
2. Forces `CHUNK_SIZE=500` / `CHUNK_OVERLAP=50` (the values the committed
   `chunk_index` markup is calibrated for). Override with
   `--chunk-size` / `--chunk-overlap` only when you intend to recompute
   the markup; the script prints a warning in that case.
3. For every `*.txt` in `eval/golden/fixtures/`, calls the same
   `ingest_bytes` used by the API.
4. With `--print-chunk-map`, prints
   `filename -> [(chunk_index, preview)]` from the resulting SQLite, which
   is what you copy into the `relevant_chunks` field of `questions.jsonl`
   when adding new entries.

Full CLI:

```text
python eval/scripts/ingest_fixtures.py
  [--fixtures-dir PATH]   default: eval/golden/fixtures
  [--storage-dir PATH]    default: eval/.cache/storage
  [--data-dir PATH]       default: eval/.cache/data
  [--chunk-size N]        default: 500  (calibrated to questions.jsonl)
  [--chunk-overlap N]     default: 50   (calibrated to questions.jsonl)
  [--clear]               wipe storage-dir before ingesting
  [--print-chunk-map]     print filename / chunk_index / preview after ingest
```

`eval/.cache/` is in `.gitignore`; only the fixtures and `questions.jsonl`
are committed.

## Updating chunk-level labels after a chunker change

If you change `CHUNK_SIZE` / `CHUNK_OVERLAP` in `app/config.py`:

1. Update the `GOLDEN_CHUNK_SIZE` / `GOLDEN_CHUNK_OVERLAP` constants in
   [`scripts/ingest_fixtures.py`](scripts/ingest_fixtures.py) and in
   [`tests/test_golden_set.py`](../tests/test_golden_set.py) so they
   stay in lockstep.
2. Re-run
   `python eval/scripts/ingest_fixtures.py --clear --print-chunk-map`
   (or pass explicit `--chunk-size` / `--chunk-overlap`).
3. For each entry in `questions.jsonl` with non-empty `relevant_chunks`,
   update `chunk_index` to match the new chunking. Doc-level
   `relevant_files` does not need to change.
4. Run `pytest tests/test_golden_set.py` to validate the schema (the
   bounds check will fail loudly if a stale `chunk_index` slipped
   through).

## Included and excluded

In:

- Definitions of the two quality layers.
- Reproducible corpus and golden questions.
- Generation rubric.
- Schema validation test.

Excluded:

- Actual metric computation (`run_eval.py`).
- LLM-as-judge automation.
- Larger corpus / multilingual extension.
