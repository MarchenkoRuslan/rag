# Generation rubric

The rubric is for **generation quality only**. Retrieval quality is measured
separately with Precision@k, Recall@k, and NDCG@k against
`relevant_files` / `relevant_chunks` in `questions.jsonl`. A wrong answer can
mean either the retrieval missed the right chunks **or** the generator
ignored / contradicted them; the two layers must be scored independently.

Each generated answer is rated on four axes. Each axis is scored on a
**1 - 5 integer scale**. The final generation score is the average of the
four axes (range 1.0 - 5.0).

## Axes

### 1. Groundedness (factual support)

Does every factual claim in the answer trace back to the retrieved context?

| Score | Meaning                                                                |
| ----- | ---------------------------------------------------------------------- |
| 5     | Every claim is directly supported by at least one retrieved chunk.     |
| 4     | All material claims supported; minor phrasing goes slightly beyond.    |
| 3     | At least one notable claim is unsupported but plausible.               |
| 2     | Multiple unsupported claims, or one clearly fabricated fact.           |
| 1     | Answer is mostly invented or contradicts the retrieved context.        |

### 2. Answer relevance

Does the answer address the user's actual question?

| Score | Meaning                                                                |
| ----- | ---------------------------------------------------------------------- |
| 5     | Direct, complete answer to the question as asked.                      |
| 4     | Correct answer with minor padding or off-topic asides.                 |
| 3     | Partially answers the question; misses an important sub-point.        |
| 2     | Mostly off-topic; touches the question only tangentially.              |
| 1     | Does not answer the question.                                          |

### 3. Citations

Are inline `[N]` citations present and correctly linked to the retrieved
context blocks, matching the format the system prompt requires (see
`SYSTEM_PROMPT` in [app/services/generation.py](../../app/services/generation.py))?

| Score | Meaning                                                                |
| ----- | ---------------------------------------------------------------------- |
| 5     | All material claims carry correct `[N]` citations to relevant chunks.  |
| 4     | Citations present and correct, with at most one missing or extra.      |
| 3     | Citations partially present; some claims uncited.                      |
| 2     | Citations missing on most claims, or several pointing to wrong chunks. |
| 1     | No citations, or citations that do not match retrieved chunks.         |

### 4. Refusal correctness (only for `answerable=false`)

For golden-set entries with `answerable: false`, the expected behavior is an
explicit refusal that mirrors the contract in
[app/services/generation.py](../../app/services/generation.py)
(`I don't have enough information to answer this question.`).

| Score | Meaning                                                                |
| ----- | ---------------------------------------------------------------------- |
| 5     | Refuses cleanly and matches the expected refusal phrasing.             |
| 4     | Refuses with a slightly different wording but no fabricated facts.     |
| 3     | Refuses but adds speculative content.                                  |
| 2     | Attempts an answer with hedging; partially fabricated.                 |
| 1     | Confidently answers with fabricated information (hallucination).       |

For `answerable=true` entries, this axis is **not scored**; the final score
is the average over the remaining three axes.

## Scoring workflow

1. Run the API or service stack against an index built from
   `eval/golden/fixtures/` (see `eval/README.md`).
2. For each question in `questions.jsonl`, capture:
   - the retrieved chunks (for retrieval metrics),
   - the generated answer (for the rubric).
3. A human (or LLM-as-judge on a sub-sample) assigns 1 - 5 to each
   applicable axis using this rubric.
4. Aggregate per-axis means across the set; flag any item with **any axis
   below 3** for triage.

## Rubric / metric pairing

| Failure mode                                  | Layer that owns it | Primary signal                         |
| --------------------------------------------- | ------------------ | -------------------------------------- |
| Right chunks not in top-k                     | Retrieval          | Recall@k drops                         |
| Right chunks present but ranked below k       | Retrieval          | NDCG@k drops, Recall@k stable          |
| Top-k contains relevant chunks, answer wrong  | Generation         | Groundedness or Answer relevance drops |
| Confident answer when corpus has no evidence  | Generation         | Refusal correctness drops (`answerable=false`) |
| Citations missing or pointing to wrong chunks | Generation         | Citations score drops                  |

This pairing is the contract between the golden set and automated metric
scripts.
