"""Timing and simple retrieval metrics helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class SegmentTimer:
    """Wall-clock segment; ``elapsed_s`` is set when the context exits."""

    __slots__ = ("_t0", "elapsed_s")

    def __init__(self) -> None:
        self._t0 = 0.0
        self.elapsed_s = 0.0

    def __enter__(self) -> SegmentTimer:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed_s = time.perf_counter() - self._t0


@dataclass
class QueryMetrics:  # pylint: disable=too-many-instance-attributes
    """Aggregated metrics for a single /query call."""

    response_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    num_sources_used: int = 0
    num_chunks_retrieved: int = 0
    mean_relevance_score: float | None = None
    max_relevance_score: float | None = None
    retrieval_accuracy_hint: float | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = {
            "response_time_ms": round(self.response_time_ms, 2),
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "generation_time_ms": round(self.generation_time_ms, 2),
            "num_sources_used": self.num_sources_used,
            "num_chunks_retrieved": self.num_chunks_retrieved,
            "mean_relevance_score": (
                round(self.mean_relevance_score, 4)
                if self.mean_relevance_score is not None
                else None
            ),
            "max_relevance_score": (
                round(self.max_relevance_score, 4) if self.max_relevance_score is not None else None
            ),
            "retrieval_accuracy_hint": (
                round(self.retrieval_accuracy_hint, 4)
                if self.retrieval_accuracy_hint is not None
                else None
            ),
        }
        return d
