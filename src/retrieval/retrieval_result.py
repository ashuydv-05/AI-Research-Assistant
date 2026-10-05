from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DocumentId = int | str


@dataclass
class RetrievalResult:
    id: DocumentId
    content: str
    title: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieval_sources: tuple[str, ...] = field(default_factory=tuple)
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    reranker_score: float | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
        if not self.retrieval_sources and self.source:
            self.retrieval_sources = (self.source,)

    @property
    def paper_id(self) -> str | None:
        value = self.metadata.get("paper_id")
        return str(value) if value is not None else None

    def to_dict(self) -> dict[str, Any]:
        metadata = {
            **self.metadata,
            "retrieval_sources": list(self.retrieval_sources),
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "rrf_score": self.rrf_score,
            "reranker_score": self.reranker_score,
        }
        return {
            "id": self.id,
            "content": self.content,
            "title": self.title,
            "score": self.score,
            "source": self.source,
            "metadata": metadata,
        }


# Backward-compatible name used by the evaluation and agent modules.
SearchResult = RetrievalResult


def default_hybrid_status() -> dict[str, str]:
    return {
        "dense": "not_run",
        "bm25": "not_run",
        "rrf": "not_run",
        "reranker": "not_run",
    }


@dataclass
class HybridSearchOutcome:
    results: list[RetrievalResult]
    retrieval_status: dict[str, str] = field(default_factory=default_hybrid_status)
    failure_reason: str | None = None
    stage_timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return bool(self.results) and all(
            self.retrieval_status.get(stage) == "success"
            for stage in ("dense", "bm25", "rrf")
        )
