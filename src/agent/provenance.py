from __future__ import annotations

from typing import Literal


SourceMode = Literal["hybrid", "web", "direct", "failed"]

SOURCE_LABELS: dict[SourceMode, str] = {
    "hybrid": "Fully Verified Hybrid RAG",
    "web": "Web Search",
    "direct": "Direct LLM",
    "failed": "No Verified Source",
}


def default_retrieval_status() -> dict[str, str]:
    return {
        "dense": "not_run",
        "bm25": "not_run",
        "rrf": "not_run",
        "reranker": "not_run",
        "validation": "not_run",
        "tavily": "not_run",
    }


def hybrid_pipeline_succeeded(status: dict[str, str] | None) -> bool:
    current = status or {}
    return all(
        (
            current.get("dense") == "success",
            current.get("bm25") == "success",
            current.get("rrf") == "success",
            current.get("validation") == "passed",
        )
    )
