from __future__ import annotations

from dataclasses import replace

from src.retrieval.errors import FusionError
from src.retrieval.retrieval_result import RetrievalResult


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    *,
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[RetrievalResult]:
    if not dense_results or not bm25_results:
        raise FusionError("RRF requires non-empty dense and BM25 result sets.")

    scores: dict[str, float] = {}
    documents: dict[str, RetrievalResult] = {}
    sources: dict[str, set[str]] = {}
    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    for source_name, results in (("dense", dense_results), ("bm25", bm25_results)):
        for rank, result in enumerate(results, start=1):
            key = str(result.id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            documents.setdefault(key, result)
            sources.setdefault(key, set()).add(source_name)
            if source_name == "dense":
                dense_ranks[key] = result.dense_rank or rank
            else:
                bm25_ranks[key] = result.bm25_rank or rank

    if not scores:
        raise FusionError("RRF produced no fused scores.")

    ranked_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
    fused: list[RetrievalResult] = []
    for key in ranked_ids:
        retrieval_sources = tuple(
            name for name in ("dense", "bm25") if name in sources[key]
        )
        fused.append(
            replace(
                documents[key],
                score=scores[key],
                source="hybrid",
                retrieval_sources=retrieval_sources,
                dense_rank=dense_ranks.get(key),
                bm25_rank=bm25_ranks.get(key),
                rrf_score=scores[key],
            )
        )
    return fused
