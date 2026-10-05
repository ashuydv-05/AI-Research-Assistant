from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from loguru import logger

from src.config.settings import settings
from src.retrieval.base import BaseRetriever
from src.retrieval.dense.qdrant_retriever import QdrantRetriever
from src.retrieval.errors import (
    BM25RetrievalError,
    DenseRetrievalError,
    FusionError,
    RerankingError,
)
from src.retrieval.fusion.rrf import reciprocal_rank_fusion
from src.retrieval.lexical.elasticsearch_retriever import ElasticsearchRetriever
from src.retrieval.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.retrieval_result import (
    HybridSearchOutcome,
    RetrievalResult,
    default_hybrid_status,
)


@dataclass(frozen=True)
class RetrievalConfig:
    prefetch_k: int = settings.hybrid_retrieval.prefetch_k
    final_k: int = settings.hybrid_retrieval.final_k
    rrf_k: int = settings.hybrid_retrieval.rrf_k
    rerank_top_k: int = settings.reranker.top_k
    reranker_enabled: bool = settings.reranker.enabled


class HybridRetriever(BaseRetriever):
    """Strict dense + BM25 retriever with optional mandatory reranking."""

    def __init__(
        self,
        dense_retriever: QdrantRetriever | None = None,
        lexical_retriever: ElasticsearchRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        config: RetrievalConfig | None = None,
        searcher=None,
    ) -> None:
        # `searcher` preserves the old constructor for existing callers/tests.
        self._legacy_searcher = searcher
        self.dense_retriever = dense_retriever or QdrantRetriever()
        self.lexical_retriever = lexical_retriever or ElasticsearchRetriever()
        self.reranker = reranker or CrossEncoderReranker()
        self.config = config or RetrievalConfig()

    @property
    def name(self) -> str:
        return "hybrid"

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query or not query.strip():
            return []
        if self._legacy_searcher is not None:
            return self._legacy_searcher.search(query=query, top_k=top_k)
        return self.search_with_status(query, top_k=top_k).results

    def dense_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        return self.dense_retriever.search(query, top_k=top_k)

    def bm25_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        return self.lexical_retriever.search(query, top_k=top_k)

    def search_with_status(
        self,
        query: str,
        top_k: int | None = None,
        prefetch: int | None = None,
    ) -> HybridSearchOutcome:
        status = default_hybrid_status()
        timings: dict[str, float] = {}
        if not query or not query.strip():
            return HybridSearchOutcome([], status, "Retrieval query is empty.", timings)

        final_k = top_k or self.config.final_k
        prefetch_k = prefetch or self.config.prefetch_k

        try:
            started = perf_counter()
            dense = self.dense_search(query, top_k=prefetch_k)
            if not dense:
                raise DenseRetrievalError("Qdrant returned no usable dense results.")
            status["dense"] = "success"
        except Exception as exc:
            status["dense"] = "failed"
            logger.error(f"[HybridRetriever] {exc}")
            return HybridSearchOutcome([], status, str(exc), timings)
        finally:
            timings["dense_retrieval"] = (perf_counter() - started) * 1000

        try:
            started = perf_counter()
            lexical = self.bm25_search(query, top_k=prefetch_k)
            if not lexical:
                raise BM25RetrievalError(
                    "Elasticsearch returned no usable BM25 results."
                )
            status["bm25"] = "success"
        except Exception as exc:
            status["bm25"] = "failed"
            logger.error(f"[HybridRetriever] {exc}")
            return HybridSearchOutcome([], status, str(exc), timings)
        finally:
            timings["bm25_retrieval"] = (perf_counter() - started) * 1000

        try:
            started = perf_counter()
            fused = reciprocal_rank_fusion(
                dense,
                lexical,
                rrf_k=self.config.rrf_k,
                top_k=self.config.final_k,
            )
            status["rrf"] = "success"
        except FusionError as exc:
            status["rrf"] = "failed"
            logger.error(f"[HybridRetriever] {exc}")
            return HybridSearchOutcome([], status, str(exc), timings)
        finally:
            timings["rrf"] = (perf_counter() - started) * 1000

        if self.config.reranker_enabled:
            try:
                started = perf_counter()
                fused = self.reranker.rerank(query, fused, final_k)
                if not fused:
                    raise RerankingError("Cross-encoder returned no results.")
                status["reranker"] = "success"
            except Exception as exc:
                status["reranker"] = "failed"
                logger.error(f"[HybridRetriever] {exc}")
                return HybridSearchOutcome([], status, str(exc), timings)
            finally:
                timings["reranker"] = (perf_counter() - started) * 1000
        else:
            status["reranker"] = "skipped"
            fused = fused[:final_k]

        return HybridSearchOutcome(fused, status, stage_timings_ms=timings)
