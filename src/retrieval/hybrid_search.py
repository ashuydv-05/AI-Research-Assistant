"""Backward-compatible exports for existing agent and evaluation imports."""

from __future__ import annotations

import os

from elasticsearch import Elasticsearch
from loguru import logger
from qdrant_client import QdrantClient

from src.config.clients import request_tavily_api_key
from src.retrieval.dense.qdrant_retriever import QdrantRetriever
from src.retrieval.fusion.rrf import reciprocal_rank_fusion
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalConfig
from src.retrieval.lexical.elasticsearch_retriever import ElasticsearchRetriever
from src.retrieval.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.retrieval_result import (
    HybridSearchOutcome,
    RetrievalResult,
    SearchResult,
)


class VectorSearch(HybridRetriever):
    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        es_client: Elasticsearch | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        super().__init__(
            dense_retriever=QdrantRetriever(client=qdrant_client),
            lexical_retriever=ElasticsearchRetriever(client=es_client),
            config=config,
        )

    @property
    def rrf_k(self) -> int:
        return self.config.rrf_k

    def dense_search(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        return self.dense_retriever.search(query, top_k)

    def bm25_search(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        return self.lexical_retriever.search(query, top_k)

    def rrf_fuse(
        self,
        results_list: list[list[RetrievalResult]],
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        if len(results_list) != 2:
            raise ValueError("RRF requires exactly dense and BM25 result lists.")
        return reciprocal_rank_fusion(
            results_list[0], results_list[1], rrf_k=self.config.rrf_k, top_k=top_k
        )

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        return CrossEncoderReranker().rerank(
            query, results, top_k or self.config.rerank_top_k
        )


class WebSearch:
    def __init__(self, api_key: str | None = None, max_results: int = 5):
        logger.info(
    f"[WebSearch] Key source - "
    f"request={bool(request_tavily_api_key.get())}, "
    f"env={bool(os.getenv('TAVILY_API_KEY'))}"
)
        self.api_key = api_key or os.getenv("TAVILY_API_KEY") or request_tavily_api_key.get() 
        self.max_results = max_results
        self.client = None
        if self.api_key:
            try:
                from tavily import TavilyClient

                self.client = TavilyClient(api_key=self.api_key)
            except ImportError:
                logger.warning("Tavily client is not installed")

    def is_available(self) -> bool:
        return self.client is not None

    def search(self, query: str) -> list[RetrievalResult]:
        if not query.strip() or not self.client:
            return []
        try:
            response = self.client.search(query=query, max_results=self.max_results)
            return [
                RetrievalResult(
                    id=f"web-{index}",
                    content=item.get("content", ""),
                    title=item.get("title", "Web Result"),
                    score=float(item.get("score") or 0.0),
                    source="web",
                    metadata={"url": item.get("url", "")},
                )
                for index, item in enumerate(response.get("results", []))
                if item.get("content")
            ]
        except Exception as exc:
            logger.error(f"[WebSearch] Search failed: {exc}")
            return []


__all__ = [
    "HybridSearchOutcome",
    "RetrievalConfig",
    "SearchResult",
    "VectorSearch",
    "WebSearch",
]
