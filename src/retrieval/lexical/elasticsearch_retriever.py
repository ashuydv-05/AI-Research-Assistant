from __future__ import annotations

from elasticsearch import Elasticsearch

from src.config.clients import get_es_client
from src.config.settings import settings
from src.retrieval.errors import BM25RetrievalError
from src.retrieval.retrieval_result import RetrievalResult


class ElasticsearchRetriever:
    def __init__(self, client: Elasticsearch | None = None) -> None:
        self.client = client if client is not None else get_es_client()

    def search(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        if not query.strip():
            raise BM25RetrievalError("BM25 retrieval requires a non-empty query.")
        if self.client is None:
            raise BM25RetrievalError("Elasticsearch is not configured.")

        es_query = {
            "multi_match": {
                "query": query,
                "fields": ["title^3.0", "content^2.0"],
                "type": "best_fields",
                "fuzziness": "AUTO",
            }
        }
        try:
            response = self.client.search(
                index=settings.elasticsearch.index,
                query=es_query,
                size=top_k,
                request_timeout=settings.elasticsearch.request_timeout,
            )
        except Exception as exc:
            raise BM25RetrievalError("Elasticsearch BM25 retrieval failed.") from exc

        results: list[RetrievalResult] = []
        for rank, hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
            source = hit.get("_source") or {}
            content = str(source.get("content", "")).strip()
            if not content:
                continue
            results.append(
                RetrievalResult(
                    id=hit.get("_id"),
                    content=content,
                    title=source.get("title") or source.get("section") or "Unknown",
                    score=float(hit.get("_score") or 0.0),
                    source="bm25",
                    metadata=dict(source),
                    retrieval_sources=("bm25",),
                    bm25_rank=rank,
                )
            )
        if not results:
            raise BM25RetrievalError("Elasticsearch returned no usable BM25 results.")
        return results
