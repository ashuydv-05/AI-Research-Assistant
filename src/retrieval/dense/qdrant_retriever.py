from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from src.config.clients import get_qdrant_client
from src.config.settings import settings
from src.retrieval.errors import DenseRetrievalError
from src.retrieval.retrieval_result import RetrievalResult


class QdrantRetriever:
    _embedding_model: Any = None

    def __init__(
        self,
        client: QdrantClient | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.client = client if client is not None else get_qdrant_client()
        self.embedder = embedder

    @classmethod
    def _get_embedding_model(cls):
        if cls._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            cls._embedding_model = SentenceTransformer(settings.embedding.dense_model)
        return cls._embedding_model

    def search(self, query: str, top_k: int = 50) -> list[RetrievalResult]:
        if not query.strip():
            raise DenseRetrievalError("Dense retrieval requires a non-empty query.")
        try:
            model = self.embedder or self._get_embedding_model()
            query_vector = model.encode(query).tolist()
            response = self.client.query_points(
                collection_name=settings.qdrant.collection,
                query=query_vector,
                using=settings.embedding.dense_vector_name,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            print("\n========== QDRANT EXACT ERROR ==========")
            print(f"Error type: {type(exc).__name__}")
            print(f"Error message: {exc}")
            print("========================================\n")

            raise
        

        results: list[RetrievalResult] = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            content = str(payload.get("content", "")).strip()
            if not content:
                continue
            results.append(
                RetrievalResult(
                    id=point.id,
                    content=content,
                    title=payload.get("title") or payload.get("section") or "Unknown",
                    score=float(point.score or 0.0),
                    source="dense",
                    metadata=dict(payload),
                    retrieval_sources=("dense",),
                    dense_rank=rank,
                )
            )
        if not results:
            raise DenseRetrievalError("Qdrant returned no usable dense results.")
        return results
