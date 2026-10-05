from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.config.settings import settings
from src.retrieval.errors import RerankingError
from src.retrieval.retrieval_result import RetrievalResult


class CrossEncoderReranker:
    _model: Any = None

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from sentence_transformers import CrossEncoder

            cls._model = CrossEncoder(settings.reranker.model)
        return cls._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not results:
            raise RerankingError("Reranking requires retrieval results.")
        try:
            model = self.model or self._get_model()
            candidates = results[: settings.reranker.candidate_k]
            pairs = [(query, f"{item.title}. {item.content}"[:2000]) for item in candidates]
            scores = model.predict(pairs)
            ranked = [
                replace(item, score=float(score), reranker_score=float(score))
                for item, score in zip(candidates, scores)
            ]
            ranked.sort(key=lambda item: item.reranker_score or 0.0, reverse=True)
            return ranked[:top_k]
        except RerankingError:
            raise
        except Exception as exc:
            raise RerankingError("Cross-encoder reranking failed.") from exc
