from __future__ import annotations

from typing import Any

from src.config.settings import settings
from src.ingestion.models import PaperChunk
from src.storage.errors import IngestionError


class DenseEmbedder:
    _model: Any = None

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(settings.embedding.dense_model)
        return cls._model

    def embed(self, chunks: list[PaperChunk]) -> list[list[float]]:
        try:
            model = self.model or self._get_model()
            vectors = model.encode(
                [chunk.content for chunk in chunks],
                batch_size=32,
                show_progress_bar=False,
            )
            result = [vector.tolist() for vector in vectors]
        except Exception as exc:
            raise IngestionError("Dense embedding generation failed.") from exc
        if len(result) != len(chunks):
            raise IngestionError("Embedding count does not match chunk count.")
        if any(len(vector) != settings.embedding.dimension for vector in result):
            raise IngestionError(
                f"Embeddings must have {settings.embedding.dimension} dimensions."
            )
        return result
