from __future__ import annotations

from qdrant_client import QdrantClient, models

from src.config.clients import get_qdrant_client
from src.config.settings import settings
from src.ingestion.models import PaperChunk
from src.storage.errors import StorageError


class QdrantStore:
    def __init__(self, client: QdrantClient | None = None) -> None:
        self.client = client or get_qdrant_client()
        self.collection = settings.qdrant.collection

    def ensure_collection(self) -> None:
        try:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config={
                        settings.embedding.dense_vector_name: models.VectorParams(
                            size=settings.embedding.dimension,
                            distance=models.Distance.COSINE,
                        )
                    },
                )
                for field_name, field_schema in (
                    ("paper_id", models.PayloadSchemaType.KEYWORD),
                    ("section", models.PayloadSchemaType.KEYWORD),
                    ("year", models.PayloadSchemaType.INTEGER),
                ):
                    self.client.create_payload_index(
                        collection_name=self.collection,
                        field_name=field_name,
                        field_schema=field_schema,
                    )
        except Exception as exc:
            raise StorageError("Qdrant collection check failed.") from exc

    def upsert(self, chunks: list[PaperChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise StorageError("Qdrant chunk/vector counts do not match.")
        self.ensure_collection()
        points = [
            models.PointStruct(
                id=chunk.id,
                vector={settings.embedding.dense_vector_name: vector},
                payload=chunk.payload(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )
        except Exception as exc:
            raise StorageError("Qdrant upsert failed.") from exc

    def delete_paper(self, paper_id: str) -> None:
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="paper_id",
                                match=models.MatchValue(value=paper_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
        except Exception as exc:
            raise StorageError("Qdrant paper deletion failed.") from exc
