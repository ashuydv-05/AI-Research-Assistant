from __future__ import annotations

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from src.config.clients import get_es_client
from src.config.settings import settings
from src.ingestion.models import PaperChunk
from src.storage.errors import StorageError


class ElasticsearchStore:
    def __init__(self, client: Elasticsearch | None = None) -> None:
        self.client = client if client is not None else get_es_client()
        self.index = settings.elasticsearch.index

    def _require_client(self) -> Elasticsearch:
        if self.client is None:
            raise StorageError("Elasticsearch Cloud is not configured.")
        return self.client

    def ensure_index(self) -> None:
        client = self._require_client()
        try:
            if not client.indices.exists(index=self.index):
                client.indices.create(
                    index=self.index,
                    mappings={
                        "properties": {
                            "paper_id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "authors": {"type": "keyword"},
                            "year": {"type": "integer"},
                            "section": {"type": "keyword"},
                            "content_type": {"type": "keyword"},
                            "page_info": {"type": "integer"},
                            "chunk_index": {"type": "integer"},
                            "content": {"type": "text"},
                        }
                    },
                )
        except Exception as exc:
            raise StorageError("Elasticsearch index check failed.") from exc

    def index_chunks(self, chunks: list[PaperChunk]) -> None:
        self.ensure_index()
        actions = [
            {"_index": self.index, "_id": chunk.id, "_source": chunk.payload()}
            for chunk in chunks
        ]
        try:
            bulk(
                self._require_client(),
                actions,
                refresh="wait_for",
                raise_on_error=True,
                request_timeout=settings.elasticsearch.request_timeout,
            )
        except Exception as exc:
            raise StorageError("Elasticsearch chunk indexing failed.") from exc

    def delete_paper(self, paper_id: str) -> None:
        try:
            self._require_client().delete_by_query(
                index=self.index,
                query={"term": {"paper_id": paper_id}},
                refresh=True,
                conflicts="proceed",
            )
        except Exception as exc:
            raise StorageError("Elasticsearch paper deletion failed.") from exc
