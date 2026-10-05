from __future__ import annotations

import uuid
from typing import Any, cast

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from src.config.settings import settings
from src.data.pdf_processor import extract_chunk_meta
from src.ingestion.models import PaperChunk, PaperMetadata
from src.storage.errors import IngestionError


CHUNK_NAMESPACE = uuid.UUID("3b9bc6a0-9e8d-4f7a-bbc5-8c43943167a4")


class DocumentChunker:
    def __init__(self, chunker=None) -> None:
        self.chunker = chunker

    def chunk(self, document, metadata: PaperMetadata) -> list[PaperChunk]:
        try:
            chunker = self.chunker or HybridChunker(
                tokenizer=cast(Any, settings.embedding.dense_model)
            )
            raw_chunks = list(chunker.chunk(document))
        except Exception as exc:
            raise IngestionError("Document chunking failed.") from exc

        chunks: list[PaperChunk] = []
        for index, raw_chunk in enumerate(raw_chunks):
            content = str(getattr(raw_chunk, "text", "")).strip()
            if not content:
                continue
            chunk_meta = extract_chunk_meta(raw_chunk)
            headings = chunk_meta.get("headings") or []
            chunk_id = str(
                uuid.uuid5(CHUNK_NAMESPACE, f"{metadata.paper_id}:{index}")
            )
            chunks.append(
                PaperChunk(
                    id=chunk_id,
                    paper_id=metadata.paper_id,
                    title=metadata.title,
                    content=content,
                    chunk_index=index,
                    authors=metadata.authors,
                    year=metadata.year,
                    section=headings[-1] if headings else "Unknown",
                    content_type=chunk_meta.get("content_type"),
                    page_info=chunk_meta.get("page_info"),
                )
            )
        if not chunks:
            raise IngestionError("The PDF produced no usable chunks.")
        return chunks
