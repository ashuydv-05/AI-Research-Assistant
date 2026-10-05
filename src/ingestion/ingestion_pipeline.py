from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

from loguru import logger

from src.ingestion.chunker import DocumentChunker
from src.ingestion.document_parser import DocumentParser
from src.ingestion.embedder import DenseEmbedder
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.models import IngestionResult
from src.ingestion.pdf_loader import PDFLoader
from src.storage.elasticsearch_store import ElasticsearchStore
from src.storage.errors import IngestionError
from src.storage.paper_store import PaperStore
from src.storage.qdrant_store import QdrantStore


class IngestionPipeline:
    def __init__(
        self,
        loader: PDFLoader | None = None,
        parser: DocumentParser | None = None,
        metadata_extractor: MetadataExtractor | None = None,
        chunker: DocumentChunker | None = None,
        embedder: DenseEmbedder | None = None,
        qdrant_store: QdrantStore | None = None,
        elasticsearch_store: ElasticsearchStore | None = None,
        paper_store: PaperStore | None = None,
    ) -> None:
        self.loader = loader or PDFLoader()
        self.parser = parser or DocumentParser()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()
        self.chunker = chunker or DocumentChunker()
        self.embedder = embedder or DenseEmbedder()
        self.qdrant_store = qdrant_store or QdrantStore()
        self.elasticsearch_store = elasticsearch_store or ElasticsearchStore()
        self.paper_store = paper_store or PaperStore()

    def ingest(
        self,
        pdf_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        path = Path(pdf_path)
        total_started = perf_counter()
        timings: dict[str, float] = {}

        started = perf_counter()
        parsed = self.parser.parse(self.loader.load(path))
        timings["pdf_processing"] = (perf_counter() - started) * 1000
        paper = self.metadata_extractor.extract(parsed, path, metadata)

        started = perf_counter()
        chunks = self.chunker.chunk(parsed, paper)
        timings["chunking"] = (perf_counter() - started) * 1000

        started = perf_counter()
        vectors = self.embedder.embed(chunks)
        timings["embedding"] = (perf_counter() - started) * 1000

        qdrant_status = "not_run"
        elasticsearch_status = "not_run"
        try:
            started = perf_counter()
            self.qdrant_store.upsert(chunks, vectors)
            timings["qdrant"] = (perf_counter() - started) * 1000
            qdrant_status = "success"
        except Exception as exc:
            timings["qdrant"] = (perf_counter() - started) * 1000
            timings["total"] = (perf_counter() - total_started) * 1000
            qdrant_status = "failed"
            result = IngestionResult(
                paper_id=paper.paper_id,
                title=paper.title,
                chunks=len(chunks),
                qdrant=qdrant_status,
                elasticsearch=elasticsearch_status,
                status="failed",
                source_file=str(path),
                error=str(exc),
                timings_ms=timings,
            )
            self.paper_store.upsert(
                {**result.to_dict(), "authors": paper.authors, "year": paper.year}
            )
            logger.exception(f"[Ingestion] Qdrant failed for paper_id={paper.paper_id}")
            raise IngestionError(
                "Paper ingestion failed; both indexes are required.",
                result=result.to_dict(),
            ) from exc

        try:
            started = perf_counter()
            self.elasticsearch_store.index_chunks(chunks)
            timings["elasticsearch"] = (perf_counter() - started) * 1000
            elasticsearch_status = "success"
        except Exception as exc:
            timings["elasticsearch"] = (perf_counter() - started) * 1000
            elasticsearch_status = "failed"
            logger.exception(f"[Ingestion] Failed for paper_id={paper.paper_id}")
            if qdrant_status == "success":
                try:
                    self.qdrant_store.delete_paper(paper.paper_id)
                    qdrant_status = "rolled_back"
                except Exception:
                    qdrant_status = "rollback_failed"
                    logger.exception("[Ingestion] Qdrant rollback failed")
            result = IngestionResult(
                paper_id=paper.paper_id,
                title=paper.title,
                chunks=len(chunks),
                qdrant=qdrant_status,
                elasticsearch=elasticsearch_status,
                status="failed",
                source_file=str(path),
                error=str(exc),
                timings_ms={
                    **timings,
                    "total": (perf_counter() - total_started) * 1000,
                },
            )
            self.paper_store.upsert({**result.to_dict(), "authors": paper.authors, "year": paper.year})
            raise IngestionError(
                "Paper ingestion failed; both indexes are required.",
                result=result.to_dict(),
            ) from exc

        timings["total"] = (perf_counter() - total_started) * 1000
        result = IngestionResult(
            paper_id=paper.paper_id,
            title=paper.title,
            chunks=len(chunks),
            qdrant=qdrant_status,
            elasticsearch=elasticsearch_status,
            status="completed",
            source_file=str(path),
            timings_ms=timings,
        )
        self.paper_store.upsert({**result.to_dict(), "authors": paper.authors, "year": paper.year})
        logger.info(
            f"[Ingestion] Completed paper_id={paper.paper_id}, chunks={len(chunks)}, "
            f"timings_ms={{{', '.join(f'{key}: {value:.1f}' for key, value in timings.items())}}}"
        )
        return result
