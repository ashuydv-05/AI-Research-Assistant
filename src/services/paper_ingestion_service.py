from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.models import IngestionResult
from src.storage.elasticsearch_store import ElasticsearchStore
from src.storage.golden_dataset_store import GoldenDatasetStore
from src.storage.paper_store import PaperStore
from src.storage.qdrant_store import QdrantStore


def normalize_paper_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    if not normalized:
        raise ValueError("A valid paper ID is required.")
    return normalized[:120]


class PaperIngestionService:
    def __init__(
        self,
        pipeline: IngestionPipeline | None = None,
        paper_store: PaperStore | None = None,
        qdrant_store: QdrantStore | None = None,
        elasticsearch_store: ElasticsearchStore | None = None,
        golden_store: GoldenDatasetStore | None = None,
        upload_dir: str | Path = "uploads",
    ) -> None:
        self.paper_store = paper_store or PaperStore()
        self.qdrant_store = qdrant_store or QdrantStore()
        self.elasticsearch_store = elasticsearch_store or ElasticsearchStore()
        self.pipeline = pipeline or IngestionPipeline(
            qdrant_store=self.qdrant_store,
            elasticsearch_store=self.elasticsearch_store,
            paper_store=self.paper_store,
        )
        self.golden_store = golden_store or GoldenDatasetStore()
        self.upload_dir = Path(upload_dir)

    def ingest_file(
        self,
        source_path: str | Path,
        metadata: dict[str, Any] | None = None,
        golden_questions: list[dict[str, Any]] | None = None,
    ) -> IngestionResult:
        source = Path(source_path)
        metadata = dict(metadata or {})
        paper_id = normalize_paper_id(str(metadata.get("paper_id") or source.stem))
        metadata["paper_id"] = paper_id
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        destination = self.upload_dir / f"{paper_id}.pdf"
        backup = self.upload_dir / f".{paper_id}.pdf.backup"
        copied = source.resolve() != destination.resolve()
        if copied:
            if destination.exists():
                shutil.copy2(destination, backup)
            shutil.copy2(source, destination)
        try:
            result = self.pipeline.ingest(destination, metadata)
        except Exception:
            if copied:
                if backup.exists():
                    shutil.move(str(backup), str(destination))
                elif destination.exists():
                    destination.unlink()
            raise
        finally:
            if backup.exists():
                backup.unlink()
        self.golden_store.save_for_paper(paper_id, golden_questions)
        return result

    def list_papers(self) -> list[dict[str, Any]]:
        return self.paper_store.list()

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        return self.paper_store.get(normalize_paper_id(paper_id))

    def delete_paper(self, paper_id: str) -> dict[str, str]:
        paper_id = normalize_paper_id(paper_id)
        if self.paper_store.get(paper_id) is None:
            raise KeyError(paper_id)
        self.qdrant_store.delete_paper(paper_id)
        self.elasticsearch_store.delete_paper(paper_id)
        self.paper_store.mark_deleted(paper_id)
        self.golden_store.delete_for_paper(paper_id)
        uploaded = self.upload_dir / f"{paper_id}.pdf"
        if uploaded.exists():
            uploaded.unlink()
        return {"paper_id": paper_id, "status": "deleted"}

    def reindex_paper(self, paper_id: str) -> IngestionResult:
        paper_id = normalize_paper_id(paper_id)
        record = self.paper_store.get(paper_id)
        if record is None:
            raise KeyError(paper_id)
        source = Path(record.get("source_file") or self.upload_dir / f"{paper_id}.pdf")
        if not source.exists():
            raise FileNotFoundError(source)
        metadata = {
            key: record.get(key)
            for key in ("paper_id", "title", "authors", "year")
            if record.get(key) is not None
        }
        return self.pipeline.ingest(source, metadata)
