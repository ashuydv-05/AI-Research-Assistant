from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.ingestion.models import PaperChunk, PaperMetadata
from src.storage.errors import IngestionError, StorageError


def build_pipeline(tmp_path: Path):
    loader = MagicMock()
    loader.load.return_value = object()
    parser = MagicMock()
    parser.parse.return_value = object()
    metadata_extractor = MagicMock()
    metadata_extractor.extract.return_value = PaperMetadata(
        paper_id="paper-1", title="Test Paper", source_file=str(tmp_path / "paper.pdf")
    )
    chunk = PaperChunk(
        id="00000000-0000-5000-8000-000000000001",
        paper_id="paper-1",
        title="Test Paper",
        content="Evidence",
        chunk_index=0,
    )
    chunker = MagicMock()
    chunker.chunk.return_value = [chunk]
    embedder = MagicMock()
    embedder.embed.return_value = [[0.0] * 384]
    qdrant = MagicMock()
    elasticsearch = MagicMock()
    paper_store = MagicMock()
    pipeline = IngestionPipeline(
        loader=loader,
        parser=parser,
        metadata_extractor=metadata_extractor,
        chunker=chunker,
        embedder=embedder,
        qdrant_store=qdrant,
        elasticsearch_store=elasticsearch,
        paper_store=paper_store,
    )
    return pipeline, qdrant, elasticsearch, paper_store


def test_ingestion_is_complete_only_after_both_writes(tmp_path):
    pipeline, qdrant, elasticsearch, paper_store = build_pipeline(tmp_path)
    result = pipeline.ingest(tmp_path / "paper.pdf")

    assert result.status == "completed"
    assert result.qdrant == "success"
    assert result.elasticsearch == "success"
    qdrant.upsert.assert_called_once()
    elasticsearch.index_chunks.assert_called_once()
    assert paper_store.upsert.call_args.args[0]["status"] == "completed"


def test_elasticsearch_failure_rolls_back_qdrant(tmp_path):
    pipeline, qdrant, elasticsearch, paper_store = build_pipeline(tmp_path)
    elasticsearch.index_chunks.side_effect = StorageError("ES failed")

    with pytest.raises(IngestionError) as captured:
        pipeline.ingest(tmp_path / "paper.pdf")

    qdrant.delete_paper.assert_called_once_with("paper-1")
    record = paper_store.upsert.call_args.args[0]
    assert record["status"] == "failed"
    assert record["qdrant"] == "rolled_back"
    assert record["elasticsearch"] == "failed"
    assert captured.value.result["status"] == "failed"
    assert captured.value.result["qdrant"] == "rolled_back"


def test_qdrant_failure_never_calls_elasticsearch(tmp_path):
    pipeline, qdrant, elasticsearch, paper_store = build_pipeline(tmp_path)
    qdrant.upsert.side_effect = StorageError("Qdrant failed")

    with pytest.raises(IngestionError) as captured:
        pipeline.ingest(tmp_path / "paper.pdf")

    elasticsearch.index_chunks.assert_not_called()
    record = paper_store.upsert.call_args.args[0]
    assert record["qdrant"] == "failed"
    assert record["elasticsearch"] == "not_run"
    assert captured.value.result["qdrant"] == "failed"
