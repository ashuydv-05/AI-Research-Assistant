from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app
from src.ingestion.models import IngestionResult


class FakePaperService:
    def ingest_file(self, source_path, metadata=None, golden_questions=None):
        assert Path(source_path).exists()
        assert metadata["paper_id"] == "paper-123"
        assert golden_questions == [{"question": "What is new?"}]
        return IngestionResult(
            paper_id="paper-123",
            title="A Paper",
            chunks=4,
            qdrant="success",
            elasticsearch="success",
            status="completed",
            source_file="uploads/paper-123.pdf",
        )

    def list_papers(self):
        return [
            {
                "paper_id": "paper-123",
                "title": "A Paper",
                "authors": [],
                "year": 2026,
                "chunks": 4,
                "status": "completed",
                "qdrant": "success",
                "elasticsearch": "success",
                "source_file": "uploads/paper-123.pdf",
            }
        ]


def test_upload_and_list_papers(monkeypatch):
    service = FakePaperService()
    monkeypatch.setattr("src.api.route.papers.get_paper_service", lambda: service)

    with TestClient(app) as client:
        upload = client.post(
            "/api/papers",
            files={"file": ("paper-123.pdf", b"%PDF-test", "application/pdf")},
            data={
                "metadata": '{"paper_id":"paper-123","title":"A Paper"}',
                "golden_questions": '[{"question":"What is new?"}]',
            },
        )
        listing = client.get("/api/papers")

    assert upload.status_code == 201
    assert upload.json()["status"] == "completed"
    assert upload.json()["qdrant"] == "success"
    assert upload.json()["elasticsearch"] == "success"
    assert listing.status_code == 200
    assert listing.json()["total"] == 1
