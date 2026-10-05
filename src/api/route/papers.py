from __future__ import annotations

import json
import os
import shutil
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.api.schemas.papers import (
    DeletePaperResponse,
    IngestionResponse,
    PaperListResponse,
    PaperResponse,
)
from src.services.paper_ingestion_service import PaperIngestionService
from src.storage.errors import IngestionError, StorageError


router = APIRouter(prefix="/papers", tags=["papers"])


@lru_cache(maxsize=1)
def get_paper_service() -> PaperIngestionService:
    return PaperIngestionService()


def _parse_json_object(value: str | None, field_name: str) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(400, f"{field_name} must be a JSON object.")
    return parsed


def _parse_json_list(value: str | None, field_name: str) -> list[dict]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"{field_name} must be valid JSON.") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise HTTPException(400, f"{field_name} must be a JSON array of objects.")
    return parsed


@router.post("", response_model=IngestionResponse, status_code=status.HTTP_201_CREATED)
def upload_paper(
    file: UploadFile = File(...),
    metadata: str | None = Form(None),
    golden_questions: str | None = Form(None),
) -> IngestionResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "A PDF file is required.")
    parsed_metadata = _parse_json_object(metadata, "metadata")
    parsed_metadata.setdefault("paper_id", Path(file.filename).stem)
    parsed_questions = _parse_json_list(golden_questions, "golden_questions")

    incoming_dir = Path("uploads") / ".incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    temporary = incoming_dir / f"{uuid.uuid4()}.pdf"
    try:
        with temporary.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)
        max_upload_bytes = int(os.getenv("MAX_PDF_UPLOAD_MB", "50")) * 1024 * 1024
        if temporary.stat().st_size > max_upload_bytes:
            raise HTTPException(413, "The PDF exceeds the configured upload limit.")
        result = get_paper_service().ingest_file(
            temporary,
            metadata=parsed_metadata,
            golden_questions=parsed_questions,
        )
        return IngestionResponse.model_validate(result.to_dict())
    except IngestionError as exc:
        raise HTTPException(502, exc.result or {"status": "failed", "error": str(exc)}) from exc
    except (StorageError, ValueError) as exc:
        raise HTTPException(502, str(exc)) from exc
    finally:
        if temporary.exists():
            temporary.unlink()


@router.get("", response_model=PaperListResponse)
def list_papers() -> PaperListResponse:
    papers = [PaperResponse.model_validate(item) for item in get_paper_service().list_papers()]
    return PaperListResponse(papers=papers, total=len(papers))


@router.get("/{paper_id}", response_model=PaperResponse)
def get_paper(paper_id: str) -> PaperResponse:
    try:
        record = get_paper_service().get_paper(paper_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if record is None:
        raise HTTPException(404, "Paper not found.")
    return PaperResponse.model_validate(record)


@router.delete("/{paper_id}", response_model=DeletePaperResponse)
def delete_paper(paper_id: str) -> DeletePaperResponse:
    try:
        result = get_paper_service().delete_paper(paper_id)
        return DeletePaperResponse.model_validate(result)
    except KeyError as exc:
        raise HTTPException(404, "Paper not found.") from exc
    except (StorageError, ValueError) as exc:
        raise HTTPException(502, str(exc)) from exc


@router.post("/{paper_id}/reindex", response_model=IngestionResponse)
def reindex_paper(paper_id: str) -> IngestionResponse:
    try:
        result = get_paper_service().reindex_paper(paper_id)
        return IngestionResponse.model_validate(result.to_dict())
    except KeyError as exc:
        raise HTTPException(404, "Paper not found.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, "The source PDF is no longer available.") from exc
    except IngestionError as exc:
        raise HTTPException(502, exc.result or {"status": "failed", "error": str(exc)}) from exc
    except (StorageError, ValueError) as exc:
        raise HTTPException(502, str(exc)) from exc
