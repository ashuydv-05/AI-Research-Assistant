from __future__ import annotations

from pydantic import BaseModel, Field


class PaperResponse(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    chunks: int = 0
    status: str
    qdrant: str
    elasticsearch: str
    source_file: str | None = None
    error: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)


class PaperListResponse(BaseModel):
    papers: list[PaperResponse]
    total: int


class IngestionResponse(PaperResponse):
    pass


class DeletePaperResponse(BaseModel):
    paper_id: str
    status: str
