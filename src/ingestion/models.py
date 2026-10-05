from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PaperMetadata:
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    source_file: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PaperChunk:
    id: str
    paper_id: str
    title: str
    content: str
    chunk_index: int
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    section: str = "Unknown"
    content_type: str | None = None
    page_info: int | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IngestionResult:
    paper_id: str
    title: str
    chunks: int
    qdrant: str
    elasticsearch: str
    status: str
    source_file: str
    error: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
