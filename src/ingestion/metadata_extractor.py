from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.pdf_processor import extract_title, extract_year
from src.ingestion.models import PaperMetadata


class MetadataExtractor:
    def extract(
        self,
        document,
        pdf_path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> PaperMetadata:
        overrides = overrides or {}
        path = Path(pdf_path)
        paper_id = str(overrides.get("paper_id") or path.stem).strip()
        title = str(overrides.get("title") or extract_title(document) or paper_id).strip()
        authors = overrides.get("authors") or []
        if isinstance(authors, str):
            authors = [item.strip() for item in authors.split(",") if item.strip()]
        year = overrides.get("year")
        if year is None:
            year = extract_year(paper_id)
        known = {"paper_id", "title", "authors", "year"}
        return PaperMetadata(
            paper_id=paper_id,
            title=title,
            authors=list(authors),
            year=int(year) if year is not None else None,
            source_file=str(path),
            extra={key: value for key, value in overrides.items() if key not in known},
        )
