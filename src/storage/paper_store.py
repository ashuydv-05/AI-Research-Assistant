from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class PaperStore:
    def __init__(
        self,
        registry_path: str | Path = "data/processed/papers.json",
        corpus_path: str | Path = "data/processed/arxiv_documents.jsonl",
    ) -> None:
        self.registry_path = Path(registry_path)
        self.corpus_path = Path(corpus_path)
        self._lock = threading.Lock()
        self._legacy_cache: dict[str, dict[str, Any]] | None = None

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        if not self.registry_path.exists():
            return {}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_registry(self, records: dict[str, dict[str, Any]]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temporary.replace(self.registry_path)

    def _load_legacy(self) -> dict[str, dict[str, Any]]:
        if self._legacy_cache is not None:
            return self._legacy_cache
        records: dict[str, dict[str, Any]] = {}
        if self.corpus_path.exists():
            with self.corpus_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    paper_id = str(item.get("paper_id") or "").strip()
                    if not paper_id:
                        continue
                    record = records.setdefault(
                        paper_id,
                        {
                            "paper_id": paper_id,
                            "title": item.get("title") or paper_id,
                            "authors": [],
                            "year": item.get("year"),
                            "chunks": 0,
                            "status": "legacy",
                            "qdrant": "existing",
                            "elasticsearch": "existing",
                            "source_file": str(Path("data/raw") / f"{paper_id}.pdf"),
                        },
                    )
                    record["chunks"] += 1
        self._legacy_cache = records
        return records

    def list(self) -> list[dict[str, Any]]:
        records = {**self._load_legacy(), **self._load_registry()}
        return sorted(
            (item for item in records.values() if item.get("status") != "deleted"),
            key=lambda item: str(item.get("paper_id")),
        )

    def get(self, paper_id: str) -> dict[str, Any] | None:
        registry = self._load_registry()
        record = registry.get(paper_id) or self._load_legacy().get(paper_id)
        if record and record.get("status") != "deleted":
            return record
        return None

    def upsert(self, record: dict[str, Any]) -> None:
        paper_id = str(record["paper_id"])
        with self._lock:
            records = self._load_registry()
            records[paper_id] = record
            self._write_registry(records)

    def mark_deleted(self, paper_id: str) -> None:
        with self._lock:
            records = self._load_registry()
            existing = records.get(paper_id) or self._load_legacy().get(paper_id) or {
                "paper_id": paper_id
            }
            records[paper_id] = {**existing, "status": "deleted"}
            self._write_registry(records)
