from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GoldenDatasetStore:
    def __init__(self, root: str | Path = "data/evaluation") -> None:
        self.root = Path(root)
        self.global_path = self.root / "golden_dataset.json"
        self.paper_dir = self.root / "papers"

    def save_for_paper(
        self, paper_id: str, questions: list[dict[str, Any]] | None
    ) -> None:
        if not questions:
            return
        normalized = []
        for index, question in enumerate(questions, start=1):
            item = dict(question)
            item.setdefault("id", f"{paper_id}-q{index:03d}")
            item.setdefault("expected_paper_ids", [paper_id])
            normalized.append(item)

        self.paper_dir.mkdir(parents=True, exist_ok=True)
        paper_path = self.paper_dir / f"{paper_id}.json"
        paper_path.write_text(
            json.dumps({"paper_id": paper_id, "questions": normalized}, indent=2),
            encoding="utf-8",
        )

        existing: dict[str, Any] = {"questions": []}
        if self.global_path.exists():
            try:
                loaded = json.loads(self.global_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except json.JSONDecodeError:
                pass
        old_questions = existing.get("questions") or []
        old_questions = [
            item for item in old_questions if paper_id not in item.get("expected_paper_ids", [])
        ]
        existing["questions"] = old_questions + normalized
        self.global_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def delete_for_paper(self, paper_id: str) -> None:
        paper_path = self.paper_dir / f"{paper_id}.json"
        if paper_path.exists():
            paper_path.unlink()
        if not self.global_path.exists():
            return
        try:
            data = json.loads(self.global_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        data["questions"] = [
            item
            for item in data.get("questions", [])
            if paper_id not in item.get("expected_paper_ids", [])
        ]
        self.global_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
