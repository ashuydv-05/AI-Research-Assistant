#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.paper_ingestion_service import PaperIngestionService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a PDF into Qdrant and Elasticsearch Cloud."
    )
    parser.add_argument("pdf", type=Path, help="Path to a research-paper PDF")
    parser.add_argument("--paper-id", help="Stable paper ID (defaults to filename)")
    parser.add_argument("--title", help="Optional title override")
    parser.add_argument("--year", type=int, help="Optional publication year")
    parser.add_argument(
        "--golden-questions",
        type=Path,
        help="Optional JSON file containing a list of golden questions",
    )
    args = parser.parse_args()

    metadata = {
        key: value
        for key, value in {
            "paper_id": args.paper_id,
            "title": args.title,
            "year": args.year,
        }.items()
        if value is not None
    }
    questions = None
    if args.golden_questions:
        questions = json.loads(args.golden_questions.read_text(encoding="utf-8"))
        if not isinstance(questions, list):
            raise ValueError("Golden questions must be a JSON array.")

    result = PaperIngestionService().ingest_file(
        args.pdf,
        metadata=metadata,
        golden_questions=questions,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
