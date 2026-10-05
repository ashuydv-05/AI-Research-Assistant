from pathlib import Path

from src.data.pdf_processor import create_converter
from src.storage.errors import IngestionError


class PDFLoader:
    def __init__(self, converter=None) -> None:
        self.converter = converter

    def load(self, pdf_path: str | Path):
        path = Path(pdf_path)
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise IngestionError("A valid PDF file is required.")
        if path.stat().st_size == 0:
            raise IngestionError("The uploaded PDF is empty.")
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise IngestionError("The uploaded file is not a valid PDF.")
        try:
            converter = self.converter or create_converter()
            return converter.convert(str(path)).document
        except Exception as exc:
            raise IngestionError("PDF processing failed.") from exc
