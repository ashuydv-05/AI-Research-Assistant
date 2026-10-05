class StorageError(RuntimeError):
    """A persistence backend operation failed."""


class IngestionError(RuntimeError):
    """The complete paper ingestion transaction failed."""

    def __init__(self, message: str, result: dict | None = None) -> None:
        super().__init__(message)
        self.result = result
