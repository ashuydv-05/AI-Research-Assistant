class DocumentParser:
    """Normalizes a loader result before metadata extraction and chunking."""

    def parse(self, document):
        if document is None or not hasattr(document, "iterate_items"):
            raise ValueError("The parsed PDF document is invalid.")
        return document
