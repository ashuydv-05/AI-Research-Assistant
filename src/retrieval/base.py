from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retrieval.retrieval_result import SearchResult
else:
    try:
        from src.retrieval.retrieval_result import SearchResult
    except ImportError:
        SearchResult = object


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Retrieve top_k documents for the given query."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the retrieval strategy."""
        pass
