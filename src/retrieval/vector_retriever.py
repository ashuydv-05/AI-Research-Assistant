from src.retrieval.base import BaseRetriever
from src.retrieval.dense.qdrant_retriever import QdrantRetriever
from src.retrieval.retrieval_result import SearchResult

'''
This is My Vector Retrieval File (Dense retrieval using Qdrant)
'''
class VectorRetriever(BaseRetriever):
    """Semantic vector-only retriever using Qdrant dense embeddings."""

    def __init__(self, searcher=None):
        self.searcher = searcher if searcher is not None else QdrantRetriever()

    @property
    def name(self) -> str:
        return "vector"

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        if hasattr(self.searcher, "dense_search"):
            return self.searcher.dense_search(query=query, top_k=top_k)
        return self.searcher.search(query=query, top_k=top_k)
