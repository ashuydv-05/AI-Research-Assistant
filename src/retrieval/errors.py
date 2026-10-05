class RetrievalError(RuntimeError):
    """Base error for a retrieval pipeline failure."""


class DenseRetrievalError(RetrievalError):
    """Qdrant dense retrieval failed or returned no usable evidence."""


class BM25RetrievalError(RetrievalError):
    """Elasticsearch lexical retrieval failed or returned no usable evidence."""


class FusionError(RetrievalError):
    """Reciprocal Rank Fusion could not produce usable results."""


class RerankingError(RetrievalError):
    """An enabled reranker failed."""
