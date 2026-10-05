import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class QdrantSettings:
    url: str = os.getenv('QDRANT_URL', 'http://localhost:6333')
    api_key: str | None = os.getenv('QDRANT_API_KEY', None)
    collection: str = os.getenv('QDRANT_COLLECTION', 'arxiv_papers')

@dataclass(frozen=True)
class ElasticsearchSettings:
    url: str = os.getenv('ES_URL', '')
    index: str = os.getenv('ES_INDEX', 'arxiv_papers')
    api_key: str | None = os.getenv('ES_API_KEY', None)
    request_timeout: float = float(os.getenv('ES_REQUEST_TIMEOUT', '10'))

@dataclass(frozen=True)
class EmbeddingSettings:
    dense_model: str = os.getenv(
        'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'
    )
    dense_vector_name: str = os.getenv('DENSE_VECTOR_NAME', 'dense')
    dimension: int = int(os.getenv('EMBEDDING_DIMENSION', '384'))

@dataclass(frozen=True)
class RerankerSettings:
    model: str = os.getenv('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    top_k: int = int(os.getenv('RERANKER_TOP_K', '5'))
    candidate_k: int = int(os.getenv('RERANKER_CANDIDATE_K', '10'))
    enabled: bool = os.getenv('RERANKER_ENABLED', 'false').lower() == 'true'

@dataclass(frozen=True)
class HybridRetrievalSettings:
    rrf_k: int = int(os.getenv('RRF_K', '60'))
    prefetch_k: int = int(os.getenv('RETRIEVAL_PREFETCH_K', '50'))
    final_k: int = int(os.getenv('RETRIEVAL_FINAL_K', '20'))

@dataclass(frozen=True)
class Settings:
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    elasticsearch: ElasticsearchSettings = field(default_factory=ElasticsearchSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    hybrid_retrieval: HybridRetrievalSettings = field(default_factory=HybridRetrievalSettings)
settings = Settings()
