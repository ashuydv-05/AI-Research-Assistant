# Hybrid Retrieval Internal Flow

```mermaid
flowchart TD
    A["Resolved search_query"] --> B["VectorSearch.search<br/>top_k=5, prefetch=50"]
    B --> C["Create Qdrant client from QDRANT_URL<br/>and optional QDRANT_API_KEY"]
    B --> D["Attempt to create Elasticsearch client<br/>from ES_URL"]

    C --> E["Load all-MiniLM-L6-v2 once<br/>and cache the model at class level"]
    E --> F["Encode query into one 384-dimensional vector"]
    F --> G["Qdrant query_points<br/>collection=arxiv_papers<br/>using=dense, limit=50, with_payload=true"]
    G --> H["Map nearest-neighbour points to SearchResult<br/>id, content, title, cosine score,<br/>source=dense and complete payload"]

    D --> I{"Elasticsearch enabled?"}
    I -->|Local URL and ES_ENABLED is not true| J["Disable Elasticsearch<br/>Return empty BM25 list"]
    I -->|Client available| K["Run Elasticsearch multi_match query"]
    K --> L["Search title with boost 3.0<br/>Search content with boost 2.0<br/>type=best_fields, fuzziness=AUTO<br/>size=50, timeout=1 second"]
    L --> M["Map hits to SearchResult<br/>source=bm25 and BM25 score"]
    K -->|Timeout or error| J

    H --> N{"Available result lists"}
    M --> N
    J --> N
    N -->|Dense and BM25| O["RRF fusion by shared numeric chunk ID"]
    O --> P["For each rank starting at 1<br/>rrf_score += 1 / (60 + rank)"]
    P --> Q["Deduplicate chunk IDs<br/>Sort by accumulated RRF score<br/>Keep top 20 candidates"]
    N -->|Only one engine worked| R["Use that engine's ranked list directly"]
    N -->|Neither worked| S["Return empty list"]

    Q --> T{"RERANKER_ENABLED=true?"}
    R --> T
    T -->|No| U["Trim candidates to requested top 5"]
    T -->|Yes| V["Load ms-marco-MiniLM-L-6-v2 CrossEncoder"]
    V --> W["Take first 10 candidates<br/>Pair query with title + first 1,000 characters"]
    W --> X["Predict cross-encoder scores<br/>Sort descending and return top 5"]
    V -->|Model error| U

    U --> Y["Return 5 or fewer documents to agent"]
    X --> Y
```

Current behavior notes:

- Dense and BM25 searches execute sequentially in Python.
- Qdrant performs its internal nearest-neighbour search; Python does not loop through all 13,666 vectors.
- RRF controls ordering, but its calculated score is not copied into the returned `SearchResult.score`.
- If one search engine fails, retrieval silently continues with the other engine.
- `RRF_K` exists in settings, but the active retrieval configuration currently hard-codes `60`.

Source files:

- `src/retrieval/hybrid_search.py`
- `src/retrieval/hybrid_retriever.py`
- `src/retrieval/vector_retriever.py`
- `src/config/settings.py`
- `src/config/clients.py`
