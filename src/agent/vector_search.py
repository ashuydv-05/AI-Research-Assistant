from loguru import logger
from src.agent.state import AgentState
from src.retrieval.hybrid_search import VectorSearch
import time


def vector_search_node(state: AgentState) -> dict:
    query = state.get("search_query") or state.get("query", "")
    existing_timings = state.get("node_timings", {})

    start_time = time.time()
    try:
        searcher = VectorSearch()
        outcome = searcher.search_with_status(query, top_k=5)
        results = outcome.results
        elapsed = (time.time() - start_time) * 1000
        new_timings = {
            **existing_timings,
            **outcome.stage_timings_ms,
            "vector_search": elapsed,
        }

        docs_as_dicts = [doc.to_dict() for doc in results]

        hybrid_succeeded = outcome.succeeded
        reason = outcome.failure_reason

        return {
            "document": docs_as_dicts,
            "retrieval_status": {
                **state.get("retrieval_status", {}),
                **outcome.retrieval_status,
            },
            "fallback_reason": None if hybrid_succeeded else (
                reason or "Strict hybrid retrieval did not return complete evidence."
            ),
            "reasoning_step": [
                (
                    f"VECTOR_SEARCH: Retrieved {len(docs_as_dicts)} documents for: {query[:50]}..."
                    if hybrid_succeeded
                    else "VECTOR_SEARCH: Strict hybrid retrieval failed; answer generation stopped"
                )
            ],
            "node_timings": new_timings,
        }
    except Exception as e:
        logger.error(f"[VectorSearch] Error: {e}")
        return {
            "document": [],
            "retrieval_status": {
                **state.get("retrieval_status", {}),
                "dense": "failed",
                "bm25": "failed",
                "rrf": "failed",
                "reranker": "not_run",
            },
            "fallback_reason": "Strict hybrid retrieval is unavailable.",
            "reasoning_step": ["VECTOR_SEARCH: Hybrid retrieval unavailable"],
            "node_timings": {
                **existing_timings,
                "vector_search": (time.time() - start_time) * 1000,
            },
        }
