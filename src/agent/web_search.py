from loguru import logger
from src.agent.state import AgentState
from src.retrieval.hybrid_search import WebSearch


def web_search_node(state: AgentState) -> dict:
    query = state.get("search_query") or state.get("query", "")
    retrieval_status = state.get("retrieval_status", {})
    try:
        searcher = WebSearch()
        if not searcher.is_available():
            logger.warning("[WebSearch] Tavily is not configured; skipping web fallback")
            return {
                "document": [],
                "source_mode": "failed",
                "source_label": "No Verified Source",
                "retrieval_verified": False,
                "retrieval_status": {**retrieval_status, "tavily": "failed"},
                "fallback_used": True,
                "fallback_reason": "Web search could not provide reliable context.",
                "reasoning_step": ["WEB_SEARCH: Skipped (Tavily not configured)"],
            }
        results = searcher.search(query) 
        #Actual Searching happens
        logger.info(f"[WebSearch] Tavily returned {len(results)} results")
        docs_as_dicts = [
            {
                "id": doc.id,
                "content": doc.content,
                "title": doc.title,
                "score": doc.score,
                "source": doc.source,
                "metadata": doc.metadata,
            }
            for doc in results
        ]
        if not docs_as_dicts:
            return {
                "document": [],
                "source_mode": "failed",
                "source_label": "No Verified Source",
                "retrieval_verified": False,
                "retrieval_status": {**retrieval_status, "tavily": "failed"},
                "fallback_used": True,
                "fallback_reason": "Web search could not provide reliable context.",
                "reasoning_step": ["WEB_SEARCH: No reliable results found"],
            }
        return {
            "document": docs_as_dicts,
            "source_mode": "web",
            "source_label": "Web Search",
            "retrieval_verified": False,
            "retrieval_status": {**retrieval_status, "tavily": "success"},
            "fallback_used": True,
            "reasoning_step": [f"WEB_SEARCH: Retrieved {len(docs_as_dicts)} result"],
        }
    except Exception as e:
        logger.error(f"[WebSearch] Error: {e}")
        return {
            "document": [],
            "source_mode": "failed",
            "source_label": "No Verified Source",
            "retrieval_verified": False,
            "retrieval_status": {**retrieval_status, "tavily": "failed"},
            "fallback_used": True,
            "fallback_reason": "Web search could not provide reliable context.",
            "reasoning_step": ["WEB_SEARCH: No reliable results found"],
        }
