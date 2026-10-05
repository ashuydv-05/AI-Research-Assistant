from langgraph.errors import NodeInterrupt
from src.config.clients import get_llm_client, extract_message_text
from src.agent.state import AgentState
from src.config.prompt import (
    GENERATE_SYSTEM_PROMPT,
    DIRECT_ANSWER_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    WEB_GENERATE_SYSTEM_PROMPT,
)
from src.agent.provenance import SOURCE_LABELS, hybrid_pipeline_succeeded
from datetime import datetime, timezone
import time


def format_documents(document: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(document, 1):
        title = doc.get("title", "Unknown")
        content = doc.get("content", "")
        metadata = doc.get("metadata", {}) or {}
        entry = f"[{i}] {title}\n{content}"
        if paper_id := metadata.get("paper_id"):
            entry += f"\n[paper_id: {paper_id}]"
        elif arxiv_id := metadata.get("arxiv_id"):
            entry += f"\n[arxiv_id: {arxiv_id}]"
        elif url := metadata.get("url"):
            entry += f"\n[url: {url}]"
        parts.append(entry)
    return "\n\n".join(parts)


def gen_node(state: AgentState) -> dict:
    query = state.get("query", "")
    decision = state.get("decision", "rag")
    document = state.get("document", [])
    chat_history = state.get("chat_history", [])
    existing_timings = state.get("node_timings", {})
    if decision == "direct_llm":
        llm = get_llm_client()
        if not llm:
            raise NodeInterrupt("Service temporarily unavailable", id="gen")
        return _gen_direct_answer(llm, query, existing_timings)
    if decision == "clarify":
        clarify_answer = "Could you please provide more details about what you're looking for?"
        return {
            "answer": clarify_answer,
            "source": [],
            "source_mode": "direct",
            "source_label": SOURCE_LABELS["direct"],
            "retrieval_verified": False,
            "reasoning_step": ["GEN: Clarify requested"],
            "chat_history": [
                {
                    "role": "assistant",
                    "content": clarify_answer,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "node_timings": existing_timings,
        }

    retrieval_status = state.get("retrieval_status", {})
    hybrid_ready = all(
        retrieval_status.get(stage) == "success"
        for stage in ("dense", "bm25", "rrf")
    )
    web_ready = retrieval_status.get("tavily") == "success"
    if not document or not (hybrid_ready or web_ready):
        return _gen_rag_answer(
            None,
            query,
            document,
            existing_timings,
            chat_history=chat_history,
            retrieval_status=retrieval_status,
            fallback_used=state.get("fallback_used", False),
            fallback_reason=state.get("fallback_reason"),
        )

    llm = get_llm_client()
    if not llm:
        raise NodeInterrupt("Service temporarily unavailable", id="gen")
    return _gen_rag_answer(
        llm,
        query,
        document,
        existing_timings,
        chat_history=chat_history,
        retrieval_status=retrieval_status,
        fallback_used=state.get("fallback_used", False),
        fallback_reason=state.get("fallback_reason"),
    )


def _gen_direct_answer(llm, query: str, existing_timings: dict) -> dict:
    messages = [
        {"role": "system", "content": DIRECT_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = (time.time() - start_time) * 1000
    new_timings = {**existing_timings, "generate": elapsed}
    answer = extract_message_text(response)

    return {
        "answer": answer,
        "source": [],
        "source_mode": "direct",
        "source_label": SOURCE_LABELS["direct"],
        "retrieval_verified": False,
        "fallback_used": False,
        "fallback_reason": None,
        "reasoning_step": ["GEN: Direct answer"],
        "chat_history": [
            {
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
        "node_timings": new_timings,
    }


def _gen_rag_answer(
    llm,
    query: str,
    document: list[dict],
    existing_timings: dict,
    chat_history: list[dict] | None = None,
    retrieval_status: dict[str, str] | None = None,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
) -> dict:
    retrieval_status = retrieval_status or {}
    if not document:
        answer = "No sufficient reliable context was found to answer this question."
        return {
            "answer": answer,
            "source": [],
            "source_mode": "failed",
            "source_label": SOURCE_LABELS["failed"],
            "retrieval_verified": False,
            "retrieval_status": retrieval_status,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason or "No reliable context was found.",
            "reasoning_step": ["GEN: No verified source available"],
            "chat_history": [
                {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "node_timings": existing_timings,
        }
    context = format_documents(document)

    is_hybrid = hybrid_pipeline_succeeded(retrieval_status)
    is_web = retrieval_status.get("tavily") == "success"
    if not is_hybrid and not is_web:
        answer = "No sufficient reliable context was found to answer this question."
        return {
            "answer": answer,
            "source": [],
            "source_mode": "failed",
            "source_label": SOURCE_LABELS["failed"],
            "retrieval_verified": False,
            "retrieval_status": retrieval_status,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason or "No reliable context was found.",
            "reasoning_step": ["GEN: No verified source available"],
            "chat_history": [
                {
                    "role": "assistant",
                    "content": answer,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "node_timings": existing_timings,
        }

    user_prompt_content = RAG_USER_TEMPLATE.format(context=context, query=query)
    if chat_history and len(chat_history) > 1:
        history_lines = []
        for msg in chat_history[:-1]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_lines.append(f"{role}: {msg.get('content', '')}")
        if history_lines:
            conv_str = "\n".join(history_lines[-4:])
            user_prompt_content = f"## Previous Conversation:\n{conv_str}\n\n{user_prompt_content}"

    messages = [
        {
            "role": "system",
            "content": GENERATE_SYSTEM_PROMPT if is_hybrid else WEB_GENERATE_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt_content,
        },
    ]

    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = (time.time() - start_time) * 1000
    new_timings = {**existing_timings, "generate": elapsed}
    answer = extract_message_text(response)

    return {
        "answer": answer,
        "source": document,
        "source_mode": "hybrid" if is_hybrid else "web",
        "source_label": SOURCE_LABELS["hybrid" if is_hybrid else "web"],
        "retrieval_verified": is_hybrid,
        "retrieval_status": retrieval_status,
        "fallback_used": fallback_used or is_web,
        "fallback_reason": fallback_reason,
        "reasoning_step": [f"GEN: RAG answer ({len(document)} source)"],
        "chat_history": [
            {
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
        "node_timings": new_timings,
    }
