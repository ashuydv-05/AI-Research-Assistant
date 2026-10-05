from typing import Literal
from loguru import logger
from pydantic import BaseModel
from src.config.clients import get_llm_client
from src.agent.state import AgentState
from src.config.prompt import VALIDATE_SYSTEM_PROMPT, VALIDATE_HUMAN_TEMPLATE
import time


class ValidationOutcome(BaseModel):
    validation: Literal["relevant", "insufficient", "off_topic"]
    reasoning: str


def validation_node(state: AgentState) -> dict:
    query = state.get("query", "")
    document = state.get("document", [])
    existing_timings = state.get("node_timings", {})
    retrieval_status = state.get("retrieval_status", {})

    hybrid_ready = all(
        retrieval_status.get(stage) == "success"
        for stage in ("dense", "bm25", "rrf")
    )
    if not hybrid_ready:
        logger.info("[Validate] Complete hybrid evidence is unavailable")
        return {
            "document": document,
            "validation_result": "insufficient",
            "retrieval_status": {**retrieval_status, "validation": "failed"},
            "fallback_used": True,
            "fallback_reason": "Hybrid retrieval could not provide sufficient context.",
            "reasoning_step": [
                "VALIDATE: Insufficient context; trying web search"
            ],
            "node_timings": existing_timings,
        }

    if not document:
        logger.info("[Validate] No documents to validate")
        return {
            "document": document,
            "validation_result": "insufficient",
            "retrieval_status": {**retrieval_status, "validation": "failed"},
            "fallback_used": True,
            "fallback_reason": "Hybrid retrieval could not provide sufficient context.",
            "reasoning_step": ["VALIDATE: No documents found"],
            "node_timings": existing_timings,
        }
    llm = get_llm_client(max_tokens=256, temperature=0.0)
    if not llm:
        logger.warning("[Validate] No LLM available, assuming relevant")
        return {
            "document": document,
            "validation_result": "insufficient",
            "retrieval_status": {**retrieval_status, "validation": "failed"},
            "fallback_used": True,
            "fallback_reason": "Hybrid retrieval validation was unavailable.",
            "reasoning_step": ["VALIDATE: Validation unavailable; trying web search"],
            "node_timings": existing_timings,
        }
    docs_text = "\n".join(
        (
            f"[{i + 1}] {d.get('title', 'Unknown')}: {d.get('content', '')[:500]}"
            for i, d in enumerate(document[:3])
        )
    )
    messages = [
        {"role": "system", "content": VALIDATE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": VALIDATE_HUMAN_TEMPLATE.format(query=query, documents=docs_text),
        },
    ]

    start_time = time.time()
    try:
        structured_llm = llm.with_structured_output(ValidationOutcome)
        result = structured_llm.invoke(messages)
        elapsed = (time.time() - start_time) * 1000
        new_timings = {**existing_timings, "validate": elapsed}

        if isinstance(result, dict):
            result = ValidationOutcome(**result)

        logger.info(f"[Validate] Result: {result.validation} - {result.reasoning}")
        return {
            "document": document,
            "validation_result": result.validation,
            "retrieval_status": {
                **retrieval_status,
                "validation": "passed" if result.validation == "relevant" else "failed",
            },
            "fallback_used": result.validation != "relevant",
            "fallback_reason": (
                None
                if result.validation == "relevant"
                else "Hybrid retrieval could not provide sufficient context."
            ),
            "reasoning_step": [f"VALIDATE: {result.validation} - {result.reasoning}"],
            "node_timings": new_timings,
        }
    except Exception as e:
        logger.error(f"[Validate] Error: {e}")
        return {
            "document": document,
            "validation_result": "insufficient",
            "retrieval_status": {**retrieval_status, "validation": "failed"},
            "fallback_used": True,
            "fallback_reason": "Hybrid retrieval validation was unavailable.",
            "reasoning_step": [
                "VALIDATE: Validation unavailable; trying web search"
            ],
            "node_timings": {**existing_timings, "validate": (time.time() - start_time) * 1000},
        }
