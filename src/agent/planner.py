from typing import Literal, Optional
import time
import uuid

from loguru import logger
from pydantic import BaseModel, Field

from src.agent.state import AgentState
from src.config.clients import get_llm_client
from src.config.prompt import PLANNER_PROMPT_TEMPLATE


# ---------------------------------------------------------
# 1. Structured output from Planner LLM
# ---------------------------------------------------------

class PlannerOutcome(BaseModel):
    decision: Literal[
        "direct_llm",
        "rag",
        "clarify",
    ]

    resolved_query: Optional[str] = Field(
        default=None,
        description=(
            "A standalone version of the user's question after "
            "resolving references using conversation history."
        ),
    )

   


# ---------------------------------------------------------
# 2. Format conversation history
# ---------------------------------------------------------

def format_chat_history(
    chat_history: list[dict],
    current_query: str,
) -> str:

    if not chat_history:
        return "(No previous conversation)"

    # Avoid sending the current user message twice
    history = chat_history

    if (
        history
        and history[-1].get("role") == "user"
        and history[-1].get("content", "").strip() == current_query.strip()
    ):
        history = history[:-1]

    if not history:
        return "(No previous conversation)"

    # Keep recent context
    history = history[-6:]

    lines = []

    for message in history:
        role = message.get("role", "user")
        content = message.get("content", "").strip()

        if not content:
            continue

        role_name = (
            "User"
            if role == "user"
            else "Assistant"
        )

        lines.append(f"{role_name}: {content}")

    return "\n".join(lines)


# ---------------------------------------------------------
# 3. Planner Node
# ---------------------------------------------------------

def plan_node(state: AgentState) -> dict:

    call_id = str(uuid.uuid4())[:8]

    query = state["query"]
    chat_history = state.get("chat_history", [])

    logger.info(
        f"[Planner] Called - id={call_id}, "
        f"query={query[:80]}..."
    )

    existing_timings = state.get("node_timings", {})

    # -----------------------------------------------------
    # Step 1: Prepare conversation context
    # -----------------------------------------------------

    history_str = format_chat_history(
        chat_history,
        query,
    )

    # -----------------------------------------------------
    # Step 2: Get LLM
    # -----------------------------------------------------

    llm = get_llm_client(
        max_tokens=512,
        temperature=0.0,
    )

    if not llm:
        raise RuntimeError(
            "Planner LLM is temporarily unavailable."
        )

    # -----------------------------------------------------
    # Step 3: Give query + history to Planner LLM
    # -----------------------------------------------------

    start_time = time.time()

    try:

        structured_llm = llm.with_structured_output(
            PlannerOutcome
        )

        prompt = PLANNER_PROMPT_TEMPLATE.invoke(
            {
                "query": query,
                "chat_history": history_str,
            }
        )

        result = structured_llm.invoke(prompt)

        elapsed = (time.time() - start_time) * 1000

        # -------------------------------------------------
        # Step 4: Validate structured output
        # -------------------------------------------------

        if isinstance(result, dict):
            result = PlannerOutcome(**result)

        # -------------------------------------------------
        # Step 5: Resolve query
        # -------------------------------------------------

        resolved_query = (
            result.resolved_query
            or query
        )

        # -------------------------------------------------
        # Step 6: Build planner reasoning
        # -------------------------------------------------

      

        # -------------------------------------------------
        # Step 7: Return state update
        # -------------------------------------------------

        return {
            "decision": result.decision,

            # Only RAG needs an actual retrieval query,
            # but keeping this in state is useful.
            "search_query": resolved_query,

            "reasoning_step": [
                f"PLANNER: {result.decision}"
            ],

            "node_timings": {
                **existing_timings,
                "planner": elapsed,
            },
        }

    except Exception as e:

        elapsed = (time.time() - start_time) * 1000

        logger.exception(
            f"[Planner] Failed - id={call_id}"
        )

        # Important:
        # Do NOT silently decide RAG if the planner fails.
        #
        # A planner failure is different from a planner decision.
        # Let the workflow/application handle the failure explicitly.

        raise RuntimeError(
            f"Planner failed: {str(e)}"
        ) from e