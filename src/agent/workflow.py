from __future__ import annotations
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from langgraph.errors import NodeInterrupt, GraphRecursionError
from loguru import logger
from src.agent.state import AgentState
from src.agent.planner import plan_node
from src.agent.vector_search import vector_search_node
from src.agent.web_search import web_search_node
from src.agent.validate import validation_node
from src.agent.gen import gen_node
from src.agent.provenance import SOURCE_LABELS, default_retrieval_status
import os
import uuid
from datetime import datetime, timezone
from langgraph.checkpoint.memory import MemorySaver


@dataclass
class WorkflowOutcome:
    answer: str
    source: list[dict] = field(default_factory=list)
    reasoning_step: list[str] = field(default_factory=list)
    node_timings: dict = field(default_factory=dict)
    trace_id: str | None = None
    langfuse_url: str | None = None
    source_mode: str = "failed"
    source_label: str = SOURCE_LABELS["failed"]
    retrieval_verified: bool = False
    retrieval_status: dict[str, str] = field(default_factory=default_retrieval_status)
    fallback_used: bool = False
    fallback_reason: str | None = None


def route_question(state: AgentState) -> str:
    decision = state.get("decision")
    logger.info(f"[Router] Route decision - decision={decision}")
    if decision in ["direct_llm", "clarify"]:
        return "generate"
    if decision == "rag":
        return "vector_search"
    logger.warning("[Router] Unknown state, defaulting to generate")
    return "generate"


def decide_to_generate(state: AgentState) -> str:
    validation = state.get("validation_result")
    logger.info(f"[Router] Validation result: {validation}")
    if validation == "relevant":
        return "generate"
    logger.info(f"[Router] Documents {validation}, falling back to web_search")
    return "web_search"


def route_after_retrieval(state: AgentState) -> str:
    status = state.get("retrieval_status", {})
    hybrid_ready = all(
        status.get(stage) == "success" for stage in ("dense", "bm25", "rrf")
    )
    if hybrid_ready and state.get("document"):
        return "validate"
    logger.warning("[Router] Strict hybrid retrieval failed; stopping grounded path")
    return "generate"


def create_workflow() -> MultiAgentWorkflow:
    return MultiAgentWorkflow()


class MultiAgentWorkflow:
    def __init__(self, checkpointer: MemorySaver | None = None):
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self.app = self.create_and_compile_graph()

    def create_and_compile_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("planner", plan_node)
        workflow.add_node("vector_search", vector_search_node)
        workflow.add_node("validate", validation_node)
        workflow.add_node("web_search", web_search_node)
        workflow.add_node("generate", gen_node)
        workflow.set_entry_point("planner")
        workflow.add_conditional_edges(
            "planner",
            route_question,
            {
                "generate": "generate",
                "vector_search": "vector_search",
                "web_search": "web_search",
            },
        )
        workflow.add_conditional_edges(
            "vector_search",
            route_after_retrieval,
            {"validate": "validate", "generate": "generate"},
        )
        workflow.add_conditional_edges(
            "validate",
            decide_to_generate,
            {"generate": "generate", "web_search": "web_search"},
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile(checkpointer=self.checkpointer)

    def run(self, query: str, session_id: str | None = None) -> WorkflowOutcome:
        thread_id = session_id or str(uuid.uuid4())

        input_state: AgentState = {
            "query": query,
            "search_query": None,
            "decision": "rag",
            "route": None,
            "document": [],
            "validation_result": None,
            "answer": "",
            "source": [],
            "reasoning_step": [],
            "chat_history": [
                {
                    "role": "user",
                    "content": query,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "node_timings": {},
            "source_mode": "failed",
            "source_label": SOURCE_LABELS["failed"],
            "retrieval_verified": False,
            "retrieval_status": default_retrieval_status(),
            "fallback_used": False,
            "fallback_reason": None,
        }

        callbacks = []
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            try:
                from langfuse.langchain import CallbackHandler
                callbacks.append(CallbackHandler())
            except Exception as e:
                logger.warning(f"Failed to initialize Langfuse callback: {e}")

        try:
            result = self.app.invoke(
                input_state,
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 10,
                    "callbacks": callbacks,
                },
            )
        except NodeInterrupt as e:
            logger.error(f"[Workflow] Node interrupted: {e}")
            return WorkflowOutcome(
                answer="Service temporarily unavailable",
                source=[],
                reasoning_step=[f"INTERRUPT: {e}"],
                fallback_reason="The answer service is temporarily unavailable.",
            )
        except GraphRecursionError as e:
            logger.error(f"[Workflow] Recursion limit exceeded: {e}")
            return WorkflowOutcome(
                answer="Service temporarily unavailable",
                source=[],
                reasoning_step=[f"RECURSION_ERROR: {e}"],
                fallback_reason="The answer service is temporarily unavailable.",
            )
        except Exception as e:
            logger.exception(f"[Workflow] Unexpected error: {e}")
            return WorkflowOutcome(
                answer="Service temporarily unavailable",
                source=[],
                reasoning_step=["ERROR: Workflow unavailable"],
                fallback_reason="The answer service is temporarily unavailable.",
            )

        return WorkflowOutcome(
            answer=result.get("answer", ""),
            source=result.get("source", []),
            reasoning_step=result.get("reasoning_step", []),
            node_timings=result.get("node_timings", {}),
            source_mode=result.get("source_mode", "failed"),
            source_label=result.get("source_label", SOURCE_LABELS["failed"]),
            retrieval_verified=result.get("retrieval_verified", False),
            retrieval_status=result.get(
                "retrieval_status", default_retrieval_status()
            ),
            fallback_used=result.get("fallback_used", False),
            fallback_reason=result.get("fallback_reason"),
        )
