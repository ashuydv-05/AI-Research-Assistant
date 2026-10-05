import time
from fastapi import APIRouter, Depends, HTTPException, Header
from loguru import logger
from src.agent.workflow import MultiAgentWorkflow, WorkflowOutcome
from src.api.dependencies import get_workflow
from src.api.models import ChatRequest, ChatResponse, ReasoningStep, Source
from src.config.clients import (
    request_groq_api_key,
    request_tavily_api_key,
)
from src.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])
chat_service = ChatService()
STEP_MAP = {
    "PLANNER:": ("planner", "Query analyzed"),
    "VECTOR_SEARCH:": ("vector_search", "Searching arXiv papers"),
    "WEB_SEARCH:": ("web_search", "Searching the web"),
    "VALIDATE:": ("validate", "Document validated"),
    "GEN:": ("generate", "Answer generated"),
    "ERROR:": ("error", "Error occurred"),
}


def generate_session_id() -> str:
    import uuid

    return str(uuid.uuid4())


def convert_sources(sources: list[dict]) -> list[Source]:
    return [Source.model_validate(s) for s in sources]


def build_reasoning_steps(result: WorkflowOutcome) -> list[ReasoningStep]:
    steps = []
    seen_steps = set()
    for step_text in result.reasoning_step:
        if step_text in seen_steps:
            continue
        seen_steps.add(step_text)
        for prefix, (action, default_obs) in STEP_MAP.items():
            if step_text.startswith(prefix):
                obs = default_obs
                if action == "retrieve":
                    obs = step_text
                steps.append(
                    ReasoningStep(thought=step_text, action=action, observation=obs)
                )
                break
        else:
            steps.append(
                ReasoningStep(thought=step_text, action="step", observation="")
            )
    return steps


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    workflow: MultiAgentWorkflow = Depends(get_workflow),
    x_groq_api_key: str | None = Header(None, alias="x-groq-api-key"),
    x_tavily_api_key: str | None = Header(None, alias="x-tavily-api-key"),
) -> ChatResponse:
    header_key = (x_groq_api_key or request.groq_api_key or "").strip()
    tavily_key = (x_tavily_api_key or request.tavily_api_key or "").strip()
    missing_keys = []
    if not header_key:
        missing_keys.append("Groq")
    if not tavily_key:
        missing_keys.append("Tavily")
    if missing_keys:
        raise HTTPException(
            status_code=428,
            detail=f"Configure {' and '.join(missing_keys)} API key{'s' if len(missing_keys) > 1 else ''} before using Chat.",
        )
    groq_token = request_groq_api_key.set(header_key or None)
    tavily_token = request_tavily_api_key.set(tavily_key or None)

    session_id = request.session_id or generate_session_id()
    logger.info(
        f"Chat request - Session: {session_id}, Message: {request.message[:50]}..."
    )
    if workflow is None:
        workflow = MultiAgentWorkflow()
    start_time = time.time()
    try:
        logger.info(f"[API] Starting workflow.run() for session {session_id}")
        result: WorkflowOutcome = chat_service.answer(
            request.message, session_id, workflow
        )
        logger.info(
            f"[API] Workflow completed. Sources count: {len(result.source)}, Reasoning steps: {len(result.reasoning_step)}"
        )
        execution_time = (time.time() - start_time) * 1000
        return ChatResponse(
            answer=result.answer,
            session_id=session_id,
            reasoning_steps=build_reasoning_steps(result),
            sources=convert_sources(result.source),
            execution_time=execution_time,
            node_timings=result.node_timings,
            source_mode=result.source_mode,
            source_label=result.source_label,
            retrieval_verified=result.retrieval_verified,
            retrieval_status=result.retrieval_status,
            fallback_used=result.fallback_used,
            fallback_reason=result.fallback_reason,
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="The research assistant could not process this request."
        )
    finally:
        request_groq_api_key.reset(groq_token)
        request_tavily_api_key.reset(tavily_token)
