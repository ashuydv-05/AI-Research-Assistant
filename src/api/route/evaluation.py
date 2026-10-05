from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from src.services.evaluation_service import EvaluationService


router = APIRouter(prefix="/evaluation", tags=["evaluation"])

RESULTS_DIR = Path("data/evaluation/results")
evaluation_service = EvaluationService(RESULTS_DIR)


class RunEvaluationRequest(BaseModel):
    dataset_path: str = "data/evaluation/evaluation_dataset.json"
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval: str = Field(
        default="both",
        description="'vector', 'hybrid', or 'both'",
    )
    llm: str = Field(
        default="both",
        description="'model_1', 'model_2', or 'both'",
    )
    max_questions: int | None = Field(
        default=None,
        description="Optional question limit",
    )

    # BYOK keys
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    tavily_api_key: str | None = None

    # Kept for backward compatibility
    gemini_api_key: str | None = None
    openai_api_key: str | None = None


@router.get("/health")
async def evaluation_health() -> dict[str, str]:
    """Health check for evaluation subsystem."""
    return {
        "status": "healthy",
        "service": "evaluation",
    }


@router.get("/config-info")
async def get_config_info() -> dict:
    """Return explicit model names and retrieval configurations for UI display."""
    return {
        "model_1": {
            "key": "model_1",
            "name": os.getenv(
                "GROQ_MODEL",
                "qwen/qwen3.8-27b",
            ),
            "provider": "Groq",
            "family": "Alibaba Qwen",
        },
        "model_2": {
            "key": "model_2",
            "name": os.getenv(
                "GROQ_MODEL_2",
                "openai/gpt-oss-20b",
            ),
            "provider": "Groq",
            "family": "OpenAI Family",
        },
        "judge": {
            "name": os.getenv(
                "OPENROUTER_MODEL",
                "openai/gpt-oss-120b",
            ),
            "provider": "OpenRouter",
            "description": "LLM-as-Judge Evaluator",
        },
        "retrievers": {
            "vector": {
                "name": "Vector Retrieval (Dense)",
                "description": "Qdrant semantic vector search",
            },
            "hybrid": {
                "name": "Hybrid Retrieval (Dense + BM25 + RRF)",
                "description": (
                    "Reciprocal Rank Fusion across Qdrant "
                    "& Elasticsearch BM25"
                ),
            },
        },
    }


@router.get("/summary")
async def get_evaluation_summary() -> dict:
    """Retrieve the latest aggregated evaluation summary and 2x2 matrix."""
    summary_path = RESULTS_DIR / "summary.json"

    if not summary_path.exists():
        return {
            "status": "not_found",
            "message": (
                "No evaluation summary found. "
                "Please run an evaluation first."
            ),
            "configurations": {},
            "comparison_matrix": {},
        }

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        results_path = RESULTS_DIR / "results.json"

        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                details = json.load(f).get(
                    "detailed_results",
                    [],
                )

            has_retrieved_documents = any(
                item.get("retrieved_documents")
                for item in details
            )

            summary["valid"] = has_retrieved_documents

            if details and not has_retrieved_documents:
                summary["warning"] = (
                    "This run retrieved zero documents, so its zero "
                    "metrics are not a valid RAG benchmark. Check "
                    "Qdrant and Elasticsearch, then run it again."
                )

        return summary

    except Exception as e:
        logger.error(
            f"Error reading evaluation summary: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to read evaluation summary: "
                f"{str(e)}"
            ),
        )


@router.get("/results")
async def get_evaluation_results() -> dict:
    """Retrieve detailed per-question results from the latest evaluation."""
    results_path = RESULTS_DIR / "results.json"

    if not results_path.exists():
        return {
            "status": "not_found",
            "message": (
                "No evaluation results found. "
                "Please run an evaluation first."
            ),
            "detailed_results": [],
        }

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        logger.error(
            f"Error reading evaluation results: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to read evaluation results: "
                f"{str(e)}"
            ),
        )


@router.post("/stream")
async def stream_evaluation(
    request: RunEvaluationRequest,
    x_groq_api_key: str | None = Header(
        None,
        alias="x-groq-api-key",
    ),
    x_openrouter_api_key: str | None = Header(
        None,
        alias="x-openrouter-api-key",
    ),
    x_tavily_api_key: str | None = Header(
        None,
        alias="x-tavily-api-key",
    ),
):
    """
    Execute evaluation and stream real-time progress events over SSE.

    Required BYOK keys:
    - Groq
    - OpenRouter
    - Tavily
    """

    groq_api_key = (
        x_groq_api_key
        or request.groq_api_key
        or ""
    ).strip()

    openrouter_api_key = (
        x_openrouter_api_key
        or request.openrouter_api_key
        or ""
    ).strip()

    tavily_api_key = (
        x_tavily_api_key
        or request.tavily_api_key
        or ""
    ).strip()

    _require_evaluation_keys(
        groq_api_key,
        openrouter_api_key,
        tavily_api_key,
    )

    def event_generator():
        event_queue = queue.Queue()

        def on_prog(data: dict):
            event_queue.put(data)

        def runner_target():
            try:
                report = evaluation_service.run(
                    dataset_path=request.dataset_path,
                    top_k=request.top_k,
                    retrieval=request.retrieval,
                    llm=request.llm,
                    max_questions=request.max_questions,

                    # Visitor-provided keys
                    groq_api_key=groq_api_key,
                    openrouter_api_key=openrouter_api_key,
                    tavily_api_key=tavily_api_key,

                    on_progress=on_prog,
                )

                event_queue.put(
                    {
                        "type": "complete",
                        "timestamp": report.timestamp,
                        "best_configuration": (
                            report.best_configuration
                        ),
                        "best_reason": report.best_reason,
                        "comparison_matrix": (
                            report.comparison_matrix
                        ),
                        "configurations": {
                            k: v.model_dump()
                            for k, v in report.configurations.items()
                        },
                        "message": (
                            "✓ Evaluation Benchmark "
                            "Completed Successfully!"
                        ),
                    }
                )

            except Exception as exc:
                logger.error(
                    f"Streaming evaluation error: {exc}",
                    exc_info=True,
                )

                event_queue.put(
                    {
                        "type": "error",
                        "message": str(exc),
                    }
                )

            finally:
                event_queue.put(None)

        thread = threading.Thread(
            target=runner_target,
            daemon=True,
        )

        thread.start()

        while True:
            item = event_queue.get()

            if item is None:
                break

            yield (
                f"data: {json.dumps(item)}\n\n"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/run")
async def run_evaluation(
    request: RunEvaluationRequest,
    x_groq_api_key: str | None = Header(
        None,
        alias="x-groq-api-key",
    ),
    x_openrouter_api_key: str | None = Header(
        None,
        alias="x-openrouter-api-key",
    ),
    x_tavily_api_key: str | None = Header(
        None,
        alias="x-tavily-api-key",
    ),
) -> dict:
    """
    Execute evaluation across selected configurations.

    Required BYOK keys:
    - Groq
    - OpenRouter
    - Tavily
    """

    try:
        groq_api_key = (
            x_groq_api_key
            or request.groq_api_key
            or ""
        ).strip()

        openrouter_api_key = (
            x_openrouter_api_key
            or request.openrouter_api_key
            or ""
        ).strip()

        tavily_api_key = (
            x_tavily_api_key
            or request.tavily_api_key
            or ""
        ).strip()

        _require_evaluation_keys(
            groq_api_key,
            openrouter_api_key,
            tavily_api_key,
        )

        report = evaluation_service.run(
            dataset_path=request.dataset_path,
            top_k=request.top_k,
            retrieval=request.retrieval,
            llm=request.llm,
            max_questions=request.max_questions,

            # Visitor-provided keys
            groq_api_key=groq_api_key,
            openrouter_api_key=openrouter_api_key,
            tavily_api_key=tavily_api_key,
        )

        return {
            "status": "success",
            "timestamp": report.timestamp,
            "best_configuration": (
                report.best_configuration
            ),
            "best_reason": report.best_reason,
            "comparison_matrix": (
                report.comparison_matrix
            ),
            "configurations": {
                k: v.model_dump()
                for k, v in report.configurations.items()
            },
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise

        logger.error(
            f"Failed to run evaluation: {e}",
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}",
        )


def _require_evaluation_keys(
    groq_api_key: str,
    openrouter_api_key: str,
    tavily_api_key: str,
) -> None:
    """Ensure all required BYOK credentials are provided."""

    missing_keys = []

    if not groq_api_key:
        missing_keys.append("Groq")

    if not openrouter_api_key:
        missing_keys.append("OpenRouter")

    if not tavily_api_key:
        missing_keys.append("Tavily")

    if missing_keys:
        raise HTTPException(
            status_code=428,
            detail=(
                f"Configure {' and '.join(missing_keys)} API key"
                f"{'s' if len(missing_keys) > 1 else ''} "
                "before running Evaluation."
            ),
        )