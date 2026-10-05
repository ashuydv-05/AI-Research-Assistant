from typing import Literal

from pydantic import BaseModel, Field

from src.agent.provenance import SOURCE_LABELS, default_retrieval_status


class ReasoningStep(BaseModel):
    thought: str
    action: str | None = None
    observation: str | None = None


class Source(BaseModel):
    title: str = "Unknown"
    content: str = ""
    id: int | str | None = None
    score: float | None = None
    source: str | None = None
    metadata: dict | None = None


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message", min_length=1)
    session_id: str | None = Field(
        None, description="Session ID for conversation continuity"
    )
    groq_api_key: str | None = Field(
        None, description="User-provided Groq API Key"
    )
    tavily_api_key: str | None = Field(
        None, description="User-provided Tavily API Key"
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Agent's response")
    session_id: str = Field(..., description="Session ID")
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    execution_time: float = Field(..., description="Execution time in milliseconds")
    node_timings: dict[str, float] = Field(default_factory=dict)
    source_mode: Literal["hybrid", "web", "direct", "failed"] = "failed"
    source_label: str = SOURCE_LABELS["failed"]
    retrieval_verified: bool = False
    retrieval_status: dict[str, str] = Field(default_factory=default_retrieval_status)
    fallback_used: bool = False
    fallback_reason: str | None = None


class HealthCheck(BaseModel):
    status: str
    version: str
    components: dict[str, str]
