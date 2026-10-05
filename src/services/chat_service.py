from __future__ import annotations

from src.agent.provenance import SOURCE_LABELS
from src.agent.workflow import MultiAgentWorkflow, WorkflowOutcome
from src.config.clients import extract_message_text, get_llm_client


class ChatService:
    def answer(
        self,
        message: str,
        session_id: str,
        workflow: MultiAgentWorkflow | None,
    ) -> WorkflowOutcome:
        active_workflow = workflow or MultiAgentWorkflow()
        result = active_workflow.run(message, session_id=session_id)
        if result.answer and str(result.answer).strip():
            return result

        if result.source_mode == "direct":
            try:
                response = get_llm_client().invoke(
                    f"Answer this general question briefly and clearly: {message}"
                )
                result.answer = extract_message_text(response)
            except Exception:
                result.answer = ""

        if not result.answer:
            result.answer = "No sufficient reliable context was found to answer this question."
            result.source_mode = "failed"
            result.source_label = SOURCE_LABELS["failed"]
            result.retrieval_verified = False
            result.source = []
            result.fallback_reason = "No reliable context was found."
        return result
