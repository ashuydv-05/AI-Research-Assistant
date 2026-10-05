from __future__ import annotations

import json
import logging
from typing import Any

from src.evaluation.llm_clients import BaseLLMClient, get_eval_llm
from src.evaluation.models import JudgeScores

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    LLM-based evaluator for generated answers.

    Important:
    - The judge must return real scores.
    - Judge failures are NOT converted into fake 50/100 scores.
    - The caller is responsible for deciding how to record failed samples.
    """

    def __init__(
        self,
        judge_model_name: str = "auto",
        judge_client: BaseLLMClient | None = None,
    ) -> None:
        if judge_client is not None:
            self.client = judge_client
        else:
            if judge_model_name in {"auto", "default", "model_1"}:
                if __import__("os").getenv("OPENROUTER_API_KEY"):
                    judge_model_name = "openrouter"
                else:
                    judge_model_name = "model_1"

            self.client = get_eval_llm(judge_model_name)

        self.judge_model_name = self.client.model_name

    def evaluate(
        self,
        question: str,
        answer: str,
        reference_answer: str,
        context: str,
    ) -> JudgeScores:
        """
        Evaluate a generated answer against the reference answer
        and retrieved context.

        Raises:
            RuntimeError:
                If the LLM judge call fails.

            ValueError:
                If the judge returns invalid structured output.
        """

        prompt = self._build_prompt(
            question=question,
            answer=answer,
            reference_answer=reference_answer,
            context=context,
        )

        try:
            raw_response = self.client.generate(prompt, "")
        except Exception as exc:
            logger.exception(
                "[LLMJudge] Judge model failed."
            )
            raise RuntimeError(
                f"LLM judge evaluation failed: {exc}"
            ) from exc

        try:
            return self._parse_response(raw_response)

        except Exception as exc:
            logger.error(
                "[LLMJudge] Failed to parse judge response: %s",
                exc,
            )

            raise ValueError(
                f"Failed to parse structured output from "
                f"LLM judge: {exc}"
            ) from exc

    def evaluate_sample(
        self,
        question: str,
        answer: str,
        reference_answer: str,
        context: str,
    ) -> JudgeScores:
        """
        Backward-compatible wrapper used by EvaluationRunner.
        """

        return self.evaluate(
            question=question,
            answer=answer,
            reference_answer=reference_answer,
            context=context,
        )

    def _build_prompt(
        self,
        *,
        question: str,
        answer: str,
        reference_answer: str,
        context: str,
    ) -> str:
        return f"""
You are an expert evaluator for a Retrieval-Augmented Generation
(RAG) system.

Evaluate the generated answer using the question, reference answer,
and retrieved context.

You must evaluate four dimensions:

1. correctness
   - Does the answer correctly answer the question?
   - Compare against the reference answer.

2. faithfulness
   - Is the answer supported by the retrieved context?
   - Penalize unsupported claims or hallucinations.

3. relevance
   - Does the answer directly address the question?
   - Penalize unnecessary or unrelated information.

4. overall
   - Overall quality of the answer considering correctness,
     faithfulness, and relevance.

Return scores from 0 to 100.

Return ONLY valid JSON in exactly this structure:

{{
    "correctness": 0,
    "faithfulness": 0,
    "relevance": 0,
    "overall": 0,
    "reason": "brief explanation"
}}

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}
""".strip()

    def _parse_response(
        self,
        raw_response: Any,
    ) -> JudgeScores:
        """
        Parse structured JSON returned by the judge.
        """

        if raw_response is None:
            raise ValueError("Judge returned an empty response.")

        if isinstance(raw_response, str):
            raw_text = raw_response.strip()
        else:
            raw_text = str(raw_response).strip()

        if not raw_text:
            raise ValueError("Judge returned an empty response.")

        # Remove markdown code fences if the model added them.
        if raw_text.startswith("```"):
            lines = raw_text.splitlines()

            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            raw_text = "\n".join(lines).strip()

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Judge response is not valid JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                "Judge response must be a JSON object."
            )

        required_fields = [
            "correctness",
            "faithfulness",
            "relevance",
            "overall",
        ]

        missing = [
            field
            for field in required_fields
            if field not in data
        ]

        if missing:
            raise ValueError(
                f"Judge response is missing fields: {missing}"
            )

        scores: dict[str, float] = {}

        for field in required_fields:
            try:
                value = float(data[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Judge field '{field}' must be numeric."
                ) from exc

            if not 0 <= value <= 100:
                raise ValueError(
                    f"Judge field '{field}' must be between 0 and 100."
                )

            scores[field] = value

        reason = str(data.get("reason", "")).strip()

        return JudgeScores(
            correctness=scores["correctness"],
            faithfulness=scores["faithfulness"],
            relevance=scores["relevance"],
            overall=scores["overall"],
            reason=reason,
        )
