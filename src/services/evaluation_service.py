from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from src.evaluation.models import EvaluationReport
from src.evaluation.runner import EvaluationRunner, load_dataset


class EvaluationService:
    def __init__(
        self,
        results_dir: str | Path = "data/evaluation/results",
    ) -> None:
        self.results_dir = Path(results_dir)

    def run(
        self,
        *,
        dataset_path: str,
        top_k: int,
        retrieval: str,
        llm: str,
        max_questions: int | None,
        groq_api_key: str,
        openrouter_api_key: str,
        tavily_api_key: str,
        on_progress: Callable[
            [dict[str, Any]],
            None,
        ]
        | None = None,
    ) -> EvaluationReport:

        # ---------------------------------------------
        # Load evaluation dataset
        # ---------------------------------------------

        questions = load_dataset(
            dataset_path
        )

        if max_questions:
            questions = questions[
                :max_questions
            ]

        # ---------------------------------------------
        # Retrieval configurations
        # ---------------------------------------------

        retrieval_keys = (
            ["vector", "hybrid"]
            if retrieval == "both"
            else [retrieval]
        )

        # ---------------------------------------------
        # LLM configurations
        # ---------------------------------------------

        llm_keys = (
            ["model_1", "model_2"]
            if llm == "both"
            else [llm]
        )

        # ---------------------------------------------
        # Create Evaluation Runner
        # ---------------------------------------------

        runner = EvaluationRunner(
            output_dir=self.results_dir,
            top_k=top_k,

            # Benchmark LLMs
            groq_api_key=groq_api_key,

            # LLM-as-Judge
            openrouter_api_key=(
                openrouter_api_key
            ),

            # Web fallback
            tavily_api_key=tavily_api_key,
        )

        # ---------------------------------------------
        # Run evaluation
        # ---------------------------------------------

        return runner.run_all_combinations(
            questions=questions,
            retrieval_keys=retrieval_keys,
            llm_keys=llm_keys,
            dataset_path=dataset_path,
            on_progress=on_progress,
        )