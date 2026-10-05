from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from src.retrieval.base import BaseRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.hybrid_search import SearchResult, WebSearch
from src.retrieval.errors import RetrievalError

from src.evaluation.llm_clients import (
    BaseLLMClient,
    get_eval_llm,
)
from src.evaluation.evaluator import LLMJudge
from src.evaluation.metrics import compute_retrieval_metrics

from src.evaluation.models import (
    EvalQuestion,
    RetrievedDoc,
    SampleResult,
    ConfigSummary,
    EvaluationReport,
)


class RetrievalUnavailableError(RuntimeError):
    """
    Raised when a benchmark cannot retrieve usable
    corpus documents.
    """

    pass


# ==========================================================
# DATASET LOADING
# ==========================================================


def load_dataset(
    dataset_path: str | Path,
) -> list[EvalQuestion]:
    """
    Load evaluation questions from a JSON file.

    Supported formats:

    {
        "questions": [...]
    }

    OR

    [...]
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found at: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    items = (
        data.get("questions", data)
        if isinstance(data, dict)
        else data
    )

    if not isinstance(items, list):
        raise ValueError(
            "Evaluation dataset must contain "
            "a list of questions."
        )

    questions: list[EvalQuestion] = []

    for item in items:

        if not isinstance(item, dict):
            continue

        qid = item.get(
            "id",
            len(questions) + 1,
        )

        q_text = (
            item.get("question")
            or item.get("query")
            or ""
        )

        expected = item.get(
            "expected",
            {},
        )

        if not isinstance(expected, dict):
            expected = {}

        ground_truth = (
            item.get("ground_truth")
            or expected.get(
                "answer",
                "",
            )
        )

        rel_docs = (
            item.get("relevant_documents")
            or expected.get(
                "paper_ids",
                [],
            )
        )

        if not isinstance(rel_docs, list):
            rel_docs = [rel_docs]

        if q_text:
            questions.append(
                EvalQuestion(
                    id=qid,
                    question=q_text,
                    ground_truth=ground_truth,
                    relevant_documents=[
                        str(p)
                        for p in rel_docs
                    ],
                )
            )

    return questions


# ==========================================================
# CONTEXT FORMATTING
# ==========================================================


def format_context(
    results: Sequence[SearchResult],
) -> str:
    """
    Format retrieved corpus documents into a
    context block for the benchmark LLM.
    """

    parts: list[str] = []

    for i, doc in enumerate(
        results,
        1,
    ):

        title = (
            doc.title
            or "Unknown"
        )

        content = (
            doc.content
            or ""
        )

        meta = (
            doc.metadata
            or {}
        )

        paper_id = (
            meta.get("paper_id")
            or meta.get("arxiv_id")
            or ""
        )

        pid_str = (
            f" [paper_id: {paper_id}]"
            if paper_id
            else ""
        )

        parts.append(
            f"[{i}] {title}{pid_str}\n"
            f"{content}"
        )

    return "\n\n".join(parts)


def format_web_context(
    results: Sequence[Any],
) -> str:
    """
    Format Tavily/WebSearch results into
    the context format used by the evaluator.
    """

    parts: list[str] = []

    for i, result in enumerate(
        results,
        1,
    ):

        title = (
            getattr(
                result,
                "title",
                None,
            )
            or "Web result"
        )

        content = (
            getattr(
                result,
                "content",
                None,
            )
            or ""
        )

        url = (
            getattr(
                result,
                "url",
                None,
            )
            or ""
        )

        # Support dictionary-style results.
        if isinstance(
            result,
            dict,
        ):
            title = (
                result.get("title")
                or "Web result"
            )

            content = (
                result.get("content")
                or result.get("snippet")
                or ""
            )

            url = (
                result.get("url")
                or ""
            )

        source_line = (
            f"\nSource: {url}"
            if url
            else ""
        )

        parts.append(
            f"[WEB {i}] {title}"
            f"{source_line}\n"
            f"{content}"
        )

    return "\n\n".join(parts)


# ==========================================================
# RESULT CONVERSION
# ==========================================================


def convert_search_results(
    results: Sequence[SearchResult],
) -> list[RetrievedDoc]:
    """
    Convert SearchResult objects into
    RetrievedDoc models used by the evaluator.
    """

    docs: list[RetrievedDoc] = []

    for result in results:

        meta = (
            result.metadata
            or {}
        )

        paper_id = (
            meta.get("paper_id")
            or meta.get("arxiv_id")
        )

        docs.append(
            RetrievedDoc(
                id=result.id,
                title=(
                    result.title
                    or "Unknown"
                ),
                paper_id=(
                    str(paper_id)
                    if paper_id
                    else None
                ),
                score=round(
                    float(result.score),
                    4,
                ),
                source=(
                    result.source
                    or "unknown"
                ),
                content_snippet=(
                    result.content[:200]
                    if result.content
                    else ""
                ),
                metadata=meta,
            )
        )

    return docs


# ==========================================================
# EVALUATION RUNNER
# ==========================================================


class EvaluationRunner:
    """
    Automated Evaluation Runner.

    Benchmark matrix:

        Vector Retrieval × Model 1
        Vector Retrieval × Model 2
        Hybrid Retrieval × Model 1
        Hybrid Retrieval × Model 2

    Benchmark models:
        - model_1 → Groq
        - model_2 → Groq

    LLM Judge:
        - OpenRouter

    Web fallback:
        - Tavily

    Important retrieval rule:

        Hybrid retrieval failure
                ↓
              STOP

        Do NOT silently fall back to Tavily.

    Tavily may be used only when the configured
    retrieval strategy explicitly allows a web
    fallback after corpus retrieval returns no
    usable results.
    """

    def __init__(
        self,
        retrievers: dict[
            str,
            BaseRetriever,
        ]
        | None = None,

        llm_clients: dict[
            str,
            BaseLLMClient,
        ]
        | None = None,

        judge: LLMJudge | None = None,

        output_dir: str | Path = (
            "data/evaluation/results"
        ),

        top_k: int = 5,

        groq_api_key: str | None = None,

        openrouter_api_key: str | None = None,

        tavily_api_key: str | None = None,

        # Kept only for backward compatibility.
        # OpenRouter is now the required judge.
        gemini_api_key: str | None = None,
    ) -> None:

        # --------------------------------------------------
        # RETRIEVERS
        # --------------------------------------------------

        self.retrievers = (
            retrievers
            or {
                "vector": VectorRetriever(),
                "hybrid": HybridRetriever(),
            }
        )

        # --------------------------------------------------
        # BENCHMARK LLMs
        # --------------------------------------------------

        if llm_clients is not None:

            self.llm_clients = llm_clients

        else:

            if not groq_api_key:
                raise ValueError(
                    "GROQ_API_KEY is required "
                    "for Evaluation because "
                    "the benchmark models run "
                    "through Groq."
                )

            self.llm_clients = {
                "model_1": get_eval_llm(
                    "model_1",
                    api_key=groq_api_key,
                ),
                "model_2": get_eval_llm(
                    "model_2",
                    api_key=groq_api_key,
                ),
            }

        # --------------------------------------------------
        # OPENROUTER JUDGE
        # --------------------------------------------------

        if judge is not None:

            self.judge = judge

        else:

            if not openrouter_api_key:
                raise ValueError(
                    "OPENROUTER_API_KEY is required "
                    "for Evaluation because "
                    "OpenRouter is the LLM judge."
                )

            self.judge = LLMJudge(
                judge_client=get_eval_llm(
                    "openrouter",
                    api_key=openrouter_api_key,
                )
            )

        # --------------------------------------------------
        # TAVILY
        # --------------------------------------------------

        if not tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY is required "
                "for Evaluation because some "
                "golden-dataset questions may "
                "require web fallback."
            )

        self.web_search = WebSearch(
            api_key=tavily_api_key
        )

        # --------------------------------------------------
        # OUTPUT
        # --------------------------------------------------

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.top_k = top_k

    # ======================================================
    # TAVILY FALLBACK
    # ======================================================

    def _web_fallback(
        self,
        query: str,
    ) -> str:
        """
        Retrieve web evidence using the
        visitor-provided Tavily API key.

        This function NEVER raises a Tavily exception
        into the main benchmark loop. It returns an
        empty string when web retrieval fails.
        """

        try:

            if not self.web_search.is_available():

                logger.warning(
                    "[Evaluation] Tavily "
                    "is not available."
                )

                return ""

            results = (
                self.web_search.search(
                    query
                )
            )

            if not results:

                logger.warning(
                    "[Evaluation] Tavily returned "
                    f"no results for: "
                    f"{query[:100]}"
                )

                return ""

            logger.info(
                "[Evaluation] Tavily fallback "
                f"returned {len(results)} results "
                f"for Q: {query[:100]}"
            )

            return format_web_context(
                results
            )

        except Exception as exc:

            logger.error(
                "[Evaluation] Tavily fallback "
                f"failed: {exc}"
            )

            return ""

    # ======================================================
    # SINGLE CONFIGURATION
    # ======================================================

    def run_single_combination(
        self,
        retrieval_key: str,
        llm_key: str,
        questions: list[EvalQuestion],
        on_progress: Any | None = None,
    ) -> list[SampleResult]:
        """
        Run one:

            Retriever × LLM

        configuration.

        Important:

        A failed retrieval, generation, or judge call
        is NOT converted into a fake evaluation score.

        Failed samples are skipped from aggregate scoring.
        """

        if retrieval_key not in self.retrievers:
            raise ValueError(
                f"Unknown retrieval key: "
                f"{retrieval_key}"
            )

        if llm_key not in self.llm_clients:
            raise ValueError(
                f"Unknown LLM key: "
                f"{llm_key}"
            )

        retriever = (
            self.retrievers[
                retrieval_key
            ]
        )

        llm = (
            self.llm_clients[
                llm_key
            ]
        )

        results: list[
            SampleResult
        ] = []

        failed_samples = 0

        logger.info(
            f"==> Evaluating "
            f"[{retrieval_key.upper()} Retrieval + "
            f"{llm_key} ({llm.model_name})] "
            f"({len(questions)} questions)"
        )

        # --------------------------------------------------
        # QUESTIONS
        # --------------------------------------------------

        for idx, q in enumerate(
            questions,
            1,
        ):

            start_time = time.time()

            retrieved_docs_models: list[
                RetrievedDoc
            ] = []

            context_str = ""

            generated_answer = ""

            web_fallback_used = False

            # ------------------------------------------------
            # QUESTION LOG
            # ------------------------------------------------

            logger.info(
                f'  [{idx}/{len(questions)}] '
                f'Q{q.id}: '
                f'"{q.question[:80]}..."'
            )

            # ------------------------------------------------
            # PROGRESS: RETRIEVAL
            # ------------------------------------------------

            if on_progress:

                on_progress(
                    {
                        "type": "step",
                        "step": "retrieval",
                        "retrieval": retrieval_key,
                        "llm": llm_key,
                        "llm_name": (
                            llm.model_name
                        ),
                        "question_index": idx,
                        "total_questions": (
                            len(questions)
                        ),
                        "question": q.question,
                        "message": (
                            f"[{retrieval_key} + "
                            f"{llm.model_name}] "
                            f"Q{idx}: "
                            "Searching documents..."
                        ),
                    }
                )

            # ==================================================
            # 1. CORPUS RETRIEVAL
            # ==================================================

            try:

                search_results = (
                    retriever.search(
                        q.question,
                        top_k=self.top_k,
                    )
                )

            except RetrievalError as exc:

                failed_samples += 1

                logger.error(
                    f"[Evaluation] "
                    f"{retrieval_key} retrieval "
                    f"failed for Q{q.id}: {exc}"
                )

                if on_progress:
                    on_progress(
                        {
                            "type": "sample_failed",
                            "retrieval": retrieval_key,
                            "llm": llm_key,
                            "llm_name": (
                                llm.model_name
                            ),
                            "question_index": idx,
                            "total_questions": (
                                len(questions)
                            ),
                            "question": q.question,
                            "error": str(exc),
                            "message": (
                                f"✗ [{retrieval_key} + "
                                f"{llm.model_name}] "
                                f"Q{idx}: "
                                f"Retrieval failed"
                            ),
                        }
                    )

                # IMPORTANT:
                # Never use Tavily to hide a strict
                # hybrid retrieval backend failure.
                continue

            except Exception as exc:

                failed_samples += 1

                logger.exception(
                    f"[Evaluation] Unexpected "
                    f"retrieval error for Q{q.id}"
                )

                if on_progress:
                    on_progress(
                        {
                            "type": "sample_failed",
                            "retrieval": retrieval_key,
                            "llm": llm_key,
                            "llm_name": (
                                llm.model_name
                            ),
                            "question_index": idx,
                            "total_questions": (
                                len(questions)
                            ),
                            "question": q.question,
                            "error": str(exc),
                            "message": (
                                f"✗ [{retrieval_key} + "
                                f"{llm.model_name}] "
                                f"Q{idx}: "
                                f"Unexpected retrieval failure"
                            ),
                        }
                    )

                continue

            # ==================================================
            # 2. HANDLE EMPTY RETRIEVAL
            # ==================================================

            if not search_results:

                # ------------------------------------------------
                # STRICT HYBRID RULE
                # ------------------------------------------------
                #
                # Hybrid means:
                #
                # Dense retrieval
                #       +
                # BM25 retrieval
                #       ↓
                # RRF
                #
                # Therefore an empty hybrid result is treated
                # as a retrieval failure.
                #
                # We DO NOT use Tavily to hide this failure.
                # ------------------------------------------------

                if retrieval_key == "hybrid":

                    failed_samples += 1

                    logger.error(
                        f"[Evaluation] Hybrid retrieval "
                        f"returned no documents for "
                        f"Q{q.id}. "
                        f"Strict hybrid evaluation "
                        f"stops without Tavily fallback."
                    )

                    if on_progress:
                        on_progress(
                            {
                                "type": "sample_failed",
                                "retrieval": retrieval_key,
                                "llm": llm_key,
                                "llm_name": (
                                    llm.model_name
                                ),
                                "question_index": idx,
                                "total_questions": (
                                    len(questions)
                                ),
                                "question": q.question,
                                "error": (
                                    "Hybrid retrieval "
                                    "returned no documents."
                                ),
                                "message": (
                                    f"✗ [{retrieval_key} + "
                                    f"{llm.model_name}] "
                                    f"Q{idx}: "
                                    "Hybrid retrieval "
                                    "returned no documents"
                                ),
                            }
                        )

                    continue

                # ------------------------------------------------
                # VECTOR RETRIEVAL
                # ------------------------------------------------
                #
                # For vector retrieval, an empty corpus result
                # may use the configured Tavily fallback.
                #
                # This does not hide a Qdrant/ES backend exception,
                # because backend exceptions were already handled
                # above.
                # ------------------------------------------------

                logger.warning(
                    f"[Evaluation] "
                    f"{retrieval_key} returned "
                    f"no corpus documents "
                    f"for Q{q.id}; "
                    f"attempting Tavily fallback."
                )

                context_str = (
                    self._web_fallback(
                        q.question
                    )
                )

                web_fallback_used = bool(
                    context_str
                )

                if not context_str:

                    failed_samples += 1

                    logger.error(
                        f"[Evaluation] Q{q.id} failed: "
                        f"no corpus documents and "
                        f"Tavily returned no usable "
                        f"evidence."
                    )

                    if on_progress:
                        on_progress(
                            {
                                "type": "sample_failed",
                                "retrieval": retrieval_key,
                                "llm": llm_key,
                                "llm_name": (
                                    llm.model_name
                                ),
                                "question_index": idx,
                                "total_questions": (
                                    len(questions)
                                ),
                                "question": q.question,
                                "error": (
                                    "No corpus documents "
                                    "and no usable Tavily "
                                    "evidence."
                                ),
                                "message": (
                                    f"✗ [{retrieval_key} + "
                                    f"{llm.model_name}] "
                                    f"Q{idx}: "
                                    "No usable evidence"
                                ),
                            }
                        )

                    continue

            else:

                # ------------------------------------------------
                # NORMAL CORPUS CONTEXT
                # ------------------------------------------------

                retrieved_docs_models = (
                    convert_search_results(
                        search_results
                    )
                )

                context_str = (
                    format_context(
                        search_results
                    )
                )

            # ==================================================
            # 3. PROGRESS: GENERATION
            # ==================================================

            if on_progress:

                on_progress(
                    {
                        "type": "step",
                        "step": "generation",
                        "retrieval": retrieval_key,
                        "llm": llm_key,
                        "llm_name": (
                            llm.model_name
                        ),
                        "question_index": idx,
                        "total_questions": (
                            len(questions)
                        ),
                        "question": q.question,
                        "web_fallback_used": (
                            web_fallback_used
                        ),
                        "message": (
                            f"[{retrieval_key} + "
                            f"{llm.model_name}] "
                            f"Q{idx}: "
                            + (
                                "Generating response "
                                "using Tavily evidence..."
                                if web_fallback_used
                                else
                                "Generating response..."
                            )
                        ),
                    }
                )

            # ==================================================
            # 4. GENERATE ANSWER
            # ==================================================

            try:

                generated_answer = (
                    llm.generate(
                        query=q.question,
                        context=context_str,
                    )
                )

            except Exception as exc:

                failed_samples += 1

                elapsed_ms = (
                    time.time()
                    - start_time
                ) * 1000

                logger.error(
                    f"[Evaluation] Generation "
                    f"failed for Q{q.id}: {exc}"
                )

                if on_progress:
                    on_progress(
                        {
                            "type": "sample_failed",
                            "retrieval": retrieval_key,
                            "llm": llm_key,
                            "llm_name": (
                                llm.model_name
                            ),
                            "question_index": idx,
                            "total_questions": (
                                len(questions)
                            ),
                            "question": q.question,
                            "error": str(exc),
                            "message": (
                                f"✗ [{retrieval_key} + "
                                f"{llm.model_name}] "
                                f"Q{idx}: "
                                "Generation failed"
                            ),
                        }
                    )

                logger.error(
                    f"     └─ Generation failed "
                    f"after {elapsed_ms:.0f}ms"
                )

                continue

            # ==================================================
            # 5. PROGRESS: JUDGE
            # ==================================================

            if on_progress:

                on_progress(
                    {
                        "type": "step",
                        "step": "evaluation",
                        "retrieval": retrieval_key,
                        "llm": llm_key,
                        "llm_name": (
                            llm.model_name
                        ),
                        "question_index": idx,
                        "total_questions": (
                            len(questions)
                        ),
                        "question": q.question,
                        "web_fallback_used": (
                            web_fallback_used
                        ),
                        "message": (
                            f"[{retrieval_key} + "
                            f"{llm.model_name}] "
                            f"Q{idx}: "
                            "LLM-as-Judge evaluating..."
                        ),
                    }
                )

            # ==================================================
            # 6. LLM JUDGE
            # ==================================================

            try:

                judge_scores = (
                    self.judge.evaluate_sample(
                        question=q.question,
                        reference_answer=(
                            q.ground_truth
                        ),
                        context=(
                            context_str
                        ),
                        answer=(
                            generated_answer
                        ),
                    )
                )

            except Exception as exc:

                failed_samples += 1

                elapsed_ms = (
                    time.time()
                    - start_time
                ) * 1000

                logger.error(
                    f"[Evaluation] Judge failed "
                    f"for Q{q.id}: {exc}"
                )

                # IMPORTANT:
                #
                # Do NOT create:
                #
                # correctness=50
                # faithfulness=50
                # relevance=50
                # overall=50
                #
                # A failed judge sample is excluded from
                # benchmark averages.
                #

                if on_progress:
                    on_progress(
                        {
                            "type": "sample_failed",
                            "retrieval": retrieval_key,
                            "llm": llm_key,
                            "llm_name": (
                                llm.model_name
                            ),
                            "question_index": idx,
                            "total_questions": (
                                len(questions)
                            ),
                            "question": q.question,
                            "error": str(exc),
                            "message": (
                                f"✗ [{retrieval_key} + "
                                f"{llm.model_name}] "
                                f"Q{idx}: "
                                "LLM judge failed"
                            ),
                        }
                    )

                logger.error(
                    f"     └─ Judge failure "
                    f"after {elapsed_ms:.0f}ms"
                )

                continue

            # ==================================================
            # 7. RETRIEVAL METRICS
            # ==================================================

            try:

                retrieval_metrics = (
                    compute_retrieval_metrics(
                        retrieved_docs=(
                            retrieved_docs_models
                        ),
                        relevant_ids=(
                            q.relevant_documents
                        ),
                        k=self.top_k,
                    )
                )

            except Exception as exc:

                logger.warning(
                    f"[Evaluation] Retrieval "
                    f"metrics failed for Q{q.id}: "
                    f"{exc}"
                )

                # Preserve the judged sample even if the
                # optional retrieval metric calculation fails.
                retrieval_metrics = None

            # ==================================================
            # 8. EXECUTION TIME
            # ==================================================

            elapsed_ms = (
                time.time()
                - start_time
            ) * 1000

            # ==================================================
            # 9. SAMPLE RESULT
            # ==================================================

            sample_res = SampleResult(
                question_id=q.id,
                retrieval=retrieval_key,
                llm=llm_key,
                question=q.question,
                ground_truth=q.ground_truth,
                generated_answer=(
                    generated_answer
                ),
                retrieved_documents=(
                    retrieved_docs_models
                ),
                judge_scores=(
                    judge_scores
                ),
                retrieval_metrics=(
                    retrieval_metrics
                ),
                execution_time_ms=round(
                    elapsed_ms,
                    2,
                ),
                error=None,
            )

            results.append(
                sample_res
            )

            # ==================================================
            # 10. LOG SCORE
            # ==================================================

            logger.info(
                f"     └─ Score: "
                f"{judge_scores.overall:.1f}% "
                f"(C:{judge_scores.correctness:.1f}%, "
                f"F:{judge_scores.faithfulness:.1f}%, "
                f"R:{judge_scores.relevance:.1f}%) "
                f"| Time: {elapsed_ms:.0f}ms"
            )

            # ==================================================
            # 11. PROGRESS: SAMPLE COMPLETE
            # ==================================================

            if on_progress:

                on_progress(
                    {
                        "type": "sample_complete",
                        "retrieval": retrieval_key,
                        "llm": llm_key,
                        "llm_name": (
                            llm.model_name
                        ),
                        "question_index": idx,
                        "total_questions": (
                            len(questions)
                        ),
                        "question": q.question,
                        "score": (
                            judge_scores.overall
                        ),
                        "correctness": (
                            judge_scores.correctness
                        ),
                        "faithfulness": (
                            judge_scores.faithfulness
                        ),
                        "relevance": (
                            judge_scores.relevance
                        ),
                        "latency_ms": round(
                            elapsed_ms,
                            1,
                        ),
                        "web_fallback_used": (
                            web_fallback_used
                        ),
                        "message": (
                            f"✓ [{retrieval_key} + "
                            f"{llm.model_name}] "
                            f"Q{idx} Scored: "
                            f"{judge_scores.overall:.1f}% "
                            f"(Latency: "
                            f"{elapsed_ms:.0f}ms)"
                        ),
                    }
                )

        logger.info(
            f"<== Completed "
            f"[{retrieval_key} + "
            f"{llm.model_name}]: "
            f"{len(results)} successful, "
            f"{failed_samples} failed, "
            f"{len(questions)} total"
        )

        return results

    # ======================================================
    # ALL COMBINATIONS
    # ======================================================

    def run_all_combinations(
        self,
        questions: list[EvalQuestion],
        retrieval_keys: list[str] | None = None,
        llm_keys: list[str] | None = None,
        dataset_path: str = (
            "data/evaluation/"
            "evaluation_dataset.json"
        ),
        on_progress: Any | None = None,
    ) -> EvaluationReport:
        """
        Run the complete:

            Vector × Model 1
            Vector × Model 2
            Hybrid × Model 1
            Hybrid × Model 2

        benchmark.
        """

        r_keys = (
            retrieval_keys
            or list(
                self.retrievers.keys()
            )
        )

        l_keys = (
            llm_keys
            or list(
                self.llm_clients.keys()
            )
        )

        all_detailed_results: list[
            SampleResult
        ] = []

        summaries: dict[
            str,
            ConfigSummary,
        ] = {}

        matrix: dict[
            str,
            dict[str, float],
        ] = {}

        total_combinations = (
            len(r_keys)
            * len(l_keys)
        )

        combination_index = 0

        # ==================================================
        # CONFIGURATIONS
        # ==================================================

        for retrieval_key in r_keys:

            if retrieval_key not in self.retrievers:
                raise ValueError(
                    f"Unknown retrieval "
                    f"configuration: "
                    f"{retrieval_key}"
                )

            if retrieval_key not in matrix:
                matrix[
                    retrieval_key
                ] = {}

            for llm_key in l_keys:

                if llm_key not in self.llm_clients:
                    raise ValueError(
                        f"Unknown LLM "
                        f"configuration: "
                        f"{llm_key}"
                    )

                combination_index += 1

                combination_key = (
                    f"{retrieval_key}+"
                    f"{llm_key}"
                )

                llm_name = (
                    self.llm_clients[
                        llm_key
                    ].model_name
                )

                # ------------------------------------------
                # COMBINATION START
                # ------------------------------------------

                if on_progress:

                    on_progress(
                        {
                            "type": (
                                "combination_start"
                            ),
                            "combination": (
                                combination_key
                            ),
                            "combination_index": (
                                combination_index
                            ),
                            "total_combinations": (
                                total_combinations
                            ),
                            "retrieval": (
                                retrieval_key
                            ),
                            "llm": (
                                llm_key
                            ),
                            "llm_name": (
                                llm_name
                            ),
                            "message": (
                                f"▶ Starting "
                                f"Configuration "
                                f"[{combination_index}/"
                                f"{total_combinations}]: "
                                f"{retrieval_key.upper()} + "
                                f"{llm_name}"
                            ),
                        }
                    )

                # ------------------------------------------
                # RUN CONFIGURATION
                # ------------------------------------------

                combination_results = (
                    self.run_single_combination(
                        retrieval_key=(
                            retrieval_key
                        ),
                        llm_key=(
                            llm_key
                        ),
                        questions=(
                            questions
                        ),
                        on_progress=(
                            on_progress
                        ),
                    )
                )

                all_detailed_results.extend(
                    combination_results
                )

                # ------------------------------------------
                # AGGREGATION
                # ------------------------------------------

                n = len(
                    combination_results
                )

                if n > 0:

                    avg_correctness = (
                        sum(
                            result
                            .judge_scores
                            .correctness
                            for result
                            in combination_results
                        )
                        / n
                    )

                    avg_faithfulness = (
                        sum(
                            result
                            .judge_scores
                            .faithfulness
                            for result
                            in combination_results
                        )
                        / n
                    )

                    avg_relevance = (
                        sum(
                            result
                            .judge_scores
                            .relevance
                            for result
                            in combination_results
                        )
                        / n
                    )

                    avg_overall = (
                        sum(
                            result
                            .judge_scores
                            .overall
                            for result
                            in combination_results
                        )
                        / n
                    )

                    avg_latency = (
                        sum(
                            result
                            .execution_time_ms
                            for result
                            in combination_results
                        )
                        / n
                    )

                    # --------------------------------------
                    # RETRIEVAL METRICS
                    # --------------------------------------

                    precision_values = [
                        result
                        .retrieval_metrics
                        .precision_at_k
                        for result
                        in combination_results
                        if (
                            result
                            .retrieval_metrics
                            is not None
                            and
                            result
                            .retrieval_metrics
                            .precision_at_k
                            is not None
                        )
                    ]

                    recall_values = [
                        result
                        .retrieval_metrics
                        .recall_at_k
                        for result
                        in combination_results
                        if (
                            result
                            .retrieval_metrics
                            is not None
                            and
                            result
                            .retrieval_metrics
                            .recall_at_k
                            is not None
                        )
                    ]

                    mrr_values = [
                        result
                        .retrieval_metrics
                        .mrr
                        for result
                        in combination_results
                        if (
                            result
                            .retrieval_metrics
                            is not None
                            and
                            result
                            .retrieval_metrics
                            .mrr
                            is not None
                        )
                    ]

                    avg_precision = (
                        sum(
                            precision_values
                        )
                        / len(
                            precision_values
                        )
                        if precision_values
                        else None
                    )

                    avg_recall = (
                        sum(
                            recall_values
                        )
                        / len(
                            recall_values
                        )
                        if recall_values
                        else None
                    )

                    avg_mrr = (
                        sum(
                            mrr_values
                        )
                        / len(
                            mrr_values
                        )
                        if mrr_values
                        else None
                    )

                else:

                    avg_correctness = 0.0
                    avg_faithfulness = 0.0
                    avg_relevance = 0.0
                    avg_overall = 0.0
                    avg_latency = 0.0

                    avg_precision = None
                    avg_recall = None
                    avg_mrr = None

                # ------------------------------------------
                # CONFIG SUMMARY
                # ------------------------------------------

                summary = ConfigSummary(
                    retrieval=(
                        retrieval_key
                    ),
                    llm=(
                        llm_key
                    ),
                    total_samples=n,

                    avg_correctness=round(
                        avg_correctness,
                        2,
                    ),

                    avg_faithfulness=round(
                        avg_faithfulness,
                        2,
                    ),

                    avg_relevance=round(
                        avg_relevance,
                        2,
                    ),

                    avg_overall=round(
                        avg_overall,
                        2,
                    ),

                    avg_precision_at_k=(
                        round(
                            avg_precision,
                            4,
                        )
                        if avg_precision
                        is not None
                        else None
                    ),

                    avg_recall_at_k=(
                        round(
                            avg_recall,
                            4,
                        )
                        if avg_recall
                        is not None
                        else None
                    ),

                    avg_mrr=(
                        round(
                            avg_mrr,
                            4,
                        )
                        if avg_mrr
                        is not None
                        else None
                    ),

                    avg_latency_ms=round(
                        avg_latency,
                        2,
                    ),
                )

                summaries[
                    combination_key
                ] = summary

                matrix[
                    retrieval_key
                ][
                    llm_key
                ] = round(
                    avg_overall,
                    2,
                )

                # ------------------------------------------
                # PROGRESS: COMBINATION COMPLETE
                # ------------------------------------------

                if on_progress:

                    on_progress(
                        {
                            "type": (
                                "combination_complete"
                            ),
                            "combination": (
                                combination_key
                            ),
                            "combination_index": (
                                combination_index
                            ),
                            "total_combinations": (
                                total_combinations
                            ),
                            "retrieval": (
                                retrieval_key
                            ),
                            "llm": (
                                llm_key
                            ),
                            "llm_name": (
                                llm_name
                            ),
                            "successful_samples": (
                                n
                            ),
                            "total_questions": (
                                len(questions)
                            ),
                            "score": (
                                round(
                                    avg_overall,
                                    2,
                                )
                            ),
                            "message": (
                                f"✓ Completed "
                                f"{combination_key}: "
                                f"{n}/{len(questions)} "
                                f"samples evaluated"
                            ),
                        }
                    )

        # ==================================================
        # BEST CONFIGURATION
        # ==================================================

        best_key = ""

        best_score = -1.0

        for key, summary in (
            summaries.items()
        ):

            # Do not call a configuration with zero
            # successfully evaluated samples a valid winner.
            if summary.total_samples <= 0:
                continue

            if (
                summary.avg_overall
                > best_score
            ):

                best_score = (
                    summary.avg_overall
                )

                best_key = key

        best_summary = (
            summaries.get(
                best_key
            )
            if best_key
            else None
        )

        best_reason = ""

        if best_summary:

            retrieval_name = (
                best_summary
                .retrieval
                .capitalize()
            )

            llm_name = (
                best_summary
                .llm
            )

            best_reason = (
                f"{retrieval_name} Retrieval + "
                f"{llm_name} achieved the "
                f"highest composite score "
                f"among configurations with "
                f"successful judge evaluations "
                f"({best_summary.avg_overall}%) "
                f"with "
                f"Correctness="
                f"{best_summary.avg_correctness}%, "
                f"Faithfulness="
                f"{best_summary.avg_faithfulness}%, "
                f"Relevance="
                f"{best_summary.avg_relevance}%, "
                f"and average latency of "
                f"{best_summary.avg_latency_ms:.1f}ms."
            )

        else:

            best_reason = (
                "No configuration produced "
                "a successful judged sample."
            )

        # ==================================================
        # REPORT
        # ==================================================

        report = EvaluationReport(
            timestamp=(
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),

            dataset_path=(
                dataset_path
            ),

            judge_model=(
                self.judge
                .judge_model_name
            ),

            top_k=self.top_k,

            configurations=(
                summaries
            ),

            best_configuration=(
                best_key
            ),

            best_reason=(
                best_reason
            ),

            comparison_matrix=(
                matrix
            ),

            detailed_results=(
                all_detailed_results
            ),
        )

        # ==================================================
        # SAVE
        # ==================================================

        self.save_results(
            report
        )

        # ==================================================
        # PRINT
        # ==================================================

        self.print_report(
            report
        )

        return report

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    def save_results(
        self,
        report: EvaluationReport,
    ) -> None:
        """
        Save evaluation results.

        Files:

            results.json
            results_<timestamp>.json
            summary.json
        """

        timestamp_str = (
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        # --------------------------------------------------
        # Latest results
        # --------------------------------------------------

        latest_results_path = (
            self.output_dir
            / "results.json"
        )

        with open(
            latest_results_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report.model_dump(),
                f,
                indent=2,
                ensure_ascii=False,
            )

        # --------------------------------------------------
        # Timestamped results
        # --------------------------------------------------

        timestamped_results_path = (
            self.output_dir
            / f"results_{timestamp_str}.json"
        )

        with open(
            timestamped_results_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report.model_dump(),
                f,
                indent=2,
                ensure_ascii=False,
            )

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

        summary_path = (
            self.output_dir
            / "summary.json"
        )

        summary_data = {
            "timestamp": (
                report.timestamp
            ),

            "dataset_path": (
                report.dataset_path
            ),

            "judge_model": (
                report.judge_model
            ),

            "best_configuration": (
                report.best_configuration
            ),

            "best_reason": (
                report.best_reason
            ),

            "comparison_matrix": (
                report.comparison_matrix
            ),

            "configurations": {
                key: value.model_dump()
                for key, value in (
                    report
                    .configurations
                    .items()
                )
            },
        }

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                summary_data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "Saved evaluation results to "
            f"{latest_results_path} and "
            f"{summary_path}"
        )

    # ======================================================
    # PRINT REPORT
    # ======================================================

    def print_report(
        self,
        report: EvaluationReport,
    ) -> None:
        """
        Print formatted CLI evaluation summary.
        """

        print(
            "\n"
            + "=" * 100
        )

        print(
            "       2 RETRIEVAL TECHNIQUES × "
            "2 LLMS EVALUATION REPORT"
        )

        print(
            "=" * 100
        )

        print(
            f"Dataset:      "
            f"{report.dataset_path}"
        )

        print(
            f"Judge Model:  "
            f"{report.judge_model}"
        )

        print(
            f"Top-K:        "
            f"{report.top_k}"
        )

        print(
            f"Timestamp:    "
            f"{report.timestamp}"
        )

        print(
            "-" * 100
        )

        # --------------------------------------------------
        # 2 × 2 MATRIX
        # --------------------------------------------------

        print(
            "\n2 × 2 OVERALL SCORE MATRIX "
            "(Composite Score 0-100%):"
        )

        print(
            "-" * 60
        )

        print(
            f"{'LLM':<18} | "
            f"{'Vector Retrieval':<18} | "
            f"{'Hybrid Retrieval':<18}"
        )

        print(
            "-" * 60
        )

        for llm_key in [
            "model_1",
            "model_2",
        ]:

            vector_score = (
                report
                .comparison_matrix
                .get(
                    "vector",
                    {},
                )
                .get(
                    llm_key,
                    0.0,
                )
            )

            hybrid_score = (
                report
                .comparison_matrix
                .get(
                    "hybrid",
                    {},
                )
                .get(
                    llm_key,
                    0.0,
                )
            )

            print(
                f"{llm_key:<18} | "
                f"{vector_score:>16.1f}% | "
                f"{hybrid_score:>16.1f}%"
            )

        print(
            "-" * 60
        )

        # --------------------------------------------------
        # DETAILED TABLE
        # --------------------------------------------------

        print(
            "\nDETAILED CONFIGURATION COMPARISON:"
        )

        print(
            "-" * 120
        )

        header = (
            f"{'Configuration':<22} | "
            f"{'Samples':<8} | "
            f"{'Correctness':<11} | "
            f"{'Faithful':<9} | "
            f"{'Relevance':<9} | "
            f"{'Overall':<8} | "
            f"{'P@K':<6} | "
            f"{'R@K':<6} | "
            f"{'MRR':<6} | "
            f"{'Latency':<8}"
        )

        print(header)

        print(
            "-" * 120
        )

        for (
            configuration_name,
            summary,
        ) in report.configurations.items():

            precision = (
                f"{summary.avg_precision_at_k:.2f}"
                if (
                    summary.avg_precision_at_k
                    is not None
                )
                else "N/A"
            )

            recall = (
                f"{summary.avg_recall_at_k:.2f}"
                if (
                    summary.avg_recall_at_k
                    is not None
                )
                else "N/A"
            )

            mrr = (
                f"{summary.avg_mrr:.2f}"
                if (
                    summary.avg_mrr
                    is not None
                )
                else "N/A"
            )

            row = (
                f"{configuration_name:<22} | "
                f"{summary.total_samples:<8} | "
                f"{summary.avg_correctness:>10.1f}% | "
                f"{summary.avg_faithfulness:>8.1f}% | "
                f"{summary.avg_relevance:>8.1f}% | "
                f"{summary.avg_overall:>7.1f}% | "
                f"{precision:>6} | "
                f"{recall:>6} | "
                f"{mrr:>6} | "
                f"{summary.avg_latency_ms:>6.0f}ms"
            )

            print(row)

        print(
            "-" * 120
        )

        # --------------------------------------------------
        # BEST CONFIGURATION
        # --------------------------------------------------

        print(
            "\nBEST OVERALL CONFIGURATION:"
        )

        if report.best_configuration:

            print(
                f"   --> "
                f"{report.best_configuration.upper()} "
                f"<--"
            )

        else:

            print(
                "   No configuration had "
                "successful evaluations."
            )

        print(
            f"   {report.best_reason}"
        )

        print(
            "=" * 100
            + "\n"
        )


# ==========================================================
# CLI
# ==========================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "2 Retrievers x 2 LLMs "
            "RAG Evaluation Runner"
        )
    )

    # ------------------------------------------------------
    # DATASET
    # ------------------------------------------------------

    parser.add_argument(
        "--dataset",
        type=str,
        default=(
            "data/evaluation/"
            "evaluation_dataset.json"
        ),
        help=(
            "Path to evaluation "
            "dataset JSON"
        ),
    )

    # ------------------------------------------------------
    # TOP-K
    # ------------------------------------------------------

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Top-K documents to retrieve"
        ),
    )

    # ------------------------------------------------------
    # RETRIEVAL
    # ------------------------------------------------------

    parser.add_argument(
        "--retrieval",
        type=str,
        choices=[
            "vector",
            "hybrid",
            "both",
        ],
        default="both",
        help=(
            "Retrieval strategy"
        ),
    )

    # ------------------------------------------------------
    # LLM
    # ------------------------------------------------------

    parser.add_argument(
        "--llm",
        type=str,
        choices=[
            "model_1",
            "model_2",
            "both",
        ],
        default="both",
        help=(
            "Benchmark LLM"
        ),
    )

    # ------------------------------------------------------
    # JUDGE
    # ------------------------------------------------------

    parser.add_argument(
        "--judge",
        type=str,
        default="openrouter",
        choices=[
            "openrouter",
        ],
        help=(
            "LLM judge provider. "
            "OpenRouter is the production "
            "evaluation judge."
        ),
    )

    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "data/evaluation/"
            "results"
        ),
        help=(
            "Output directory"
        ),
    )

    # ------------------------------------------------------
    # MAX QUESTIONS
    # ------------------------------------------------------

    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=(
            "Optional limit on "
            "number of questions"
        ),
    )

    args = parser.parse_args()

    # ======================================================
    # DATASET
    # ======================================================

    questions = load_dataset(
        args.dataset
    )

    if args.max_questions:

        questions = questions[
            :args.max_questions
        ]

    if not questions:

        raise ValueError(
            "No evaluation questions "
            "were loaded from the dataset."
        )

    logger.info(
        f"Loaded {len(questions)} "
        f"evaluation questions from "
        f"{args.dataset}"
    )

    # ======================================================
    # RETRIEVAL CONFIGURATION
    # ======================================================

    retrieval_keys = (
        [
            "vector",
            "hybrid",
        ]
        if args.retrieval == "both"
        else [
            args.retrieval
        ]
    )

    # ======================================================
    # LLM CONFIGURATION
    # ======================================================

    llm_keys = (
        [
            "model_1",
            "model_2",
        ]
        if args.llm == "both"
        else [
            args.llm
        ]
    )

    # ======================================================
    # API KEYS
    # ======================================================
    #
    # CLI evaluation reads keys from environment variables.
    #
    # Public web evaluation does NOT use this function.
    # The API route passes visitor-provided keys directly
    # into EvaluationService -> EvaluationRunner.
    #
    # ======================================================

    groq_api_key = (
        os.getenv(
            "GROQ_API_KEY"
        )
        or ""
    ).strip()

    openrouter_api_key = (
        os.getenv(
            "OPENROUTER_API_KEY"
        )
        or ""
    ).strip()

    tavily_api_key = (
        os.getenv(
            "TAVILY_API_KEY"
        )
        or ""
    ).strip()

    # ======================================================
    # RUNNER
    # ======================================================

    runner = EvaluationRunner(
        output_dir=args.output,
        top_k=args.top_k,

        groq_api_key=(
            groq_api_key
        ),

        openrouter_api_key=(
            openrouter_api_key
        ),

        tavily_api_key=(
            tavily_api_key
        ),
    )

    # ======================================================
    # RUN
    # ======================================================

    runner.run_all_combinations(
        questions=questions,
        retrieval_keys=(
            retrieval_keys
        ),
        llm_keys=(
            llm_keys
        ),
        dataset_path=(
            args.dataset
        ),
    )


if __name__ == "__main__":
    main()
