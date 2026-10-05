from unittest.mock import MagicMock

import pytest

from src.retrieval.errors import BM25RetrievalError, DenseRetrievalError
from src.retrieval.hybrid_retriever import HybridRetriever, RetrievalConfig
from src.retrieval.retrieval_result import RetrievalResult
from src.agent.workflow import route_after_retrieval


def result(doc_id: int, source: str) -> RetrievalResult:
    return RetrievalResult(
        id=doc_id,
        content=f"content {doc_id}",
        title=f"paper {doc_id}",
        score=1.0,
        source=source,
        metadata={"paper_id": f"p{doc_id}"},
    )


def make_retriever(dense, lexical) -> HybridRetriever:
    dense_retriever = MagicMock()
    dense_retriever.search.side_effect = dense if isinstance(dense, Exception) else None
    if not isinstance(dense, Exception):
        dense_retriever.search.return_value = dense
    lexical_retriever = MagicMock()
    lexical_retriever.search.side_effect = lexical if isinstance(lexical, Exception) else None
    if not isinstance(lexical, Exception):
        lexical_retriever.search.return_value = lexical
    return HybridRetriever(
        dense_retriever=dense_retriever,
        lexical_retriever=lexical_retriever,
        config=RetrievalConfig(reranker_enabled=False),
    )


def test_both_backends_succeed_then_rrf_runs():
    retriever = make_retriever(
        [result(1, "dense"), result(2, "dense")],
        [result(2, "bm25"), result(3, "bm25")],
    )
    outcome = retriever.search_with_status("hybrid retrieval", top_k=3)

    assert outcome.succeeded is True
    assert outcome.retrieval_status["dense"] == "success"
    assert outcome.retrieval_status["bm25"] == "success"
    assert outcome.retrieval_status["rrf"] == "success"
    shared = next(item for item in outcome.results if item.id == 2)
    assert shared.retrieval_sources == ("dense", "bm25")
    assert shared.dense_rank == 2
    assert shared.bm25_rank == 1


def test_dense_success_bm25_failure_stops_before_rrf():
    retriever = make_retriever(
        [result(1, "dense")], BM25RetrievalError("BM25 unavailable")
    )
    outcome = retriever.search_with_status("query")

    assert outcome.results == []
    assert outcome.retrieval_status == {
        "dense": "success",
        "bm25": "failed",
        "rrf": "not_run",
        "reranker": "not_run",
    }


def test_dense_failure_stops_before_bm25_and_rrf():
    retriever = make_retriever(
        DenseRetrievalError("Qdrant unavailable"), [result(1, "bm25")]
    )
    outcome = retriever.search_with_status("query")

    assert outcome.results == []
    assert outcome.retrieval_status["dense"] == "failed"
    assert outcome.retrieval_status["bm25"] == "not_run"
    assert outcome.retrieval_status["rrf"] == "not_run"
    retriever.lexical_retriever.search.assert_not_called()


def test_both_unavailable_returns_no_results():
    retriever = make_retriever(
        DenseRetrievalError("Qdrant unavailable"),
        BM25RetrievalError("Elasticsearch unavailable"),
    )
    outcome = retriever.search_with_status("query")

    assert outcome.results == []
    assert outcome.succeeded is False
    retriever.lexical_retriever.search.assert_not_called()


def test_langgraph_stops_before_validation_when_hybrid_failed():
    assert route_after_retrieval(
        {
            "document": [],
            "retrieval_status": {
                "dense": "success",
                "bm25": "failed",
                "rrf": "not_run",
            },
        }
    ) == "generate"
