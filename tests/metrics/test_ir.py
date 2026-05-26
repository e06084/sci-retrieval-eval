"""Tests for IR metric formulas."""

import pytest

from eval_platform.metrics import RankedDoc, aggregate_query_metrics, compute_query_metrics


def _doc(rank: int, doc_id: str) -> RankedDoc:
    return RankedDoc(
        rank=rank,
        doc_id=doc_id,
        score=1.0 / rank,
        source_chunk_id=f"chunk-{doc_id}",
        source_chunk_rank=rank,
        source_chunk_score=10.0 - rank,
    )


def test_compute_query_metrics_perfect_ranking() -> None:
    metrics = compute_query_metrics(
        [_doc(1, "doc-1"), _doc(2, "doc-2")],
        {"doc-1": 1.0, "doc-2": 1.0},
        [1, 2],
    )

    assert metrics["precision_at_1"] == 1.0
    assert metrics["recall_at_1"] == 0.5
    assert metrics["hit_rate_at_1"] == 1.0
    assert metrics["mrr_at_1"] == 1.0
    assert metrics["map_at_2"] == 1.0
    assert metrics["ndcg_at_2"] == 1.0


def test_compute_query_metrics_no_hit() -> None:
    metrics = compute_query_metrics([], {"doc-1": 1.0}, [3])

    assert metrics["precision_at_3"] == 0.0
    assert metrics["recall_at_3"] == 0.0
    assert metrics["hit_rate_at_3"] == 0.0
    assert metrics["mrr_at_3"] == 0.0
    assert metrics["map_at_3"] == 0.0
    assert metrics["ndcg_at_3"] == 0.0


def test_compute_query_metrics_partial_hit() -> None:
    metrics = compute_query_metrics(
        [_doc(1, "doc-x"), _doc(2, "doc-2")],
        {"doc-1": 1.0, "doc-2": 1.0},
        [2],
    )

    assert metrics["precision_at_2"] == 0.5
    assert metrics["recall_at_2"] == 0.5
    assert metrics["mrr_at_2"] == 0.5
    assert metrics["map_at_2"] == 0.25


def test_compute_query_metrics_uses_graded_relevance_for_ndcg() -> None:
    metrics = compute_query_metrics(
        [_doc(1, "doc-low"), _doc(2, "doc-high")],
        {"doc-high": 3.0, "doc-low": 1.0},
        [2],
    )

    expected = (1.0 + 3.0 / 1.584962500721156) / (3.0 + 1.0 / 1.584962500721156)
    assert metrics["ndcg_at_2"] == pytest.approx(expected)


def test_aggregate_query_metrics_averages_queries() -> None:
    aggregate = aggregate_query_metrics(
        [
            {"ndcg_at_1": 1.0, "precision_at_1": 1.0},
            {"ndcg_at_1": 0.0, "precision_at_1": 0.0},
        ],
        [1],
    )

    assert aggregate["ndcg_at_1"] == 0.5
    assert aggregate["precision_at_1"] == 0.5
