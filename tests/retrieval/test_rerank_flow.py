"""Tests for retrieval rerank flow helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from eval_platform.retrieval.rerank_flow import maybe_rerank, rank_hits
from eval_platform.retrieval.schema import RetrievalHit


@dataclass
class RerankConfig:
    rerank_enabled: bool = True
    rerank_candidate_cap: int = 2
    rerank_cross_path_topk: int = 2


class FakeRerankClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        self.calls.append((query, [hit.chunk_id for hit in hits], top_n))
        return [
            hit.model_copy(update={"score": 100.0 - index})
            for index, hit in enumerate(reversed(hits[:top_n]), start=1)
        ]


def _hit(chunk_id: str, score: float) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=f"text {chunk_id}",
        score=score,
        recall_source="test",
    )


def test_maybe_rerank_caps_head_preserves_tail_and_dedupes_reranked_tail_ids() -> None:
    rerank = FakeRerankClient()
    trace: dict[str, Any] = {}

    result = maybe_rerank(
        "alpha",
        [_hit("a", 10.0), _hit("b", 9.0), _hit("c", 8.0), _hit("a", 1.0)],
        RerankConfig(),
        rerank,
        trace,
    )

    assert rerank.calls == [("alpha", ["a", "b"], 2)]
    assert [hit.chunk_id for hit in result] == ["b", "a", "c"]
    assert [hit["chunk_id"] for hit in trace["rerank_input"]] == ["a", "b"]
    assert [hit["chunk_id"] for hit in trace["rerank_hits"]] == ["b", "a"]


def test_maybe_rerank_returns_candidates_when_disabled() -> None:
    candidates = [_hit("a", 10.0), _hit("b", 9.0)]

    assert maybe_rerank(
        "alpha",
        candidates,
        RerankConfig(rerank_enabled=False),
        rerank_client=None,
        trace={},
    ) == candidates


def test_rank_hits_assigns_stable_one_based_ranks() -> None:
    ranked = rank_hits([_hit("a", 10.0), _hit("b", 9.0)])

    assert [hit.rank for hit in ranked] == [1, 2]
