"""Rerank ordering helpers for retrieval runs."""

from __future__ import annotations

from typing import Any, Protocol

from eval_platform.retrieval.clients import RerankClient
from eval_platform.retrieval.errors import RetrievalRunError
from eval_platform.retrieval.schema import RetrievalHit


class RerankFlowConfig(Protocol):
    rerank_enabled: bool
    rerank_candidate_cap: int
    rerank_cross_path_topk: int


def maybe_rerank(
    query_text: str,
    candidates: list[RetrievalHit],
    config: RerankFlowConfig,
    rerank_client: RerankClient | None,
    trace: dict[str, Any],
) -> list[RetrievalHit]:
    """Optionally rerank the highest-scoring candidate head and preserve the tail."""

    if not config.rerank_enabled:
        return candidates
    if rerank_client is None:
        raise RetrievalRunError("rerank_client is required when rerank_enabled=True")
    ordered = sorted(candidates, key=lambda hit: (-hit.score, hit.chunk_id))
    head = (
        ordered[: config.rerank_candidate_cap]
        if config.rerank_candidate_cap > 0
        else ordered
    )
    tail = ordered[len(head) :]
    trace["rerank_input"] = [hit.model_dump(mode="json") for hit in head]
    top_n = config.rerank_cross_path_topk if config.rerank_cross_path_topk > 0 else len(head)
    reranked = rerank_client.rerank(query_text, head, top_n)
    trace["rerank_hits"] = [hit.model_dump(mode="json") for hit in reranked]
    reranked_ids = {hit.chunk_id for hit in reranked}
    return list(reranked) + [hit for hit in tail if hit.chunk_id not in reranked_ids]


def rank_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """Assign stable 1-based ranks without mutating source hit objects."""

    return [hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(hits, start=1)]
