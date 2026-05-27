"""Trace assembly helpers for retrieval runs."""

from __future__ import annotations

from typing import Any

from eval_platform.retrieval.schema import RetrievalHit


def hits_trace(hits: list[RetrievalHit]) -> list[dict[str, Any]]:
    """Serialize hits for replay trace payloads."""

    return [hit.model_dump(mode="json") for hit in hits]


def new_live_trace(queries: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create a live query trace and return its mutable per-query list."""

    per_query_trace: list[dict[str, Any]] = []
    trace: dict[str, Any] = {
        "rewrite_queries": queries,
        "per_query": per_query_trace,
        "rerank_input": [],
        "rerank_hits": [],
    }
    return trace, per_query_trace


def append_recall_trace(
    per_query_trace: list[dict[str, Any]],
    *,
    query: str,
    es_hits: list[RetrievalHit],
    milvus_hits: list[RetrievalHit],
    fused_hits: list[RetrievalHit],
) -> None:
    """Append one query-path recall trace record."""

    per_query_trace.append(
        {
            "query": query,
            "es_hits": hits_trace(es_hits),
            "milvus_hits": hits_trace(milvus_hits),
            "fused_hits": hits_trace(fused_hits),
        }
    )


def build_error_trace(query_text: str, exc: Exception) -> dict[str, Any]:
    """Build the replay trace stored when one query fails."""

    return {
        "rewrite_queries": [query_text.strip()],
        "per_query": [],
        "rerank_input": [],
        "rerank_hits": [],
        "final_hits": [],
        "error": str(exc),
        "error_stage": "unknown",
    }
