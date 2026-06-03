"""Shared defaults for Sciverse benchmark v1."""

from __future__ import annotations

from typing import Any

DEFAULT_RETRIEVAL_TOP_K = 100
DEFAULT_HYBRID_PER_SOURCE_TOPK = 50
DEFAULT_RRF_PATH_TOPK = 25
DEFAULT_RERANK_CROSS_PATH_TOPK = 50
DEFAULT_RERANK_CANDIDATE_CAP = 0

DEFAULT_MILVUS_PRIMARY_KEY_FIELD = "chunk_id"
DEFAULT_MILVUS_VECTOR_FIELD = "vector"
DEFAULT_MILVUS_METRIC_TYPE = "COSINE"
DEFAULT_MILVUS_INDEX_TYPE = "HNSW"
DEFAULT_MILVUS_HNSW_PARAMS = {"M": 16, "efConstruction": 200}
DEFAULT_MILVUS_SEARCH_PARAMS: dict[str, Any] = {}
DEFAULT_MILVUS_TITLE_MAX_LENGTH = 65535
DEFAULT_MILVUS_TEXT_MAX_LENGTH = 65535


def default_milvus_index_params() -> dict[str, Any]:
    """Return fresh default Milvus HNSW index params."""

    return {
        "index_type": DEFAULT_MILVUS_INDEX_TYPE,
        "metric_type": DEFAULT_MILVUS_METRIC_TYPE,
        "params": dict(DEFAULT_MILVUS_HNSW_PARAMS),
    }


def default_milvus_search_params(
    *,
    metric_type: str = DEFAULT_MILVUS_METRIC_TYPE,
) -> dict[str, Any]:
    """Return fresh default Milvus vector search params."""

    return {
        "metric_type": metric_type,
        "params": dict(DEFAULT_MILVUS_SEARCH_PARAMS),
    }
