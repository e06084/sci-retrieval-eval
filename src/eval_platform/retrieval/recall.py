"""ES, Milvus, and hybrid recall helpers for retrieval runs."""

from __future__ import annotations

from typing import Literal, Protocol

from eval_platform.embeddings import EmbeddingClient
from eval_platform.retrieval.clients import (
    ElasticsearchRetrievalClient,
    MilvusRetrievalClient,
)
from eval_platform.retrieval.errors import RetrievalRunError
from eval_platform.retrieval.fusion import rrf_fuse
from eval_platform.retrieval.schema import RetrievalHit


class RecallConfig(Protocol):
    retrieval_mode: Literal["es", "milvus", "hybrid"]
    top_k: int
    index_name: str | None
    collection_name: str | None
    hybrid_per_source_topk: int
    rrf_path_topk: int


def recall_one(
    *,
    query: str,
    config: RecallConfig,
    es_client: ElasticsearchRetrievalClient | None,
    milvus_client: MilvusRetrievalClient | None,
    embedding_client: EmbeddingClient | None,
    vector: list[float] | None = None,
) -> tuple[list[RetrievalHit], list[RetrievalHit], list[RetrievalHit], list[RetrievalHit]]:
    """Run one query path through the configured recall mode."""

    if es_client is None or config.index_name is None:
        raise RetrievalRunError("es_client and index_name are required")
    if config.retrieval_mode == "es":
        es_hits = es_client.search_bm25(config.index_name, query, config.top_k)
        return es_hits, es_hits, [], es_hits

    if milvus_client is None or config.collection_name is None:
        raise RetrievalRunError("milvus_client and collection_name are required")
    if vector is None:
        if embedding_client is None:
            raise RetrievalRunError("embedding_client is required")
        vector = embedding_client.embed_texts([query])[0]

    milvus_top_k = (
        config.top_k
        if config.retrieval_mode == "milvus"
        else max(config.hybrid_per_source_topk, config.rrf_path_topk)
    )
    milvus_hits = milvus_client.search(config.collection_name, vector, milvus_top_k)
    enriched_milvus_hits = es_client.enrich_by_chunk_ids(config.index_name, milvus_hits)
    if config.retrieval_mode == "milvus":
        return enriched_milvus_hits, [], milvus_hits, enriched_milvus_hits

    es_top_k = max(config.hybrid_per_source_topk, config.rrf_path_topk)
    es_hits = es_client.search_bm25(config.index_name, query, es_top_k)
    fused_hits = rrf_fuse(enriched_milvus_hits, es_hits, config.rrf_path_topk)
    enriched_fused_hits = es_client.enrich_by_chunk_ids(config.index_name, fused_hits)
    return enriched_fused_hits, es_hits, milvus_hits, enriched_fused_hits
