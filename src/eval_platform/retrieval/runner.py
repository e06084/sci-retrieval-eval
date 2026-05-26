"""Retrieval run orchestration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest, ArtifactStore
from eval_platform.datasets import (
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    read_normalized_dataset_artifact,
)
from eval_platform.embeddings import EmbeddingClient
from eval_platform.indexes import ELASTICSEARCH_INDEX_ARTIFACT_TYPE, MILVUS_COLLECTION_ARTIFACT_TYPE
from eval_platform.retrieval.artifact import (
    RETRIEVAL_RUN_ARTIFACT_TYPE,
    read_retrieval_run_artifact,
    write_retrieval_run_artifact,
)
from eval_platform.retrieval.clients import (
    ElasticsearchRetrievalClient,
    MilvusRetrievalClient,
    RerankClient,
    RewriteClient,
)
from eval_platform.retrieval.fusion import dedupe_by_chunk_id, dedupe_sequential, rrf_fuse
from eval_platform.retrieval.schema import RetrievalHit, RetrievalQueryResult


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class RetrievalRunError(Exception):
    """Raised when a retrieval run cannot be executed."""


class RetrievalRunConfig(BaseModel):
    """Configuration for a retrieval_run artifact."""

    source_normalized_dataset_artifact_id: str
    output_artifact_id: str
    retrieval_mode: Literal["es", "milvus", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, gt=0)
    query_limit: int | None = Field(default=None, gt=0)
    queries_per_shard: int = Field(default=1000, gt=0)
    trace_mode: Literal["replay", "none"] = "replay"
    execution_mode: Literal["live", "replay"] = "live"
    replay_source_retrieval_run_artifact_id: str | None = None

    elasticsearch_index_artifact_id: str | None = None
    milvus_collection_artifact_id: str | None = None

    index_name: str | None = None
    collection_name: str | None = None

    sub_queries: int = Field(default=0, ge=0)
    rewrite_enabled: bool = False
    rerank_enabled: bool = False
    hybrid_per_source_topk: int = Field(default=50, gt=0)
    rrf_path_topk: int = Field(default=10, gt=0)
    rerank_cross_path_topk: int = Field(default=50, ge=0)
    rerank_candidate_cap: int = Field(default=0, ge=0)

    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_normalized_dataset_artifact_id",
        "output_artifact_id",
    )
    @classmethod
    def validate_required_ids(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator(
        "elasticsearch_index_artifact_id",
        "milvus_collection_artifact_id",
        "replay_source_retrieval_run_artifact_id",
        "index_name",
        "collection_name",
    )
    @classmethod
    def validate_optional_strings(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        if value is None:
            return None
        return _non_empty_string(value, info.field_name or "field")

    @model_validator(mode="after")
    def validate_artifact_ids(self) -> RetrievalRunConfig:
        if self.execution_mode == "replay":
            if self.replay_source_retrieval_run_artifact_id is None:
                raise ValueError(
                    "replay_source_retrieval_run_artifact_id is required for replay execution"
                )
            if self.trace_mode != "replay":
                raise ValueError("execution_mode='replay' requires trace_mode='replay'")
            return self
        if self.retrieval_mode in {"es", "hybrid"} and self.elasticsearch_index_artifact_id is None:
            raise ValueError("elasticsearch_index_artifact_id is required for es/hybrid retrieval")
        if (
            self.retrieval_mode in {"milvus", "hybrid"}
            and self.milvus_collection_artifact_id is None
        ):
            raise ValueError(
                "milvus_collection_artifact_id is required for milvus/hybrid retrieval"
            )
        if self.retrieval_mode == "milvus" and self.elasticsearch_index_artifact_id is None:
            raise ValueError("elasticsearch_index_artifact_id is required for milvus ES enrich")
        return self


def run_retrieval(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: RetrievalRunConfig,
    *,
    es_client: ElasticsearchRetrievalClient | None = None,
    milvus_client: MilvusRetrievalClient | None = None,
    embedding_client: EmbeddingClient | None = None,
    rewrite_client: RewriteClient | None = None,
    rerank_client: RerankClient | None = None,
) -> ArtifactManifest:
    """Run retrieval for normalized queries and write a retrieval_run artifact."""

    if config.execution_mode == "replay":
        return _run_retrieval_replay(source_store, output_store, config)

    _validate_runtime_dependencies(
        config,
        es_client=es_client,
        milvus_client=milvus_client,
        embedding_client=embedding_client,
        rewrite_client=rewrite_client,
        rerank_client=rerank_client,
    )
    dataset = read_normalized_dataset_artifact(
        source_store,
        config.source_normalized_dataset_artifact_id,
    )
    queries = dataset.queries[: config.query_limit] if config.query_limit else dataset.queries
    results: list[RetrievalQueryResult] = []
    for query in queries:
        try:
            results.append(
                _retrieve_one_query(
                    query_id=query.query_id,
                    query_text=query.text,
                    config=config,
                    es_client=es_client,
                    milvus_client=milvus_client,
                    embedding_client=embedding_client,
                    rewrite_client=rewrite_client,
                    rerank_client=rerank_client,
                )
            )
        except Exception as exc:
            results.append(
                RetrievalQueryResult(
                    query_id=query.query_id,
                    query_text=query.text,
                    hits=[],
                    trace=(
                        _build_error_trace(query.text, exc)
                        if config.trace_mode == "replay"
                        else None
                    ),
                    error=str(exc),
                )
            )

    metadata = _build_manifest_metadata(config)
    return write_retrieval_run_artifact(
        output_store,
        config.output_artifact_id,
        results,
        queries_per_shard=config.queries_per_shard,
        metadata=metadata,
        dependencies=_build_dependencies(config),
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
    )


def _run_retrieval_replay(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: RetrievalRunConfig,
) -> ArtifactManifest:
    if config.replay_source_retrieval_run_artifact_id is None:
        raise RetrievalRunError(
            "replay_source_retrieval_run_artifact_id is required for replay execution"
        )

    source_records = read_retrieval_run_artifact(
        source_store,
        config.replay_source_retrieval_run_artifact_id,
    )
    if any(record.trace is None for record in source_records):
        raise RetrievalRunError("replay source retrieval_run artifact is missing replay trace")

    return write_retrieval_run_artifact(
        output_store,
        config.output_artifact_id,
        source_records,
        queries_per_shard=config.queries_per_shard,
        metadata=_build_manifest_metadata(config),
        dependencies=_build_dependencies(config),
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
    )


def _validate_runtime_dependencies(
    config: RetrievalRunConfig,
    *,
    es_client: ElasticsearchRetrievalClient | None,
    milvus_client: MilvusRetrievalClient | None,
    embedding_client: EmbeddingClient | None,
    rewrite_client: RewriteClient | None,
    rerank_client: RerankClient | None,
) -> None:
    if config.index_name is None:
        raise RetrievalRunError("index_name is required because ES enrich is required")
    if es_client is None:
        raise RetrievalRunError("es_client is required because ES enrich is required")
    if config.retrieval_mode in {"milvus", "hybrid"}:
        if config.collection_name is None:
            raise RetrievalRunError("collection_name is required for milvus/hybrid retrieval")
        if milvus_client is None:
            raise RetrievalRunError("milvus_client is required for milvus/hybrid retrieval")
        if embedding_client is None:
            raise RetrievalRunError("embedding_client is required for milvus/hybrid retrieval")
    if config.rerank_enabled and rerank_client is None:
        raise RetrievalRunError("rerank_client is required when rerank_enabled=True")
    if config.rewrite_enabled and config.sub_queries > 0 and rewrite_client is None:
        raise RetrievalRunError("rewrite_client is required when rewrite is enabled")


def _retrieve_one_query(
    *,
    query_id: str,
    query_text: str,
    config: RetrievalRunConfig,
    es_client: ElasticsearchRetrievalClient | None,
    milvus_client: MilvusRetrievalClient | None,
    embedding_client: EmbeddingClient | None,
    rewrite_client: RewriteClient | None,
    rerank_client: RerankClient | None,
) -> RetrievalQueryResult:
    queries = _resolve_query_paths(query_text, config, rewrite_client)
    vectors = _embed_query_paths(queries, config, embedding_client)
    hit_lists: list[list[RetrievalHit]] = []
    per_query_trace: list[dict[str, Any]] = []
    trace: dict[str, Any] = {
        "rewrite_queries": queries,
        "per_query": per_query_trace,
        "rerank_input": [],
        "rerank_hits": [],
    }

    for index, query in enumerate(queries):
        hits, es_hits, milvus_hits, fused_hits = _recall_one(
            query=query,
            config=config,
            es_client=es_client,
            milvus_client=milvus_client,
            embedding_client=embedding_client,
            vector=vectors[index] if vectors else None,
        )
        hit_lists.append(hits)
        per_query_trace.append(
            {
                "query": query,
                "es_hits": [hit.model_dump(mode="json") for hit in es_hits],
                "milvus_hits": [hit.model_dump(mode="json") for hit in milvus_hits],
                "fused_hits": [hit.model_dump(mode="json") for hit in fused_hits],
            }
        )

    if len(hit_lists) == 1:
        candidates = hit_lists[0]
        single_trace = per_query_trace[0]
        trace["es_hits"] = single_trace["es_hits"]
        trace["milvus_hits"] = single_trace["milvus_hits"]
        trace["fused_hits"] = single_trace["fused_hits"]
    elif config.rerank_enabled:
        candidates = dedupe_by_chunk_id([hit for hits in hit_lists for hit in hits])
        trace["fused_hits"] = [hit.model_dump(mode="json") for hit in candidates]
    else:
        candidates = dedupe_sequential(hit_lists, max_total=250)
        trace["fused_hits"] = [hit.model_dump(mode="json") for hit in candidates]

    final_hits = _maybe_rerank(query_text, candidates, config, rerank_client, trace)
    ranked_hits = _rank_hits(final_hits[: config.top_k])
    trace["final_hits"] = [hit.model_dump(mode="json") for hit in ranked_hits]
    return RetrievalQueryResult(
        query_id=query_id,
        query_text=query_text,
        hits=ranked_hits,
        trace=trace if config.trace_mode == "replay" else None,
        error=None,
    )


def _build_error_trace(query_text: str, exc: Exception) -> dict[str, Any]:
    return {
        "rewrite_queries": [query_text.strip()],
        "per_query": [],
        "rerank_input": [],
        "rerank_hits": [],
        "final_hits": [],
        "error": str(exc),
        "error_stage": "unknown",
    }


def _resolve_query_paths(
    query_text: str,
    config: RetrievalRunConfig,
    rewrite_client: RewriteClient | None,
) -> list[str]:
    queries = [query_text.strip()]
    if config.rewrite_enabled and config.sub_queries > 0:
        if rewrite_client is None:
            raise RetrievalRunError("rewrite_client is required when rewrite is enabled")
        queries.extend(rewrite_client.rewrite(query_text, config.sub_queries))
    return _dedupe_queries(queries, max_count=1 + config.sub_queries)


def _dedupe_queries(values: list[str], *, max_count: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = value.strip()
        key = query.lower()
        if not query or key in seen:
            continue
        seen.add(key)
        out.append(query)
        if len(out) >= max_count:
            break
    return out


def _embed_query_paths(
    queries: list[str],
    config: RetrievalRunConfig,
    embedding_client: EmbeddingClient | None,
) -> list[list[float]]:
    if config.retrieval_mode not in {"milvus", "hybrid"}:
        return []
    if embedding_client is None:
        raise RetrievalRunError("embedding_client is required for milvus/hybrid retrieval")
    vectors = embedding_client.embed_texts(queries)
    if len(vectors) != len(queries):
        raise RetrievalRunError("embedding client returned a different number of vectors")
    return vectors


def _recall_one(
    *,
    query: str,
    config: RetrievalRunConfig,
    es_client: ElasticsearchRetrievalClient | None,
    milvus_client: MilvusRetrievalClient | None,
    embedding_client: EmbeddingClient | None,
    vector: list[float] | None = None,
) -> tuple[list[RetrievalHit], list[RetrievalHit], list[RetrievalHit], list[RetrievalHit]]:
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


def _maybe_rerank(
    query_text: str,
    candidates: list[RetrievalHit],
    config: RetrievalRunConfig,
    rerank_client: RerankClient | None,
    trace: dict[str, Any],
) -> list[RetrievalHit]:
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


def _rank_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    return [hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(hits, start=1)]


def _build_manifest_metadata(config: RetrievalRunConfig) -> dict[str, Any]:
    metadata = dict(config.metadata)
    metadata.update(
        {
            "source_normalized_dataset_artifact_id": config.source_normalized_dataset_artifact_id,
            "elasticsearch_index_artifact_id": config.elasticsearch_index_artifact_id,
            "milvus_collection_artifact_id": config.milvus_collection_artifact_id,
            "retrieval_mode": config.retrieval_mode,
            "top_k": config.top_k,
            "trace_mode": config.trace_mode,
            "execution_mode": config.execution_mode,
            "replay_source_retrieval_run_artifact_id": (
                config.replay_source_retrieval_run_artifact_id
            ),
            "sub_queries": config.sub_queries,
            "rewrite_enabled": config.rewrite_enabled,
            "rerank_enabled": config.rerank_enabled,
            "hybrid_per_source_topk": config.hybrid_per_source_topk,
            "rrf_path_topk": config.rrf_path_topk,
            "rerank_cross_path_topk": config.rerank_cross_path_topk,
            "rerank_candidate_cap": config.rerank_candidate_cap,
            "index_name": config.index_name,
            "collection_name": config.collection_name,
        }
    )
    return metadata


def _build_dependencies(config: RetrievalRunConfig) -> list[ArtifactDependency]:
    dependencies = [
        ArtifactDependency(
            artifact_type=NORMALIZED_DATASET_ARTIFACT_TYPE,
            artifact_id=config.source_normalized_dataset_artifact_id,
        )
    ]
    if (
        config.execution_mode == "replay"
        and config.replay_source_retrieval_run_artifact_id is not None
    ):
        dependencies.append(
            ArtifactDependency(
                artifact_type=RETRIEVAL_RUN_ARTIFACT_TYPE,
                artifact_id=config.replay_source_retrieval_run_artifact_id,
            )
        )
        return dependencies
    if config.elasticsearch_index_artifact_id is not None:
        dependencies.append(
            ArtifactDependency(
                artifact_type=ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
                artifact_id=config.elasticsearch_index_artifact_id,
            )
        )
    if config.retrieval_mode in {"milvus", "hybrid"} and config.milvus_collection_artifact_id:
        dependencies.append(
            ArtifactDependency(
                artifact_type=MILVUS_COLLECTION_ARTIFACT_TYPE,
                artifact_id=config.milvus_collection_artifact_id,
            )
        )
    return dependencies
