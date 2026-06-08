"""Retrieval run orchestration."""

from __future__ import annotations

import gc
import hashlib as _hashlib
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from eval_platform.artifacts import (
    ArtifactDependency,
    ArtifactFile,
    ArtifactManifest,
    ArtifactStore,
)
from eval_platform.assets import (
    add_asset_fingerprint_metadata,
    build_asset_fingerprint,
    manifest_asset_fingerprint_sha256,
    retrieval_run_fingerprint_components,
)
from eval_platform.chunking.progress import ProgressReporter, report_progress
from eval_platform.datasets import (
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    read_normalized_dataset_artifact,
)
from eval_platform.defaults import (
    DEFAULT_HYBRID_PER_SOURCE_TOPK,
    DEFAULT_PAPER_CAP,
    DEFAULT_RERANK_CANDIDATE_CAP,
    DEFAULT_RERANK_CROSS_PATH_TOPK,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_RRF_PATH_TOPK,
)
from eval_platform.embeddings import EmbeddingClient
from eval_platform.indexes import ELASTICSEARCH_INDEX_ARTIFACT_TYPE, MILVUS_COLLECTION_ARTIFACT_TYPE
from eval_platform.retrieval.artifact import RETRIEVAL_RUN_ARTIFACT_TYPE
from eval_platform.retrieval.clients import (
    ElasticsearchRetrievalClient,
    MilvusRetrievalClient,
    RerankClient,
    RewriteClient,
)
from eval_platform.retrieval.errors import RetrievalRunError
from eval_platform.retrieval.fusion import (
    dedupe_by_chunk_id,
    dedupe_sequential,
    limit_hits_per_paper,
)
from eval_platform.retrieval.jsonl import dump_retrieval_results_jsonl
from eval_platform.retrieval.query_paths import embed_query_paths, resolve_query_paths
from eval_platform.retrieval.recall import recall_one
from eval_platform.retrieval.replay import run_retrieval_replay
from eval_platform.retrieval.rerank_flow import maybe_rerank, rank_hits
from eval_platform.retrieval.schema import RetrievalHit, RetrievalQueryResult
from eval_platform.retrieval.trace import (
    append_recall_trace,
    build_error_trace,
    hits_trace,
    hits_trace_light,
    new_live_trace,
)


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class RetrievalRunConfig(BaseModel):
    """Configuration for a retrieval_run artifact."""

    source_normalized_dataset_artifact_id: str
    output_artifact_id: str
    retrieval_mode: Literal["es", "milvus", "hybrid"] = "hybrid"
    top_k: int = Field(default=DEFAULT_RETRIEVAL_TOP_K, gt=0)
    query_limit: int | None = Field(default=None, gt=0)
    queries_per_shard: int = Field(default=1000, gt=0)
    trace_mode: Literal["replay", "light", "none"] = "replay"
    execution_mode: Literal["live", "replay"] = "live"
    replay_source_retrieval_run_artifact_id: str | None = None

    elasticsearch_index_artifact_id: str | None = None
    milvus_collection_artifact_id: str | None = None

    index_name: str | None = None
    collection_name: str | None = None

    sub_queries: int = Field(default=0, ge=0)
    rewrite_enabled: bool = False
    rerank_enabled: bool = False
    hybrid_per_source_topk: int = Field(default=DEFAULT_HYBRID_PER_SOURCE_TOPK, gt=0)
    rrf_path_topk: int = Field(default=DEFAULT_RRF_PATH_TOPK, gt=0)
    paper_cap: int = Field(default=DEFAULT_PAPER_CAP, ge=0)
    rerank_cross_path_topk: int = Field(default=DEFAULT_RERANK_CROSS_PATH_TOPK, ge=0)
    rerank_candidate_cap: int = Field(default=DEFAULT_RERANK_CANDIDATE_CAP, ge=0)

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
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Run retrieval for normalized queries and write a retrieval_run artifact."""

    if config.execution_mode == "replay":
        return run_retrieval_replay(
            source_store,
            output_store,
            config,
            build_manifest_metadata=lambda replay_config: _build_manifest_metadata(
                replay_config,
                source_store=source_store,
            ),
            build_dependencies=_build_dependencies,
        )

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
    total_queries = len(queries)
    report_progress(
        progress_reporter,
        stage="retrieval_run",
        current=0,
        total=total_queries,
        message="Starting retrieval queries",
        metadata=_progress_metadata(config),
    )
    shard_buffer: list[RetrievalQueryResult] = []
    failed_query_count = 0
    shard_index = 0
    files: list[ArtifactFile] = []
    total_record_count = 0

    for query_index, query in enumerate(queries, start=1):
        query_error: str | None = None
        try:
            shard_buffer.append(
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
            query_error = str(exc)
            failed_query_count += 1
            shard_buffer.append(
                RetrievalQueryResult(
                    query_id=query.query_id,
                    query_text=query.text,
                    hits=[],
                    trace=(
                        build_error_trace(query.text, exc)
                        if config.trace_mode != "none"
                        else None
                    ),
                    error=query_error,
                )
            )
        report_progress(
            progress_reporter,
            stage="retrieval_run",
            current=query_index,
            total=total_queries,
            message="Processed retrieval query",
            metadata={
                **_progress_metadata(config),
                "query_id": query.query_id,
                "failed_query_count": failed_query_count,
                "query_error": query_error,
            },
        )
        if len(shard_buffer) >= config.queries_per_shard:
            files.append(
                _flush_shard(output_store, config.output_artifact_id, shard_index, shard_buffer)
            )
            total_record_count += len(shard_buffer)
            shard_index += 1
            shard_buffer = []
            gc.collect()

    if shard_buffer:
        files.append(
            _flush_shard(output_store, config.output_artifact_id, shard_index, shard_buffer)
        )
        total_record_count += len(shard_buffer)
        shard_buffer = []
        gc.collect()

    metadata = _build_manifest_metadata(config, source_store=source_store)
    succeeded_query_count = total_record_count - failed_query_count
    manifest_metadata = {
        key: value
        for key, value in metadata.items()
        if key
        not in {
            "stage",
            "query_count",
            "succeeded_query_count",
            "failed_query_count",
            "queries_per_shard",
            "result_file_count",
            "result_record_count",
        }
    }
    manifest_metadata.update(
        {
            "stage": "retrieval_run",
            "query_count": total_record_count,
            "succeeded_query_count": succeeded_query_count,
            "failed_query_count": failed_query_count,
            "queries_per_shard": config.queries_per_shard,
            "result_file_count": len(files),
            "result_record_count": total_record_count,
        }
    )
    manifest = ArtifactManifest(
        artifact_id=config.output_artifact_id,
        artifact_type=RETRIEVAL_RUN_ARTIFACT_TYPE,
        created_at=datetime.now(UTC),
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
        dependencies=_build_dependencies(config),
        metadata=manifest_metadata,
        files=files,
    )
    output_store.write_manifest(
        RETRIEVAL_RUN_ARTIFACT_TYPE, config.output_artifact_id, manifest
    )
    output_store.mark_success(RETRIEVAL_RUN_ARTIFACT_TYPE, config.output_artifact_id)
    return manifest


def build_retrieval_run_fingerprint_sha256(
    source_store: ArtifactStore,
    config: RetrievalRunConfig,
) -> str | None:
    """Return the expected retrieval_run asset fingerprint for a config, if available."""

    components = _retrieval_asset_fingerprint_components(config, source_store=source_store)
    if components is None:
        return None
    return build_asset_fingerprint(
        artifact_type=RETRIEVAL_RUN_ARTIFACT_TYPE,
        components=components,
    ).sha256


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
    queries = resolve_query_paths(query_text, config, rewrite_client)
    vectors = embed_query_paths(queries, config, embedding_client)
    hit_lists: list[list[RetrievalHit]] = []
    trace, per_query_trace = new_live_trace(queries)
    _hits_fn = hits_trace if config.trace_mode == "replay" else hits_trace_light

    for index, query in enumerate(queries):
        hits, es_hits, milvus_hits, fused_hits = recall_one(
            query=query,
            config=config,
            es_client=es_client,
            milvus_client=milvus_client,
            embedding_client=embedding_client,
            vector=vectors[index] if vectors else None,
        )
        hit_lists.append(hits)
        append_recall_trace(
            per_query_trace,
            query=query,
            es_hits=es_hits,
            milvus_hits=milvus_hits,
            fused_hits=fused_hits,
            hits_fn=_hits_fn,
        )

    if len(hit_lists) == 1:
        candidates = hit_lists[0]
        single_trace = per_query_trace[0]
        trace["es_hits"] = single_trace["es_hits"]
        trace["milvus_hits"] = single_trace["milvus_hits"]
        trace["fused_hits"] = single_trace["fused_hits"]
    elif config.rerank_enabled:
        candidates = dedupe_by_chunk_id([hit for hits in hit_lists for hit in hits])
        trace["fused_hits"] = _hits_fn(candidates)
    else:
        candidates = dedupe_sequential(hit_lists, max_total=250)
        trace["fused_hits"] = _hits_fn(candidates)

    if config.retrieval_mode == "hybrid":
        capped_candidates = limit_hits_per_paper(
            candidates,
            paper_cap=config.paper_cap,
            max_total=config.rrf_path_topk,
        )
    else:
        capped_candidates = candidates
    trace["paper_capped_hits"] = _hits_fn(capped_candidates)
    final_hits = maybe_rerank(
        query_text, capped_candidates, config, rerank_client, trace, hits_fn=_hits_fn
    )
    ranked_hits = rank_hits(final_hits[: config.top_k])
    trace["final_hits"] = _hits_fn(ranked_hits)
    return RetrievalQueryResult(
        query_id=query_id,
        query_text=query_text,
        hits=ranked_hits,
        trace=trace if config.trace_mode != "none" else None,
        error=None,
    )


_RESULTS_DIR = "results"


def _flush_shard(
    store: ArtifactStore,
    artifact_id: str,
    shard_index: int,
    records: list[RetrievalQueryResult],
    *,
    strip_hit_text: bool = True,
) -> ArtifactFile:
    """Write one shard of retrieval results and return its file metadata."""
    if strip_hit_text:
        records = [_strip_record_text(r) for r in records]
    payload = dump_retrieval_results_jsonl(records).encode("utf-8")
    path = f"{_RESULTS_DIR}/part-{shard_index:05d}.jsonl"
    store.put_file(RETRIEVAL_RUN_ARTIFACT_TYPE, artifact_id, path, payload)
    return ArtifactFile(
        path=path,
        size_bytes=len(payload),
        sha256=_hashlib.sha256(payload).hexdigest(),
    )


def _strip_record_text(record: RetrievalQueryResult) -> RetrievalQueryResult:
    """Remove bulky text/title from hits to reduce artifact size and memory on read-back."""
    if not record.hits:
        return record
    stripped_hits = [
        hit.model_copy(update={"text": "", "title": None}) for hit in record.hits
    ]
    return record.model_copy(update={"hits": stripped_hits})



def _build_manifest_metadata(
    config: RetrievalRunConfig,
    *,
    source_store: ArtifactStore | None = None,
) -> dict[str, Any]:
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
            "paper_cap": config.paper_cap,
            "rerank_cross_path_topk": config.rerank_cross_path_topk,
            "rerank_candidate_cap": config.rerank_candidate_cap,
            "index_name": config.index_name,
            "collection_name": config.collection_name,
        }
    )
    add_asset_fingerprint_metadata(
        metadata,
        artifact_type=RETRIEVAL_RUN_ARTIFACT_TYPE,
        components=_retrieval_asset_fingerprint_components(
            config,
            source_store=source_store,
        ),
    )
    return metadata


def _retrieval_asset_fingerprint_components(
    config: RetrievalRunConfig,
    *,
    source_store: ArtifactStore | None,
) -> dict[str, Any] | None:
    if source_store is None:
        return None

    normalized_fingerprint = _manifest_fingerprint(
        source_store,
        NORMALIZED_DATASET_ARTIFACT_TYPE,
        config.source_normalized_dataset_artifact_id,
    )
    if normalized_fingerprint is None:
        return None

    es_fingerprint = (
        _manifest_fingerprint(
            source_store,
            ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
            config.elasticsearch_index_artifact_id,
        )
        if config.elasticsearch_index_artifact_id is not None
        else None
    )
    milvus_fingerprint = (
        _manifest_fingerprint(
            source_store,
            MILVUS_COLLECTION_ARTIFACT_TYPE,
            config.milvus_collection_artifact_id,
        )
        if config.milvus_collection_artifact_id is not None
        else None
    )
    if config.retrieval_mode in {"es", "hybrid"} and es_fingerprint is None:
        return None
    if config.retrieval_mode in {"milvus", "hybrid"} and milvus_fingerprint is None:
        return None

    metadata = config.metadata
    return retrieval_run_fingerprint_components(
        normalized_dataset_fingerprint=normalized_fingerprint,
        retrieval_mode=config.retrieval_mode,
        elasticsearch_index_fingerprint=es_fingerprint,
        milvus_collection_fingerprint=milvus_fingerprint,
        query_source={"query_limit": config.query_limit},
        query_embedding=_optional_mapping(metadata.get("query_embedding")),
        search_params={
            "top_k": config.top_k,
            "sub_queries": config.sub_queries,
            "es": {
                "enabled": config.retrieval_mode in {"es", "hybrid"},
                "top_k": (
                    config.top_k
                    if config.retrieval_mode == "es"
                    else max(config.hybrid_per_source_topk, config.rrf_path_topk)
                ),
            },
            "milvus": {
                "enabled": config.retrieval_mode in {"milvus", "hybrid"},
                "top_k": (
                    config.top_k
                    if config.retrieval_mode == "milvus"
                    else max(config.hybrid_per_source_topk, config.rrf_path_topk)
                ),
            },
            "fusion": {
                "method": "rrf" if config.retrieval_mode == "hybrid" else None,
                "path_topk": config.rrf_path_topk,
                "paper_cap": config.paper_cap,
            },
        },
        rewrite=_optional_mapping(metadata.get("rewrite"))
        or (
            {"enabled": True, "sub_queries": config.sub_queries}
            if config.rewrite_enabled
            else None
        ),
        rerank=_optional_mapping(metadata.get("rerank"))
        or (
            {
                "enabled": True,
                "candidate_cap": config.rerank_candidate_cap,
                "cross_path_topk": config.rerank_cross_path_topk,
            }
            if config.rerank_enabled
            else None
        ),
        trace_mode=config.trace_mode,
    )


def _manifest_fingerprint(
    store: ArtifactStore,
    artifact_type: str,
    artifact_id: str | None,
) -> str | None:
    if artifact_id is None:
        return None
    try:
        manifest = store.read_manifest(artifact_type, artifact_id)
    except Exception:
        return None
    return manifest_asset_fingerprint_sha256(manifest)


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    return None


def _progress_metadata(config: RetrievalRunConfig) -> dict[str, Any]:
    metadata = dict(config.metadata)
    metadata.update(
        {
            "output_artifact_id": config.output_artifact_id,
            "source_normalized_dataset_artifact_id": (
                config.source_normalized_dataset_artifact_id
            ),
            "retrieval_mode": config.retrieval_mode,
            "execution_mode": config.execution_mode,
            "trace_mode": config.trace_mode,
            "query_limit": config.query_limit,
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
