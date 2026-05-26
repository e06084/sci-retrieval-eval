"""Embedding runner orchestration."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from eval_platform.artifacts import ArtifactManifest, ArtifactStore
from eval_platform.chunking import CHUNKED_CORPUS_ARTIFACT_TYPE
from eval_platform.chunking.artifact import iter_chunk_shards
from eval_platform.chunking.progress import ProgressReporter, report_progress
from eval_platform.embeddings.artifact import EmbeddingShard, write_embedding_shards_artifact
from eval_platform.embeddings.client import EmbeddingClient
from eval_platform.embeddings.schema import (
    EmbeddingConsistencyCheckResult,
    EmbeddingProvenance,
    EmbeddingRecord,
)


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class EmbeddingRunError(Exception):
    """Raised when embedding runner validation fails."""


def _batched_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    return [texts[index : index + batch_size] for index in range(0, len(texts), batch_size)]


def _source_user_metadata(source_manifest_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in source_manifest_metadata.items()
        if key
        not in {
            "chunk_count",
            "unique_doc_count",
            "chunker",
            "chunk_params",
            "sharding",
            "shards",
        }
    }


class EmbeddingRunConfig(BaseModel):
    """Configuration for an embedding run."""

    source_artifact_id: str
    output_artifact_id: str
    model_name: str
    embedding_dim: int = Field(gt=0)
    provider: str | None = None
    api_version: str | None = None
    normalized: bool | None = None
    endpoint_id: str | None = None
    endpoint_ids: list[str] = Field(default_factory=list)
    batch_size: int | None = Field(default=None, gt=0)
    timeout_seconds: float | None = Field(default=None, gt=0)
    consistency_check: EmbeddingConsistencyCheckResult | None = None
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_artifact_id", "output_artifact_id", "model_name")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("endpoint_id")
    @classmethod
    def validate_endpoint_id(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("endpoint_ids")
    @classmethod
    def validate_endpoint_ids(cls, value: list[str]) -> list[str]:
        return [_non_empty_string(item, "endpoint_ids") for item in value]


def run_embedding(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: EmbeddingRunConfig,
    client: EmbeddingClient,
    *,
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Read chunked corpus, compute embeddings, and write an embeddings artifact."""
    endpoint_ids = list(config.endpoint_ids)
    if config.endpoint_id and config.endpoint_id not in endpoint_ids:
        endpoint_ids.insert(0, config.endpoint_id)

    if len(endpoint_ids) > 1 and config.consistency_check is None:
        raise EmbeddingRunError(
            "Multi-endpoint embedding runs require a consistency_check result"
        )
    if config.consistency_check is not None and config.consistency_check.passed is False:
        raise EmbeddingRunError("Embedding consistency check failed; refusing to write artifact")

    source_manifest = source_store.read_manifest(
        CHUNKED_CORPUS_ARTIFACT_TYPE,
        config.source_artifact_id,
    )
    source_sharding = source_manifest.metadata.get("sharding", {})
    shard_total = (
        len(source_manifest.metadata.get("shards", []))
        if source_sharding.get("enabled")
        else 1
    )
    batch_size = config.batch_size or 0

    runtime_parameters: dict[str, Any] = {}
    if config.batch_size is not None:
        runtime_parameters["batch_size"] = config.batch_size
    if config.timeout_seconds is not None:
        runtime_parameters["timeout_seconds"] = config.timeout_seconds

    provenance = EmbeddingProvenance(
        model_name=config.model_name,
        provider=config.provider,
        api_version=config.api_version,
        embedding_dim=config.embedding_dim,
        normalized=config.normalized,
        endpoint_id=config.endpoint_id or (endpoint_ids[0] if len(endpoint_ids) == 1 else None),
        endpoint_ids=endpoint_ids,
        consistency_check=config.consistency_check,
        runtime_parameters=runtime_parameters,
        metadata={"source_file_record_num": source_sharding.get("file_record_num")},
    )

    def _iter_output_shards() -> Iterator[EmbeddingShard]:
        for shard_index, chunk_shard in enumerate(
            iter_chunk_shards(source_store, config.source_artifact_id),
            start=1,
        ):
            shard_records: list[EmbeddingRecord] = []
            processed_in_shard = 0
            shard_batch_size = batch_size or max(1, len(chunk_shard.chunks))
            shard_texts = [chunk.text for chunk in chunk_shard.chunks]
            shard_batches = (
                _batched_texts(shard_texts, shard_batch_size) if shard_texts else []
            )
            cursor = 0

            for batch_index, batch_texts in enumerate(shard_batches, start=1):
                vectors = client.embed_texts(batch_texts)
                if len(vectors) != len(batch_texts):
                    raise EmbeddingRunError(
                        "Embedding client returned a different number of vectors"
                    )

                batch_chunks = chunk_shard.chunks[cursor : cursor + len(batch_texts)]
                cursor += len(batch_texts)
                for chunk, vector in zip(batch_chunks, vectors, strict=True):
                    if len(vector) != config.embedding_dim:
                        raise EmbeddingRunError(
                            "Embedding client returned vectors with unexpected dimension"
                        )
                    record_metadata = dict(chunk.metadata)
                    if chunk.title is not None:
                        record_metadata["title"] = chunk.title
                    record_metadata["chunk_index"] = chunk.chunk_index
                    if chunk.start_offset is not None:
                        record_metadata["start_offset"] = chunk.start_offset
                    if chunk.end_offset is not None:
                        record_metadata["end_offset"] = chunk.end_offset
                    shard_records.append(
                        EmbeddingRecord(
                            chunk_id=chunk.chunk_id,
                            doc_id=chunk.doc_id,
                            vector=vector,
                            metadata=record_metadata,
                        )
                    )

                processed_in_shard += len(batch_texts)
                report_progress(
                    progress_reporter,
                    stage="embedding",
                    current=processed_in_shard,
                    total=len(chunk_shard.chunks),
                    message="Embedded shard batch",
                    metadata={
                        "kind": "batch",
                        "shard_id": chunk_shard.shard_id,
                        "batch_index": batch_index,
                        "batch_size": len(batch_texts),
                        "chunk_count": len(chunk_shard.chunks),
                    },
                )

            report_progress(
                progress_reporter,
                stage="embedding",
                current=shard_index,
                total=shard_total,
                message="Completed embedding shard",
                metadata={
                    "kind": "shard",
                    "shard_id": chunk_shard.shard_id,
                    "chunk_count": len(chunk_shard.chunks),
                    "source_chunk_file": chunk_shard.path,
                },
            )
            yield EmbeddingShard(
                shard_id=chunk_shard.shard_id,
                source_chunk_file=chunk_shard.path,
                embedding_file=(
                    f"embeddings/{chunk_shard.shard_id}.jsonl"
                    if chunk_shard.path != "chunks.jsonl"
                    else "embeddings.jsonl"
                ),
                source_chunk_count=len(chunk_shard.chunks),
                embedding_count=len(shard_records),
                first_chunk_id=shard_records[0].chunk_id if shard_records else None,
                last_chunk_id=shard_records[-1].chunk_id if shard_records else None,
                embeddings=shard_records,
            )

    manifest_metadata = _source_user_metadata(source_manifest.metadata)
    manifest_metadata.update(config.metadata)

    return write_embedding_shards_artifact(
        output_store,
        config.output_artifact_id,
        _iter_output_shards(),
        provenance=provenance,
        source_artifact_id=config.source_artifact_id,
        source_artifact_type="chunked_corpus",
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
        metadata=manifest_metadata,
    )
