"""Corpus build runner orchestration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest, ArtifactStore
from eval_platform.artifacts.metadata_keys import (
    METADATA_KEY_COLLECTION_NAME,
    METADATA_KEY_INDEX_NAME,
)
from eval_platform.artifacts.types import (
    CHUNKED_CORPUS_ARTIFACT_TYPE,
    CORPUS_BUILD_ARTIFACT_TYPE,
    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
    EMBEDDINGS_ARTIFACT_TYPE,
    MILVUS_COLLECTION_ARTIFACT_TYPE,
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    RAW_DATASET_ARTIFACT_TYPE,
)
from eval_platform.chunking import (
    ChunkingRunConfig,
    ExternalChunker,
    ProgressReporter,
    run_chunking,
)
from eval_platform.chunking.progress import report_progress
from eval_platform.datasets import (
    RawFileOpener,
    RawToNormalizedConfig,
    import_raw_dataset_from_local_dir,
    import_raw_dataset_from_s3_prefix,
    normalize_raw_dataset_artifact,
)
from eval_platform.datasets.raw_normalize import SUPPORTED_RAW_NORMALIZER_DATASET_NAMES
from eval_platform.embeddings import (
    EmbeddingClient,
    EmbeddingRunConfig,
    run_embedding,
)
from eval_platform.indexes import (
    ElasticsearchClientProtocol,
    ElasticsearchIngestConfig,
    MilvusClientProtocol,
    MilvusIngestConfig,
    run_elasticsearch_ingest,
    run_milvus_ingest,
)

_SENSITIVE_METADATA_KEY_PARTS = (
    "access_key",
    "api_key",
    "password",
    "secret",
    "token",
)


class CorpusBuildError(Exception):
    """Raised when corpus build orchestration fails."""


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class RawSourceSpec(BaseModel):
    """Raw source location for corpus build."""

    source_type: Literal["s3_prefix", "local_dir"]
    uri: str
    dataset_revision: str | None = None
    import_parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


class CorpusBuildArtifactIds(BaseModel):
    """Artifact ids used by one corpus build run."""

    raw_dataset: str
    normalized_dataset: str
    chunked_corpus: str
    embeddings: str
    elasticsearch_index: str | None = None
    milvus_collection: str | None = None

    @field_validator("raw_dataset", "normalized_dataset", "chunked_corpus", "embeddings")
    @classmethod
    def validate_required_ids(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("elasticsearch_index", "milvus_collection")
    @classmethod
    def validate_optional_ids(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _non_empty_string(value, info.field_name or "field")


class CorpusBuildConfig(BaseModel):
    """Configuration for a corpus build run."""

    run_id: str
    dataset_name: str = "IFIRNFCorpus"
    raw_source: RawSourceSpec
    artifact_ids: CorpusBuildArtifactIds | None = None
    enable_elasticsearch: bool = True
    enable_milvus: bool = True
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "dataset_name")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @model_validator(mode="after")
    def validate_dataset_name(self) -> CorpusBuildConfig:
        if self.dataset_name not in SUPPORTED_RAW_NORMALIZER_DATASET_NAMES:
            raise ValueError(
                "corpus build v1 supports only registered raw normalizer datasets: "
                f"{sorted(SUPPORTED_RAW_NORMALIZER_DATASET_NAMES)}"
            )
        return self


def default_corpus_build_artifact_ids(
    run_id: str,
    *,
    enable_elasticsearch: bool = True,
    enable_milvus: bool = True,
) -> CorpusBuildArtifactIds:
    """Generate deterministic default artifact ids for a corpus build run."""

    run_id = _non_empty_string(run_id, "run_id")
    return CorpusBuildArtifactIds(
        raw_dataset=f"{run_id}_raw",
        normalized_dataset=f"{run_id}_normalized",
        chunked_corpus=f"{run_id}_chunks",
        embeddings=f"{run_id}_embeddings",
        elasticsearch_index=f"{run_id}_es_index" if enable_elasticsearch else None,
        milvus_collection=f"{run_id}_milvus_collection" if enable_milvus else None,
    )


def _safe_user_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe_metadata: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized_key = key.lower()
        if any(part in normalized_key for part in _SENSITIVE_METADATA_KEY_PARTS):
            continue
        safe_metadata[key] = value
    return safe_metadata


def _safe_raw_source_metadata(raw_source: RawSourceSpec) -> dict[str, Any]:
    payload = raw_source.model_dump(mode="json")
    import_parameters = payload.get("import_parameters")
    if isinstance(import_parameters, dict):
        payload["import_parameters"] = _safe_user_metadata(import_parameters)
    return payload


def _parse_s3_prefix_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise CorpusBuildError(f"Invalid s3_prefix uri: {uri!r}")
    prefix = parsed.path.lstrip("/")
    return parsed.netloc, prefix


def _local_dir_from_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise CorpusBuildError(f"Unsupported local_dir uri: {uri!r}")
    return Path(uri)


def _require_equal(actual: str | None, expected: str | None, label: str) -> None:
    if actual != expected:
        raise CorpusBuildError(
            f"{label} artifact id mismatch: expected {expected!r}, got {actual!r}"
        )


def _validate_stage_configs(
    artifact_ids: CorpusBuildArtifactIds,
    *,
    raw_to_normalized_config: RawToNormalizedConfig,
    chunking_config: ChunkingRunConfig,
    embedding_config: EmbeddingRunConfig,
    elasticsearch_config: ElasticsearchIngestConfig | None,
    milvus_config: MilvusIngestConfig | None,
    enable_elasticsearch: bool,
    enable_milvus: bool,
) -> None:
    _require_equal(
        raw_to_normalized_config.source_artifact_id,
        artifact_ids.raw_dataset,
        "raw_to_normalized.source_artifact_id",
    )
    _require_equal(
        raw_to_normalized_config.output_artifact_id,
        artifact_ids.normalized_dataset,
        "raw_to_normalized.output_artifact_id",
    )
    _require_equal(
        chunking_config.source_artifact_id,
        artifact_ids.normalized_dataset,
        "chunking.source_artifact_id",
    )
    _require_equal(
        chunking_config.output_artifact_id,
        artifact_ids.chunked_corpus,
        "chunking.output_artifact_id",
    )
    _require_equal(
        embedding_config.source_artifact_id,
        artifact_ids.chunked_corpus,
        "embedding.source_artifact_id",
    )
    _require_equal(
        embedding_config.output_artifact_id,
        artifact_ids.embeddings,
        "embedding.output_artifact_id",
    )

    if enable_elasticsearch:
        if elasticsearch_config is None:
            raise CorpusBuildError("elasticsearch_config is required when Elasticsearch is enabled")
        if artifact_ids.elasticsearch_index is None:
            raise CorpusBuildError("elasticsearch_index artifact id is required")
        _require_equal(
            elasticsearch_config.source_artifact_id,
            artifact_ids.chunked_corpus,
            "elasticsearch.source_artifact_id",
        )
        _require_equal(
            elasticsearch_config.output_artifact_id,
            artifact_ids.elasticsearch_index,
            "elasticsearch.output_artifact_id",
        )

    if enable_milvus:
        if milvus_config is None:
            raise CorpusBuildError("milvus_config is required when Milvus is enabled")
        if artifact_ids.milvus_collection is None:
            raise CorpusBuildError("milvus_collection artifact id is required")
        _require_equal(
            milvus_config.chunked_corpus_artifact_id,
            artifact_ids.chunked_corpus,
            "milvus.chunked_corpus_artifact_id",
        )
        _require_equal(
            milvus_config.embeddings_artifact_id,
            artifact_ids.embeddings,
            "milvus.embeddings_artifact_id",
        )
        _require_equal(
            milvus_config.output_artifact_id,
            artifact_ids.milvus_collection,
            "milvus.output_artifact_id",
        )


def _metadata_summary(manifest: ArtifactManifest) -> dict[str, Any]:
    metadata = manifest.metadata
    summary_keys_by_type = {
        RAW_DATASET_ARTIFACT_TYPE: {
            "source_type",
            "dataset_name",
            "dataset_revision",
            "content_fingerprint_sha256",
            "file_count",
        },
        NORMALIZED_DATASET_ARTIFACT_TYPE: {
            "corpus_count",
            "query_count",
            "qrel_count",
            "task_name",
            "split",
            "normalizer_name",
            "normalized_schema_version",
        },
        CHUNKED_CORPUS_ARTIFACT_TYPE: {
            "chunk_count",
            "unique_doc_count",
            "sharding",
        },
        EMBEDDINGS_ARTIFACT_TYPE: {
            "embedding_count",
            "unique_chunk_count",
            "unique_doc_count",
            "embedding_dim",
            "embedding_dtype",
            "vector_encoding",
        },
        ELASTICSEARCH_INDEX_ARTIFACT_TYPE: {
            METADATA_KEY_INDEX_NAME,
            "mapping_sha256",
            "indexed_count",
            "failed_count",
            "verified_document_count",
        },
        MILVUS_COLLECTION_ARTIFACT_TYPE: {
            METADATA_KEY_COLLECTION_NAME,
            "schema_sha256",
            "inserted_count",
            "failed_count",
            "verified_entity_count",
            "vector_dim",
            "metric_type",
        },
    }
    selected_keys = summary_keys_by_type.get(manifest.artifact_type, set())
    summary = {
        key: value
        for key, value in metadata.items()
        if key in selected_keys
        and not any(part in key.lower() for part in _SENSITIVE_METADATA_KEY_PARTS)
    }

    if manifest.artifact_type == EMBEDDINGS_ARTIFACT_TYPE:
        provenance = metadata.get("provenance")
        if isinstance(provenance, dict):
            summary["provenance"] = {
                key: value
                for key, value in provenance.items()
                if key in {"model_name", "provider", "api_version", "embedding_dim", "normalized"}
            }
    return summary


def _stage_record(
    *,
    stage: str,
    manifest: ArtifactManifest,
    store: ArtifactStore,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "artifact_type": manifest.artifact_type,
        "artifact_id": manifest.artifact_id,
        "artifact_uri": store.artifact_uri(manifest.artifact_type, manifest.artifact_id),
        "metadata_summary": _metadata_summary(manifest),
    }


def _report_stage_start(
    progress_reporter: ProgressReporter | None,
    *,
    current: int,
    total: int,
    stage_name: str,
) -> None:
    report_progress(
        progress_reporter,
        stage=CORPUS_BUILD_ARTIFACT_TYPE,
        current=current,
        total=total,
        message=f"Starting corpus build stage: {stage_name}",
        metadata={"kind": "stage_start", "stage_name": stage_name},
    )


def _report_stage_done(
    progress_reporter: ProgressReporter | None,
    *,
    current: int,
    total: int,
    stage_name: str,
    manifest: ArtifactManifest,
) -> None:
    report_progress(
        progress_reporter,
        stage=CORPUS_BUILD_ARTIFACT_TYPE,
        current=current,
        total=total,
        message=f"Completed corpus build stage: {stage_name}",
        metadata={
            "kind": "stage_done",
            "stage_name": stage_name,
            "artifact_type": manifest.artifact_type,
            "artifact_id": manifest.artifact_id,
        },
    )


def _run_stage(
    *,
    stage_name: str,
    stage_index: int,
    total_stages: int,
    progress_reporter: ProgressReporter | None,
    operation: Callable[[], ArtifactManifest],
) -> ArtifactManifest:
    try:
        _report_stage_start(
            progress_reporter,
            current=stage_index,
            total=total_stages,
            stage_name=stage_name,
        )
        manifest = operation()
        _report_stage_done(
            progress_reporter,
            current=stage_index,
            total=total_stages,
            stage_name=stage_name,
            manifest=manifest,
        )
        return manifest
    except Exception as exc:
        if isinstance(exc, CorpusBuildError):
            raise
        raise CorpusBuildError(f"Corpus build failed at stage {stage_name}") from exc


def run_corpus_build(
    store: ArtifactStore,
    config: CorpusBuildConfig,
    *,
    raw_import_client: Any | None = None,
    raw_file_opener: RawFileOpener,
    chunker: ExternalChunker,
    chunking_config: ChunkingRunConfig,
    embedding_client: EmbeddingClient,
    embedding_config: EmbeddingRunConfig,
    raw_to_normalized_config: RawToNormalizedConfig,
    elasticsearch_client: ElasticsearchClientProtocol | None = None,
    elasticsearch_config: ElasticsearchIngestConfig | None = None,
    milvus_client: MilvusClientProtocol | None = None,
    milvus_config: MilvusIngestConfig | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Run the v1 corpus build pipeline and write a run-level manifest."""

    artifact_ids = config.artifact_ids or default_corpus_build_artifact_ids(
        config.run_id,
        enable_elasticsearch=config.enable_elasticsearch,
        enable_milvus=config.enable_milvus,
    )
    _validate_stage_configs(
        artifact_ids,
        raw_to_normalized_config=raw_to_normalized_config,
        chunking_config=chunking_config,
        embedding_config=embedding_config,
        elasticsearch_config=elasticsearch_config,
        milvus_config=milvus_config,
        enable_elasticsearch=config.enable_elasticsearch,
        enable_milvus=config.enable_milvus,
    )
    if config.enable_elasticsearch and elasticsearch_client is None:
        raise CorpusBuildError("elasticsearch_client is required when Elasticsearch is enabled")
    if config.enable_milvus and milvus_client is None:
        raise CorpusBuildError("milvus_client is required when Milvus is enabled")

    stage_names = ["raw_import", "raw_to_normalized", "chunking", "embedding"]
    if config.enable_elasticsearch:
        stage_names.append("elasticsearch_ingest")
    if config.enable_milvus:
        stage_names.append("milvus_ingest")
    total_stages = len(stage_names)

    stage_records: list[dict[str, Any]] = []
    dependencies: list[ArtifactDependency] = []

    def record_stage(stage_name: str, manifest: ArtifactManifest) -> None:
        stage_records.append(_stage_record(stage=stage_name, manifest=manifest, store=store))
        dependencies.append(
            ArtifactDependency(
                artifact_type=manifest.artifact_type,
                artifact_id=manifest.artifact_id,
            )
        )

    stage_index = 1
    raw_manifest = _run_stage(
        stage_name="raw_import",
        stage_index=stage_index,
        total_stages=total_stages,
        progress_reporter=progress_reporter,
        operation=lambda: _run_raw_import(store, config, artifact_ids, raw_import_client),
    )
    record_stage("raw_import", raw_manifest)

    stage_index += 1
    normalized_manifest = _run_stage(
        stage_name="raw_to_normalized",
        stage_index=stage_index,
        total_stages=total_stages,
        progress_reporter=progress_reporter,
        operation=lambda: normalize_raw_dataset_artifact(
            store,
            store,
            raw_to_normalized_config,
            opener=raw_file_opener,
            progress_reporter=progress_reporter,
        ),
    )
    record_stage("raw_to_normalized", normalized_manifest)

    stage_index += 1
    chunk_manifest = _run_stage(
        stage_name="chunking",
        stage_index=stage_index,
        total_stages=total_stages,
        progress_reporter=progress_reporter,
        operation=lambda: run_chunking(
            store,
            chunking_config,
            chunker,
            progress_reporter=progress_reporter,
        ),
    )
    record_stage("chunking", chunk_manifest)

    stage_index += 1
    embedding_manifest = _run_stage(
        stage_name="embedding",
        stage_index=stage_index,
        total_stages=total_stages,
        progress_reporter=progress_reporter,
        operation=lambda: run_embedding(
            store,
            store,
            embedding_config,
            embedding_client,
            progress_reporter=progress_reporter,
        ),
    )
    record_stage("embedding", embedding_manifest)

    if config.enable_elasticsearch:
        assert elasticsearch_config is not None
        assert elasticsearch_client is not None
        stage_index += 1
        es_manifest = _run_stage(
            stage_name="elasticsearch_ingest",
            stage_index=stage_index,
            total_stages=total_stages,
            progress_reporter=progress_reporter,
            operation=lambda: run_elasticsearch_ingest(
                store,
                store,
                elasticsearch_config,
                elasticsearch_client,
                progress_reporter=progress_reporter,
            ),
        )
        record_stage("elasticsearch_ingest", es_manifest)

    if config.enable_milvus:
        assert milvus_config is not None
        assert milvus_client is not None
        stage_index += 1
        milvus_manifest = _run_stage(
            stage_name="milvus_ingest",
            stage_index=stage_index,
            total_stages=total_stages,
            progress_reporter=progress_reporter,
            operation=lambda: run_milvus_ingest(
                store,
                store,
                store,
                milvus_config,
                milvus_client,
                progress_reporter=progress_reporter,
            ),
        )
        record_stage("milvus_ingest", milvus_manifest)

    manifest_metadata: dict[str, Any] = {}
    manifest_metadata.update(_safe_user_metadata(config.metadata))
    manifest_metadata.update(
        {
            "run_id": config.run_id,
            "dataset_name": config.dataset_name,
            "raw_source": _safe_raw_source_metadata(config.raw_source),
            "artifact_ids": artifact_ids.model_dump(mode="json"),
            "enabled_stages": {
                "elasticsearch": config.enable_elasticsearch,
                "milvus": config.enable_milvus,
            },
            "stage_manifests": stage_records,
        }
    )
    manifest = ArtifactManifest(
        artifact_id=config.run_id,
        artifact_type=CORPUS_BUILD_ARTIFACT_TYPE,
        created_at=datetime.now(UTC),
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
        dependencies=dependencies,
        metadata=manifest_metadata,
        files=[],
    )

    try:
        store.write_manifest(CORPUS_BUILD_ARTIFACT_TYPE, config.run_id, manifest)
        report_progress(
            progress_reporter,
            stage=CORPUS_BUILD_ARTIFACT_TYPE,
            current=total_stages,
            total=total_stages,
            message="Completed corpus build run",
            metadata={"kind": "run_done", "run_id": config.run_id},
        )
        store.mark_success(CORPUS_BUILD_ARTIFACT_TYPE, config.run_id)
    except Exception as exc:
        if isinstance(exc, CorpusBuildError):
            raise
        raise CorpusBuildError("Corpus build failed while writing final manifest") from exc
    return manifest


def _run_raw_import(
    store: ArtifactStore,
    config: CorpusBuildConfig,
    artifact_ids: CorpusBuildArtifactIds,
    raw_import_client: Any | None,
) -> ArtifactManifest:
    if config.raw_source.source_type == "local_dir":
        source_dir = _local_dir_from_uri(config.raw_source.uri)
        return import_raw_dataset_from_local_dir(
            store,
            artifact_ids.raw_dataset,
            source_dir,
            dataset_name=config.dataset_name,
            dataset_revision=config.raw_source.dataset_revision,
            source_uri=config.raw_source.uri,
            import_parameters=_safe_user_metadata(config.raw_source.import_parameters),
            created_by=config.created_by,
            code_git_sha=config.code_git_sha,
            metadata={"corpus_build_run_id": config.run_id},
        )

    if config.raw_source.source_type == "s3_prefix":
        if raw_import_client is None:
            raise CorpusBuildError("raw_import_client is required for s3_prefix raw source")
        bucket, prefix = _parse_s3_prefix_uri(config.raw_source.uri)
        return import_raw_dataset_from_s3_prefix(
            store,
            artifact_ids.raw_dataset,
            client=raw_import_client,
            bucket=bucket,
            prefix=prefix,
            dataset_name=config.dataset_name,
            dataset_revision=config.raw_source.dataset_revision,
            source_uri=config.raw_source.uri,
            import_parameters=_safe_user_metadata(config.raw_source.import_parameters),
            created_by=config.created_by,
            code_git_sha=config.code_git_sha,
            metadata={"corpus_build_run_id": config.run_id},
        )

    raise CorpusBuildError(f"Unsupported raw source type: {config.raw_source.source_type}")
