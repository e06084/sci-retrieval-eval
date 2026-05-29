"""Inventory helpers for corpus asset planning."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from eval_platform.artifacts import ArtifactManifest, ArtifactStore
from eval_platform.artifacts.metadata_keys import (
    METADATA_KEY_ASSET_FINGERPRINT_SHA256,
    METADATA_KEY_CHUNKED_CORPUS_ARTIFACT_ID,
    METADATA_KEY_COLLECTION_NAME,
    METADATA_KEY_EMBEDDINGS_ARTIFACT_ID,
    METADATA_KEY_INDEX_NAME,
    METADATA_KEY_RAW_DATASET_ARTIFACT_ID,
    METADATA_KEY_SOURCE_CHUNKED_CORPUS_ARTIFACT_ID,
    METADATA_KEY_SOURCE_EMBEDDINGS_ARTIFACT_ID,
    METADATA_KEY_SOURCE_NORMALIZED_DATASET_ARTIFACT_ID,
)
from eval_platform.artifacts.store import SUCCESS_MARKER
from eval_platform.corpus_assets.naming import (
    ARTIFACT_STAGE_ORDER,
    raw_prefix_key,
    s3_uri,
)
from eval_platform.corpus_assets.registry import TARGET_DATASETS, DatasetSpec
from eval_platform.corpus_assets.s3 import raw_prefix_exists, redact_sensitive_values

_MANIFEST_SUMMARY_KEYS = {
    "dataset",
    "dataset_name",
    "task_name",
    "split",
    "normalizer_name",
    "corpus_count",
    "query_count",
    "qrel_count",
    "chunk_count",
    "embedding_count",
    "unique_chunk_count",
    "unique_doc_count",
    "embedding_dim",
    METADATA_KEY_ASSET_FINGERPRINT_SHA256,
    "source_artifact_id",
    METADATA_KEY_RAW_DATASET_ARTIFACT_ID,
    METADATA_KEY_SOURCE_NORMALIZED_DATASET_ARTIFACT_ID,
    METADATA_KEY_SOURCE_CHUNKED_CORPUS_ARTIFACT_ID,
    METADATA_KEY_SOURCE_EMBEDDINGS_ARTIFACT_ID,
    METADATA_KEY_CHUNKED_CORPUS_ARTIFACT_ID,
    METADATA_KEY_EMBEDDINGS_ARTIFACT_ID,
    METADATA_KEY_INDEX_NAME,
    METADATA_KEY_COLLECTION_NAME,
    "indexed_count",
    "inserted_count",
    "failed_count",
    "verified_count",
    "verified_document_count",
    "verified_entity_count",
}


def _manifest_metadata_summary(manifest: ArtifactManifest) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in _MANIFEST_SUMMARY_KEYS:
        if key in manifest.metadata:
            summary[key] = manifest.metadata[key]
    if manifest.dependencies:
        summary["dependencies"] = [
            dependency.model_dump(mode="json")
            for dependency in manifest.dependencies
        ]
    return redact_sensitive_values(summary)


def _manifest_dataset_match(manifest: ArtifactManifest, spec: DatasetSpec) -> bool:
    metadata = manifest.metadata
    candidates = {
        str(metadata.get("dataset", "")),
        str(metadata.get("dataset_name", "")),
        str(metadata.get("task_name", "")),
        str(metadata.get("dataset_slug", "")),
    }
    if spec.task_name in candidates or spec.slug in candidates:
        return True
    return manifest.artifact_id == spec.slug or manifest.artifact_id.startswith(
        f"{spec.slug}_"
    )


def inventory_corpus_assets(
    *,
    store: ArtifactStore,
    raw_client: Any,
    bucket: str,
    raw_prefix: str,
    datasets: list[DatasetSpec] | None = None,
) -> dict[str, Any]:
    """Inventory raw prefixes and corpus/index artifacts for target datasets."""

    selected = datasets or list(TARGET_DATASETS)
    artifact_manifests: dict[str, list[ArtifactManifest]] = {}
    for artifact_type in ARTIFACT_STAGE_ORDER:
        manifests: list[ArtifactManifest] = []
        for _current_type, artifact_id in store.list_artifacts(artifact_type):
            try:
                manifests.append(store.read_manifest(artifact_type, artifact_id))
            except Exception:
                manifests.append(
                    ArtifactManifest(
                        artifact_id=artifact_id,
                        artifact_type=artifact_type,
                        created_at=datetime.fromtimestamp(0, tz=UTC),
                        metadata={"manifest_read_error": True},
                    )
                )
        artifact_manifests[artifact_type] = manifests

    dataset_results: dict[str, Any] = {}
    for spec in selected:
        prefix_key = raw_prefix_key(raw_prefix, spec)
        raw_exists = raw_prefix_exists(raw_client, bucket=bucket, prefix=prefix_key)
        artifacts_by_type: dict[str, list[dict[str, Any]]] = {}
        missing: list[str] = []

        for artifact_type in ARTIFACT_STAGE_ORDER:
            records: list[dict[str, Any]] = []
            for manifest in artifact_manifests.get(artifact_type, []):
                if not _manifest_dataset_match(manifest, spec):
                    continue
                complete = store.is_complete(artifact_type, manifest.artifact_id)
                records.append(
                    {
                        "artifact_id": manifest.artifact_id,
                        "complete": complete,
                        "has_manifest": not manifest.metadata.get(
                            "manifest_read_error",
                            False,
                        ),
                        "has_success": store.exists(
                            artifact_type,
                            manifest.artifact_id,
                            SUCCESS_MARKER,
                        ),
                        "artifact_uri": store.artifact_uri(
                            artifact_type,
                            manifest.artifact_id,
                        ),
                        "metadata_summary": _manifest_metadata_summary(manifest),
                    }
                )
            artifacts_by_type[artifact_type] = sorted(
                records,
                key=lambda item: str(item["artifact_id"]),
            )
            if not any(record["complete"] for record in records):
                missing.append(artifact_type)

        if not raw_exists:
            missing.insert(0, "raw_prefix")

        dataset_results[spec.task_name] = {
            "slug": spec.slug,
            "raw_format": spec.raw_format,
            "raw_prefix": s3_uri(bucket, prefix_key),
            "raw_prefix_exists": raw_exists,
            "expected_raw_files": list(spec.expected_raw_files),
            "artifacts": artifacts_by_type,
            "missing": missing,
        }

    return {
        "datasets": dataset_results,
        "artifact_stage_order": ARTIFACT_STAGE_ORDER,
    }
