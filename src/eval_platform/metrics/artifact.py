"""Metrics run artifact read/write helpers."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from eval_platform.artifacts import (
    ArtifactDependency,
    ArtifactFile,
    ArtifactIncompleteError,
    ArtifactManifest,
    ArtifactStore,
)
from eval_platform.metrics.jsonl import dump_query_metrics_jsonl, load_query_metrics_jsonl
from eval_platform.metrics.schema import MetricsRunData, QueryMetricsRecord

METRICS_RUN_ARTIFACT_TYPE = "metrics_run"
METRICS_FILENAME = "metrics.json"
QUERY_METRICS_DIR = "query_metrics"
_SYSTEM_METADATA_FIELDS = {
    "stage",
    "main_score",
    "query_count",
    "evaluated_query_count",
    "queries_per_shard",
    "query_metric_file_count",
    "query_metric_record_count",
}


class MetricsArtifactError(Exception):
    """Raised when metrics artifact validation fails."""


def _sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _query_metric_shard_path(index: int) -> str:
    return f"{QUERY_METRICS_DIR}/part-{index:05d}.jsonl"


def write_metrics_run_artifact(
    store: ArtifactStore,
    artifact_id: str,
    data: MetricsRunData,
    *,
    queries_per_shard: int = 1000,
    metadata: dict[str, Any] | None = None,
    dependencies: list[ArtifactDependency] | None = None,
    created_at: datetime | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
) -> ArtifactManifest:
    """Write a metrics_run artifact."""

    if queries_per_shard <= 0:
        raise MetricsArtifactError("queries_per_shard must be greater than 0")

    summary_payload = data.model_dump_json(
        exclude={"query_metrics", "metadata"},
    ).encode("utf-8")
    store.put_file(METRICS_RUN_ARTIFACT_TYPE, artifact_id, METRICS_FILENAME, summary_payload)
    files = [
        ArtifactFile(
            path=METRICS_FILENAME,
            size_bytes=len(summary_payload),
            sha256=_sha256_hexdigest(summary_payload),
        )
    ]

    shard_count = max(1, (len(data.query_metrics) + queries_per_shard - 1) // queries_per_shard)
    for shard_index in range(shard_count):
        start = shard_index * queries_per_shard
        stop = start + queries_per_shard
        shard_records = data.query_metrics[start:stop]
        payload = dump_query_metrics_jsonl(shard_records).encode("utf-8")
        path = _query_metric_shard_path(shard_index)
        store.put_file(METRICS_RUN_ARTIFACT_TYPE, artifact_id, path, payload)
        files.append(
            ArtifactFile(
                path=path,
                size_bytes=len(payload),
                sha256=_sha256_hexdigest(payload),
            )
        )

    manifest_metadata = {
        key: value for key, value in (metadata or {}).items() if key not in _SYSTEM_METADATA_FIELDS
    }
    manifest_metadata.update(data.metadata)
    manifest_metadata.update(
        {
            "stage": "metrics_run",
            "main_score": data.main_score,
            "query_count": len(data.query_metrics),
            "evaluated_query_count": len(data.query_metrics),
            "queries_per_shard": queries_per_shard,
            "query_metric_file_count": len(files) - 1,
            "query_metric_record_count": len(data.query_metrics),
        }
    )

    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=METRICS_RUN_ARTIFACT_TYPE,
        created_at=created_at or datetime.now(UTC),
        created_by=created_by,
        code_git_sha=code_git_sha,
        dependencies=list(dependencies or []),
        metadata=manifest_metadata,
        files=files,
    )
    store.write_manifest(METRICS_RUN_ARTIFACT_TYPE, artifact_id, manifest)
    store.mark_success(METRICS_RUN_ARTIFACT_TYPE, artifact_id)
    return manifest


def read_metrics_run_artifact(
    store: ArtifactStore,
    artifact_id: str,
    *,
    require_complete: bool = True,
) -> MetricsRunData:
    """Read a metrics_run artifact."""

    if require_complete and not store.is_complete(METRICS_RUN_ARTIFACT_TYPE, artifact_id):
        raise ArtifactIncompleteError(
            f"Artifact is incomplete: {METRICS_RUN_ARTIFACT_TYPE}/{artifact_id}"
        )

    summary_payload = json.loads(
        store.get_file(METRICS_RUN_ARTIFACT_TYPE, artifact_id, METRICS_FILENAME).decode("utf-8")
    )
    manifest = store.read_manifest(METRICS_RUN_ARTIFACT_TYPE, artifact_id)
    records: list[QueryMetricsRecord] = []
    for artifact_file in manifest.files:
        if not artifact_file.path.startswith(f"{QUERY_METRICS_DIR}/"):
            continue
        payload = store.get_file(
            METRICS_RUN_ARTIFACT_TYPE,
            artifact_id,
            artifact_file.path,
        ).decode("utf-8")
        records.extend(load_query_metrics_jsonl(payload))
    metadata = {
        key: value
        for key, value in manifest.metadata.items()
        if key not in _SYSTEM_METADATA_FIELDS
    }
    return MetricsRunData.model_validate(
        {**summary_payload, "query_metrics": records, "metadata": metadata}
    )
