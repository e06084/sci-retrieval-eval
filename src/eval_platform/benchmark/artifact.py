"""Benchmark run artifact read/write helpers."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from eval_platform.artifacts import (
    ArtifactDependency,
    ArtifactFile,
    ArtifactIncompleteError,
    ArtifactManifest,
    ArtifactStore,
)
from eval_platform.benchmark.schema import BenchmarkRunSummary

BENCHMARK_RUN_ARTIFACT_TYPE = "benchmark_run"
SUMMARY_FILENAME = "summary.json"
_SYSTEM_METADATA_FIELDS = {
    "stage",
    "source_normalized_dataset_artifact_id",
    "retrieval_run_artifact_id",
    "metrics_run_artifact_id",
    "main_score_metric",
    "main_score",
}


class BenchmarkArtifactError(Exception):
    """Raised when benchmark artifact validation fails."""


def _sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_benchmark_run_artifact(
    store: ArtifactStore,
    artifact_id: str,
    summary: BenchmarkRunSummary,
    *,
    metadata: dict[str, Any] | None = None,
    dependencies: list[ArtifactDependency] | None = None,
    created_at: datetime | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
) -> ArtifactManifest:
    """Write a benchmark_run artifact."""

    if summary.benchmark_run_artifact_id != artifact_id:
        raise BenchmarkArtifactError("summary benchmark_run_artifact_id must match artifact_id")

    payload = summary.model_dump_json().encode("utf-8")
    store.put_file(BENCHMARK_RUN_ARTIFACT_TYPE, artifact_id, SUMMARY_FILENAME, payload)
    manifest_metadata = {
        key: value for key, value in (metadata or {}).items() if key not in _SYSTEM_METADATA_FIELDS
    }
    manifest_metadata.update(
        {
            "stage": "benchmark_run",
            "source_normalized_dataset_artifact_id": summary.source_normalized_dataset_artifact_id,
            "retrieval_run_artifact_id": summary.retrieval_run_artifact_id,
            "metrics_run_artifact_id": summary.metrics_run_artifact_id,
            "main_score_metric": summary.main_score_metric,
            "main_score": summary.main_score,
        }
    )
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=BENCHMARK_RUN_ARTIFACT_TYPE,
        created_at=created_at or datetime.now(UTC),
        created_by=created_by,
        code_git_sha=code_git_sha,
        dependencies=list(dependencies or []),
        metadata=manifest_metadata,
        files=[
            ArtifactFile(
                path=SUMMARY_FILENAME,
                size_bytes=len(payload),
                sha256=_sha256_hexdigest(payload),
            )
        ],
    )
    store.write_manifest(BENCHMARK_RUN_ARTIFACT_TYPE, artifact_id, manifest)
    store.mark_success(BENCHMARK_RUN_ARTIFACT_TYPE, artifact_id)
    return manifest


def read_benchmark_run_artifact(
    store: ArtifactStore,
    artifact_id: str,
    *,
    require_complete: bool = True,
) -> BenchmarkRunSummary:
    """Read a benchmark_run summary artifact."""

    if require_complete and not store.is_complete(BENCHMARK_RUN_ARTIFACT_TYPE, artifact_id):
        raise ArtifactIncompleteError(
            f"Artifact is incomplete: {BENCHMARK_RUN_ARTIFACT_TYPE}/{artifact_id}"
        )
    return BenchmarkRunSummary.model_validate_json(
        store.get_file(BENCHMARK_RUN_ARTIFACT_TYPE, artifact_id, SUMMARY_FILENAME)
    )
