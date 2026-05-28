"""Benchmark suite artifact read/write helpers."""

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
from eval_platform.artifacts.types import BENCHMARK_SUITE_RUN_ARTIFACT_TYPE
from eval_platform.benchmark.suite import BenchmarkSuiteRunSummary

SUITE_SUMMARY_FILENAME = "summary.json"
_SYSTEM_METADATA_FIELDS = {
    "stage",
    "suite_run_id",
    "dataset_count",
    "setting_count",
    "item_count",
}


class BenchmarkSuiteArtifactError(Exception):
    """Raised when benchmark suite artifact validation fails."""


def _sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_benchmark_suite_run_artifact(
    store: ArtifactStore,
    artifact_id: str,
    summary: BenchmarkSuiteRunSummary,
    *,
    metadata: dict[str, Any] | None = None,
    dependencies: list[ArtifactDependency] | None = None,
    created_at: datetime | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
) -> ArtifactManifest:
    """Write a benchmark_suite_run artifact."""

    if summary.suite_run_id != artifact_id:
        raise BenchmarkSuiteArtifactError("summary suite_run_id must match artifact_id")

    payload = summary.model_dump_json().encode("utf-8")
    store.put_file(
        BENCHMARK_SUITE_RUN_ARTIFACT_TYPE,
        artifact_id,
        SUITE_SUMMARY_FILENAME,
        payload,
    )
    manifest_metadata = {
        key: value for key, value in (metadata or {}).items() if key not in _SYSTEM_METADATA_FIELDS
    }
    manifest_metadata.update(
        {
            "stage": BENCHMARK_SUITE_RUN_ARTIFACT_TYPE,
            "suite_run_id": summary.suite_run_id,
            "dataset_count": summary.dataset_count,
            "setting_count": summary.setting_count,
            "item_count": summary.item_count,
        }
    )
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=BENCHMARK_SUITE_RUN_ARTIFACT_TYPE,
        created_at=created_at or datetime.now(UTC),
        created_by=created_by,
        code_git_sha=code_git_sha,
        dependencies=list(dependencies or []),
        metadata=manifest_metadata,
        files=[
            ArtifactFile(
                path=SUITE_SUMMARY_FILENAME,
                size_bytes=len(payload),
                sha256=_sha256_hexdigest(payload),
            )
        ],
    )
    store.write_manifest(BENCHMARK_SUITE_RUN_ARTIFACT_TYPE, artifact_id, manifest)
    store.mark_success(BENCHMARK_SUITE_RUN_ARTIFACT_TYPE, artifact_id)
    return manifest


def read_benchmark_suite_run_artifact(
    store: ArtifactStore,
    artifact_id: str,
    *,
    require_complete: bool = True,
) -> BenchmarkSuiteRunSummary:
    """Read a benchmark_suite_run summary artifact."""

    if require_complete and not store.is_complete(
        BENCHMARK_SUITE_RUN_ARTIFACT_TYPE,
        artifact_id,
    ):
        raise ArtifactIncompleteError(
            f"Artifact is incomplete: {BENCHMARK_SUITE_RUN_ARTIFACT_TYPE}/{artifact_id}"
        )
    return BenchmarkSuiteRunSummary.model_validate_json(
        store.get_file(
            BENCHMARK_SUITE_RUN_ARTIFACT_TYPE,
            artifact_id,
            SUITE_SUMMARY_FILENAME,
        )
    )
