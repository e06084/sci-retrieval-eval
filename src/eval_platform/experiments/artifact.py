"""experiment_run artifact read/write helpers."""

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
from eval_platform.artifacts.types import EXPERIMENT_RUN_ARTIFACT_TYPE
from eval_platform.experiments.schema import ExperimentRunSummary

EXPERIMENT_SUMMARY_FILENAME = "summary.json"
_SYSTEM_METADATA_FIELDS = {
    "stage",
    "experiment_run_id",
    "dataset_count",
    "setting_count",
    "item_count",
}


class ExperimentArtifactError(Exception):
    """Raised when experiment artifact validation fails."""


def _sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_experiment_run_artifact(
    store: ArtifactStore,
    artifact_id: str,
    summary: ExperimentRunSummary,
    *,
    metadata: dict[str, Any] | None = None,
    dependencies: list[ArtifactDependency] | None = None,
    created_at: datetime | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
) -> ArtifactManifest:
    """Write an experiment_run summary artifact."""

    if summary.experiment_run_id != artifact_id:
        raise ExperimentArtifactError("summary experiment_run_id must match artifact_id")

    payload = summary.model_dump_json().encode("utf-8")
    store.put_file(
        EXPERIMENT_RUN_ARTIFACT_TYPE,
        artifact_id,
        EXPERIMENT_SUMMARY_FILENAME,
        payload,
    )
    manifest_metadata = {
        key: value for key, value in (metadata or {}).items() if key not in _SYSTEM_METADATA_FIELDS
    }
    manifest_metadata.update(
        {
            "stage": EXPERIMENT_RUN_ARTIFACT_TYPE,
            "experiment_run_id": summary.experiment_run_id,
            "dataset_count": summary.dataset_count,
            "setting_count": summary.setting_count,
            "item_count": summary.item_count,
        }
    )
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=EXPERIMENT_RUN_ARTIFACT_TYPE,
        created_at=created_at or datetime.now(UTC),
        created_by=created_by,
        code_git_sha=code_git_sha,
        dependencies=list(dependencies or []),
        metadata=manifest_metadata,
        files=[
            ArtifactFile(
                path=EXPERIMENT_SUMMARY_FILENAME,
                size_bytes=len(payload),
                sha256=_sha256_hexdigest(payload),
            )
        ],
    )
    store.write_manifest(EXPERIMENT_RUN_ARTIFACT_TYPE, artifact_id, manifest)
    store.mark_success(EXPERIMENT_RUN_ARTIFACT_TYPE, artifact_id)
    return manifest


def read_experiment_run_artifact(
    store: ArtifactStore,
    artifact_id: str,
    *,
    require_complete: bool = True,
) -> ExperimentRunSummary:
    """Read an experiment_run summary artifact."""

    if require_complete and not store.is_complete(EXPERIMENT_RUN_ARTIFACT_TYPE, artifact_id):
        raise ArtifactIncompleteError(
            f"Artifact is incomplete: {EXPERIMENT_RUN_ARTIFACT_TYPE}/{artifact_id}"
        )
    return ExperimentRunSummary.model_validate_json(
        store.get_file(
            EXPERIMENT_RUN_ARTIFACT_TYPE,
            artifact_id,
            EXPERIMENT_SUMMARY_FILENAME,
        )
    )
