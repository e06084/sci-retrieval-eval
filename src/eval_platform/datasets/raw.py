"""Raw dataset artifact schema and manifest-only read/write helpers."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from eval_platform.artifacts.manifest import ArtifactManifest
from eval_platform.artifacts.store import ArtifactIncompleteError, ArtifactStore
from eval_platform.artifacts.types import RAW_DATASET_ARTIFACT_TYPE

_SYSTEM_METADATA_KEYS = {
    "stage",
    "source_type",
    "source_uri",
    "dataset_name",
    "dataset_revision",
    "file_count",
    "total_size_bytes",
    "files",
    "content_fingerprint_sha256",
    "import_parameters",
}


class RawDatasetArtifactError(Exception):
    """Raised when raw dataset artifact validation fails."""


class RawDatasetFile(BaseModel):
    """Metadata describing one immutable raw source file."""

    path: str
    uri: str
    size_bytes: int = Field(ge=0)
    sha256: str


class RawDatasetSnapshot(BaseModel):
    """In-memory representation of a raw dataset snapshot artifact."""

    source_type: str
    source_uri: str
    dataset_name: str
    dataset_revision: str | None = None
    files: list[RawDatasetFile] = Field(default_factory=list)
    content_fingerprint_sha256: str
    import_parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_content_fingerprint_sha256(files: list[RawDatasetFile]) -> str:
    """Build a deterministic dataset-level fingerprint from sorted file metadata."""
    digest = hashlib.sha256()
    for file in sorted(files, key=lambda item: item.path):
        digest.update(file.path.encode("utf-8"))
        digest.update(b"\t")
        digest.update(file.uri.encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(file.size_bytes).encode("utf-8"))
        digest.update(b"\t")
        digest.update(file.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_raw_dataset_artifact(
    store: ArtifactStore,
    artifact_id: str,
    snapshot: RawDatasetSnapshot,
    *,
    created_at: datetime | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Write a manifest-only raw dataset snapshot artifact to the given store."""
    sorted_files = sorted(snapshot.files, key=lambda item: item.path)
    expected_fingerprint = build_content_fingerprint_sha256(sorted_files)
    if snapshot.content_fingerprint_sha256 != expected_fingerprint:
        raise RawDatasetArtifactError("content_fingerprint_sha256 does not match file metadata")

    total_size_bytes = sum(file.size_bytes for file in sorted_files)

    manifest_metadata: dict[str, Any] = {}
    manifest_metadata.update(snapshot.metadata)
    if metadata:
        manifest_metadata.update(metadata)
    manifest_metadata.update(
        {
            "stage": "raw_dataset",
            "source_type": snapshot.source_type,
            "source_uri": snapshot.source_uri,
            "dataset_name": snapshot.dataset_name,
            "dataset_revision": snapshot.dataset_revision,
            "file_count": len(sorted_files),
            "total_size_bytes": total_size_bytes,
            "files": [file.model_dump(mode="json") for file in sorted_files],
            "content_fingerprint_sha256": snapshot.content_fingerprint_sha256,
            "import_parameters": dict(snapshot.import_parameters),
        }
    )

    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=RAW_DATASET_ARTIFACT_TYPE,
        created_at=created_at or datetime.now(UTC),
        created_by=created_by,
        code_git_sha=code_git_sha,
        metadata=manifest_metadata,
        files=[],
    )
    store.write_manifest(RAW_DATASET_ARTIFACT_TYPE, artifact_id, manifest)
    store.mark_success(RAW_DATASET_ARTIFACT_TYPE, artifact_id)
    return manifest


def read_raw_dataset_artifact(
    store: ArtifactStore,
    artifact_id: str,
    *,
    require_complete: bool = True,
) -> RawDatasetSnapshot:
    """Read a raw dataset artifact manifest from the given store."""
    if require_complete and not store.is_complete(RAW_DATASET_ARTIFACT_TYPE, artifact_id):
        raise ArtifactIncompleteError(
            f"Artifact is incomplete: {RAW_DATASET_ARTIFACT_TYPE}/{artifact_id}"
        )

    manifest = store.read_manifest(RAW_DATASET_ARTIFACT_TYPE, artifact_id)
    snapshot_metadata = {
        key: value for key, value in manifest.metadata.items() if key not in _SYSTEM_METADATA_KEYS
    }
    return RawDatasetSnapshot(
        source_type=str(manifest.metadata["source_type"]),
        source_uri=str(manifest.metadata["source_uri"]),
        dataset_name=str(manifest.metadata["dataset_name"]),
        dataset_revision=manifest.metadata.get("dataset_revision"),
        files=[RawDatasetFile.model_validate(item) for item in manifest.metadata["files"]],
        content_fingerprint_sha256=str(manifest.metadata["content_fingerprint_sha256"]),
        import_parameters=dict(manifest.metadata.get("import_parameters") or {}),
        metadata=snapshot_metadata,
    )
