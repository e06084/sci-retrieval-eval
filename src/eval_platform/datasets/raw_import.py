"""Import raw dataset metadata from local directories or S3 prefixes."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any

from eval_platform.artifacts.manifest import ArtifactManifest
from eval_platform.artifacts.s3 import normalize_prefix
from eval_platform.artifacts.store import ArtifactStore
from eval_platform.datasets.raw import (
    RawDatasetFile,
    RawDatasetSnapshot,
    build_content_fingerprint_sha256,
    write_raw_dataset_artifact,
)

_STREAM_CHUNK_SIZE = 1024 * 1024


class RawDatasetImportError(Exception):
    """Raised when raw dataset import validation fails."""


def _hash_stream(stream: Any) -> tuple[int, str]:
    digest = hashlib.sha256()
    total_size = 0

    while True:
        chunk = stream.read(_STREAM_CHUNK_SIZE)
        if not chunk:
            break
        if not isinstance(chunk, bytes):
            raise RawDatasetImportError("Stream chunk must be bytes")
        digest.update(chunk)
        total_size += len(chunk)

    return total_size, digest.hexdigest()


def _hash_local_file(path: Path) -> tuple[int, str]:
    with path.open("rb") as handle:
        return _hash_stream(handle)


def import_raw_dataset_from_local_dir(
    store: ArtifactStore,
    artifact_id: str,
    source_dir: Path,
    *,
    dataset_name: str,
    dataset_revision: str | None = None,
    source_uri: str | None = None,
    import_parameters: dict[str, Any] | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Import local raw file metadata into a raw_dataset snapshot artifact."""
    if not source_dir.is_dir():
        raise RawDatasetImportError(f"source_dir is not a directory: {source_dir}")

    files: list[RawDatasetFile] = []
    source_paths = sorted(path for path in source_dir.rglob("*") if path.is_file())

    for path in source_paths:
        relative_path = path.relative_to(source_dir).as_posix()
        size_bytes, sha256 = _hash_local_file(path)
        files.append(
            RawDatasetFile(
                path=relative_path,
                uri=path.resolve().as_uri(),
                size_bytes=size_bytes,
                sha256=sha256,
            )
        )

    snapshot = RawDatasetSnapshot(
        source_type="local_dir",
        source_uri=source_uri or source_dir.resolve().as_uri(),
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        files=files,
        content_fingerprint_sha256=build_content_fingerprint_sha256(files),
        import_parameters=dict(import_parameters or {}),
        metadata=dict(metadata or {}),
    )
    return write_raw_dataset_artifact(
        store,
        artifact_id,
        snapshot,
        created_by=created_by,
        code_git_sha=code_git_sha,
    )


def _list_s3_keys(client: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    continuation_token: str | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token is not None:
            request["ContinuationToken"] = continuation_token

        response = client.list_objects_v2(**request)
        keys.extend(item["Key"] for item in response.get("Contents", []))

        if not response.get("IsTruncated"):
            break

        continuation_token = response.get("NextContinuationToken")
        if not continuation_token:
            break

    return sorted(key for key in keys if key != prefix and not key.endswith("/"))


def import_raw_dataset_from_s3_prefix(
    store: ArtifactStore,
    artifact_id: str,
    *,
    client: Any,
    bucket: str,
    prefix: str,
    dataset_name: str,
    dataset_revision: str | None = None,
    source_uri: str | None = None,
    import_parameters: dict[str, Any] | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Import S3 raw file metadata into a raw_dataset snapshot artifact."""
    normalized_prefix = normalize_prefix(prefix)
    list_prefix = f"{normalized_prefix}/" if normalized_prefix else ""

    files: list[RawDatasetFile] = []

    for key in _list_s3_keys(client, bucket, list_prefix):
        relative_path = key[len(list_prefix) :] if list_prefix else key
        relative_path = PurePosixPath(relative_path).as_posix()
        response = client.get_object(Bucket=bucket, Key=key)
        size_bytes, sha256 = _hash_stream(response["Body"])
        files.append(
            RawDatasetFile(
                path=relative_path,
                uri=f"s3://{bucket}/{key}",
                size_bytes=size_bytes,
                sha256=sha256,
            )
        )

    snapshot = RawDatasetSnapshot(
        source_type="s3_prefix",
        source_uri=source_uri or f"s3://{bucket}/{list_prefix}".rstrip("/"),
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        files=files,
        content_fingerprint_sha256=build_content_fingerprint_sha256(files),
        import_parameters=dict(import_parameters or {}),
        metadata=dict(metadata or {}),
    )
    return write_raw_dataset_artifact(
        store,
        artifact_id,
        snapshot,
        created_by=created_by,
        code_git_sha=code_git_sha,
    )
