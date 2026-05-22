"""S3 artifact store backend."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any

from eval_platform.artifacts.manifest import ArtifactManifest
from eval_platform.artifacts.paths import (
    InvalidArtifactPathError,
    ManifestMismatchError,
    validate_coordinate,
    validate_relative_path,
)
from eval_platform.artifacts.store import (
    MANIFEST_FILENAME,
    SUCCESS_MARKER,
    ArtifactNotFoundError,
    ArtifactStore,
    ManifestNotFoundError,
)


def normalize_prefix(prefix: str) -> str:
    """Normalize and validate an S3 key prefix."""
    if not prefix:
        return ""

    normalized = prefix.strip("/")
    if not normalized:
        return ""

    for part in normalized.split("/"):
        if part in {"", ".", ".."}:
            raise InvalidArtifactPathError("prefix contains invalid segment")

    return normalized


class S3ArtifactStore(ArtifactStore):
    """Store artifacts in an S3 bucket."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        client: Any | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("bucket must not be empty")

        self._bucket = bucket
        self._prefix = normalize_prefix(prefix)
        self._client = client if client is not None else self._create_default_client()

    @staticmethod
    def _create_default_client() -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3ArtifactStore. "
                "Install with: pip install 'sci-retrieval-eval[s3]'"
            ) from exc
        return boto3.client("s3")

    def _join_key_parts(self, *parts: str) -> str:
        segments = [part for part in parts if part]
        return "/".join(segments)

    def _artifact_prefix(self, artifact_type: str, artifact_id: str) -> str:
        safe_type = validate_coordinate(artifact_type, "artifact_type")
        safe_id = validate_coordinate(artifact_id, "artifact_id")
        return self._join_key_parts(self._prefix, safe_type, safe_id)

    def _object_key(
        self,
        artifact_type: str,
        artifact_id: str,
        relative_path: str,
    ) -> str:
        safe_path = validate_relative_path(relative_path)
        artifact_prefix = self._artifact_prefix(artifact_type, artifact_id)
        path_parts = PurePosixPath(safe_path.replace("\\", "/")).parts
        return self._join_key_parts(artifact_prefix, *path_parts)

    @staticmethod
    def _is_not_found_error(exc: BaseException) -> bool:
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return True
        return False

    def _put_object(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def _get_object_bytes(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if self._is_not_found_error(exc):
                raise ArtifactNotFoundError(f"File not found: s3://{self._bucket}/{key}") from exc
            raise

        body = response["Body"]
        return body.read() if hasattr(body, "read") else body

    def _object_exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if self._is_not_found_error(exc):
                return False
            raise
        return True

    def _parse_artifact_key(self, key: str) -> tuple[str, str] | None:
        relative = key
        if self._prefix:
            prefix_with_slash = f"{self._prefix}/"
            if relative.startswith(prefix_with_slash):
                relative = relative[len(prefix_with_slash) :]
            elif relative == self._prefix:
                return None
            else:
                return None

        parts = relative.split("/")
        if len(parts) < 2:
            return None

        return parts[0], parts[1]

    def put_file(
        self,
        artifact_type: str,
        artifact_id: str,
        relative_path: str,
        data: bytes,
    ) -> None:
        key = self._object_key(artifact_type, artifact_id, relative_path)
        self._put_object(key, data)

    def get_file(self, artifact_type: str, artifact_id: str, relative_path: str) -> bytes:
        key = self._object_key(artifact_type, artifact_id, relative_path)
        return self._get_object_bytes(key)

    def exists(self, artifact_type: str, artifact_id: str, relative_path: str) -> bool:
        key = self._object_key(artifact_type, artifact_id, relative_path)
        return self._object_exists(key)

    def list_artifacts(self, artifact_type: str | None = None) -> list[tuple[str, str]]:
        if artifact_type is not None:
            validate_coordinate(artifact_type, "artifact_type")
            list_prefix = self._join_key_parts(self._prefix, artifact_type)
        else:
            list_prefix = self._prefix

        list_prefix = f"{list_prefix}/" if list_prefix else ""

        artifacts: set[tuple[str, str]] = set()
        for key in self._list_object_keys(list_prefix):
            parsed = self._parse_artifact_key(key)
            if parsed is None:
                continue
            current_type, _current_id = parsed
            if artifact_type is not None and current_type != artifact_type:
                continue
            artifacts.add(parsed)

        return sorted(artifacts)

    def _list_object_keys(self, list_prefix: str) -> list[str]:
        keys: list[str] = []
        continuation_token: str | None = None

        while True:
            request: dict[str, str] = {"Bucket": self._bucket, "Prefix": list_prefix}
            if continuation_token is not None:
                request["ContinuationToken"] = continuation_token

            response = self._client.list_objects_v2(**request)
            keys.extend(item["Key"] for item in response.get("Contents", []))

            if not response.get("IsTruncated"):
                break

            continuation_token = response.get("NextContinuationToken")
            if not continuation_token:
                break

        return keys

    def write_manifest(
        self,
        artifact_type: str,
        artifact_id: str,
        manifest: ArtifactManifest,
    ) -> None:
        if manifest.artifact_type != artifact_type:
            raise ManifestMismatchError(
                "manifest.artifact_type does not match storage path artifact_type"
            )
        if manifest.artifact_id != artifact_id:
            raise ManifestMismatchError(
                "manifest.artifact_id does not match storage path artifact_id"
            )

        key = self._object_key(artifact_type, artifact_id, MANIFEST_FILENAME)
        payload = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
        self._put_object(key, payload.encode("utf-8"))

    def read_manifest(self, artifact_type: str, artifact_id: str) -> ArtifactManifest:
        key = self._object_key(artifact_type, artifact_id, MANIFEST_FILENAME)
        try:
            payload = self._get_object_bytes(key)
        except ArtifactNotFoundError as exc:
            raise ManifestNotFoundError(
                f"Manifest not found: s3://{self._bucket}/{key}"
            ) from exc

        return ArtifactManifest.model_validate(json.loads(payload.decode("utf-8")))

    def mark_success(self, artifact_type: str, artifact_id: str) -> None:
        key = self._object_key(artifact_type, artifact_id, SUCCESS_MARKER)
        self._put_object(key, b"")

    def is_complete(self, artifact_type: str, artifact_id: str) -> bool:
        manifest_key = self._object_key(artifact_type, artifact_id, MANIFEST_FILENAME)
        success_key = self._object_key(artifact_type, artifact_id, SUCCESS_MARKER)
        return self._object_exists(manifest_key) and self._object_exists(success_key)

    def artifact_uri(self, artifact_type: str, artifact_id: str) -> str:
        artifact_prefix = self._artifact_prefix(artifact_type, artifact_id)
        return f"s3://{self._bucket}/{artifact_prefix}/"
