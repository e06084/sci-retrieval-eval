"""Local filesystem artifact store."""

import json
from pathlib import Path, PurePosixPath

from eval_platform.artifacts.manifest import ArtifactManifest
from eval_platform.artifacts.paths import (
    ManifestMismatchError,
    resolve_under_root,
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


class LocalArtifactStore(ArtifactStore):
    """Store artifacts on the local filesystem."""

    def __init__(self, root_dir: Path) -> None:
        self._root = root_dir

    def artifact_dir(self, artifact_type: str, artifact_id: str) -> Path:
        safe_type = validate_coordinate(artifact_type, "artifact_type")
        safe_id = validate_coordinate(artifact_id, "artifact_id")
        return resolve_under_root(self._root, safe_type, safe_id)

    def artifact_uri(self, artifact_type: str, artifact_id: str) -> str:
        return self.artifact_dir(artifact_type, artifact_id).resolve().as_uri()

    def _file_path(self, artifact_type: str, artifact_id: str, relative_path: str) -> Path:
        safe_path = validate_relative_path(relative_path)
        artifact_path = self.artifact_dir(artifact_type, artifact_id)
        return resolve_under_root(artifact_path, *PurePosixPath(safe_path).parts)

    def put_file(
        self,
        artifact_type: str,
        artifact_id: str,
        relative_path: str,
        data: bytes,
    ) -> None:
        target = self._file_path(artifact_type, artifact_id, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def get_file(self, artifact_type: str, artifact_id: str, relative_path: str) -> bytes:
        target = self._file_path(artifact_type, artifact_id, relative_path)
        if not target.is_file():
            raise ArtifactNotFoundError(
                f"File not found: {artifact_type}/{artifact_id}/{relative_path}"
            )
        return target.read_bytes()

    def exists(self, artifact_type: str, artifact_id: str, relative_path: str) -> bool:
        target = self._file_path(artifact_type, artifact_id, relative_path)
        return target.is_file()

    def list_artifacts(self, artifact_type: str | None = None) -> list[tuple[str, str]]:
        if not self._root.exists():
            return []

        if artifact_type is not None:
            validate_coordinate(artifact_type, "artifact_type")
            type_dirs = [self._root / artifact_type]
        else:
            type_dirs = sorted(self._root.iterdir())

        artifacts: list[tuple[str, str]] = []

        for type_dir in type_dirs:
            if not type_dir.is_dir():
                continue
            current_type = type_dir.name
            for artifact_dir in sorted(type_dir.iterdir()):
                if artifact_dir.is_dir():
                    artifacts.append((current_type, artifact_dir.name))

        return artifacts

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

        artifact_path = self.artifact_dir(artifact_type, artifact_id)
        artifact_path.mkdir(parents=True, exist_ok=True)
        manifest_path = artifact_path / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def read_manifest(self, artifact_type: str, artifact_id: str) -> ArtifactManifest:
        manifest_path = self.artifact_dir(artifact_type, artifact_id) / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ManifestNotFoundError(
                f"Manifest not found: {artifact_type}/{artifact_id}/{MANIFEST_FILENAME}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ArtifactManifest.model_validate(payload)

    def mark_success(self, artifact_type: str, artifact_id: str) -> None:
        artifact_path = self.artifact_dir(artifact_type, artifact_id)
        artifact_path.mkdir(parents=True, exist_ok=True)
        (artifact_path / SUCCESS_MARKER).touch()

    def is_complete(self, artifact_type: str, artifact_id: str) -> bool:
        artifact_path = self.artifact_dir(artifact_type, artifact_id)
        return (artifact_path / MANIFEST_FILENAME).is_file() and (
            artifact_path / SUCCESS_MARKER
        ).is_file()
