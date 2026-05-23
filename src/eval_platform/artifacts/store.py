"""Artifact store abstract interface."""

from abc import ABC, abstractmethod

from eval_platform.artifacts.manifest import ArtifactManifest

MANIFEST_FILENAME = "_MANIFEST.json"
SUCCESS_MARKER = "_SUCCESS"


class ArtifactStoreError(Exception):
    """Base error for artifact store operations."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when an artifact or file cannot be found."""


class ManifestNotFoundError(ArtifactStoreError):
    """Raised when an artifact manifest is missing."""


class ArtifactIncompleteError(ArtifactStoreError):
    """Raised when an artifact is missing _MANIFEST.json or _SUCCESS."""


class ArtifactStore(ABC):
    """Storage backend for pipeline artifacts."""

    @abstractmethod
    def put_file(
        self,
        artifact_type: str,
        artifact_id: str,
        relative_path: str,
        data: bytes,
    ) -> None:
        """Write a file into an artifact directory."""

    @abstractmethod
    def get_file(self, artifact_type: str, artifact_id: str, relative_path: str) -> bytes:
        """Read a file from an artifact directory."""

    @abstractmethod
    def exists(self, artifact_type: str, artifact_id: str, relative_path: str) -> bool:
        """Return whether a file exists in an artifact directory."""

    @abstractmethod
    def list_artifacts(self, artifact_type: str | None = None) -> list[tuple[str, str]]:
        """List stored artifacts as (artifact_type, artifact_id) pairs."""

    @abstractmethod
    def write_manifest(
        self,
        artifact_type: str,
        artifact_id: str,
        manifest: ArtifactManifest,
    ) -> None:
        """Persist an artifact manifest."""

    @abstractmethod
    def read_manifest(self, artifact_type: str, artifact_id: str) -> ArtifactManifest:
        """Load an artifact manifest."""

    @abstractmethod
    def mark_success(self, artifact_type: str, artifact_id: str) -> None:
        """Mark an artifact as successfully completed."""

    @abstractmethod
    def is_complete(self, artifact_type: str, artifact_id: str) -> bool:
        """Return whether an artifact has both manifest and success marker."""

    @abstractmethod
    def artifact_uri(self, artifact_type: str, artifact_id: str) -> str:
        """Return a backend-specific URI for an artifact."""
