"""Path validation helpers for artifact storage."""

from pathlib import Path, PurePosixPath

from eval_platform.artifacts.store import ArtifactStoreError


class InvalidArtifactPathError(ArtifactStoreError):
    """Raised when artifact coordinates or relative paths are unsafe."""


class ManifestMismatchError(ArtifactStoreError):
    """Raised when manifest coordinates do not match the storage path."""


def validate_coordinate(value: str, field_name: str) -> str:
    """Validate a single-segment artifact coordinate."""
    if not value:
        raise InvalidArtifactPathError(f"{field_name} must not be empty")
    if value in {".", ".."}:
        raise InvalidArtifactPathError(f"{field_name} must not be '.' or '..'")
    if "/" in value or "\\" in value:
        raise InvalidArtifactPathError(f"{field_name} must not contain path separators")
    if ".." in value:
        raise InvalidArtifactPathError(f"{field_name} must not contain '..'")
    return value


def validate_relative_path(relative_path: str) -> str:
    """Validate a relative file path within an artifact directory."""
    if not relative_path:
        raise InvalidArtifactPathError("relative_path must not be empty")

    path = PurePosixPath(relative_path.replace("\\", "/"))
    if path.is_absolute():
        raise InvalidArtifactPathError("relative_path must not be absolute")
    if ".." in path.parts:
        raise InvalidArtifactPathError("relative_path must not contain '..'")
    if any(part in {"", "."} for part in path.parts):
        raise InvalidArtifactPathError("relative_path must not contain empty or '.' segments")

    return relative_path


def resolve_under_root(root: Path, *parts: str) -> Path:
    """Resolve a path and ensure it stays under root."""
    target = root.joinpath(*parts).resolve()
    root_resolved = root.resolve()
    if not target.is_relative_to(root_resolved):
        raise InvalidArtifactPathError("path escapes artifact root")
    return target
