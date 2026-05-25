"""External chunker repository version checks."""

from __future__ import annotations

from pydantic import BaseModel, ValidationInfo, field_validator

from eval_platform.chunking.git import GitRepoState, ensure_git_repo_clean


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class ExternalChunkerRepoError(Exception):
    """Base error for external chunker repository validation."""


class ExternalChunkerRepoMismatchError(ExternalChunkerRepoError):
    """Raised when the checked-out repo does not match the requested version."""


class ExternalChunkerRepoSpec(BaseModel):
    """Expected external chunker checkout state."""

    repo_path: str
    expected_remote_url: str
    expected_commit_sha: str

    @field_validator("repo_path", "expected_remote_url", "expected_commit_sha")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


def verify_external_chunker_repo(spec: ExternalChunkerRepoSpec) -> GitRepoState:
    """Validate repo cleanliness, remote URL, and commit SHA."""
    state = ensure_git_repo_clean(spec.repo_path)

    if state.repo_url != spec.expected_remote_url:
        raise ExternalChunkerRepoMismatchError(
            "External chunker repo remote URL mismatch: "
            f"expected {spec.expected_remote_url!r}, got {state.repo_url!r}"
        )
    if state.commit_sha != spec.expected_commit_sha:
        raise ExternalChunkerRepoMismatchError(
            "External chunker repo commit SHA mismatch: "
            f"expected {spec.expected_commit_sha!r}, got {state.commit_sha!r}"
        )

    return state
