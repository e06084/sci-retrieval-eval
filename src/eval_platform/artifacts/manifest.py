"""Artifact manifest schema."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ArtifactDependency(BaseModel):
    """Reference to an upstream artifact."""

    artifact_id: str
    artifact_type: str | None = None


class ArtifactFile(BaseModel):
    """Metadata for a file belonging to an artifact."""

    path: str
    size_bytes: int | None = None
    sha256: str | None = None


class ArtifactManifest(BaseModel):
    """Manifest describing a pipeline artifact."""

    schema_version: str = "1"
    artifact_id: str
    artifact_type: str
    created_at: datetime
    created_by: str | None = None
    code_git_sha: str | None = None
    dependencies: list[ArtifactDependency] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    files: list[ArtifactFile] = Field(default_factory=list)
