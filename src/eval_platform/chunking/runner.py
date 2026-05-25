"""Chunking runner orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from eval_platform.artifacts.manifest import ArtifactDependency, ArtifactManifest
from eval_platform.artifacts.store import ArtifactStore
from eval_platform.chunking.artifact import write_chunked_corpus_artifact
from eval_platform.chunking.git import ensure_git_repo_clean
from eval_platform.chunking.schema import ChunkedCorpus, ChunkerProvenance, ChunkRecord
from eval_platform.datasets.normalized import read_normalized_dataset_artifact
from eval_platform.datasets.schema import NormalizedDataset


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


@runtime_checkable
class ExternalChunker(Protocol):
    """Protocol for injectable external chunker implementations."""

    def chunk_corpus(self, dataset: NormalizedDataset) -> Iterable[ChunkRecord]:
        """Chunk all documents in a normalized dataset."""


class ChunkingRunConfig(BaseModel):
    """Configuration for a chunking run."""

    source_artifact_id: str
    output_artifact_id: str
    chunker_name: str
    chunker_repo_path: str
    chunk_params: dict[str, Any] = Field(default_factory=dict)
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "source_artifact_id",
        "output_artifact_id",
        "chunker_name",
        "chunker_repo_path",
    )
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


def run_chunking(
    store: ArtifactStore,
    config: ChunkingRunConfig,
    chunker: ExternalChunker,
) -> ArtifactManifest:
    """Run chunking against a normalized dataset artifact and write output."""
    git_state = ensure_git_repo_clean(config.chunker_repo_path)
    dataset = read_normalized_dataset_artifact(store, config.source_artifact_id)
    chunks = list(chunker.chunk_corpus(dataset))

    chunked_corpus = ChunkedCorpus(
        chunks=chunks,
        metadata=dict(dataset.metadata),
    )

    chunker_provenance = ChunkerProvenance(
        name=config.chunker_name,
        repo_url=git_state.repo_url,
        repo_path=git_state.repo_path,
        commit_sha=git_state.commit_sha,
        branch=git_state.branch,
        is_dirty=git_state.is_dirty,
    )

    source_dependency = ArtifactDependency(
        artifact_id=config.source_artifact_id,
        artifact_type="normalized_dataset",
    )

    return write_chunked_corpus_artifact(
        store,
        config.output_artifact_id,
        chunked_corpus,
        source_dependency=source_dependency,
        chunker=chunker_provenance,
        chunk_params=config.chunk_params,
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
        metadata=config.metadata,
    )
