"""Chunking runner orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from eval_platform.artifacts.manifest import ArtifactDependency, ArtifactManifest
from eval_platform.artifacts.store import ArtifactStore
from eval_platform.chunking.artifact import build_chunk_shards, write_chunked_corpus_artifact
from eval_platform.chunking.git import ensure_git_repo_clean
from eval_platform.chunking.progress import ProgressReporter, report_progress
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
    file_record_num: int | None = Field(default=None, gt=0)
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
    *,
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Run chunking against a normalized dataset artifact and write output."""
    git_state = ensure_git_repo_clean(config.chunker_repo_path)
    dataset = read_normalized_dataset_artifact(store, config.source_artifact_id)
    chunks: list[ChunkRecord] = []
    completed_doc_count = 0
    current_doc_id: str | None = None
    current_doc_chunk_count = 0
    total_docs = len(dataset.corpus)

    for chunk in chunker.chunk_corpus(dataset):
        if current_doc_id is None:
            current_doc_id = chunk.doc_id
        elif chunk.doc_id != current_doc_id:
            completed_doc_count += 1
            report_progress(
                progress_reporter,
                stage="chunking",
                current=completed_doc_count,
                total=total_docs,
                message="Completed source document chunking",
                metadata={
                    "kind": "source_doc",
                    "doc_id": current_doc_id,
                    "chunk_count": current_doc_chunk_count,
                },
            )
            current_doc_id = chunk.doc_id
            current_doc_chunk_count = 0
        current_doc_chunk_count += 1
        chunks.append(chunk)

    if current_doc_id is not None:
        completed_doc_count += 1
        report_progress(
            progress_reporter,
            stage="chunking",
            current=completed_doc_count,
            total=total_docs,
            message="Completed source document chunking",
            metadata={
                "kind": "source_doc",
                "doc_id": current_doc_id,
                "chunk_count": current_doc_chunk_count,
            },
        )

    chunked_corpus = ChunkedCorpus(
        chunks=chunks,
        metadata=dict(dataset.metadata),
    )

    shard_plan = build_chunk_shards(chunks, file_record_num=config.file_record_num)
    if config.file_record_num is not None:
        for index, shard in enumerate(shard_plan, start=1):
            report_progress(
                progress_reporter,
                stage="chunking",
                current=index,
                total=len(shard_plan),
                message="Prepared chunk shard",
                metadata={
                    "kind": "shard",
                    "shard_id": shard.shard_id,
                    "source_doc_count": shard.source_doc_count,
                    "chunk_count": shard.chunk_count,
                },
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
        file_record_num=config.file_record_num,
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
        metadata=config.metadata,
    )
