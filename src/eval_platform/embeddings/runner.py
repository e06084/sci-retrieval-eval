"""Embedding runner orchestration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from eval_platform.artifacts import ArtifactManifest, ArtifactStore
from eval_platform.chunking import read_chunked_corpus_artifact
from eval_platform.embeddings.artifact import write_embeddings_artifact
from eval_platform.embeddings.client import EmbeddingClient
from eval_platform.embeddings.schema import EmbeddedCorpus, EmbeddingProvenance, EmbeddingRecord


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class EmbeddingRunError(Exception):
    """Raised when embedding runner validation fails."""


class EmbeddingRunConfig(BaseModel):
    """Configuration for an embedding run."""

    source_artifact_id: str
    output_artifact_id: str
    model_name: str
    embedding_dim: int = Field(gt=0)
    provider: str | None = None
    api_version: str | None = None
    normalized: bool | None = None
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_artifact_id", "output_artifact_id", "model_name")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


def run_embedding(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: EmbeddingRunConfig,
    client: EmbeddingClient,
) -> ArtifactManifest:
    """Read chunked corpus, compute embeddings, and write an embeddings artifact."""
    chunked_corpus = read_chunked_corpus_artifact(source_store, config.source_artifact_id)
    texts = [chunk.text for chunk in chunked_corpus.chunks]
    vectors = client.embed_texts(texts)

    if len(vectors) != len(chunked_corpus.chunks):
        raise EmbeddingRunError("Embedding client returned a different number of vectors")

    records: list[EmbeddingRecord] = []
    for chunk, vector in zip(chunked_corpus.chunks, vectors, strict=True):
        if len(vector) != config.embedding_dim:
            raise EmbeddingRunError("Embedding client returned vectors with unexpected dimension")
        record_metadata = dict(chunk.metadata)
        if chunk.title is not None:
            record_metadata["title"] = chunk.title
        record_metadata["chunk_index"] = chunk.chunk_index
        if chunk.start_offset is not None:
            record_metadata["start_offset"] = chunk.start_offset
        if chunk.end_offset is not None:
            record_metadata["end_offset"] = chunk.end_offset
        records.append(
            EmbeddingRecord(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                vector=vector,
                metadata=record_metadata,
            )
        )

    embedded_corpus = EmbeddedCorpus(
        embeddings=records,
        metadata=dict(chunked_corpus.metadata),
    )
    provenance = EmbeddingProvenance(
        model_name=config.model_name,
        provider=config.provider,
        api_version=config.api_version,
        embedding_dim=config.embedding_dim,
        normalized=config.normalized,
    )

    return write_embeddings_artifact(
        output_store,
        config.output_artifact_id,
        embedded_corpus,
        provenance=provenance,
        source_artifact_id=config.source_artifact_id,
        source_artifact_type="chunked_corpus",
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
        metadata=config.metadata,
    )
