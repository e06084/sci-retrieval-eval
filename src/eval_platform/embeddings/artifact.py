"""Embeddings artifact read/write helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from eval_platform.artifacts import (
    ArtifactDependency,
    ArtifactFile,
    ArtifactIncompleteError,
    ArtifactManifest,
    ArtifactStore,
)
from eval_platform.embeddings.jsonl import dump_embeddings_jsonl, load_embeddings_jsonl
from eval_platform.embeddings.schema import EmbeddedCorpus, EmbeddingProvenance

EMBEDDINGS_ARTIFACT_TYPE = "embeddings"
EMBEDDINGS_FILENAME = "embeddings.jsonl"
_SYSTEM_METADATA_FIELDS = {
    "embedding_count",
    "unique_chunk_count",
    "unique_doc_count",
    "embedding_dim",
    "provenance",
}


class EmbeddingArtifactError(Exception):
    """Raised when embeddings artifact validation fails."""


def _validate_embedding_dimensions(
    embedded_corpus: EmbeddedCorpus,
    provenance: EmbeddingProvenance,
) -> None:
    dims = {len(record.vector) for record in embedded_corpus.embeddings}
    if len(dims) > 1:
        raise EmbeddingArtifactError("All embedding vectors must have the same dimension")
    if dims and next(iter(dims)) != provenance.embedding_dim:
        raise EmbeddingArtifactError(
            "Embedding vector dimension does not match provenance.embedding_dim"
        )


def write_embeddings_artifact(
    store: ArtifactStore,
    artifact_id: str,
    embedded_corpus: EmbeddedCorpus,
    *,
    provenance: EmbeddingProvenance,
    source_artifact_id: str | None = None,
    source_artifact_type: str | None = "chunked_corpus",
    created_at: datetime | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Write an embeddings artifact to the given store."""
    _validate_embedding_dimensions(embedded_corpus, provenance)

    embeddings_bytes = dump_embeddings_jsonl(embedded_corpus.embeddings).encode("utf-8")
    store.put_file(EMBEDDINGS_ARTIFACT_TYPE, artifact_id, EMBEDDINGS_FILENAME, embeddings_bytes)

    manifest_metadata: dict[str, Any] = {}
    manifest_metadata.update(embedded_corpus.metadata)
    if metadata:
        manifest_metadata.update(metadata)
    manifest_metadata.update(
        {
            "embedding_count": len(embedded_corpus.embeddings),
            "unique_chunk_count": len({record.chunk_id for record in embedded_corpus.embeddings}),
            "unique_doc_count": len({record.doc_id for record in embedded_corpus.embeddings}),
            "embedding_dim": provenance.embedding_dim,
            "provenance": provenance.model_dump(mode="json"),
        }
    )

    dependencies: list[ArtifactDependency] = []
    if source_artifact_id is not None:
        dependencies.append(
            ArtifactDependency(
                artifact_id=source_artifact_id,
                artifact_type=source_artifact_type or "chunked_corpus",
            )
        )

    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=EMBEDDINGS_ARTIFACT_TYPE,
        created_at=created_at or datetime.now(UTC),
        created_by=created_by,
        code_git_sha=code_git_sha,
        dependencies=dependencies,
        metadata=manifest_metadata,
        files=[ArtifactFile(path=EMBEDDINGS_FILENAME, size_bytes=len(embeddings_bytes))],
    )
    store.write_manifest(EMBEDDINGS_ARTIFACT_TYPE, artifact_id, manifest)
    store.mark_success(EMBEDDINGS_ARTIFACT_TYPE, artifact_id)
    return manifest


def read_embeddings_artifact(
    store: ArtifactStore,
    artifact_id: str,
    *,
    require_complete: bool = True,
) -> EmbeddedCorpus:
    """Read an embeddings artifact from the given store."""
    if require_complete and not store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, artifact_id):
        raise ArtifactIncompleteError(
            f"Artifact is incomplete: {EMBEDDINGS_ARTIFACT_TYPE}/{artifact_id}"
        )

    manifest = store.read_manifest(EMBEDDINGS_ARTIFACT_TYPE, artifact_id)
    corpus_metadata = {
        key: value for key, value in manifest.metadata.items() if key not in _SYSTEM_METADATA_FIELDS
    }
    embeddings_text = store.get_file(
        EMBEDDINGS_ARTIFACT_TYPE, artifact_id, EMBEDDINGS_FILENAME
    ).decode("utf-8")

    return EmbeddedCorpus(
        embeddings=load_embeddings_jsonl(embeddings_text),
        metadata=corpus_metadata,
    )
