"""Chunked corpus artifact read/write helpers."""

from datetime import UTC, datetime
from typing import Any

from eval_platform.artifacts.manifest import (
    ArtifactDependency,
    ArtifactFile,
    ArtifactManifest,
)
from eval_platform.artifacts.store import ArtifactIncompleteError, ArtifactStore
from eval_platform.chunking.jsonl import dump_chunks_jsonl, load_chunks_jsonl
from eval_platform.chunking.schema import ChunkedCorpus, ChunkerProvenance

CHUNKED_CORPUS_ARTIFACT_TYPE = "chunked_corpus"
CHUNKS_FILENAME = "chunks.jsonl"


def write_chunked_corpus_artifact(
    store: ArtifactStore,
    artifact_id: str,
    corpus: ChunkedCorpus,
    *,
    created_at: datetime | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
    metadata: dict[str, Any] | None = None,
    chunker: ChunkerProvenance | None = None,
    chunk_params: dict[str, Any] | None = None,
    source_dependency: ArtifactDependency | None = None,
) -> ArtifactManifest:
    """Write a chunked corpus artifact to the given store."""
    chunks_bytes = dump_chunks_jsonl(corpus.chunks).encode("utf-8")
    store.put_file(CHUNKED_CORPUS_ARTIFACT_TYPE, artifact_id, CHUNKS_FILENAME, chunks_bytes)

    manifest_metadata: dict[str, Any] = {}
    manifest_metadata.update(corpus.metadata)
    if metadata:
        manifest_metadata.update(metadata)
    if chunker is not None:
        manifest_metadata["chunker"] = chunker.model_dump(mode="json")
    if chunk_params is not None:
        manifest_metadata["chunk_params"] = chunk_params
    manifest_metadata.update(
        {
            "chunk_count": len(corpus.chunks),
            "unique_doc_count": len({chunk.doc_id for chunk in corpus.chunks}),
        }
    )

    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=CHUNKED_CORPUS_ARTIFACT_TYPE,
        created_at=created_at or datetime.now(UTC),
        created_by=created_by,
        code_git_sha=code_git_sha,
        dependencies=[source_dependency] if source_dependency is not None else [],
        metadata=manifest_metadata,
        files=[ArtifactFile(path=CHUNKS_FILENAME, size_bytes=len(chunks_bytes))],
    )
    store.write_manifest(CHUNKED_CORPUS_ARTIFACT_TYPE, artifact_id, manifest)
    store.mark_success(CHUNKED_CORPUS_ARTIFACT_TYPE, artifact_id)
    return manifest


def read_chunked_corpus_artifact(
    store: ArtifactStore,
    artifact_id: str,
    *,
    require_complete: bool = True,
) -> ChunkedCorpus:
    """Read a chunked corpus artifact from the given store."""
    if require_complete and not store.is_complete(CHUNKED_CORPUS_ARTIFACT_TYPE, artifact_id):
        raise ArtifactIncompleteError(
            f"Artifact is incomplete: {CHUNKED_CORPUS_ARTIFACT_TYPE}/{artifact_id}"
        )

    manifest = store.read_manifest(CHUNKED_CORPUS_ARTIFACT_TYPE, artifact_id)
    corpus_metadata = {
        key: value
        for key, value in manifest.metadata.items()
        if key not in {"chunk_count", "unique_doc_count", "chunker", "chunk_params"}
    }

    chunks_text = store.get_file(
        CHUNKED_CORPUS_ARTIFACT_TYPE, artifact_id, CHUNKS_FILENAME
    ).decode("utf-8")

    return ChunkedCorpus(
        chunks=load_chunks_jsonl(chunks_text),
        metadata=corpus_metadata,
    )
