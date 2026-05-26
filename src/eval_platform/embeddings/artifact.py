"""Embeddings artifact read/write helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from eval_platform.artifacts import (
    ArtifactDependency,
    ArtifactFile,
    ArtifactIncompleteError,
    ArtifactManifest,
    ArtifactStore,
)
from eval_platform.embeddings.jsonl import (
    VECTOR_DTYPE,
    VECTOR_ENCODING,
    dump_embeddings_jsonl,
    load_embeddings_jsonl,
)
from eval_platform.embeddings.schema import EmbeddedCorpus, EmbeddingProvenance, EmbeddingRecord

EMBEDDINGS_ARTIFACT_TYPE = "embeddings"
EMBEDDINGS_FILENAME = "embeddings.jsonl"
_SYSTEM_METADATA_FIELDS = {
    "embedding_count",
    "unique_chunk_count",
    "unique_doc_count",
    "embedding_dim",
    "embedding_dtype",
    "vector_encoding",
    "provenance",
    "source_chunked_corpus_artifact_id",
    "alignment_key",
    "alignment_order",
    "sharding",
    "shards",
}


class EmbeddingArtifactError(Exception):
    """Raised when embeddings artifact validation fails."""


class EmbeddingShardDescriptor(BaseModel):
    """Manifest metadata for one embedding shard."""

    shard_id: str
    source_chunk_file: str
    embedding_file: str
    source_chunk_count: int = Field(ge=0)
    embedding_count: int = Field(ge=0)
    first_chunk_id: str | None = None
    last_chunk_id: str | None = None
    sha256: str


class EmbeddingShard(BaseModel):
    """Loaded embedding shard with records."""

    shard_id: str
    source_chunk_file: str
    embedding_file: str
    source_chunk_count: int = Field(ge=0)
    embedding_count: int = Field(ge=0)
    first_chunk_id: str | None = None
    last_chunk_id: str | None = None
    sha256: str | None = None
    embeddings: list[EmbeddingRecord] = Field(default_factory=list)


def _sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _resolve_embedding_shard_descriptors(
    manifest: ArtifactManifest,
) -> list[EmbeddingShardDescriptor]:
    sharding = manifest.metadata.get("sharding")
    if isinstance(sharding, dict) and sharding.get("enabled"):
        return [
            EmbeddingShardDescriptor.model_validate(payload)
            for payload in manifest.metadata.get("shards", [])
        ]

    file_sha256 = None
    for artifact_file in manifest.files:
        if artifact_file.path == EMBEDDINGS_FILENAME:
            file_sha256 = artifact_file.sha256
            break
    embedding_count = int(manifest.metadata.get("embedding_count", 0))
    return [
        EmbeddingShardDescriptor(
            shard_id="part-00000",
            source_chunk_file="chunks.jsonl",
            embedding_file=EMBEDDINGS_FILENAME,
            source_chunk_count=embedding_count,
            embedding_count=embedding_count,
            first_chunk_id=None,
            last_chunk_id=None,
            sha256=file_sha256 or "",
        )
    ]


def iter_embedding_shards(
    store: ArtifactStore,
    artifact_id: str,
    *,
    require_complete: bool = True,
) -> Iterator[EmbeddingShard]:
    """Yield embedding shards in manifest order without preloading all shards."""

    if require_complete and not store.is_complete(EMBEDDINGS_ARTIFACT_TYPE, artifact_id):
        raise ArtifactIncompleteError(
            f"Artifact is incomplete: {EMBEDDINGS_ARTIFACT_TYPE}/{artifact_id}"
        )

    manifest = store.read_manifest(EMBEDDINGS_ARTIFACT_TYPE, artifact_id)
    for descriptor in _resolve_embedding_shard_descriptors(manifest):
        shard_text = store.get_file(
            EMBEDDINGS_ARTIFACT_TYPE, artifact_id, descriptor.embedding_file
        ).decode("utf-8")
        embeddings = load_embeddings_jsonl(shard_text)
        yield EmbeddingShard(
            shard_id=descriptor.shard_id,
            source_chunk_file=descriptor.source_chunk_file,
            embedding_file=descriptor.embedding_file,
            source_chunk_count=descriptor.source_chunk_count,
            embedding_count=descriptor.embedding_count or len(embeddings),
            first_chunk_id=descriptor.first_chunk_id
            or (embeddings[0].chunk_id if embeddings else None),
            last_chunk_id=descriptor.last_chunk_id
            or (embeddings[-1].chunk_id if embeddings else None),
            sha256=descriptor.sha256 or _sha256_hexdigest(shard_text.encode("utf-8")),
            embeddings=embeddings,
        )


def write_embedding_shards_artifact(
    store: ArtifactStore,
    artifact_id: str,
    shards: Iterable[EmbeddingShard],
    *,
    provenance: EmbeddingProvenance,
    metadata: dict[str, Any] | None = None,
    source_artifact_id: str | None = None,
    source_artifact_type: str | None = "chunked_corpus",
    created_at: datetime | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
) -> ArtifactManifest:
    """Write an embeddings artifact by streaming one shard at a time."""

    total_embedding_count = 0
    unique_chunk_ids: set[str] = set()
    unique_doc_ids: set[str] = set()
    files: list[ArtifactFile] = []
    shard_metadata: list[dict[str, Any]] = []
    sharded_output = False

    for shard in shards:
        shard_corpus = EmbeddedCorpus(embeddings=shard.embeddings)
        _validate_embedding_dimensions(shard_corpus, provenance)
        if shard.embedding_count != len(shard.embeddings):
            raise EmbeddingArtifactError("Embedding shard count does not match shard rows")
        shard_bytes = dump_embeddings_jsonl(shard.embeddings).encode("utf-8")
        shard_sha256 = _sha256_hexdigest(shard_bytes)
        shard.sha256 = shard_sha256
        store.put_file(EMBEDDINGS_ARTIFACT_TYPE, artifact_id, shard.embedding_file, shard_bytes)
        files.append(
            ArtifactFile(
                path=shard.embedding_file,
                size_bytes=len(shard_bytes),
                sha256=shard_sha256,
            )
        )
        shard_metadata.append(
            EmbeddingShardDescriptor(
                shard_id=shard.shard_id,
                source_chunk_file=shard.source_chunk_file,
                embedding_file=shard.embedding_file,
                source_chunk_count=shard.source_chunk_count,
                embedding_count=shard.embedding_count,
                first_chunk_id=shard.first_chunk_id,
                last_chunk_id=shard.last_chunk_id,
                sha256=shard_sha256,
            ).model_dump(mode="json")
        )
        total_embedding_count += shard.embedding_count
        unique_chunk_ids.update(record.chunk_id for record in shard.embeddings)
        unique_doc_ids.update(record.doc_id for record in shard.embeddings)
        sharded_output = sharded_output or shard.embedding_file != EMBEDDINGS_FILENAME

    manifest_metadata: dict[str, Any] = {}
    if metadata:
        manifest_metadata.update(metadata)
    manifest_metadata.update(
        {
            "embedding_count": total_embedding_count,
            "unique_chunk_count": len(unique_chunk_ids),
            "unique_doc_count": len(unique_doc_ids),
            "embedding_dim": provenance.embedding_dim,
            "embedding_dtype": VECTOR_DTYPE,
            "vector_encoding": VECTOR_ENCODING,
            "provenance": provenance.model_dump(mode="json"),
            "source_chunked_corpus_artifact_id": source_artifact_id,
            "alignment_key": "chunk_id",
            "alignment_order": "source_chunk_order",
            "sharding": {
                "enabled": sharded_output,
                "source_strategy": "source_doc_count" if sharded_output else None,
                "file_record_num": (
                    provenance.metadata.get("source_file_record_num")
                    if sharded_output
                    else None
                ),
            },
            "shards": shard_metadata if sharded_output else [],
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
        files=files,
    )
    store.write_manifest(EMBEDDINGS_ARTIFACT_TYPE, artifact_id, manifest)
    store.mark_success(EMBEDDINGS_ARTIFACT_TYPE, artifact_id)
    return manifest


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
    shards: list[EmbeddingShard] | None = None,
) -> ArtifactManifest:
    """Write an embeddings artifact to the given store."""

    _validate_embedding_dimensions(embedded_corpus, provenance)

    if shards is None:
        shards = [
            EmbeddingShard(
                shard_id="part-00000",
                source_chunk_file="chunks.jsonl",
                embedding_file=EMBEDDINGS_FILENAME,
                source_chunk_count=len(embedded_corpus.embeddings),
                embedding_count=len(embedded_corpus.embeddings),
                first_chunk_id=embedded_corpus.embeddings[0].chunk_id
                if embedded_corpus.embeddings
                else None,
                last_chunk_id=embedded_corpus.embeddings[-1].chunk_id
                if embedded_corpus.embeddings
                else None,
                embeddings=list(embedded_corpus.embeddings),
            )
        ]

    total_embedding_count = sum(shard.embedding_count for shard in shards)
    if total_embedding_count != len(embedded_corpus.embeddings):
        raise EmbeddingArtifactError("Shard embedding_count does not match total embedding_count")

    combined_metadata = dict(embedded_corpus.metadata)
    if metadata:
        combined_metadata.update(metadata)
    return write_embedding_shards_artifact(
        store,
        artifact_id,
        shards,
        provenance=provenance,
        metadata=combined_metadata,
        source_artifact_id=source_artifact_id,
        source_artifact_type=source_artifact_type,
        created_at=created_at,
        created_by=created_by,
        code_git_sha=code_git_sha,
    )


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
    embeddings: list[EmbeddingRecord] = []
    for shard in iter_embedding_shards(store, artifact_id, require_complete=False):
        embeddings.extend(shard.embeddings)

    corpus_metadata = {
        key: value for key, value in manifest.metadata.items() if key not in _SYSTEM_METADATA_FIELDS
    }
    return EmbeddedCorpus(
        embeddings=embeddings,
        metadata=corpus_metadata,
    )
