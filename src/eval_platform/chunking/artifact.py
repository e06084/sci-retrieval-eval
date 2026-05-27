"""Chunked corpus artifact read/write helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from eval_platform.artifacts.manifest import (
    ArtifactDependency,
    ArtifactFile,
    ArtifactManifest,
)
from eval_platform.artifacts.store import ArtifactIncompleteError, ArtifactStore
from eval_platform.artifacts.types import CHUNKED_CORPUS_ARTIFACT_TYPE
from eval_platform.chunking.jsonl import dump_chunks_jsonl, load_chunks_jsonl
from eval_platform.chunking.schema import ChunkedCorpus, ChunkerProvenance, ChunkRecord

CHUNKS_FILENAME = "chunks.jsonl"
CHUNK_SHARD_DIRNAME = "chunks"
_SYSTEM_METADATA_FIELDS = {
    "chunk_count",
    "unique_doc_count",
    "chunker",
    "chunk_params",
    "sharding",
    "shards",
}


class ChunkShardDescriptor(BaseModel):
    """Manifest-level shard metadata for one chunk file."""

    shard_id: str
    path: str
    source_doc_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    first_chunk_id: str | None = None
    last_chunk_id: str | None = None
    sha256: str


class ChunkShard(BaseModel):
    """Loaded chunk shard with both metadata and chunk rows."""

    shard_id: str
    path: str
    source_doc_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    first_chunk_id: str | None = None
    last_chunk_id: str | None = None
    sha256: str | None = None
    chunks: list[ChunkRecord] = Field(default_factory=list)


def _sha256_hexdigest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chunk_shard_path(index: int) -> str:
    return f"{CHUNK_SHARD_DIRNAME}/part-{index:05d}.jsonl"


def _group_chunks_by_source_doc(chunks: list[ChunkRecord]) -> list[list[ChunkRecord]]:
    groups: list[list[ChunkRecord]] = []
    current_doc_id: str | None = None
    current_group: list[ChunkRecord] = []

    for chunk in chunks:
        if current_doc_id is None or chunk.doc_id != current_doc_id:
            if current_group:
                groups.append(current_group)
            current_doc_id = chunk.doc_id
            current_group = [chunk]
            continue
        current_group.append(chunk)

    if current_group:
        groups.append(current_group)
    return groups


def build_chunk_shards(
    chunks: list[ChunkRecord],
    *,
    file_record_num: int | None,
) -> list[ChunkShard]:
    """Partition chunks into shard-aligned groups."""

    if file_record_num is None:
        return [
            ChunkShard(
                shard_id="part-00000",
                path=CHUNKS_FILENAME,
                source_doc_count=len({chunk.doc_id for chunk in chunks}),
                chunk_count=len(chunks),
                first_chunk_id=chunks[0].chunk_id if chunks else None,
                last_chunk_id=chunks[-1].chunk_id if chunks else None,
                chunks=list(chunks),
            )
        ]

    if file_record_num <= 0:
        raise ValueError("file_record_num must be a positive integer")

    doc_groups = _group_chunks_by_source_doc(chunks)
    shards: list[ChunkShard] = []
    for index in range(0, len(doc_groups), file_record_num):
        shard_groups = doc_groups[index : index + file_record_num]
        shard_chunks = [chunk for group in shard_groups for chunk in group]
        shards.append(
            ChunkShard(
                shard_id=f"part-{len(shards):05d}",
                path=_chunk_shard_path(len(shards)),
                source_doc_count=len(shard_groups),
                chunk_count=len(shard_chunks),
                first_chunk_id=shard_chunks[0].chunk_id if shard_chunks else None,
                last_chunk_id=shard_chunks[-1].chunk_id if shard_chunks else None,
                chunks=shard_chunks,
            )
        )
    return shards


def _sharding_metadata(file_record_num: int | None) -> dict[str, Any]:
    if file_record_num is None:
        return {
            "enabled": False,
            "strategy": None,
            "file_record_num": None,
            "alignment_key": "chunk_id",
            "alignment_order": "source_chunk_order",
        }
    return {
        "enabled": True,
        "strategy": "source_doc_count",
        "file_record_num": file_record_num,
        "alignment_key": "chunk_id",
        "alignment_order": "source_chunk_order",
    }


def _read_user_corpus_metadata(manifest: ArtifactManifest) -> dict[str, Any]:
    return {
        key: value for key, value in manifest.metadata.items() if key not in _SYSTEM_METADATA_FIELDS
    }


def _resolve_chunk_shard_descriptors(manifest: ArtifactManifest) -> list[ChunkShardDescriptor]:
    sharding = manifest.metadata.get("sharding")
    if isinstance(sharding, dict) and sharding.get("enabled"):
        shard_payloads = manifest.metadata.get("shards", [])
        return [ChunkShardDescriptor.model_validate(payload) for payload in shard_payloads]

    file_sha256 = None
    for artifact_file in manifest.files:
        if artifact_file.path == CHUNKS_FILENAME:
            file_sha256 = artifact_file.sha256
            break
    chunk_count = int(manifest.metadata.get("chunk_count", 0))
    unique_doc_count = int(manifest.metadata.get("unique_doc_count", 0))
    return [
        ChunkShardDescriptor(
            shard_id="part-00000",
            path=CHUNKS_FILENAME,
            source_doc_count=unique_doc_count,
            chunk_count=chunk_count,
            first_chunk_id=None,
            last_chunk_id=None,
            sha256=file_sha256 or "",
        )
    ]


def iter_chunk_shards(
    store: ArtifactStore,
    artifact_id: str,
    *,
    require_complete: bool = True,
) -> Iterator[ChunkShard]:
    """Yield chunk shards in manifest order without preloading all shards."""

    if require_complete and not store.is_complete(CHUNKED_CORPUS_ARTIFACT_TYPE, artifact_id):
        raise ArtifactIncompleteError(
            f"Artifact is incomplete: {CHUNKED_CORPUS_ARTIFACT_TYPE}/{artifact_id}"
        )

    manifest = store.read_manifest(CHUNKED_CORPUS_ARTIFACT_TYPE, artifact_id)
    for descriptor in _resolve_chunk_shard_descriptors(manifest):
        shard_text = store.get_file(
            CHUNKED_CORPUS_ARTIFACT_TYPE, artifact_id, descriptor.path
        ).decode("utf-8")
        loaded_chunks = load_chunks_jsonl(shard_text)
        yield ChunkShard(
            shard_id=descriptor.shard_id,
            path=descriptor.path,
            source_doc_count=descriptor.source_doc_count,
            chunk_count=descriptor.chunk_count or len(loaded_chunks),
            first_chunk_id=descriptor.first_chunk_id
            or (loaded_chunks[0].chunk_id if loaded_chunks else None),
            last_chunk_id=descriptor.last_chunk_id
            or (loaded_chunks[-1].chunk_id if loaded_chunks else None),
            sha256=descriptor.sha256 or _sha256_hexdigest(shard_text.encode("utf-8")),
            chunks=loaded_chunks,
        )


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
    file_record_num: int | None = None,
) -> ArtifactManifest:
    """Write a chunked corpus artifact to the given store."""

    shards = build_chunk_shards(corpus.chunks, file_record_num=file_record_num)
    files: list[ArtifactFile] = []
    shard_metadata: list[dict[str, Any]] = []

    for shard in shards:
        shard_bytes = dump_chunks_jsonl(shard.chunks).encode("utf-8")
        shard_sha256 = _sha256_hexdigest(shard_bytes)
        shard.sha256 = shard_sha256
        store.put_file(CHUNKED_CORPUS_ARTIFACT_TYPE, artifact_id, shard.path, shard_bytes)
        files.append(
            ArtifactFile(path=shard.path, size_bytes=len(shard_bytes), sha256=shard_sha256)
        )
        shard_metadata.append(
            ChunkShardDescriptor(
                shard_id=shard.shard_id,
                path=shard.path,
                source_doc_count=shard.source_doc_count,
                chunk_count=shard.chunk_count,
                first_chunk_id=shard.first_chunk_id,
                last_chunk_id=shard.last_chunk_id,
                sha256=shard_sha256,
            ).model_dump(mode="json")
        )

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
            "sharding": _sharding_metadata(file_record_num),
            "shards": shard_metadata if file_record_num is not None else [],
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
        files=files,
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
    chunks: list[ChunkRecord] = []
    for shard in iter_chunk_shards(store, artifact_id, require_complete=False):
        chunks.extend(shard.chunks)

    return ChunkedCorpus(
        chunks=chunks,
        metadata=_read_user_corpus_metadata(manifest),
    )
