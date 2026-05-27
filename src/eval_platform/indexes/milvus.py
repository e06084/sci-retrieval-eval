"""Milvus ingest helpers for aligned chunk and embedding artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import zip_longest
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest, ArtifactStore
from eval_platform.artifacts.metadata_keys import (
    METADATA_KEY_COLLECTION_NAME,
    METADATA_KEY_SOURCE_CHUNKED_CORPUS_ARTIFACT_ID,
    METADATA_KEY_SOURCE_EMBEDDINGS_ARTIFACT_ID,
)
from eval_platform.artifacts.types import MILVUS_COLLECTION_ARTIFACT_TYPE
from eval_platform.chunking import CHUNKED_CORPUS_ARTIFACT_TYPE, ChunkRecord, iter_chunk_shards
from eval_platform.chunking.artifact import ChunkShard
from eval_platform.chunking.progress import ProgressReporter, report_progress
from eval_platform.embeddings import (
    EMBEDDINGS_ARTIFACT_TYPE,
    EmbeddingRecord,
    iter_embedding_shards,
)
from eval_platform.embeddings.artifact import EmbeddingShard

DEFAULT_PRIMARY_KEY_FIELD = "chunk_id"
DEFAULT_VECTOR_FIELD = "vector"
_SENSITIVE_METADATA_KEY_PARTS = (
    "access_key",
    "api_key",
    "password",
    "secret",
    "token",
)


class MilvusIngestError(Exception):
    """Raised when Milvus ingest fails."""


class MilvusInsertFailure(BaseModel):
    """One failed Milvus insert item summary."""

    primary_key: str
    error: str | None = None


class MilvusRow(BaseModel):
    """One Milvus insert row with its primary key."""

    primary_key: str
    row: dict[str, Any]


class MilvusInsertResult(BaseModel):
    """Result returned by Milvus insert."""

    inserted_count: int
    failed_items: list[MilvusInsertFailure] = Field(default_factory=list)


class MilvusClientProtocol(Protocol):
    """Minimal Milvus client protocol used by the ingest runner."""

    def collection_exists(self, collection_name: str) -> bool:
        """Return whether a collection exists."""
        ...

    def create_collection(
        self,
        collection_name: str,
        schema: dict[str, Any],
        index_params: dict[str, Any],
    ) -> None:
        """Create a collection with explicit schema and index params."""
        ...

    def drop_collection(self, collection_name: str) -> None:
        """Drop an existing collection."""
        ...

    def insert_rows(
        self,
        collection_name: str,
        rows: Sequence[MilvusRow],
    ) -> MilvusInsertResult:
        """Insert one ordered batch of rows."""
        ...

    def flush_collection(self, collection_name: str) -> None:
        """Flush a collection."""
        ...

    def count_entities(self, collection_name: str) -> int:
        """Return the number of entities in a collection."""
        ...


class MilvusIngestConfig(BaseModel):
    """Configuration for chunked_corpus + embeddings to Milvus ingest."""

    model_config = ConfigDict(populate_by_name=True)

    chunked_corpus_artifact_id: str
    embeddings_artifact_id: str
    output_artifact_id: str
    collection_name: str
    batch_size: int = Field(default=500, gt=0)
    overwrite_existing: bool = False
    flush: bool = True
    verify_count: bool = True
    vector_dim: int | None = Field(default=None, gt=0)
    primary_key_field: str = DEFAULT_PRIMARY_KEY_FIELD
    vector_field: str = DEFAULT_VECTOR_FIELD
    metric_type: str = "COSINE"
    index_params: dict[str, Any] = Field(default_factory=dict)
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "chunked_corpus_artifact_id",
        "embeddings_artifact_id",
        "output_artifact_id",
        "collection_name",
        "primary_key_field",
        "vector_field",
        "metric_type",
    )
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name or 'field'} must not be empty")
        return value


class PymilvusMilvusClientConfig(BaseModel):
    """Configuration for the lazy-import pymilvus adapter."""

    uri: str
    token: str | None = None
    username: str | None = None
    password: str | None = None
    db_name: str | None = None

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name or 'field'} must not be empty")
        return value


class PymilvusMilvusClient:
    """Small lazy-import pymilvus adapter covering only ingest runner needs."""

    def __init__(self, config: PymilvusMilvusClientConfig, *, client: Any | None = None) -> None:
        self._config = config
        if client is not None:
            self._client = client
            return

        try:
            from pymilvus import MilvusClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MilvusIngestError(
                "pymilvus is required for PymilvusMilvusClient; install the milvus extra"
            ) from exc

        kwargs: dict[str, Any] = {"uri": config.uri}
        if config.token is not None:
            kwargs["token"] = config.token
        if config.username is not None:
            kwargs["user"] = config.username
        if config.password is not None:
            kwargs["password"] = config.password
        if config.db_name is not None:
            kwargs["db_name"] = config.db_name
        self._client = MilvusClient(**kwargs)

    def collection_exists(self, collection_name: str) -> bool:
        return bool(self._client.has_collection(collection_name))

    def create_collection(
        self,
        collection_name: str,
        schema: dict[str, Any],
        index_params: dict[str, Any],
    ) -> None:
        pymilvus_schema = self._build_pymilvus_schema(schema)
        pymilvus_index_params = self._build_pymilvus_index_params(schema, index_params)
        self._client.create_collection(
            collection_name=collection_name,
            schema=pymilvus_schema,
            index_params=pymilvus_index_params,
        )

    def drop_collection(self, collection_name: str) -> None:
        self._client.drop_collection(collection_name=collection_name)

    def insert_rows(
        self,
        collection_name: str,
        rows: Sequence[MilvusRow],
    ) -> MilvusInsertResult:
        result = self._client.insert(
            collection_name=collection_name,
            data=[row.row for row in rows],
        )
        if isinstance(result, dict):
            insert_count = result.get("insert_count")
            ids = result.get("ids")
            if isinstance(insert_count, int):
                return MilvusInsertResult(inserted_count=insert_count)
            if isinstance(ids, list):
                return MilvusInsertResult(inserted_count=len(ids))
        return MilvusInsertResult(inserted_count=len(rows))

    def flush_collection(self, collection_name: str) -> None:
        self._client.flush(collection_name=collection_name)

    def count_entities(self, collection_name: str) -> int:
        result = self._client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        if not result:
            raise MilvusIngestError("Milvus count query returned no rows")
        count = result[0].get("count(*)") if isinstance(result[0], dict) else None
        if not isinstance(count, int):
            raise MilvusIngestError("Milvus count query missing integer count")
        return count

    @staticmethod
    def _build_pymilvus_schema(schema: dict[str, Any]) -> Any:
        try:
            from pymilvus import DataType, MilvusClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MilvusIngestError(
                "pymilvus is required for PymilvusMilvusClient; install the milvus extra"
            ) from exc

        schema_obj = MilvusClient.create_schema(
            auto_id=bool(schema.get("auto_id", False)),
            description=schema.get("description", ""),
        )
        for field in schema.get("fields", []):
            if not isinstance(field, dict):
                raise MilvusIngestError("Milvus schema field must be a mapping")
            field_name = field.get("name")
            dtype_name = field.get("dtype")
            if not isinstance(field_name, str) or not field_name.strip():
                raise MilvusIngestError("Milvus schema field missing name")
            if not isinstance(dtype_name, str) or not dtype_name.strip():
                raise MilvusIngestError(f"Milvus schema field {field_name} missing dtype")
            datatype = getattr(DataType, dtype_name.upper(), None)
            if datatype is None:
                raise MilvusIngestError(f"Unsupported Milvus dtype: {dtype_name}")

            field_kwargs = {
                key: value
                for key, value in field.items()
                if key not in {"name", "dtype"} and value is not None
            }
            schema_obj.add_field(
                field_name=field_name,
                datatype=datatype,
                **field_kwargs,
            )
        return schema_obj

    @staticmethod
    def _build_pymilvus_index_params(
        schema: dict[str, Any],
        index_params: dict[str, Any],
    ) -> Any:
        try:
            from pymilvus import MilvusClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MilvusIngestError(
                "pymilvus is required for PymilvusMilvusClient; install the milvus extra"
            ) from exc

        vector_field = _find_vector_field_name(schema)
        raw_params = dict(index_params)
        index_type = str(raw_params.pop("index_type", "AUTOINDEX")).upper()
        metric_type = str(raw_params.pop("metric_type", "COSINE")).upper()
        index_name = str(raw_params.pop("index_name", ""))
        params = raw_params.pop("params", None)
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise MilvusIngestError("Milvus index params.params must be a mapping")

        pymilvus_index_params = MilvusClient.prepare_index_params()
        pymilvus_index_params.add_index(
            field_name=vector_field,
            index_type=index_type,
            index_name=index_name,
            metric_type=metric_type,
            params=params,
            **raw_params,
        )
        return pymilvus_index_params


def default_milvus_schema(
    *,
    vector_dim: int,
    primary_key_field: str = DEFAULT_PRIMARY_KEY_FIELD,
    vector_field: str = DEFAULT_VECTOR_FIELD,
) -> dict[str, Any]:
    """Build the default explicit Milvus collection schema."""

    return {
        "description": "Chunk text and embedding vectors for retrieval evaluation.",
        "auto_id": False,
        "fields": [
            {
                "name": primary_key_field,
                "dtype": "VARCHAR",
                "is_primary": True,
                "max_length": 512,
            },
            {"name": "doc_id", "dtype": "VARCHAR", "max_length": 512},
            {"name": "title", "dtype": "VARCHAR", "max_length": 4096, "nullable": True},
            {"name": "text", "dtype": "VARCHAR", "max_length": 65535},
            {"name": "chunk_index", "dtype": "INT64"},
            {"name": "start_offset", "dtype": "INT64", "nullable": True},
            {"name": "end_offset", "dtype": "INT64", "nullable": True},
            {"name": "metadata", "dtype": "JSON"},
            {
                "name": METADATA_KEY_SOURCE_CHUNKED_CORPUS_ARTIFACT_ID,
                "dtype": "VARCHAR",
                "max_length": 1024,
            },
            {
                "name": METADATA_KEY_SOURCE_EMBEDDINGS_ARTIFACT_ID,
                "dtype": "VARCHAR",
                "max_length": 1024,
            },
            {"name": "source_chunk_file", "dtype": "VARCHAR", "max_length": 4096},
            {"name": "source_embedding_file", "dtype": "VARCHAR", "max_length": 4096},
            {"name": "shard_id", "dtype": "VARCHAR", "max_length": 512},
            {"name": vector_field, "dtype": "FLOAT_VECTOR", "dim": vector_dim},
        ],
    }


def stable_schema_sha256(schema: dict[str, Any]) -> str:
    """Compute a stable sha256 for a Milvus schema body."""

    payload = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _find_vector_field_name(schema: dict[str, Any]) -> str:
    for field in schema.get("fields", []):
        if isinstance(field, dict) and str(field.get("dtype", "")).upper() == "FLOAT_VECTOR":
            field_name = field.get("name")
            if isinstance(field_name, str) and field_name.strip():
                return field_name
    raise MilvusIngestError("Milvus schema must contain a FLOAT_VECTOR field")


def _safe_user_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop secret-looking supplemental metadata keys before writing manifests."""

    safe_metadata: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized_key = key.lower()
        if any(part in normalized_key for part in _SENSITIVE_METADATA_KEY_PARTS):
            continue
        safe_metadata[key] = value
    return safe_metadata


def _resolve_vector_dim(
    *,
    embedding_store: ArtifactStore,
    embeddings_artifact_id: str,
    config_vector_dim: int | None,
) -> int:
    manifest = embedding_store.read_manifest(EMBEDDINGS_ARTIFACT_TYPE, embeddings_artifact_id)
    manifest_vector_dim = manifest.metadata.get("embedding_dim")

    if config_vector_dim is not None:
        if isinstance(manifest_vector_dim, int) and manifest_vector_dim != config_vector_dim:
            raise MilvusIngestError(
                "Milvus vector_dim does not match embeddings manifest embedding_dim"
            )
        return config_vector_dim

    if isinstance(manifest_vector_dim, int) and manifest_vector_dim > 0:
        return manifest_vector_dim

    raise MilvusIngestError("Unable to determine Milvus vector_dim")


def chunk_embedding_to_milvus_row(
    chunk: ChunkRecord,
    embedding: EmbeddingRecord,
    *,
    chunked_corpus_artifact_id: str,
    embeddings_artifact_id: str,
    source_chunk_file: str,
    source_embedding_file: str,
    shard_id: str,
    primary_key_field: str = DEFAULT_PRIMARY_KEY_FIELD,
    vector_field: str = DEFAULT_VECTOR_FIELD,
) -> dict[str, Any]:
    """Convert one aligned chunk/embedding pair to a Milvus row."""

    return {
        primary_key_field: chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "metadata": dict(chunk.metadata),
        METADATA_KEY_SOURCE_CHUNKED_CORPUS_ARTIFACT_ID: chunked_corpus_artifact_id,
        METADATA_KEY_SOURCE_EMBEDDINGS_ARTIFACT_ID: embeddings_artifact_id,
        "source_chunk_file": source_chunk_file,
        "source_embedding_file": source_embedding_file,
        "shard_id": shard_id,
        vector_field: list(embedding.vector),
    }


def _raise_for_insert_failures(result: MilvusInsertResult) -> None:
    if not result.failed_items:
        return
    sample = [
        {"primary_key": item.primary_key, "error": item.error}
        for item in result.failed_items[:3]
    ]
    raise MilvusIngestError(
        f"Milvus insert failed for {len(result.failed_items)} item(s): {sample}"
    )


def _validate_shard_alignment(
    chunk_shard: ChunkShard,
    embedding_shard: EmbeddingShard,
) -> None:
    if chunk_shard.shard_id != embedding_shard.shard_id:
        raise MilvusIngestError(
            "Chunk and embedding shard_id mismatch: "
            f"{chunk_shard.shard_id} != {embedding_shard.shard_id}"
        )
    if chunk_shard.path != embedding_shard.source_chunk_file:
        raise MilvusIngestError(
            "Chunk shard path does not match embedding source_chunk_file: "
            f"{chunk_shard.path} != {embedding_shard.source_chunk_file}"
        )
    if chunk_shard.chunk_count != embedding_shard.embedding_count:
        raise MilvusIngestError(
            "Chunk and embedding shard row count mismatch: "
            f"{chunk_shard.chunk_count} != {embedding_shard.embedding_count}"
        )


def run_milvus_ingest(
    chunk_store: ArtifactStore,
    embedding_store: ArtifactStore,
    output_store: ArtifactStore,
    config: MilvusIngestConfig,
    client: MilvusClientProtocol,
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Stream aligned chunk and embedding shards into Milvus and write a manifest."""

    vector_dim = _resolve_vector_dim(
        embedding_store=embedding_store,
        embeddings_artifact_id=config.embeddings_artifact_id,
        config_vector_dim=config.vector_dim,
    )
    schema = config.schema_ or default_milvus_schema(
        vector_dim=vector_dim,
        primary_key_field=config.primary_key_field,
        vector_field=config.vector_field,
    )
    schema_sha256 = stable_schema_sha256(schema)
    index_params = dict(config.index_params)
    if "metric_type" not in index_params:
        index_params["metric_type"] = config.metric_type

    collection_exists = client.collection_exists(config.collection_name)
    if collection_exists and not config.overwrite_existing:
        raise MilvusIngestError(f"Milvus collection already exists: {config.collection_name}")
    if collection_exists and config.overwrite_existing:
        client.drop_collection(config.collection_name)

    client.create_collection(config.collection_name, schema, index_params)
    report_progress(
        progress_reporter,
        stage="milvus_ingest",
        current=0,
        total=None,
        message="Prepared Milvus collection",
        metadata={
            "kind": "collection",
            "collection_name": config.collection_name,
            "overwrite_existing": config.overwrite_existing,
        },
    )

    inserted_count = 0
    shard_metadata: list[dict[str, Any]] = []
    pending_rows: list[MilvusRow] = []

    def flush_batch(*, shard_id: str) -> None:
        nonlocal inserted_count
        if not pending_rows:
            return
        result = client.insert_rows(config.collection_name, pending_rows)
        _raise_for_insert_failures(result)
        if result.inserted_count != len(pending_rows):
            raise MilvusIngestError("Milvus insert result inserted_count does not match row count")
        inserted_count += result.inserted_count
        report_progress(
            progress_reporter,
            stage="milvus_ingest",
            current=inserted_count,
            total=None,
            message="Completed Milvus insert batch",
            metadata={
                "kind": "batch",
                "collection_name": config.collection_name,
                "shard_id": shard_id,
                "batch_size": len(pending_rows),
                "inserted_count": inserted_count,
            },
        )
        pending_rows.clear()

    chunk_shards = iter_chunk_shards(chunk_store, config.chunked_corpus_artifact_id)
    embedding_shards = iter_embedding_shards(embedding_store, config.embeddings_artifact_id)
    sentinel = object()

    for shard_index, pair in enumerate(
        zip_longest(chunk_shards, embedding_shards, fillvalue=sentinel),
        start=1,
    ):
        chunk_shard, embedding_shard = pair
        if chunk_shard is sentinel or embedding_shard is sentinel:
            raise MilvusIngestError("Chunk and embedding shard counts do not match")
        if not isinstance(chunk_shard, ChunkShard) or not isinstance(
            embedding_shard, EmbeddingShard
        ):
            raise MilvusIngestError("Invalid chunk or embedding shard type")

        _validate_shard_alignment(chunk_shard, embedding_shard)
        shard_inserted_count = 0
        for row_index, row_pair in enumerate(
            zip_longest(chunk_shard.chunks, embedding_shard.embeddings, fillvalue=sentinel),
            start=1,
        ):
            chunk, embedding = row_pair
            if chunk is sentinel or embedding is sentinel:
                raise MilvusIngestError(
                    f"Chunk and embedding row counts do not match in shard {chunk_shard.shard_id}"
                )
            if not isinstance(chunk, ChunkRecord) or not isinstance(embedding, EmbeddingRecord):
                raise MilvusIngestError("Invalid chunk or embedding row type")
            if chunk.chunk_id != embedding.chunk_id:
                raise MilvusIngestError(
                    "Chunk and embedding chunk_id mismatch in shard "
                    f"{chunk_shard.shard_id} row {row_index}: "
                    f"{chunk.chunk_id} != {embedding.chunk_id}"
                )
            if chunk.doc_id != embedding.doc_id:
                raise MilvusIngestError(
                    "Chunk and embedding doc_id mismatch in shard "
                    f"{chunk_shard.shard_id} row {row_index}: "
                    f"{chunk.doc_id} != {embedding.doc_id}"
                )
            if len(embedding.vector) != vector_dim:
                raise MilvusIngestError(
                    "Embedding vector dimension mismatch in shard "
                    f"{chunk_shard.shard_id} row {row_index}: "
                    f"expected {vector_dim}, got {len(embedding.vector)}"
                )

            pending_rows.append(
                MilvusRow(
                    primary_key=chunk.chunk_id,
                    row=chunk_embedding_to_milvus_row(
                        chunk,
                        embedding,
                        chunked_corpus_artifact_id=config.chunked_corpus_artifact_id,
                        embeddings_artifact_id=config.embeddings_artifact_id,
                        source_chunk_file=chunk_shard.path,
                        source_embedding_file=embedding_shard.embedding_file,
                        shard_id=chunk_shard.shard_id,
                        primary_key_field=config.primary_key_field,
                        vector_field=config.vector_field,
                    ),
                )
            )
            if len(pending_rows) >= config.batch_size:
                before_flush_count = inserted_count
                flush_batch(shard_id=chunk_shard.shard_id)
                shard_inserted_count += inserted_count - before_flush_count

        before_flush_count = inserted_count
        flush_batch(shard_id=chunk_shard.shard_id)
        shard_inserted_count += inserted_count - before_flush_count
        shard_metadata.append(
            {
                "shard_id": chunk_shard.shard_id,
                "source_chunk_file": chunk_shard.path,
                "source_embedding_file": embedding_shard.embedding_file,
                "chunk_count": chunk_shard.chunk_count,
                "embedding_count": embedding_shard.embedding_count,
                "inserted_count": shard_inserted_count,
                "failed_count": 0,
                "first_chunk_id": chunk_shard.first_chunk_id,
                "last_chunk_id": chunk_shard.last_chunk_id,
            }
        )
        report_progress(
            progress_reporter,
            stage="milvus_ingest",
            current=shard_index,
            total=None,
            message="Completed Milvus shard ingest",
            metadata={
                "kind": "shard",
                "collection_name": config.collection_name,
                "shard_id": chunk_shard.shard_id,
                "source_chunk_file": chunk_shard.path,
                "source_embedding_file": embedding_shard.embedding_file,
                "inserted_count": shard_inserted_count,
            },
        )

    if config.flush:
        client.flush_collection(config.collection_name)
        report_progress(
            progress_reporter,
            stage="milvus_ingest",
            current=inserted_count,
            total=inserted_count,
            message="Flushed Milvus collection",
            metadata={
                "kind": "flush",
                "collection_name": config.collection_name,
                "inserted_count": inserted_count,
            },
        )

    verified_entity_count: int | None = None
    if config.verify_count:
        verified_entity_count = client.count_entities(config.collection_name)
        if verified_entity_count != inserted_count:
            raise MilvusIngestError(
                "Milvus entity count verification failed: "
                f"expected {inserted_count}, got {verified_entity_count}"
            )
        report_progress(
            progress_reporter,
            stage="milvus_ingest",
            current=verified_entity_count,
            total=inserted_count,
            message="Verified Milvus entity count",
            metadata={
                "kind": "verify",
                "collection_name": config.collection_name,
                "verified_entity_count": verified_entity_count,
            },
        )

    manifest_metadata: dict[str, Any] = {}
    manifest_metadata.update(_safe_user_metadata(config.metadata))
    manifest_metadata.update(
        {
            METADATA_KEY_SOURCE_CHUNKED_CORPUS_ARTIFACT_ID: (
                config.chunked_corpus_artifact_id
            ),
            METADATA_KEY_SOURCE_EMBEDDINGS_ARTIFACT_ID: config.embeddings_artifact_id,
            METADATA_KEY_COLLECTION_NAME: config.collection_name,
            "primary_key_field": config.primary_key_field,
            "vector_field": config.vector_field,
            "vector_dim": vector_dim,
            "metric_type": config.metric_type,
            "batch_size": config.batch_size,
            "overwrite_existing": config.overwrite_existing,
            "flush": config.flush,
            "verify_count": config.verify_count,
            "schema_sha256": schema_sha256,
            "schema": schema,
            "index_params": index_params,
            "alignment_key": "chunk_id",
            "alignment_order": "source_chunk_order",
            "inserted_count": inserted_count,
            "failed_count": 0,
            "verified_entity_count": verified_entity_count,
            "shards": shard_metadata,
        }
    )

    manifest = ArtifactManifest(
        artifact_id=config.output_artifact_id,
        artifact_type=MILVUS_COLLECTION_ARTIFACT_TYPE,
        created_at=datetime.now(UTC),
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
        dependencies=[
            ArtifactDependency(
                artifact_type=CHUNKED_CORPUS_ARTIFACT_TYPE,
                artifact_id=config.chunked_corpus_artifact_id,
            ),
            ArtifactDependency(
                artifact_type=EMBEDDINGS_ARTIFACT_TYPE,
                artifact_id=config.embeddings_artifact_id,
            ),
        ],
        metadata=manifest_metadata,
        files=[],
    )
    output_store.write_manifest(
        MILVUS_COLLECTION_ARTIFACT_TYPE,
        config.output_artifact_id,
        manifest,
    )
    output_store.mark_success(
        MILVUS_COLLECTION_ARTIFACT_TYPE,
        config.output_artifact_id,
    )
    return manifest
