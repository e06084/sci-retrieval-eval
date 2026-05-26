"""Elasticsearch ingest helpers for chunked corpus artifacts."""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest, ArtifactStore
from eval_platform.chunking import CHUNKED_CORPUS_ARTIFACT_TYPE, ChunkRecord, iter_chunk_shards
from eval_platform.chunking.progress import ProgressReporter, report_progress

ELASTICSEARCH_INDEX_ARTIFACT_TYPE = "elasticsearch_index"
DOCUMENT_ID_FIELD = "chunk_id"
_SENSITIVE_METADATA_KEY_PARTS = (
    "access_key",
    "api_key",
    "password",
    "secret",
    "token",
)

DEFAULT_ELASTICSEARCH_MAPPING: dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "title": {"type": "text"},
            "text": {"type": "text"},
            "chunk_index": {"type": "integer"},
            "start_offset": {"type": "integer"},
            "end_offset": {"type": "integer"},
            "source_chunked_corpus_artifact_id": {"type": "keyword"},
            "source_chunk_file": {"type": "keyword"},
            "shard_id": {"type": "keyword"},
            "metadata": {"type": "flattened"},
        },
    },
}


class ElasticsearchIngestError(Exception):
    """Raised when Elasticsearch ingest fails."""


class ElasticsearchBulkFailure(BaseModel):
    """One failed bulk item summary."""

    document_id: str
    status: int | None = None
    error: str | None = None


class ElasticsearchBulkAction(BaseModel):
    """One bulk index action."""

    document_id: str
    document: dict[str, Any]


class ElasticsearchBulkResult(BaseModel):
    """Result returned by Elasticsearch bulk indexing."""

    indexed_count: int
    failed_items: list[ElasticsearchBulkFailure] = Field(default_factory=list)


class ElasticsearchClientProtocol(Protocol):
    """Minimal Elasticsearch client protocol used by the ingest runner."""

    def index_exists(self, index_name: str) -> bool:
        """Return whether an index exists."""
        ...

    def create_index(self, index_name: str, body: dict[str, Any]) -> None:
        """Create an index with an explicit mapping/settings body."""
        ...

    def delete_index(self, index_name: str) -> None:
        """Delete an existing index."""
        ...

    def bulk_index(
        self,
        index_name: str,
        actions: Sequence[ElasticsearchBulkAction],
    ) -> ElasticsearchBulkResult:
        """Index one ordered bulk batch."""
        ...

    def refresh_index(self, index_name: str) -> None:
        """Refresh an index."""
        ...

    def count_documents(self, index_name: str) -> int:
        """Return the number of indexed documents."""
        ...


class ElasticsearchIngestConfig(BaseModel):
    """Configuration for chunked_corpus to Elasticsearch ingest."""

    source_artifact_id: str
    output_artifact_id: str
    index_name: str
    bulk_size: int = Field(default=500, gt=0)
    overwrite_existing: bool = False
    refresh: bool = True
    verify_count: bool = True
    mapping: dict[str, Any] | None = None
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_artifact_id", "output_artifact_id", "index_name")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name or 'field'} must not be empty")
        return value


class HTTPElasticsearchClientConfig(BaseModel):
    """Configuration for the lightweight standard-library HTTP ES client."""

    base_url: str
    username: str | None = None
    password: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name or 'field'} must not be empty")
        return value.rstrip("/")


class HTTPElasticsearchClient:
    """Small Elasticsearch HTTP client covering only ingest runner needs."""

    def __init__(
        self,
        config: HTTPElasticsearchClientConfig,
        *,
        transport: Callable[[urllib.request.Request, float], tuple[int, bytes]] | None = None,
    ) -> None:
        self._config = config
        self._transport = transport or self._default_transport

    @staticmethod
    def _default_transport(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | str | None = None,
        content_type: str = "application/json",
    ) -> tuple[int, bytes]:
        data: bytes | None = None
        if isinstance(body, dict):
            data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            data = body.encode("utf-8")

        headers: dict[str, str] = {}
        if data is not None:
            headers["Content-Type"] = content_type
        if self._config.username is not None and self._config.password is not None:
            token = f"{self._config.username}:{self._config.password}".encode()
            headers["Authorization"] = f"Basic {base64.b64encode(token).decode('ascii')}"

        request = urllib.request.Request(
            f"{self._config.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        return self._transport(request, self._config.timeout_seconds)

    @staticmethod
    def _require_success(status: int, operation: str) -> None:
        if not 200 <= status < 300:
            raise ElasticsearchIngestError(f"Elasticsearch {operation} failed with status {status}")

    def index_exists(self, index_name: str) -> bool:
        status, _ = self._request("HEAD", f"/{index_name}")
        if status == 404:
            return False
        self._require_success(status, "index exists check")
        return True

    def create_index(self, index_name: str, body: dict[str, Any]) -> None:
        status, _ = self._request("PUT", f"/{index_name}", body=body)
        self._require_success(status, "create index")

    def delete_index(self, index_name: str) -> None:
        status, _ = self._request("DELETE", f"/{index_name}")
        if status == 404:
            return
        self._require_success(status, "delete index")

    def bulk_index(
        self,
        index_name: str,
        actions: Sequence[ElasticsearchBulkAction],
    ) -> ElasticsearchBulkResult:
        lines: list[str] = []
        for action in actions:
            lines.append(json.dumps({"index": {"_id": action.document_id}}, separators=(",", ":")))
            lines.append(json.dumps(action.document, separators=(",", ":"), ensure_ascii=False))
        status, payload = self._request(
            "POST",
            f"/{index_name}/_bulk",
            body="\n".join(lines) + "\n",
            content_type="application/x-ndjson",
        )
        self._require_success(status, "bulk index")
        try:
            result = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ElasticsearchIngestError("Elasticsearch bulk response is not valid JSON") from exc

        failed_items: list[ElasticsearchBulkFailure] = []
        for item, action in zip(result.get("items", []), actions, strict=False):
            index_result = item.get("index", {}) if isinstance(item, dict) else {}
            item_status = index_result.get("status")
            if isinstance(item_status, int) and item_status >= 300:
                failed_items.append(
                    ElasticsearchBulkFailure(
                        document_id=action.document_id,
                        status=item_status,
                        error=str(index_result.get("error")),
                    )
                )
        return ElasticsearchBulkResult(
            indexed_count=len(actions) - len(failed_items),
            failed_items=failed_items,
        )

    def refresh_index(self, index_name: str) -> None:
        status, _ = self._request("POST", f"/{index_name}/_refresh")
        self._require_success(status, "refresh index")

    def count_documents(self, index_name: str) -> int:
        status, payload = self._request("GET", f"/{index_name}/_count")
        self._require_success(status, "count documents")
        try:
            result = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ElasticsearchIngestError(
                "Elasticsearch count response is not valid JSON"
            ) from exc
        count = result.get("count")
        if not isinstance(count, int):
            raise ElasticsearchIngestError("Elasticsearch count response missing integer count")
        return count


def stable_mapping_sha256(mapping: dict[str, Any]) -> str:
    """Compute a stable sha256 for an ES mapping/settings body."""

    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_user_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop secret-looking supplemental metadata keys before writing manifests."""

    safe_metadata: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized_key = key.lower()
        if any(part in normalized_key for part in _SENSITIVE_METADATA_KEY_PARTS):
            continue
        safe_metadata[key] = value
    return safe_metadata


def chunk_to_elasticsearch_document(
    chunk: ChunkRecord,
    *,
    source_artifact_id: str,
    source_chunk_file: str,
    shard_id: str,
) -> dict[str, Any]:
    """Convert one chunk to the stable ES document shape."""

    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "metadata": dict(chunk.metadata),
        "source_chunked_corpus_artifact_id": source_artifact_id,
        "source_chunk_file": source_chunk_file,
        "shard_id": shard_id,
    }


def _raise_for_bulk_failures(result: ElasticsearchBulkResult) -> None:
    if not result.failed_items:
        return
    sample = [
        {
            "document_id": item.document_id,
            "status": item.status,
            "error": item.error,
        }
        for item in result.failed_items[:3]
    ]
    raise ElasticsearchIngestError(
        f"Elasticsearch bulk index failed for {len(result.failed_items)} item(s): {sample}"
    )


def run_elasticsearch_ingest(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: ElasticsearchIngestConfig,
    client: ElasticsearchClientProtocol,
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Stream chunked_corpus shards into an Elasticsearch index and write a manifest."""

    mapping = config.mapping or DEFAULT_ELASTICSEARCH_MAPPING
    mapping_sha256 = stable_mapping_sha256(mapping)
    index_exists = client.index_exists(config.index_name)

    if index_exists and not config.overwrite_existing:
        raise ElasticsearchIngestError(
            f"Elasticsearch index already exists: {config.index_name}"
        )
    if index_exists and config.overwrite_existing:
        client.delete_index(config.index_name)

    client.create_index(config.index_name, mapping)
    report_progress(
        progress_reporter,
        stage="elasticsearch_ingest",
        current=0,
        total=None,
        message="Prepared Elasticsearch index",
        metadata={
            "kind": "index",
            "index_name": config.index_name,
            "overwrite_existing": config.overwrite_existing,
        },
    )

    indexed_count = 0
    shard_metadata: list[dict[str, Any]] = []
    pending_actions: list[ElasticsearchBulkAction] = []

    def flush_bulk(*, shard_id: str) -> None:
        nonlocal indexed_count
        if not pending_actions:
            return
        result = client.bulk_index(config.index_name, pending_actions)
        _raise_for_bulk_failures(result)
        if result.indexed_count != len(pending_actions):
            raise ElasticsearchIngestError(
                "Elasticsearch bulk result indexed_count does not match action count"
            )
        indexed_count += result.indexed_count
        report_progress(
            progress_reporter,
            stage="elasticsearch_ingest",
            current=indexed_count,
            total=None,
            message="Completed Elasticsearch bulk",
            metadata={
                "kind": "bulk",
                "index_name": config.index_name,
                "shard_id": shard_id,
                "bulk_size": len(pending_actions),
                "indexed_count": indexed_count,
            },
        )
        pending_actions.clear()

    for shard_index, chunk_shard in enumerate(
        iter_chunk_shards(source_store, config.source_artifact_id),
        start=1,
    ):
        shard_indexed_count = 0
        for chunk in chunk_shard.chunks:
            pending_actions.append(
                ElasticsearchBulkAction(
                    document_id=chunk.chunk_id,
                    document=chunk_to_elasticsearch_document(
                        chunk,
                        source_artifact_id=config.source_artifact_id,
                        source_chunk_file=chunk_shard.path,
                        shard_id=chunk_shard.shard_id,
                    ),
                )
            )
            if len(pending_actions) >= config.bulk_size:
                before_flush_count = indexed_count
                flush_bulk(shard_id=chunk_shard.shard_id)
                shard_indexed_count += indexed_count - before_flush_count
        before_flush_count = indexed_count
        flush_bulk(shard_id=chunk_shard.shard_id)
        shard_indexed_count += indexed_count - before_flush_count
        shard_metadata.append(
            {
                "shard_id": chunk_shard.shard_id,
                "source_chunk_file": chunk_shard.path,
                "chunk_count": chunk_shard.chunk_count,
                "indexed_count": shard_indexed_count,
                "failed_count": 0,
                "first_chunk_id": chunk_shard.first_chunk_id,
                "last_chunk_id": chunk_shard.last_chunk_id,
            }
        )
        report_progress(
            progress_reporter,
            stage="elasticsearch_ingest",
            current=shard_index,
            total=None,
            message="Completed Elasticsearch shard ingest",
            metadata={
                "kind": "shard",
                "index_name": config.index_name,
                "shard_id": chunk_shard.shard_id,
                "source_chunk_file": chunk_shard.path,
                "indexed_count": shard_indexed_count,
            },
        )

    if config.refresh:
        client.refresh_index(config.index_name)

    verified_document_count: int | None = None
    if config.verify_count:
        verified_document_count = client.count_documents(config.index_name)
        if verified_document_count != indexed_count:
            raise ElasticsearchIngestError(
                "Elasticsearch document count verification failed: "
                f"expected {indexed_count}, got {verified_document_count}"
            )
        report_progress(
            progress_reporter,
            stage="elasticsearch_ingest",
            current=verified_document_count,
            total=indexed_count,
            message="Verified Elasticsearch document count",
            metadata={
                "kind": "verify",
                "index_name": config.index_name,
                "verified_document_count": verified_document_count,
            },
        )

    manifest_metadata: dict[str, Any] = {}
    manifest_metadata.update(_safe_user_metadata(config.metadata))
    manifest_metadata.update(
        {
            "source_chunked_corpus_artifact_id": config.source_artifact_id,
            "index_name": config.index_name,
            "document_id_field": DOCUMENT_ID_FIELD,
            "bulk_size": config.bulk_size,
            "overwrite_existing": config.overwrite_existing,
            "refresh": config.refresh,
            "verify_count": config.verify_count,
            "mapping_sha256": mapping_sha256,
            "mapping": mapping,
            "indexed_count": indexed_count,
            "failed_count": 0,
            "verified_document_count": verified_document_count,
            "shards": shard_metadata,
        }
    )

    manifest = ArtifactManifest(
        artifact_id=config.output_artifact_id,
        artifact_type=ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
        created_at=datetime.now(UTC),
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
        dependencies=[
            ArtifactDependency(
                artifact_type=CHUNKED_CORPUS_ARTIFACT_TYPE,
                artifact_id=config.source_artifact_id,
            )
        ],
        metadata=manifest_metadata,
        files=[],
    )
    output_store.write_manifest(
        ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
        config.output_artifact_id,
        manifest,
    )
    output_store.mark_success(
        ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
        config.output_artifact_id,
    )
    return manifest
