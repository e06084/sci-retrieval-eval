"""Elasticsearch retrieval adapter."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from eval_platform.config import ElasticsearchConfig
from eval_platform.retrieval.schema import RetrievalHit

_SOURCE_FIELDS = [
    "chunk_id",
    "doc_id",
    "title",
    "text",
    "chunk_index",
    "start_offset",
    "end_offset",
    "metadata",
]
_METADATA_FIELDS = ("chunk_index", "start_offset", "end_offset")


class ElasticsearchRetrievalAdapterError(Exception):
    """Raised when Elasticsearch retrieval fails."""


class HTTPElasticsearchRetrievalClientConfig(BaseModel):
    """Configuration for the lightweight standard-library HTTP ES retrieval client."""

    base_url: str
    username: str | None = None
    password: str | None = None
    timeout_seconds: float = Field(default=60.0, gt=0)
    text_fields: list[str] = Field(default_factory=lambda: ["title^1.5", "text"])

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name or 'field'} must not be empty")
        return value.rstrip("/")

    @field_validator("text_fields")
    @classmethod
    def validate_text_fields(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("text_fields must not be empty")
        for field in value:
            if not field.strip():
                raise ValueError("text_fields must not contain empty fields")
        return value


class HTTPElasticsearchRetrievalClient:
    """Small Elasticsearch HTTP adapter for retrieval."""

    def __init__(
        self,
        config: HTTPElasticsearchRetrievalClientConfig,
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

    def search_bm25(self, index_name: str, query: str, top_k: int) -> list[RetrievalHit]:
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": self._config.text_fields,
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"chunk_id": {"order": "asc"}},
            ],
            "_source": _SOURCE_FIELDS,
        }
        payload = self._request_json("POST", f"/{index_name}/_search", body, "search")
        hits = _extract_hits(payload)
        return [_hit_from_es_hit(hit, recall_source="es") for hit in hits]

    def enrich_by_chunk_ids(
        self,
        index_name: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        if not hits:
            return []
        body = {
            "docs": [
                {"_id": hit.chunk_id, "_source": _SOURCE_FIELDS}
                for hit in hits
            ]
        }
        payload = self._request_json("POST", f"/{index_name}/_mget", body, "mget")
        docs = payload.get("docs")
        if not isinstance(docs, list):
            raise ElasticsearchRetrievalAdapterError("Elasticsearch mget response missing docs")

        docs_by_id = {
            str(doc.get("_id")): doc
            for doc in docs
            if isinstance(doc, dict) and bool(doc.get("found", True))
        }
        enriched: list[RetrievalHit] = []
        for hit in hits:
            doc = docs_by_id.get(hit.chunk_id)
            if doc is None:
                metadata = dict(hit.metadata)
                metadata["enrich_missing"] = True
                enriched.append(hit.model_copy(update={"metadata": metadata}))
                continue
            source = doc.get("_source") if isinstance(doc, dict) else None
            if not isinstance(source, dict):
                metadata = dict(hit.metadata)
                metadata["enrich_missing"] = True
                enriched.append(hit.model_copy(update={"metadata": metadata}))
                continue
            metadata = _source_metadata(source)
            enriched.append(
                hit.model_copy(
                    update={
                        "doc_id": str(source.get("doc_id") or hit.doc_id),
                        "title": source.get("title"),
                        "text": str(source.get("text") or hit.text),
                        "metadata": {**metadata, **hit.metadata},
                    }
                )
            )
        return enriched

    def _request_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any],
        operation: str,
    ) -> dict[str, Any]:
        status, payload = self._request(method, path, body=body)
        if not 200 <= status < 300:
            raise ElasticsearchRetrievalAdapterError(
                f"Elasticsearch {operation} failed with status {status}"
            )
        try:
            result = json.loads(payload.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ElasticsearchRetrievalAdapterError(
                f"Elasticsearch {operation} response is not valid JSON"
            ) from exc
        if not isinstance(result, dict):
            raise ElasticsearchRetrievalAdapterError(
                f"Elasticsearch {operation} response must be an object"
            )
        return result

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
    ) -> tuple[int, bytes]:
        data = None
        headers: dict[str, str] = {}
        if body is not None:
            data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
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


def elasticsearch_retrieval_client_from_config(
    config: ElasticsearchConfig,
) -> HTTPElasticsearchRetrievalClient:
    """Create an Elasticsearch retrieval client from platform config."""

    if config.url is None or not config.url.strip():
        raise ElasticsearchRetrievalAdapterError("elasticsearch.url is required")
    return HTTPElasticsearchRetrievalClient(
        HTTPElasticsearchRetrievalClientConfig(
            base_url=config.url,
            username=config.username,
            password=config.password,
        )
    )


def _extract_hits(payload: dict[str, Any]) -> list[dict[str, Any]]:
    hits_obj = payload.get("hits")
    if not isinstance(hits_obj, dict):
        raise ElasticsearchRetrievalAdapterError("Elasticsearch search response missing hits")
    hits = hits_obj.get("hits")
    if not isinstance(hits, list):
        raise ElasticsearchRetrievalAdapterError("Elasticsearch search response missing hits.hits")
    return [hit for hit in hits if isinstance(hit, dict)]


def _hit_from_es_hit(hit: dict[str, Any], *, recall_source: str) -> RetrievalHit:
    source = hit.get("_source")
    if not isinstance(source, dict):
        source = {}
    chunk_id = str(source.get("chunk_id") or hit.get("_id") or "")
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=str(source.get("doc_id") or ""),
        title=source.get("title"),
        text=str(source.get("text") or ""),
        score=float(hit.get("_score") or 0.0),
        recall_source=recall_source,
        origin_es_score=float(hit.get("_score") or 0.0),
        metadata=_source_metadata(source),
    )


def _source_metadata(source: dict[str, Any]) -> dict[str, Any]:
    raw_metadata = source.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    for field in _METADATA_FIELDS:
        if field in source:
            metadata[field] = source[field]
    return metadata
