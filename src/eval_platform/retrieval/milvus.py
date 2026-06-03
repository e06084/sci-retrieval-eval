"""Milvus retrieval adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from eval_platform.config import MilvusConfig
from eval_platform.defaults import (
    DEFAULT_MILVUS_METRIC_TYPE,
    DEFAULT_MILVUS_PRIMARY_KEY_FIELD,
    DEFAULT_MILVUS_VECTOR_FIELD,
    default_milvus_search_params,
)
from eval_platform.retrieval.schema import RetrievalHit

_DEFAULT_OUTPUT_FIELDS = [
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


class MilvusRetrievalAdapterError(Exception):
    """Raised when Milvus retrieval fails."""


class PymilvusRetrievalClientConfig(BaseModel):
    """Configuration for the lazy-import pymilvus retrieval client."""

    address: str
    username: str | None = None
    password: str | None = None
    db_name: str | None = None
    primary_key_field: str = DEFAULT_MILVUS_PRIMARY_KEY_FIELD
    vector_field: str = DEFAULT_MILVUS_VECTOR_FIELD
    output_fields: list[str] = Field(default_factory=lambda: list(_DEFAULT_OUTPUT_FIELDS))
    metric_type: str = DEFAULT_MILVUS_METRIC_TYPE
    search_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("address", "primary_key_field", "vector_field", "metric_type")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name or 'field'} must not be empty")
        return value

    @field_validator("output_fields")
    @classmethod
    def validate_output_fields(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("output_fields must not be empty")
        for field in value:
            if not field.strip():
                raise ValueError("output_fields must not contain empty fields")
        return value


class PymilvusRetrievalClient:
    """Small lazy-import pymilvus adapter for vector retrieval."""

    def __init__(self, config: PymilvusRetrievalClientConfig, *, client: Any | None = None) -> None:
        self._config = config
        if client is not None:
            self._client = client
            return

        try:
            from pymilvus import MilvusClient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise MilvusRetrievalAdapterError(
                "pymilvus is required for PymilvusRetrievalClient; install the milvus extra"
            ) from exc

        kwargs: dict[str, Any] = {"uri": config.address}
        if config.username is not None:
            kwargs["user"] = config.username
        if config.password is not None:
            kwargs["password"] = config.password
        if config.db_name is not None:
            kwargs["db_name"] = config.db_name
        self._client = MilvusClient(**kwargs)

    def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        result = self._client.search(
            collection_name=collection_name,
            data=[list(vector)],
            anns_field=self._config.vector_field,
            limit=top_k,
            output_fields=self._config.output_fields,
            search_params=default_milvus_search_params(
                metric_type=self._config.metric_type,
            )
            | {"params": dict(self._config.search_params)},
        )
        return [_hit_from_milvus_hit(hit, self._config) for hit in _first_result_set(result)]


def milvus_retrieval_client_from_config(config: MilvusConfig) -> PymilvusRetrievalClient:
    """Create a Milvus retrieval client from platform config."""

    if config.address is None or not config.address.strip():
        raise MilvusRetrievalAdapterError("milvus.address is required")
    return PymilvusRetrievalClient(
        PymilvusRetrievalClientConfig(
            address=config.address,
            username=config.username,
            password=config.password,
            db_name=config.db_name,
        )
    )


def _first_result_set(result: Any) -> list[dict[str, Any]]:
    if not isinstance(result, list) or not result:
        return []
    first = result[0]
    if not isinstance(first, list):
        raise MilvusRetrievalAdapterError("Milvus search response must be a list of hit lists")
    return [hit for hit in first if isinstance(hit, dict)]


def _hit_from_milvus_hit(
    hit: dict[str, Any],
    config: PymilvusRetrievalClientConfig,
) -> RetrievalHit:
    entity = hit.get("entity")
    if not isinstance(entity, dict):
        entity = {}
    chunk_id = str(entity.get(config.primary_key_field) or hit.get("id") or "")
    score_value = hit.get("distance", hit.get("score", 0.0))
    metadata = _entity_metadata(entity, config.vector_field)
    return RetrievalHit(
        chunk_id=chunk_id,
        doc_id=str(entity.get("doc_id") or ""),
        title=entity.get("title"),
        text=str(entity.get("text") or ""),
        score=float(score_value or 0.0),
        recall_source="milvus",
        origin_milvus_score=float(score_value or 0.0),
        metadata=metadata,
    )


def _entity_metadata(entity: dict[str, Any], vector_field: str) -> dict[str, Any]:
    raw_metadata = entity.get("metadata")
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    for field in _METADATA_FIELDS:
        if field in entity:
            metadata[field] = entity[field]
    metadata.pop(vector_field, None)
    metadata.pop("vector", None)
    return metadata
