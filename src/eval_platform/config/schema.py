"""Strongly-typed platform configuration schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EndpointConfig(StrictConfigModel):
    url: str | None = None
    api_key: str | None = None


class S3Config(StrictConfigModel):
    endpoint: str | None = None
    bucket: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    prefix: str | None = None


class ElasticsearchConfig(StrictConfigModel):
    url: str | None = None
    username: str | None = None
    password: str | None = None


class MilvusConfig(StrictConfigModel):
    address: str | None = None
    username: str | None = None
    password: str | None = None
    index_type: str | None = None
    db_name: str | None = None


class EmbeddingConfig(StrictConfigModel):
    model: str | None = None
    dim: int | None = None
    batch_size: int | None = None
    max_concurrency: int | None = None
    per_endpoint_concurrency: int | None = None
    timeout_sec: float | None = None
    max_retries: int | None = None
    endpoints: list[EndpointConfig] = Field(default_factory=list)


class RerankConfig(StrictConfigModel):
    model: str | None = None
    timeout_sec: float | None = None
    max_retries: int | None = None
    endpoints: list[EndpointConfig] = Field(default_factory=list)


class RewriteConfig(StrictConfigModel):
    enabled: bool | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout_sec: float | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class SearchRuntimeConfig(StrictConfigModel):
    go_api_url: str | None = None
    request_timeout_sec: float | None = None
    rewrite: RewriteConfig | None = None
    search: dict[str, Any] = Field(default_factory=dict)
    es_index: str | None = None
    milvus_collection: str | None = None


class RawSourceConfig(StrictConfigModel):
    uri: str | None = None
    source_type: str | None = None
    dataset_revision: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkingConfig(StrictConfigModel):
    repo_path: str | None = None
    repo_remote: str | None = None
    commit_sha: str | None = None
    chunker_name: str | None = None
    chunk_params: dict[str, Any] = Field(default_factory=dict)


class PlatformConfig(StrictConfigModel):
    s3: S3Config = Field(default_factory=S3Config)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    rerank: RerankConfig | None = None
    elasticsearch: ElasticsearchConfig | None = None
    milvus: MilvusConfig | None = None
    search_runtime: SearchRuntimeConfig | None = None
    raw_sources: dict[str, RawSourceConfig] = Field(default_factory=dict)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
