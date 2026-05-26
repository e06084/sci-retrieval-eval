"""Embedding record schemas."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class EmbeddingProvenance(BaseModel):
    """Provenance metadata for embedding generation."""

    model_name: str
    provider: str | None = None
    api_version: str | None = None
    embedding_dim: int = Field(gt=0)
    normalized: bool | None = None
    endpoint_id: str | None = None
    endpoint_ids: list[str] = Field(default_factory=list)
    consistency_check: EmbeddingConsistencyCheckResult | None = None
    runtime_parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("endpoint_id")
    @classmethod
    def validate_endpoint_id(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("endpoint_ids")
    @classmethod
    def validate_endpoint_ids(cls, value: list[str]) -> list[str]:
        return [_non_empty_string(item, "endpoint_ids") for item in value]


class EmbeddingConsistencyCheckResult(BaseModel):
    """Result of a multi-endpoint embedding consistency pre-check."""

    input_text: str
    endpoint_ids: list[str]
    passed: bool
    failure_reason: str | None = None
    max_abs_diff: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input_text")
    @classmethod
    def validate_input_text(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("endpoint_ids")
    @classmethod
    def validate_result_endpoint_ids(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("endpoint_ids must not be empty")
        return [_non_empty_string(item, "endpoint_ids") for item in value]

    @model_validator(mode="after")
    def validate_semantics(self) -> EmbeddingConsistencyCheckResult:
        if self.passed:
            if self.failure_reason is not None:
                raise ValueError("failure_reason must be omitted when passed is True")
        else:
            if self.failure_reason is None or not self.failure_reason.strip():
                raise ValueError("failure_reason must be provided when passed is False")
            self.failure_reason = self.failure_reason.strip()
        return self


class EmbeddingRecord(BaseModel):
    """A single embedding aligned to a chunk/document pair."""

    chunk_id: str
    doc_id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id", "doc_id")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("vector must not be empty")
        for item in value:
            if not math.isfinite(item):
                raise ValueError("vector values must be finite numbers")
        return value


class EmbeddedCorpus(BaseModel):
    """In-memory container for an embedded corpus."""

    embeddings: list[EmbeddingRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
