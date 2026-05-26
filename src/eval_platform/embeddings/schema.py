"""Embedding record schemas."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


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
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


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
