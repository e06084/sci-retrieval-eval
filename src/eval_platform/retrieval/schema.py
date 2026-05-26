"""Retrieval run record schemas."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class RetrievalHit(BaseModel):
    """One retrieval hit for a query."""

    chunk_id: str
    doc_id: str = ""
    title: str | None = None
    text: str = ""
    score: float = 0.0
    recall_source: str = ""
    rank: int | None = Field(default=None, gt=0)
    origin_es_score: float | None = None
    origin_milvus_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunk_id")
    @classmethod
    def validate_chunk_id(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @field_validator("score", "origin_es_score", "origin_milvus_score")
    @classmethod
    def validate_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("score values must be finite")
        return value


class RetrievalQueryResult(BaseModel):
    """Retrieval results for one normalized query."""

    query_id: str
    query_text: str
    hits: list[RetrievalHit] = Field(default_factory=list)
    trace: dict[str, Any] | None = None
    error: str | None = None

    @field_validator("query_id", "query_text")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @model_validator(mode="after")
    def validate_hit_ranks(self) -> RetrievalQueryResult:
        for expected_rank, hit in enumerate(self.hits, start=1):
            if hit.rank != expected_rank:
                raise ValueError("hits must have stable ranks starting at 1")
        return self


RetrievalMode = Literal["es", "milvus", "hybrid"]
