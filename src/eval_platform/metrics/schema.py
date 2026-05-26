"""Metrics run schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class RankedDoc(BaseModel):
    """One doc-level result projected from chunk-level retrieval hits."""

    rank: int = Field(gt=0)
    doc_id: str
    score: float
    source_chunk_id: str
    source_chunk_rank: int = Field(gt=0)
    source_chunk_score: float

    @field_validator("doc_id", "source_chunk_id")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


class ProjectionStats(BaseModel):
    """Stats collected while projecting chunk hits to doc results."""

    input_hit_count: int = Field(ge=0)
    ranked_doc_count: int = Field(ge=0)
    missing_doc_id_hit_count: int = Field(ge=0)
    duplicate_doc_hit_count: int = Field(ge=0)


class QueryMetricsRecord(BaseModel):
    """Metrics and projected docs for one query."""

    query_id: str
    query_text: str = ""
    retrieval_error: str | None = None
    ranked_docs: list[RankedDoc] = Field(default_factory=list)
    relevant_docs: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    projection_stats: ProjectionStats

    @field_validator("query_id")
    @classmethod
    def validate_query_id(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


class MetricsRunData(BaseModel):
    """In-memory representation of a metrics_run artifact."""

    aggregate: dict[str, float]
    k_values: list[int]
    main_score: float
    main_score_metric: str
    query_metrics: list[QueryMetricsRecord]
    metadata: dict[str, Any] = Field(default_factory=dict)
