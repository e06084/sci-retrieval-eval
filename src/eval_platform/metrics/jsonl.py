"""JSONL helpers for metrics run query records."""

from __future__ import annotations

from collections.abc import Iterable

from eval_platform.datasets.jsonl import dump_jsonl, load_jsonl
from eval_platform.metrics.schema import QueryMetricsRecord


def dump_query_metrics_jsonl(records: Iterable[QueryMetricsRecord]) -> str:
    """Serialize query metrics records to JSONL."""

    return dump_jsonl(records)


def load_query_metrics_jsonl(text: str) -> list[QueryMetricsRecord]:
    """Load query metrics records from JSONL."""

    return load_jsonl(text, QueryMetricsRecord)
