"""Metrics and reporting layer."""

from eval_platform.metrics.artifact import (
    METRICS_FILENAME,
    METRICS_RUN_ARTIFACT_TYPE,
    QUERY_METRICS_DIR,
    MetricsArtifactError,
    read_metrics_run_artifact,
    write_metrics_run_artifact,
)
from eval_platform.metrics.ir import aggregate_query_metrics, compute_query_metrics
from eval_platform.metrics.projection import project_retrieval_result_to_docs
from eval_platform.metrics.runner import MetricsRunConfig, run_metrics
from eval_platform.metrics.schema import (
    MetricsRunData,
    ProjectionStats,
    QueryMetricsRecord,
    RankedDoc,
)

__all__ = [
    "METRICS_FILENAME",
    "METRICS_RUN_ARTIFACT_TYPE",
    "QUERY_METRICS_DIR",
    "MetricsArtifactError",
    "MetricsRunConfig",
    "MetricsRunData",
    "ProjectionStats",
    "QueryMetricsRecord",
    "RankedDoc",
    "aggregate_query_metrics",
    "compute_query_metrics",
    "project_retrieval_result_to_docs",
    "read_metrics_run_artifact",
    "run_metrics",
    "write_metrics_run_artifact",
]
