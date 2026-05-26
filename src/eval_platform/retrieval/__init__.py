"""Retrieval pipeline layer."""

from eval_platform.retrieval.artifact import (
    RESULTS_DIR,
    RETRIEVAL_RUN_ARTIFACT_TYPE,
    RetrievalArtifactError,
    read_retrieval_run_artifact,
    write_retrieval_run_artifact,
)
from eval_platform.retrieval.clients import (
    ElasticsearchRetrievalClient,
    MilvusRetrievalClient,
    RerankClient,
    RewriteClient,
)
from eval_platform.retrieval.elasticsearch import (
    ElasticsearchRetrievalAdapterError,
    HTTPElasticsearchRetrievalClient,
    HTTPElasticsearchRetrievalClientConfig,
    elasticsearch_retrieval_client_from_config,
)
from eval_platform.retrieval.fusion import dedupe_by_chunk_id, dedupe_sequential, rrf_fuse
from eval_platform.retrieval.jsonl import (
    dump_retrieval_results_jsonl,
    load_retrieval_results_jsonl,
)
from eval_platform.retrieval.milvus import (
    MilvusRetrievalAdapterError,
    PymilvusRetrievalClient,
    PymilvusRetrievalClientConfig,
    milvus_retrieval_client_from_config,
)
from eval_platform.retrieval.runner import RetrievalRunConfig, RetrievalRunError, run_retrieval
from eval_platform.retrieval.schema import (
    RetrievalHit,
    RetrievalMode,
    RetrievalQueryResult,
)

__all__ = [
    "RETRIEVAL_RUN_ARTIFACT_TYPE",
    "RESULTS_DIR",
    "ElasticsearchRetrievalClient",
    "ElasticsearchRetrievalAdapterError",
    "HTTPElasticsearchRetrievalClient",
    "HTTPElasticsearchRetrievalClientConfig",
    "MilvusRetrievalClient",
    "MilvusRetrievalAdapterError",
    "PymilvusRetrievalClient",
    "PymilvusRetrievalClientConfig",
    "RerankClient",
    "RetrievalArtifactError",
    "RetrievalHit",
    "RetrievalMode",
    "RetrievalQueryResult",
    "RetrievalRunConfig",
    "RetrievalRunError",
    "RewriteClient",
    "dedupe_by_chunk_id",
    "dedupe_sequential",
    "dump_retrieval_results_jsonl",
    "elasticsearch_retrieval_client_from_config",
    "load_retrieval_results_jsonl",
    "milvus_retrieval_client_from_config",
    "read_retrieval_run_artifact",
    "rrf_fuse",
    "run_retrieval",
    "write_retrieval_run_artifact",
]
