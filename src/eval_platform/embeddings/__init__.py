"""Embedding schema, artifact, and runner helpers."""

from eval_platform.embeddings.artifact import (
    EMBEDDINGS_ARTIFACT_TYPE,
    EMBEDDINGS_FILENAME,
    EmbeddingArtifactError,
    EmbeddingShard,
    EmbeddingShardDescriptor,
    iter_embedding_shards,
    read_embeddings_artifact,
    write_embedding_shards_artifact,
    write_embeddings_artifact,
)
from eval_platform.embeddings.client import (
    EmbeddingClient,
    EmbeddingClientError,
    EmbeddingConsistencyTolerance,
    FakeEmbeddingClient,
    HTTPEmbeddingClient,
    HTTPEmbeddingClientConfig,
    MultiEndpointEmbeddingConfig,
    http_embedding_client_from_env,
    run_embedding_consistency_check,
)
from eval_platform.embeddings.jsonl import (
    VECTOR_DTYPE,
    VECTOR_ENCODING,
    dump_embeddings_jsonl,
    load_embeddings_jsonl,
)
from eval_platform.embeddings.runner import EmbeddingRunConfig, EmbeddingRunError, run_embedding
from eval_platform.embeddings.schema import (
    EmbeddedCorpus,
    EmbeddingConsistencyCheckResult,
    EmbeddingProvenance,
    EmbeddingRecord,
)

__all__ = [
    "EMBEDDINGS_ARTIFACT_TYPE",
    "EMBEDDINGS_FILENAME",
    "EmbeddingArtifactError",
    "EmbeddingClient",
    "EmbeddingClientError",
    "EmbeddingConsistencyCheckResult",
    "EmbeddingConsistencyTolerance",
    "EmbeddingProvenance",
    "EmbeddingRecord",
    "EmbeddingShard",
    "EmbeddingShardDescriptor",
    "EmbeddedCorpus",
    "EmbeddingRunConfig",
    "EmbeddingRunError",
    "FakeEmbeddingClient",
    "HTTPEmbeddingClient",
    "HTTPEmbeddingClientConfig",
    "MultiEndpointEmbeddingConfig",
    "VECTOR_DTYPE",
    "VECTOR_ENCODING",
    "dump_embeddings_jsonl",
    "http_embedding_client_from_env",
    "load_embeddings_jsonl",
    "iter_embedding_shards",
    "read_embeddings_artifact",
    "run_embedding_consistency_check",
    "run_embedding",
    "write_embedding_shards_artifact",
    "write_embeddings_artifact",
]
