"""Embedding schema, artifact, and runner helpers."""

from eval_platform.embeddings.artifact import (
    EMBEDDINGS_ARTIFACT_TYPE,
    EMBEDDINGS_FILENAME,
    EmbeddingArtifactError,
    read_embeddings_artifact,
    write_embeddings_artifact,
)
from eval_platform.embeddings.client import (
    EmbeddingClient,
    EmbeddingClientError,
    FakeEmbeddingClient,
    HTTPEmbeddingClient,
    HTTPEmbeddingClientConfig,
    http_embedding_client_from_env,
)
from eval_platform.embeddings.jsonl import dump_embeddings_jsonl, load_embeddings_jsonl
from eval_platform.embeddings.runner import EmbeddingRunConfig, EmbeddingRunError, run_embedding
from eval_platform.embeddings.schema import EmbeddedCorpus, EmbeddingProvenance, EmbeddingRecord

__all__ = [
    "EMBEDDINGS_ARTIFACT_TYPE",
    "EMBEDDINGS_FILENAME",
    "EmbeddingArtifactError",
    "EmbeddingClient",
    "EmbeddingClientError",
    "EmbeddingProvenance",
    "EmbeddingRecord",
    "EmbeddedCorpus",
    "EmbeddingRunConfig",
    "EmbeddingRunError",
    "FakeEmbeddingClient",
    "HTTPEmbeddingClient",
    "HTTPEmbeddingClientConfig",
    "dump_embeddings_jsonl",
    "http_embedding_client_from_env",
    "load_embeddings_jsonl",
    "read_embeddings_artifact",
    "run_embedding",
    "write_embeddings_artifact",
]
