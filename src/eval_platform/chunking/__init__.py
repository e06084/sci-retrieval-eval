"""Chunking pipeline helpers."""

from eval_platform.chunking.artifact import (
    CHUNKED_CORPUS_ARTIFACT_TYPE,
    CHUNKS_FILENAME,
    ChunkShard,
    ChunkShardDescriptor,
    build_chunk_shards,
    iter_chunk_shards,
    read_chunked_corpus_artifact,
    write_chunked_corpus_artifact,
)
from eval_platform.chunking.external_adapter import (
    ExternalChunkerAdapterError,
    PythonCallableChunkerConfig,
    PythonCallableExternalChunker,
    run_version_pinned_external_chunking,
)
from eval_platform.chunking.external_repo import (
    ExternalChunkerRepoError,
    ExternalChunkerRepoMismatchError,
    ExternalChunkerRepoSpec,
    verify_external_chunker_repo,
)
from eval_platform.chunking.git import (
    GitRepoDirtyError,
    GitRepoError,
    GitRepoState,
    ensure_git_repo_clean,
    inspect_git_repo,
)
from eval_platform.chunking.jsonl import dump_chunks_jsonl, load_chunks_jsonl
from eval_platform.chunking.progress import ProgressEvent, ProgressReporter
from eval_platform.chunking.runner import ChunkingRunConfig, ExternalChunker, run_chunking
from eval_platform.chunking.schema import ChunkedCorpus, ChunkerProvenance, ChunkRecord
from eval_platform.chunking.sciverse_adapter import (
    SciverseAdapterError,
    SciverseAdminIngestChunkerConfig,
    SciverseAdminIngestExternalChunker,
    run_version_pinned_sciverse_chunking,
)

__all__ = [
    "CHUNKED_CORPUS_ARTIFACT_TYPE",
    "CHUNKS_FILENAME",
    "ChunkShard",
    "ChunkShardDescriptor",
    "ChunkRecord",
    "ChunkedCorpus",
    "ChunkerProvenance",
    "ChunkingRunConfig",
    "ExternalChunkerAdapterError",
    "ExternalChunker",
    "ExternalChunkerRepoError",
    "ExternalChunkerRepoMismatchError",
    "ExternalChunkerRepoSpec",
    "GitRepoDirtyError",
    "GitRepoError",
    "GitRepoState",
    "PythonCallableChunkerConfig",
    "PythonCallableExternalChunker",
    "ProgressEvent",
    "ProgressReporter",
    "SciverseAdapterError",
    "SciverseAdminIngestChunkerConfig",
    "SciverseAdminIngestExternalChunker",
    "build_chunk_shards",
    "dump_chunks_jsonl",
    "ensure_git_repo_clean",
    "inspect_git_repo",
    "iter_chunk_shards",
    "load_chunks_jsonl",
    "read_chunked_corpus_artifact",
    "run_chunking",
    "run_version_pinned_external_chunking",
    "run_version_pinned_sciverse_chunking",
    "verify_external_chunker_repo",
    "write_chunked_corpus_artifact",
]
