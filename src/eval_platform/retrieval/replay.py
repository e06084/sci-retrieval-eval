"""Replay execution for retrieval runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest, ArtifactStore
from eval_platform.retrieval.artifact import (
    read_retrieval_run_artifact,
    write_retrieval_run_artifact,
)
from eval_platform.retrieval.errors import RetrievalRunError


class ReplayConfig(Protocol):
    output_artifact_id: str
    queries_per_shard: int
    replay_source_retrieval_run_artifact_id: str | None
    created_by: str | None
    code_git_sha: str | None


def run_retrieval_replay(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: ReplayConfig,
    *,
    build_manifest_metadata: Callable[[Any], dict[str, Any]],
    build_dependencies: Callable[[Any], list[ArtifactDependency]],
) -> ArtifactManifest:
    """Copy a trace-complete source retrieval run into a replay artifact."""

    if config.replay_source_retrieval_run_artifact_id is None:
        raise RetrievalRunError(
            "replay_source_retrieval_run_artifact_id is required for replay execution"
        )

    source_records = read_retrieval_run_artifact(
        source_store,
        config.replay_source_retrieval_run_artifact_id,
    )
    if any(record.trace is None for record in source_records):
        raise RetrievalRunError("replay source retrieval_run artifact is missing replay trace")

    return write_retrieval_run_artifact(
        output_store,
        config.output_artifact_id,
        source_records,
        queries_per_shard=config.queries_per_shard,
        metadata=build_manifest_metadata(config),
        dependencies=build_dependencies(config),
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
    )
