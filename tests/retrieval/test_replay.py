"""Tests for retrieval replay execution helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from eval_platform.artifacts import ArtifactDependency, LocalArtifactStore
from eval_platform.retrieval import RETRIEVAL_RUN_ARTIFACT_TYPE
from eval_platform.retrieval.artifact import (
    read_retrieval_run_artifact,
    write_retrieval_run_artifact,
)
from eval_platform.retrieval.errors import RetrievalRunError
from eval_platform.retrieval.replay import run_retrieval_replay
from eval_platform.retrieval.schema import RetrievalHit, RetrievalQueryResult


@dataclass
class ReplayConfig:
    output_artifact_id: str = "replayed-run"
    queries_per_shard: int = 1000
    replay_source_retrieval_run_artifact_id: str | None = "source-run"
    created_by: str | None = None
    code_git_sha: str | None = None


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _metadata(_config: ReplayConfig) -> dict[str, object]:
    return {"execution_mode": "replay"}


def _dependencies(config: ReplayConfig) -> list[ArtifactDependency]:
    assert config.replay_source_retrieval_run_artifact_id is not None
    return [
        ArtifactDependency(
            artifact_type=RETRIEVAL_RUN_ARTIFACT_TYPE,
            artifact_id=config.replay_source_retrieval_run_artifact_id,
        )
    ]


def test_run_retrieval_replay_copies_source_records(store: LocalArtifactStore) -> None:
    source_records = [
        RetrievalQueryResult(
            query_id="q-1",
            query_text="alpha",
            hits=[
                RetrievalHit(
                    rank=1,
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    text="text",
                    score=1.0,
                    recall_source="hybrid",
                )
            ],
            trace={"rewrite_queries": ["alpha"], "per_query": [], "final_hits": []},
        )
    ]
    write_retrieval_run_artifact(store, "source-run", source_records)

    manifest = run_retrieval_replay(
        store,
        store,
        ReplayConfig(),
        build_manifest_metadata=_metadata,
        build_dependencies=_dependencies,
    )
    replay_records = read_retrieval_run_artifact(store, "replayed-run")

    assert [record.model_dump(mode="json") for record in replay_records] == [
        record.model_dump(mode="json") for record in source_records
    ]
    assert manifest.metadata["execution_mode"] == "replay"
    assert store.is_complete(RETRIEVAL_RUN_ARTIFACT_TYPE, "replayed-run") is True


def test_run_retrieval_replay_fails_when_source_lacks_trace(
    store: LocalArtifactStore,
) -> None:
    write_retrieval_run_artifact(
        store,
        "source-run",
        [RetrievalQueryResult(query_id="q-1", query_text="alpha", hits=[])],
    )

    with pytest.raises(RetrievalRunError, match="missing replay trace"):
        run_retrieval_replay(
            store,
            store,
            ReplayConfig(),
            build_manifest_metadata=_metadata,
            build_dependencies=_dependencies,
        )

    assert store.is_complete(RETRIEVAL_RUN_ARTIFACT_TYPE, "replayed-run") is False
