"""Tests for chunking runner orchestration."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking import (
    CHUNKED_CORPUS_ARTIFACT_TYPE,
    CHUNKS_FILENAME,
    ChunkRecord,
    ProgressEvent,
    read_chunked_corpus_artifact,
    run_chunking,
)
from eval_platform.chunking.git import GitRepoDirtyError
from eval_platform.chunking.runner import ChunkingRunConfig, ExternalChunker
from eval_platform.datasets import (
    CorpusRecord,
    NormalizedDataset,
    QueryRecord,
    write_normalized_dataset_artifact,
)


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


pytestmark = pytest.mark.skipif(not _git_available(), reason="git not available")


def _run_git(repo_path: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo_path, check=True, capture_output=True)


def _init_git_repo(repo_path: Path) -> None:
    _run_git(repo_path, "init")
    _run_git(repo_path, "config", "user.name", "test-user")
    _run_git(repo_path, "config", "user.email", "test@example.com")
    (repo_path / "README.md").write_text("initial\n", encoding="utf-8")
    _run_git(repo_path, "add", "README.md")
    _run_git(repo_path, "commit", "-m", "initial")


def _git_head(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class FakeChunker:
    """In-memory chunker used for runner tests."""

    def __init__(self) -> None:
        self.call_count = 0

    def chunk_corpus(self, dataset: NormalizedDataset) -> Iterable[ChunkRecord]:
        self.call_count += 1
        return [
            ChunkRecord(
                chunk_id=f"{doc.doc_id}-0",
                doc_id=doc.doc_id,
                text=f"chunk from {doc.doc_id}",
                chunk_index=0,
            )
            for doc in dataset.corpus
        ]


class MultiChunkPerDocChunker:
    def chunk_corpus(self, dataset: NormalizedDataset) -> Iterable[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for doc in dataset.corpus:
            chunks.extend(
                [
                    ChunkRecord(
                        chunk_id=f"{doc.doc_id}-0",
                        doc_id=doc.doc_id,
                        text=f"chunk-0 from {doc.doc_id}",
                        chunk_index=0,
                    ),
                    ChunkRecord(
                        chunk_id=f"{doc.doc_id}-1",
                        doc_id=doc.doc_id,
                        text=f"chunk-1 from {doc.doc_id}",
                        chunk_index=1,
                    ),
                ]
            )
        return chunks


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo_path = tmp_path / "chunker-repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)
    return repo_path


@pytest.fixture
def source_artifact_id(store: LocalArtifactStore) -> str:
    artifact_id = "litsearch_test"
    dataset = NormalizedDataset(
        corpus=[
            CorpusRecord(doc_id="doc-1", text="first document"),
            CorpusRecord(doc_id="doc-2", text="second document"),
        ],
        queries=[QueryRecord(query_id="q-1", text="query")],
        qrels=[],
        metadata={"source": "unit-test"},
    )
    write_normalized_dataset_artifact(store, artifact_id, dataset)
    return artifact_id


def _run_config(
    source_artifact_id: str,
    git_repo: Path,
    *,
    output_artifact_id: str = "litsearch_test_chunks",
) -> ChunkingRunConfig:
    return ChunkingRunConfig(
        source_artifact_id=source_artifact_id,
        output_artifact_id=output_artifact_id,
        chunker_name="fake-chunker",
        chunker_repo_path=str(git_repo),
    )


def test_run_chunking_writes_chunked_corpus_artifact(
    store: LocalArtifactStore,
    git_repo: Path,
    source_artifact_id: str,
) -> None:
    config = _run_config(source_artifact_id, git_repo)
    config.chunk_params = {"max_tokens": 512}

    manifest = run_chunking(store, config, FakeChunker())

    assert manifest.artifact_id == "litsearch_test_chunks"
    assert store.is_complete(CHUNKED_CORPUS_ARTIFACT_TYPE, "litsearch_test_chunks") is True


def test_run_chunking_round_trip_reads_expected_chunks(
    store: LocalArtifactStore,
    git_repo: Path,
    source_artifact_id: str,
) -> None:
    output_artifact_id = "litsearch_test_chunks"
    config = _run_config(source_artifact_id, git_repo, output_artifact_id=output_artifact_id)

    run_chunking(store, config, FakeChunker())
    loaded = read_chunked_corpus_artifact(store, output_artifact_id)

    source_doc_ids = {"doc-1", "doc-2"}
    assert len(loaded.chunks) == len(source_doc_ids)
    for chunk in loaded.chunks:
        assert chunk.doc_id in source_doc_ids
        assert chunk.chunk_id == f"{chunk.doc_id}-0"
        assert chunk.chunk_index == 0
        assert chunk.text == f"chunk from {chunk.doc_id}"


def test_run_chunking_records_source_dependency(
    store: LocalArtifactStore,
    git_repo: Path,
    source_artifact_id: str,
) -> None:
    manifest = run_chunking(store, _run_config(source_artifact_id, git_repo), FakeChunker())

    assert len(manifest.dependencies) == 1
    assert manifest.dependencies[0].artifact_id == source_artifact_id
    assert manifest.dependencies[0].artifact_type == "normalized_dataset"


def test_run_chunking_records_chunker_provenance(
    store: LocalArtifactStore,
    git_repo: Path,
    source_artifact_id: str,
) -> None:
    config = _run_config(source_artifact_id, git_repo)
    config.chunk_params = {"max_tokens": 512, "overlap": 64}

    manifest = run_chunking(store, config, FakeChunker())
    chunker = manifest.metadata["chunker"]

    assert chunker["name"] == "fake-chunker"
    assert chunker["commit_sha"] == _git_head(git_repo)
    assert chunker["is_dirty"] is False
    assert manifest.metadata["chunk_params"] == {"max_tokens": 512, "overlap": 64}


def test_run_chunking_raises_for_dirty_chunker_repo_without_side_effects(
    store: LocalArtifactStore,
    git_repo: Path,
    source_artifact_id: str,
) -> None:
    output_artifact_id = "litsearch_test_chunks"
    (git_repo / "README.md").write_text("modified\n", encoding="utf-8")
    chunker = FakeChunker()
    config = _run_config(source_artifact_id, git_repo, output_artifact_id=output_artifact_id)

    with pytest.raises(GitRepoDirtyError):
        run_chunking(store, config, chunker)

    assert chunker.call_count == 0
    assert store.is_complete(CHUNKED_CORPUS_ARTIFACT_TYPE, output_artifact_id) is False
    assert not store.exists(CHUNKED_CORPUS_ARTIFACT_TYPE, output_artifact_id, CHUNKS_FILENAME)


def test_run_chunking_calls_fake_chunker_once(
    store: LocalArtifactStore,
    git_repo: Path,
    source_artifact_id: str,
) -> None:
    chunker = FakeChunker()

    run_chunking(store, _run_config(source_artifact_id, git_repo), chunker)

    assert chunker.call_count == 1


def test_run_chunking_shards_by_source_doc_count(
    store: LocalArtifactStore,
    git_repo: Path,
    source_artifact_id: str,
) -> None:
    config = _run_config(source_artifact_id, git_repo)
    config.file_record_num = 1

    manifest = run_chunking(store, config, MultiChunkPerDocChunker())

    assert manifest.metadata["sharding"]["enabled"] is True
    assert len(manifest.metadata["shards"]) == 2
    assert manifest.metadata["shards"][0]["source_doc_count"] == 1
    assert manifest.metadata["shards"][0]["chunk_count"] == 2
    assert manifest.metadata["shards"][1]["source_doc_count"] == 1
    assert manifest.metadata["shards"][1]["chunk_count"] == 2


def test_run_chunking_reports_doc_and_shard_progress(
    store: LocalArtifactStore,
    git_repo: Path,
    source_artifact_id: str,
) -> None:
    config = _run_config(source_artifact_id, git_repo)
    config.file_record_num = 1
    events: list[ProgressEvent] = []

    run_chunking(
        store,
        config,
        MultiChunkPerDocChunker(),
        progress_reporter=events.append,
    )

    doc_events = [event for event in events if event.metadata.get("kind") == "source_doc"]
    shard_events = [event for event in events if event.metadata.get("kind") == "shard"]
    assert [event.current for event in doc_events] == [1, 2]
    assert doc_events[0].total == 2
    assert [event.metadata["shard_id"] for event in shard_events] == ["part-00000", "part-00001"]


def test_run_chunking_progress_reporter_failure_does_not_write_success(
    store: LocalArtifactStore,
    git_repo: Path,
    source_artifact_id: str,
) -> None:
    config = _run_config(source_artifact_id, git_repo)
    config.file_record_num = 1

    def failing_reporter(_: ProgressEvent) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_chunking(
            store,
            config,
            MultiChunkPerDocChunker(),
            progress_reporter=failing_reporter,
        )

    assert store.is_complete(CHUNKED_CORPUS_ARTIFACT_TYPE, "litsearch_test_chunks") is False
    assert not store.exists(CHUNKED_CORPUS_ARTIFACT_TYPE, "litsearch_test_chunks", CHUNKS_FILENAME)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("source_artifact_id", ""),
        ("source_artifact_id", " "),
        ("output_artifact_id", ""),
        ("output_artifact_id", " "),
        ("chunker_name", ""),
        ("chunker_name", " "),
        ("chunker_repo_path", ""),
        ("chunker_repo_path", " "),
    ],
)
def test_chunking_run_config_rejects_empty_or_blank_strings(
    field_name: str,
    value: str,
) -> None:
    payload = {
        "source_artifact_id": "litsearch_test",
        "output_artifact_id": "litsearch_test_chunks",
        "chunker_name": "fake-chunker",
        "chunker_repo_path": "/tmp/chunker",
        field_name: value,
    }

    with pytest.raises(ValidationError):
        ChunkingRunConfig.model_validate(payload)


def test_chunking_run_config_default_dicts_are_independent() -> None:
    first = ChunkingRunConfig(
        source_artifact_id="source-a",
        output_artifact_id="output-a",
        chunker_name="chunker-a",
        chunker_repo_path="/tmp/chunker-a",
    )
    second = ChunkingRunConfig(
        source_artifact_id="source-b",
        output_artifact_id="output-b",
        chunker_name="chunker-b",
        chunker_repo_path="/tmp/chunker-b",
    )

    first.chunk_params["max_tokens"] = 512
    first.metadata["pipeline_step"] = "chunk"
    second.chunk_params["max_tokens"] = 256
    second.metadata["pipeline_step"] = "other"

    assert first.chunk_params == {"max_tokens": 512}
    assert first.metadata == {"pipeline_step": "chunk"}
    assert second.chunk_params == {"max_tokens": 256}
    assert second.metadata == {"pipeline_step": "other"}


def test_chunking_run_config_rejects_non_positive_file_record_num() -> None:
    with pytest.raises(ValidationError):
        ChunkingRunConfig(
            source_artifact_id="source-a",
            output_artifact_id="output-a",
            chunker_name="chunker-a",
            chunker_repo_path="/tmp/chunker-a",
            file_record_num=0,
        )


def test_external_chunker_protocol_is_structural() -> None:
    chunker = FakeChunker()

    assert isinstance(chunker, ExternalChunker)
