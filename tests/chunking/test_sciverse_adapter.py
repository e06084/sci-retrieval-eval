"""Tests for the sciverse admin-ingest adapter."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking import (
    CHUNKED_CORPUS_ARTIFACT_TYPE,
    ExternalChunkerRepoSpec,
    SciverseAdapterError,
    SciverseAdminIngestChunkerConfig,
    SciverseAdminIngestExternalChunker,
    read_chunked_corpus_artifact,
    run_version_pinned_sciverse_chunking,
)
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


def _run_git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


def _init_fake_sciverse_repo(repo_path: Path) -> tuple[str, str]:
    _run_git(repo_path, "init")
    _run_git(repo_path, "config", "user.name", "test-user")
    _run_git(repo_path, "config", "user.email", "test@example.com")
    (repo_path / "README.md").write_text("fake sciverse repo\n", encoding="utf-8")

    _write(
        repo_path / "python_services" / "admin-ingest" / "admin_ingest" / "__init__.py",
        "",
    )
    _write(
        repo_path
        / "python_services"
        / "admin-ingest"
        / "admin_ingest"
        / "chunk"
        / "__init__.py",
        "",
    )
    _write(
        repo_path
        / "python_services"
        / "admin-ingest"
        / "admin_ingest"
        / "pipeline"
        / "__init__.py",
        "",
    )
    _write(
        repo_path
        / "python_services"
        / "admin-ingest"
        / "admin_ingest"
        / "chunk"
        / "recursive_split.py",
        """
        from dataclasses import dataclass

        @dataclass
        class RecursiveChunkOptions:
            chunk_size: int = 600
            chunk_overlap: int = 0
            keep_separator: bool = True
            structured_min_chunk_size: int = 200
            structured_max_chunk_size: int = 800
            structured_ideal_tolerance: int = 100
        """,
    )
    _write(
        repo_path
        / "python_services"
        / "admin-ingest"
        / "admin_ingest"
        / "pipeline"
        / "steps.py",
        """
        import json
        from dataclasses import dataclass


        @dataclass
        class FakeTextChunkLine:
            chunk_id: str
            doc_id: str
            title: str
            text: str
            offset: int
            chunk_no: int
            url: str = ""
            page_no: int = 0
            source_type: str = "benchmark"
            lang: str = "unknown"
            category: str = ""
            job_id: str = ""
            task_id: str = ""
            extra_info: bytes | None = None


        def chunk_ndjson_records(
            input_fp,
            chunk_opts,
            *,
            file_ext="",
            job_source_type="",
            prechunked_jsonl=False,
            default_job_id="",
        ):
            records = []
            input_lines = 0
            for raw in input_fp:
                if not raw.strip():
                    continue
                input_lines += 1
                obj = json.loads(raw.decode("utf-8"))
                text = obj["text"]
                midpoint = min(len(text), max(1, len(text) // 2))
                parts = [text[:midpoint], text[midpoint:]] if len(text) > midpoint else [text]
                offset = 0
                for chunk_no, part in enumerate(parts):
                    if not part:
                        continue
                    records.append(
                        FakeTextChunkLine(
                            chunk_id=f"{obj['doc_id']}-{chunk_no}",
                            doc_id=obj["doc_id"],
                            title=obj.get("title", ""),
                            text=part,
                            offset=offset,
                            chunk_no=chunk_no,
                            source_type=obj.get("source_type") or job_source_type or "benchmark",
                            job_id=default_job_id,
                            extra_info=json.dumps(
                                obj.get("extra_info", {}),
                                ensure_ascii=False,
                            ).encode("utf-8"),
                        )
                    )
                    offset += len(part.encode("utf-8"))
            return input_lines, records
        """,
    )

    _run_git(repo_path, "add", ".")
    _run_git(repo_path, "commit", "-m", "initial")
    _run_git(repo_path, "remote", "add", "origin", "https://example.com/sciverse.git")
    commit_sha = _run_git(repo_path, "rev-parse", "HEAD")
    remote_url = _run_git(repo_path, "config", "--get", "remote.origin.url")
    return commit_sha, remote_url


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def source_artifact_id(store: LocalArtifactStore) -> str:
    artifact_id = "ifirnf_test"
    dataset = NormalizedDataset(
        corpus=[
            CorpusRecord(
                doc_id="doc-1",
                title="First",
                text="abcdefghij",
                metadata={"origin_url": "https://example.com/doc-1"},
            ),
            CorpusRecord(
                doc_id="doc-2",
                title="Second",
                text="klmnopqrst",
                metadata={"origin_url": "https://example.com/doc-2"},
            ),
        ],
        queries=[QueryRecord(query_id="q-1", text="query")],
        qrels=[],
        metadata={"source": "unit-test"},
    )
    write_normalized_dataset_artifact(store, artifact_id, dataset)
    return artifact_id


def test_sciverse_admin_ingest_external_chunker_success(tmp_path: Path) -> None:
    repo_path = tmp_path / "sciverse-repo"
    repo_path.mkdir()
    _init_fake_sciverse_repo(repo_path)

    dataset = NormalizedDataset(
        corpus=[CorpusRecord(doc_id="doc-1", title="First", text="abcdefghij")],
        queries=[],
        qrels=[],
    )
    chunker = SciverseAdminIngestExternalChunker(
        SciverseAdminIngestChunkerConfig(repo_path=str(repo_path))
    )

    chunks = list(chunker.chunk_corpus(dataset))

    assert [chunk.chunk_id for chunk in chunks] == ["doc-1-0", "doc-1-1"]
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[0].start_offset == 0
    assert chunks[1].start_offset == len(chunks[0].text.encode("utf-8"))


def test_sciverse_admin_ingest_external_chunker_requires_package_root(tmp_path: Path) -> None:
    repo_path = tmp_path / "missing-sciverse-repo"
    repo_path.mkdir()

    dataset = NormalizedDataset(corpus=[], queries=[], qrels=[])
    chunker = SciverseAdminIngestExternalChunker(
        SciverseAdminIngestChunkerConfig(repo_path=str(repo_path))
    )

    with pytest.raises(SciverseAdapterError, match="package root not found"):
        list(chunker.chunk_corpus(dataset))


def test_sciverse_admin_ingest_chunker_config_rejects_inverted_structured_bounds() -> None:
    with pytest.raises(ValidationError, match="structured_min_chunk_size"):
        SciverseAdminIngestChunkerConfig(
            repo_path="/tmp/sciverse-repo",
            structured_min_chunk_size=900,
            structured_max_chunk_size=800,
        )


def test_run_version_pinned_sciverse_chunking_success(
    tmp_path: Path,
    store: LocalArtifactStore,
    source_artifact_id: str,
) -> None:
    repo_path = tmp_path / "sciverse-repo"
    repo_path.mkdir()
    commit_sha, remote_url = _init_fake_sciverse_repo(repo_path)

    manifest = run_version_pinned_sciverse_chunking(
        store,
        source_artifact_id=source_artifact_id,
        output_artifact_id="ifirnf_test_chunks",
        chunker_name="sciverse-admin-ingest",
        repo_spec=ExternalChunkerRepoSpec(
            repo_path=str(repo_path),
            expected_remote_url=remote_url,
            expected_commit_sha=commit_sha,
        ),
        chunker_config=SciverseAdminIngestChunkerConfig(
            repo_path=str(repo_path),
            chunk_size=700,
            chunk_overlap=50,
        ),
        chunk_params={"dataset": "ifirnf"},
        created_by="test-suite",
    )

    loaded = read_chunked_corpus_artifact(store, "ifirnf_test_chunks")

    assert store.is_complete(CHUNKED_CORPUS_ARTIFACT_TYPE, "ifirnf_test_chunks") is True
    assert len(loaded.chunks) == 4
    assert loaded.chunks[0].metadata["extra_info"]["origin_url"] == "https://example.com/doc-1"
    assert manifest.dependencies[0].artifact_id == source_artifact_id
    assert manifest.metadata["chunker"]["repo_url"] == remote_url
    assert manifest.metadata["chunker"]["commit_sha"] == commit_sha
    assert manifest.metadata["chunker"]["is_dirty"] is False
    assert manifest.metadata["chunk_params"] == {
        "dataset": "ifirnf",
        "chunk_size": 700,
        "chunk_overlap": 50,
        "keep_separator": True,
        "structured_min_chunk_size": 200,
        "structured_max_chunk_size": 800,
        "structured_ideal_tolerance": 100,
        "source_type": "benchmark",
        "file_ext": ".jsonl",
        "adapter_type": "sciverse_admin_ingest",
        "adapter_package_subdir": "python_services/admin-ingest",
    }


def test_run_version_pinned_sciverse_chunking_fails_for_repo_path_mismatch(
    tmp_path: Path,
    store: LocalArtifactStore,
    source_artifact_id: str,
) -> None:
    repo_path = tmp_path / "sciverse-repo"
    repo_path.mkdir()
    other_repo_path = tmp_path / "other-sciverse-repo"
    other_repo_path.mkdir()
    commit_sha, remote_url = _init_fake_sciverse_repo(repo_path)

    with pytest.raises(SciverseAdapterError, match="must match repo_spec.repo_path"):
        run_version_pinned_sciverse_chunking(
            store,
            source_artifact_id=source_artifact_id,
            output_artifact_id="ifirnf_test_chunks",
            chunker_name="sciverse-admin-ingest",
            repo_spec=ExternalChunkerRepoSpec(
                repo_path=str(repo_path),
                expected_remote_url=remote_url,
                expected_commit_sha=commit_sha,
            ),
            chunker_config=SciverseAdminIngestChunkerConfig(repo_path=str(other_repo_path)),
        )
