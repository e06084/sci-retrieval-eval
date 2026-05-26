"""Tests for version-pinned external chunking helper."""

from __future__ import annotations

import subprocess
import textwrap
import uuid
from pathlib import Path

import pytest

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking import (
    CHUNKED_CORPUS_ARTIFACT_TYPE,
    ExternalChunkerRepoMismatchError,
    ExternalChunkerRepoSpec,
    GitRepoDirtyError,
    PythonCallableChunkerConfig,
    read_chunked_corpus_artifact,
    run_version_pinned_external_chunking,
)
from eval_platform.chunking.external_adapter import ExternalChunkerAdapterError
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


def _init_git_repo(repo_path: Path, module_name: str) -> tuple[str, str]:
    _run_git(repo_path, "init")
    _run_git(repo_path, "config", "user.name", "test-user")
    _run_git(repo_path, "config", "user.email", "test@example.com")
    (repo_path / "README.md").write_text("initial\n", encoding="utf-8")
    (repo_path / f"{module_name}.py").write_text(
        textwrap.dedent(
            """
            def chunk_dataset(dataset, prefix="chunk"):
                for doc in dataset.corpus:
                    yield {
                        "chunk_id": f"{prefix}-{doc.doc_id}",
                        "doc_id": doc.doc_id,
                        "text": doc.text,
                        "title": doc.title,
                        "chunk_index": 0,
                    }
            """
        ),
        encoding="utf-8",
    )
    _run_git(repo_path, "add", "README.md", f"{module_name}.py")
    _run_git(repo_path, "commit", "-m", "initial")
    _run_git(repo_path, "remote", "add", "origin", "https://example.com/chunker.git")
    commit_sha = _run_git(repo_path, "rev-parse", "HEAD")
    remote_url = _run_git(repo_path, "config", "--get", "remote.origin.url")
    return commit_sha, remote_url


def _module_name() -> str:
    return f"fake_versioned_chunker_{uuid.uuid4().hex}"


def _different_sha(commit_sha: str) -> str:
    replacement = "0" if commit_sha[-1] != "0" else "1"
    return f"{commit_sha[:-1]}{replacement}"


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def source_artifact_id(store: LocalArtifactStore) -> str:
    artifact_id = "litsearch_test"
    dataset = NormalizedDataset(
        corpus=[
            CorpusRecord(doc_id="doc-1", text="first document", title="First"),
            CorpusRecord(doc_id="doc-2", text="second document", title="Second"),
        ],
        queries=[QueryRecord(query_id="q-1", text="query")],
        qrels=[],
        metadata={"source": "unit-test"},
    )
    write_normalized_dataset_artifact(store, artifact_id, dataset)
    return artifact_id


def test_run_version_pinned_external_chunking_success(
    tmp_path: Path,
    store: LocalArtifactStore,
    source_artifact_id: str,
) -> None:
    repo_path = tmp_path / "external-repo"
    repo_path.mkdir()
    module_name = _module_name()
    commit_sha, remote_url = _init_git_repo(repo_path, module_name)

    manifest = run_version_pinned_external_chunking(
        store,
        source_artifact_id=source_artifact_id,
        output_artifact_id="litsearch_test_chunks",
        chunker_name="real-external-chunker",
        repo_spec=ExternalChunkerRepoSpec(
            repo_path=str(repo_path),
            expected_remote_url=remote_url,
            expected_commit_sha=commit_sha,
        ),
        adapter_config=PythonCallableChunkerConfig(
            repo_path=str(repo_path),
            module=module_name,
            callable_name="chunk_dataset",
            callable_kwargs={"prefix": "ext"},
        ),
        chunk_params={"window": 256},
        created_by="test-suite",
    )

    loaded = read_chunked_corpus_artifact(store, "litsearch_test_chunks")
    assert store.is_complete(CHUNKED_CORPUS_ARTIFACT_TYPE, "litsearch_test_chunks") is True
    assert [chunk.chunk_id for chunk in loaded.chunks] == ["ext-doc-1", "ext-doc-2"]
    assert manifest.dependencies[0].artifact_id == source_artifact_id
    assert manifest.metadata["chunker"]["repo_url"] == remote_url
    assert manifest.metadata["chunker"]["commit_sha"] == commit_sha
    assert manifest.metadata["chunker"]["is_dirty"] is False
    assert manifest.metadata["chunk_params"] == {
        "window": 256,
        "adapter_type": "python_callable",
        "adapter_module": module_name,
        "adapter_callable": "chunk_dataset",
    }


def test_run_version_pinned_external_chunking_fails_for_dirty_repo(
    tmp_path: Path,
    store: LocalArtifactStore,
    source_artifact_id: str,
) -> None:
    repo_path = tmp_path / "external-repo"
    repo_path.mkdir()
    module_name = _module_name()
    commit_sha, remote_url = _init_git_repo(repo_path, module_name)
    (repo_path / "README.md").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(GitRepoDirtyError):
        run_version_pinned_external_chunking(
            store,
            source_artifact_id=source_artifact_id,
            output_artifact_id="litsearch_test_chunks",
            chunker_name="real-external-chunker",
            repo_spec=ExternalChunkerRepoSpec(
                repo_path=str(repo_path),
                expected_remote_url=remote_url,
                expected_commit_sha=commit_sha,
            ),
            adapter_config=PythonCallableChunkerConfig(
                repo_path=str(repo_path),
                module=module_name,
                callable_name="chunk_dataset",
            ),
        )

    assert store.is_complete(CHUNKED_CORPUS_ARTIFACT_TYPE, "litsearch_test_chunks") is False


def test_run_version_pinned_external_chunking_fails_for_remote_mismatch(
    tmp_path: Path,
    store: LocalArtifactStore,
    source_artifact_id: str,
) -> None:
    repo_path = tmp_path / "external-repo"
    repo_path.mkdir()
    module_name = _module_name()
    commit_sha, _remote_url = _init_git_repo(repo_path, module_name)

    with pytest.raises(ExternalChunkerRepoMismatchError, match="remote URL mismatch"):
        run_version_pinned_external_chunking(
            store,
            source_artifact_id=source_artifact_id,
            output_artifact_id="litsearch_test_chunks",
            chunker_name="real-external-chunker",
            repo_spec=ExternalChunkerRepoSpec(
                repo_path=str(repo_path),
                expected_remote_url="https://example.com/other.git",
                expected_commit_sha=commit_sha,
            ),
            adapter_config=PythonCallableChunkerConfig(
                repo_path=str(repo_path),
                module=module_name,
                callable_name="chunk_dataset",
            ),
        )


def test_run_version_pinned_external_chunking_fails_for_commit_mismatch(
    tmp_path: Path,
    store: LocalArtifactStore,
    source_artifact_id: str,
) -> None:
    repo_path = tmp_path / "external-repo"
    repo_path.mkdir()
    module_name = _module_name()
    commit_sha, remote_url = _init_git_repo(repo_path, module_name)

    with pytest.raises(ExternalChunkerRepoMismatchError, match="commit SHA mismatch"):
        run_version_pinned_external_chunking(
            store,
            source_artifact_id=source_artifact_id,
            output_artifact_id="litsearch_test_chunks",
            chunker_name="real-external-chunker",
            repo_spec=ExternalChunkerRepoSpec(
                repo_path=str(repo_path),
                expected_remote_url=remote_url,
                expected_commit_sha=_different_sha(commit_sha),
            ),
            adapter_config=PythonCallableChunkerConfig(
                repo_path=str(repo_path),
                module=module_name,
                callable_name="chunk_dataset",
            ),
        )


def test_run_version_pinned_external_chunking_fails_for_adapter_repo_path_mismatch(
    tmp_path: Path,
    store: LocalArtifactStore,
    source_artifact_id: str,
) -> None:
    repo_path = tmp_path / "external-repo"
    repo_path.mkdir()
    other_repo_path = tmp_path / "other-external-repo"
    other_repo_path.mkdir()
    module_name = _module_name()
    commit_sha, remote_url = _init_git_repo(repo_path, module_name)

    with pytest.raises(ExternalChunkerAdapterError, match="must match repo_spec.repo_path"):
        run_version_pinned_external_chunking(
            store,
            source_artifact_id=source_artifact_id,
            output_artifact_id="litsearch_test_chunks",
            chunker_name="real-external-chunker",
            repo_spec=ExternalChunkerRepoSpec(
                repo_path=str(repo_path),
                expected_remote_url=remote_url,
                expected_commit_sha=commit_sha,
            ),
            adapter_config=PythonCallableChunkerConfig(
                repo_path=str(other_repo_path),
                module=module_name,
                callable_name="chunk_dataset",
            ),
        )
