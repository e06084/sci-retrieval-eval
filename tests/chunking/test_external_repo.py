"""Tests for version-pinned external chunker repository validation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_platform.chunking.external_repo import (
    ExternalChunkerRepoMismatchError,
    ExternalChunkerRepoSpec,
    verify_external_chunker_repo,
)
from eval_platform.chunking.git import GitRepoDirtyError


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


def _init_git_repo(repo_path: Path, *, remote_url: str = "https://example.com/chunker.git") -> str:
    _run_git(repo_path, "init")
    _run_git(repo_path, "config", "user.name", "test-user")
    _run_git(repo_path, "config", "user.email", "test@example.com")
    (repo_path / "README.md").write_text("initial\n", encoding="utf-8")
    _run_git(repo_path, "add", "README.md")
    _run_git(repo_path, "commit", "-m", "initial")
    _run_git(repo_path, "remote", "add", "origin", remote_url)
    return _run_git(repo_path, "rev-parse", "HEAD")


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[Path, str]:
    repo_path = tmp_path / "external-repo"
    repo_path.mkdir()
    commit_sha = _init_git_repo(repo_path)
    return repo_path, commit_sha


def test_repo_spec_constructs_for_valid_input(git_repo: tuple[Path, str]) -> None:
    repo_path, commit_sha = git_repo
    spec = ExternalChunkerRepoSpec(
        repo_path=str(repo_path),
        expected_remote_url="https://example.com/chunker.git",
        expected_commit_sha=commit_sha,
    )
    assert spec.expected_commit_sha == commit_sha


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("repo_path", ""),
        ("repo_path", " "),
        ("expected_remote_url", ""),
        ("expected_remote_url", " "),
        ("expected_commit_sha", ""),
        ("expected_commit_sha", " "),
    ],
)
def test_repo_spec_rejects_blank_values(
    git_repo: tuple[Path, str],
    field_name: str,
    value: str,
) -> None:
    repo_path, commit_sha = git_repo
    payload = {
        "repo_path": str(repo_path),
        "expected_remote_url": "https://example.com/chunker.git",
        "expected_commit_sha": commit_sha,
        field_name: value,
    }
    with pytest.raises(ValidationError):
        ExternalChunkerRepoSpec.model_validate(payload)


def test_verify_external_chunker_repo_returns_state_for_matching_repo(
    git_repo: tuple[Path, str],
) -> None:
    repo_path, commit_sha = git_repo
    spec = ExternalChunkerRepoSpec(
        repo_path=str(repo_path),
        expected_remote_url="https://example.com/chunker.git",
        expected_commit_sha=commit_sha,
    )
    state = verify_external_chunker_repo(spec)
    assert state.repo_url == "https://example.com/chunker.git"
    assert state.commit_sha == commit_sha
    assert state.is_dirty is False


def test_verify_external_chunker_repo_raises_for_remote_mismatch(
    git_repo: tuple[Path, str],
) -> None:
    repo_path, commit_sha = git_repo
    spec = ExternalChunkerRepoSpec(
        repo_path=str(repo_path),
        expected_remote_url="https://example.com/other.git",
        expected_commit_sha=commit_sha,
    )
    with pytest.raises(ExternalChunkerRepoMismatchError, match="remote URL mismatch"):
        verify_external_chunker_repo(spec)


def test_verify_external_chunker_repo_raises_for_commit_mismatch(
    git_repo: tuple[Path, str],
) -> None:
    repo_path, commit_sha = git_repo
    wrong_commit_sha = ("0" if commit_sha[0] != "0" else "1") + commit_sha[1:]
    spec = ExternalChunkerRepoSpec(
        repo_path=str(repo_path),
        expected_remote_url="https://example.com/chunker.git",
        expected_commit_sha=wrong_commit_sha,
    )
    with pytest.raises(ExternalChunkerRepoMismatchError, match="commit SHA mismatch"):
        verify_external_chunker_repo(spec)


def test_verify_external_chunker_repo_raises_for_dirty_repo(
    git_repo: tuple[Path, str],
) -> None:
    repo_path, commit_sha = git_repo
    (repo_path / "README.md").write_text("dirty\n", encoding="utf-8")
    spec = ExternalChunkerRepoSpec(
        repo_path=str(repo_path),
        expected_remote_url="https://example.com/chunker.git",
        expected_commit_sha=commit_sha,
    )
    with pytest.raises(GitRepoDirtyError):
        verify_external_chunker_repo(spec)
