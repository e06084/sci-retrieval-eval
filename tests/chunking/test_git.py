"""Tests for git repository inspection helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from eval_platform.chunking.git import (
    GitRepoDirtyError,
    GitRepoError,
    ensure_git_repo_clean,
    inspect_git_repo,
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


def _init_git_repo(repo_path: Path, *, remote_url: str | None = None) -> None:
    _run_git(repo_path, "init")
    _run_git(repo_path, "config", "user.name", "test-user")
    _run_git(repo_path, "config", "user.email", "test@example.com")
    (repo_path / "README.md").write_text("initial\n", encoding="utf-8")
    _run_git(repo_path, "add", "README.md")
    _run_git(repo_path, "commit", "-m", "initial")
    if remote_url is not None:
        _run_git(repo_path, "remote", "add", "origin", remote_url)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo_path = tmp_path / "chunker-repo"
    repo_path.mkdir()
    _init_git_repo(repo_path)
    return repo_path


def test_inspect_git_repo_returns_commit_sha(git_repo: Path) -> None:
    expected_sha = _run_git_output(git_repo, "rev-parse", "HEAD")

    state = inspect_git_repo(git_repo)

    assert state.commit_sha == expected_sha


def test_inspect_git_repo_returns_branch(git_repo: Path) -> None:
    expected_branch = _run_git_output(git_repo, "rev-parse", "--abbrev-ref", "HEAD")

    state = inspect_git_repo(git_repo)

    assert state.branch == expected_branch


def test_inspect_git_repo_returns_remote_origin_url(tmp_path: Path) -> None:
    repo_path = tmp_path / "with-remote"
    repo_path.mkdir()
    _init_git_repo(repo_path, remote_url="https://example.com/chunker.git")

    state = inspect_git_repo(repo_path)

    assert state.repo_url == "https://example.com/chunker.git"


def test_inspect_git_repo_without_remote_has_none_url(git_repo: Path) -> None:
    state = inspect_git_repo(git_repo)

    assert state.repo_url is None


def test_inspect_git_repo_clean_repo_is_not_dirty(git_repo: Path) -> None:
    state = inspect_git_repo(git_repo)

    assert state.is_dirty is False


def test_inspect_git_repo_modified_repo_is_dirty(git_repo: Path) -> None:
    (git_repo / "README.md").write_text("modified\n", encoding="utf-8")

    state = inspect_git_repo(git_repo)

    assert state.is_dirty is True


def test_ensure_git_repo_clean_returns_state_for_clean_repo(git_repo: Path) -> None:
    state = ensure_git_repo_clean(git_repo)

    assert state.is_dirty is False
    assert state.repo_path == str(git_repo.resolve())


def test_ensure_git_repo_clean_raises_for_dirty_repo(git_repo: Path) -> None:
    (git_repo / "README.md").write_text("modified\n", encoding="utf-8")

    with pytest.raises(GitRepoDirtyError):
        ensure_git_repo_clean(git_repo)


def test_inspect_git_repo_raises_for_non_git_directory(tmp_path: Path) -> None:
    repo_path = tmp_path / "not-a-repo"
    repo_path.mkdir()

    with pytest.raises(GitRepoError):
        inspect_git_repo(repo_path)


def _run_git_output(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
