"""Git repository inspection helpers for external chunker provenance."""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel


class GitRepoError(Exception):
    """Raised when git inspection fails."""


class GitRepoDirtyError(GitRepoError):
    """Raised when a git repository has uncommitted changes."""


class GitRepoState(BaseModel):
    """Snapshot of a local git repository state."""

    repo_path: str
    repo_url: str | None = None
    commit_sha: str
    branch: str | None = None
    is_dirty: bool


def _run_git(repo_path: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        message = stderr or stdout or f"git {' '.join(args)} failed with exit code {exc.returncode}"
        raise GitRepoError(message) from exc
    except FileNotFoundError as exc:
        raise GitRepoError("git executable not found") from exc
    return result.stdout.strip()


def inspect_git_repo(repo_path: str | Path) -> GitRepoState:
    """Inspect a local git repository and return its current state."""
    path = Path(repo_path).resolve()
    if not path.is_dir():
        raise GitRepoError(f"Repository path does not exist: {path}")

    commit_sha = _run_git(path, "rev-parse", "HEAD")
    branch = _run_git(path, "rev-parse", "--abbrev-ref", "HEAD")

    try:
        repo_url = _run_git(path, "config", "--get", "remote.origin.url") or None
    except GitRepoError:
        repo_url = None

    status_output = _run_git(path, "status", "--porcelain")
    is_dirty = bool(status_output)

    return GitRepoState(
        repo_path=str(path),
        repo_url=repo_url,
        commit_sha=commit_sha,
        branch=branch,
        is_dirty=is_dirty,
    )


def ensure_git_repo_clean(repo_path: str | Path) -> GitRepoState:
    """Inspect a git repository and fail if it has uncommitted changes."""
    state = inspect_git_repo(repo_path)
    if state.is_dirty:
        raise GitRepoDirtyError(f"Git repository is dirty: {state.repo_path}")
    return state
