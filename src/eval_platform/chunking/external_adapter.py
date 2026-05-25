"""Thin adapter for calling external Python chunker code."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator

from eval_platform.artifacts.manifest import ArtifactManifest
from eval_platform.artifacts.store import ArtifactStore
from eval_platform.chunking.external_repo import (
    ExternalChunkerRepoSpec,
    verify_external_chunker_repo,
)
from eval_platform.chunking.runner import ChunkingRunConfig, run_chunking
from eval_platform.chunking.schema import ChunkRecord
from eval_platform.datasets.schema import NormalizedDataset


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class ExternalChunkerAdapterError(Exception):
    """Raised when an external chunker callable cannot be loaded or normalized."""


class PythonCallableChunkerConfig(BaseModel):
    """How to load a Python callable from an external repository checkout."""

    repo_path: str
    module: str
    callable_name: str
    callable_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("repo_path", "module", "callable_name")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


@contextmanager
def _temporary_module_import_path(path: str, module_name: str) -> Iterator[None]:
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == module_name or name.startswith(f"{module_name}.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    sys.path.insert(0, path)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass
        for name in list(sys.modules):
            if name == module_name or name.startswith(f"{module_name}."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


class PythonCallableExternalChunker:
    """External chunker adapter backed by a Python callable."""

    def __init__(self, config: PythonCallableChunkerConfig) -> None:
        self._config = config

    def _load_callable(self) -> Any:
        with _temporary_module_import_path(self._config.repo_path, self._config.module):
            try:
                module = importlib.import_module(self._config.module)
            except Exception as exc:  # pragma: no cover - exercised by tests
                raise ExternalChunkerAdapterError(
                    f"Failed to import external chunker module {self._config.module!r}"
                ) from exc

        chunk_callable = getattr(module, self._config.callable_name, None)
        if chunk_callable is None or not callable(chunk_callable):
            raise ExternalChunkerAdapterError(
                f"External chunker callable not found: "
                f"{self._config.module}.{self._config.callable_name}"
            )
        return chunk_callable

    def chunk_corpus(self, dataset: NormalizedDataset) -> Iterable[ChunkRecord]:
        chunk_callable = self._load_callable()
        try:
            rows = chunk_callable(dataset, **self._config.callable_kwargs)
        except Exception as exc:
            raise ExternalChunkerAdapterError("External chunker callable raised an error") from exc

        if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
            raise ExternalChunkerAdapterError("External chunker callable must return an iterable")

        normalized_rows: list[ChunkRecord] = []
        for row in rows:
            if isinstance(row, ChunkRecord):
                normalized_rows.append(row)
                continue
            if isinstance(row, dict):
                try:
                    normalized_rows.append(ChunkRecord.model_validate(row))
                except ValidationError as exc:
                    raise ExternalChunkerAdapterError(
                        "External chunker dict row could not be converted to ChunkRecord"
                    ) from exc
                continue
            raise ExternalChunkerAdapterError(
                f"Unsupported external chunker row type: {type(row).__name__}"
            )
        return normalized_rows


def run_version_pinned_external_chunking(
    store: ArtifactStore,
    *,
    source_artifact_id: str,
    output_artifact_id: str,
    chunker_name: str,
    repo_spec: ExternalChunkerRepoSpec,
    adapter_config: PythonCallableChunkerConfig,
    chunk_params: dict[str, Any] | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Validate the external repo version and run chunking through the thin adapter."""
    verify_external_chunker_repo(repo_spec)

    if adapter_config.repo_path != repo_spec.repo_path:
        raise ExternalChunkerAdapterError(
            "adapter_config.repo_path must match repo_spec.repo_path"
        )

    final_chunk_params: dict[str, Any] = {}
    if chunk_params:
        final_chunk_params.update(chunk_params)
    final_chunk_params.update(
        {
            "adapter_type": "python_callable",
            "adapter_module": adapter_config.module,
            "adapter_callable": adapter_config.callable_name,
        }
    )

    return run_chunking(
        store,
        ChunkingRunConfig(
            source_artifact_id=source_artifact_id,
            output_artifact_id=output_artifact_id,
            chunker_name=chunker_name,
            chunker_repo_path=repo_spec.repo_path,
            chunk_params=final_chunk_params,
            created_by=created_by,
            code_git_sha=code_git_sha,
            metadata=metadata or {},
        ),
        PythonCallableExternalChunker(adapter_config),
    )
