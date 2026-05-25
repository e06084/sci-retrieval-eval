"""Sciverse admin-ingest external chunker adapter."""

from __future__ import annotations

import importlib
import io
import json
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from eval_platform.artifacts.manifest import ArtifactManifest
from eval_platform.artifacts.store import ArtifactStore
from eval_platform.chunking.external_repo import (
    ExternalChunkerRepoSpec,
    verify_external_chunker_repo,
)
from eval_platform.chunking.runner import ChunkingRunConfig, run_chunking
from eval_platform.chunking.schema import ChunkRecord
from eval_platform.datasets.schema import NormalizedDataset

_SCIVERSE_PACKAGE_SUBDIR = Path("python_services") / "admin-ingest"
_SCIVERSE_PACKAGE_PREFIX = "admin_ingest"


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class SciverseAdapterError(Exception):
    """Raised when the Sciverse admin-ingest adapter cannot be loaded or normalized."""


class SciverseAdminIngestChunkerConfig(BaseModel):
    """Config for chunking through sciverse_clean admin-ingest."""

    repo_path: str
    chunk_size: int = Field(default=600, ge=1)
    chunk_overlap: int = Field(default=0, ge=0)
    keep_separator: bool = True
    structured_min_chunk_size: int = Field(default=200, ge=1)
    structured_max_chunk_size: int = Field(default=800, ge=1)
    structured_ideal_tolerance: int = Field(default=100, ge=0)
    source_type: str = "benchmark"
    file_ext: str = ".jsonl"
    default_job_id: str = ""

    @field_validator("repo_path", "source_type", "file_ext")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")

    @model_validator(mode="after")
    def validate_structured_chunk_bounds(self) -> SciverseAdminIngestChunkerConfig:
        if self.structured_min_chunk_size > self.structured_max_chunk_size:
            raise ValueError(
                "structured_min_chunk_size must be less than or equal to "
                "structured_max_chunk_size"
            )
        return self

    @property
    def package_root(self) -> Path:
        return Path(self.repo_path).resolve() / _SCIVERSE_PACKAGE_SUBDIR

    def chunk_params(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "keep_separator": self.keep_separator,
            "structured_min_chunk_size": self.structured_min_chunk_size,
            "structured_max_chunk_size": self.structured_max_chunk_size,
            "structured_ideal_tolerance": self.structured_ideal_tolerance,
            "source_type": self.source_type,
            "file_ext": self.file_ext,
        }

    def recursive_chunk_options_params(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "keep_separator": self.keep_separator,
            "structured_min_chunk_size": self.structured_min_chunk_size,
            "structured_max_chunk_size": self.structured_max_chunk_size,
            "structured_ideal_tolerance": self.structured_ideal_tolerance,
        }


@contextmanager
def _temporary_package_root(path: Path, package_prefix: str) -> Iterator[None]:
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == package_prefix or name.startswith(f"{package_prefix}.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        importlib.invalidate_caches()
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass
        for name in list(sys.modules):
            if name == package_prefix or name.startswith(f"{package_prefix}."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def _encode_sciverse_row(doc: Any, source_type: str) -> bytes:
    payload: dict[str, Any] = {
        "doc_id": doc.doc_id,
        "title": doc.title or "",
        "text": doc.text,
        "content": doc.text,
        "source_type": source_type,
    }
    if doc.metadata:
        payload["extra_info"] = doc.metadata
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _decode_extra_info(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    elif isinstance(raw, str):
        text = raw
    else:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


def _chunk_record_from_sciverse_row(row: Any) -> ChunkRecord:
    metadata: dict[str, Any] = {}
    extra_info = _decode_extra_info(row.extra_info if hasattr(row, "extra_info") else None)
    if extra_info:
        metadata["extra_info"] = extra_info
    for attr in ("url", "page_no", "source_type", "lang", "category", "job_id", "task_id"):
        value = getattr(row, attr, None)
        if value not in (None, "", []):
            metadata[attr] = value

    text = row.text
    offset = row.offset if hasattr(row, "offset") else 0
    if not isinstance(offset, int):
        offset = int(offset)
    chunk_no = row.chunk_no if hasattr(row, "chunk_no") else 0
    title = row.title if hasattr(row, "title") else None

    return ChunkRecord(
        chunk_id=row.chunk_id,
        doc_id=row.doc_id,
        title=title or None,
        text=text,
        chunk_index=chunk_no,
        start_offset=offset,
        end_offset=offset + len(text.encode("utf-8")),
        metadata=metadata,
    )


class SciverseAdminIngestExternalChunker:
    """External chunker adapter backed by sciverse_clean admin-ingest."""

    def __init__(self, config: SciverseAdminIngestChunkerConfig) -> None:
        self._config = config

    def chunk_corpus(self, dataset: NormalizedDataset) -> Iterable[ChunkRecord]:
        package_root = self._config.package_root
        if not package_root.is_dir():
            raise SciverseAdapterError(
                f"sciverse admin-ingest package root not found: {package_root}"
            )

        lines = b"".join(
            _encode_sciverse_row(doc, self._config.source_type) + b"\n" for doc in dataset.corpus
        )

        with _temporary_package_root(package_root, _SCIVERSE_PACKAGE_PREFIX):
            try:
                recursive_split = importlib.import_module(
                    "admin_ingest.chunk.recursive_split"
                )
                pipeline_steps = importlib.import_module("admin_ingest.pipeline.steps")
            except Exception as exc:  # pragma: no cover - exercised by tests
                raise SciverseAdapterError(
                    "Failed to import sciverse admin-ingest chunking modules"
                ) from exc

            try:
                options_cls = recursive_split.RecursiveChunkOptions
                chunk_records = pipeline_steps.chunk_ndjson_records
            except AttributeError as exc:
                raise SciverseAdapterError(
                    "sciverse admin-ingest is missing required chunking symbols"
                ) from exc

            options = options_cls(**self._config.recursive_chunk_options_params())

            try:
                _input_lines, rows = chunk_records(
                    io.BytesIO(lines),
                    options,
                    file_ext=self._config.file_ext,
                    job_source_type=self._config.source_type,
                    prechunked_jsonl=False,
                    default_job_id=self._config.default_job_id,
                )
            except Exception as exc:
                raise SciverseAdapterError(
                    "sciverse admin-ingest chunking callable raised an error"
                ) from exc

        if isinstance(rows, (str, bytes)) or not isinstance(rows, Iterable):
            raise SciverseAdapterError("sciverse admin-ingest chunker must return an iterable")

        return [_chunk_record_from_sciverse_row(row) for row in rows]


def run_version_pinned_sciverse_chunking(
    store: ArtifactStore,
    *,
    source_artifact_id: str,
    output_artifact_id: str,
    chunker_name: str,
    repo_spec: ExternalChunkerRepoSpec,
    chunker_config: SciverseAdminIngestChunkerConfig,
    chunk_params: dict[str, Any] | None = None,
    created_by: str | None = None,
    code_git_sha: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Validate repo version and run chunking through sciverse admin-ingest."""
    verify_external_chunker_repo(repo_spec)

    if Path(chunker_config.repo_path).resolve() != Path(repo_spec.repo_path).resolve():
        raise SciverseAdapterError(
            "chunker_config.repo_path must match repo_spec.repo_path"
        )

    final_chunk_params: dict[str, Any] = {}
    if chunk_params:
        final_chunk_params.update(chunk_params)
    final_chunk_params.update(chunker_config.chunk_params())
    final_chunk_params.update(
        {
            "adapter_type": "sciverse_admin_ingest",
            "adapter_package_subdir": str(_SCIVERSE_PACKAGE_SUBDIR),
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
        SciverseAdminIngestExternalChunker(chunker_config),
    )
