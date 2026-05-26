"""Raw snapshot to normalized dataset conversion helpers."""

from __future__ import annotations

import csv
import io
import json
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Protocol

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest
from eval_platform.artifacts.store import ArtifactStore
from eval_platform.chunking.progress import ProgressReporter, report_progress
from eval_platform.datasets.normalized import write_normalized_dataset_artifact
from eval_platform.datasets.raw import (
    RAW_DATASET_ARTIFACT_TYPE,
    RawDatasetFile,
    read_raw_dataset_artifact,
)
from eval_platform.datasets.schema import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
)

_NORMALIZED_SCHEMA_VERSION = "1"
_IFIR_NFCORPUS_NORMALIZER = "ifir_nfcorpus_raw_jsonl_tsv_v1"


def _non_empty_string(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value


class RawNormalizeError(Exception):
    """Raised when a raw snapshot cannot be normalized."""


class S3RawFileOpener:
    """Open raw files referenced by `s3://bucket/key` URIs."""

    def __init__(self, client: Any | None = None) -> None:
        self._client = client if client is not None else self._create_default_client()

    @staticmethod
    def _create_default_client() -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3RawFileOpener. "
                "Install with: pip install 'sci-retrieval-eval[s3]'"
            ) from exc
        return boto3.client("s3")

    @staticmethod
    def _parse_s3_uri(uri: str) -> tuple[str, str]:
        if not uri.startswith("s3://"):
            raise RawNormalizeError(f"Unsupported raw file URI: {uri!r}")

        without_scheme = uri[len("s3://") :]
        bucket, separator, key = without_scheme.partition("/")
        if not bucket or not separator or not key:
            raise RawNormalizeError(f"Invalid S3 raw file URI: {uri!r}")
        return bucket, key

    def open(self, uri: str) -> BinaryIO:
        bucket, key = self._parse_s3_uri(uri)
        response = self._client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        if hasattr(body, "read"):
            return body
        if isinstance(body, bytes):
            return io.BytesIO(body)
        raise RawNormalizeError("S3 get_object response body is not readable")


class RawFileOpener(Protocol):
    """Open one raw source URI as a binary readable stream."""

    def open(self, uri: str) -> BinaryIO:
        """Return a readable binary stream for the given raw source URI."""


class RawToNormalizedConfig(BaseModel):
    """Configuration for one raw-to-normalized conversion."""

    source_artifact_id: str
    output_artifact_id: str
    dataset_name: str
    split: str = "test"
    normalizer_name: str | None = None
    created_by: str | None = None
    code_git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_artifact_id", "output_artifact_id", "dataset_name", "split")
    @classmethod
    def validate_non_empty(cls, value: str, info: ValidationInfo) -> str:
        return _non_empty_string(value, info.field_name or "field")


def _read_jsonl_records(stream: BinaryIO) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in stream:
        if not raw_line.strip():
            continue
        records.append(json.loads(raw_line.decode("utf-8")))
    return records


def _read_tsv_rows(stream: BinaryIO) -> list[dict[str, str]]:
    text_stream = (line.decode("utf-8") for line in stream)
    return list(csv.DictReader(text_stream, delimiter="\t"))


def _find_required_file(files: list[RawDatasetFile], relative_path: str) -> RawDatasetFile:
    normalized_target = PurePosixPath(relative_path).as_posix()
    for file in files:
        if PurePosixPath(file.path).as_posix() == normalized_target:
            return file
    raise RawNormalizeError(f"Required raw file missing from snapshot: {relative_path}")


def _load_ifir_nfcorpus_dataset(
    snapshot_files: list[RawDatasetFile],
    opener: RawFileOpener,
    *,
    progress_reporter: ProgressReporter | None = None,
) -> NormalizedDataset:
    corpus_file = _find_required_file(snapshot_files, "corpus.jsonl")
    queries_file = _find_required_file(snapshot_files, "queries.jsonl")
    instructions_file = _find_required_file(snapshot_files, "instructions.jsonl")
    qrels_file = _find_required_file(snapshot_files, "qrels/test.tsv")

    completed_steps = 0
    total_steps = 4

    with opener.open(corpus_file.uri) as corpus_stream:
        corpus_rows = _read_jsonl_records(corpus_stream)
    completed_steps += 1
    report_progress(
        progress_reporter,
        stage="raw_to_normalized",
        current=completed_steps,
        total=total_steps,
        message="Loaded raw corpus records",
        metadata={"kind": "corpus", "record_count": len(corpus_rows), "path": corpus_file.path},
    )
    with opener.open(queries_file.uri) as queries_stream:
        query_rows = _read_jsonl_records(queries_stream)
    completed_steps += 1
    report_progress(
        progress_reporter,
        stage="raw_to_normalized",
        current=completed_steps,
        total=total_steps,
        message="Loaded raw query records",
        metadata={"kind": "queries", "record_count": len(query_rows), "path": queries_file.path},
    )
    with opener.open(instructions_file.uri) as instructions_stream:
        instruction_rows = _read_jsonl_records(instructions_stream)
    completed_steps += 1
    report_progress(
        progress_reporter,
        stage="raw_to_normalized",
        current=completed_steps,
        total=total_steps,
        message="Loaded raw instruction records",
        metadata={
            "kind": "instructions",
            "record_count": len(instruction_rows),
            "path": instructions_file.path,
        },
    )
    with opener.open(qrels_file.uri) as qrels_stream:
        qrel_rows = _read_tsv_rows(qrels_stream)
    completed_steps += 1
    report_progress(
        progress_reporter,
        stage="raw_to_normalized",
        current=completed_steps,
        total=total_steps,
        message="Loaded raw qrel rows",
        metadata={"kind": "qrels", "record_count": len(qrel_rows), "path": qrels_file.path},
    )

    instructions_by_query_id = {
        str(row["query-id"]): str(row["instruction"])
        for row in instruction_rows
    }

    return NormalizedDataset(
        corpus=[
            CorpusRecord(
                doc_id=str(row["_id"]),
                title=str(row["title"]) if row.get("title") is not None else None,
                text=str(row["text"]),
            )
            for row in corpus_rows
        ],
        queries=[
            QueryRecord(
                query_id=str(row["_id"]),
                text=str(row["text"]),
                metadata={
                    "instruction": instructions_by_query_id[str(row["_id"])]
                }
                if str(row["_id"]) in instructions_by_query_id
                else {},
            )
            for row in query_rows
        ],
        qrels=[
            QrelRecord(
                query_id=str(row["query-id"]),
                doc_id=str(row["corpus-id"]),
                relevance=float(row["score"]),
            )
            for row in qrel_rows
        ],
    )


def _resolve_normalizer_name(config: RawToNormalizedConfig) -> str:
    if config.normalizer_name is not None:
        return config.normalizer_name
    if config.dataset_name == "IFIRNFCorpus":
        return _IFIR_NFCORPUS_NORMALIZER
    raise RawNormalizeError(f"No default raw normalizer for dataset_name={config.dataset_name!r}")


def normalize_raw_dataset_artifact(
    source_store: ArtifactStore,
    output_store: ArtifactStore,
    config: RawToNormalizedConfig,
    *,
    opener: RawFileOpener,
    progress_reporter: ProgressReporter | None = None,
) -> ArtifactManifest:
    """Normalize one raw snapshot artifact into a normalized dataset artifact."""
    snapshot = read_raw_dataset_artifact(source_store, config.source_artifact_id)
    normalizer_name = _resolve_normalizer_name(config)

    if normalizer_name != _IFIR_NFCORPUS_NORMALIZER:
        raise RawNormalizeError(f"Unsupported raw normalizer: {normalizer_name}")

    dataset = _load_ifir_nfcorpus_dataset(
        snapshot.files,
        opener,
        progress_reporter=progress_reporter,
    )
    raw_source_uri = snapshot.source_uri
    if snapshot.files:
        prefix = "qrels/test.tsv"
        for candidate in snapshot.files:
            if PurePosixPath(candidate.path).as_posix() == prefix:
                raw_source_uri = candidate.uri.rsplit("/qrels/test.tsv", 1)[0]
                break

    normalized_metadata: dict[str, Any] = {}
    normalized_metadata.update(config.metadata)
    normalized_metadata.update(
        {
            "source": "raw_dataset",
            "task_name": config.dataset_name,
            "split": config.split,
            "normalizer_name": normalizer_name,
            "raw_dataset_artifact_id": config.source_artifact_id,
            "raw_dataset_fingerprint": snapshot.content_fingerprint_sha256,
            "raw_source_uri": raw_source_uri,
            "normalized_schema_version": _NORMALIZED_SCHEMA_VERSION,
        }
    )
    dataset.metadata.update(normalized_metadata)

    return write_normalized_dataset_artifact(
        output_store,
        config.output_artifact_id,
        dataset,
        created_by=config.created_by,
        code_git_sha=config.code_git_sha,
        dependencies=[
            ArtifactDependency(
                artifact_type=RAW_DATASET_ARTIFACT_TYPE,
                artifact_id=config.source_artifact_id,
            )
        ],
    )
