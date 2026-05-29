"""Raw snapshot to normalized dataset conversion helpers."""

from __future__ import annotations

import csv
import io
import json
from pathlib import PurePosixPath
from typing import Any, BinaryIO, Literal, Protocol

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest
from eval_platform.artifacts.metadata_keys import METADATA_KEY_ASSET_FINGERPRINT_SHA256
from eval_platform.artifacts.store import ArtifactStore
from eval_platform.assets import manifest_asset_fingerprint_sha256
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


class RawNormalizerSpec(BaseModel):
    """Explicit raw normalizer registration for one dataset."""

    dataset_name: str
    normalizer_name: str
    raw_format: Literal["jsonl_tsv", "parquet_dir_shards"]
    has_instructions: bool = False


RAW_NORMALIZER_SPECS: dict[str, RawNormalizerSpec] = {
    "IFIRNFCorpus": RawNormalizerSpec(
        dataset_name="IFIRNFCorpus",
        normalizer_name="ifir_nfcorpus_raw_jsonl_tsv_v1",
        raw_format="jsonl_tsv",
        has_instructions=True,
    ),
    "IFIRScifact": RawNormalizerSpec(
        dataset_name="IFIRScifact",
        normalizer_name="ifir_scifact_raw_jsonl_tsv_v1",
        raw_format="jsonl_tsv",
        has_instructions=True,
    ),
    "NFCorpus": RawNormalizerSpec(
        dataset_name="NFCorpus",
        normalizer_name="nfcorpus_raw_jsonl_tsv_v1",
        raw_format="jsonl_tsv",
    ),
    "SciFact": RawNormalizerSpec(
        dataset_name="SciFact",
        normalizer_name="scifact_raw_jsonl_tsv_v1",
        raw_format="jsonl_tsv",
    ),
    "LitSearchRetrieval": RawNormalizerSpec(
        dataset_name="LitSearchRetrieval",
        normalizer_name="litsearch_raw_parquet_v1",
        raw_format="parquet_dir_shards",
    ),
}
SUPPORTED_RAW_NORMALIZER_DATASET_NAMES = frozenset(RAW_NORMALIZER_SPECS)


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


def _rows_to_corpus(corpus_rows: list[dict[str, Any]]) -> list[CorpusRecord]:
    return [
        CorpusRecord(
            doc_id=str(row["_id"]),
            title=str(row["title"]) if row.get("title") is not None else None,
            text=str(row["text"]),
        )
        for row in corpus_rows
    ]


def _first_non_empty_string(row: dict[str, Any], field_names: tuple[str, ...]) -> str | None:
    for field_name in field_names:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _rows_to_litsearch_corpus(
    corpus_rows: list[dict[str, Any]],
) -> tuple[list[CorpusRecord], set[str], int]:
    corpus: list[CorpusRecord] = []
    retained_doc_ids: set[str] = set()
    dropped_count = 0

    for row in corpus_rows:
        text = _first_non_empty_string(row, ("text", "abstract", "title"))
        if text is None:
            dropped_count += 1
            continue

        doc_id = str(row["_id"])
        title = row.get("title")
        corpus.append(
            CorpusRecord(
                doc_id=doc_id,
                title=title if isinstance(title, str) else None,
                text=text,
            )
        )
        retained_doc_ids.add(doc_id)

    return corpus, retained_doc_ids, dropped_count


def _rows_to_queries(
    query_rows: list[dict[str, Any]],
    *,
    instructions_by_query_id: dict[str, str] | None = None,
) -> list[QueryRecord]:
    instructions = instructions_by_query_id or {}
    return [
        QueryRecord(
            query_id=str(row["_id"]),
            text=str(row["text"]),
            metadata={"instruction": instructions[str(row["_id"])]}
            if str(row["_id"]) in instructions
            else {},
        )
        for row in query_rows
    ]


def _rows_to_qrels(qrel_rows: list[dict[str, Any]]) -> list[QrelRecord]:
    return [
        QrelRecord(
            query_id=str(row["query-id"]),
            doc_id=str(row["corpus-id"]),
            relevance=float(row["score"]),
        )
        for row in qrel_rows
    ]


def _filter_litsearch_rows(
    *,
    query_rows: list[dict[str, Any]],
    qrel_rows: list[dict[str, Any]],
    retained_doc_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    doc_filtered_qrels = [
        row for row in qrel_rows if str(row["corpus-id"]) in retained_doc_ids
    ]
    qrel_query_ids = {str(row["query-id"]) for row in doc_filtered_qrels}
    filtered_query_rows = [
        row for row in query_rows if str(row["_id"]) in qrel_query_ids
    ]
    retained_query_ids = {str(row["_id"]) for row in filtered_query_rows}
    filtered_qrel_rows = [
        row for row in doc_filtered_qrels if str(row["query-id"]) in retained_query_ids
    ]
    return (
        filtered_query_rows,
        filtered_qrel_rows,
        len(query_rows) - len(filtered_query_rows),
        len(qrel_rows) - len(filtered_qrel_rows),
    )


def _load_jsonl_tsv_dataset(
    snapshot_files: list[RawDatasetFile],
    opener: RawFileOpener,
    spec: RawNormalizerSpec,
    *,
    progress_reporter: ProgressReporter | None = None,
) -> NormalizedDataset:
    corpus_file = _find_required_file(snapshot_files, "corpus.jsonl")
    queries_file = _find_required_file(snapshot_files, "queries.jsonl")
    instructions_file = (
        _find_required_file(snapshot_files, "instructions.jsonl")
        if spec.has_instructions
        else None
    )
    qrels_file = _find_required_file(snapshot_files, "qrels/test.tsv")

    completed_steps = 0
    total_steps = 4 if spec.has_instructions else 3

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
    instruction_rows: list[dict[str, Any]] = []
    if instructions_file is not None:
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
        corpus=_rows_to_corpus(corpus_rows),
        queries=_rows_to_queries(query_rows, instructions_by_query_id=instructions_by_query_id),
        qrels=_rows_to_qrels(qrel_rows),
    )


def _read_parquet_records(file: RawDatasetFile, opener: RawFileOpener) -> list[dict[str, Any]]:
    with opener.open(file.uri) as stream:
        payload = stream.read()

    try:
        import pandas as pd  # type: ignore[import-not-found]

        return list(pd.read_parquet(io.BytesIO(payload)).to_dict(orient="records"))
    except ImportError:
        pass

    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RawNormalizeError(
            "pandas or pyarrow is required to normalize parquet raw datasets"
        ) from exc

    return list(pq.read_table(io.BytesIO(payload)).to_pylist())


def _find_parquet_shard_files(
    files: list[RawDatasetFile],
    directory: str,
) -> list[RawDatasetFile]:
    normalized_directory = PurePosixPath(directory).as_posix()
    shard_files = [
        file
        for file in files
        if PurePosixPath(file.path).parent.as_posix() == normalized_directory
        and PurePosixPath(file.path).suffix == ".parquet"
    ]
    if not shard_files:
        raise RawNormalizeError(
            f"Required raw parquet shards missing from snapshot: {normalized_directory}/*.parquet"
        )
    return sorted(shard_files, key=lambda file: PurePosixPath(file.path).as_posix())


def _read_parquet_shard_rows(
    shard_files: list[RawDatasetFile],
    opener: RawFileOpener,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shard_file in shard_files:
        rows.extend(_read_parquet_records(shard_file, opener))
    return rows


def _load_parquet_dataset(
    snapshot_files: list[RawDatasetFile],
    opener: RawFileOpener,
    *,
    progress_reporter: ProgressReporter | None = None,
) -> NormalizedDataset:
    corpus_files = _find_parquet_shard_files(snapshot_files, "corpus")
    queries_files = _find_parquet_shard_files(snapshot_files, "queries")
    qrels_files = _find_parquet_shard_files(snapshot_files, "qrels")
    total_steps = 3

    corpus_rows = _read_parquet_shard_rows(corpus_files, opener)
    report_progress(
        progress_reporter,
        stage="raw_to_normalized",
        current=1,
        total=total_steps,
        message="Loaded raw corpus records",
        metadata={
            "kind": "corpus",
            "record_count": len(corpus_rows),
            "path": "corpus/*.parquet",
            "shard_count": len(corpus_files),
            "shard_paths": [file.path for file in corpus_files],
        },
    )
    query_rows = _read_parquet_shard_rows(queries_files, opener)
    report_progress(
        progress_reporter,
        stage="raw_to_normalized",
        current=2,
        total=total_steps,
        message="Loaded raw query records",
        metadata={
            "kind": "queries",
            "record_count": len(query_rows),
            "path": "queries/*.parquet",
            "shard_count": len(queries_files),
            "shard_paths": [file.path for file in queries_files],
        },
    )
    qrel_rows = _read_parquet_shard_rows(qrels_files, opener)
    report_progress(
        progress_reporter,
        stage="raw_to_normalized",
        current=3,
        total=total_steps,
        message="Loaded raw qrel rows",
        metadata={
            "kind": "qrels",
            "record_count": len(qrel_rows),
            "path": "qrels/*.parquet",
            "shard_count": len(qrels_files),
            "shard_paths": [file.path for file in qrels_files],
        },
    )
    corpus, retained_doc_ids, dropped_corpus_count = _rows_to_litsearch_corpus(corpus_rows)
    filtered_query_rows, filtered_qrel_rows, dropped_query_count, dropped_qrel_count = (
        _filter_litsearch_rows(
            query_rows=query_rows,
            qrel_rows=qrel_rows,
            retained_doc_ids=retained_doc_ids,
        )
    )
    metadata: dict[str, Any] = {}
    if dropped_corpus_count or dropped_query_count or dropped_qrel_count:
        metadata.update(
            {
                "filtered_corpus_count": len(corpus),
                "dropped_corpus_count": dropped_corpus_count,
                "dropped_qrel_count": dropped_qrel_count,
                "dropped_query_count": dropped_query_count,
            }
        )

    return NormalizedDataset(
        corpus=corpus,
        queries=_rows_to_queries(filtered_query_rows),
        qrels=_rows_to_qrels(filtered_qrel_rows),
        metadata=metadata,
    )


def _resolve_raw_normalizer_spec(config: RawToNormalizedConfig) -> RawNormalizerSpec:
    spec = RAW_NORMALIZER_SPECS.get(config.dataset_name)
    if spec is None:
        raise RawNormalizeError(f"No raw normalizer for dataset_name={config.dataset_name!r}")
    if config.normalizer_name is not None and config.normalizer_name != spec.normalizer_name:
        raise RawNormalizeError(
            "Raw normalizer mismatch for dataset "
            f"{config.dataset_name!r}: expected {spec.normalizer_name!r}, "
            f"got {config.normalizer_name!r}"
        )
    return spec


def _resolve_raw_source_uri(
    snapshot_files: list[RawDatasetFile],
    snapshot_uri: str,
    spec: RawNormalizerSpec,
) -> str:
    if spec.raw_format == "jsonl_tsv":
        target_paths = ["qrels/test.tsv"]
    elif spec.raw_format == "parquet_dir_shards":
        target_paths = [
            PurePosixPath(file.path).as_posix()
            for file in _find_parquet_shard_files(snapshot_files, "qrels")
        ]
    else:
        target_paths = []

    for candidate in snapshot_files:
        candidate_path = PurePosixPath(candidate.path).as_posix()
        if candidate_path in target_paths:
            return candidate.uri.rsplit(f"/{candidate_path}", 1)[0]
    return snapshot_uri


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
    raw_manifest = source_store.read_manifest(
        RAW_DATASET_ARTIFACT_TYPE,
        config.source_artifact_id,
    )
    spec = _resolve_raw_normalizer_spec(config)

    if spec.raw_format == "jsonl_tsv":
        dataset = _load_jsonl_tsv_dataset(
            snapshot.files,
            opener,
            spec,
            progress_reporter=progress_reporter,
        )
    elif spec.raw_format == "parquet_dir_shards":
        dataset = _load_parquet_dataset(
            snapshot.files,
            opener,
            progress_reporter=progress_reporter,
        )
    else:
        raise RawNormalizeError(f"Unsupported raw format: {spec.raw_format}")

    raw_source_uri = _resolve_raw_source_uri(snapshot.files, snapshot.source_uri, spec)

    normalized_metadata: dict[str, Any] = {}
    normalized_metadata.update(config.metadata)
    normalized_metadata.update(
        {
            "source": "raw_dataset",
            "task_name": config.dataset_name,
            "split": config.split,
            "normalizer_name": spec.normalizer_name,
            "normalizer_version": "1",
            "normalizer_params": {
                "split": config.split,
                "raw_format": spec.raw_format,
                "has_instructions": spec.has_instructions,
            },
            "raw_format": spec.raw_format,
            "has_instructions": spec.has_instructions,
            "raw_dataset_artifact_id": config.source_artifact_id,
            "raw_dataset_fingerprint": snapshot.content_fingerprint_sha256,
            "raw_dataset_asset_fingerprint_sha256": (
                manifest_asset_fingerprint_sha256(raw_manifest)
                or raw_manifest.metadata.get(METADATA_KEY_ASSET_FINGERPRINT_SHA256)
                or snapshot.content_fingerprint_sha256
            ),
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
