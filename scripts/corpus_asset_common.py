"""Shared helpers for five-dataset corpus asset inventory and dry-run planning."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eval_platform.artifacts import ArtifactManifest, ArtifactStore, S3ArtifactStore
from eval_platform.artifacts.store import SUCCESS_MARKER
from eval_platform.config import PlatformConfig, dump_redacted_config, load_platform_config

ARTIFACT_STAGE_ORDER = [
    "raw_dataset",
    "normalized_dataset",
    "chunked_corpus",
    "embeddings",
    "elasticsearch_index",
    "milvus_collection",
]

STAGE_SUFFIX = {
    "raw_dataset": "raw",
    "normalized_dataset": "normalized",
    "chunked_corpus": "chunks",
    "embeddings": "embeddings",
    "elasticsearch_index": "es_index",
    "milvus_collection": "milvus_collection",
}

_SENSITIVE_KEY_PARTS = (
    "access_key",
    "api_key",
    "authorization",
    "password",
    "secret",
    "token",
)

_MANIFEST_SUMMARY_KEYS = {
    "dataset",
    "dataset_name",
    "task_name",
    "split",
    "normalizer_name",
    "corpus_count",
    "query_count",
    "qrel_count",
    "chunk_count",
    "embedding_count",
    "unique_chunk_count",
    "unique_doc_count",
    "embedding_dim",
    "source_artifact_id",
    "raw_dataset_artifact_id",
    "source_normalized_dataset_artifact_id",
    "source_chunked_corpus_artifact_id",
    "source_embeddings_artifact_id",
    "chunked_corpus_artifact_id",
    "embeddings_artifact_id",
    "index_name",
    "collection_name",
    "indexed_count",
    "inserted_count",
    "failed_count",
    "verified_count",
    "verified_document_count",
    "verified_entity_count",
}

_DEPENDENCY_METADATA_KEYS = {
    "raw_dataset": ("raw_dataset_artifact_id",),
    "normalized_dataset": ("source_normalized_dataset_artifact_id",),
    "chunked_corpus": (
        "source_chunked_corpus_artifact_id",
        "chunked_corpus_artifact_id",
    ),
    "embeddings": ("source_embeddings_artifact_id", "embeddings_artifact_id"),
}


class CorpusAssetError(Exception):
    """Raised when corpus asset inventory or planning fails."""


@dataclass(frozen=True)
class DatasetSpec:
    """One target dataset and its immutable raw S3 layout."""

    task_name: str
    slug: str
    raw_dir: str
    raw_format: str
    expected_raw_files: tuple[str, ...]
    notes: str


TARGET_DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        task_name="IFIRNFCorpus",
        slug="ifir_nfcorpus",
        raw_dir="ifir_nfcorpus",
        raw_format="jsonl_tsv",
        expected_raw_files=(
            "corpus.jsonl",
            "queries.jsonl",
            "instructions.jsonl",
            "qrels/test.tsv",
        ),
        notes="IFIR layout with query instructions.",
    ),
    DatasetSpec(
        task_name="NFCorpus",
        slug="nfcorpus",
        raw_dir="nfcorpus",
        raw_format="jsonl_tsv",
        expected_raw_files=("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"),
        notes="BEIR-style JSONL corpus/queries plus test qrels TSV.",
    ),
    DatasetSpec(
        task_name="IFIRScifact",
        slug="ifir_scifact",
        raw_dir="ifir_scifact",
        raw_format="jsonl_tsv",
        expected_raw_files=(
            "corpus.jsonl",
            "queries.jsonl",
            "instructions.jsonl",
            "qrels/test.tsv",
        ),
        notes="IFIR layout with query instructions.",
    ),
    DatasetSpec(
        task_name="SciFact",
        slug="scifact",
        raw_dir="scifact",
        raw_format="jsonl_tsv",
        expected_raw_files=("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"),
        notes="BEIR-style JSONL corpus/queries plus test qrels TSV.",
    ),
    DatasetSpec(
        task_name="LitSearchRetrieval",
        slug="litsearch",
        raw_dir="litsearch",
        raw_format="parquet_dir_shards",
        expected_raw_files=(
            "corpus/test-00000-of-00001.parquet",
            "queries/test-00000-of-00001.parquet",
            "qrels/test-00000-of-00001.parquet",
        ),
        notes="MTEB parquet shard layout.",
    ),
)

DATASETS_BY_NAME = {spec.task_name: spec for spec in TARGET_DATASETS}
DATASETS_BY_SLUG = {spec.slug: spec for spec in TARGET_DATASETS}


def dataset_specs_for_selection(selection: str) -> list[DatasetSpec]:
    """Return target dataset specs for one CLI selection."""

    if selection == "all":
        return list(TARGET_DATASETS)
    if selection in DATASETS_BY_NAME:
        return [DATASETS_BY_NAME[selection]]
    if selection in DATASETS_BY_SLUG:
        return [DATASETS_BY_SLUG[selection]]
    valid = sorted([spec.task_name for spec in TARGET_DATASETS] + ["all"])
    raise CorpusAssetError(f"Unknown dataset {selection!r}; expected one of {valid}")


def artifact_ids_for_dataset(spec: DatasetSpec, run_id: str) -> dict[str, str]:
    """Generate stable artifact ids for one dataset/run."""

    if not run_id.strip():
        raise CorpusAssetError("run_id must not be empty")
    return {
        artifact_type: f"{spec.slug}_{run_id}_{suffix}"
        for artifact_type, suffix in STAGE_SUFFIX.items()
    }


def index_name_for_dataset(spec: DatasetSpec, run_id: str) -> str:
    """Generate the Elasticsearch index name for one dataset/run."""

    return f"{spec.slug}_{run_id}_es"


def collection_name_for_dataset(spec: DatasetSpec, run_id: str) -> str:
    """Generate the Milvus collection name for one dataset/run."""

    return f"{spec.slug}_{run_id}_milvus"


def s3_uri(bucket: str, *parts: str) -> str:
    """Build an s3:// URI from a bucket and key parts."""

    key = "/".join(part.strip("/") for part in parts if part.strip("/"))
    return f"s3://{bucket}/{key}" if key else f"s3://{bucket}"


def raw_prefix_key(raw_prefix: str, spec: DatasetSpec) -> str:
    """Return the immutable raw prefix key for a dataset."""

    return "/".join(part.strip("/") for part in (raw_prefix, spec.raw_dir) if part.strip("/"))


def raw_prefix_uri(bucket: str, raw_prefix: str, spec: DatasetSpec) -> str:
    """Return the immutable raw prefix URI for a dataset."""

    return s3_uri(bucket, raw_prefix_key(raw_prefix, spec))


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
                out[key] = "***"
            else:
                out[key] = _redact(item)
        return out
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def safe_json_dumps(payload: Any) -> str:
    """Serialize a payload after redacting sensitive keys."""

    return json.dumps(_redact(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def make_s3_client(config: PlatformConfig) -> Any:
    """Create a boto3 S3 client from platform config."""

    try:
        import boto3
    except ImportError as exc:
        raise CorpusAssetError(
            "boto3 is required for real S3 inventory; install the s3 extra"
        ) from exc
    return boto3.client(
        "s3",
        endpoint_url=config.s3.endpoint,
        aws_access_key_id=config.s3.access_key_id,
        aws_secret_access_key=config.s3.secret_access_key,
    )


def make_s3_artifact_store(
    *,
    config: PlatformConfig,
    s3_prefix: str,
    client: Any,
) -> S3ArtifactStore:
    """Create the artifact store for a target S3 prefix."""

    if not config.s3.bucket:
        raise CorpusAssetError("config.s3.bucket is required")
    return S3ArtifactStore(
        bucket=config.s3.bucket,
        prefix=s3_prefix.strip("/"),
        client=client,
    )


def raw_prefix_exists(client: Any, *, bucket: str, prefix: str) -> bool:
    """Return whether a raw S3 prefix contains at least one object."""

    list_prefix = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
    response = client.list_objects_v2(Bucket=bucket, Prefix=list_prefix, MaxKeys=1)
    return bool(response.get("Contents"))


def _manifest_metadata_summary(manifest: ArtifactManifest) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in _MANIFEST_SUMMARY_KEYS:
        if key in manifest.metadata:
            summary[key] = manifest.metadata[key]
    if manifest.dependencies:
        summary["dependencies"] = [
            dependency.model_dump(mode="json")
            for dependency in manifest.dependencies
        ]
    return _redact(summary)


def _manifest_dataset_match(manifest: ArtifactManifest, spec: DatasetSpec) -> bool:
    metadata = manifest.metadata
    candidates = {
        str(metadata.get("dataset", "")),
        str(metadata.get("dataset_name", "")),
        str(metadata.get("task_name", "")),
        str(metadata.get("dataset_slug", "")),
    }
    if spec.task_name in candidates or spec.slug in candidates:
        return True
    return manifest.artifact_id == spec.slug or manifest.artifact_id.startswith(f"{spec.slug}_")


def inventory_corpus_assets(
    *,
    store: ArtifactStore,
    raw_client: Any,
    bucket: str,
    raw_prefix: str,
    datasets: list[DatasetSpec] | None = None,
) -> dict[str, Any]:
    """Inventory raw prefixes and corpus/index artifacts for target datasets."""

    selected = datasets or list(TARGET_DATASETS)
    artifact_manifests: dict[str, list[ArtifactManifest]] = {}
    for artifact_type in ARTIFACT_STAGE_ORDER:
        manifests: list[ArtifactManifest] = []
        for _current_type, artifact_id in store.list_artifacts(artifact_type):
            try:
                manifests.append(store.read_manifest(artifact_type, artifact_id))
            except Exception:
                # Incomplete or corrupt manifests are represented through the artifact record.
                manifests.append(
                    ArtifactManifest(
                        artifact_id=artifact_id,
                        artifact_type=artifact_type,
                        created_at=datetime.fromtimestamp(0, tz=UTC),
                        metadata={"manifest_read_error": True},
                    )
                )
        artifact_manifests[artifact_type] = manifests

    dataset_results: dict[str, Any] = {}
    for spec in selected:
        prefix_key = raw_prefix_key(raw_prefix, spec)
        raw_exists = raw_prefix_exists(raw_client, bucket=bucket, prefix=prefix_key)
        artifacts_by_type: dict[str, list[dict[str, Any]]] = {}
        missing: list[str] = []

        for artifact_type in ARTIFACT_STAGE_ORDER:
            records: list[dict[str, Any]] = []
            for manifest in artifact_manifests.get(artifact_type, []):
                if not _manifest_dataset_match(manifest, spec):
                    continue
                complete = store.is_complete(artifact_type, manifest.artifact_id)
                records.append(
                    {
                        "artifact_id": manifest.artifact_id,
                        "complete": complete,
                        "has_manifest": not manifest.metadata.get("manifest_read_error", False),
                        "has_success": store.exists(
                            artifact_type,
                            manifest.artifact_id,
                            SUCCESS_MARKER,
                        ),
                        "artifact_uri": store.artifact_uri(artifact_type, manifest.artifact_id),
                        "metadata_summary": _manifest_metadata_summary(manifest),
                    }
                )
            artifacts_by_type[artifact_type] = sorted(
                records,
                key=lambda item: str(item["artifact_id"]),
            )
            if not any(record["complete"] for record in records):
                missing.append(artifact_type)

        if not raw_exists:
            missing.insert(0, "raw_prefix")

        dataset_results[spec.task_name] = {
            "slug": spec.slug,
            "raw_format": spec.raw_format,
            "raw_prefix": s3_uri(bucket, prefix_key),
            "raw_prefix_exists": raw_exists,
            "expected_raw_files": list(spec.expected_raw_files),
            "artifacts": artifacts_by_type,
            "missing": missing,
        }

    return {
        "datasets": dataset_results,
        "artifact_stage_order": ARTIFACT_STAGE_ORDER,
    }


def build_plan_for_datasets(
    *,
    datasets: list[DatasetSpec],
    run_id: str,
    bucket: str,
    raw_prefix: str,
    s3_prefix: str,
    raw_exists_by_slug: dict[str, bool],
    reuse_existing: bool = False,
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a dry-run plan for corpus/index artifacts."""

    dataset_plans: dict[str, Any] = {}
    for spec in datasets:
        if not raw_exists_by_slug.get(spec.slug, False):
            raise CorpusAssetError(f"Raw prefix does not exist for {spec.task_name}")

        generated_artifact_ids = artifact_ids_for_dataset(spec, run_id)
        complete_records = _complete_inventory_records(inventory, spec.task_name)
        records_by_id = _records_by_id(complete_records)
        reuse_artifact_ids = (
            _resolve_reusable_artifact_chain(records_by_id)
            if reuse_existing and inventory is not None
            else {}
        )
        resolved_artifact_ids: dict[str, str] = {}
        generated_resource_names = {
            "elasticsearch_index": index_name_for_dataset(spec, run_id),
            "milvus_collection": collection_name_for_dataset(spec, run_id),
        }
        resolved_resource_names: dict[str, str] = {}
        steps: list[dict[str, Any]] = []
        source_artifact_id: str | None = None

        for artifact_type in ARTIFACT_STAGE_ORDER:
            artifact_id = reuse_artifact_ids.get(
                artifact_type,
                generated_artifact_ids[artifact_type],
            )
            action = "reuse" if artifact_type in reuse_artifact_ids else "create"
            resolved_artifact_ids[artifact_type] = artifact_id
            reused_record = (
                records_by_id.get(artifact_type, {}).get(artifact_id)
                if action == "reuse"
                else None
            )

            step: dict[str, Any] = {
                "stage": artifact_type,
                "action": action,
                "artifact_type": artifact_type,
                "artifact_id": artifact_id,
                "artifact_uri": s3_uri(bucket, s3_prefix, artifact_type, artifact_id),
            }
            if source_artifact_id is not None:
                step["source_artifact_id"] = source_artifact_id
            if artifact_type == "raw_dataset":
                step["raw_source_uri"] = raw_prefix_uri(bucket, raw_prefix, spec)
            if artifact_type == "elasticsearch_index":
                if reused_record is not None:
                    step["index_name"] = _required_metadata_value(
                        reused_record,
                        "index_name",
                    )
                    step["source_artifact_id"] = _dependency_id(
                        reused_record,
                        "chunked_corpus",
                    )
                else:
                    step["index_name"] = generated_resource_names[
                        "elasticsearch_index"
                    ]
                    step["source_artifact_id"] = resolved_artifact_ids["chunked_corpus"]
                resolved_resource_names["elasticsearch_index"] = step["index_name"]
            if artifact_type == "milvus_collection":
                step.pop("source_artifact_id", None)
                if reused_record is not None:
                    step["collection_name"] = _required_metadata_value(
                        reused_record,
                        "collection_name",
                    )
                    step["chunked_corpus_artifact_id"] = _dependency_id(
                        reused_record,
                        "chunked_corpus",
                    )
                    step["embeddings_artifact_id"] = _dependency_id(
                        reused_record,
                        "embeddings",
                    )
                else:
                    step["chunked_corpus_artifact_id"] = resolved_artifact_ids[
                        "chunked_corpus"
                    ]
                    step["embeddings_artifact_id"] = resolved_artifact_ids["embeddings"]
                    step["collection_name"] = generated_resource_names[
                        "milvus_collection"
                    ]
                resolved_resource_names["milvus_collection"] = step["collection_name"]
            steps.append(step)
            source_artifact_id = artifact_id

        dataset_plans[spec.task_name] = {
            "slug": spec.slug,
            "raw_format": spec.raw_format,
            "artifact_ids": generated_artifact_ids,
            "generated_artifact_ids": generated_artifact_ids,
            "resolved_artifact_ids": resolved_artifact_ids,
            "generated_resource_names": generated_resource_names,
            "resolved_resource_names": resolved_resource_names,
            "elasticsearch_index_name": resolved_resource_names["elasticsearch_index"],
            "milvus_collection_name": resolved_resource_names["milvus_collection"],
            "steps": steps,
        }

    return {
        "mode": "dry_run",
        "run_id": run_id,
        "s3_prefix": s3_prefix,
        "reuse_existing": reuse_existing,
        "datasets": dataset_plans,
    }


def _complete_inventory_records(
    inventory: dict[str, Any] | None,
    task_name: str,
) -> dict[str, list[dict[str, Any]]]:
    if inventory is None:
        return {artifact_type: [] for artifact_type in ARTIFACT_STAGE_ORDER}
    dataset = inventory.get("datasets", {}).get(task_name, {})
    return {
        artifact_type: [
            record
            for record in dataset.get("artifacts", {}).get(artifact_type, [])
            if record.get("complete")
        ]
        for artifact_type in ARTIFACT_STAGE_ORDER
    }


def _records_by_id(
    records_by_type: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        artifact_type: {
            str(record["artifact_id"]): record
            for record in records
            if record.get("artifact_id") is not None
        }
        for artifact_type, records in records_by_type.items()
    }


def _resolve_reusable_artifact_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, str]:
    """Resolve one dependency-consistent reusable artifact chain."""

    for record in records_by_id.get("milvus_collection", {}).values():
        chain = _trace_milvus_chain(records_by_id, str(record["artifact_id"]))
        if chain is None:
            continue
        matching_es_id = _find_dependent_record_id(
            records_by_id,
            "elasticsearch_index",
            "chunked_corpus",
            chain["chunked_corpus"],
        )
        if matching_es_id is not None:
            chain["elasticsearch_index"] = matching_es_id
        return chain

    for record in records_by_id.get("elasticsearch_index", {}).values():
        chain = _trace_elasticsearch_chain(records_by_id, str(record["artifact_id"]))
        if chain is None:
            continue
        matching_embedding_id = _find_dependent_record_id(
            records_by_id,
            "embeddings",
            "chunked_corpus",
            chain["chunked_corpus"],
        )
        if matching_embedding_id is not None:
            chain["embeddings"] = matching_embedding_id
            matching_milvus_id = _find_milvus_record_id(
                records_by_id,
                chunked_corpus_id=chain["chunked_corpus"],
                embeddings_id=matching_embedding_id,
            )
            if matching_milvus_id is not None:
                chain["milvus_collection"] = matching_milvus_id
        return chain

    for record in records_by_id.get("embeddings", {}).values():
        chain = _trace_embedding_chain(records_by_id, str(record["artifact_id"]))
        if chain is not None:
            return chain

    for record in records_by_id.get("chunked_corpus", {}).values():
        chain = _trace_chunked_chain(records_by_id, str(record["artifact_id"]))
        if chain is not None:
            return chain

    for record in records_by_id.get("normalized_dataset", {}).values():
        chain = _trace_normalized_chain(records_by_id, str(record["artifact_id"]))
        if chain is not None:
            return chain

    for record in records_by_id.get("raw_dataset", {}).values():
        return {"raw_dataset": str(record["artifact_id"])}

    return {}


def _trace_milvus_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    artifact_id: str,
) -> dict[str, str] | None:
    record = records_by_id.get("milvus_collection", {}).get(artifact_id)
    if record is None:
        return None
    chunked_corpus_id = _dependency_id(record, "chunked_corpus")
    embeddings_id = _dependency_id(record, "embeddings")
    if chunked_corpus_id is None or embeddings_id is None:
        return None
    chunk_chain = _trace_chunked_chain(records_by_id, chunked_corpus_id)
    embedding_chain = _trace_embedding_chain(records_by_id, embeddings_id)
    if chunk_chain is None or embedding_chain is None:
        return None
    if embedding_chain["chunked_corpus"] != chunked_corpus_id:
        return None
    return {
        **chunk_chain,
        "embeddings": embeddings_id,
        "milvus_collection": artifact_id,
    }


def _trace_elasticsearch_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    artifact_id: str,
) -> dict[str, str] | None:
    record = records_by_id.get("elasticsearch_index", {}).get(artifact_id)
    if record is None:
        return None
    chunked_corpus_id = _dependency_id(record, "chunked_corpus")
    if chunked_corpus_id is None:
        return None
    chunk_chain = _trace_chunked_chain(records_by_id, chunked_corpus_id)
    if chunk_chain is None:
        return None
    return {**chunk_chain, "elasticsearch_index": artifact_id}


def _trace_embedding_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    artifact_id: str,
) -> dict[str, str] | None:
    record = records_by_id.get("embeddings", {}).get(artifact_id)
    if record is None:
        return None
    chunked_corpus_id = _dependency_id(record, "chunked_corpus")
    if chunked_corpus_id is None:
        return None
    chunk_chain = _trace_chunked_chain(records_by_id, chunked_corpus_id)
    if chunk_chain is None:
        return None
    return {**chunk_chain, "embeddings": artifact_id}


def _trace_chunked_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    artifact_id: str,
) -> dict[str, str] | None:
    record = records_by_id.get("chunked_corpus", {}).get(artifact_id)
    if record is None:
        return None
    normalized_id = _dependency_id(record, "normalized_dataset")
    if normalized_id is None:
        return None
    normalized_chain = _trace_normalized_chain(records_by_id, normalized_id)
    if normalized_chain is None:
        return None
    return {**normalized_chain, "chunked_corpus": artifact_id}


def _trace_normalized_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    artifact_id: str,
) -> dict[str, str] | None:
    record = records_by_id.get("normalized_dataset", {}).get(artifact_id)
    if record is None:
        return None
    raw_id = _dependency_id(record, "raw_dataset")
    if raw_id is None:
        return None
    if raw_id not in records_by_id.get("raw_dataset", {}):
        return None
    return {"raw_dataset": raw_id, "normalized_dataset": artifact_id}


def _find_dependent_record_id(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    artifact_type: str,
    dependency_type: str,
    dependency_id: str,
) -> str | None:
    for artifact_id, record in records_by_id.get(artifact_type, {}).items():
        if _dependency_id(record, dependency_type) == dependency_id:
            return artifact_id
    return None


def _find_milvus_record_id(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    *,
    chunked_corpus_id: str,
    embeddings_id: str,
) -> str | None:
    for artifact_id, record in records_by_id.get("milvus_collection", {}).items():
        if (
            _dependency_id(record, "chunked_corpus") == chunked_corpus_id
            and _dependency_id(record, "embeddings") == embeddings_id
        ):
            return artifact_id
    return None


def _dependency_id(record: dict[str, Any], artifact_type: str) -> str | None:
    metadata = record.get("metadata_summary", {})
    for dependency in metadata.get("dependencies", []):
        if dependency.get("artifact_type") == artifact_type:
            artifact_id = dependency.get("artifact_id")
            return str(artifact_id) if artifact_id else None
    for key in _DEPENDENCY_METADATA_KEYS.get(artifact_type, ()):
        artifact_id = metadata.get(key)
        if artifact_id:
            return str(artifact_id)
    return None


def _required_metadata_value(record: dict[str, Any], key: str) -> str:
    metadata = record.get("metadata_summary", {})
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value
    if value is not None and str(value).strip():
        return str(value)
    artifact_id = record.get("artifact_id", "<unknown>")
    raise CorpusAssetError(
        f"Reused artifact {artifact_id!r} is missing required metadata {key!r}"
    )


def load_config_and_client(config_path: Path) -> tuple[PlatformConfig, Any]:
    """Load platform config and create an S3 client."""

    config = load_platform_config(config_path)
    return config, make_s3_client(config)


def output_payload(payload: dict[str, Any], output: Path | None) -> None:
    """Print JSON and optionally write it to a local file."""

    text = safe_json_dumps(payload)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add shared config/S3 arguments."""

    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--s3-prefix", default="test_sciverse_benchmark")
    parser.add_argument("--raw-prefix", default="sciverse_benchmark/raw")
    parser.add_argument("--output", type=Path, default=None)


def redacted_config_summary(config: PlatformConfig) -> dict[str, Any]:
    """Return a safe config summary for reports."""

    return dump_redacted_config(config)
