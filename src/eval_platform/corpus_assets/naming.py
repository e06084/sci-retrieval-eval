"""Naming helpers for corpus asset planning."""

from __future__ import annotations

from eval_platform.artifacts.types import (
    CHUNKED_CORPUS_ARTIFACT_TYPE,
    CORPUS_ASSET_STAGE_ORDER,
    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
    EMBEDDINGS_ARTIFACT_TYPE,
    MILVUS_COLLECTION_ARTIFACT_TYPE,
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    RAW_DATASET_ARTIFACT_TYPE,
)
from eval_platform.corpus_assets.registry import CorpusAssetError, DatasetSpec

ARTIFACT_STAGE_ORDER = CORPUS_ASSET_STAGE_ORDER

STAGE_SUFFIX = {
    RAW_DATASET_ARTIFACT_TYPE: "raw",
    NORMALIZED_DATASET_ARTIFACT_TYPE: "normalized",
    CHUNKED_CORPUS_ARTIFACT_TYPE: "chunks",
    EMBEDDINGS_ARTIFACT_TYPE: "embeddings",
    ELASTICSEARCH_INDEX_ARTIFACT_TYPE: "es_index",
    MILVUS_COLLECTION_ARTIFACT_TYPE: "milvus_collection",
}


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

    return "/".join(
        part.strip("/") for part in (raw_prefix, spec.raw_dir) if part.strip("/")
    )


def raw_prefix_uri(bucket: str, raw_prefix: str, spec: DatasetSpec) -> str:
    """Return the immutable raw prefix URI for a dataset."""

    return s3_uri(bucket, raw_prefix_key(raw_prefix, spec))
