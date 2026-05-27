"""Dry-run corpus asset planning."""

from __future__ import annotations

from typing import Any

from eval_platform.corpus_assets.naming import (
    ARTIFACT_STAGE_ORDER,
    artifact_ids_for_dataset,
    collection_name_for_dataset,
    index_name_for_dataset,
    raw_prefix_uri,
    s3_uri,
)
from eval_platform.corpus_assets.registry import CorpusAssetError, DatasetSpec

_DEPENDENCY_METADATA_KEYS = {
    "raw_dataset": ("raw_dataset_artifact_id",),
    "normalized_dataset": ("source_normalized_dataset_artifact_id",),
    "chunked_corpus": (
        "source_chunked_corpus_artifact_id",
        "chunked_corpus_artifact_id",
    ),
    "embeddings": ("source_embeddings_artifact_id", "embeddings_artifact_id"),
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
                    step["source_artifact_id"] = resolved_artifact_ids[
                        "chunked_corpus"
                    ]
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
