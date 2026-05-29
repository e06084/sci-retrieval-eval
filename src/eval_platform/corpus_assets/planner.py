"""Dry-run corpus asset planning."""

from __future__ import annotations

from typing import Any

from eval_platform.artifacts.metadata_keys import (
    DEPENDENCY_METADATA_KEYS_BY_ARTIFACT_TYPE,
    METADATA_KEY_ASSET_FINGERPRINT_SHA256,
    METADATA_KEY_CHUNKED_CORPUS_ARTIFACT_ID,
    METADATA_KEY_COLLECTION_NAME,
    METADATA_KEY_EMBEDDINGS_ARTIFACT_ID,
    METADATA_KEY_INDEX_NAME,
)
from eval_platform.artifacts.types import (
    CHUNKED_CORPUS_ARTIFACT_TYPE,
    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
    EMBEDDINGS_ARTIFACT_TYPE,
    MILVUS_COLLECTION_ARTIFACT_TYPE,
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    RAW_DATASET_ARTIFACT_TYPE,
)
from eval_platform.corpus_assets.naming import (
    ARTIFACT_STAGE_ORDER,
    artifact_ids_for_dataset,
    collection_name_for_dataset,
    index_name_for_dataset,
    raw_prefix_uri,
    s3_uri,
)
from eval_platform.corpus_assets.registry import CorpusAssetError, DatasetSpec


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
    expected_asset_fingerprints_by_slug: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a dry-run plan for corpus/index artifacts."""

    dataset_plans: dict[str, Any] = {}
    for spec in datasets:
        if not raw_exists_by_slug.get(spec.slug, False):
            raise CorpusAssetError(f"Raw prefix does not exist for {spec.task_name}")

        generated_artifact_ids = artifact_ids_for_dataset(spec, run_id)
        complete_records = _complete_inventory_records(inventory, spec.task_name)
        expected_asset_fingerprints = _expected_fingerprints_for_spec(
            spec,
            expected_asset_fingerprints_by_slug,
        )
        complete_records = _filter_records_by_expected_asset_fingerprints(
            complete_records,
            expected_asset_fingerprints,
        )
        records_by_id = _records_by_id(complete_records)
        reuse_artifact_ids = (
            _resolve_reusable_artifact_chain(records_by_id)
            if reuse_existing and inventory is not None
            else {}
        )
        resolved_artifact_ids: dict[str, str] = {}
        generated_resource_names = {
            ELASTICSEARCH_INDEX_ARTIFACT_TYPE: index_name_for_dataset(spec, run_id),
            MILVUS_COLLECTION_ARTIFACT_TYPE: collection_name_for_dataset(spec, run_id),
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
            if reused_record is not None:
                asset_fingerprint = _record_asset_fingerprint_sha256(reused_record)
                if asset_fingerprint is not None:
                    step[METADATA_KEY_ASSET_FINGERPRINT_SHA256] = asset_fingerprint
            if artifact_type == RAW_DATASET_ARTIFACT_TYPE:
                step["raw_source_uri"] = raw_prefix_uri(bucket, raw_prefix, spec)
            if artifact_type == ELASTICSEARCH_INDEX_ARTIFACT_TYPE:
                if reused_record is not None:
                    step[METADATA_KEY_INDEX_NAME] = _required_metadata_value(
                        reused_record,
                        METADATA_KEY_INDEX_NAME,
                    )
                    step["source_artifact_id"] = _dependency_id(
                        reused_record,
                        CHUNKED_CORPUS_ARTIFACT_TYPE,
                    )
                else:
                    step[METADATA_KEY_INDEX_NAME] = generated_resource_names[
                        ELASTICSEARCH_INDEX_ARTIFACT_TYPE
                    ]
                    step["source_artifact_id"] = resolved_artifact_ids[
                        CHUNKED_CORPUS_ARTIFACT_TYPE
                    ]
                resolved_resource_names[ELASTICSEARCH_INDEX_ARTIFACT_TYPE] = step[
                    METADATA_KEY_INDEX_NAME
                ]
            if artifact_type == MILVUS_COLLECTION_ARTIFACT_TYPE:
                step.pop("source_artifact_id", None)
                if reused_record is not None:
                    step[METADATA_KEY_COLLECTION_NAME] = _required_metadata_value(
                        reused_record,
                        METADATA_KEY_COLLECTION_NAME,
                    )
                    step[METADATA_KEY_CHUNKED_CORPUS_ARTIFACT_ID] = _dependency_id(
                        reused_record,
                        CHUNKED_CORPUS_ARTIFACT_TYPE,
                    )
                    step[METADATA_KEY_EMBEDDINGS_ARTIFACT_ID] = _dependency_id(
                        reused_record,
                        EMBEDDINGS_ARTIFACT_TYPE,
                    )
                else:
                    step[METADATA_KEY_CHUNKED_CORPUS_ARTIFACT_ID] = resolved_artifact_ids[
                        CHUNKED_CORPUS_ARTIFACT_TYPE
                    ]
                    step[METADATA_KEY_EMBEDDINGS_ARTIFACT_ID] = resolved_artifact_ids[
                        EMBEDDINGS_ARTIFACT_TYPE
                    ]
                    step[METADATA_KEY_COLLECTION_NAME] = generated_resource_names[
                        MILVUS_COLLECTION_ARTIFACT_TYPE
                    ]
                resolved_resource_names[MILVUS_COLLECTION_ARTIFACT_TYPE] = step[
                    METADATA_KEY_COLLECTION_NAME
                ]
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
            "expected_asset_fingerprints": expected_asset_fingerprints,
            "elasticsearch_index_name": resolved_resource_names[
                ELASTICSEARCH_INDEX_ARTIFACT_TYPE
            ],
            "milvus_collection_name": resolved_resource_names[
                MILVUS_COLLECTION_ARTIFACT_TYPE
            ],
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


def _expected_fingerprints_for_spec(
    spec: DatasetSpec,
    expected_asset_fingerprints_by_slug: dict[str, dict[str, str]] | None,
) -> dict[str, str]:
    if expected_asset_fingerprints_by_slug is None:
        return {}
    return dict(
        expected_asset_fingerprints_by_slug.get(spec.slug)
        or expected_asset_fingerprints_by_slug.get(spec.task_name)
        or {}
    )


def _filter_records_by_expected_asset_fingerprints(
    records_by_type: dict[str, list[dict[str, Any]]],
    expected_asset_fingerprints: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    if not expected_asset_fingerprints:
        return records_by_type
    filtered: dict[str, list[dict[str, Any]]] = {}
    for artifact_type, records in records_by_type.items():
        expected = expected_asset_fingerprints.get(artifact_type)
        if expected is None:
            filtered[artifact_type] = records
            continue
        filtered[artifact_type] = [
            record
            for record in records
            if _record_asset_fingerprint_sha256(record) == expected
        ]
    return filtered


def _record_asset_fingerprint_sha256(record: dict[str, Any]) -> str | None:
    metadata = record.get("metadata_summary", {})
    value = metadata.get(METADATA_KEY_ASSET_FINGERPRINT_SHA256)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _resolve_reusable_artifact_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, str]:
    """Resolve one dependency-consistent reusable artifact chain."""

    for record in records_by_id.get(MILVUS_COLLECTION_ARTIFACT_TYPE, {}).values():
        chain = _trace_milvus_chain(records_by_id, str(record["artifact_id"]))
        if chain is None:
            continue
        matching_es_id = _find_dependent_record_id(
            records_by_id,
            ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
            CHUNKED_CORPUS_ARTIFACT_TYPE,
            chain[CHUNKED_CORPUS_ARTIFACT_TYPE],
        )
        if matching_es_id is not None:
            chain[ELASTICSEARCH_INDEX_ARTIFACT_TYPE] = matching_es_id
        return chain

    for record in records_by_id.get(ELASTICSEARCH_INDEX_ARTIFACT_TYPE, {}).values():
        chain = _trace_elasticsearch_chain(records_by_id, str(record["artifact_id"]))
        if chain is None:
            continue
        matching_embedding_id = _find_dependent_record_id(
            records_by_id,
            EMBEDDINGS_ARTIFACT_TYPE,
            CHUNKED_CORPUS_ARTIFACT_TYPE,
            chain[CHUNKED_CORPUS_ARTIFACT_TYPE],
        )
        if matching_embedding_id is not None:
            chain[EMBEDDINGS_ARTIFACT_TYPE] = matching_embedding_id
            matching_milvus_id = _find_milvus_record_id(
                records_by_id,
                chunked_corpus_id=chain[CHUNKED_CORPUS_ARTIFACT_TYPE],
                embeddings_id=matching_embedding_id,
            )
            if matching_milvus_id is not None:
                chain[MILVUS_COLLECTION_ARTIFACT_TYPE] = matching_milvus_id
        return chain

    for record in records_by_id.get(EMBEDDINGS_ARTIFACT_TYPE, {}).values():
        chain = _trace_embedding_chain(records_by_id, str(record["artifact_id"]))
        if chain is not None:
            return chain

    for record in records_by_id.get(CHUNKED_CORPUS_ARTIFACT_TYPE, {}).values():
        chain = _trace_chunked_chain(records_by_id, str(record["artifact_id"]))
        if chain is not None:
            return chain

    for record in records_by_id.get(NORMALIZED_DATASET_ARTIFACT_TYPE, {}).values():
        chain = _trace_normalized_chain(records_by_id, str(record["artifact_id"]))
        if chain is not None:
            return chain

    for record in records_by_id.get(RAW_DATASET_ARTIFACT_TYPE, {}).values():
        return {RAW_DATASET_ARTIFACT_TYPE: str(record["artifact_id"])}

    return {}


def _trace_milvus_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    artifact_id: str,
) -> dict[str, str] | None:
    record = records_by_id.get(MILVUS_COLLECTION_ARTIFACT_TYPE, {}).get(artifact_id)
    if record is None:
        return None
    chunked_corpus_id = _dependency_id(record, CHUNKED_CORPUS_ARTIFACT_TYPE)
    embeddings_id = _dependency_id(record, EMBEDDINGS_ARTIFACT_TYPE)
    if chunked_corpus_id is None or embeddings_id is None:
        return None
    chunk_chain = _trace_chunked_chain(records_by_id, chunked_corpus_id)
    embedding_chain = _trace_embedding_chain(records_by_id, embeddings_id)
    if chunk_chain is None or embedding_chain is None:
        return None
    if embedding_chain[CHUNKED_CORPUS_ARTIFACT_TYPE] != chunked_corpus_id:
        return None
    return {
        **chunk_chain,
        EMBEDDINGS_ARTIFACT_TYPE: embeddings_id,
        MILVUS_COLLECTION_ARTIFACT_TYPE: artifact_id,
    }


def _trace_elasticsearch_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    artifact_id: str,
) -> dict[str, str] | None:
    record = records_by_id.get(ELASTICSEARCH_INDEX_ARTIFACT_TYPE, {}).get(artifact_id)
    if record is None:
        return None
    chunked_corpus_id = _dependency_id(record, CHUNKED_CORPUS_ARTIFACT_TYPE)
    if chunked_corpus_id is None:
        return None
    chunk_chain = _trace_chunked_chain(records_by_id, chunked_corpus_id)
    if chunk_chain is None:
        return None
    return {**chunk_chain, ELASTICSEARCH_INDEX_ARTIFACT_TYPE: artifact_id}


def _trace_embedding_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    artifact_id: str,
) -> dict[str, str] | None:
    record = records_by_id.get(EMBEDDINGS_ARTIFACT_TYPE, {}).get(artifact_id)
    if record is None:
        return None
    chunked_corpus_id = _dependency_id(record, CHUNKED_CORPUS_ARTIFACT_TYPE)
    if chunked_corpus_id is None:
        return None
    chunk_chain = _trace_chunked_chain(records_by_id, chunked_corpus_id)
    if chunk_chain is None:
        return None
    return {**chunk_chain, EMBEDDINGS_ARTIFACT_TYPE: artifact_id}


def _trace_chunked_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    artifact_id: str,
) -> dict[str, str] | None:
    record = records_by_id.get(CHUNKED_CORPUS_ARTIFACT_TYPE, {}).get(artifact_id)
    if record is None:
        return None
    normalized_id = _dependency_id(record, NORMALIZED_DATASET_ARTIFACT_TYPE)
    if normalized_id is None:
        return None
    normalized_chain = _trace_normalized_chain(records_by_id, normalized_id)
    if normalized_chain is None:
        return None
    return {**normalized_chain, CHUNKED_CORPUS_ARTIFACT_TYPE: artifact_id}


def _trace_normalized_chain(
    records_by_id: dict[str, dict[str, dict[str, Any]]],
    artifact_id: str,
) -> dict[str, str] | None:
    record = records_by_id.get(NORMALIZED_DATASET_ARTIFACT_TYPE, {}).get(artifact_id)
    if record is None:
        return None
    raw_id = _dependency_id(record, RAW_DATASET_ARTIFACT_TYPE)
    if raw_id is None:
        return None
    if raw_id not in records_by_id.get(RAW_DATASET_ARTIFACT_TYPE, {}):
        return None
    return {
        RAW_DATASET_ARTIFACT_TYPE: raw_id,
        NORMALIZED_DATASET_ARTIFACT_TYPE: artifact_id,
    }


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
    for artifact_id, record in records_by_id.get(
        MILVUS_COLLECTION_ARTIFACT_TYPE,
        {},
    ).items():
        if (
            _dependency_id(record, CHUNKED_CORPUS_ARTIFACT_TYPE) == chunked_corpus_id
            and _dependency_id(record, EMBEDDINGS_ARTIFACT_TYPE) == embeddings_id
        ):
            return artifact_id
    return None


def _dependency_id(record: dict[str, Any], artifact_type: str) -> str | None:
    metadata = record.get("metadata_summary", {})
    for dependency in metadata.get("dependencies", []):
        if dependency.get("artifact_type") == artifact_type:
            artifact_id = dependency.get("artifact_id")
            return str(artifact_id) if artifact_id else None
    for key in DEPENDENCY_METADATA_KEYS_BY_ARTIFACT_TYPE.get(artifact_type, ()):
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
