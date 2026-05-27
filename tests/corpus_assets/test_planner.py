"""Tests for corpus asset dry-run planning."""

from __future__ import annotations

from typing import Any

import pytest

from eval_platform.corpus_assets import (
    ARTIFACT_STAGE_ORDER,
    DATASETS_BY_NAME,
    CorpusAssetError,
    build_plan_for_datasets,
)


def _complete_record(
    artifact_id: str,
    *,
    dependencies: list[tuple[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_summary = dict(metadata or {})
    if dependencies is not None:
        metadata_summary["dependencies"] = [
            {"artifact_type": artifact_type, "artifact_id": dependency_id}
            for artifact_type, dependency_id in dependencies
        ]
    return {
        "artifact_id": artifact_id,
        "complete": True,
        "metadata_summary": metadata_summary,
    }


def test_build_plan_orders_stages_and_uses_stable_names() -> None:
    spec = DATASETS_BY_NAME["IFIRNFCorpus"]

    plan = build_plan_for_datasets(
        datasets=[spec],
        run_id="five_ds_001",
        bucket="bucket",
        raw_prefix="sciverse_benchmark/raw",
        s3_prefix="test_sciverse_benchmark",
        raw_exists_by_slug={"ifir_nfcorpus": True},
    )

    dataset_plan = plan["datasets"]["IFIRNFCorpus"]
    assert [step["stage"] for step in dataset_plan["steps"]] == ARTIFACT_STAGE_ORDER
    assert dataset_plan["artifact_ids"] == dataset_plan["generated_artifact_ids"]
    assert dataset_plan["artifact_ids"]["raw_dataset"] == "ifir_nfcorpus_five_ds_001_raw"
    assert dataset_plan["resolved_artifact_ids"] == dataset_plan["generated_artifact_ids"]
    assert dataset_plan["generated_resource_names"] == {
        "elasticsearch_index": "ifir_nfcorpus_five_ds_001_es",
        "milvus_collection": "ifir_nfcorpus_five_ds_001_milvus",
    }
    assert dataset_plan["resolved_resource_names"] == dataset_plan[
        "generated_resource_names"
    ]
    assert dataset_plan["artifact_ids"]["elasticsearch_index"] == (
        "ifir_nfcorpus_five_ds_001_es_index"
    )
    assert dataset_plan["elasticsearch_index_name"] == "ifir_nfcorpus_five_ds_001_es"
    assert dataset_plan["milvus_collection_name"] == "ifir_nfcorpus_five_ds_001_milvus"
    assert dataset_plan["steps"][0]["raw_source_uri"] == (
        "s3://bucket/sciverse_benchmark/raw/ifir_nfcorpus"
    )
    assert dataset_plan["steps"][4]["source_artifact_id"] == (
        "ifir_nfcorpus_five_ds_001_chunks"
    )
    assert dataset_plan["steps"][5]["chunked_corpus_artifact_id"] == (
        "ifir_nfcorpus_five_ds_001_chunks"
    )
    assert dataset_plan["steps"][5]["embeddings_artifact_id"] == (
        "ifir_nfcorpus_five_ds_001_embeddings"
    )
    assert "source_artifact_id" not in dataset_plan["steps"][5]


def test_build_plan_raises_when_raw_prefix_missing() -> None:
    with pytest.raises(CorpusAssetError, match="Raw prefix does not exist"):
        build_plan_for_datasets(
            datasets=[DATASETS_BY_NAME["SciFact"]],
            run_id="five_ds_001",
            bucket="bucket",
            raw_prefix="sciverse_benchmark/raw",
            s3_prefix="test_sciverse_benchmark",
            raw_exists_by_slug={"scifact": False},
        )


def test_build_plan_can_reuse_existing_complete_artifacts() -> None:
    spec = DATASETS_BY_NAME["NFCorpus"]
    inventory: dict[str, Any] = {
        "datasets": {
            "NFCorpus": {
                "artifacts": {
                    "raw_dataset": [_complete_record("nfcorpus_old_raw")],
                    "normalized_dataset": [
                        _complete_record(
                            "nfcorpus_old_normalized",
                            dependencies=[("raw_dataset", "nfcorpus_old_raw")],
                        )
                    ],
                    "chunked_corpus": [
                        _complete_record(
                            "nfcorpus_old_chunks",
                            dependencies=[
                                ("normalized_dataset", "nfcorpus_old_normalized")
                            ],
                        )
                    ],
                    "embeddings": [
                        _complete_record(
                            "nfcorpus_old_embeddings",
                            dependencies=[("chunked_corpus", "nfcorpus_old_chunks")],
                        )
                    ],
                    "elasticsearch_index": [],
                    "milvus_collection": [],
                }
            }
        }
    }

    plan = build_plan_for_datasets(
        datasets=[spec],
        run_id="five_ds_001",
        bucket="bucket",
        raw_prefix="sciverse_benchmark/raw",
        s3_prefix="test_sciverse_benchmark",
        raw_exists_by_slug={"nfcorpus": True},
        reuse_existing=True,
        inventory=inventory,
    )

    dataset_plan = plan["datasets"]["NFCorpus"]
    first_step = dataset_plan["steps"][0]
    es_step = dataset_plan["steps"][4]
    milvus_step = dataset_plan["steps"][5]

    assert first_step["action"] == "reuse"
    assert first_step["artifact_id"] == "nfcorpus_old_raw"
    assert dataset_plan["generated_artifact_ids"]["chunked_corpus"] == (
        "nfcorpus_five_ds_001_chunks"
    )
    assert dataset_plan["generated_artifact_ids"]["embeddings"] == (
        "nfcorpus_five_ds_001_embeddings"
    )
    assert dataset_plan["resolved_artifact_ids"]["chunked_corpus"] == (
        "nfcorpus_old_chunks"
    )
    assert dataset_plan["resolved_artifact_ids"]["embeddings"] == (
        "nfcorpus_old_embeddings"
    )
    assert es_step["source_artifact_id"] == "nfcorpus_old_chunks"
    assert milvus_step["chunked_corpus_artifact_id"] == "nfcorpus_old_chunks"
    assert milvus_step["embeddings_artifact_id"] == "nfcorpus_old_embeddings"


def test_build_plan_reuse_existing_selects_one_consistent_downstream_chain() -> None:
    spec = DATASETS_BY_NAME["IFIRNFCorpus"]
    inventory: dict[str, Any] = {
        "datasets": {
            "IFIRNFCorpus": {
                "artifacts": {
                    "raw_dataset": [
                        _complete_record("depcheck_raw"),
                        _complete_record("full_raw"),
                    ],
                    "normalized_dataset": [
                        _complete_record(
                            "depcheck_normalized",
                            dependencies=[("raw_dataset", "depcheck_raw")],
                        ),
                        _complete_record(
                            "full_normalized",
                            dependencies=[("raw_dataset", "full_raw")],
                        ),
                    ],
                    "chunked_corpus": [
                        _complete_record(
                            "depcheck_chunks",
                            dependencies=[
                                ("normalized_dataset", "depcheck_normalized")
                            ],
                        ),
                        _complete_record(
                            "full_chunks",
                            dependencies=[("normalized_dataset", "full_normalized")],
                        ),
                    ],
                    "embeddings": [
                        _complete_record(
                            "depcheck_embeddings",
                            dependencies=[("chunked_corpus", "depcheck_chunks")],
                        ),
                        _complete_record(
                            "full_embeddings",
                            dependencies=[("chunked_corpus", "full_chunks")],
                        ),
                    ],
                    "elasticsearch_index": [
                        _complete_record(
                            "full_es_index",
                            dependencies=[("chunked_corpus", "full_chunks")],
                            metadata={"index_name": "full_real_es"},
                        )
                    ],
                    "milvus_collection": [
                        _complete_record(
                            "full_milvus_collection",
                            dependencies=[
                                ("chunked_corpus", "full_chunks"),
                                ("embeddings", "full_embeddings"),
                            ],
                            metadata={"collection_name": "full_real_milvus"},
                        )
                    ],
                }
            }
        }
    }

    plan = build_plan_for_datasets(
        datasets=[spec],
        run_id="validator_reuse_check",
        bucket="bucket",
        raw_prefix="sciverse_benchmark/raw",
        s3_prefix="test_sciverse_benchmark",
        raw_exists_by_slug={"ifir_nfcorpus": True},
        reuse_existing=True,
        inventory=inventory,
    )

    dataset_plan = plan["datasets"]["IFIRNFCorpus"]
    assert dataset_plan["resolved_artifact_ids"] == {
        "raw_dataset": "full_raw",
        "normalized_dataset": "full_normalized",
        "chunked_corpus": "full_chunks",
        "embeddings": "full_embeddings",
        "elasticsearch_index": "full_es_index",
        "milvus_collection": "full_milvus_collection",
    }
    assert {step["action"] for step in dataset_plan["steps"]} == {"reuse"}
    assert dataset_plan["steps"][4]["source_artifact_id"] == "full_chunks"
    assert dataset_plan["steps"][4]["index_name"] == "full_real_es"
    assert dataset_plan["steps"][5]["chunked_corpus_artifact_id"] == "full_chunks"
    assert dataset_plan["steps"][5]["embeddings_artifact_id"] == "full_embeddings"
    assert dataset_plan["steps"][5]["collection_name"] == "full_real_milvus"
    assert dataset_plan["generated_resource_names"] == {
        "elasticsearch_index": "ifir_nfcorpus_validator_reuse_check_es",
        "milvus_collection": "ifir_nfcorpus_validator_reuse_check_milvus",
    }
    assert dataset_plan["resolved_resource_names"] == {
        "elasticsearch_index": "full_real_es",
        "milvus_collection": "full_real_milvus",
    }
    assert dataset_plan["elasticsearch_index_name"] == "full_real_es"
    assert dataset_plan["milvus_collection_name"] == "full_real_milvus"


def test_reused_index_steps_use_manifest_dependencies() -> None:
    spec = DATASETS_BY_NAME["SciFact"]
    inventory: dict[str, Any] = {
        "datasets": {
            "SciFact": {
                "artifacts": {
                    "raw_dataset": [_complete_record("manifest_raw")],
                    "normalized_dataset": [
                        _complete_record(
                            "manifest_normalized",
                            metadata={"raw_dataset_artifact_id": "manifest_raw"},
                        )
                    ],
                    "chunked_corpus": [
                        _complete_record(
                            "manifest_chunks",
                            metadata={
                                "source_normalized_dataset_artifact_id": (
                                    "manifest_normalized"
                                )
                            },
                        )
                    ],
                    "embeddings": [
                        _complete_record(
                            "manifest_embeddings",
                            metadata={
                                "source_chunked_corpus_artifact_id": "manifest_chunks"
                            },
                        )
                    ],
                    "elasticsearch_index": [
                        _complete_record(
                            "manifest_es",
                            metadata={
                                "index_name": "manifest_real_es",
                                "source_chunked_corpus_artifact_id": "manifest_chunks",
                            },
                        )
                    ],
                    "milvus_collection": [
                        _complete_record(
                            "manifest_milvus",
                            metadata={
                                "collection_name": "manifest_real_milvus",
                                "source_chunked_corpus_artifact_id": "manifest_chunks",
                                "source_embeddings_artifact_id": "manifest_embeddings",
                            },
                        )
                    ],
                }
            }
        }
    }

    plan = build_plan_for_datasets(
        datasets=[spec],
        run_id="newrun",
        bucket="bucket",
        raw_prefix="sciverse_benchmark/raw",
        s3_prefix="test_sciverse_benchmark",
        raw_exists_by_slug={"scifact": True},
        reuse_existing=True,
        inventory=inventory,
    )

    steps = plan["datasets"]["SciFact"]["steps"]
    assert steps[4]["action"] == "reuse"
    assert steps[4]["source_artifact_id"] == "manifest_chunks"
    assert steps[4]["index_name"] == "manifest_real_es"
    assert steps[5]["action"] == "reuse"
    assert steps[5]["chunked_corpus_artifact_id"] == "manifest_chunks"
    assert steps[5]["embeddings_artifact_id"] == "manifest_embeddings"
    assert steps[5]["collection_name"] == "manifest_real_milvus"


def test_reused_elasticsearch_index_requires_manifest_index_name() -> None:
    spec = DATASETS_BY_NAME["SciFact"]
    inventory: dict[str, Any] = {
        "datasets": {
            "SciFact": {
                "artifacts": {
                    "raw_dataset": [_complete_record("raw")],
                    "normalized_dataset": [
                        _complete_record(
                            "normalized",
                            dependencies=[("raw_dataset", "raw")],
                        )
                    ],
                    "chunked_corpus": [
                        _complete_record(
                            "chunks",
                            dependencies=[("normalized_dataset", "normalized")],
                        )
                    ],
                    "embeddings": [],
                    "elasticsearch_index": [
                        _complete_record(
                            "es_index",
                            dependencies=[("chunked_corpus", "chunks")],
                        )
                    ],
                    "milvus_collection": [],
                }
            }
        }
    }

    with pytest.raises(
        CorpusAssetError,
        match="Reused artifact 'es_index' is missing required metadata 'index_name'",
    ):
        build_plan_for_datasets(
            datasets=[spec],
            run_id="newrun",
            bucket="bucket",
            raw_prefix="sciverse_benchmark/raw",
            s3_prefix="test_sciverse_benchmark",
            raw_exists_by_slug={"scifact": True},
            reuse_existing=True,
            inventory=inventory,
        )


def test_reused_milvus_collection_requires_manifest_collection_name() -> None:
    spec = DATASETS_BY_NAME["SciFact"]
    inventory: dict[str, Any] = {
        "datasets": {
            "SciFact": {
                "artifacts": {
                    "raw_dataset": [_complete_record("raw")],
                    "normalized_dataset": [
                        _complete_record(
                            "normalized",
                            dependencies=[("raw_dataset", "raw")],
                        )
                    ],
                    "chunked_corpus": [
                        _complete_record(
                            "chunks",
                            dependencies=[("normalized_dataset", "normalized")],
                        )
                    ],
                    "embeddings": [
                        _complete_record(
                            "embeddings",
                            dependencies=[("chunked_corpus", "chunks")],
                        )
                    ],
                    "elasticsearch_index": [],
                    "milvus_collection": [
                        _complete_record(
                            "milvus_collection",
                            dependencies=[
                                ("chunked_corpus", "chunks"),
                                ("embeddings", "embeddings"),
                            ],
                        )
                    ],
                }
            }
        }
    }

    with pytest.raises(
        CorpusAssetError,
        match=(
            "Reused artifact 'milvus_collection' is missing required metadata "
            "'collection_name'"
        ),
    ):
        build_plan_for_datasets(
            datasets=[spec],
            run_id="newrun",
            bucket="bucket",
            raw_prefix="sciverse_benchmark/raw",
            s3_prefix="test_sciverse_benchmark",
            raw_exists_by_slug={"scifact": True},
            reuse_existing=True,
            inventory=inventory,
        )


def test_build_plan_is_dry_run_and_has_no_external_clients() -> None:
    plan = build_plan_for_datasets(
        datasets=[DATASETS_BY_NAME["LitSearchRetrieval"]],
        run_id="five_ds_001",
        bucket="bucket",
        raw_prefix="sciverse_benchmark/raw",
        s3_prefix="test_sciverse_benchmark",
        raw_exists_by_slug={"litsearch": True},
    )

    assert plan["mode"] == "dry_run"
    for step in plan["datasets"]["LitSearchRetrieval"]["steps"]:
        assert step["action"] == "create"
        assert "client" not in step
        assert "api_key" not in step
        assert "password" not in step
