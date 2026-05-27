"""Tests for corpus asset registry and naming helpers."""

from __future__ import annotations

import pytest

from eval_platform.corpus_assets import (
    DATASETS_BY_NAME,
    CorpusAssetError,
    artifact_ids_for_dataset,
    collection_name_for_dataset,
    dataset_specs_for_selection,
    index_name_for_dataset,
    raw_prefix_key,
    raw_prefix_uri,
)


def test_dataset_selection_supports_task_name_slug_and_all() -> None:
    assert [spec.task_name for spec in dataset_specs_for_selection("IFIRNFCorpus")] == [
        "IFIRNFCorpus"
    ]
    assert [spec.task_name for spec in dataset_specs_for_selection("ifir_nfcorpus")] == [
        "IFIRNFCorpus"
    ]
    assert len(dataset_specs_for_selection("all")) == 5


def test_dataset_selection_rejects_unknown_dataset() -> None:
    with pytest.raises(CorpusAssetError, match="Unknown dataset"):
        dataset_specs_for_selection("unknown")


def test_artifact_id_and_resource_name_mapping() -> None:
    spec = DATASETS_BY_NAME["IFIRNFCorpus"]

    assert artifact_ids_for_dataset(spec, "run_001") == {
        "raw_dataset": "ifir_nfcorpus_run_001_raw",
        "normalized_dataset": "ifir_nfcorpus_run_001_normalized",
        "chunked_corpus": "ifir_nfcorpus_run_001_chunks",
        "embeddings": "ifir_nfcorpus_run_001_embeddings",
        "elasticsearch_index": "ifir_nfcorpus_run_001_es_index",
        "milvus_collection": "ifir_nfcorpus_run_001_milvus_collection",
    }
    assert index_name_for_dataset(spec, "run_001") == "ifir_nfcorpus_run_001_es"
    assert collection_name_for_dataset(spec, "run_001") == (
        "ifir_nfcorpus_run_001_milvus"
    )


def test_artifact_ids_reject_empty_run_id() -> None:
    with pytest.raises(CorpusAssetError, match="run_id must not be empty"):
        artifact_ids_for_dataset(DATASETS_BY_NAME["IFIRNFCorpus"], " ")


def test_raw_prefix_key_and_uri() -> None:
    spec = DATASETS_BY_NAME["LitSearchRetrieval"]

    assert raw_prefix_key("/sciverse_benchmark/raw/", spec) == (
        "sciverse_benchmark/raw/litsearch"
    )
    assert raw_prefix_uri("bucket", "/sciverse_benchmark/raw/", spec) == (
        "s3://bucket/sciverse_benchmark/raw/litsearch"
    )
