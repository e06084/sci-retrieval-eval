from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.assets import (
    AssetFingerprintError,
    build_asset_fingerprint,
    canonical_json_hash,
    chunked_corpus_fingerprint_components,
    elasticsearch_index_fingerprint_components,
    embeddings_fingerprint_components,
    metrics_run_fingerprint_components,
    milvus_collection_fingerprint_components,
    normalized_dataset_fingerprint_components,
    raw_dataset_fingerprint_components,
    retrieval_run_fingerprint_components,
)
from eval_platform.datasets import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
    write_normalized_dataset_artifact,
)
from eval_platform.metrics import MetricsRunConfig, read_metrics_run_artifact, run_metrics
from eval_platform.retrieval import (
    RetrievalHit,
    RetrievalRunConfig,
    read_retrieval_run_artifact,
    run_retrieval,
)


def test_canonical_hash_is_stable_for_key_order() -> None:
    first = {"b": 2, "a": {"y": 1, "x": [3, 4]}}
    second = {"a": {"x": [3, 4], "y": 1}, "b": 2}

    assert canonical_json_hash(first) == canonical_json_hash(second)


def test_canonical_hash_changes_for_different_value() -> None:
    assert canonical_json_hash({"a": 1}) != canonical_json_hash({"a": 2})


def test_canonical_hash_rejects_non_json_serializable_value() -> None:
    with pytest.raises(AssetFingerprintError):
        canonical_json_hash({"bad": object()})


def test_canonical_hash_rejects_secret_key() -> None:
    with pytest.raises(AssetFingerprintError):
        canonical_json_hash({"api_key": "redacted"})


def test_canonical_hash_rejects_nested_secret_key() -> None:
    with pytest.raises(AssetFingerprintError):
        canonical_json_hash({"embedding": {"Access_Key": "redacted"}})


def test_canonical_hash_rejects_secret_key_inside_list_of_dicts() -> None:
    with pytest.raises(AssetFingerprintError):
        canonical_json_hash({"files": [{"path": "a"}, {"token": "redacted"}]})


@pytest.mark.parametrize(
    "secret_key",
    [
        "api_key",
        "access_key",
        "client_secret",
        "password",
        "token",
        "Authorization",
    ],
)
def test_canonical_hash_rejects_all_secret_key_fragments(secret_key: str) -> None:
    with pytest.raises(AssetFingerprintError):
        canonical_json_hash({secret_key: "redacted"})


def test_canonical_hash_rejects_operational_identity_key() -> None:
    with pytest.raises(AssetFingerprintError):
        canonical_json_hash({"run_id": "experiment-001"})


@pytest.mark.parametrize(
    "timestamp_key",
    [
        "created_at",
        "updated_at",
        "started_at",
        "completed_at",
        "timestamp",
        "created_time",
        "updated_time",
    ],
)
def test_canonical_hash_rejects_timestamp_identity_keys(timestamp_key: str) -> None:
    with pytest.raises(AssetFingerprintError):
        canonical_json_hash({timestamp_key: "2026-05-30T00:00:00Z"})


def test_build_asset_fingerprint_returns_stable_fingerprint() -> None:
    first = build_asset_fingerprint(
        artifact_type="retrieval_run",
        components={"model": "demo", "params": {"top_k": 10}},
    )
    second = build_asset_fingerprint(
        artifact_type="retrieval_run",
        components={"params": {"top_k": 10}, "model": "demo"},
    )

    assert first.sha256 == second.sha256
    assert first.fingerprint_version == 1
    assert first.artifact_type == "retrieval_run"


def test_build_asset_fingerprint_rejects_empty_artifact_type() -> None:
    with pytest.raises(ValidationError):
        build_asset_fingerprint(artifact_type=" ", components={})


def test_build_asset_fingerprint_rejects_invalid_version() -> None:
    with pytest.raises(ValidationError):
        build_asset_fingerprint(
            artifact_type="retrieval_run",
            components={},
            fingerprint_version=0,
        )


def test_build_asset_fingerprint_does_not_mutate_input_components() -> None:
    components = {"params": {"b": 2, "a": 1}}
    before = {"params": {"b": 2, "a": 1}}

    fingerprint = build_asset_fingerprint(artifact_type="embeddings", components=components)

    assert components == before
    assert fingerprint.components == before
    assert fingerprint.components is not components


def test_raw_dataset_components() -> None:
    components = raw_dataset_fingerprint_components(
        dataset_name="NFCorpus",
        raw_source_uri="s3://bucket/raw/nfcorpus/",
        raw_format="jsonl",
        split="test",
        file_fingerprints=[
            {"path": "corpus.jsonl", "size_bytes": 123, "sha256": "corpus-sha"},
            {"path": "queries.jsonl", "size_bytes": 45, "sha256": "queries-sha"},
        ],
    )

    assert components == {
        "dataset_name": "NFCorpus",
        "raw_source_uri": "s3://bucket/raw/nfcorpus/",
        "raw_format": "jsonl",
        "split": "test",
        "file_fingerprints": [
            {"path": "corpus.jsonl", "size_bytes": 123, "sha256": "corpus-sha"},
            {"path": "queries.jsonl", "size_bytes": 45, "sha256": "queries-sha"},
        ],
    }
    assert "run_id" not in components
    assert "artifact_id" not in components


def test_raw_dataset_file_fingerprints_are_canonical_ordered() -> None:
    first = raw_dataset_fingerprint_components(
        dataset_name="NFCorpus",
        raw_source_uri="s3://bucket/raw/nfcorpus/",
        raw_format="jsonl",
        split="test",
        file_fingerprints=[
            {"path": "queries.jsonl", "size_bytes": 45, "sha256": "queries-sha"},
            {"path": "corpus.jsonl", "size_bytes": 123, "sha256": "corpus-sha"},
        ],
    )
    second = raw_dataset_fingerprint_components(
        dataset_name="NFCorpus",
        raw_source_uri="s3://bucket/raw/nfcorpus/",
        raw_format="jsonl",
        split="test",
        file_fingerprints=[
            {"path": "corpus.jsonl", "size_bytes": 123, "sha256": "corpus-sha"},
            {"path": "queries.jsonl", "size_bytes": 45, "sha256": "queries-sha"},
        ],
    )

    assert first == second
    assert [item["path"] for item in first["file_fingerprints"] or []] == [
        "corpus.jsonl",
        "queries.jsonl",
    ]
    assert _fingerprint("raw_dataset", first).sha256 == _fingerprint(
        "raw_dataset",
        second,
    ).sha256


@pytest.mark.parametrize(
    ("field_name", "new_value"),
    [
        ("path", "corpus-v2.jsonl"),
        ("sha256", "corpus-sha-v2"),
        ("size_bytes", 124),
    ],
)
def test_changing_file_fingerprint_changes_raw_dataset_fingerprint(
    field_name: str,
    new_value: Any,
) -> None:
    base_files = [
        {"path": "corpus.jsonl", "size_bytes": 123, "sha256": "corpus-sha"},
        {"path": "queries.jsonl", "size_bytes": 45, "sha256": "queries-sha"},
    ]
    changed_files = [dict(item) for item in base_files]
    changed_files[0][field_name] = new_value

    base = _fingerprint(
        "raw_dataset",
        raw_dataset_fingerprint_components(
            dataset_name="NFCorpus",
            raw_source_uri="s3://bucket/raw/nfcorpus/",
            raw_format="jsonl",
            split="test",
            file_fingerprints=base_files,
        ),
    )
    changed = _fingerprint(
        "raw_dataset",
        raw_dataset_fingerprint_components(
            dataset_name="NFCorpus",
            raw_source_uri="s3://bucket/raw/nfcorpus/",
            raw_format="jsonl",
            split="test",
            file_fingerprints=changed_files,
        ),
    )

    assert base.sha256 != changed.sha256


def test_normalized_dataset_components() -> None:
    components = normalized_dataset_fingerprint_components(
        raw_dataset_fingerprint="raw-sha",
        normalizer_name="mteb",
        normalizer_version="1",
        schema_version="normalized-v1",
        normalizer_params={"doc_id_field": "doc_id"},
    )

    assert components == {
        "raw_dataset_fingerprint": "raw-sha",
        "normalizer_name": "mteb",
        "normalizer_version": "1",
        "schema_version": "normalized-v1",
        "normalizer_params": {"doc_id_field": "doc_id"},
    }


def test_chunked_corpus_components() -> None:
    components = chunked_corpus_fingerprint_components(
        normalized_dataset_fingerprint="normalized-sha",
        chunker_source="sciverse",
        chunker_name="sciverse-admin-ingest",
        source_git_remote_url="git@github.com:example/chunker.git",
        git_commit="abc123",
        chunker_entrypoint=None,
        chunk_params={"chunk_size": 512, "chunk_overlap": 64, "chunk_type": "sentence"},
        schema_version="chunked_corpus.v1",
    )

    assert components == {
        "normalized_dataset_fingerprint": "normalized-sha",
        "chunker_source": "sciverse",
        "chunker_name": "sciverse-admin-ingest",
        "source_git_remote_url": "git@github.com:example/chunker.git",
        "git_commit": "abc123",
        "chunker_entrypoint": None,
        "chunk_params": {
            "chunk_size": 512,
            "chunk_overlap": 64,
            "chunk_type": "sentence",
        },
        "schema_version": "chunked_corpus.v1",
    }


def test_chunked_corpus_components_validate_required_fields_and_params() -> None:
    with pytest.raises(AssetFingerprintError):
        chunked_corpus_fingerprint_components(
            normalized_dataset_fingerprint="normalized-sha",
            chunker_source=" ",
            chunker_name="chunker",
            source_git_remote_url="git@github.com:example/chunker.git",
            git_commit="abc123",
            chunk_params={},
            schema_version="chunked_corpus.v1",
        )
    with pytest.raises(AssetFingerprintError):
        chunked_corpus_fingerprint_components(
            normalized_dataset_fingerprint="normalized-sha",
            chunker_source="sciverse",
            chunker_name="chunker",
            source_git_remote_url="git@github.com:example/chunker.git",
            git_commit="abc123",
            chunker_entrypoint=" ",
            chunk_params={},
            schema_version="chunked_corpus.v1",
        )
    with pytest.raises(AssetFingerprintError):
        chunked_corpus_fingerprint_components(
            normalized_dataset_fingerprint="normalized-sha",
            chunker_source="sciverse",
            chunker_name="chunker",
            source_git_remote_url="git@github.com:example/chunker.git",
            git_commit="abc123",
            chunk_params={"api_token": "redacted"},
            schema_version="chunked_corpus.v1",
        )
    with pytest.raises(AssetFingerprintError):
        chunked_corpus_fingerprint_components(
            normalized_dataset_fingerprint="normalized-sha",
            chunker_source="sciverse",
            chunker_name="chunker",
            source_git_remote_url="git@github.com:example/chunker.git",
            git_commit="abc123",
            chunk_params={"run_id": "bad"},
            schema_version="chunked_corpus.v1",
        )


def test_embeddings_components() -> None:
    components = embeddings_fingerprint_components(
        chunked_corpus_fingerprint="chunk-sha",
        embedding_source="sciverse_internal",
        model_name="bge-large",
        model_revision="rev-1",
        embedding_dim=1024,
        endpoint_alias="embedding-prod",
        api_version="2026-05",
        input_field="text",
        call_params={"input_type": "document"},
        normalized=True,
        storage_type="json_float",
    )

    assert components == {
        "chunked_corpus_fingerprint": "chunk-sha",
        "embedding_source": "sciverse_internal",
        "model_name": "bge-large",
        "model_revision": "rev-1",
        "embedding_dim": 1024,
        "endpoint_alias": "embedding-prod",
        "api_version": "2026-05",
        "input_field": "text",
        "call_params": {"input_type": "document"},
        "normalized": True,
        "storage_type": "json_float",
    }


def test_embeddings_components_validate_embedding_dim() -> None:
    with pytest.raises(AssetFingerprintError):
        embeddings_fingerprint_components(
            chunked_corpus_fingerprint="chunk-sha",
            embedding_source="sciverse_internal",
            model_name="bge-large",
            model_revision=None,
            embedding_dim=0,
        )


def test_embeddings_call_params_reject_real_endpoint_url() -> None:
    with pytest.raises(AssetFingerprintError):
        _embeddings_components(call_params={"endpoint_url": "http://real-endpoint/v1"})


def test_elasticsearch_index_components() -> None:
    components = elasticsearch_index_fingerprint_components(
        chunked_corpus_fingerprint="chunk-sha",
        builder_source="sci-retrieval-eval",
        code_git_commit="code-sha",
        builder_entrypoint="eval_platform.indexes.elasticsearch.run_elasticsearch_ingest",
        builder_params={
            "id_field": "chunk_id",
            "text_fields": ["text"],
            "metadata_fields": ["doc_id"],
            "empty_text_policy": "skip",
        },
        mapping={"properties": {"text": {"type": "text"}}},
        settings={"analysis": {"analyzer": {"default": {"type": "standard"}}}},
        ingest_params={},
    )

    assert components["chunked_corpus_fingerprint"] == "chunk-sha"
    assert components["builder_source"] == "sci-retrieval-eval"
    assert components["code_git_commit"] == "code-sha"
    assert components["builder_params"]["text_fields"] == ["text"]
    assert components["mapping"] == {"properties": {"text": {"type": "text"}}}
    assert components["settings"] == {
        "analysis": {"analyzer": {"default": {"type": "standard"}}}
    }
    assert components["ingest_params"] == {}
    assert "index_name" not in components
    assert "elasticsearch_url" not in components


def test_elasticsearch_builder_params_reject_physical_index_name() -> None:
    with pytest.raises(AssetFingerprintError):
        _es_components(builder_params={"index_name": "physical-index"})


def test_milvus_collection_components() -> None:
    components = milvus_collection_fingerprint_components(
        chunked_corpus_fingerprint="chunk-sha",
        embeddings_fingerprint="embeddings-sha",
        builder_source="sci-retrieval-eval",
        code_git_commit="code-sha",
        builder_entrypoint="eval_platform.indexes.milvus.run_milvus_ingest",
        builder_params={
            "id_field": "chunk_id",
            "doc_id_field": "doc_id",
            "vector_field": "embedding",
            "metadata_fields": ["doc_id"],
        },
        schema={"primary_field": "chunk_id", "vector_field": "embedding", "dim": 1024},
        metric_type="IP",
        index_type="HNSW",
        index_params={"M": 16},
    )

    assert components["chunked_corpus_fingerprint"] == "chunk-sha"
    assert components["embeddings_fingerprint"] == "embeddings-sha"
    assert components["builder_params"]["vector_field"] == "embedding"
    assert components["schema"] == {
        "primary_field": "chunk_id",
        "vector_field": "embedding",
        "dim": 1024,
    }
    assert components["metric_type"] == "IP"
    assert components["index_params"] == {"M": 16}
    assert "collection_name" not in components
    assert "milvus_uri" not in components


def test_milvus_builder_params_reject_physical_collection_name() -> None:
    with pytest.raises(AssetFingerprintError):
        _milvus_components(builder_params={"collection_name": "physical-collection"})


def test_retrieval_run_components() -> None:
    components = retrieval_run_fingerprint_components(
        normalized_dataset_fingerprint="normalized-sha",
        retrieval_mode="hybrid",
        elasticsearch_index_fingerprint="es-sha",
        milvus_collection_fingerprint="milvus-sha",
        query_source={"query_limit": 50},
        query_embedding={
            "embedding_source": "sciverse_internal",
            "model_name": "bge-large",
            "model_revision": "rev-1",
            "embedding_dim": 1024,
            "endpoint_alias": "embedding-prod",
            "api_version": None,
            "input_field": "query_text",
            "call_params": {"input_type": "query"},
            "normalized": True,
        },
        search_params={
            "es": {"top_k": 50, "query_fields": ["text"]},
            "milvus": {"top_k": 50, "search_params": {"ef": 128}},
            "fusion": {"method": "rrf", "path_topk": 25, "k": 60},
        },
        rerank={
            "rerank_source": "sciverse_internal",
            "model_name": "bge-reranker",
            "model_revision": "rev-1",
            "endpoint_alias": "rerank-prod",
            "top_n": 50,
        },
        rewrite=None,
        trace_mode="replay",
    )

    assert components["normalized_dataset_fingerprint"] == "normalized-sha"
    assert components["retrieval_mode"] == "hybrid"
    assert components["elasticsearch_index_fingerprint"] == "es-sha"
    assert components["milvus_collection_fingerprint"] == "milvus-sha"
    assert components["query_source"] == {"query_limit": 50}
    assert components["query_embedding"]["input_field"] == "query_text"
    assert components["search_params"]["fusion"]["method"] == "rrf"
    assert components["rerank"]["model_name"] == "bge-reranker"
    assert components["rewrite"] is None
    assert components["trace_mode"] == "replay"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("index_name", "physical-index"),
        ("collection_name", "physical-collection"),
        ("endpoint_url", "http://real-endpoint/v1"),
        ("elasticsearch_url", "http://real-es:9200"),
        ("milvus_uri", "http://real-milvus:19530"),
        ("url", "http://real-service"),
        ("uri", "http://real-service"),
        ("host", "127.0.0.1"),
        ("port", 9200),
        ("request_id", "request-1"),
        ("trace_file", "trace.jsonl"),
        ("trace_path", "/tmp/trace.jsonl"),
        ("timestamp", "2026-05-30T00:00:00Z"),
        ("updated_at", "2026-05-30T00:00:00Z"),
        ("started_at", "2026-05-30T00:00:00Z"),
        ("completed_at", "2026-05-30T00:00:00Z"),
        ("created_time", "2026-05-30T00:00:00Z"),
        ("updated_time", "2026-05-30T00:00:00Z"),
    ],
)
def test_retrieval_free_params_reject_runtime_and_physical_keys(
    key: str,
    value: Any,
) -> None:
    with pytest.raises(AssetFingerprintError):
        _retrieval_components(search_params={"backend": {key: value}})


@pytest.mark.parametrize(
    "query_embedding",
    [
        {"endpoint_url": "http://real-endpoint/v1/embeddings"},
        {"url": "http://real-endpoint/v1/embeddings"},
    ],
)
def test_retrieval_query_embedding_rejects_real_endpoint_keys(
    query_embedding: dict[str, Any],
) -> None:
    with pytest.raises(AssetFingerprintError):
        _retrieval_components(query_embedding=query_embedding)


def test_stable_identity_url_fields_are_allowed() -> None:
    raw = raw_dataset_fingerprint_components(
        dataset_name="NFCorpus",
        raw_source_uri="s3://bucket/raw/nfcorpus/",
        raw_format="jsonl",
    )
    chunked = _chunked_components(
        source_git_remote_url="https://github.com/example/chunker.git",
    )
    embeddings = _embeddings_components(endpoint_alias="embedding-prod")

    assert raw["raw_source_uri"] == "s3://bucket/raw/nfcorpus/"
    assert chunked["source_git_remote_url"] == "https://github.com/example/chunker.git"
    assert embeddings["endpoint_alias"] == "embedding-prod"


def test_metrics_run_components() -> None:
    components = metrics_run_fingerprint_components(
        normalized_dataset_fingerprint="normalized-sha",
        retrieval_run_fingerprint="retrieval-sha",
        metrics_source="sci-retrieval-eval",
        code_git_commit="code-sha",
        metrics_entrypoint="eval_platform.metrics.runner.run_metrics",
        metric_params={
            "k_values": [1, 3, 5, 10],
            "main_metric": "ndcg_at_10",
            "projection": {
                "from": "chunk",
                "to": "doc",
                "dedupe_policy": "first_chunk_rank",
            },
            "missing_query_policy": "zero",
        },
    )

    assert components == {
        "normalized_dataset_fingerprint": "normalized-sha",
        "retrieval_run_fingerprint": "retrieval-sha",
        "metrics_source": "sci-retrieval-eval",
        "code_git_commit": "code-sha",
        "metrics_entrypoint": "eval_platform.metrics.runner.run_metrics",
        "metric_params": {
            "k_values": [1, 3, 5, 10],
            "main_metric": "ndcg_at_10",
            "projection": {
                "from": "chunk",
                "to": "doc",
                "dedupe_policy": "first_chunk_rank",
            },
            "missing_query_policy": "zero",
        },
    }


def test_changing_embedding_model_changes_embeddings_fingerprint() -> None:
    first = _fingerprint(
        "embeddings",
        _embeddings_components(model_name="model-a"),
    )
    second = _fingerprint(
        "embeddings",
        _embeddings_components(model_name="model-b"),
    )

    assert first.sha256 != second.sha256


def test_changing_embedding_model_does_not_change_chunked_corpus_fingerprint() -> None:
    components = _chunked_components()

    first = _fingerprint("chunked_corpus", components)
    second = _fingerprint("chunked_corpus", components)

    assert first.sha256 == second.sha256


def test_changing_chunk_inputs_changes_chunked_corpus_fingerprint() -> None:
    base = _fingerprint("chunked_corpus", _chunked_components())
    changed_commit = _fingerprint(
        "chunked_corpus",
        _chunked_components(git_commit="def456"),
    )
    changed_params = _fingerprint(
        "chunked_corpus",
        _chunked_components(chunk_params={"chunk_size": 1024}),
    )
    changed_entrypoint = _fingerprint(
        "chunked_corpus",
        _chunked_components(chunker_entrypoint="chunkers.sentence_chunk"),
    )

    assert base.sha256 != changed_commit.sha256
    assert base.sha256 != changed_params.sha256
    assert base.sha256 != changed_entrypoint.sha256


def test_changing_chunk_params_changes_downstream_embeddings_fingerprint() -> None:
    chunk_a = _fingerprint("chunked_corpus", _chunked_components(chunk_params={"chunk_size": 512}))
    chunk_b = _fingerprint("chunked_corpus", _chunked_components(chunk_params={"chunk_size": 1024}))
    embeddings_a = _fingerprint(
        "embeddings",
        _embeddings_components(chunked_corpus_fingerprint=chunk_a.sha256),
    )
    embeddings_b = _fingerprint(
        "embeddings",
        _embeddings_components(chunked_corpus_fingerprint=chunk_b.sha256),
    )

    assert embeddings_a.sha256 != embeddings_b.sha256


def test_changing_elasticsearch_index_inputs_changes_fingerprint() -> None:
    base = _fingerprint("elasticsearch_index", _es_components())
    changed_builder = _fingerprint(
        "elasticsearch_index",
        _es_components(builder_params={"text_fields": ["title", "text"]}),
    )
    changed_mapping = _fingerprint(
        "elasticsearch_index",
        _es_components(mapping={"properties": {"text": {"type": "keyword"}}}),
    )
    changed_settings = _fingerprint(
        "elasticsearch_index",
        _es_components(settings={"analysis": {"analyzer": {"default": {"type": "whitespace"}}}}),
    )

    assert base.sha256 != changed_builder.sha256
    assert base.sha256 != changed_mapping.sha256
    assert base.sha256 != changed_settings.sha256


def test_changing_milvus_collection_inputs_changes_fingerprint() -> None:
    base = _fingerprint("milvus_collection", _milvus_components())
    changed_metric = _fingerprint("milvus_collection", _milvus_components(metric_type="COSINE"))
    changed_index = _fingerprint(
        "milvus_collection",
        _milvus_components(index_params={"M": 32}),
    )
    changed_schema = _fingerprint(
        "milvus_collection",
        _milvus_components(schema={"primary_field": "chunk_id", "dim": 1024}),
    )

    assert base.sha256 != changed_metric.sha256
    assert base.sha256 != changed_index.sha256
    assert base.sha256 != changed_schema.sha256


def test_changing_retrieval_params_changes_retrieval_fingerprint() -> None:
    base = _fingerprint("retrieval_run", _retrieval_components())
    changed_search = _fingerprint(
        "retrieval_run",
        _retrieval_components(search_params={"fusion": {"method": "rrf", "path_topk": 50}}),
    )
    changed_rerank = _fingerprint(
        "retrieval_run",
        _retrieval_components(rerank={"model_name": "other-reranker"}),
    )
    changed_trace = _fingerprint("retrieval_run", _retrieval_components(trace_mode="none"))

    assert base.sha256 != changed_search.sha256
    assert base.sha256 != changed_rerank.sha256
    assert base.sha256 != changed_trace.sha256


def test_retrieval_fingerprint_changes_when_paper_cap_changes() -> None:
    base = _fingerprint("retrieval_run", _retrieval_components())
    changed_paper_cap = _fingerprint(
        "retrieval_run",
        _retrieval_components(
            search_params={"fusion": {"method": "rrf", "path_topk": 25, "paper_cap": 1}}
        ),
    )

    assert base.sha256 != changed_paper_cap.sha256


def test_changing_metric_params_changes_metrics_but_not_retrieval_fingerprint() -> None:
    retrieval = _fingerprint("retrieval_run", _retrieval_components())
    metrics_a = _fingerprint(
        "metrics_run",
        _metrics_components(retrieval_run_fingerprint=retrieval.sha256),
    )
    metrics_b = _fingerprint(
        "metrics_run",
        _metrics_components(
            retrieval_run_fingerprint=retrieval.sha256,
            metric_params={"main_metric": "recall_at_10"},
        ),
    )

    assert metrics_a.sha256 != metrics_b.sha256
    assert retrieval.sha256 == _fingerprint("retrieval_run", retrieval.components).sha256


def test_minimal_e4_eval_builds_core_fingerprints_and_runs_metrics(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    write_normalized_dataset_artifact(
        store,
        "mini-normalized",
        NormalizedDataset(
            corpus=[
                CorpusRecord(doc_id="doc-1", text="alpha document"),
                CorpusRecord(doc_id="doc-2", text="beta document"),
            ],
            queries=[QueryRecord(query_id="q-1", text="alpha query")],
            qrels=[QrelRecord(query_id="q-1", doc_id="doc-1", relevance=1.0)],
        ),
    )

    run_retrieval(
        store,
        store,
        RetrievalRunConfig(
            source_normalized_dataset_artifact_id="mini-normalized",
            output_artifact_id="mini-retrieval",
            retrieval_mode="hybrid",
            top_k=2,
            query_limit=1,
            elasticsearch_index_artifact_id="mini-es",
            milvus_collection_artifact_id="mini-milvus",
            index_name="mini-index",
            collection_name="mini-collection",
            hybrid_per_source_topk=2,
            rrf_path_topk=2,
            rerank_enabled=True,
            rerank_candidate_cap=2,
            rerank_cross_path_topk=2,
            trace_mode="replay",
        ),
        es_client=MiniElasticsearchClient(),
        milvus_client=MiniMilvusClient(),
        embedding_client=MiniEmbeddingClient(),
        rerank_client=MiniRerankClient(),
    )
    run_metrics(
        store,
        store,
        MetricsRunConfig(
            source_normalized_dataset_artifact_id="mini-normalized",
            source_retrieval_run_artifact_id="mini-retrieval",
            output_artifact_id="mini-metrics",
        ),
    )

    retrieval_records = read_retrieval_run_artifact(store, "mini-retrieval")
    metrics = read_metrics_run_artifact(store, "mini-metrics")
    trace = retrieval_records[0].trace

    assert trace is not None
    assert trace["es_hits"]
    assert trace["milvus_hits"]
    assert trace["fused_hits"]
    assert trace["rerank_input"]
    assert trace["rerank_hits"]
    assert trace["final_hits"]
    assert retrieval_records[0].hits[0].doc_id == "doc-1"
    assert metrics.main_score > 0

    fingerprints = _minimal_eval_fingerprints()

    assert set(fingerprints) == {
        "raw_dataset",
        "normalized_dataset",
        "chunked_corpus",
        "embeddings",
        "elasticsearch_index",
        "milvus_collection",
        "retrieval_run",
        "metrics_run",
    }
    assert all(fingerprint.sha256 for fingerprint in fingerprints.values())
    assert fingerprints["metrics_run"].components["retrieval_run_fingerprint"] == (
        fingerprints["retrieval_run"].sha256
    )


class MiniEmbeddingClient:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]


class MiniElasticsearchClient:
    def search_bm25(self, index_name: str, query: str, top_k: int) -> list[RetrievalHit]:
        hits = [
            RetrievalHit(
                chunk_id="chunk-1",
                doc_id="doc-1",
                text="alpha document",
                score=10.0,
                recall_source="es",
            ),
            RetrievalHit(
                chunk_id="chunk-2",
                doc_id="doc-2",
                text="beta document",
                score=5.0,
                recall_source="es",
            ),
        ]
        return hits[:top_k]

    def enrich_by_chunk_ids(
        self,
        index_name: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        docs = {
            "chunk-1": {"doc_id": "doc-1", "text": "alpha document"},
            "chunk-2": {"doc_id": "doc-2", "text": "beta document"},
        }
        return [
            hit.model_copy(
                update={
                    "doc_id": hit.doc_id or docs[hit.chunk_id]["doc_id"],
                    "text": hit.text or docs[hit.chunk_id]["text"],
                }
            )
            for hit in hits
        ]


class MiniMilvusClient:
    def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        hits = [
            RetrievalHit(
                chunk_id="chunk-1",
                doc_id="doc-1",
                score=0.91,
                recall_source="milvus",
            ),
            RetrievalHit(
                chunk_id="chunk-2",
                doc_id="doc-2",
                score=0.72,
                recall_source="milvus",
            ),
        ]
        return hits[:top_k]


class MiniRerankClient:
    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        top_n: int,
    ) -> list[RetrievalHit]:
        return [
            hit.model_copy(update={"score": 100.0 - index})
            for index, hit in enumerate(hits[:top_n], start=1)
        ]


def _fingerprint(artifact_type: str, components: dict[str, Any]):
    return build_asset_fingerprint(artifact_type=artifact_type, components=components)


def _chunked_components(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "normalized_dataset_fingerprint": "normalized-sha",
        "chunker_source": "sciverse",
        "chunker_name": "sciverse-admin-ingest",
        "source_git_remote_url": "git@github.com:example/chunker.git",
        "git_commit": "abc123",
        "chunker_entrypoint": None,
        "chunk_params": {"chunk_size": 512},
        "schema_version": "chunked_corpus.v1",
    }
    payload.update(overrides)
    return chunked_corpus_fingerprint_components(**payload)


def _embeddings_components(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chunked_corpus_fingerprint": "chunk-sha",
        "embedding_source": "sciverse_internal",
        "model_name": "model-a",
        "model_revision": "rev-1",
        "embedding_dim": 768,
        "endpoint_alias": "embedding-prod",
        "api_version": None,
        "input_field": "text",
        "call_params": {"input_type": "document"},
        "normalized": True,
        "storage_type": "json_float",
    }
    payload.update(overrides)
    return embeddings_fingerprint_components(**payload)


def _es_components(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chunked_corpus_fingerprint": "chunk-sha",
        "builder_source": "sci-retrieval-eval",
        "code_git_commit": "code-sha",
        "builder_entrypoint": "eval_platform.indexes.elasticsearch.run_elasticsearch_ingest",
        "builder_params": {"text_fields": ["text"]},
        "mapping": {"properties": {"text": {"type": "text"}}},
        "settings": {"analysis": {"analyzer": {"default": {"type": "standard"}}}},
        "ingest_params": {},
    }
    payload.update(overrides)
    return elasticsearch_index_fingerprint_components(**payload)


def _milvus_components(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "chunked_corpus_fingerprint": "chunk-sha",
        "embeddings_fingerprint": "embedding-sha",
        "builder_source": "sci-retrieval-eval",
        "code_git_commit": "code-sha",
        "builder_entrypoint": "eval_platform.indexes.milvus.run_milvus_ingest",
        "builder_params": {"vector_field": "embedding"},
        "schema": {"primary_field": "chunk_id", "dim": 768},
        "metric_type": "IP",
        "index_type": "HNSW",
        "index_params": {"M": 16},
    }
    payload.update(overrides)
    return milvus_collection_fingerprint_components(**payload)


def _retrieval_components(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "normalized_dataset_fingerprint": "normalized-sha",
        "retrieval_mode": "hybrid",
        "elasticsearch_index_fingerprint": "es-sha",
        "milvus_collection_fingerprint": "milvus-sha",
        "query_source": {"query_limit": 1},
        "query_embedding": {"model_name": "model-a", "input_field": "query_text"},
        "search_params": {"fusion": {"method": "rrf", "path_topk": 25}},
        "rewrite": None,
        "rerank": {"model_name": "reranker-a"},
        "trace_mode": "replay",
    }
    payload.update(overrides)
    return retrieval_run_fingerprint_components(**payload)


def _metrics_components(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "normalized_dataset_fingerprint": "normalized-sha",
        "retrieval_run_fingerprint": "retrieval-sha",
        "metrics_source": "sci-retrieval-eval",
        "code_git_commit": "code-sha",
        "metrics_entrypoint": "eval_platform.metrics.runner.run_metrics",
        "metric_params": {
            "k_values": [1, 3, 5, 10],
            "main_metric": "ndcg_at_10",
            "projection": {"dedupe_policy": "first_chunk_rank"},
        },
    }
    payload.update(overrides)
    return metrics_run_fingerprint_components(**payload)


def _minimal_eval_fingerprints():
    raw = _fingerprint(
        "raw_dataset",
        raw_dataset_fingerprint_components(
            dataset_name="MiniEval",
            raw_source_uri="memory://mini-eval",
            raw_format="jsonl",
            split="test",
            file_fingerprints=[
                {"path": "corpus.jsonl", "size_bytes": 10, "sha256": "corpus-sha"},
                {"path": "queries.jsonl", "size_bytes": 5, "sha256": "queries-sha"},
                {"path": "qrels.jsonl", "size_bytes": 3, "sha256": "qrels-sha"},
            ],
        ),
    )
    normalized = _fingerprint(
        "normalized_dataset",
        normalized_dataset_fingerprint_components(
            raw_dataset_fingerprint=raw.sha256,
            normalizer_name="mini-normalizer",
            normalizer_version="v1",
            schema_version="normalized_dataset.v1",
            normalizer_params={},
        ),
    )
    chunked = _fingerprint(
        "chunked_corpus",
        _chunked_components(normalized_dataset_fingerprint=normalized.sha256),
    )
    embeddings = _fingerprint(
        "embeddings",
        _embeddings_components(chunked_corpus_fingerprint=chunked.sha256),
    )
    es = _fingerprint(
        "elasticsearch_index",
        _es_components(chunked_corpus_fingerprint=chunked.sha256),
    )
    milvus = _fingerprint(
        "milvus_collection",
        _milvus_components(
            chunked_corpus_fingerprint=chunked.sha256,
            embeddings_fingerprint=embeddings.sha256,
        ),
    )
    retrieval = _fingerprint(
        "retrieval_run",
        _retrieval_components(
            normalized_dataset_fingerprint=normalized.sha256,
            elasticsearch_index_fingerprint=es.sha256,
            milvus_collection_fingerprint=milvus.sha256,
        ),
    )
    metrics = _fingerprint(
        "metrics_run",
        _metrics_components(
            normalized_dataset_fingerprint=normalized.sha256,
            retrieval_run_fingerprint=retrieval.sha256,
        ),
    )
    return {
        "raw_dataset": raw,
        "normalized_dataset": normalized,
        "chunked_corpus": chunked,
        "embeddings": embeddings,
        "elasticsearch_index": es,
        "milvus_collection": milvus,
        "retrieval_run": retrieval,
        "metrics_run": metrics,
    }
