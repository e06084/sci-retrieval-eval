"""Tests for asset fingerprints written into artifact manifests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from eval_platform.artifacts import (
    METADATA_KEY_ASSET_FINGERPRINT,
    METADATA_KEY_ASSET_FINGERPRINT_SHA256,
    ArtifactManifest,
    LocalArtifactStore,
)
from eval_platform.chunking import (
    ChunkedCorpus,
    ChunkerProvenance,
    ChunkRecord,
    write_chunked_corpus_artifact,
)
from eval_platform.datasets import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
    RawDatasetFile,
    RawDatasetSnapshot,
    build_content_fingerprint_sha256,
    write_normalized_dataset_artifact,
    write_raw_dataset_artifact,
)
from eval_platform.embeddings import (
    EmbeddedCorpus,
    EmbeddingProvenance,
    EmbeddingRecord,
    write_embeddings_artifact,
)
from eval_platform.indexes import (
    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
    MILVUS_COLLECTION_ARTIFACT_TYPE,
)
from eval_platform.metrics import MetricsRunConfig, run_metrics
from eval_platform.retrieval import RetrievalHit, RetrievalRunConfig, run_retrieval


def test_raw_dataset_manifest_fingerprint_ignores_artifact_instance_fields(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    files = [
        RawDatasetFile(
            path="corpus.jsonl",
            uri="s3://bucket/raw/corpus.jsonl",
            size_bytes=10,
            sha256="corpus-sha",
        )
    ]
    snapshot = RawDatasetSnapshot(
        source_type="s3_prefix",
        source_uri="s3://bucket/raw",
        dataset_name="NFCorpus",
        files=files,
        content_fingerprint_sha256=build_content_fingerprint_sha256(files),
        import_parameters={"raw_format": "jsonl_tsv", "split": "test"},
    )

    first = write_raw_dataset_artifact(
        store,
        "raw-a",
        snapshot,
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    second = write_raw_dataset_artifact(
        store,
        "raw-b",
        snapshot,
        created_at=datetime(2026, 5, 2, tzinfo=UTC),
    )

    assert _asset_sha(first) == _asset_sha(second)
    assert first.metadata[METADATA_KEY_ASSET_FINGERPRINT]["artifact_type"] == "raw_dataset"


def test_normalized_chunked_and_embeddings_manifests_write_asset_fingerprints(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    normalized = write_normalized_dataset_artifact(
        store,
        "normalized",
        NormalizedDataset(
            corpus=[CorpusRecord(doc_id="doc-1", text="alpha")],
            queries=[QueryRecord(query_id="q-1", text="alpha")],
            qrels=[QrelRecord(query_id="q-1", doc_id="doc-1", relevance=1.0)],
            metadata={
                "raw_dataset_fingerprint": "raw-sha",
                "normalizer_name": "nfcorpus_raw_jsonl_tsv_v1",
                "normalizer_version": "1",
                "normalized_schema_version": "1",
                "normalizer_params": {"split": "test", "raw_format": "jsonl_tsv"},
            },
        ),
    )
    chunked = write_chunked_corpus_artifact(
        store,
        "chunks",
        ChunkedCorpus(
            chunks=[
                ChunkRecord(
                    chunk_id="chunk-1",
                    doc_id="doc-1",
                    text="alpha",
                    chunk_index=0,
                )
            ]
        ),
        metadata={"normalized_dataset_fingerprint": _asset_sha(normalized)},
        chunker=ChunkerProvenance(
            name="fake-chunker",
            repo_url="https://github.com/example/chunker.git",
            commit_sha="abc123",
        ),
        chunk_params={"chunk_size": 512},
    )
    embeddings = write_embeddings_artifact(
        store,
        "embeddings",
        EmbeddedCorpus(
            embeddings=[
                EmbeddingRecord(chunk_id="chunk-1", doc_id="doc-1", vector=[0.1, 0.2])
            ]
        ),
        provenance=EmbeddingProvenance(
            model_name="bge-m3",
            provider="sciverse_internal",
            embedding_dim=2,
            normalized=True,
            endpoint_id="embedding-3886",
        ),
        source_artifact_id="chunks",
        metadata={"chunked_corpus_fingerprint": _asset_sha(chunked)},
    )

    assert _asset_sha(normalized)
    assert _asset_sha(chunked)
    assert _asset_sha(embeddings)
    assert embeddings.metadata[METADATA_KEY_ASSET_FINGERPRINT]["components"][
        "chunked_corpus_fingerprint"
    ] == _asset_sha(chunked)


def test_retrieval_and_metrics_fingerprints_use_logical_assets_not_resource_names(
    tmp_path: Path,
) -> None:
    store = LocalArtifactStore(tmp_path)
    normalized = write_normalized_dataset_artifact(
        store,
        "normalized",
        NormalizedDataset(
            corpus=[CorpusRecord(doc_id="doc-1", text="alpha document")],
            queries=[QueryRecord(query_id="q-1", text="alpha query")],
            qrels=[QrelRecord(query_id="q-1", doc_id="doc-1", relevance=1.0)],
            metadata={
                "raw_dataset_fingerprint": "raw-sha",
                "normalizer_name": "mini",
                "normalizer_version": "1",
                "normalized_schema_version": "1",
            },
        ),
    )
    _write_manifest_with_asset_sha(
        store,
        ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
        "es-index-artifact",
        "es-fingerprint",
    )
    _write_manifest_with_asset_sha(
        store,
        MILVUS_COLLECTION_ARTIFACT_TYPE,
        "milvus-collection-artifact",
        "milvus-fingerprint",
    )

    first = _run_retrieval(
        store,
        "retrieval-a",
        index_name="physical-index-a",
        collection_name="physical-collection-a",
    )
    second = _run_retrieval(
        store,
        "retrieval-b",
        index_name="physical-index-b",
        collection_name="physical-collection-b",
    )
    metrics = run_metrics(
        store,
        store,
        MetricsRunConfig(
            source_normalized_dataset_artifact_id="normalized",
            source_retrieval_run_artifact_id="retrieval-a",
            output_artifact_id="metrics-a",
        ),
    )

    assert _asset_sha(normalized)
    assert _asset_sha(first) == _asset_sha(second)
    assert _asset_sha(metrics)
    assert first.metadata[METADATA_KEY_ASSET_FINGERPRINT]["components"][
        "normalized_dataset_fingerprint"
    ] == _asset_sha(normalized)
    assert metrics.metadata[METADATA_KEY_ASSET_FINGERPRINT]["components"][
        "retrieval_run_fingerprint"
    ] == _asset_sha(first)


def _run_retrieval(
    store: LocalArtifactStore,
    output_artifact_id: str,
    *,
    index_name: str,
    collection_name: str,
) -> ArtifactManifest:
    return run_retrieval(
        store,
        store,
        RetrievalRunConfig(
            source_normalized_dataset_artifact_id="normalized",
            output_artifact_id=output_artifact_id,
            retrieval_mode="hybrid",
            top_k=2,
            query_limit=1,
            elasticsearch_index_artifact_id="es-index-artifact",
            milvus_collection_artifact_id="milvus-collection-artifact",
            index_name=index_name,
            collection_name=collection_name,
            metadata={
                "query_embedding": {
                    "embedding_source": "sciverse_internal",
                    "model_name": "bge-m3",
                    "endpoint_alias": "embedding-3886",
                    "embedding_dim": 2,
                    "input_field": "query_text",
                }
            },
        ),
        es_client=MiniElasticsearchClient(),
        milvus_client=MiniMilvusClient(),
        embedding_client=MiniEmbeddingClient(),
    )


def _write_manifest_with_asset_sha(
    store: LocalArtifactStore,
    artifact_type: str,
    artifact_id: str,
    sha256: str,
) -> None:
    manifest = ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
        metadata={METADATA_KEY_ASSET_FINGERPRINT_SHA256: sha256},
    )
    store.write_manifest(artifact_type, artifact_id, manifest)
    store.mark_success(artifact_type, artifact_id)


def _asset_sha(manifest: ArtifactManifest) -> str:
    value = manifest.metadata[METADATA_KEY_ASSET_FINGERPRINT_SHA256]
    assert isinstance(value, str)
    return value


class MiniEmbeddingClient:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, float(len(text))] for text in texts]


class MiniElasticsearchClient:
    def search_bm25(self, index_name: str, query: str, top_k: int) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                chunk_id="chunk-1",
                doc_id="doc-1",
                text="alpha document",
                score=10.0,
                recall_source="es",
            )
        ][:top_k]

    def enrich_by_chunk_ids(
        self,
        index_name: str,
        hits: Sequence[RetrievalHit],
    ) -> list[RetrievalHit]:
        return list(hits)


class MiniMilvusClient:
    def search(
        self,
        collection_name: str,
        vector: Sequence[float],
        top_k: int,
    ) -> list[RetrievalHit]:
        return [
            RetrievalHit(
                chunk_id="chunk-1",
                doc_id="doc-1",
                text="alpha document",
                score=0.9,
                recall_source="milvus",
            )
        ][:top_k]
