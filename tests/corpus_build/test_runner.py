"""Tests for corpus build runner orchestration."""

from __future__ import annotations

import io
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO
from urllib.parse import unquote, urlparse

import pytest

from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking import (
    ChunkingRunConfig,
    ChunkRecord,
    ProgressEvent,
    ProgressReporter,
)
from eval_platform.corpus_build import (
    CORPUS_BUILD_ARTIFACT_TYPE,
    CorpusBuildConfig,
    CorpusBuildError,
    RawSourceSpec,
    default_corpus_build_artifact_ids,
    run_corpus_build,
)
from eval_platform.datasets import NormalizedDataset, RawToNormalizedConfig
from eval_platform.embeddings import EmbeddingClient, EmbeddingRunConfig, FakeEmbeddingClient
from eval_platform.indexes import (
    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
    MILVUS_COLLECTION_ARTIFACT_TYPE,
    ElasticsearchBulkAction,
    ElasticsearchBulkResult,
    ElasticsearchIngestConfig,
    MilvusIngestConfig,
    MilvusInsertResult,
    MilvusRow,
)


class LocalUriOpener:
    def open(self, uri: str) -> BinaryIO:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise AssertionError(f"unexpected uri: {uri}")
        return Path(unquote(parsed.path)).open("rb")


class FakeChunker:
    def chunk_corpus(self, dataset: NormalizedDataset) -> list[ChunkRecord]:
        corpus = dataset.corpus
        return [
            ChunkRecord(
                chunk_id=f"chunk-{record.doc_id}",
                doc_id=record.doc_id,
                title=record.title,
                text=record.text,
                chunk_index=0,
                metadata={"source": "fake_chunker"},
            )
            for record in corpus
        ]


class WrongCountEmbeddingClient:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts[:-1]]


class FakeElasticsearchClient:
    def __init__(self, *, fail_bulk: bool = False) -> None:
        self.fail_bulk = fail_bulk
        self.created_indices: list[str] = []
        self.bulk_calls: list[list[ElasticsearchBulkAction]] = []
        self.documents: dict[str, dict[str, object]] = {}

    def index_exists(self, index_name: str) -> bool:
        return False

    def create_index(self, index_name: str, body: dict[str, object]) -> None:
        self.created_indices.append(index_name)

    def delete_index(self, index_name: str) -> None:
        raise AssertionError("delete should not be called")

    def bulk_index(
        self,
        index_name: str,
        actions: Sequence[ElasticsearchBulkAction],
    ) -> ElasticsearchBulkResult:
        self.bulk_calls.append(list(actions))
        if self.fail_bulk:
            return ElasticsearchBulkResult(indexed_count=0)
        for action in actions:
            self.documents[action.document_id] = action.document
        return ElasticsearchBulkResult(indexed_count=len(actions))

    def refresh_index(self, index_name: str) -> None:
        return None

    def count_documents(self, index_name: str) -> int:
        return len(self.documents)


class FakeMilvusClient:
    def __init__(self, *, fail_insert: bool = False) -> None:
        self.fail_insert = fail_insert
        self.created_collections: list[str] = []
        self.insert_calls: list[list[MilvusRow]] = []
        self.rows: dict[str, dict[str, object]] = {}

    def collection_exists(self, collection_name: str) -> bool:
        return False

    def create_collection(
        self,
        collection_name: str,
        schema: dict[str, object],
        index_params: dict[str, object],
    ) -> None:
        self.created_collections.append(collection_name)

    def drop_collection(self, collection_name: str) -> None:
        raise AssertionError("drop should not be called")

    def insert_rows(
        self,
        collection_name: str,
        rows: Sequence[MilvusRow],
    ) -> MilvusInsertResult:
        self.insert_calls.append(list(rows))
        if self.fail_insert:
            return MilvusInsertResult(inserted_count=0)
        for row in rows:
            self.rows[row.primary_key] = row.row
        return MilvusInsertResult(inserted_count=len(rows))

    def flush_collection(self, collection_name: str) -> None:
        return None

    def count_entities(self, collection_name: str) -> int:
        return len(self.rows)


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        '{"_id":"doc-1","title":"Doc 1","text":"First document."}\n'
        '{"_id":"doc-2","title":"Doc 2","text":"Second document."}\n',
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        '{"_id":"q-1","text":"first query"}\n',
        encoding="utf-8",
    )
    (root / "instructions.jsonl").write_text(
        '{"query-id":"q-1","instruction":"Find relevant documents."}\n',
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq-1\tdoc-1\t1\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def chunker_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "chunker_repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "README.md").write_text("fake chunker\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


def _configs(
    *,
    raw_dir: Path,
    chunker_repo: Path,
    run_id: str = "test_ifir",
    dataset_name: str = "IFIRNFCorpus",
    enable_elasticsearch: bool = True,
    enable_milvus: bool = True,
) -> tuple[
    CorpusBuildConfig,
    RawToNormalizedConfig,
    ChunkingRunConfig,
    EmbeddingRunConfig,
    ElasticsearchIngestConfig | None,
    MilvusIngestConfig | None,
]:
    ids = default_corpus_build_artifact_ids(
        run_id,
        enable_elasticsearch=enable_elasticsearch,
        enable_milvus=enable_milvus,
    )
    build_config = CorpusBuildConfig(
        run_id=run_id,
        dataset_name=dataset_name,
        raw_source=RawSourceSpec(
            source_type="local_dir",
            uri=str(raw_dir),
            import_parameters={"api_key": "secret", "format": "ifir"},
        ),
        enable_elasticsearch=enable_elasticsearch,
        enable_milvus=enable_milvus,
        metadata={"password": "secret", "stage": "test"},
    )
    raw_to_normalized_config = RawToNormalizedConfig(
        source_artifact_id=ids.raw_dataset,
        output_artifact_id=ids.normalized_dataset,
        dataset_name=dataset_name,
        created_by="test",
    )
    chunking_config = ChunkingRunConfig(
        source_artifact_id=ids.normalized_dataset,
        output_artifact_id=ids.chunked_corpus,
        chunker_name="fake-chunker",
        chunker_repo_path=str(chunker_repo),
        file_record_num=1,
        created_by="test",
    )
    embedding_config = EmbeddingRunConfig(
        source_artifact_id=ids.chunked_corpus,
        output_artifact_id=ids.embeddings,
        model_name="fake-embedding-model",
        embedding_dim=2,
        provider="fake",
        created_by="test",
    )
    elasticsearch_config = (
        ElasticsearchIngestConfig(
            source_artifact_id=ids.chunked_corpus,
            output_artifact_id=ids.elasticsearch_index or "",
            index_name="test-ifir-es",
            bulk_size=1,
            created_by="test",
        )
        if enable_elasticsearch
        else None
    )
    milvus_config = (
        MilvusIngestConfig(
            chunked_corpus_artifact_id=ids.chunked_corpus,
            embeddings_artifact_id=ids.embeddings,
            output_artifact_id=ids.milvus_collection or "",
            collection_name="test_ifir_milvus",
            batch_size=1,
            vector_dim=2,
            created_by="test",
        )
        if enable_milvus
        else None
    )
    return (
        build_config,
        raw_to_normalized_config,
        chunking_config,
        embedding_config,
        elasticsearch_config,
        milvus_config,
    )


def _run_build(
    store: LocalArtifactStore,
    *,
    raw_dir: Path,
    chunker_repo: Path,
    dataset_name: str = "IFIRNFCorpus",
    enable_elasticsearch: bool = True,
    enable_milvus: bool = True,
    embedding_client: EmbeddingClient | None = None,
    elasticsearch_client: FakeElasticsearchClient | None = None,
    milvus_client: FakeMilvusClient | None = None,
    progress_reporter: ProgressReporter | None = None,
):
    (
        build_config,
        raw_to_normalized_config,
        chunking_config,
        embedding_config,
        elasticsearch_config,
        milvus_config,
    ) = _configs(
        raw_dir=raw_dir,
        chunker_repo=chunker_repo,
        dataset_name=dataset_name,
        enable_elasticsearch=enable_elasticsearch,
        enable_milvus=enable_milvus,
    )
    return run_corpus_build(
        store,
        build_config,
        raw_file_opener=LocalUriOpener(),
        chunker=FakeChunker(),
        chunking_config=chunking_config,
        embedding_client=embedding_client or FakeEmbeddingClient(2),
        embedding_config=embedding_config,
        raw_to_normalized_config=raw_to_normalized_config,
        elasticsearch_client=elasticsearch_client,
        elasticsearch_config=elasticsearch_config,
        milvus_client=milvus_client,
        milvus_config=milvus_config,
        progress_reporter=progress_reporter,
    )


def test_default_corpus_build_artifact_ids() -> None:
    ids = default_corpus_build_artifact_ids("run-1")

    assert ids.raw_dataset == "run-1_raw"
    assert ids.normalized_dataset == "run-1_normalized"
    assert ids.chunked_corpus == "run-1_chunks"
    assert ids.embeddings == "run-1_embeddings"
    assert ids.elasticsearch_index == "run-1_es_index"
    assert ids.milvus_collection == "run-1_milvus_collection"


def test_default_corpus_build_artifact_ids_respect_disabled_stages() -> None:
    ids = default_corpus_build_artifact_ids(
        "run-1",
        enable_elasticsearch=False,
        enable_milvus=False,
    )

    assert ids.elasticsearch_index is None
    assert ids.milvus_collection is None


@pytest.mark.parametrize(
    "dataset_name",
    ["IFIRNFCorpus", "IFIRScifact", "NFCorpus", "SciFact", "LitSearchRetrieval"],
)
def test_corpus_build_config_accepts_registered_raw_normalizer_datasets(
    raw_dir: Path,
    dataset_name: str,
) -> None:
    config = CorpusBuildConfig(
        run_id="test",
        dataset_name=dataset_name,
        raw_source=RawSourceSpec(source_type="local_dir", uri=str(raw_dir)),
    )

    assert config.dataset_name == dataset_name


def test_corpus_build_config_rejects_unsupported_dataset(raw_dir: Path) -> None:
    with pytest.raises(ValueError, match="registered raw normalizer datasets"):
        CorpusBuildConfig(
            run_id="test",
            dataset_name="UnknownDataset",
            raw_source=RawSourceSpec(source_type="local_dir", uri=str(raw_dir)),
        )


def test_run_corpus_build_happy_path_calls_stages_in_order(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    es_client = FakeElasticsearchClient()
    milvus_client = FakeMilvusClient()

    manifest = _run_build(
        store,
        raw_dir=raw_dir,
        chunker_repo=chunker_repo,
        elasticsearch_client=es_client,
        milvus_client=milvus_client,
    )

    assert store.is_complete(CORPUS_BUILD_ARTIFACT_TYPE, "test_ifir") is True
    assert manifest.metadata["run_id"] == "test_ifir"
    assert [stage["stage"] for stage in manifest.metadata["stage_manifests"]] == [
        "raw_import",
        "raw_to_normalized",
        "chunking",
        "embedding",
        "elasticsearch_ingest",
        "milvus_ingest",
    ]
    assert es_client.created_indices == ["test-ifir-es"]
    assert milvus_client.created_collections == ["test_ifir_milvus"]


def test_run_corpus_build_manifest_dependencies_include_enabled_stages(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    manifest = _run_build(
        store,
        raw_dir=raw_dir,
        chunker_repo=chunker_repo,
        elasticsearch_client=FakeElasticsearchClient(),
        milvus_client=FakeMilvusClient(),
    )

    assert [dependency.artifact_type for dependency in manifest.dependencies] == [
        "raw_dataset",
        "normalized_dataset",
        "chunked_corpus",
        "embeddings",
        "elasticsearch_index",
        "milvus_collection",
    ]
    assert [dependency.artifact_id for dependency in manifest.dependencies] == [
        "test_ifir_raw",
        "test_ifir_normalized",
        "test_ifir_chunks",
        "test_ifir_embeddings",
        "test_ifir_es_index",
        "test_ifir_milvus_collection",
    ]


def test_run_corpus_build_can_disable_elasticsearch(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    manifest = _run_build(
        store,
        raw_dir=raw_dir,
        chunker_repo=chunker_repo,
        enable_elasticsearch=False,
        milvus_client=FakeMilvusClient(),
    )

    assert ELASTICSEARCH_INDEX_ARTIFACT_TYPE not in [
        dependency.artifact_type for dependency in manifest.dependencies
    ]
    assert manifest.metadata["enabled_stages"]["elasticsearch"] is False


def test_run_corpus_build_can_disable_milvus(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    manifest = _run_build(
        store,
        raw_dir=raw_dir,
        chunker_repo=chunker_repo,
        enable_milvus=False,
        elasticsearch_client=FakeElasticsearchClient(),
    )

    assert MILVUS_COLLECTION_ARTIFACT_TYPE not in [
        dependency.artifact_type for dependency in manifest.dependencies
    ]
    assert manifest.metadata["enabled_stages"]["milvus"] is False


def test_run_corpus_build_non_ifir_dataset_not_blocked_by_runner(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    manifest = _run_build(
        store,
        raw_dir=raw_dir,
        chunker_repo=chunker_repo,
        dataset_name="SciFact",
        enable_elasticsearch=False,
        enable_milvus=False,
    )

    assert manifest.metadata["dataset_name"] == "SciFact"
    assert [dependency.artifact_type for dependency in manifest.dependencies] == [
        "raw_dataset",
        "normalized_dataset",
        "chunked_corpus",
        "embeddings",
    ]


def test_run_corpus_build_rejects_stage_config_artifact_id_mismatch(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    (
        build_config,
        raw_to_normalized_config,
        chunking_config,
        embedding_config,
        elasticsearch_config,
        milvus_config,
    ) = _configs(raw_dir=raw_dir, chunker_repo=chunker_repo)
    embedding_config.source_artifact_id = "wrong"

    with pytest.raises(CorpusBuildError, match="embedding.source_artifact_id"):
        run_corpus_build(
            store,
            build_config,
            raw_file_opener=LocalUriOpener(),
            chunker=FakeChunker(),
            chunking_config=chunking_config,
            embedding_client=FakeEmbeddingClient(2),
            embedding_config=embedding_config,
            raw_to_normalized_config=raw_to_normalized_config,
            elasticsearch_client=FakeElasticsearchClient(),
            elasticsearch_config=elasticsearch_config,
            milvus_client=FakeMilvusClient(),
            milvus_config=milvus_config,
        )

    assert store.is_complete(CORPUS_BUILD_ARTIFACT_TYPE, "test_ifir") is False


def test_run_corpus_build_raw_import_failure_does_not_write_success(
    store: LocalArtifactStore,
    tmp_path: Path,
    chunker_repo: Path,
) -> None:
    missing_raw_dir = tmp_path / "missing"

    with pytest.raises(CorpusBuildError, match="raw_import"):
        _run_build(
            store,
            raw_dir=missing_raw_dir,
            chunker_repo=chunker_repo,
            elasticsearch_client=FakeElasticsearchClient(),
            milvus_client=FakeMilvusClient(),
        )

    assert store.is_complete(CORPUS_BUILD_ARTIFACT_TYPE, "test_ifir") is False


def test_run_corpus_build_embedding_failure_keeps_prior_stage_artifacts(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    with pytest.raises(CorpusBuildError, match="embedding"):
        _run_build(
            store,
            raw_dir=raw_dir,
            chunker_repo=chunker_repo,
            embedding_client=WrongCountEmbeddingClient(),
            elasticsearch_client=FakeElasticsearchClient(),
            milvus_client=FakeMilvusClient(),
        )

    assert store.is_complete(CORPUS_BUILD_ARTIFACT_TYPE, "test_ifir") is False
    assert store.is_complete("raw_dataset", "test_ifir_raw") is True
    assert store.is_complete("normalized_dataset", "test_ifir_normalized") is True
    assert store.is_complete("chunked_corpus", "test_ifir_chunks") is True


def test_run_corpus_build_es_failure_does_not_write_success(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    with pytest.raises(CorpusBuildError, match="elasticsearch_ingest"):
        _run_build(
            store,
            raw_dir=raw_dir,
            chunker_repo=chunker_repo,
            elasticsearch_client=FakeElasticsearchClient(fail_bulk=True),
            milvus_client=FakeMilvusClient(),
        )

    assert store.is_complete(CORPUS_BUILD_ARTIFACT_TYPE, "test_ifir") is False


def test_run_corpus_build_milvus_failure_does_not_write_success(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    with pytest.raises(CorpusBuildError, match="milvus_ingest"):
        _run_build(
            store,
            raw_dir=raw_dir,
            chunker_repo=chunker_repo,
            elasticsearch_client=FakeElasticsearchClient(),
            milvus_client=FakeMilvusClient(fail_insert=True),
        )

    assert store.is_complete(CORPUS_BUILD_ARTIFACT_TYPE, "test_ifir") is False


def test_run_corpus_build_reports_progress(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    events: list[ProgressEvent] = []

    _run_build(
        store,
        raw_dir=raw_dir,
        chunker_repo=chunker_repo,
        elasticsearch_client=FakeElasticsearchClient(),
        milvus_client=FakeMilvusClient(),
        progress_reporter=events.append,
    )

    corpus_events = [event for event in events if event.stage == "corpus_build"]
    assert "stage_start" in [event.metadata.get("kind") for event in corpus_events]
    assert "stage_done" in [event.metadata.get("kind") for event in corpus_events]
    assert "run_done" in [event.metadata.get("kind") for event in corpus_events]


def test_run_corpus_build_progress_failure_does_not_write_success(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    def fail_reporter(_: ProgressEvent) -> None:
        raise RuntimeError("progress failed")

    with pytest.raises(CorpusBuildError, match="raw_import"):
        _run_build(
            store,
            raw_dir=raw_dir,
            chunker_repo=chunker_repo,
            elasticsearch_client=FakeElasticsearchClient(),
            milvus_client=FakeMilvusClient(),
            progress_reporter=fail_reporter,
        )

    assert store.is_complete(CORPUS_BUILD_ARTIFACT_TYPE, "test_ifir") is False


def test_run_corpus_build_final_manifest_does_not_contain_secrets(
    store: LocalArtifactStore,
    raw_dir: Path,
    chunker_repo: Path,
) -> None:
    manifest = _run_build(
        store,
        raw_dir=raw_dir,
        chunker_repo=chunker_repo,
        elasticsearch_client=FakeElasticsearchClient(),
        milvus_client=FakeMilvusClient(),
    )

    manifest_text = str(manifest.model_dump(mode="json")).lower()
    assert "password" not in manifest_text
    assert "token" not in manifest_text
    assert "access_key" not in manifest_text
    assert "api_key" not in manifest_text
    assert manifest.metadata["stage"] == "test"


def test_run_corpus_build_supports_s3_prefix_with_fake_client(
    store: LocalArtifactStore,
    tmp_path: Path,
    chunker_repo: Path,
) -> None:
    class FakeBody(io.BytesIO):
        pass

    class FakeS3Client:
        objects = {
            "prefix/corpus.jsonl": (
                b'{"_id":"doc-1","title":"Doc 1","text":"First document."}\n'
            ),
            "prefix/queries.jsonl": b'{"_id":"q-1","text":"first query"}\n',
            "prefix/instructions.jsonl": (
                b'{"query-id":"q-1","instruction":"Find relevant documents."}\n'
            ),
            "prefix/qrels/test.tsv": b"query-id\tcorpus-id\tscore\nq-1\tdoc-1\t1\n",
        }

        def list_objects_v2(self, **kwargs: object) -> dict[str, object]:
            return {
                "Contents": [{"Key": key} for key in sorted(self.objects)],
                "IsTruncated": False,
            }

        def get_object(self, **kwargs: object) -> dict[str, object]:
            return {"Body": FakeBody(self.objects[str(kwargs["Key"])])}

    class FakeS3Opener:
        def open(self, uri: str) -> BinaryIO:
            key = uri.removeprefix("s3://bucket/")
            return FakeBody(FakeS3Client.objects[key])

    ids = default_corpus_build_artifact_ids(
        "test_ifir_s3",
        enable_elasticsearch=False,
        enable_milvus=False,
    )
    build_config = CorpusBuildConfig(
        run_id="test_ifir_s3",
        raw_source=RawSourceSpec(source_type="s3_prefix", uri="s3://bucket/prefix"),
        enable_elasticsearch=False,
        enable_milvus=False,
    )
    raw_to_normalized_config = RawToNormalizedConfig(
        source_artifact_id=ids.raw_dataset,
        output_artifact_id=ids.normalized_dataset,
        dataset_name="IFIRNFCorpus",
    )
    chunking_config = ChunkingRunConfig(
        source_artifact_id=ids.normalized_dataset,
        output_artifact_id=ids.chunked_corpus,
        chunker_name="fake-chunker",
        chunker_repo_path=str(chunker_repo),
    )
    embedding_config = EmbeddingRunConfig(
        source_artifact_id=ids.chunked_corpus,
        output_artifact_id=ids.embeddings,
        model_name="fake",
        embedding_dim=2,
    )

    manifest = run_corpus_build(
        store,
        build_config,
        raw_import_client=FakeS3Client(),
        raw_file_opener=FakeS3Opener(),  # type: ignore[arg-type]
        chunker=FakeChunker(),
        chunking_config=chunking_config,
        embedding_client=FakeEmbeddingClient(2),
        embedding_config=embedding_config,
        raw_to_normalized_config=raw_to_normalized_config,
    )

    assert store.is_complete(CORPUS_BUILD_ARTIFACT_TYPE, "test_ifir_s3") is True
    assert manifest.metadata["raw_source"]["source_type"] == "s3_prefix"


def test_run_corpus_build_requires_s3_client_for_s3_prefix(
    store: LocalArtifactStore,
    chunker_repo: Path,
) -> None:
    ids = default_corpus_build_artifact_ids(
        "test_ifir_s3",
        enable_elasticsearch=False,
        enable_milvus=False,
    )
    build_config = CorpusBuildConfig(
        run_id="test_ifir_s3",
        raw_source=RawSourceSpec(source_type="s3_prefix", uri="s3://bucket/prefix"),
        enable_elasticsearch=False,
        enable_milvus=False,
    )

    with pytest.raises(CorpusBuildError, match="raw_import_client"):
        run_corpus_build(
            store,
            build_config,
            raw_file_opener=LocalUriOpener(),
            chunker=FakeChunker(),
            chunking_config=ChunkingRunConfig(
                source_artifact_id=ids.normalized_dataset,
                output_artifact_id=ids.chunked_corpus,
                chunker_name="fake-chunker",
                chunker_repo_path=str(chunker_repo),
            ),
            embedding_client=FakeEmbeddingClient(2),
            embedding_config=EmbeddingRunConfig(
                source_artifact_id=ids.chunked_corpus,
                output_artifact_id=ids.embeddings,
                model_name="fake",
                embedding_dim=2,
            ),
            raw_to_normalized_config=RawToNormalizedConfig(
                source_artifact_id=ids.raw_dataset,
                output_artifact_id=ids.normalized_dataset,
                dataset_name="IFIRNFCorpus",
            ),
        )
