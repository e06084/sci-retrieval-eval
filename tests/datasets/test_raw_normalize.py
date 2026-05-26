"""Tests for raw snapshot to normalized dataset conversion."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

import eval_platform.datasets.raw_normalize as raw_normalize_module
from eval_platform.artifacts import LocalArtifactStore
from eval_platform.chunking.progress import ProgressEvent
from eval_platform.datasets import (
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    RAW_DATASET_ARTIFACT_TYPE,
    RawDatasetFile,
    RawDatasetSnapshot,
    RawNormalizeError,
    RawToNormalizedConfig,
    S3RawFileOpener,
    build_content_fingerprint_sha256,
    normalize_raw_dataset_artifact,
    read_normalized_dataset_artifact,
    write_raw_dataset_artifact,
)


class RecordingBinaryStream(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.read_calls: list[int] = []
        self.iterated = False

    def read(self, size: int | None = -1) -> bytes:
        recorded_size = -1 if size is None else size
        self.read_calls.append(recorded_size)
        return super().read(size)

    def __iter__(self) -> RecordingBinaryStream:
        self.iterated = True
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if line == b"":
            raise StopIteration
        return line


class FakeRawFileOpener:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.streams: dict[str, RecordingBinaryStream] = {}

    def open(self, uri: str) -> RecordingBinaryStream:
        stream = RecordingBinaryStream(self.payloads[uri])
        self.streams[uri] = stream
        return stream


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, io.BytesIO]:
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


@pytest.fixture
def store(tmp_path: Path) -> LocalArtifactStore:
    return LocalArtifactStore(tmp_path)


def _jsonl_tsv_snapshot(
    *,
    dataset_name: str = "IFIRNFCorpus",
    slug: str = "ifir_nfcorpus",
    has_instructions: bool = True,
) -> tuple[RawDatasetSnapshot, FakeRawFileOpener]:
    uris_to_payloads = {
        f"s3://bucket/raw/{slug}/corpus.jsonl": (
            b'{"_id":"d1","title":"Title 1","text":"Doc 1"}\n'
            b'{"_id":"d2","title":"Title 2","text":"Doc 2"}\n'
        ),
        f"s3://bucket/raw/{slug}/queries.jsonl": (
            b'{"_id":"q1","text":"Query 1"}\n'
            b'{"_id":"q2","text":"Query 2"}\n'
        ),
        f"s3://bucket/raw/{slug}/qrels/test.tsv": (
            b"query-id\tcorpus-id\tscore\n"
            b"q1\td1\t1\n"
            b"q2\td2\t2\n"
        ),
    }
    if has_instructions:
        uris_to_payloads[f"s3://bucket/raw/{slug}/instructions.jsonl"] = (
            b'{"query-id":"q1","instruction":"Instruction 1"}\n'
            b'{"query-id":"q2","instruction":"Instruction 2"}\n'
        )
    files = [
        RawDatasetFile(
            path="corpus.jsonl",
            uri=f"s3://bucket/raw/{slug}/corpus.jsonl",
            size_bytes=len(uris_to_payloads[f"s3://bucket/raw/{slug}/corpus.jsonl"]),
            sha256="0" * 64,
        ),
        RawDatasetFile(
            path="qrels/test.tsv",
            uri=f"s3://bucket/raw/{slug}/qrels/test.tsv",
            size_bytes=len(uris_to_payloads[f"s3://bucket/raw/{slug}/qrels/test.tsv"]),
            sha256="1" * 64,
        ),
        RawDatasetFile(
            path="queries.jsonl",
            uri=f"s3://bucket/raw/{slug}/queries.jsonl",
            size_bytes=len(uris_to_payloads[f"s3://bucket/raw/{slug}/queries.jsonl"]),
            sha256="2" * 64,
        ),
    ]
    if has_instructions:
        files.append(
            RawDatasetFile(
                path="instructions.jsonl",
                uri=f"s3://bucket/raw/{slug}/instructions.jsonl",
                size_bytes=len(uris_to_payloads[f"s3://bucket/raw/{slug}/instructions.jsonl"]),
                sha256="3" * 64,
            )
        )
    snapshot = RawDatasetSnapshot(
        source_type="s3_prefix",
        source_uri=f"s3://bucket/raw/{slug}",
        dataset_name=dataset_name,
        files=files,
        content_fingerprint_sha256=build_content_fingerprint_sha256(files),
        import_parameters={"split": "test"},
    )
    return snapshot, FakeRawFileOpener(uris_to_payloads)


def _ifir_nfcorpus_snapshot() -> tuple[RawDatasetSnapshot, FakeRawFileOpener]:
    return _jsonl_tsv_snapshot()


def _litsearch_snapshot(
    *,
    shard_paths: tuple[str, ...] = (
        "corpus/test-00000-of-00001.parquet",
        "queries/test-00000-of-00001.parquet",
        "qrels/test-00000-of-00001.parquet",
    ),
) -> tuple[RawDatasetSnapshot, FakeRawFileOpener]:
    uris_to_payloads = {
        f"s3://bucket/raw/litsearch/{path}": f"fake-{path}".encode()
        for path in shard_paths
    }
    files = [
        RawDatasetFile(
            path=path,
            uri=f"s3://bucket/raw/litsearch/{path}",
            size_bytes=len(uris_to_payloads[f"s3://bucket/raw/litsearch/{path}"]),
            sha256=f"{index % 16:x}" * 64,
        )
        for index, path in enumerate(shard_paths, start=4)
    ]
    snapshot = RawDatasetSnapshot(
        source_type="s3_prefix",
        source_uri="s3://bucket/raw/litsearch",
        dataset_name="LitSearchRetrieval",
        files=files,
        content_fingerprint_sha256=build_content_fingerprint_sha256(files),
    )
    return snapshot, FakeRawFileOpener(uris_to_payloads)


def test_normalize_raw_dataset_artifact_ifir_nfcorpus(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _ifir_nfcorpus_snapshot()
    write_raw_dataset_artifact(store, "raw_ifir_nfcorpus_001", snapshot)

    manifest = normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id="raw_ifir_nfcorpus_001",
            output_artifact_id="normalized_ifir_nfcorpus_001",
            dataset_name="IFIRNFCorpus",
            metadata={"note": "smoke"},
        ),
        opener=opener,
    )
    loaded = read_normalized_dataset_artifact(store, "normalized_ifir_nfcorpus_001")

    assert store.is_complete(NORMALIZED_DATASET_ARTIFACT_TYPE, "normalized_ifir_nfcorpus_001")
    assert len(loaded.corpus) == 2
    assert len(loaded.queries) == 2
    assert len(loaded.qrels) == 2
    assert loaded.queries[0].metadata["instruction"] == "Instruction 1"
    assert loaded.qrels[1].relevance == 2.0
    assert manifest.dependencies[0].artifact_type == RAW_DATASET_ARTIFACT_TYPE
    assert manifest.dependencies[0].artifact_id == "raw_ifir_nfcorpus_001"
    assert manifest.metadata["source"] == "raw_dataset"
    assert manifest.metadata["task_name"] == "IFIRNFCorpus"
    assert manifest.metadata["split"] == "test"
    assert manifest.metadata["normalizer_name"] == "ifir_nfcorpus_raw_jsonl_tsv_v1"
    assert manifest.metadata["raw_format"] == "jsonl_tsv"
    assert manifest.metadata["has_instructions"] is True
    assert manifest.metadata["corpus_count"] == 2
    assert manifest.metadata["query_count"] == 2
    assert manifest.metadata["qrel_count"] == 2
    assert manifest.metadata["raw_dataset_artifact_id"] == "raw_ifir_nfcorpus_001"
    assert manifest.metadata["raw_dataset_fingerprint"] == snapshot.content_fingerprint_sha256
    assert manifest.metadata["raw_source_uri"] == "s3://bucket/raw/ifir_nfcorpus"
    assert manifest.metadata["normalized_schema_version"] == "1"
    assert manifest.metadata["note"] == "smoke"


@pytest.mark.parametrize(
    ("dataset_name", "slug", "normalizer_name", "has_instructions"),
    [
        ("IFIRNFCorpus", "ifir_nfcorpus", "ifir_nfcorpus_raw_jsonl_tsv_v1", True),
        ("IFIRScifact", "ifir_scifact", "ifir_scifact_raw_jsonl_tsv_v1", True),
        ("NFCorpus", "nfcorpus", "nfcorpus_raw_jsonl_tsv_v1", False),
        ("SciFact", "scifact", "scifact_raw_jsonl_tsv_v1", False),
    ],
)
def test_normalize_jsonl_tsv_raw_datasets(
    store: LocalArtifactStore,
    dataset_name: str,
    slug: str,
    normalizer_name: str,
    has_instructions: bool,
) -> None:
    snapshot, opener = _jsonl_tsv_snapshot(
        dataset_name=dataset_name,
        slug=slug,
        has_instructions=has_instructions,
    )
    write_raw_dataset_artifact(store, f"raw_{slug}_001", snapshot)

    manifest = normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id=f"raw_{slug}_001",
            output_artifact_id=f"normalized_{slug}_001",
            dataset_name=dataset_name,
        ),
        opener=opener,
    )
    loaded = read_normalized_dataset_artifact(store, f"normalized_{slug}_001")

    assert len(loaded.corpus) == 2
    assert len(loaded.queries) == 2
    assert len(loaded.qrels) == 2
    assert manifest.metadata["normalizer_name"] == normalizer_name
    assert manifest.metadata["raw_format"] == "jsonl_tsv"
    assert manifest.metadata["has_instructions"] is has_instructions
    if has_instructions:
        assert loaded.queries[0].metadata["instruction"] == "Instruction 1"
    else:
        assert loaded.queries[0].metadata == {}


def test_normalize_raw_dataset_artifact_streams_corpus_jsonl(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _ifir_nfcorpus_snapshot()
    write_raw_dataset_artifact(store, "raw_ifir_nfcorpus_001", snapshot)

    normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id="raw_ifir_nfcorpus_001",
            output_artifact_id="normalized_ifir_nfcorpus_001",
            dataset_name="IFIRNFCorpus",
        ),
        opener=opener,
    )

    corpus_stream = opener.streams["s3://bucket/raw/ifir_nfcorpus/corpus.jsonl"]
    assert corpus_stream.iterated is True
    assert -1 not in corpus_stream.read_calls


def test_normalize_raw_dataset_artifact_rejects_unknown_normalizer(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _ifir_nfcorpus_snapshot()
    write_raw_dataset_artifact(store, "raw_ifir_nfcorpus_001", snapshot)

    with pytest.raises(RawNormalizeError, match="Raw normalizer mismatch"):
        normalize_raw_dataset_artifact(
            store,
            store,
            RawToNormalizedConfig(
                source_artifact_id="raw_ifir_nfcorpus_001",
                output_artifact_id="normalized_ifir_nfcorpus_001",
                dataset_name="IFIRNFCorpus",
                normalizer_name="unknown",
            ),
            opener=opener,
        )


def test_normalize_raw_dataset_artifact_rejects_unsupported_dataset(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _ifir_nfcorpus_snapshot()
    write_raw_dataset_artifact(store, "raw_unknown_001", snapshot)

    with pytest.raises(RawNormalizeError, match="No raw normalizer"):
        normalize_raw_dataset_artifact(
            store,
            store,
            RawToNormalizedConfig(
                source_artifact_id="raw_unknown_001",
                output_artifact_id="normalized_unknown_001",
                dataset_name="UnknownDataset",
            ),
            opener=opener,
        )


def test_normalize_raw_dataset_artifact_rejects_missing_required_file(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _jsonl_tsv_snapshot(
        dataset_name="NFCorpus",
        slug="nfcorpus",
        has_instructions=False,
    )
    snapshot.files = [file for file in snapshot.files if file.path != "qrels/test.tsv"]
    snapshot.content_fingerprint_sha256 = build_content_fingerprint_sha256(snapshot.files)
    write_raw_dataset_artifact(store, "raw_nfcorpus_001", snapshot)

    with pytest.raises(RawNormalizeError, match="Required raw file missing"):
        normalize_raw_dataset_artifact(
            store,
            store,
            RawToNormalizedConfig(
                source_artifact_id="raw_nfcorpus_001",
                output_artifact_id="normalized_nfcorpus_001",
                dataset_name="NFCorpus",
            ),
            opener=opener,
        )


def test_normalize_litsearch_parquet_dataset(
    store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, opener = _litsearch_snapshot()
    write_raw_dataset_artifact(store, "raw_litsearch_001", snapshot)

    def fake_read_parquet_records(
        file: RawDatasetFile,
        opener: object,
    ) -> list[dict[str, object]]:
        rows_by_path: dict[str, list[dict[str, object]]] = {
            "corpus/test-00000-of-00001.parquet": [
                {"_id": "doc-1", "title": "Doc 1", "text": "First document."}
            ],
            "queries/test-00000-of-00001.parquet": [{"_id": "q-1", "text": "first query"}],
            "qrels/test-00000-of-00001.parquet": [
                {"query-id": "q-1", "corpus-id": "doc-1", "score": 1}
            ],
        }
        return rows_by_path[file.path]

    monkeypatch.setattr(raw_normalize_module, "_read_parquet_records", fake_read_parquet_records)

    manifest = normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id="raw_litsearch_001",
            output_artifact_id="normalized_litsearch_001",
            dataset_name="LitSearchRetrieval",
        ),
        opener=opener,
    )
    loaded = read_normalized_dataset_artifact(store, "normalized_litsearch_001")

    assert loaded.corpus[0].doc_id == "doc-1"
    assert loaded.queries[0].query_id == "q-1"
    assert loaded.qrels[0].relevance == 1.0
    assert manifest.metadata["normalizer_name"] == "litsearch_raw_parquet_v1"
    assert manifest.metadata["raw_format"] == "parquet_dir_shards"
    assert manifest.metadata["has_instructions"] is False
    assert manifest.metadata["raw_source_uri"] == "s3://bucket/raw/litsearch"


def test_normalize_litsearch_parquet_filters_empty_text_docs_and_orphans(
    store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, opener = _litsearch_snapshot()
    write_raw_dataset_artifact(store, "raw_litsearch_001", snapshot)

    def fake_read_parquet_records(
        file: RawDatasetFile,
        opener: object,
    ) -> list[dict[str, object]]:
        rows_by_path: dict[str, list[dict[str, object]]] = {
            "corpus/test-00000-of-00001.parquet": [
                {"_id": "doc-text", "title": "Text title", "text": "Body"},
                {
                    "_id": "doc-abstract",
                    "title": "Abstract title",
                    "text": "",
                    "abstract": "Abstract body",
                },
                {"_id": "doc-title", "title": "Title only", "text": ""},
                {"_id": "doc-empty", "title": " ", "text": "", "abstract": ""},
            ],
            "queries/test-00000-of-00001.parquet": [
                {"_id": "q-keep", "text": "kept query"},
                {"_id": "q-empty-doc-only", "text": "dropped doc query"},
                {"_id": "q-missing-doc-only", "text": "missing doc query"},
                {"_id": "q-no-qrels", "text": "no qrels query"},
            ],
            "qrels/test-00000-of-00001.parquet": [
                {"query-id": "q-keep", "corpus-id": "doc-title", "score": 1},
                {"query-id": "q-keep", "corpus-id": "doc-abstract", "score": 2},
                {"query-id": "q-keep", "corpus-id": "doc-empty", "score": 1},
                {
                    "query-id": "q-empty-doc-only",
                    "corpus-id": "doc-empty",
                    "score": 1,
                },
                {
                    "query-id": "q-missing-doc-only",
                    "corpus-id": "doc-missing",
                    "score": 1,
                },
            ],
        }
        return rows_by_path[file.path]

    monkeypatch.setattr(raw_normalize_module, "_read_parquet_records", fake_read_parquet_records)

    manifest = normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id="raw_litsearch_001",
            output_artifact_id="normalized_litsearch_001",
            dataset_name="LitSearchRetrieval",
        ),
        opener=opener,
    )
    loaded = read_normalized_dataset_artifact(store, "normalized_litsearch_001")

    assert [(record.doc_id, record.text) for record in loaded.corpus] == [
        ("doc-text", "Body"),
        ("doc-abstract", "Abstract body"),
        ("doc-title", "Title only"),
    ]
    assert [query.query_id for query in loaded.queries] == ["q-keep"]
    assert [(qrel.query_id, qrel.doc_id, qrel.relevance) for qrel in loaded.qrels] == [
        ("q-keep", "doc-title", 1.0),
        ("q-keep", "doc-abstract", 2.0),
    ]
    assert manifest.metadata["filtered_corpus_count"] == 3
    assert manifest.metadata["dropped_corpus_count"] == 1
    assert manifest.metadata["dropped_qrel_count"] == 3
    assert manifest.metadata["dropped_query_count"] == 3


def test_normalize_litsearch_parquet_merges_shards_in_path_order(
    store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, opener = _litsearch_snapshot(
        shard_paths=(
            "qrels/test-00001-of-00002.parquet",
            "queries/test-00001-of-00002.parquet",
            "corpus/test-00001-of-00002.parquet",
            "qrels/test-00000-of-00002.parquet",
            "queries/test-00000-of-00002.parquet",
            "corpus/test-00000-of-00002.parquet",
        )
    )
    write_raw_dataset_artifact(store, "raw_litsearch_001", snapshot)
    read_order: list[str] = []

    def fake_read_parquet_records(
        file: RawDatasetFile,
        opener: object,
    ) -> list[dict[str, object]]:
        read_order.append(file.path)
        rows_by_path: dict[str, list[dict[str, object]]] = {
            "corpus/test-00000-of-00002.parquet": [
                {"_id": "doc-0", "title": "Doc 0", "text": "First shard"}
            ],
            "corpus/test-00001-of-00002.parquet": [
                {"_id": "doc-1", "title": "Doc 1", "text": "Second shard"}
            ],
            "queries/test-00000-of-00002.parquet": [{"_id": "q-0", "text": "first query"}],
            "queries/test-00001-of-00002.parquet": [{"_id": "q-1", "text": "second query"}],
            "qrels/test-00000-of-00002.parquet": [
                {"query-id": "q-0", "corpus-id": "doc-0", "score": 1}
            ],
            "qrels/test-00001-of-00002.parquet": [
                {"query-id": "q-1", "corpus-id": "doc-1", "score": 2}
            ],
        }
        return rows_by_path[file.path]

    monkeypatch.setattr(raw_normalize_module, "_read_parquet_records", fake_read_parquet_records)

    normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id="raw_litsearch_001",
            output_artifact_id="normalized_litsearch_001",
            dataset_name="LitSearchRetrieval",
        ),
        opener=opener,
    )
    loaded = read_normalized_dataset_artifact(store, "normalized_litsearch_001")

    assert read_order == [
        "corpus/test-00000-of-00002.parquet",
        "corpus/test-00001-of-00002.parquet",
        "queries/test-00000-of-00002.parquet",
        "queries/test-00001-of-00002.parquet",
        "qrels/test-00000-of-00002.parquet",
        "qrels/test-00001-of-00002.parquet",
    ]
    assert [record.doc_id for record in loaded.corpus] == ["doc-0", "doc-1"]
    assert [record.query_id for record in loaded.queries] == ["q-0", "q-1"]
    assert [record.relevance for record in loaded.qrels] == [1.0, 2.0]


@pytest.mark.parametrize(
    ("missing_directory", "error_pattern"),
    [
        ("corpus", r"corpus/\*\.parquet"),
        ("queries", r"queries/\*\.parquet"),
        ("qrels", r"qrels/\*\.parquet"),
    ],
)
def test_normalize_litsearch_parquet_rejects_missing_shard_group(
    store: LocalArtifactStore,
    missing_directory: str,
    error_pattern: str,
) -> None:
    snapshot, opener = _litsearch_snapshot()
    snapshot.files = [
        file for file in snapshot.files if not file.path.startswith(f"{missing_directory}/")
    ]
    snapshot.content_fingerprint_sha256 = build_content_fingerprint_sha256(snapshot.files)
    write_raw_dataset_artifact(store, "raw_litsearch_001", snapshot)

    with pytest.raises(RawNormalizeError, match=error_pattern):
        normalize_raw_dataset_artifact(
            store,
            store,
            RawToNormalizedConfig(
                source_artifact_id="raw_litsearch_001",
                output_artifact_id="normalized_litsearch_001",
                dataset_name="LitSearchRetrieval",
            ),
            opener=opener,
        )


def test_s3_raw_file_opener_reads_bytes_from_fake_client() -> None:
    client = FakeS3Client()
    client.put_object(Bucket="raw-bucket", Key="path/data.jsonl", Body=b'{"_id":"1"}\n')

    opener = S3RawFileOpener(client=client)

    with opener.open("s3://raw-bucket/path/data.jsonl") as stream:
        assert stream.read() == b'{"_id":"1"}\n'


def test_s3_raw_file_opener_rejects_invalid_uri() -> None:
    opener = S3RawFileOpener(client=FakeS3Client())

    with pytest.raises(RawNormalizeError, match="Unsupported raw file URI"):
        opener.open("file:///tmp/not-s3.jsonl")


def test_normalize_raw_dataset_artifact_reports_progress(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _ifir_nfcorpus_snapshot()
    write_raw_dataset_artifact(store, "raw_ifir_nfcorpus_001", snapshot)
    events: list[ProgressEvent] = []

    normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id="raw_ifir_nfcorpus_001",
            output_artifact_id="normalized_ifir_nfcorpus_001",
            dataset_name="IFIRNFCorpus",
        ),
        opener=opener,
        progress_reporter=events.append,
    )

    assert [event.metadata["kind"] for event in events] == [
        "corpus",
        "queries",
        "instructions",
        "qrels",
    ]
    assert events[-1].current == 4
    assert events[-1].total == 4


def test_normalize_jsonl_tsv_without_instructions_reports_three_progress_events(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _jsonl_tsv_snapshot(
        dataset_name="SciFact",
        slug="scifact",
        has_instructions=False,
    )
    write_raw_dataset_artifact(store, "raw_scifact_001", snapshot)
    events: list[ProgressEvent] = []

    normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id="raw_scifact_001",
            output_artifact_id="normalized_scifact_001",
            dataset_name="SciFact",
        ),
        opener=opener,
        progress_reporter=events.append,
    )

    assert [event.metadata["kind"] for event in events] == ["corpus", "queries", "qrels"]
    assert events[-1].current == 3
    assert events[-1].total == 3


def test_normalize_litsearch_parquet_reports_progress(
    store: LocalArtifactStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, opener = _litsearch_snapshot()
    write_raw_dataset_artifact(store, "raw_litsearch_001", snapshot)
    events: list[ProgressEvent] = []

    def fake_read_parquet_records(
        file: RawDatasetFile,
        opener: object,
    ) -> list[dict[str, object]]:
        rows_by_path: dict[str, list[dict[str, object]]] = {
            "corpus/test-00000-of-00001.parquet": [
                {"_id": "doc-1", "title": None, "text": "Doc"}
            ],
            "queries/test-00000-of-00001.parquet": [{"_id": "q-1", "text": "Query"}],
            "qrels/test-00000-of-00001.parquet": [
                {"query-id": "q-1", "corpus-id": "doc-1", "score": 1}
            ],
        }
        return rows_by_path[file.path]

    monkeypatch.setattr(raw_normalize_module, "_read_parquet_records", fake_read_parquet_records)

    normalize_raw_dataset_artifact(
        store,
        store,
        RawToNormalizedConfig(
            source_artifact_id="raw_litsearch_001",
            output_artifact_id="normalized_litsearch_001",
            dataset_name="LitSearchRetrieval",
        ),
        opener=opener,
        progress_reporter=events.append,
    )

    assert [event.metadata["kind"] for event in events] == ["corpus", "queries", "qrels"]
    assert [event.metadata["path"] for event in events] == [
        "corpus/*.parquet",
        "queries/*.parquet",
        "qrels/*.parquet",
    ]
    assert [event.metadata["shard_paths"] for event in events] == [
        ["corpus/test-00000-of-00001.parquet"],
        ["queries/test-00000-of-00001.parquet"],
        ["qrels/test-00000-of-00001.parquet"],
    ]


def test_normalize_raw_dataset_artifact_reporter_failure_does_not_write_success(
    store: LocalArtifactStore,
) -> None:
    snapshot, opener = _ifir_nfcorpus_snapshot()
    write_raw_dataset_artifact(store, "raw_ifir_nfcorpus_001", snapshot)

    def failing_reporter(_: ProgressEvent) -> None:
        raise RuntimeError("progress failed")

    with pytest.raises(RuntimeError, match="progress failed"):
        normalize_raw_dataset_artifact(
            store,
            store,
            RawToNormalizedConfig(
                source_artifact_id="raw_ifir_nfcorpus_001",
                output_artifact_id="normalized_ifir_nfcorpus_001",
                dataset_name="IFIRNFCorpus",
            ),
            opener=opener,
            progress_reporter=failing_reporter,
        )

    assert (
        store.is_complete(NORMALIZED_DATASET_ARTIFACT_TYPE, "normalized_ifir_nfcorpus_001")
        is False
    )
