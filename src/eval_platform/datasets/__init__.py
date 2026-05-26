"""Dataset loading and normalization."""

from eval_platform.datasets.jsonl import dump_jsonl, load_jsonl
from eval_platform.datasets.normalized import (
    CORPUS_FILENAME,
    NORMALIZED_DATASET_ARTIFACT_TYPE,
    QRELS_FILENAME,
    QUERIES_FILENAME,
    read_normalized_dataset_artifact,
    write_normalized_dataset_artifact,
)
from eval_platform.datasets.raw import (
    RAW_DATASET_ARTIFACT_TYPE,
    RawDatasetArtifactError,
    RawDatasetFile,
    RawDatasetSnapshot,
    build_content_fingerprint_sha256,
    read_raw_dataset_artifact,
    write_raw_dataset_artifact,
)
from eval_platform.datasets.raw_import import (
    RawDatasetImportError,
    import_raw_dataset_from_local_dir,
    import_raw_dataset_from_s3_prefix,
)
from eval_platform.datasets.schema import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
)

__all__ = [
    "CORPUS_FILENAME",
    "NORMALIZED_DATASET_ARTIFACT_TYPE",
    "QRELS_FILENAME",
    "QUERIES_FILENAME",
    "RAW_DATASET_ARTIFACT_TYPE",
    "CorpusRecord",
    "NormalizedDataset",
    "QrelRecord",
    "QueryRecord",
    "RawDatasetArtifactError",
    "RawDatasetFile",
    "RawDatasetImportError",
    "RawDatasetSnapshot",
    "build_content_fingerprint_sha256",
    "dump_jsonl",
    "import_raw_dataset_from_local_dir",
    "import_raw_dataset_from_s3_prefix",
    "load_jsonl",
    "read_normalized_dataset_artifact",
    "read_raw_dataset_artifact",
    "write_normalized_dataset_artifact",
    "write_raw_dataset_artifact",
]
