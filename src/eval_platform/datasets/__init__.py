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
    "CorpusRecord",
    "NormalizedDataset",
    "QrelRecord",
    "QueryRecord",
    "dump_jsonl",
    "load_jsonl",
    "read_normalized_dataset_artifact",
    "write_normalized_dataset_artifact",
]
