"""Base classes and shared helpers for MTEB task normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from eval_platform.datasets.schema import NormalizedDataset
from eval_platform.mteb_adapter.convert import convert_retrieval_data_to_normalized_dataset


class MTEBAdapterError(Exception):
    """Raised when MTEB task data cannot be loaded or extracted."""


def _load_task_data(task: Any, split: str) -> None:
    load_data = getattr(task, "load_data", None)
    if load_data is None:
        return

    attempts = [
        lambda: load_data(eval_splits=[split]),
        lambda: load_data(splits=[split]),
        lambda: load_data(split=split),
        lambda: load_data(),
    ]

    last_type_error: TypeError | None = None
    for attempt in attempts:
        try:
            attempt()
            return
        except TypeError as exc:
            last_type_error = exc

    if last_type_error is not None:
        raise last_type_error


def _select_split(data: Any, split: str) -> Any:
    if isinstance(data, Mapping) and split in data:
        return data[split]
    return data


def _require_mapping(name: str, data: Any) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise MTEBAdapterError(f"MTEB task is missing {name}")
    return data


def _rows_to_mapping(data: Any) -> Any:
    if isinstance(data, Mapping):
        return data
    if not isinstance(data, Iterable) or isinstance(data, (str, bytes)):
        return data

    result: dict[str, Any] = {}
    for row in data:
        if not isinstance(row, Mapping):
            return data
        row_id = row.get("id")
        if row_id is None:
            row_id = row.get("_id")
        if row_id is None:
            return data
        normalized_row = dict(row)
        normalized_row.pop("id", None)
        normalized_row.pop("_id", None)
        result[str(row_id)] = normalized_row
    return result


def _extract_from_dataset_field(
    task: Any,
    split: str,
) -> tuple[Any, Any, Any] | None:
    dataset = getattr(task, "dataset", None)
    if not isinstance(dataset, Mapping):
        return None

    for subset_payload in dataset.values():
        split_payload = _select_split(subset_payload, split)
        if not isinstance(split_payload, Mapping):
            continue

        corpus = _rows_to_mapping(split_payload.get("corpus"))
        queries = _rows_to_mapping(split_payload.get("queries"))
        qrels = split_payload.get("relevant_docs")
        if qrels is None:
            qrels = split_payload.get("qrels")

        if corpus is None and queries is None and qrels is None:
            continue
        return corpus, queries, qrels

    return None


def extract_generic_retrieval_task_data(
    task: Any,
    split: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Mapping[str, int | float]]]:
    """Extract corpus, queries, and qrels from a loaded retrieval task."""
    _load_task_data(task, split)

    corpus = _select_split(getattr(task, "corpus", None), split)
    queries = _select_split(getattr(task, "queries", None), split)

    qrels = getattr(task, "relevant_docs", None)
    if qrels is None:
        qrels = getattr(task, "qrels", None)
    qrels = _select_split(qrels, split)

    if corpus is None or queries is None or qrels is None:
        extracted = _extract_from_dataset_field(task, split)
        if extracted is not None:
            if corpus is None:
                corpus = extracted[0]
            if queries is None:
                queries = extracted[1]
            if qrels is None:
                qrels = extracted[2]

    corpus_mapping = _require_mapping("corpus", corpus)
    queries_mapping = _require_mapping("queries", queries)
    qrels_mapping = _require_mapping("qrels", qrels)

    return corpus_mapping, queries_mapping, qrels_mapping


class MTEBTaskNormalizer(ABC):
    """Normalizer interface for a specific MTEB retrieval task."""

    task_name: str

    @property
    def normalizer_name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def extract_raw(
        self,
        task: Any,
        split: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Mapping[str, int | float]]]:
        """Extract raw corpus, queries, and qrels from a loaded task."""

    def normalize(self, task: Any, split: str = "test") -> NormalizedDataset:
        corpus, queries, qrels = self.extract_raw(task, split)
        return convert_retrieval_data_to_normalized_dataset(
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            metadata={
                "source": "mteb",
                "task_name": self.task_name,
                "split": split,
                "normalizer_name": self.normalizer_name,
            },
        )


class GenericRetrievalTaskNormalizer(MTEBTaskNormalizer):
    """Dataset-specific normalizer that reuses the shared retrieval extractor."""

    def extract_raw(
        self,
        task: Any,
        split: str,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Mapping[str, int | float]]]:
        return extract_generic_retrieval_task_data(task, split)
