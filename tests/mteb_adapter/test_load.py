"""Tests for MTEB task loading helpers."""

from types import SimpleNamespace

import pytest

from eval_platform.mteb_adapter import MTEBAdapterError, extract_retrieval_data_from_mteb_task
from eval_platform.mteb_adapter.registry import NORMALIZER_REGISTRY


class SplitAwareFakeTask:
    def __init__(self) -> None:
        self.load_data_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.corpus = {
            "test": {"doc-1": {"text": "Body"}},
        }
        self.queries = {
            "test": {"q-1": "query text"},
        }
        self.relevant_docs = {
            "test": {"q-1": {"doc-1": 1}},
        }

    def load_data(self, eval_splits: list[str] | None = None) -> None:
        self.load_data_calls.append(((), {"eval_splits": eval_splits}))


class FlatFakeTask:
    def __init__(self) -> None:
        self.load_data_called = False
        self.corpus = {"doc-1": {"text": "Body"}}
        self.queries = {"q-1": "query text"}
        self.qrels = {"q-1": {"doc-1": 1.0}}

    def load_data(self) -> None:
        self.load_data_called = True


class QrelsOnlyFakeTask:
    def __init__(self) -> None:
        self.corpus = {"doc-1": {"text": "Body"}}
        self.queries = {"q-1": "query text"}
        self.qrels = {"q-1": {"doc-1": 1}}


class FallbackLoadDataTask:
    def __init__(self) -> None:
        self.fallback_used = False
        self.corpus = {"doc-1": {"text": "Body"}}
        self.queries = {"q-1": "query text"}
        self.relevant_docs = {"q-1": {"doc-1": 1}}

    def load_data(self, eval_splits: list[str] | None = None) -> None:
        if eval_splits is not None:
            raise TypeError("unexpected keyword argument 'eval_splits'")
        self.fallback_used = True


class MissingFieldsTask:
    def load_data(self) -> None:
        return None


class SplitsParamFakeTask:
    def __init__(self) -> None:
        self.used_splits: list[str] | None = None
        self.corpus = {"doc-1": {"text": "Body"}}
        self.queries = {"q-1": "query text"}
        self.relevant_docs = {"q-1": {"doc-1": 1}}

    def load_data(self, splits: list[str] | None = None) -> None:
        if splits is None:
            raise TypeError("missing required argument: 'splits'")
        self.used_splits = splits


class SplitKwargFakeTask:
    def __init__(self) -> None:
        self.used_split: str | None = None
        self.corpus = {"doc-1": {"text": "Body"}}
        self.queries = {"q-1": "query text"}
        self.relevant_docs = {"q-1": {"doc-1": 1}}

    def load_data(self, split: str | None = None) -> None:
        if split is None:
            raise TypeError("missing required argument: 'split'")
        self.used_split = split


class EvalSplitsRuntimeErrorTask:
    def __init__(self) -> None:
        self.corpus = {"doc-1": {"text": "Body"}}
        self.queries = {"q-1": "query text"}
        self.relevant_docs = {"q-1": {"doc-1": 1}}

    def load_data(self, eval_splits: list[str] | None = None) -> None:
        if eval_splits is not None:
            raise RuntimeError("eval_splits failed")


class DatasetFieldTask:
    def __init__(self) -> None:
        self.dataset = {
            "default": {
                "test": {
                    "corpus": [{"id": "doc-1", "text": "Body"}],
                    "queries": [{"id": "q-1", "text": "query text"}],
                    "relevant_docs": {"q-1": {"doc-1": 1}},
                    "top_ranked": {},
                }
            }
        }

    def load_data(self, eval_splits: list[str] | None = None) -> None:
        return None


class DatasetFieldUnderscoreIdTask:
    def __init__(self) -> None:
        self.dataset = {
            "default": {
                "test": {
                    "corpus": [{"_id": "doc-1", "text": "Body"}],
                    "queries": [{"_id": "q-1", "text": "query text"}],
                    "qrels": {"q-1": {"doc-1": 1}},
                }
            }
        }

    def load_data(self, eval_splits: list[str] | None = None) -> None:
        return None


class MultiSubsetDatasetFieldTask:
    def __init__(self) -> None:
        self.dataset = {
            "unused": {
                "train": {
                    "corpus": [{"id": "doc-train", "text": "Train"}],
                    "queries": [{"id": "q-train", "text": "train query"}],
                    "relevant_docs": {"q-train": {"doc-train": 1}},
                }
            },
            "default": {
                "test": {
                    "corpus": [{"id": "doc-1", "text": "Body"}],
                    "queries": [{"id": "q-1", "text": "query text"}],
                    "relevant_docs": {"q-1": {"doc-1": 1}},
                }
            },
        }

    def load_data(self, eval_splits: list[str] | None = None) -> None:
        return None


class NamedTaskForRegistryDispatch:
    def __init__(self) -> None:
        self.metadata = SimpleNamespace(name="LitSearchRetrieval")


class RegistryDispatchNormalizer:
    task_name = "LitSearchRetrieval"

    def extract_raw(self, task: object, split: str):
        return {"doc-1": {"text": "Body"}}, {"q-1": "query text"}, {"q-1": {"doc-1": 1}}


def test_extract_from_split_aware_task() -> None:
    task = SplitAwareFakeTask()

    corpus, queries, qrels = extract_retrieval_data_from_mteb_task(task, split="test")

    assert corpus == {"doc-1": {"text": "Body"}}
    assert queries == {"q-1": "query text"}
    assert qrels == {"q-1": {"doc-1": 1}}
    assert task.load_data_calls == [((), {"eval_splits": ["test"]})]


def test_extract_from_flat_task() -> None:
    task = FlatFakeTask()

    corpus, queries, qrels = extract_retrieval_data_from_mteb_task(task, split="test")

    assert corpus == {"doc-1": {"text": "Body"}}
    assert queries == {"q-1": "query text"}
    assert qrels == {"q-1": {"doc-1": 1.0}}
    assert task.load_data_called is True


def test_extract_uses_relevant_docs_when_present() -> None:
    task = SplitAwareFakeTask()

    _, _, qrels = extract_retrieval_data_from_mteb_task(task, split="test")

    assert qrels == {"q-1": {"doc-1": 1}}


def test_extract_uses_qrels_when_relevant_docs_missing() -> None:
    task = QrelsOnlyFakeTask()

    _, _, qrels = extract_retrieval_data_from_mteb_task(task, split="test")

    assert qrels == {"q-1": {"doc-1": 1}}


def test_extract_raises_when_required_fields_missing() -> None:
    task = MissingFieldsTask()

    with pytest.raises(MTEBAdapterError, match="missing corpus"):
        extract_retrieval_data_from_mteb_task(task, split="test")


def test_load_data_prefers_eval_splits_when_supported() -> None:
    task = SplitAwareFakeTask()

    extract_retrieval_data_from_mteb_task(task, split="test")

    assert task.load_data_calls[-1] == ((), {"eval_splits": ["test"]})


def test_load_data_falls_back_when_eval_splits_unsupported() -> None:
    task = FallbackLoadDataTask()

    corpus, queries, qrels = extract_retrieval_data_from_mteb_task(task, split="test")

    assert corpus == {"doc-1": {"text": "Body"}}
    assert queries == {"q-1": "query text"}
    assert qrels == {"q-1": {"doc-1": 1}}
    assert task.fallback_used is True


def test_load_data_uses_splits_parameter_when_supported() -> None:
    task = SplitsParamFakeTask()

    extract_retrieval_data_from_mteb_task(task, split="test")

    assert task.used_splits == ["test"]


def test_load_data_uses_split_keyword_when_supported() -> None:
    task = SplitKwargFakeTask()

    extract_retrieval_data_from_mteb_task(task, split="test")

    assert task.used_split == "test"


def test_load_data_does_not_swallow_non_type_error_from_eval_splits() -> None:
    task = EvalSplitsRuntimeErrorTask()

    with pytest.raises(RuntimeError, match="eval_splits failed"):
        extract_retrieval_data_from_mteb_task(task, split="test")


def test_extract_raises_when_queries_missing() -> None:
    task = SimpleNamespace(corpus={"doc-1": {"text": "Body"}}, relevant_docs={"q-1": {"doc-1": 1}})

    with pytest.raises(MTEBAdapterError, match="missing queries"):
        extract_retrieval_data_from_mteb_task(task, split="test")


def test_extract_from_dataset_field_layout() -> None:
    task = DatasetFieldTask()

    corpus, queries, qrels = extract_retrieval_data_from_mteb_task(task, split="test")

    assert corpus == {"doc-1": {"text": "Body"}}
    assert queries == {"q-1": {"text": "query text"}}
    assert qrels == {"q-1": {"doc-1": 1}}


def test_extract_from_dataset_field_layout_with_underscore_id() -> None:
    task = DatasetFieldUnderscoreIdTask()

    corpus, queries, qrels = extract_retrieval_data_from_mteb_task(task, split="test")

    assert corpus == {"doc-1": {"text": "Body"}}
    assert queries == {"q-1": {"text": "query text"}}
    assert qrels == {"q-1": {"doc-1": 1}}


def test_extract_from_first_subset_with_matching_split_and_fields() -> None:
    task = MultiSubsetDatasetFieldTask()

    corpus, queries, qrels = extract_retrieval_data_from_mteb_task(task, split="test")

    assert corpus == {"doc-1": {"text": "Body"}}
    assert queries == {"q-1": {"text": "query text"}}
    assert qrels == {"q-1": {"doc-1": 1}}


def test_extract_dispatches_to_registered_normalizer_when_task_name_is_known(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = NamedTaskForRegistryDispatch()
    monkeypatch.setitem(NORMALIZER_REGISTRY, "LitSearchRetrieval", RegistryDispatchNormalizer())

    corpus, queries, qrels = extract_retrieval_data_from_mteb_task(task, split="test")

    assert corpus == {"doc-1": {"text": "Body"}}
    assert queries == {"q-1": "query text"}
    assert qrels == {"q-1": {"doc-1": 1}}
