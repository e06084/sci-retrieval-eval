"""Tests for the MTEB normalizer registry."""

from types import SimpleNamespace

import pytest

from eval_platform.datasets.schema import NormalizedDataset
from eval_platform.mteb_adapter import (
    NORMALIZER_REGISTRY,
    MTEBAdapterError,
    get_mteb_task_normalizer,
    load_mteb_retrieval_dataset,
)
from eval_platform.mteb_adapter.base import MTEBTaskNormalizer


@pytest.mark.parametrize(
    ("task_name", "normalizer_name"),
    [
        ("LitSearchRetrieval", "LitSearchRetrievalNormalizer"),
        ("SciFact", "SciFactNormalizer"),
        ("IFIRScifact", "IFIRScifactNormalizer"),
        ("IFIRNFCorpus", "IFIRNFCorpusNormalizer"),
        ("NFCorpus", "NFCorpusNormalizer"),
    ],
)
def test_registry_contains_explicit_normalizer_for_target_task(
    task_name: str,
    normalizer_name: str,
) -> None:
    normalizer = get_mteb_task_normalizer(task_name)

    assert normalizer.task_name == task_name
    assert type(normalizer).__name__ == normalizer_name


def test_registry_raises_for_unknown_task() -> None:
    with pytest.raises(MTEBAdapterError, match="No MTEB normalizer registered"):
        get_mteb_task_normalizer("UnknownTask")


class FakeNormalizer(MTEBTaskNormalizer):
    task_name = "SciFact"

    def __init__(self) -> None:
        self.called_with: tuple[object, str] | None = None

    def extract_raw(self, task: object, split: str):
        raise AssertionError("extract_raw should not be called directly in this test")

    def normalize(self, task: object, split: str = "test") -> NormalizedDataset:
        self.called_with = (task, split)
        return NormalizedDataset(corpus=[], queries=[], qrels=[], metadata={"ok": True})


def test_load_mteb_retrieval_dataset_dispatches_to_registered_normalizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_task = SimpleNamespace(name="fake-task")
    fake_normalizer = FakeNormalizer()

    monkeypatch.setattr(
        "eval_platform.mteb_adapter.load.load_mteb_task",
        lambda task_name: fake_task,
    )
    monkeypatch.setitem(NORMALIZER_REGISTRY, "SciFact", fake_normalizer)

    dataset = load_mteb_retrieval_dataset("SciFact", split="validation")

    assert dataset.metadata == {"ok": True}
    assert fake_normalizer.called_with == (fake_task, "validation")
