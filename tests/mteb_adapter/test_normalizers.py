"""Tests for dataset-specific MTEB normalizers."""

from types import SimpleNamespace

from eval_platform.mteb_adapter.normalizers.litsearch import LitSearchRetrievalNormalizer


def test_litsearch_normalizer_drops_empty_corpus_docs_and_orphan_queries() -> None:
    task = SimpleNamespace(
        corpus={
            "doc-1": {"title": "Title only", "text": ""},
            "doc-2": {"title": "", "text": ""},
            "doc-3": {"text": "Body"},
        },
        queries={
            "q-1": {"text": "query 1"},
            "q-2": {"text": "query 2"},
        },
        qrels={
            "q-1": {"doc-1": 1, "doc-2": 1},
            "q-2": {"doc-2": 1},
        },
    )

    dataset = LitSearchRetrievalNormalizer().normalize(task, split="test")

    assert [doc.doc_id for doc in dataset.corpus] == ["doc-1", "doc-3"]
    assert [query.query_id for query in dataset.queries] == ["q-1"]
    assert [(qrel.query_id, qrel.doc_id) for qrel in dataset.qrels] == [("q-1", "doc-1")]
    assert dataset.metadata["task_name"] == "LitSearchRetrieval"
    assert dataset.metadata["normalizer_name"] == "LitSearchRetrievalNormalizer"
