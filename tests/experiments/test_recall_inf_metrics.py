"""Tests for _compute_recall_inf_metrics and its helper functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_platform.artifacts import ArtifactDependency, ArtifactManifest, LocalArtifactStore
from eval_platform.artifacts.types import (
    ELASTICSEARCH_INDEX_ARTIFACT_TYPE,
    MILVUS_COLLECTION_ARTIFACT_TYPE,
)
from eval_platform.datasets import (
    CorpusRecord,
    NormalizedDataset,
    QrelRecord,
    QueryRecord,
    write_normalized_dataset_artifact,
)
from eval_platform.retrieval import RetrievalHit, RetrievalQueryResult, write_retrieval_run_artifact
from eval_platform.experiments.runner import (
    _compute_recall_inf_metrics,
    _mean,
    _recall_inf,
    _record_doc_ids,
    _trace_doc_ids,
    _trace_hit_doc_id,
)


class TestTraceHitDocId:
    def test_returns_doc_id_when_present(self) -> None:
        assert _trace_hit_doc_id({"doc_id": "d1", "chunk_id": "c1"}) == "d1"

    def test_returns_doc_id_stripped(self) -> None:
        assert _trace_hit_doc_id({"doc_id": "  d1  ", "chunk_id": "c1"}) == "d1"

    def test_falls_back_to_metadata_paper_id(self) -> None:
        hit = {"doc_id": "", "chunk_id": "c1", "metadata": {"paper_id": "p1"}}
        assert _trace_hit_doc_id(hit) == "p1"

    def test_falls_back_to_chunk_id(self) -> None:
        hit = {"doc_id": "", "chunk_id": "c1", "metadata": {}}
        assert _trace_hit_doc_id(hit) == "c1"

    def test_returns_empty_when_all_missing(self) -> None:
        assert _trace_hit_doc_id({}) == ""

    def test_ignores_non_dict_metadata(self) -> None:
        hit = {"doc_id": "", "chunk_id": "c1", "metadata": "bad"}
        assert _trace_hit_doc_id(hit) == "c1"

    def test_ignores_non_string_paper_id(self) -> None:
        hit = {"doc_id": "", "chunk_id": "c1", "metadata": {"paper_id": 123}}
        assert _trace_hit_doc_id(hit) == "c1"

    def test_ignores_blank_paper_id(self) -> None:
        hit = {"doc_id": "", "chunk_id": "c1", "metadata": {"paper_id": "   "}}
        assert _trace_hit_doc_id(hit) == "c1"


class TestTraceDocs:
    def test_extracts_from_top_level_key(self) -> None:
        trace = {
            "es_hits": [
                {"doc_id": "d1", "chunk_id": "c1"},
                {"doc_id": "d2", "chunk_id": "c2"},
            ]
        }
        assert _trace_doc_ids(trace, "es_hits") == {"d1", "d2"}

    def test_extracts_from_per_query(self) -> None:
        trace = {
            "per_query": [
                {
                    "milvus_hits": [
                        {"doc_id": "d1", "chunk_id": "c1"},
                        {"doc_id": "d2", "chunk_id": "c2"},
                    ]
                },
                {
                    "milvus_hits": [
                        {"doc_id": "d3", "chunk_id": "c3"},
                    ]
                },
            ]
        }
        assert _trace_doc_ids(trace, "milvus_hits") == {"d1", "d2", "d3"}

    def test_merges_top_level_and_per_query(self) -> None:
        trace = {
            "es_hits": [{"doc_id": "d1", "chunk_id": "c1"}],
            "per_query": [
                {"es_hits": [{"doc_id": "d2", "chunk_id": "c2"}]},
            ],
        }
        assert _trace_doc_ids(trace, "es_hits") == {"d1", "d2"}

    def test_returns_empty_for_missing_key(self) -> None:
        trace = {"es_hits": [{"doc_id": "d1", "chunk_id": "c1"}]}
        assert _trace_doc_ids(trace, "milvus_hits") == set()

    def test_skips_non_dict_hits(self) -> None:
        trace = {"es_hits": [{"doc_id": "d1", "chunk_id": "c1"}, "not-a-dict", 42]}
        assert _trace_doc_ids(trace, "es_hits") == {"d1"}

    def test_skips_hits_with_empty_doc_id(self) -> None:
        trace = {"es_hits": [{"doc_id": "", "chunk_id": "", "metadata": {}}]}
        assert _trace_doc_ids(trace, "es_hits") == set()

    def test_empty_trace(self) -> None:
        assert _trace_doc_ids({}, "es_hits") == set()

    def test_per_query_non_list(self) -> None:
        trace = {"per_query": "not_a_list"}
        assert _trace_doc_ids(trace, "es_hits") == set()


class TestRecordDocIds:
    def test_extracts_doc_ids_from_hits(self) -> None:
        hits = [
            RetrievalHit(chunk_id="c1", doc_id="d1", score=1.0),
            RetrievalHit(chunk_id="c2", doc_id="d2", score=0.9),
        ]
        assert _record_doc_ids(hits) == {"d1", "d2"}

    def test_falls_back_to_chunk_id_when_doc_id_empty(self) -> None:
        hits = [
            RetrievalHit(chunk_id="c1", doc_id="", score=1.0),
        ]
        assert _record_doc_ids(hits) == {"c1"}

    def test_deduplicates(self) -> None:
        hits = [
            RetrievalHit(chunk_id="c1", doc_id="d1", score=1.0, rank=1),
            RetrievalHit(chunk_id="c2", doc_id="d1", score=0.9, rank=2),
        ]
        assert _record_doc_ids(hits) == {"d1"}

    def test_uses_metadata_paper_id_fallback(self) -> None:
        hits = [
            RetrievalHit(
                chunk_id="c1",
                doc_id="",
                score=1.0,
                metadata={"paper_id": "p1"},
            ),
        ]
        assert _record_doc_ids(hits) == {"p1"}


class TestRecallInf:
    def test_perfect_recall(self) -> None:
        assert _recall_inf({"d1", "d2", "d3"}, {"d1", "d2", "d3"}) == 1.0

    def test_partial_recall(self) -> None:
        assert _recall_inf({"d1", "d2"}, {"d1", "d2", "d3"}) == pytest.approx(2.0 / 3.0)

    def test_zero_recall(self) -> None:
        assert _recall_inf({"x", "y"}, {"d1", "d2"}) == 0.0

    def test_empty_relevant_docs(self) -> None:
        assert _recall_inf({"d1"}, set()) == 0.0

    def test_empty_retrieved_docs(self) -> None:
        assert _recall_inf(set(), {"d1", "d2"}) == 0.0

    def test_superset_retrieval(self) -> None:
        assert _recall_inf({"d1", "d2", "d3", "d4"}, {"d1", "d2"}) == 1.0


class TestMean:
    def test_normal(self) -> None:
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_single_value(self) -> None:
        assert _mean([0.5]) == 0.5

    def test_empty_returns_zero(self) -> None:
        assert _mean([]) == 0.0


class TestComputeRecallInfMetrics:
    def test_computes_all_recall_inf_keys(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path)
        normalized_id = "norm-ds1"
        retrieval_id = "ret-ds1"

        write_normalized_dataset_artifact(
            store,
            normalized_id,
            NormalizedDataset(
                corpus=[
                    CorpusRecord(doc_id="d1", text="text1"),
                    CorpusRecord(doc_id="d2", text="text2"),
                    CorpusRecord(doc_id="d3", text="text3"),
                ],
                queries=[
                    QueryRecord(query_id="q1", text="query1"),
                ],
                qrels=[
                    QrelRecord(query_id="q1", doc_id="d1", relevance=1.0),
                    QrelRecord(query_id="q1", doc_id="d2", relevance=1.0),
                ],
            ),
            metadata={
                "raw_dataset_asset_fingerprint_sha256": "fp-raw",
                "normalizer_name": "test",
                "normalizer_version": "1",
                "normalized_schema_version": "1",
            },
        )

        write_retrieval_run_artifact(
            store,
            retrieval_id,
            [
                RetrievalQueryResult(
                    query_id="q1",
                    query_text="query1",
                    hits=[
                        RetrievalHit(chunk_id="c1", doc_id="d1", score=1.0, rank=1),
                        RetrievalHit(chunk_id="c2", doc_id="d2", score=0.9, rank=2),
                    ],
                    trace={
                        "es_hits": [
                            {"doc_id": "d1", "chunk_id": "c1", "score": 1.0},
                            {"doc_id": "d3", "chunk_id": "c3", "score": 0.5},
                        ],
                        "milvus_hits": [
                            {"doc_id": "d2", "chunk_id": "c2", "score": 0.9},
                        ],
                        "paper_capped_hits": [
                            {"doc_id": "d1", "chunk_id": "c1", "score": 1.0},
                            {"doc_id": "d2", "chunk_id": "c2", "score": 0.9},
                        ],
                    },
                ),
            ],
        )

        result = _compute_recall_inf_metrics(
            store,
            source_normalized_dataset_artifact_id=normalized_id,
            retrieval_run_artifact_id=retrieval_id,
        )

        assert "es_recall_at_inf" in result
        assert "milvus_recall_at_inf" in result
        assert "rrf_recall_at_inf" in result
        assert "rrf_intersect_es_recall_at_inf" in result
        assert "rrf_intersect_milvus_recall_at_inf" in result
        # q1 relevant={d1, d2}; es_hits={d1, d3} -> recall=1/2
        assert result["es_recall_at_inf"] == pytest.approx(0.5)
        # milvus_hits={d2} -> recall=1/2
        assert result["milvus_recall_at_inf"] == pytest.approx(0.5)
        # paper_capped_hits={d1, d2} -> recall=2/2
        assert result["rrf_recall_at_inf"] == pytest.approx(1.0)
        # rrf ∩ es = {d1,d2} ∩ {d1,d3} = {d1} -> recall=1/2
        assert result["rrf_intersect_es_recall_at_inf"] == pytest.approx(0.5)
        # rrf ∩ milvus = {d1,d2} ∩ {d2} = {d2} -> recall=1/2
        assert result["rrf_intersect_milvus_recall_at_inf"] == pytest.approx(0.5)

    def test_falls_back_to_fused_hits_when_paper_capped_missing(
        self, tmp_path: Path
    ) -> None:
        store = LocalArtifactStore(tmp_path)
        normalized_id = "norm-ds2"
        retrieval_id = "ret-ds2"

        write_normalized_dataset_artifact(
            store,
            normalized_id,
            NormalizedDataset(
                corpus=[
                    CorpusRecord(doc_id="d1", text="t1"),
                    CorpusRecord(doc_id="d2", text="t2"),
                ],
                queries=[QueryRecord(query_id="q1", text="q")],
                qrels=[QrelRecord(query_id="q1", doc_id="d1", relevance=1.0)],
            ),
            metadata={
                "raw_dataset_asset_fingerprint_sha256": "fp",
                "normalizer_name": "test",
                "normalizer_version": "1",
                "normalized_schema_version": "1",
            },
        )

        write_retrieval_run_artifact(
            store,
            retrieval_id,
            [
                RetrievalQueryResult(
                    query_id="q1",
                    query_text="q",
                    hits=[RetrievalHit(chunk_id="c1", doc_id="d1", score=1.0, rank=1)],
                    trace={
                        "es_hits": [{"doc_id": "d1", "chunk_id": "c1"}],
                        "milvus_hits": [],
                        "fused_hits": [{"doc_id": "d1", "chunk_id": "c1"}],
                    },
                ),
            ],
        )

        result = _compute_recall_inf_metrics(
            store,
            source_normalized_dataset_artifact_id=normalized_id,
            retrieval_run_artifact_id=retrieval_id,
        )

        assert result["rrf_recall_at_inf"] == pytest.approx(1.0)

    def test_falls_back_to_record_hits_when_trace_keys_missing(
        self, tmp_path: Path
    ) -> None:
        store = LocalArtifactStore(tmp_path)
        normalized_id = "norm-ds3"
        retrieval_id = "ret-ds3"

        write_normalized_dataset_artifact(
            store,
            normalized_id,
            NormalizedDataset(
                corpus=[CorpusRecord(doc_id="d1", text="t1")],
                queries=[QueryRecord(query_id="q1", text="q")],
                qrels=[QrelRecord(query_id="q1", doc_id="d1", relevance=1.0)],
            ),
            metadata={
                "raw_dataset_asset_fingerprint_sha256": "fp",
                "normalizer_name": "test",
                "normalizer_version": "1",
                "normalized_schema_version": "1",
            },
        )

        write_retrieval_run_artifact(
            store,
            retrieval_id,
            [
                RetrievalQueryResult(
                    query_id="q1",
                    query_text="q",
                    hits=[RetrievalHit(chunk_id="c1", doc_id="d1", score=1.0, rank=1)],
                    trace={},
                ),
            ],
        )

        result = _compute_recall_inf_metrics(
            store,
            source_normalized_dataset_artifact_id=normalized_id,
            retrieval_run_artifact_id=retrieval_id,
        )

        # rrf falls back to record hits: {d1} -> recall=1/1
        assert result["rrf_recall_at_inf"] == pytest.approx(1.0)
        # es/milvus have no trace -> empty set -> recall=0
        assert result["es_recall_at_inf"] == pytest.approx(0.0)
        assert result["milvus_recall_at_inf"] == pytest.approx(0.0)

    def test_returns_empty_dict_when_artifact_missing(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path)
        result = _compute_recall_inf_metrics(
            store,
            source_normalized_dataset_artifact_id="nonexistent",
            retrieval_run_artifact_id="nonexistent",
        )
        assert result == {}

    def test_handles_error_queries(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path)
        normalized_id = "norm-ds4"
        retrieval_id = "ret-ds4"

        write_normalized_dataset_artifact(
            store,
            normalized_id,
            NormalizedDataset(
                corpus=[CorpusRecord(doc_id="d1", text="t1")],
                queries=[
                    QueryRecord(query_id="q1", text="q1"),
                    QueryRecord(query_id="q2", text="q2"),
                ],
                qrels=[
                    QrelRecord(query_id="q1", doc_id="d1", relevance=1.0),
                    QrelRecord(query_id="q2", doc_id="d1", relevance=1.0),
                ],
            ),
            metadata={
                "raw_dataset_asset_fingerprint_sha256": "fp",
                "normalizer_name": "test",
                "normalizer_version": "1",
                "normalized_schema_version": "1",
            },
        )

        write_retrieval_run_artifact(
            store,
            retrieval_id,
            [
                RetrievalQueryResult(
                    query_id="q1",
                    query_text="q1",
                    hits=[RetrievalHit(chunk_id="c1", doc_id="d1", score=1.0, rank=1)],
                    trace={
                        "es_hits": [{"doc_id": "d1", "chunk_id": "c1"}],
                        "milvus_hits": [{"doc_id": "d1", "chunk_id": "c1"}],
                    },
                ),
                RetrievalQueryResult(
                    query_id="q2",
                    query_text="q2",
                    error="timeout",
                ),
            ],
        )

        result = _compute_recall_inf_metrics(
            store,
            source_normalized_dataset_artifact_id=normalized_id,
            retrieval_run_artifact_id=retrieval_id,
        )

        # q1: es recall=1.0, q2: error -> 0.0, mean = 0.5
        assert result["es_recall_at_inf"] == pytest.approx(0.5)
        assert result["milvus_recall_at_inf"] == pytest.approx(0.5)

    def test_skips_zero_relevance_qrels(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path)
        normalized_id = "norm-ds5"
        retrieval_id = "ret-ds5"

        write_normalized_dataset_artifact(
            store,
            normalized_id,
            NormalizedDataset(
                corpus=[
                    CorpusRecord(doc_id="d1", text="t1"),
                    CorpusRecord(doc_id="d2", text="t2"),
                ],
                queries=[QueryRecord(query_id="q1", text="q")],
                qrels=[
                    QrelRecord(query_id="q1", doc_id="d1", relevance=1.0),
                    QrelRecord(query_id="q1", doc_id="d2", relevance=0.0),
                ],
            ),
            metadata={
                "raw_dataset_asset_fingerprint_sha256": "fp",
                "normalizer_name": "test",
                "normalizer_version": "1",
                "normalized_schema_version": "1",
            },
        )

        write_retrieval_run_artifact(
            store,
            retrieval_id,
            [
                RetrievalQueryResult(
                    query_id="q1",
                    query_text="q",
                    hits=[RetrievalHit(chunk_id="c1", doc_id="d1", score=1.0, rank=1)],
                    trace={
                        "es_hits": [{"doc_id": "d1", "chunk_id": "c1"}],
                        "milvus_hits": [{"doc_id": "d2", "chunk_id": "c2"}],
                    },
                ),
            ],
        )

        result = _compute_recall_inf_metrics(
            store,
            source_normalized_dataset_artifact_id=normalized_id,
            retrieval_run_artifact_id=retrieval_id,
        )

        # only d1 is relevant (d2 has relevance=0)
        assert result["es_recall_at_inf"] == pytest.approx(1.0)
        assert result["milvus_recall_at_inf"] == pytest.approx(0.0)

    def test_per_query_trace_structure(self, tmp_path: Path) -> None:
        store = LocalArtifactStore(tmp_path)
        normalized_id = "norm-ds6"
        retrieval_id = "ret-ds6"

        write_normalized_dataset_artifact(
            store,
            normalized_id,
            NormalizedDataset(
                corpus=[
                    CorpusRecord(doc_id="d1", text="t1"),
                    CorpusRecord(doc_id="d2", text="t2"),
                ],
                queries=[QueryRecord(query_id="q1", text="q")],
                qrels=[
                    QrelRecord(query_id="q1", doc_id="d1", relevance=1.0),
                    QrelRecord(query_id="q1", doc_id="d2", relevance=1.0),
                ],
            ),
            metadata={
                "raw_dataset_asset_fingerprint_sha256": "fp",
                "normalizer_name": "test",
                "normalizer_version": "1",
                "normalized_schema_version": "1",
            },
        )

        write_retrieval_run_artifact(
            store,
            retrieval_id,
            [
                RetrievalQueryResult(
                    query_id="q1",
                    query_text="q",
                    hits=[
                        RetrievalHit(chunk_id="c1", doc_id="d1", score=1.0, rank=1),
                        RetrievalHit(chunk_id="c2", doc_id="d2", score=0.9, rank=2),
                    ],
                    trace={
                        "per_query": [
                            {
                                "es_hits": [{"doc_id": "d1", "chunk_id": "c1"}],
                                "milvus_hits": [{"doc_id": "d2", "chunk_id": "c2"}],
                            }
                        ]
                    },
                ),
            ],
        )

        result = _compute_recall_inf_metrics(
            store,
            source_normalized_dataset_artifact_id=normalized_id,
            retrieval_run_artifact_id=retrieval_id,
        )

        assert result["es_recall_at_inf"] == pytest.approx(0.5)
        assert result["milvus_recall_at_inf"] == pytest.approx(0.5)
