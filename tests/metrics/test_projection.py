"""Tests for retrieval hit to doc ranking projection."""

from eval_platform.metrics import project_retrieval_result_to_docs
from eval_platform.retrieval import RetrievalHit, RetrievalQueryResult


def test_first_chunk_rank_projection_dedupes_docs_and_reassigns_rank() -> None:
    ranked_docs, stats = project_retrieval_result_to_docs(
        RetrievalQueryResult(
            query_id="q-1",
            query_text="query",
            hits=[
                RetrievalHit(rank=1, chunk_id="c-1", doc_id="doc-1", score=9.0),
                RetrievalHit(rank=2, chunk_id="c-2", doc_id="doc-1", score=8.0),
                RetrievalHit(rank=3, chunk_id="c-3", doc_id="", score=7.0),
                RetrievalHit(rank=4, chunk_id="c-4", doc_id="doc-2", score=6.0),
            ],
        )
    )

    assert [doc.doc_id for doc in ranked_docs] == ["doc-1", "doc-2"]
    assert [doc.rank for doc in ranked_docs] == [1, 2]
    assert ranked_docs[0].score == 1.0
    assert ranked_docs[1].score == 0.25
    assert ranked_docs[1].source_chunk_id == "c-4"
    assert ranked_docs[1].source_chunk_rank == 4
    assert stats.input_hit_count == 4
    assert stats.ranked_doc_count == 2
    assert stats.missing_doc_id_hit_count == 1
    assert stats.duplicate_doc_hit_count == 1
