"""Tests for retrieval fusion helpers."""

from eval_platform.retrieval import RetrievalHit, rrf_fuse
from eval_platform.retrieval.fusion import limit_hits_per_paper


def test_rrf_fuse_merges_sources_and_origin_scores() -> None:
    milvus_hits = [
        RetrievalHit(chunk_id="chunk-1", doc_id="doc-1", score=0.9, recall_source="milvus"),
        RetrievalHit(chunk_id="chunk-2", doc_id="doc-2", score=0.8, recall_source="milvus"),
    ]
    es_hits = [
        RetrievalHit(chunk_id="chunk-2", doc_id="doc-2", score=5.0, recall_source="es"),
        RetrievalHit(chunk_id="chunk-3", doc_id="doc-3", score=4.0, recall_source="es"),
    ]

    fused = rrf_fuse(milvus_hits, es_hits, out_top_k=3)

    merged = next(hit for hit in fused if hit.chunk_id == "chunk-2")
    assert merged.recall_source == "milvus|es"
    assert merged.origin_milvus_score == 0.8
    assert merged.origin_es_score == 5.0
    assert [hit.chunk_id for hit in fused] == ["chunk-2", "chunk-1", "chunk-3"]


def test_rrf_fuse_tie_breaks_by_chunk_id() -> None:
    fused = rrf_fuse(
        [RetrievalHit(chunk_id="b", score=1.0)],
        [RetrievalHit(chunk_id="a", score=1.0)],
        out_top_k=2,
    )

    assert [hit.chunk_id for hit in fused] == ["a", "b"]


def test_limit_hits_per_paper_uses_metadata_paper_id_then_doc_id_then_chunk_id() -> None:
    hits = [
        RetrievalHit(chunk_id="c1", doc_id="doc-a", metadata={"paper_id": "paper-1"}),
        RetrievalHit(chunk_id="c2", doc_id="doc-b", metadata={"paper_id": "paper-1"}),
        RetrievalHit(chunk_id="c3", doc_id="doc-2"),
        RetrievalHit(chunk_id="c4", doc_id="doc-2"),
        RetrievalHit(chunk_id="c5", doc_id="", metadata={}),
        RetrievalHit(chunk_id="c5", doc_id="", metadata={}),
        RetrievalHit(chunk_id="c6", doc_id="", metadata={}),
    ]

    limited = limit_hits_per_paper(hits, paper_cap=1, max_total=10)

    assert [hit.chunk_id for hit in limited] == ["c1", "c3", "c5", "c6"]
