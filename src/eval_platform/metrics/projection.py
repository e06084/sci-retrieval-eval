"""Chunk-hit to doc-ranking projection."""

from __future__ import annotations

from eval_platform.metrics.schema import ProjectionStats, RankedDoc
from eval_platform.retrieval import RetrievalQueryResult


def project_retrieval_result_to_docs(
    result: RetrievalQueryResult,
) -> tuple[list[RankedDoc], ProjectionStats]:
    """Project chunk-level retrieval hits to a deterministic doc-level ranking."""

    ranked_docs: list[RankedDoc] = []
    seen_doc_ids: set[str] = set()
    missing_doc_id_hit_count = 0
    duplicate_doc_hit_count = 0

    for hit in sorted(result.hits, key=lambda item: item.rank or 0):
        if not hit.doc_id.strip():
            missing_doc_id_hit_count += 1
            continue
        if hit.doc_id in seen_doc_ids:
            duplicate_doc_hit_count += 1
            continue
        if hit.rank is None:
            continue
        seen_doc_ids.add(hit.doc_id)
        ranked_docs.append(
            RankedDoc(
                rank=len(ranked_docs) + 1,
                doc_id=hit.doc_id,
                score=1.0 / hit.rank,
                source_chunk_id=hit.chunk_id,
                source_chunk_rank=hit.rank,
                source_chunk_score=hit.score,
            )
        )

    return ranked_docs, ProjectionStats(
        input_hit_count=len(result.hits),
        ranked_doc_count=len(ranked_docs),
        missing_doc_id_hit_count=missing_doc_id_hit_count,
        duplicate_doc_hit_count=duplicate_doc_hit_count,
    )
