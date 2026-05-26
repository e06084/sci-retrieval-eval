"""Retrieval hit fusion and dedupe helpers."""

from __future__ import annotations

from eval_platform.retrieval.schema import RetrievalHit


def rrf_fuse(
    milvus_hits: list[RetrievalHit],
    es_hits: list[RetrievalHit],
    out_top_k: int,
    *,
    k: int = 60,
) -> list[RetrievalHit]:
    """Fuse Milvus and Elasticsearch hits using reciprocal rank fusion."""

    if out_top_k <= 0:
        return []
    if not milvus_hits:
        return [hit.model_copy() for hit in es_hits[:out_top_k]]
    if not es_hits:
        return [hit.model_copy() for hit in milvus_hits[:out_top_k]]

    milvus_rank = _first_ranks([hit.chunk_id for hit in milvus_hits])
    es_rank = _first_ranks([hit.chunk_id for hit in es_hits])
    rows: list[tuple[str, float]] = []
    for chunk_id in set(milvus_rank) | set(es_rank):
        score = 0.0
        if chunk_id in milvus_rank:
            score += 1.0 / (k + milvus_rank[chunk_id])
        if chunk_id in es_rank:
            score += 1.0 / (k + es_rank[chunk_id])
        rows.append((chunk_id, score))
    rows.sort(key=lambda row: (-row[1], row[0]))

    milvus_by_id = {hit.chunk_id: hit for hit in milvus_hits}
    es_by_id = {hit.chunk_id: hit for hit in es_hits}
    return [
        _coalesce_hits(milvus_by_id.get(chunk_id), es_by_id.get(chunk_id), score)
        for chunk_id, score in rows[:out_top_k]
    ]


def dedupe_sequential(
    hit_lists: list[list[RetrievalHit]],
    *,
    max_total: int = 250,
) -> list[RetrievalHit]:
    """Dedupe hits by chunk id while preserving query/list order."""

    out: list[RetrievalHit] = []
    seen: set[str] = set()
    for hits in hit_lists:
        for hit in hits:
            chunk_id = hit.chunk_id.strip()
            if not chunk_id or chunk_id in seen:
                continue
            seen.add(chunk_id)
            out.append(hit)
            if max_total > 0 and len(out) >= max_total:
                return out
    return out


def dedupe_by_chunk_id(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    """Dedupe hits by first chunk id occurrence."""

    out: list[RetrievalHit] = []
    seen: set[str] = set()
    for hit in hits:
        chunk_id = hit.chunk_id.strip()
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append(hit)
    return out


def _first_ranks(chunk_ids: list[str]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for rank, chunk_id in enumerate(chunk_ids, start=1):
        normalized = chunk_id.strip()
        if normalized and normalized not in ranks:
            ranks[normalized] = rank
    return ranks


def _coalesce_hits(
    milvus_hit: RetrievalHit | None,
    es_hit: RetrievalHit | None,
    score: float,
) -> RetrievalHit:
    if milvus_hit is not None and es_hit is not None:
        data = milvus_hit.model_dump()
        data["score"] = score
        data["recall_source"] = "milvus|es"
        data["origin_milvus_score"] = milvus_hit.score
        data["origin_es_score"] = es_hit.score
        data["doc_id"] = data.get("doc_id") or es_hit.doc_id
        data["title"] = data.get("title") or es_hit.title
        data["text"] = data.get("text") or es_hit.text
        data["metadata"] = {**es_hit.metadata, **milvus_hit.metadata}
        return RetrievalHit.model_validate(data)

    hit = milvus_hit or es_hit
    if hit is None:
        raise ValueError("cannot coalesce empty hits")
    data = hit.model_dump()
    data["score"] = score
    return RetrievalHit.model_validate(data)
