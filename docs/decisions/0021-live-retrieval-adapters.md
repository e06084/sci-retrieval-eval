# 0021. Live Retrieval Adapters

- Status: Accepted
- Date: 2026-05-27

## Context

The platform can now produce index artifacts, retrieval run artifacts, and metrics run artifacts.
`run_retrieval(...)` already supports injectable retrieval protocols, but only fake clients were
available in tests. To run live retrieval against produced indexes, the platform needs concrete
Elasticsearch and Milvus retrieval adapters.

## Decision

Add two concrete retrieval adapters:

- `HTTPElasticsearchRetrievalClient`
- `PymilvusRetrievalClient`

The Elasticsearch adapter uses the Python standard library HTTP stack with an injectable transport
for tests. BM25 search uses `multi_match` over `title^1.5` and `text`, deterministic sorting by
`_score desc` then `chunk_id asc`, and source fields needed to construct `RetrievalHit`.

The Elasticsearch adapter also implements `enrich_by_chunk_ids(...)` using `_mget`, preserving input
hit order and original hit score/source fields. Missing enrich results are kept in place with
`metadata["enrich_missing"] = True`.

The Milvus adapter lazy-imports `pymilvus.MilvusClient` and accepts an injected fake client in tests.
Search calls pass `collection_name`, `data=[vector]`, `anns_field`, `limit`, `output_fields`, and
metric search params. Vector fields are not copied into retrieval metadata.

Factory functions create clients from platform config:

- `elasticsearch_retrieval_client_from_config(...)`
- `milvus_retrieval_client_from_config(...)`

## Consequences

Positive effects:

- `run_retrieval(...)` can now be wired to live Elasticsearch and Milvus backends.
- Tests remain hermetic via fake HTTP transport and fake Milvus client.
- Adapter errors avoid leaking credentials.

Tradeoffs:

- Real connectivity smoke tests remain separate from unit tests.
- Rewrite and rerank adapters remain future work.

Non-goals:

- No benchmark runner.
- No metrics logic changes.
- No CLI or HTTP server.
- No real ES/Milvus network access in tests.
