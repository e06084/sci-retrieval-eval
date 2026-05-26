# Current Status

## Current Phase

Elasticsearch ingest artifact in progress.

## Implemented

- Local/S3 artifact store
- Raw dataset snapshot artifact
- Raw snapshot to normalized dataset API
- Normalized dataset schema
- MTEB per-dataset normalizer registry
- Chunked corpus schema
- Chunking runner
- Embedding schema and artifact read/write
- Embedding runner with injectable client and separate source/output stores
- HTTP embedding client
- Multi-endpoint embedding consistency pre-check helper
- Platform config system
- Shard-aware `chunked_corpus` artifact layout
- Shard-aware `embeddings` artifact layout aligned to source chunk shards
- Stream-oriented `iter_chunk_shards(...)` / `iter_embedding_shards(...)`
- Stream-oriented embedding shard writer without full-corpus accumulation
- Reusable progress reporter for raw-to-normalized / chunking / embedding
- Elasticsearch ingest runner from sharded `chunked_corpus` to auditable `elasticsearch_index` artifact

## In Progress

- Elasticsearch ingest PR validation

## Not Implemented

- Corpus build runner
- Milvus ingest
- Retrieval pipeline
- Metrics and reports
- Frontend dashboard

## Next Task

Open and merge Elasticsearch ingest PR, then start Milvus ingest on top of shard zip join.
