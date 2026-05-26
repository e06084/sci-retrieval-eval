# Current Status

## Current Phase

Corpus build runner v1 in progress.

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
- Milvus ingest runner from aligned sharded `chunked_corpus` + `embeddings` to auditable `milvus_collection` artifact
- Corpus build runner v1 for IFIRNFCorpus with run-level `corpus_build` artifact

## In Progress

- Corpus build runner PR validation

## Not Implemented

- Retrieval pipeline
- Metrics and reports
- Frontend dashboard
- Multi-dataset raw normalizer expansion
- Corpus build CLI / scheduler

## Next Task

Open and merge corpus build runner PR, then start retrieval pipeline design.
