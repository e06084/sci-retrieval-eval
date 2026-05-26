# Current Status

## Current Phase

Embedding consistency hardening ready for merge review.

## Implemented

- Local/S3 artifact store
- Normalized dataset schema
- MTEB per-dataset normalizer registry
- Chunked corpus schema
- Chunking runner
- Embedding schema and artifact read/write
- Embedding runner with injectable client and separate source/output stores
- HTTP embedding client
- HTTP embedding request payload fix merged on `main`
- Multi-endpoint embedding config
- Multi-endpoint embedding consistency pre-check helper
- Embedding provenance endpoint and consistency metadata

## In Progress

- None for the current phase

## Not Implemented

- Milvus index builder
- ES index builder
- Retrieval pipeline
- Metrics and reports
- Frontend dashboard
 
## Next Task
Open and merge embedding consistency hardening PR, then start ES / Milvus ingest artifact identity work.
