# Current Status

## Current Phase

Post-embedding artifact baseline; preparing embedding consistency hardening.

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

## In Progress

- Embedding consistency hardening planning

## Not Implemented

- Real embedding API client
- Milvus index builder
- ES index builder
- Retrieval pipeline
- Metrics and reports
- Frontend dashboard
 
## Next Task
Harden embedding stage, especially multi-endpoint consistency checks, before starting ES / Milvus ingest artifacts.
