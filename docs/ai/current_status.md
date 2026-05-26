# Current Status

## Current Phase

Raw dataset artifact ready for merge review.

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
- Raw dataset artifact schema and import helpers, if merged

## In Progress

- None for the current phase

## Not Implemented

- Raw dataset -> normalized dataset dependency wiring
- Milvus index builder
- ES index builder
- Retrieval pipeline
- Metrics and reports
- Frontend dashboard
 
## Next Task
Open and merge raw dataset artifact PR, then start raw_dataset -> normalized_dataset dependency wiring.
