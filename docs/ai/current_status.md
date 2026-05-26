# Current Status

## Current Phase

Embedding runner to S3 PR ready for merge review.

## Implemented

- Local/S3 artifact store
- Normalized dataset schema
- MTEB per-dataset normalizer registry
- Chunked corpus schema
- Chunking runner
- Embedding schema and artifact read/write
- Embedding runner with injectable client and separate source/output stores

## In Progress

- PR / merge review for `feat/embedding-runner-s3`

## Not Implemented

- Real embedding API client
- Milvus index builder
- ES index builder
- Retrieval pipeline
- Metrics and reports
- Frontend dashboard

## Next Task

Open and merge embedding runner S3 PR, then start `feat/embedding-api-client`.
