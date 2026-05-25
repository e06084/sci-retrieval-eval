# Current Status

## Current Phase

Chunking schema PR ready for merge review.

## Implemented

- Project rules (`AGENTS.md`)
- Architecture and AI collaboration docs (`docs/`)
- Local artifact store (`LocalArtifactStore`)
- S3 artifact store (`S3ArtifactStore`)
- Normalized dataset schema and artifact read/write
- MTEB dataset adapter (convert, load, export)
- Chunked corpus schema with external chunker provenance metadata
- Chunk JSONL helpers (`dump_chunks_jsonl`, `load_chunks_jsonl`)
- Chunked corpus source artifact dependency in manifest
- ChunkRecord validation and artifact read/write tests

## In Progress

Nothing.

## Not Implemented

- Chunking runner and external chunker invocation
- Embedding pipeline
- ES/Milvus index builder
- Retrieval pipeline
- Metrics and reports
- Frontend dashboard

## Current Risks

- External chunker git cleanliness is not enforced until the runner stage.
- MTEB doc-level evaluation and internal chunk-level evidence evaluation need to be clearly separated.

## Next Task

Open and merge chunking schema PR, then start `feat/chunking-runner`.
