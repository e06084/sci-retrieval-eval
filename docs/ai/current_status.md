# Current Status

## Current Phase

Chunking runner PR ready for merge review.

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
- Chunking runner with git clean-state inspection

## In Progress

Nothing.

## Not Implemented

- Real external chunker adapter
- Embedding pipeline
- ES/Milvus index builder
- Retrieval pipeline
- Metrics and reports
- Frontend dashboard

## Current Risks

- MTEB doc-level evaluation and internal chunk-level evidence evaluation need to be clearly separated.
- External chunker API shape may differ from the injected `ExternalChunker` protocol.

## Next Task

Open and merge chunking runner PR, then start `feat/embedding-schema`.
