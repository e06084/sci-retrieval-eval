# Current Status

## Current Phase

Version-pinned external chunker adapter / Sciverse admin-ingest adapter completed locally and pending PR / merge review.

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
- Version-pinned external chunker adapter
- Sciverse admin-ingest thin adapter with version-pinned repo checks

## In Progress

- PR / merge review
- Real `sciverse_clean` smoke validation

## Not Implemented

- Embedding pipeline
- ES/Milvus index builder
- Retrieval pipeline
- Metrics and reports
- Frontend dashboard

## Current Risks

- MTEB doc-level evaluation and internal chunk-level evidence evaluation need to be clearly separated.
- `sciverse_clean` 真实输出字段可能与 fake repo 测试 still differ，需要再做一次真实 smoke。

## Next Task

Open and merge the Sciverse-path adapter PR, then start `feat/embedding-schema`.
