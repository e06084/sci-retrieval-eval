# Current Status

## Current Phase

MTEB per-dataset normalizer registry completed locally and pending PR / merge review.

## Implemented

- Project rules (`AGENTS.md`)
- Architecture and AI collaboration docs (`docs/`)
- Local artifact store (`LocalArtifactStore`)
- S3 artifact store (`S3ArtifactStore`)
- Normalized dataset schema and artifact read/write
- MTEB dataset adapter
- MTEB dataset layout compatibility fix
- MTEB per-dataset normalizer registry for:
  - `LitSearchRetrieval`
  - `SciFact`
  - `IFIRScifact`
  - `IFIRNFCorpus`
  - `NFCorpus`
- Chunked corpus schema with external chunker provenance metadata
- Chunk JSONL helpers (`dump_chunks_jsonl`, `load_chunks_jsonl`)
- Chunked corpus source artifact dependency in manifest
- Chunking runner with git clean-state inspection
- Version-pinned external chunker adapter
- Sciverse admin-ingest thin adapter with version-pinned repo checks

## In Progress

- PR / merge review for `feat/mteb-normalizer-registry`
- Follow-up integration work on embedding schema

## Not Implemented

- Embedding pipeline
- ES/Milvus index builder
- Retrieval pipeline
- Metrics and reports
- Frontend dashboard

## Current Risks

- The MTEB target set is stable now, but any future new retrieval task should add an explicit normalizer instead of implicit fallback logic.
- `LitSearchRetrieval` currently requires dataset-specific cleanup for empty corpus rows; future upstream dataset revisions may change the cleanup counts.

## Next Task

Open and merge the MTEB normalizer registry PR, then start `feat/embedding-schema`.
