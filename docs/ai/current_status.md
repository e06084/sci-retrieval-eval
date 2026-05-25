# Current Status

## Current Phase

MTEB dataset adapter PR ready for merge review.

## Implemented

- Project rules (`AGENTS.md`)
- Architecture and AI collaboration docs (`docs/`)
- Package skeleton under `src/eval_platform/`
- CLI entry point: `evalctl version`
- Dev tooling: pytest, ruff, mypy (configured in `pyproject.toml`)
- Local artifact store (`LocalArtifactStore`)
- S3 artifact store (`S3ArtifactStore`)
- Artifact manifest schema and `ArtifactStore` interface
- Normalized dataset schema, JSONL helpers, and artifact read/write
- MTEB dataset adapter (convert, load, export)

## In Progress

Nothing.

## Not Implemented

- Chunking pipeline
- Embedding pipeline
- ES/Milvus index builder
- Retrieval pipeline
- Metrics and reports
- Frontend dashboard

## Current Risks

- AI agents may create unmaintainable scripts if project rules are not strict.
- MTEB doc-level evaluation and internal chunk-level evidence evaluation need to be clearly separated.
- MTEB task APIs may vary across versions; extraction logic must stay defensive.

## Next Task

Open and merge MTEB dataset adapter PR, then start `feat/chunking-schema`.
