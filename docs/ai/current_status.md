# Current Status

## Current Phase

Normalized dataset schema PR ready for merge review.

## Implemented

- Project rules (`AGENTS.md`)
- Architecture and AI collaboration docs (`docs/`)
- Package skeleton under `src/eval_platform/`
- CLI entry point: `evalctl version`
- Dev tooling: pytest, ruff, mypy (configured in `pyproject.toml`)
- Local artifact store (`LocalArtifactStore`)
- S3 artifact store (`S3ArtifactStore`)
- Artifact manifest schema and `ArtifactStore` interface
- Normalized dataset schema (`CorpusRecord`, `QueryRecord`, `QrelRecord`, `NormalizedDataset`)
- JSONL helpers and normalized dataset artifact read/write

## In Progress

Nothing.

## Not Implemented

- MTEB dataset adapter
- Chunking pipeline
- Embedding pipeline
- ES/Milvus index builder
- Retrieval pipeline
- Metrics and reports
- Frontend dashboard

## Current Risks

- AI agents may create unmaintainable scripts if project rules are not strict.
- MTEB doc-level evaluation and internal chunk-level evidence evaluation need to be clearly separated.
- Manifest schema evolution must remain backward compatible via `schema_version`.

## Next Task

Open and merge dataset schema PR, then start `feat/mteb-dataset-adapter`.
