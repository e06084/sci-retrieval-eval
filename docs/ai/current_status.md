# Current Status

## Current Phase

Project bootstrap (complete).

## Implemented

- Project rules (`AGENTS.md`)
- Architecture and AI collaboration docs (`docs/`)
- Package skeleton under `src/eval_platform/`
- CLI entry point: `evalctl version`
- Dev tooling: pytest, ruff, mypy (configured in `pyproject.toml`)

## In Progress

Nothing.

## Not Implemented

- Artifact store
- Dataset adapter
- Chunking pipeline
- Embedding pipeline
- ES/Milvus index builder
- Retrieval pipeline
- MTEB adapter
- Metrics and reports
- Frontend dashboard

## Current Risks

- AI agents may create unmaintainable scripts if project rules are not strict.
- MTEB doc-level evaluation and internal chunk-level evidence evaluation need to be clearly separated.
- Artifact versioning and manifest schema need to be defined before implementing pipelines.

## Next Task

Implement artifact manifest schema and local artifact store.
