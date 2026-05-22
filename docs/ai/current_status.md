# Current Status

## Current Phase

S3 artifact backend PR ready for merge review.

## Implemented

- Project rules (`AGENTS.md`)
- Architecture and AI collaboration docs (`docs/`)
- Package skeleton under `src/eval_platform/`
- CLI entry point: `evalctl version`
- Dev tooling: pytest, ruff, mypy (configured in `pyproject.toml`)
- Artifact manifest schema (`ArtifactManifest`, `ArtifactFile`, `ArtifactDependency`)
- Artifact store abstract interface (`ArtifactStore`) with `artifact_uri()`
- Local artifact store (`LocalArtifactStore`) with path safety and manifest consistency checks
- S3 artifact store (`S3ArtifactStore`) with injectable client for testing

## In Progress

Nothing.

## Not Implemented

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
- Manifest schema evolution must remain backward compatible via `schema_version`.

## Next Task

Open and merge S3 artifact store PR, then start `feat/dataset-schema`.
