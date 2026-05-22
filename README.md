# sci-retrieval-eval

Artifact-driven evaluation platform for scientific literature retrieval.

## Scope

This repository is intended to become the clean project home for:

- benchmark dataset preparation
- artifact management on S3
- chunking and embedding pipelines
- ES and Milvus index building
- retrieval inference
- MTEB-compatible evaluation
- report generation

The first milestone is project bootstrap. At this stage, the repository provides:

- project rules
- architecture boundaries
- package skeleton
- CLI entry point (`evalctl version`)
- documentation entry points

## Layout

- `AGENTS.md`: AI collaboration and engineering rules
- `docs/ai/project_brief.md`: project context for coding agents
- `docs/ai/current_status.md`: current implementation status
- `docs/architecture.md`: system architecture
- `src/eval_platform/`: implementation modules
- `tests/`: unit and integration tests

## Planned Modules

- `src/eval_platform/artifacts/`
- `src/eval_platform/datasets/`
- `src/eval_platform/chunking/`
- `src/eval_platform/embeddings/`
- `src/eval_platform/indexes/`
- `src/eval_platform/retrieval/`
- `src/eval_platform/mteb_adapter/`
- `src/eval_platform/metrics/`
- `src/eval_platform/frontend/`
- `src/eval_platform/cli/`

## Development

```bash
pip install -e ".[dev]"
evalctl version
pytest
ruff check .
```

Project rules for AI coding agents are defined in `AGENTS.md`.

Architecture documents are in `docs/`.

## Current Status

See `docs/ai/current_status.md`.

## Next Step

Implement the first core module: artifact store.
