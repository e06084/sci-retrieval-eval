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
- documentation entry points

## Layout

- `AGENTS.md`: AI collaboration and engineering rules
- `docs/ai/project_brief.md`: project context for coding agents
- `src/eval_platform/`: future implementation modules
- `tests/`: unit and integration tests

## Planned Modules

- `eval_platform/artifacts/`
- `eval_platform/datasets/`
- `eval_platform/chunking/`
- `eval_platform/embeddings/`
- `eval_platform/indexes/`
- `eval_platform/retrieval/`
- `eval_platform/mteb_adapter/`
- `eval_platform/metrics/`
- `eval_platform/frontend/`
- `eval_platform/cli/`

## Next Step

Implement the first core module: artifact store.
