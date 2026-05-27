# sci-retrieval-eval

Artifact-driven offline evaluation platform for scientific literature retrieval.

The project exists to make corpus builds, retrieval runs, metric runs, and benchmark comparisons reproducible and auditable across the研发团队.

## Layout

- `AGENTS.md`: agent collaboration rules and file ownership.
- `TASK.md`: local ignored task file maintained by the validator session.
- `report.md`: development report updated by the dev session and included in PR evidence.
- `docs/architecture.md`: project background, architecture, current stage, and engineering principles.
- `docs/decisions/`: accepted ADRs and design history.
- `docs/operations/`: real-environment runbooks.
- `src/eval_platform/`: implementation modules.
- `tests/`: tests organized by module.

## Main Modules

- `artifacts`: local/S3 artifact stores and manifests.
- `datasets`: raw and normalized dataset schemas and converters.
- `chunking`: chunk schema, runner, and external chunker provenance.
- `embeddings`: embedding clients, runner, and artifact IO.
- `indexes`: Elasticsearch and Milvus ingest artifacts.
- `retrieval`: retrieval adapters, fusion, rerank integration, traces, replay.
- `metrics`: MTEB-style metric computation from retrieval artifacts.
- `benchmark`: retrieval + metrics orchestration.
- `mteb_adapter`: MTEB data and interface adapters.

## Development

```bash
pip install -e ".[dev]"
evalctl version
pytest
ruff check .
```

Project rules for AI coding agents are defined in `AGENTS.md`.

Architecture and roadmap context are in `docs/architecture.md`.
