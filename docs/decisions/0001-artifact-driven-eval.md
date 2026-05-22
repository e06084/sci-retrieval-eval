# 0001: Use Artifact-Driven Evaluation Design

## Status
仓库
Accepted

## Context

The evaluation platform needs to prepare corpora, build indexes, run inference, compute metrics, and support error analysis.

The system should avoid unnecessary infrastructure complexity in the first version. We do not want to introduce SQL, Redis, Airflow, MLflow, or DVC at this stage.

## Decision

Use S3 as the persistent artifact registry.

Each pipeline step will:

- read input artifacts
- validate manifests
- produce output artifacts
- write `_MANIFEST.json`
- write `_SUCCESS` only after successful completion

Local storage may be used as cache or temporary workspace.

## Consequences

Benefits:

- simpler infrastructure
- easier reproducibility
- easier debugging
- suitable for batch evaluation workflows

Tradeoffs:

- no transactional database semantics
- artifact discovery depends on manifest scanning
- frontend should initially be read-only
