# AGENTS.md

## Project Goal

This repository implements an artifact-driven evaluation platform for a scientific literature retrieval system.

The target retrieval system uses:
- Elasticsearch lexical retrieval
- Milvus vector retrieval
- RRF fusion
- reranker
- MTEB retrieval tasks including LitSearch, SciFact, IFIR-SciFact, IFIR-NFCorpus, and NFCorpus

The platform should prepare corpora, build indexes, run retrieval inference, compute metrics, and generate evaluation reports.

## Core Architecture

The system is artifact-driven.

Each pipeline step consumes artifacts and produces artifacts. S3 is the persistent artifact registry. Local storage may be used as a cache or temporary workspace.

Expected artifact types include:
- raw dataset
- normalized dataset
- chunked corpus
- embeddings
- ES/Milvus index metadata
- inference predictions
- retrieval traces
- metric reports

## Hard Rules

- Do not create one-off scripts outside `eval_platform/cli/`.
- Every pipeline step must be idempotent.
- Every output artifact must include `_MANIFEST.json`.
- Completed artifacts must include `_SUCCESS`.
- Downstream steps must not consume artifacts without `_SUCCESS`.
- Do not introduce SQL, Redis, Airflow, MLflow, DVC, or Celery without explicit approval.
- Do not hardcode S3 buckets, API keys, ES endpoints, Milvus endpoints, model names, or credentials.
- Do not access production ES, Milvus, or S3 resources in tests.
- New features must include tests.
- Prefer typed configuration models over loose dictionaries.
- Keep changes small and focused.
- Do not modify unrelated modules.
- Do not silently change public schemas.
- Do not change project architecture without updating docs.

## Module Boundaries

- `eval_platform/artifacts/`: artifact storage, manifests, checksums, local cache, S3 backend
- `eval_platform/datasets/`: MTEB dataset loading and normalized schemas
- `eval_platform/chunking/`: wrapper around sciverse chunking logic
- `eval_platform/embeddings/`: embedding API client and batch embedding writer
- `eval_platform/indexes/`: ES and Milvus index builders
- `eval_platform/retrieval/`: ES recall, Milvus recall, RRF fusion, reranker pipeline
- `eval_platform/mteb_adapter/`: MTEB SearchProtocol-compatible integration
- `eval_platform/metrics/`: metric computation and report generation
- `eval_platform/frontend/`: read-only dashboard
- `eval_platform/cli/`: official command-line entry points only

## Testing Rules

Tests must not call real external services unless explicitly marked as integration tests.

Use mocks, local fixtures, or fake clients for:
- S3
- ES
- Milvus
- embedding API
- reranker API

Before completing a coding task, run:
- `pytest`
- `ruff check .`
- `mypy .` if mypy is configured

## Done Means

A task is not done unless:
- Tests pass.
- No credentials are committed.
- No unrelated files are changed.
- New schemas or configs are documented.
- The task summary explains what changed and why.
- Any known limitations are explicitly documented.
