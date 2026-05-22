# Project Brief

## Background

We are building an evaluation platform for a scientific literature retrieval system.

The production retrieval system currently uses:
- Elasticsearch for lexical retrieval
- Milvus for vector retrieval
- RRF for fusion
- reranker for final ranking

The source data is mainly parsed from PDF files. The parsed text is chunked, indexed into Elasticsearch, embedded, and inserted into Milvus.

At query time, the system retrieves document chunks as evidence candidates. These candidates are intended to be used by an Agent for scientific research.

## Evaluation Tasks

The initial evaluation datasets are based on MTEB retrieval tasks:
- LitSearch
- SciFact
- IFIR-SciFact
- IFIR-NFCorpus
- NFCorpus

## Main Goal

Build a reproducible, artifact-driven evaluation system that can:
1. Prepare corpus artifacts.
2. Build ES and Milvus indexes.
3. Run retrieval inference.
4. Save detailed traces.
5. Compute MTEB-compatible metrics.
6. Generate evaluation reports.
7. Support per-query error analysis.

## Non-Goals for the First Version

The first version should not introduce:
- SQL database
- Redis
- Airflow
- MLflow
- DVC
- complex frontend task scheduling
- production-grade distributed workflow engine

S3 should be used as the persistent artifact registry. Local storage can be used for cache and temporary files.

## Important Design Principle

The platform should be artifact-driven.

Each step should:
- read input artifacts
- validate manifests
- produce output artifacts
- write manifest and success marker

No step should rely on hidden mutable state.
