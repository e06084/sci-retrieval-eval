# Architecture

## Overview

This project implements an artifact-driven evaluation platform for scientific literature retrieval.

The evaluation pipeline is:

```text
raw dataset
  -> normalized dataset
  -> chunked corpus
  -> embeddings
  -> ES/Milvus indexes
  -> retrieval inference
  -> predictions and traces
  -> metrics and reports
```

## Artifact-Driven Design

Each step consumes one or more input artifacts and produces one or more output artifacts.

Each artifact directory must contain:

- `_MANIFEST.json`
- `_SUCCESS` if complete
- data files or part files

Downstream steps must validate `_MANIFEST.json` and `_SUCCESS` before consuming an artifact.

## Main Modules

### artifacts

Responsible for:

- artifact manifest
- checksum
- S3 backend
- local backend
- local cache
- resume support

### datasets

Responsible for:

- loading MTEB datasets
- converting raw data into normalized corpus/query/qrels artifacts

### chunking

Responsible for:

- calling sciverse chunking logic
- storing chunk metadata
- preserving mapping from chunk ID to source document ID

### embeddings

Responsible for:

- calling embedding API
- batching
- retry
- storing embedding shards

### indexes

Responsible for:

- building ES indexes
- building Milvus collections
- validating index completeness
- writing index artifact metadata

### retrieval

Responsible for:

- ES recall
- Milvus recall
- RRF fusion
- reranking
- final candidate generation
- per-query trace recording

### mteb_adapter

Responsible for:

- exposing the retrieval system through an MTEB-compatible search interface
- returning query-to-document score mappings

### metrics

Responsible for:

- MTEB-compatible scoring
- internal evidence scoring
- report generation

### frontend

Responsible for:

- read-only artifact registry view
- run result view
- query-level result exploration

## Evaluation Modes

### MTEB-Standard Mode

Used for comparable MTEB metrics.

Results should be scored at the document level or the dataset's original relevance unit.

### Internal Evidence Mode

Used for evaluating chunk-level evidence quality for Agent consumption.

These metrics are internal and should not be compared directly with MTEB leaderboard results.
