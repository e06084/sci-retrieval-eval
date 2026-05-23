# 0002: Use Normalized Dataset JSONL Schema

## Status

Accepted

## Context

Downstream MTEB adapter, chunking, embedding, indexing, and inference steps all need a stable,
dataset-agnostic representation of corpus, queries, and relevance judgments.

Different upstream sources use different formats. The evaluation platform needs one normalized
schema before pipeline steps can share artifacts reliably.

## Decision

Use a normalized dataset artifact with three JSONL files:

- `corpus.jsonl`
- `queries.jsonl`
- `qrels.jsonl`

Each qrel record contains `query_id`, `doc_id`, and graded `relevance`.

This schema represents document-level relevance (or the dataset's original relevance unit). It is
not a chunk-level evidence schema.

## Consequences

Benefits:

- one artifact format for local and S3 backends
- simple JSONL read/write without pandas or HuggingFace
- graded relevance is supported from the first version

Tradeoffs:

- the MTEB adapter must convert MTEB-native structures into this schema
- chunking must preserve mappings from chunk IDs back to document IDs
- MTEB-standard metrics and internal chunk evidence metrics must remain separate
