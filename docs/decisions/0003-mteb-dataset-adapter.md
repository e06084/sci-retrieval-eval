# 0003: MTEB Dataset Adapter Scope

## Status

Accepted

## Context

MTEB retrieval tasks are the first benchmark source for this platform, but downstream pipeline
steps consume normalized dataset artifacts rather than MTEB-native structures.

The project already defines corpus, query, and qrel JSONL schemas and artifact read/write helpers.

## Decision

The MTEB adapter layer will only:

- load MTEB retrieval tasks
- convert them into `NormalizedDataset`
- export normalized dataset artifacts through `ArtifactStore`

This stage will not implement chunking, embedding, indexing, retrieval, or metrics.

`mteb` will be an optional dependency via the `[mteb]` extra. Unit tests will use fake task
objects and must not download real MTEB data or access the network.

## Consequences

Benefits:

- clear boundary between benchmark ingestion and pipeline execution
- normalized artifacts can be consumed by local and S3 backends consistently
- tests remain fast and offline

Tradeoffs:

- raw MTEB download artifacts and dataset version pinning are deferred
- MTEB-standard scoring remains in a later metrics stage
- chunking must preserve chunk-to-document mappings separately
