# 0005: Chunking Runner with External Chunker Provenance

## Status

Accepted

## Context

Actual chunking logic lives in an external repository. Chunked corpus artifacts must record
which external chunker commit produced them. A dirty external chunker repository breaks
reproducibility because uncommitted changes are not captured by commit SHA alone.

The chunked corpus schema stage defined `ChunkerProvenance` and artifact metadata conventions
without executing git commands or invoking external chunkers.

## Decision

Introduce a chunking runner that:

- reads a normalized dataset artifact from an `ArtifactStore`
- checks the external chunker repository is clean before execution
- fails immediately when the external repo is dirty
- records `ChunkerProvenance` on the output chunked corpus manifest
- accepts an injected `ExternalChunker` protocol implementation

Git inspection uses local `git` commands via `subprocess` with `cwd=repo_path`. This PR does
not bind to a specific external chunker library API.

## Consequences

Benefits:

- reproducibility requires both platform code SHA and external chunker commit SHA
- dirty external repos are rejected before chunking runs
- real sciverse or other chunkers can be integrated via a thin adapter

Tradeoffs:

- runner tests depend on a local `git` executable
- no real external chunker adapter is included in this PR
- batch-only external APIs may need adapter wrappers later
