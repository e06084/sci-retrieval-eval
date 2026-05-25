# 0006: Version-Pinned External Chunker Adapter

## Status

Accepted

## Context

The chunking pipeline already supports:

- git clean-state inspection
- external chunker provenance in chunked corpus manifests
- normalized dataset input artifacts
- chunked corpus output artifacts

However, real chunk logic lives in another repository. We need a way to call that external logic
without copying its code into this project, while still making runs reproducible.

## Decision

Add a version-pinned external chunker adapter layer with:

- explicit repo path
- explicit expected remote URL
- explicit expected commit SHA
- explicit clean-state validation

Use a thin Python callable adapter:

- validate the external repo checkout first
- dynamically import a Python module from that checkout
- call a specified callable
- normalize returned rows into `ChunkRecord`
- delegate artifact writing to existing `run_chunking`

This project will not:

- git fetch
- git checkout
- auto-fix dirty repos
- copy external chunker code

## Consequences

Benefits:

- reproducible chunking runs pinned to a specific remote URL and commit
- early failure for dirty repos or wrong checkouts
- thin integration boundary between this repo and external chunk logic

Tradeoffs:

- users must prepare the correct external checkout ahead of time
- external API shape changes still require adapter-compatible output
- dynamic imports make runtime validation more important than static linkage
