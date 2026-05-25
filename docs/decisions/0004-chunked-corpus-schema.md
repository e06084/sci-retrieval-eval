# 0004: Chunked Corpus Schema with External Chunker Provenance

## Status

Accepted

## Context

Chunking logic lives in an external repository. The evaluation platform must record which
external chunker implementation and commit produced a chunked corpus artifact.

The evaluation platform repository commit (`code_git_sha` on the artifact manifest) and the
external chunker repository commit are different provenance signals and must not be conflated.

Chunked corpus artifacts also depend on upstream normalized dataset artifacts. Manifest
`dependencies` must record that lineage.

## Decision

Define a `ChunkerProvenance` schema and store it in chunked corpus artifact manifest metadata
under the `chunker` key. Optional `chunk_params` are stored alongside provenance.

Expose chunking-specific JSONL helpers (`dump_chunks_jsonl`, `load_chunks_jsonl`) as the
public chunking module interface. They may delegate to generic dataset JSONL helpers internally.

`write_chunked_corpus_artifact()` accepts an optional `source_dependency: ArtifactDependency`
and writes it to `ArtifactManifest.dependencies`.

This stage only defines schema and artifact metadata conventions. It does not:

- execute git commands
- verify external repositories are clean
- call external chunker libraries
- run real chunking

A later `feat/chunking-runner` stage will:

- verify the external chunker repository is clean before execution
- capture the external chunker commit SHA
- invoke the real chunker implementation
- write chunked corpus artifacts using this schema

## Consequences

Benefits:

- reproducibility links artifacts to both platform code and external chunker code
- manifest dependencies express upstream normalized dataset lineage
- schema-only stage keeps the PR small and testable offline
- downstream embedding and indexing can trust manifest provenance fields

Tradeoffs:

- git cleanliness checks are deferred to the runner stage
- chunk file sharding (`chunks/part-*.jsonl`) may evolve in later PRs
