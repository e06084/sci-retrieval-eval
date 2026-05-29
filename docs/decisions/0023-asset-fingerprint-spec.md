# 0023. Asset Fingerprint Spec

- Status: Accepted
- Date: 2026-05-30

## Context

Track B needs a reliable way to decide whether a requested logical asset is equivalent to an
existing artifact. Existing artifact dependencies and complete markers are necessary but not enough:

- `_SUCCESS` only proves an artifact was completed.
- artifact dependencies prove lineage, not logical equivalence.
- `artifact_id`, `run_id`, ES index name, and Milvus collection name are physical or operational
  identities, not asset identities.

The platform therefore defines `asset_fingerprint` for the core reusable stages:

```text
raw_dataset
normalized_dataset
chunked_corpus
embeddings
elasticsearch_index
milvus_collection
retrieval_run
metrics_run
```

`benchmark_run` and `benchmark_suite_run` are orchestration / aggregation artifacts and are not
part of the current fingerprint equivalence scope.

## Global Rules

Every asset fingerprint must exclude:

- `run_id`
- `artifact_id`
- `created_at`, `updated_at`, `started_at`, `completed_at`, `timestamp`, or other timestamps
- `created_by`
- API keys, access keys, tokens, passwords, Authorization headers, and other secrets
- physical resource names such as Elasticsearch index name or Milvus collection name
- real service addresses such as direct endpoint URLs, hosts, ports, ES URLs, or Milvus URIs
- request ids, trace file names, and trace paths

Every reusable artifact should eventually be reused only when all checks pass:

- artifact complete marker exists
- artifact type matches the requested stage
- dependency chain is compatible with the requested upstream assets
- `asset_fingerprint` matches the requested logical asset

Result summaries, counts, validation flags, connection aliases, and physical locations are useful
artifact manifest metadata, but they do not enter the fingerprint unless they directly define asset
semantics.

`raw_source_uri` and `source_git_remote_url` are allowed stable identity fields: they identify the
raw data source snapshot and chunker source repository, respectively. They are not interchangeable
with real service connection fields in free-form parameter dictionaries. `endpoint_alias` is also
allowed because it is a controlled logical alias rather than a direct service URL.

Free-form parameter dictionaries such as `builder_params`, `ingest_params`, `search_params`,
`query_embedding`, `rewrite`, `rerank`, `call_params`, and `metric_params` must reject physical
connection or run-instance keys including `index_name`, `collection_name`, endpoint URLs, hosts,
ports, request ids, and trace paths.

## Raw Dataset

`raw_dataset` identity is the source data snapshot.

Fingerprint includes:

- `dataset_name`
- `raw_source_uri`
- `raw_format`
- `split`
- `file_fingerprints`
  - `path`
  - `size_bytes`
  - `sha256`

Audit metadata, not fingerprint:

- object-store `etag`
- object-store `version_id`
- `record_count` or `row_count`
- import time
- local cache path

Notes:

- `raw_source_uri` alone is not enough because a URI can be overwritten.
- `sha256` is the preferred content identity. `etag` is audit metadata unless its content-hash
  semantics are explicitly guaranteed.
- `file_fingerprints` has set semantics. Builders canonical-sort entries by `path`, `sha256`, and
  `size_bytes` before hashing so object-store listing order cannot change the fingerprint.

## Normalized Dataset

`normalized_dataset` identity is the deterministic normalization of a raw dataset into the platform
schema.

Fingerprint includes:

- `raw_dataset_fingerprint`
- `normalizer_name`
- `normalizer_version`
- `schema_version`
- `normalizer_params`

Audit metadata, not fingerprint:

- `corpus_count`
- `query_count`
- `qrel_count`
- output file paths

Counts are important sanity checks, but the identity is defined by upstream raw data and
normalization rules.

## Chunked Corpus

`chunked_corpus` identity is the deterministic chunking of a normalized dataset.

Fingerprint includes:

- `normalized_dataset_fingerprint`
- `chunker_source`
  - for example `sciverse` or `sci-retrieval-eval`
- `chunker_name`
- `source_git_remote_url`
- `git_commit`
- `chunker_entrypoint`
  - optional, explicitly `None` when not needed
- `chunk_params`
  - for example `chunk_size`, `chunk_overlap`, `chunk_type`
  - schema can vary by chunker implementation
- `schema_version`

Audit metadata or build preconditions, not fingerprint:

- `git_status`
- `git_dirty`
- `is_dirty`
- `commit_reachable_from_remote`
- `doc_count`
- `chunk_count`
- `failed_doc_count`
- chunk output path

The build should require a clean working tree and a commit recorded or reachable on the remote, but
those checks are validation metadata rather than fingerprint input. Do not introduce a separate
`chunker_fingerprint` for now; keep chunker provenance directly in the chunked corpus components.

## Embeddings

`embeddings` identity is the vectorization of `chunk.text` under a specific model and API call
configuration. The platform does not concatenate title or metadata into corpus embedding input.

Fingerprint includes:

- `chunked_corpus_fingerprint`
- `embedding_source`
  - for example `sciverse_internal`, `openai`, or `local_sentence_transformers`
- `model_name`
- `model_revision`
  - optional when a provider exposes only a stable model name
- `embedding_dim`
- `endpoint_alias`
- `api_version`
- `input_field`
  - `text` for corpus embeddings
- `call_params`
  - provider/API parameters that affect embedding output
- `normalized`
  - whether stored vectors are normalized
- `storage_type`
  - for example `json_float`

Audit metadata, not fingerprint:

- API key or token
- real endpoint URL
- request ids
- retry policy
- timeout
- batch size, unless it can change model output

Provider-side text normalization, punctuation handling, and tokenization are represented by
`embedding_source`, `model_name`, `model_revision`, `endpoint_alias`, `api_version`, and
`call_params`; the platform should not add hidden preprocessing outside those components.

## Elasticsearch Index

`elasticsearch_index` identity is a lexical index built from a chunked corpus under a specific
builder, mapping, settings, and ingest policy.

Fingerprint includes:

- `chunked_corpus_fingerprint`
- `builder_source`
  - normally `sci-retrieval-eval`
- `code_git_commit`
- `builder_entrypoint`
- `builder_params`
  - how chunk records become ES documents, such as id field, text fields, metadata fields, and
    empty-text policy
- `mapping`
- `settings`
  - especially `analysis`, analyzers, tokenizers, and filters
- `ingest_params`
  - only params that change indexed content or retrieval semantics

Artifact manifest metadata, not fingerprint:

- ES index name
- ES URL, if it contains no credentials
- ES cluster alias or connection profile name
- username/password/token
- bulk batch size
- retry policy
- timeout
- refresh timing, unless it changes validated asset semantics

Elasticsearch access information should be recorded in artifact manifest metadata so a completed
artifact can be located and queried later. Prefer a stable `cluster_alias` or connection profile
plus `index_name`; if an ES URL is recorded directly, it must not contain credentials.

## Milvus Collection

`milvus_collection` identity is a vector index built from a chunked corpus and an embeddings asset.

Fingerprint includes:

- `chunked_corpus_fingerprint`
- `embeddings_fingerprint`
- `builder_source`
  - normally `sci-retrieval-eval`
- `code_git_commit`
- `builder_entrypoint`
- `builder_params`
  - how chunk and embedding records become Milvus rows
- `schema`
- `metric_type`
- `index_type`
- `index_params`

Artifact manifest metadata, not fingerprint:

- Milvus collection name
- Milvus URI, if it contains no credentials
- Milvus alias or connection profile name
- username/password/token
- ingest batch size
- retry policy
- timeout
- search-time params

Index build params belong to the collection fingerprint. Search-time params such as HNSW `ef` or
IVF `nprobe` belong to `retrieval_run`.

## Retrieval Run

`retrieval_run` identity is the retrieval configuration over a normalized query set and one or more
index assets.

Fingerprint includes:

- `normalized_dataset_fingerprint`
- `retrieval_mode`
  - `es`, `milvus`, or `hybrid`
- `elasticsearch_index_fingerprint`
- `milvus_collection_fingerprint`
- `query_source`
  - for example `query_limit` or an explicit query id selection fingerprint
- `query_embedding`
  - query embedding identity, independent from corpus embeddings because query/document call params
    can differ
- `search_params`
  - ES top-k, query fields, boosts, operators
  - Milvus top-k and search-time params such as `ef` or `nprobe`
  - fusion method and RRF/candidate parameters
- `rewrite`
- `rerank`
- `trace_mode`

Audit metadata, not fingerprint:

- retrieval run artifact id
- request ids
- real service URLs
- credentials
- trace file paths
- timeout / retry

Current `trace_mode="replay"` records the actual configured ES hits, Milvus hits, fused hits,
rerank input, rerank hits, and final hits. "All recall data" means all hits requested by the
configured top-k / candidate parameters, not every possible match in the backing services.

## Metrics Run

`metrics_run` identity is metric computation over a normalized dataset and a retrieval run.

Fingerprint includes:

- `normalized_dataset_fingerprint`
- `retrieval_run_fingerprint`
- `metrics_source`
  - normally `sci-retrieval-eval`
- `code_git_commit`
- `metrics_entrypoint`
- `metric_params`
  - metrics and cutoffs
  - main metric
  - chunk-to-doc projection policy
  - missing-query policy
  - relevance threshold or qrel filtering policy, if configured

Audit metadata, not fingerprint:

- metrics run artifact id
- aggregate score summaries
- per-query output paths
- compute timestamp

The chunk-to-doc projection policy is part of metric semantics because retrieval can return
chunk-level hits while qrels are doc-level.

## Rebuild Impact

Changing raw dataset identity rebuilds every downstream asset.

Changing normalized dataset identity rebuilds:

```text
chunked_corpus
embeddings
elasticsearch_index
milvus_collection
retrieval_run
metrics_run
```

Changing `git_commit`, `chunk_params`, or `chunker_entrypoint` rebuilds:

```text
chunked_corpus
embeddings
elasticsearch_index
milvus_collection
retrieval_run
metrics_run
```

Changing embedding identity rebuilds:

```text
embeddings
milvus_collection
retrieval_run
metrics_run
```

Changing Elasticsearch or Milvus index identity rebuilds:

```text
retrieval_run
metrics_run
```

Changing rewrite, rerank, search params, query source, or trace mode rebuilds:

```text
retrieval_run
metrics_run
```

Changing metric params rebuilds:

```text
metrics_run
```

Changing only physical names such as `run_id`, `artifact_id`, ES index name, or Milvus collection
name should not change logical asset fingerprints.

## Consequences

Positive effects:

- Asset reuse can become independent of physical artifact ids.
- Minimal rebuild planning can reason over logical equivalence instead of naming conventions.
- Downstream fingerprints naturally change when upstream logical assets change.

Tradeoffs:

- Existing artifact writers still need follow-up work to write `asset_fingerprint` into manifest
  metadata.
- Reuse planner behavior should change only after fingerprints are written consistently.
- Query embedding, rewrite, and rerank identities are represented as retrieval components rather
  than standalone artifact types for now.

Non-goals for PR1:

- No planner behavior change.
- No artifact writer integration.
- No benchmark run or benchmark suite fingerprint.
- No force-rebuild or pinned-artifact semantics.
- No benchmark variant spec.
- No real external service access.
