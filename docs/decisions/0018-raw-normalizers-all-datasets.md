# 0018. Raw Normalizers For Target Datasets

- Status: Accepted
- Date: 2026-05-26

## Context

The platform can already import immutable `raw_dataset` snapshots and convert a raw snapshot into a `normalized_dataset` artifact. The first raw normalizer only supported `IFIRNFCorpus`, so `run_corpus_build(...)` also rejected all other datasets at config validation time.

The standard offline corpus build path needs to start from raw source for the five target datasets:

- `IFIRNFCorpus`
- `IFIRScifact`
- `NFCorpus`
- `SciFact`
- `LitSearchRetrieval`

The downstream stages should continue to consume one unified `NormalizedDataset` schema and should not know which raw layout produced it.

## Decision

Add an explicit raw normalizer registry in `raw_normalize.py`.

Each registered dataset has a `RawNormalizerSpec`:

- `dataset_name`
- `normalizer_name`
- `raw_format`
- `has_instructions`

Registered normalizers:

| dataset_name | normalizer_name | raw_format | has_instructions |
| --- | --- | --- | --- |
| `IFIRNFCorpus` | `ifir_nfcorpus_raw_jsonl_tsv_v1` | `jsonl_tsv` | `true` |
| `IFIRScifact` | `ifir_scifact_raw_jsonl_tsv_v1` | `jsonl_tsv` | `true` |
| `NFCorpus` | `nfcorpus_raw_jsonl_tsv_v1` | `jsonl_tsv` | `false` |
| `SciFact` | `scifact_raw_jsonl_tsv_v1` | `jsonl_tsv` | `false` |
| `LitSearchRetrieval` | `litsearch_raw_parquet_v1` | `parquet_dir_shards` | `false` |

For `jsonl_tsv` raw datasets:

- `corpus.jsonl` maps `_id`, `title`, `text` to `CorpusRecord`.
- `queries.jsonl` maps `_id`, `text` to `QueryRecord`.
- `qrels/test.tsv` maps `query-id`, `corpus-id`, `score` to `QrelRecord`.
- `instructions.jsonl` is read only when `has_instructions=True` and is stored in query metadata as `instruction`.

For `LitSearchRetrieval` parquet raw datasets:

- `corpus/*.parquet`
- `queries/*.parquet`
- `qrels/*.parquet`

Parquet reading uses lazy imports. It first tries `pandas`, then `pyarrow`; if neither is available, normalization raises a clear `RawNormalizeError`.

Parquet shard handling is deterministic:

- Each shard group is discovered by relative path.
- Shards are sorted by `PurePosixPath(file.path).as_posix()`.
- Rows from all shards in the group are merged before constructing `NormalizedDataset`.
- Missing shard groups fail with a message naming `corpus/*.parquet`, `queries/*.parquet`, or `qrels/*.parquet`.

LitSearch corpus text uses the same data-quality semantics as the MTEB LitSearch normalizer:

- `CorpusRecord.text` is the first non-empty value from `text`, `abstract`, then `title`.
- Documents without usable `text`, `abstract`, or `title` are dropped.
- Qrels pointing to dropped documents are dropped.
- Queries without any remaining qrel are dropped.
- When filtering occurs, manifest metadata records filtered and dropped counts.

The `normalized_dataset` manifest continues to record the raw upstream identity and now also records:

- `raw_format`
- `has_instructions`

`CorpusBuildConfig.dataset_name` now validates against the raw normalizer registry instead of a separate IFIR-only constant. The runner stage order and stage semantics are unchanged.

## Consequences

Positive effects:

- The Python corpus build runner can start from raw source for all five target datasets.
- Raw parsing rules are explicit and auditable per dataset.
- The allowlist is maintained in one place, avoiding duplicated runner-specific dataset checks.
- Downstream chunking, embedding, indexing, retrieval, and metrics continue to depend only on `NormalizedDataset`.

Tradeoffs:

- This still does not run a real five-dataset external smoke against production S3.
- LitSearch requires `pandas` or `pyarrow` only when the parquet shard normalizer is actually used.
- LitSearch raw normalization intentionally drops unusable documents and orphan judgments so the output satisfies `NormalizedDataset` invariants.
- The one-off scripts remain as historical references and are not runtime dependencies.

Non-goals:

- No CLI or scheduler changes.
- No retrieval, metrics, or frontend changes.
- No ES, Milvus, embedding, or chunking semantic changes.
- No real S3 or external service access in unit tests.
