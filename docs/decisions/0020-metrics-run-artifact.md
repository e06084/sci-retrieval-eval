# 0020. Metrics Run Artifact

- Status: Accepted
- Date: 2026-05-27

## Context

The pipeline now writes normalized datasets and retrieval run artifacts. Retrieval outputs are
chunk-level, while MTEB and pytrec_eval-style metrics operate on doc-level rankings and qrels.
Metrics must be reproducible without calling live retrieval services.

## Decision

Add a new artifact type:

```text
metrics_run
```

Artifact layout:

```text
metrics_run/<artifact_id>/
  metrics.json
  query_metrics/part-00000.jsonl
  ...
  _MANIFEST.json
  _SUCCESS
```

`metrics_run` consumes only:

- `normalized_dataset` qrels
- `retrieval_run` query results

Chunk hits are projected to doc rankings with `first_chunk_rank`:

- Skip hits without `doc_id` and count them.
- Keep only the first chunk for each `doc_id`.
- Re-rank docs consecutively from 1.
- Score docs as `1 / source_chunk_rank`.

Metrics are computed per query, then averaged across evaluated queries. The evaluated query
universe is the set of qrel queries with at least one positive relevance judgment.

Implemented metrics:

- `ndcg_at_k`
- `map_at_k`
- `recall_at_k`
- `precision_at_k`
- `mrr_at_k`
- `hit_rate_at_k`

Default `k_values` are:

```text
[1, 3, 5, 10, 20, 100, 1000]
```

Queries with missing retrieval results or retrieval errors are evaluated with empty results and
therefore zero metrics. Retrieval results not present in positive qrels are ignored and counted.

## Consequences

Positive effects:

- Metrics are deterministic for a fixed `normalized_dataset` and `retrieval_run`.
- Metrics do not call Elasticsearch, Milvus, embedding, rewrite, or rerank services.
- Later benchmark runners can reuse the same `retrieval_run` with different metric settings.

Tradeoffs:

- This PR implements built-in formulas rather than depending on `pytrec_eval`.
- The current doc projection strategy is intentionally narrow: first chunk rank only.

Non-goals:

- No retrieval execution.
- No real ES / Milvus / embedding / rewrite / rerank adapters.
- No complete benchmark runner.
- No CLI or HTTP server.
