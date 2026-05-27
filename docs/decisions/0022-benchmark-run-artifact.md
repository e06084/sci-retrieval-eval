# 0022. Benchmark Run Artifact

- Status: Accepted
- Date: 2026-05-27

## Context

The platform now has separate `retrieval_run` and `metrics_run` artifacts. A benchmark experiment
needs a small orchestration artifact that records which retrieval setting was run, which metrics
setting was computed, and which child artifacts were produced.

## Decision

Add a new artifact type:

```text
benchmark_run
```

Artifact layout:

```text
benchmark_run/<artifact_id>/
  summary.json
  _MANIFEST.json
  _SUCCESS
```

The benchmark runner:

1. Calls `run_retrieval(...)`.
2. Calls `run_metrics(...)` against the generated retrieval artifact.
3. Writes a compact benchmark summary and manifest.

The summary records only run-level fields:

- benchmark artifact id
- setting name
- source normalized dataset artifact id
- retrieval run artifact id
- metrics run artifact id
- main score and metric
- aggregate metrics

It intentionally does not duplicate per-query metrics or retrieval hits.

The runner supports both retrieval execution modes:

- `live`: clients are passed through to `run_retrieval(...)`.
- `replay`: an existing retrieval run is replayed, then metrics are recomputed.

## Consequences

Positive effects:

- Benchmark experiments are auditable as first-class artifacts.
- Retrieval and metrics stay separated; benchmark orchestration does not compute metrics itself.
- Replay benchmarks can recompute metrics without touching retrieval services.

Tradeoffs:

- This is a minimal Python runner, not a CLI or batch scheduler.
- Multi-setting comparison and report generation remain future work.

Non-goals:

- No real connectivity smoke.
- No rewrite/rerank adapters.
- No HTML/Markdown report generation.
