# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`metrics_run artifact`
- 当前分支：`feat/metrics-run-artifact`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-27
- 完成时间：2026-05-27

## 2. 本次改动

- 新增 `metrics_run` schema：
  - `RankedDoc`
  - `ProjectionStats`
  - `QueryMetricsRecord`
  - `MetricsRunData`
- 新增 chunk-level hits 到 doc-level ranking 的投影：
  - `project_retrieval_result_to_docs(...)`
- 新增 IR metrics 公式：
  - `compute_query_metrics(...)`
  - `aggregate_query_metrics(...)`
- 新增 `metrics_run` artifact IO：
  - `write_metrics_run_artifact(...)`
  - `read_metrics_run_artifact(...)`
  - manifest metadata 写入 `queries_per_shard`，并作为系统字段防止用户 metadata 覆盖
- 新增 metrics runner：
  - `MetricsRunConfig`
  - `run_metrics(...)`
- 更新公共导出：
  - `src/eval_platform/metrics/__init__.py`
- 新增 ADR：
  - `docs/decisions/0020-metrics-run-artifact.md`
- 更新：
  - `docs/ai/current_status.md`
  - `report.md`
- 新增测试：
  - `tests/metrics/test_projection.py`
  - `tests/metrics/test_ir.py`
  - `tests/metrics/test_artifact.py`
  - `tests/metrics/test_runner.py`

## 3. Artifact 结构

新增 artifact type：

```text
metrics_run
```

文件结构：

```text
metrics_run/<artifact_id>/
  metrics.json
  query_metrics/part-00000.jsonl
  query_metrics/part-00001.jsonl
  _MANIFEST.json
  _SUCCESS
```

`metrics.json` 保存：

- `aggregate`
- `k_values`
- `main_score`
- `main_score_metric`

`query_metrics/*.jsonl` 每条记录保存：

- `query_id`
- `query_text`
- `retrieval_error`
- `ranked_docs`
- `relevant_docs`
- `metrics`
- `projection_stats`

## 4. Projection 规则

默认策略：

```text
doc_aggregation = first_chunk_rank
doc_score = reciprocal_first_chunk_rank
```

规则：

- 按 chunk hit 的 `rank` 升序处理。
- `doc_id` 为空的 hit 跳过，并计入 `missing_doc_id_hit_count`。
- 同一个 `doc_id` 只保留第一次出现的 chunk，并计入后续重复 `duplicate_doc_hit_count`。
- doc-level rank 从 1 重新连续编号。
- doc-level score 使用 `1 / source_chunk_rank`。
- 不在 retrieval 阶段提前做 doc 聚合。

## 5. Metrics 公式

默认 `k_values`：

```text
[1, 3, 5, 10, 20, 100, 1000]
```

实现指标：

- `precision_at_k = top_k positive doc count / k`
- `recall_at_k = top_k positive doc count / positive doc count`
- `hit_rate_at_k = top_k 是否至少命中一个 positive doc`
- `mrr_at_k = 1 / 第一个 positive doc rank`
- `map_at_k = sum(precision@rank for positive hits <= k) / positive doc count`
- `ndcg_at_k = dcg / idcg`，使用 graded relevance

聚合规则：

- 先计算 per-query metrics。
- aggregate 是 evaluated queries 的算术平均。
- retrieval error / missing result query 只要有 positive qrels，就计入平均且指标为 0。

## 6. Query Universe 和异常处理

- evaluated query universe 以 `normalized_dataset.qrels` 中 `relevance > 0` 的 query 为准。
- 有 positive qrels 但缺 retrieval result：空结果计分，并计入 `missing_result_query_count`。
- retrieval result 有 `error`：空结果计分，并计入 `failed_retrieval_query_count`。
- retrieval result 不在 positive qrels 中：忽略，并计入 `ignored_result_query_count`。
- qrels 中没有 positive doc 的 query：跳过，并计入 `skipped_no_positive_qrels_query_count`。

## 7. Manifest

manifest metadata 记录：

- `stage = metrics_run`
- `source_normalized_dataset_artifact_id`
- `source_retrieval_run_artifact_id`
- `k_values`
- `doc_aggregation`
- `doc_score`
- `main_score_metric`
- `main_score`
- `query_count`
- `evaluated_query_count`
- `missing_result_query_count`
- `failed_retrieval_query_count`
- `ignored_result_query_count`
- `skipped_no_positive_qrels_query_count`
- `missing_doc_id_hit_count`
- `duplicate_doc_hit_count`
- `queries_per_shard`
- `query_metric_file_count`
- `query_metric_record_count`

dependencies 记录：

- `normalized_dataset`
- `retrieval_run`

## 8. 范围自检

- 是否重新跑检索：`no`
- 是否访问真实 ES / Milvus / embedding / rewrite / rerank 服务：`no`
- 是否实现真实 retrieval adapter：`no`
- 是否实现完整 benchmark runner：`no`
- 是否实现 CLI / HTTP server：`no`
- 是否提交 `.local_artifacts` 或真实 config / 密钥：`no`
- 是否修改流程控制文档：`no`

## 9. 自检结果

### 9.1 已运行命令

```bash
pytest tests/metrics
pytest tests/metrics tests/retrieval
ruff check .
mypy .
pytest
```

### 9.2 输出摘要

- `pytest tests/metrics`
  - 通过，`13 passed`
- `pytest tests/metrics tests/retrieval`
  - 通过，`32 passed`
- `ruff check .`
  - 通过，`All checks passed!`
- `mypy .`
  - 通过，`Success: no issues found in 128 source files`
- `pytest`
  - 通过，`501 passed`

## 10. 风险与未决项

- 本轮实现内置 IR 公式，没有引入 `pytrec_eval` 依赖；后续可加可选 cross-check。
- 当前 doc projection 只实现 `first_chunk_rank` / `reciprocal_first_chunk_rank`。
- `main_score_metric` 固定为 `ndcg_at_10`。
- 本轮不实现完整 benchmark runner 或 CLI。

## 11. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无

## 12. 提交信息

- 是否已提交：`yes`
- commit subject：`Add metrics run artifact`
- 返修 commit subject：`Record metrics shard size in manifest`
- 验收者确认的最终 commit：由验收者用 `git log -1 --oneline` 确认
