# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`benchmark_run v1`
- 当前分支：`feat/benchmark-runner-v1`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-27
- 完成时间：2026-05-27

## 2. 本次改动

- 新增 benchmark schema：
  - `BenchmarkRunConfig`
  - `BenchmarkRunSummary`
- 新增 `benchmark_run` artifact IO：
  - `write_benchmark_run_artifact(...)`
  - `read_benchmark_run_artifact(...)`
- 新增 benchmark runner：
  - `run_benchmark(...)`
- 更新公共导出：
  - `src/eval_platform/benchmark/__init__.py`
- 新增 ADR：
  - `docs/decisions/0022-benchmark-run-artifact.md`
- 更新：
  - `docs/ai/current_status.md`
  - `report.md`
- 新增测试：
  - `tests/benchmark/test_artifact.py`
  - `tests/benchmark/test_runner.py`

## 3. Artifact 结构

新增 artifact type：

```text
benchmark_run
```

文件结构：

```text
benchmark_run/<artifact_id>/
  summary.json
  _MANIFEST.json
  _SUCCESS
```

`summary.json` 保存：

- `benchmark_run_artifact_id`
- `setting_name`
- `retrieval_run_artifact_id`
- `metrics_run_artifact_id`
- `source_normalized_dataset_artifact_id`
- `main_score`
- `main_score_metric`
- `aggregate_metrics`

不写入完整 per-query metrics 或 retrieval hits。

## 4. Runner 行为

`run_benchmark(...)` 执行顺序：

1. 调用 `run_retrieval(...)` 产出 `retrieval_run`。
2. 调用 `run_metrics(...)` 产出 `metrics_run`。
3. 读取 metrics artifact 的 aggregate / main score。
4. 写 `benchmark_run` summary / manifest / `_SUCCESS`。

如果 retrieval 或 metrics 任一步失败，不会写出 `benchmark_run/_SUCCESS`。

## 5. Live / Replay 路径

Live retrieval：

- `retrieval.execution_mode="live"`。
- benchmark runner 只透传 ES / Milvus / embedding / rewrite / rerank clients。
- 缺必要 client 的错误沿用 `run_retrieval(...)`。

Replay retrieval：

- `retrieval.execution_mode="replay"`。
- benchmark runner 不要求 ES / Milvus / embedding clients。
- replay 由 `run_retrieval(...)` 复制已有 retrieval run。
- 然后 `run_metrics(...)` 只消费 replay 后的新 retrieval artifact。

metrics 永远只消费 `normalized_dataset + retrieval_run`，不会重新跑检索。

## 6. Config 校验

`BenchmarkRunConfig` 校验：

- `output_artifact_id` 非空。
- `source_normalized_dataset_artifact_id` 非空。
- `retrieval.source_normalized_dataset_artifact_id` 必须匹配 benchmark source。
- `metrics.source_normalized_dataset_artifact_id` 必须匹配 benchmark source。
- `metrics.source_retrieval_run_artifact_id` 必须等于 `retrieval.output_artifact_id`。
- `setting_name` 如提供必须非空。
- `tags` 去空白，并按输入顺序去重。

## 7. Manifest

manifest metadata 记录：

- `stage = benchmark_run`
- `source_normalized_dataset_artifact_id`
- `retrieval_run_artifact_id`
- `metrics_run_artifact_id`
- `setting_name`
- `description`
- `tags`
- `retrieval_mode`
- `retrieval_execution_mode`
- `retrieval_trace_mode`
- `top_k`
- `sub_queries`
- `rewrite_enabled`
- `rerank_enabled`
- `metrics_k_values`
- `doc_aggregation`
- `doc_score`
- `main_score_metric`
- `main_score`
- `retrieval_failed_query_count`
- `metrics_evaluated_query_count`

dependencies 记录：

- `normalized_dataset`
- `retrieval_run`
- `metrics_run`

## 8. 范围自检

- 是否实现 CLI：`no`
- 是否实现 HTTP server：`no`
- 是否访问真实 ES / Milvus / embedding / rewrite / rerank 服务：`no`
- 是否实现 rewrite / rerank adapter：`no`
- 是否修改 corpus build 主链路：`no`
- 是否修改 metrics 公式：`no`
- 是否提交 `.local_artifacts` 或真实 config / 密钥：`no`
- 是否修改流程控制文档：`no`

## 9. 自检结果

### 9.1 已运行命令

```bash
pytest tests/benchmark
pytest tests/benchmark tests/retrieval tests/metrics
ruff check .
mypy .
pytest
```

### 9.2 输出摘要

- `pytest tests/benchmark`
  - 通过，`10 passed`
- `pytest tests/benchmark tests/retrieval tests/metrics`
  - 通过，`54 passed`
- `ruff check .`
  - 通过，`All checks passed!`
- `mypy .`
  - 通过，`Success: no issues found in 139 source files`
- `pytest`
  - 通过，`523 passed`

## 10. 风险与未决项

- 本轮是最小 Python runner，不实现 CLI / batch scheduler。
- 本轮不做真实 S3 / ES / Milvus connectivity smoke。
- 本轮不实现多 setting 批量执行和报告生成。
- benchmark manifest 不展开 ES/Milvus index artifact 依赖；这些仍保留在 retrieval manifest。

## 11. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无

## 12. 提交信息

- 是否已提交：`yes`
- commit subject：`Add benchmark run artifact`
- 验收者确认的最终 commit：由验收者用 `git log -1 --oneline` 确认
