# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`默认实验指标关注 recall@5/10/20`
- 当前分支：`feat/default-recall-at-5-10-20`
- 基线：`origin/main`
- 完成时间：2026-06-03

## 2. 本轮实现

本轮把后续实验的默认 metric 关注点改为浅层 recall：

- 新增共享默认值：
  - `DEFAULT_METRICS_K_VALUES = (5, 10, 20)`
  - `DEFAULT_MAIN_SCORE_METRIC = "recall_at_10"`
  - `default_metrics_k_values()`
- `MetricsRunConfig.k_values` 默认改为 `[5, 10, 20]`。
- `MetricsRunData.main_score_metric` 默认改为 `recall_at_10`。
- `BenchmarkSuiteRunConfig.metrics_k_values` 默认改为 `[5, 10, 20]`。
- `ExperimentRunConfig.metrics_k_values` 默认改为 `[5, 10, 20]`。
- experiment planner 构造 metrics fingerprint 时使用共享 `DEFAULT_MAIN_SCORE_METRIC`，避免 plan/reuse 和真实 metrics run 的 fingerprint 不一致。

## 3. 行为语义

- 默认实验优先比较：
  - `recall_at_5`
  - `recall_at_10`
  - `recall_at_20`
- 默认主指标是 `recall_at_10`。
- 显式传入 `k_values` / `metrics_k_values` 的旧实验仍保持可覆盖能力。
- 如果要复现历史 README 表格中的 `ndcg10`、`mrr10`、`r100`，需要显式传入对应 cutoff。
- 本轮不改变 retrieval top_k、Milvus 默认索引、rerank 参数、trace 策略或 metric 公式。

## 4. 测试覆盖

新增/更新测试覆盖：

- `MetricsRunConfig` 默认 `k_values == [5, 10, 20]`。
- 默认 metrics run 输出 `recall_at_5/10/20`，且 `main_score == aggregate["recall_at_10"]`。
- `BenchmarkSuiteRunConfig` 默认 `metrics_k_values == [5, 10, 20]`。
- `ExperimentRunConfig` 默认 `metrics_k_values == [5, 10, 20]`。
- benchmark / suite summary 的默认 `main_score_metric` 更新为 `recall_at_10`。

## 5. 验证结果

已运行：

```bash
env PYTHONPATH=src pytest tests/metrics tests/benchmark tests/experiments tests/assets
env PYTHONPATH=src pytest
ruff check .
mypy .
```

结果：

- `tests/metrics tests/benchmark tests/experiments tests/assets`: `126 passed`
- 全量测试：`714 passed`
- `ruff check .`: `All checks passed!`
- `mypy .`: `Success: no issues found in 188 source files`

## 6. 未实现项

- 未实现报告比较视图或图表层的 recall@5/10/20 专门展示。
- 未移除历史结果中的 `ndcg10`、`mrr10`、`r100` 表格；这些仍用于历史 baseline 对齐。
