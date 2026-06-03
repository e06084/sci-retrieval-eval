# E1-E4 Benchmark Suite Runbook

本文档说明如何从已完成的五数据集 corpus/index artifacts 进入 E1-E4 benchmark suite smoke 和 full baseline。本文档只是 runbook，不新增 CLI，不包含真实密钥或真实 `config.yaml` 内容。

## 1. 输入 Artifacts

每个 dataset 必须显式提供一组 asset spec：

```text
dataset_key
task_name
normalized_dataset_artifact_id
elasticsearch_index_artifact_id
milvus_collection_artifact_id
index_name
collection_name
```

这些输入必须来自同一条 artifact chain，不能混用不同 run 或不同版本的 artifact：

```text
normalized_dataset -> chunked_corpus -> embeddings -> elasticsearch_index / milvus_collection
```

最低检查：

- `normalized_dataset_artifact_id` 对应的 normalized artifact 已 complete。
- `elasticsearch_index_artifact_id` 的 manifest dependency 指向同一条 chain 的 `chunked_corpus`。
- `milvus_collection_artifact_id` 的 manifest dependencies 指向同一条 chain 的 `chunked_corpus` 和 `embeddings`。
- `index_name` 必须来自 Elasticsearch index artifact manifest metadata。
- `collection_name` 必须来自 Milvus collection artifact manifest metadata。

不要用新 run_id 猜测真实 ES index 或 Milvus collection 名。复用现有 ES/Milvus 资源时，资源名必须来自 reused artifact manifest。

## 2. E1-E4 Settings

默认 E1-E4 baseline settings：

```text
E1-milvus: milvus + ES enrich, no rewrite, no rerank
E2-es: ES BM25, no rewrite, no rerank
E3-hybrid: Milvus + ES + RRF, no rewrite, no rerank
E4-hybrid-rerank: Milvus + ES + RRF + rerank, no rewrite
```

Baseline 必须保持：

```text
top_k = 100
hybrid_per_source_topk = 50
rrf_path_topk = 25
rerank_cross_path_topk = 50
rerank_candidate_cap = 0
metrics_k_values = [5, 10, 20]
main_score_metric = recall_at_10
trace_mode = replay
```

不要在 baseline 中关闭 trace。trace 是后续 replay、排查失败 query、比较 setting 行为的基础审计数据。

默认实验报告应优先看 `recall_at_5`、`recall_at_10`、`recall_at_20`。如果需要和历史 README
baseline 对比 `ndcg10`、`mrr10` 或 `r100`，应在实验配置中显式传入对应 `metrics_k_values`。

Milvus collection 建议使用 Sciverse benchmark v1 默认协议：

```text
index_type = HNSW
metric_type = COSINE
index params = {"M": 16, "efConstruction": 200}
search params = {"metric_type": "COSINE", "params": {}}
primary key = chunk_id
vector field = vector
title.max_length = 65535
```

`vector_dim` 不应写成固定默认值；它必须来自 embedding artifact manifest 或显式配置。

## 3. 推荐执行顺序

建议按以下顺序推进，逐步扩大运行规模：

1. `inventory` / dry-run 确认五数据集 corpus/index artifacts。
2. 1 dataset x E1-E4 x `query_limit=3` smoke。
3. 5 datasets x E1-E4 x `query_limit=3` smoke。
4. 5 datasets x E1-E4 x `query_limit=50` 稳定性验证。
5. 5 datasets x E1-E4 full baseline，此时 `query_limit=None`。

`query_limit=3` 或 `query_limit=5` 用于快速链路 smoke；`query_limit=50` 用于中等规模稳定性验证；`query_limit=None` 表示全量运行。

## 4. 输出检查

每轮运行后检查：

```text
retrieval_run complete
metrics_run complete
benchmark_run complete
benchmark_suite_run complete
suite item_count = dataset_count x setting_count
metrics 有 main_score
retrieval_run 有 trace
suite dependencies 指向 child benchmark_run
```

建议逐项确认：

- 每个 suite item 都有 `benchmark_run_artifact_id`、`retrieval_run_artifact_id`、`metrics_run_artifact_id`。
- 每个 child `benchmark_run` manifest dependencies 指向对应 `retrieval_run` 和 `metrics_run`。
- `benchmark_suite_run` manifest dependencies 指向所有 child `benchmark_run`。
- `benchmark_suite_run` manifest metadata 记录 `dataset_count`、`setting_count`、`item_count` 和本轮 `query_limit`。
- `metrics_run` summary 中 `main_score_metric` 和 `main_score` 非空。
- `retrieval_run` record 在 baseline 中保留 replay trace。

## 5. 进度输出

真实运行脚本必须显式传入 `progress_reporter`，建议按 JSONL 输出到 stdout 或 stderr，避免长时间看不到状态：

```python
def progress_reporter(event):
    print(event.model_dump_json(), flush=True)

run_benchmark_suite(..., progress_reporter=progress_reporter)
```

当前会输出：

- `benchmark_suite_run`: suite 开始、每个 dataset x setting item 开始/完成。
- `benchmark_run`: retrieval 完成、metrics 完成、benchmark artifact 写入前。
- `retrieval_run`: query 级处理进度和失败 query 计数。
- `metrics_run`: query 级指标计算进度、missing/failed query 计数。

## 6. 本轮未实现内容

本文档是 runbook，不新增 CLI。真实运行仍需后续单独执行。

本轮不实现：

- E5/E6。
- rewrite setting。
- variant spec。
- query analysis。
- comparison report。
- 并发调度。
- 真实五数据集运行。
