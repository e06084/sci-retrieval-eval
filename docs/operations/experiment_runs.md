# Experiment Runs

本文档说明新的实验调用层如何工作。它不替代 corpus asset 构建，而是把已有资产和 benchmark runner 串成可复用、可补全的实验流程。

## 流程

推荐调用顺序：

```text
1. 准备或盘点 corpus assets
2. 构造 ExperimentRunConfig，可以显式传 BenchmarkDatasetSpec，也可以用 corpus_assets 从数据集选择自动解析
3. plan_experiment: 计算 retrieval / metrics / benchmark 的预期 fingerprint 和 reuse/create 决策
4. run_experiment: 只执行缺失的 stage，并写 experiment_run summary
5. 可选写入 artifact_catalog/default/records.jsonl，后续实验优先按 catalog 查找可复用 artifact
```

## Corpus Asset Selection

如果 corpus/index 资产已经由资产准备流程产出，实验配置不需要手写每个数据集的
normalized / ES / Milvus artifact id。可以使用：

```python
ExperimentRunConfig(
    experiment_run_id="e1_e4_bge_m3_fp_20260530_experiment",
    corpus_assets=ExperimentCorpusAssetConfig(
        dataset_selection="all",
        corpus_run_id="e1_e4_bge_m3_fp_20260530",
        bucket="scibase-service",
        raw_prefix="sciverse_benchmark/raw",
        s3_prefix="sciverse_benchmark/assets",
    ),
    settings=settings_for_selection(),
    code_git_sha="<current-git-sha>",
)
```

系统会调用 corpus asset inventory/planner，解析出每个数据集的：

```text
normalized_dataset_artifact_id
elasticsearch_index_artifact_id
milvus_collection_artifact_id
index_name
collection_name
```

`plan_experiment` 返回的 `corpus_asset_plan` 会保留 corpus 资产链的 reuse/create
判断。实验 runner 只消费 corpus assets；如果 normalized / ES / Milvus 缺失，会在
计划阶段明确失败，要求先运行 corpus asset 准备流程。

## Metrics Defaults

默认实验关注浅层召回能力：

```text
metrics_k_values = [5, 10, 20]
main_score_metric = recall_at_10
```

也就是说，默认报告优先比较 `recall_at_5`、`recall_at_10`、`recall_at_20`。如果需要和历史
README baseline 对比 `ndcg10`、`mrr10`、`r100`，或者需要更深的 recall cutoff，必须在
`ExperimentRunConfig.metrics_k_values` 中显式传入对应 cutoff。

## 复用规则

`run_experiment` 会按顺序处理每个 dataset x setting：

```text
retrieval_run -> metrics_run -> benchmark_run -> experiment_run
```

每个 stage 的复用优先级：

```text
1. catalog 中有 complete artifact 且 asset_fingerprint_sha256 匹配，复用。
2. artifact store 中扫描到 complete artifact 且 asset_fingerprint_sha256 匹配，复用。
3. artifact_id 精确命中且 complete，复用。
4. 否则创建。
```

因此，不同 experiment run 之间可以复用已有 retrieval、metrics、benchmark，只要逻辑 fingerprint 一致。

## Catalog

Catalog 是轻量索引，不是唯一真相：

```text
唯一真相: artifact 目录里的 _MANIFEST.json + _SUCCESS
查询加速: artifact_catalog/default/records.jsonl
```

可用能力：

```text
build_artifact_catalog_record
upsert_artifact_catalog_record
read_artifact_catalog
sync_artifact_catalog_from_store
find_catalog_record_by_fingerprint
```

S3 没有原子 append，本实现采用单写者模式下的 read-merge-overwrite。后续如需并发写，可以改成按日期/run_id 分片 catalog。

## 当前边界

本轮实现的是实验层补全：

```text
已有 corpus assets -> 自动解析 benchmark datasets -> retrieval/metrics/benchmark 自动复用或补跑
```

尚未把 corpus asset 缺失时的真实构建执行并入 `run_experiment`。目前 corpus asset 仍由
`corpus_assets` planner 和 `corpus_build` runner 负责。
