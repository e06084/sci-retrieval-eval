# Experiment Variants and Asset Reuse

本文档记录后续实验变体规划的资产复用原则。当前实现只提供
`asset_fingerprint` 基础设施，不改变既有 planner 行为，也不接真实服务。

## 1. Identity Boundary

`run_id` 是一次构建或实验运行的操作性身份，可以继续出现在 artifact id、ES index
name、Milvus collection name 中。它不应进入资产 fingerprint，也不应作为判断两个资产
是否逻辑等价的依据。

资产逻辑身份由 `asset_fingerprint` 表示。fingerprint 只包含影响资产内容或行为的稳定
components，例如上游资产 fingerprint、dataset、normalizer、chunker Git commit、
chunk params、embedding model/revision/call params、index schema、retrieval search
params、rerank/rewrite 身份和 metric params。

fingerprint 不包含物理资源名、真实服务地址、运行实例 ID、时间戳、request id、trace
path 或凭证。ES URL、Milvus URI、ES index name、Milvus collection name 应作为 artifact
manifest metadata 或运行配置记录，用来定位外部资源，而不是参与资产等价判断。
`raw_source_uri` 和 `source_git_remote_url` 是例外：它们分别描述 raw 数据快照来源和
chunker 源码身份，属于稳定 identity 字段。`endpoint_alias` 也可进入 fingerprint，
因为它是受控的逻辑服务别名，不是直接访问地址。

自由参数字典必须保持语义参数边界，例如 `builder_params`、`ingest_params`、
`search_params`、`query_embedding`、`rewrite`、`rerank`、`call_params` 中不能夹带
`index_name`、`collection_name`、真实 endpoint URL、host/port、request id 或 trace
path。`raw_dataset.file_fingerprints` 是文件集合语义，计算前会按 canonical order 排序，
因此不会因为对象存储 listing 顺序不同而改变 fingerprint。

## 2. Reuse Checks

复用已有 artifact 时，后续 planner 至少应同时检查：

- artifact complete marker 存在。
- artifact type 与当前 stage 匹配。
- artifact dependency chain 与当前需要的上游链路兼容。
- artifact manifest 中的 `asset_fingerprint` 与当前需要的 fingerprint 匹配。

dependency chain 只证明 lineage，不能单独证明等价。artifact id 只定位物理产物，不能单独
证明等价。

## 3. Minimal Rebuild Examples

只改变 embedding model：

```text
reuse raw_dataset
reuse normalized_dataset
reuse chunked_corpus
reuse elasticsearch_index
rebuild embeddings
rebuild milvus_collection
rerun retrieval_run
rerun metrics_run
rerun benchmark_run / benchmark_suite_run
```

只改变 chunker `git_commit` 或 `chunker_entrypoint`：

```text
reuse raw_dataset
reuse normalized_dataset
rebuild chunked_corpus
rebuild embeddings
rebuild elasticsearch_index
rebuild milvus_collection
rerun retrieval_run
rerun metrics_run
rerun benchmark_run / benchmark_suite_run
```

只改变 chunk params：

```text
reuse raw_dataset
reuse normalized_dataset
rebuild chunked_corpus
rebuild embeddings
rebuild elasticsearch_index
rebuild milvus_collection
rerun retrieval_run
rerun metrics_run
rerun benchmark_run / benchmark_suite_run
```

只改变 ES mapping/settings/builder params：

```text
reuse raw_dataset
reuse normalized_dataset
reuse chunked_corpus
reuse embeddings
rebuild elasticsearch_index
reuse milvus_collection
rerun retrieval_run
rerun metrics_run
rerun benchmark_run / benchmark_suite_run
```

只改变 Milvus schema/metric/index params：

```text
reuse raw_dataset
reuse normalized_dataset
reuse chunked_corpus
reuse embeddings
reuse elasticsearch_index
rebuild milvus_collection
rerun retrieval_run
rerun metrics_run
rerun benchmark_run / benchmark_suite_run
```

只改变 rerank 配置：

```text
reuse raw_dataset
reuse normalized_dataset
reuse chunked_corpus
reuse embeddings
reuse elasticsearch_index
reuse milvus_collection
rerun retrieval_run
rerun metrics_run
rerun benchmark_run / benchmark_suite_run
```

只改变 metric params：

```text
reuse raw_dataset
reuse normalized_dataset
reuse chunked_corpus
reuse embeddings
reuse elasticsearch_index
reuse milvus_collection
reuse retrieval_run
rerun metrics_run
rerun benchmark_run / benchmark_suite_run
```

## 4. Future Planner Inputs

后续 Track B2 / B3 可以在不改变 fingerprint 语义的前提下增加：

- stage override，例如强制重建某些 stage。
- pinned artifact，例如显式指定某个 artifact id，但仍需要校验 type、complete、dependency
  和 fingerprint。
- variant spec，例如同一 dataset 下多组 embedding、chunk、retrieval、rerank 或 metric
  参数矩阵。

这些能力不在当前 B1 范围内实现。
