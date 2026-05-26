# 0017. Corpus Build Runner V1

- Status: Accepted
- Date: 2026-05-26

## Context

主线已经具备单阶段 artifact 能力：

- `raw_dataset`
- `normalized_dataset`
- `chunked_corpus`
- `embeddings`
- `elasticsearch_index`
- `milvus_collection`

这些能力可以独立测试和复用，但完整 corpus 构建仍主要依赖 one-off 拼接脚本。团队需要一个主库内的正式 Python runner，用来固定阶段顺序、artifact id、依赖关系、进度事件和最终 run-level manifest。

本阶段只支持 IFIRNFCorpus，因为当前主库 raw normalizer 只正式支持该数据集。

## Decision

新增 `corpus_build` artifact 类型。

成功输出：

```text
corpus_build/<run_id>/
  _MANIFEST.json
  _SUCCESS
```

该 artifact 可以没有 payload 文件，`files=[]`。

新增 runner：

```python
run_corpus_build(...)
```

固定阶段顺序：

1. raw import
2. raw to normalized
3. chunking
4. embedding
5. Elasticsearch ingest，可关闭
6. Milvus ingest，可关闭
7. write `corpus_build` manifest

设计约束：

- runner 不读取真实 `config.yaml`。
- runner 不创建真实 S3 / ES / Milvus / embedding client。
- 调用方必须注入 raw opener、chunker、embedding client、ES client、Milvus client。
- stage config 的 artifact id 必须与 runner 使用的 artifact id 一致，否则拒绝运行。
- 阶段之间只通过 artifact id 传递，不绕过 artifact 直接传内存对象。
- 任一阶段失败时不写最终 `corpus_build/_SUCCESS`。
- 已成功写出的 stage artifact 不回滚。

默认 artifact id：

```text
raw_dataset:         <run_id>_raw
normalized_dataset:  <run_id>_normalized
chunked_corpus:      <run_id>_chunks
embeddings:          <run_id>_embeddings
elasticsearch_index: <run_id>_es_index
milvus_collection:   <run_id>_milvus_collection
corpus_build:        <run_id>
```

Final manifest dependencies 包含所有成功且启用的 stage artifact。

Final manifest metadata 至少记录：

- `run_id`
- `dataset_name`
- `raw_source`
- `artifact_ids`
- `enabled_stages`
- `stage_manifests`

`stage_manifests` 只保留审计必要摘要，例如计数、fingerprint、schema / mapping hash、模型名、index / collection 名，不嵌入完整平台配置或连接密钥。

## Consequences

正面影响：

- 完整 corpus build 可以由主库 API 复现。
- 每次构建有 run-level artifact 可审计。
- 后续 CLI / scheduler 可以直接复用该 runner。
- 单元测试可用 fake dependency 覆盖完整链路，不访问真实外部服务。

代价：

- V1 只支持 IFIRNFCorpus。
- V1 不实现 DAG / resume / retry。
- V1 不迁移或删除 one-off scripts。

非目标：

- 不实现 retrieval。
- 不实现 metrics。
- 不扩展五个数据集 normalizer。
- 不做真实外部服务测试。
