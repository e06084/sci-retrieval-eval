# 0016. Milvus Collection Artifact

- Status: Accepted
- Date: 2026-05-26

## Context

当前主线已经支持：

- sharded `chunked_corpus` artifact
- 与 source chunk shard 对齐的 sharded `embeddings` artifact
- `iter_chunk_shards(...)`
- `iter_embedding_shards(...)`
- Elasticsearch ingest artifact

Milvus 入库需要同时消费 chunk text 和 embedding vector。为了避免大 corpus 下全量加载 `chunked_corpus` 与 `embeddings`，Milvus ingest 必须按 shard 流式 zip join，并校验 shard 级和行级对齐关系。

## Decision

新增 `milvus_collection` artifact 类型。

成功入库后的输出形态：

```text
milvus_collection/<artifact_id>/
  _MANIFEST.json
  _SUCCESS
```

该 artifact 可以没有 payload 文件，`files` 允许为空。

Milvus 入库 runner 使用：

```python
run_milvus_ingest(chunk_store, embedding_store, output_store, config, client)
```

设计约束：

- 输入 chunk 只通过 `iter_chunk_shards(...)` 流式读取。
- 输入 embedding 只通过 `iter_embedding_shards(...)` 流式读取。
- 不调用 `read_chunked_corpus_artifact(...)` 或 `read_embeddings_artifact(...)`。
- shard 数量、`shard_id`、`source_chunk_file`、行数必须一致。
- 每一行必须校验 `chunk_id`、`doc_id` 和 vector 维度。
- Milvus 主键固定默认使用 `chunk_id`。
- 默认 schema 显式声明字段，不依赖 Milvus 自动推断。
- schema 使用稳定 JSON 序列化计算 `schema_sha256`。
- collection 已存在且 `overwrite_existing=False` 时直接失败。
- collection 已存在且 `overwrite_existing=True` 时先 drop 再 create。
- 不支持 append 模式。
- insert 失败、对齐失败、count 校验失败、progress reporter 失败时都不写 `_SUCCESS`。

Manifest dependencies 记录两个 upstream artifact：

```json
[
  {
    "artifact_type": "chunked_corpus",
    "artifact_id": "<chunked_corpus_artifact_id>"
  },
  {
    "artifact_type": "embeddings",
    "artifact_id": "<embeddings_artifact_id>"
  }
]
```

Manifest metadata 至少记录：

- `source_chunked_corpus_artifact_id`
- `source_embeddings_artifact_id`
- `collection_name`
- `primary_key_field`
- `vector_field`
- `vector_dim`
- `metric_type`
- `batch_size`
- `overwrite_existing`
- `flush`
- `verify_count`
- `schema_sha256`
- `index_params`
- `alignment_key`
- `alignment_order`
- `inserted_count`
- `failed_count`
- `verified_entity_count`
- `shards`

用户传入的 supplemental metadata 只能补充字段，不能覆盖系统字段。Secret-looking metadata key 会被过滤，避免 manifest 写入 password / token / access key。

## Consequences

正面影响：

- Milvus 入库结果可通过 artifact manifest 审计。
- 后续 corpus build runner 可以复用同一 runner。
- 下游能明确知道哪份 chunk artifact 与哪份 embedding artifact 入了哪个 collection。
- shard zip join 降低大 corpus 下的内存峰值。
- 失败路径不会产生伪成功 artifact。

代价：

- 本阶段只用 fake client 做单元测试，不做真实 Milvus 集成测试。
- `PymilvusMilvusClient` 是最小 lazy-import adapter，仅覆盖本阶段 ingest 所需 API。
- Milvus collection 本体在外部系统中，artifact 只记录可审计元数据。

非目标：

- 不实现 Elasticsearch ingest 变更。
- 不实现 retrieval。
- 不实现 metrics。
- 不实现正式 CLI runner。
