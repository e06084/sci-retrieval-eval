# 0015. Elasticsearch Index Artifact

- Status: Accepted
- Date: 2026-05-26

## Context

当前主线已经能产出 `chunked_corpus` artifact，并且 `chunked_corpus` 已支持 shard-aware 布局与 `iter_chunk_shards(...)` 流式读取。

后续 corpus 构建需要把 chunk text 入库到 Elasticsearch，供关键词检索或混合检索使用。但本阶段不实现 retrieval，也不接真实 ES 服务做集成测试。

本阶段需要固定的是：

- 从哪个 `chunked_corpus` artifact 入库
- 入库到哪个 ES index
- 使用什么 mapping
- 写入多少 chunk document
- 哪些 shard 被处理
- 失败时不产生成功 artifact

## Decision

新增 `elasticsearch_index` artifact 类型。

成功入库后的输出形态：

```text
elasticsearch_index/<artifact_id>/
  _MANIFEST.json
  _SUCCESS
```

该 artifact 可以没有 payload 文件，`files` 允许为空。

ES 入库 runner 使用：

```python
run_elasticsearch_ingest(source_store, output_store, config, client, progress_reporter=None)
```

设计约束：

- 输入只通过 `iter_chunk_shards(...)` 流式读取 `chunked_corpus`。
- 不调用 `read_chunked_corpus_artifact(...)`。
- ES `_id` 固定使用 `chunk_id`。
- 每个 chunk 写入一个 ES document。
- 默认 mapping 显式声明字段，不依赖 dynamic mapping。
- mapping 使用稳定 JSON 序列化计算 `mapping_sha256`。
- index 已存在且 `overwrite_existing=False` 时直接失败。
- index 已存在且 `overwrite_existing=True` 时先 delete 再 create。
- 不支持 append 模式。
- bulk item 失败、count 校验失败、progress reporter 失败时都不写 `_SUCCESS`。

Manifest dependency 记录 source `chunked_corpus`：

```json
{
  "artifact_type": "chunked_corpus",
  "artifact_id": "<source_artifact_id>"
}
```

Manifest metadata 至少记录：

- `source_chunked_corpus_artifact_id`
- `index_name`
- `document_id_field`
- `bulk_size`
- `overwrite_existing`
- `refresh`
- `verify_count`
- `mapping_sha256`
- `indexed_count`
- `failed_count`
- `verified_document_count`
- `shards`

用户传入的 supplemental metadata 只能补充字段，不能覆盖系统字段。Secret-looking metadata key 会被过滤，避免 manifest 写入 password / token / access key。

## Consequences

正面影响：

- ES 入库结果可通过 artifact manifest 审计。
- 后续正式 corpus build runner 可以复用同一 runner。
- 下游能明确知道哪个 chunk artifact 入了哪个 ES index。
- 失败路径不会产生伪成功 artifact。

代价：

- 当前只提供最小 HTTP ES client 和 fake-client 单元测试，不做真实 ES 集成测试。
- ES index 本体在外部系统中，artifact 只记录可审计元数据。

非目标：

- 不实现 Milvus ingest。
- 不实现 retrieval。
- 不实现 metrics。
- 不实现正式 CLI runner。
