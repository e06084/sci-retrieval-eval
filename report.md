# 开发报告

本文件由开发 session 维护。

验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`chunked_corpus + embeddings 到 Milvus 的可审计入库 artifact`
- 当前分支：`feat/milvus-ingest`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-26
- 完成时间：2026-05-26

## 2. 本次改动

- 新增 `src/eval_platform/indexes/milvus.py`
  - 定义 `MILVUS_COLLECTION_ARTIFACT_TYPE = "milvus_collection"`
  - 定义默认 Milvus schema
  - 定义 `MilvusIngestConfig`
  - 定义 `MilvusClientProtocol`
  - 定义 `PymilvusMilvusClient`
  - `PymilvusMilvusClient` 将内部 dict schema / index params 转换为 pymilvus 需要的 `CollectionSchema` / `IndexParams`
  - 实现 `run_milvus_ingest(...)`
- 更新 `src/eval_platform/indexes/__init__.py`
  - 导出 Milvus ingest 相关公共接口
- 更新 `pyproject.toml`
  - 新增 optional dependency：`milvus = ["pymilvus>=2.4"]`
- 新增 `tests/indexes/test_milvus_ingest.py`
  - 使用 fake Milvus client 覆盖成功路径、失败路径、shard 对齐、manifest、进度回调和流式读取约束
  - 使用 fake `pymilvus` module 覆盖生产 adapter 的 schema / index params 转换逻辑，不依赖真实 pymilvus 服务
- 新增 ADR：
  - `docs/decisions/0016-milvus-collection-artifact.md`
- 更新：
  - `docs/ai/current_status.md`
  - `report.md`

## 3. 涉及文件

- `pyproject.toml`
- `src/eval_platform/indexes/__init__.py`
- `src/eval_platform/indexes/milvus.py`
- `tests/indexes/test_milvus_ingest.py`
- `docs/decisions/0016-milvus-collection-artifact.md`
- `docs/ai/current_status.md`
- `report.md`

### 3.1 范围自检

- 是否改动了流程控制文档：`no`
- 如果是，改动理由：无
- 是否修改 ES ingest 语义：`no`
- 是否实现 retrieval / metrics / frontend：`no`
- 单元测试是否访问真实 Milvus：`no`
- 验收 smoke 是否访问真实 Milvus：`yes`

## 4. 实现说明

### 4.1 入库语义

`run_milvus_ingest(...)` 的输入是一个完整 `chunked_corpus` artifact 和一个完整 `embeddings` artifact，输出是一个 `milvus_collection` artifact manifest。

语义固定为：

1. 如果 collection 不存在，先创建 collection。
2. 如果 collection 已存在且 `overwrite_existing=False`，直接抛 `MilvusIngestError`。
3. 如果 collection 已存在且 `overwrite_existing=True`，先 drop，再 create。
4. 不支持 append 模式。
5. 每个 aligned chunk + embedding 写一个 Milvus row。
6. 主键默认使用 `chunk_id`。
7. 默认 `flush=True`、`verify_count=True`。
8. count 校验不一致时失败，不写 `_SUCCESS`。

### 4.2 shard zip join 与对齐校验

Milvus ingest 路径使用：

```python
iter_chunk_shards(chunk_store, config.chunked_corpus_artifact_id)
iter_embedding_shards(embedding_store, config.embeddings_artifact_id)
```

不会调用：

```python
read_chunked_corpus_artifact(...)
read_embeddings_artifact(...)
```

对齐校验包括：

1. shard 数量必须一致。
2. `chunk_shard.shard_id == embedding_shard.shard_id`。
3. `chunk_shard.path == embedding_shard.source_chunk_file`。
4. `chunk_shard.chunk_count == embedding_shard.embedding_count`。
5. 行级 `chunk_id` 必须一致。
6. 行级 `doc_id` 必须一致。
7. `len(embedding.vector) == vector_dim`。

任一对齐失败都会抛 `MilvusIngestError`，且不会写 `_SUCCESS`。

### 4.3 vector_dim 与 schema

`vector_dim` 解析顺序：

1. 优先使用 `config.vector_dim`。
2. 如果 `config.vector_dim` 未传，则读取 embeddings manifest 的 `embedding_dim`。
3. 如果两者都存在且不一致，直接失败。
4. 如果无法确定维度，直接失败。

默认 schema 显式声明字段，不依赖 Milvus 自动推断。

默认 row 字段包括：

1. `chunk_id`
2. `doc_id`
3. `title`
4. `text`
5. `chunk_index`
6. `start_offset`
7. `end_offset`
8. `metadata`
9. `source_chunked_corpus_artifact_id`
10. `source_embeddings_artifact_id`
11. `source_chunk_file`
12. `source_embedding_file`
13. `shard_id`
14. `vector`

schema 通过稳定 JSON 序列化计算 `schema_sha256`。

### 4.4 Manifest

成功后写出：

```text
milvus_collection/<output_artifact_id>/_MANIFEST.json
milvus_collection/<output_artifact_id>/_SUCCESS
```

这类 artifact 可以没有 payload 文件，因此 `files=[]`。

Manifest dependencies 记录：

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

Manifest metadata 记录：

1. `source_chunked_corpus_artifact_id`
2. `source_embeddings_artifact_id`
3. `collection_name`
4. `primary_key_field`
5. `vector_field`
6. `vector_dim`
7. `metric_type`
8. `batch_size`
9. `overwrite_existing`
10. `flush`
11. `verify_count`
12. `schema_sha256`
13. `schema`
14. `index_params`
15. `alignment_key`
16. `alignment_order`
17. `inserted_count`
18. `failed_count`
19. `verified_entity_count`
20. `shards`

用户传入的 `metadata` 只能作为补充字段，不能覆盖系统字段。

为避免误写密钥，manifest 会过滤用户 metadata 中 secret-looking key：

- `password`
- `token`
- `access_key`
- `api_key`
- `secret`

### 4.5 生产 Milvus client

本轮实现了最小 `PymilvusMilvusClient`：

1. `pymilvus` 仅在初始化该 client 时 lazy import。
2. 依赖放在 optional extra：`milvus = ["pymilvus>=2.4"]`。
3. 单元测试不依赖 `pymilvus`。
4. 本 client 只覆盖本轮 ingest 需要的 API：
   - collection exists
   - create
   - drop
   - insert
   - flush
   - count
5. 真实 Milvus smoke 曾暴露出 dict schema 不能直接传给 pymilvus 的问题；已修复为显式构建 `CollectionSchema` 和 `IndexParams`。

### 4.6 失败路径

以下情况都会抛错，且不会写 `_SUCCESS`：

1. collection 已存在且未启用 overwrite
2. insert item 失败
3. insert 返回成功数量与 batch row 数不一致
4. shard 数量不一致
5. shard id 不一致
6. source chunk file 不一致
7. shard 内 chunk / embedding 行数不一致
8. 行级 `chunk_id` 不一致
9. 行级 `doc_id` 不一致
10. vector 维度不一致
11. count 校验失败
12. progress reporter 抛异常

## 5. 自检结果

### 5.1 必跑命令

```bash
pytest tests/indexes tests/chunking/test_artifact.py tests/embeddings/test_artifact.py
ruff check .
mypy .
pytest
```

### 5.2 输出摘要

- `pytest tests/indexes tests/chunking/test_artifact.py tests/embeddings/test_artifact.py`：
  - 已运行，`87 passed`
- `ruff check .`：
  - 已运行，通过
- `mypy .`：
  - 已运行，通过，`Success: no issues found in 103 source files`
- `pytest`：
  - 通过，`432 passed`

### 5.3 真实入库 smoke

使用已有完整 IFIRNFCorpus artifact：

- `chunked_corpus/ifir_nfcorpus_full_20260526_1945_chunks`
- `embeddings/ifir_nfcorpus_full_20260526_1945_embeddings`

实际写入结果：

- ES index：`ifir_nfcorpus_real_ingest_20260526_220102_es`
- ES artifact：`s3://scibase-service/test_sciverse_benchmark/elasticsearch_index/ifir_nfcorpus_real_ingest_20260526_220102_es_index/`
- ES `indexed_count=11962`
- ES `verified_document_count=11962`
- Milvus collection：`ifir_nfcorpus_real_ingest_20260526_220102_milvus`
- Milvus artifact：`s3://scibase-service/test_sciverse_benchmark/milvus_collection/ifir_nfcorpus_real_ingest_20260526_220102_milvus_collection/`
- Milvus `inserted_count=11962`
- Milvus `verified_entity_count=11962`

## 6. 风险与未决项

- 已知风险：
  - `PymilvusMilvusClient` 是最小 lazy-import adapter，只覆盖本轮 ingest 所需 API。
  - 真实 smoke 已验证 `create_collection`、insert、flush、count 和 manifest 写入，但还没有纳入自动化集成测试。
- 未覆盖场景：
  - 不覆盖正式 corpus build runner。
  - 不覆盖 retrieval / metrics。
- 需要验收者重点检查的点：
  - 是否满足通过 `iter_chunk_shards(...)` / `iter_embedding_shards(...)` 流式消费 artifact。
  - 对齐错误和失败路径是否确实不会写 `_SUCCESS`。
  - manifest 是否足以审计 source chunk artifact、source embedding artifact、collection、schema、index params 和 shard 入库结果。

## 7. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无

## 8. 提交信息

- 是否已提交：`yes`
- commit subject：`Add Milvus ingest artifact`
- 验收者确认的最终 commit：
