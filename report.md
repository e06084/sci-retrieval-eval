# 开发报告

本文件由开发 session 维护。

验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`chunked_corpus 到 Elasticsearch 的可审计入库 artifact`
- 当前分支：`feat/es-ingest`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-26
- 完成时间：2026-05-26

## 2. 本次改动

- 新增 `src/eval_platform/indexes/elasticsearch.py`
  - 定义 `ELASTICSEARCH_INDEX_ARTIFACT_TYPE = "elasticsearch_index"`
  - 定义默认 Elasticsearch mapping
  - 定义 `ElasticsearchIngestConfig`
  - 定义 `ElasticsearchClientProtocol`
  - 定义 `HTTPElasticsearchClient`
  - 实现 `run_elasticsearch_ingest(...)`
- 更新 `src/eval_platform/indexes/__init__.py`
  - 导出 ES ingest 相关公共接口
- 新增 `tests/indexes/test_elasticsearch_ingest.py`
  - 使用 fake ES client 覆盖成功路径、失败路径、manifest、进度回调和流式读取约束
- 新增 ADR：
  - `docs/decisions/0015-elasticsearch-index-artifact.md`
- 更新：
  - `docs/ai/current_status.md`
  - `report.md`

## 3. 涉及文件

- `src/eval_platform/indexes/__init__.py`
- `src/eval_platform/indexes/elasticsearch.py`
- `tests/indexes/__init__.py`
- `tests/indexes/test_elasticsearch_ingest.py`
- `docs/decisions/0015-elasticsearch-index-artifact.md`
- `docs/ai/current_status.md`
- `report.md`

### 3.1 范围自检

- 是否改动了流程控制文档：`no`
- 如果是，改动理由：无
- 是否实现 Milvus：`no`
- 是否实现 retrieval / metrics / frontend：`no`
- 是否访问真实 ES：`no`

## 4. 实现说明

### 4.1 入库语义

`run_elasticsearch_ingest(...)` 的输入是一个完整 `chunked_corpus` artifact，输出是一个 `elasticsearch_index` artifact manifest。

语义固定为：

1. 如果 index 不存在，先创建 index。
2. 如果 index 已存在且 `overwrite_existing=False`，直接抛 `ElasticsearchIngestError`。
3. 如果 index 已存在且 `overwrite_existing=True`，先 delete，再 create。
4. 不支持 append 模式。
5. 每个 chunk 写一个 ES document。
6. ES `_id` 固定使用 `chunk_id`。
7. 默认 `refresh=True`、`verify_count=True`。
8. count 校验不一致时失败，不写 `_SUCCESS`。

### 4.2 流式读取

ES ingest 路径使用：

```python
iter_chunk_shards(source_store, config.source_artifact_id)
```

不会调用：

```python
read_chunked_corpus_artifact(...)
```

测试中通过 monkeypatch full reader 为抛错，验证 ingest 仍可成功。

### 4.3 ES document 格式

每个 chunk 转成稳定 document：

```json
{
  "chunk_id": "...",
  "doc_id": "...",
  "title": "...",
  "text": "...",
  "chunk_index": 0,
  "start_offset": 0,
  "end_offset": 100,
  "metadata": {},
  "source_chunked_corpus_artifact_id": "...",
  "source_chunk_file": "chunks/part-00000.jsonl",
  "shard_id": "part-00000"
}
```

本轮不把 embedding vector 写入 ES。

### 4.4 Manifest

成功后写出：

```text
elasticsearch_index/<output_artifact_id>/_MANIFEST.json
elasticsearch_index/<output_artifact_id>/_SUCCESS
```

这类 artifact 可以没有 payload 文件，因此 `files=[]`。

Manifest dependency 记录：

```json
{
  "artifact_type": "chunked_corpus",
  "artifact_id": "<source_artifact_id>"
}
```

Manifest metadata 记录：

1. `source_chunked_corpus_artifact_id`
2. `index_name`
3. `document_id_field`
4. `bulk_size`
5. `overwrite_existing`
6. `refresh`
7. `verify_count`
8. `mapping_sha256`
9. `mapping`
10. `indexed_count`
11. `failed_count`
12. `verified_document_count`
13. `shards`

用户传入的 `metadata` 只能作为补充字段，不能覆盖系统字段。

为避免误写密钥，manifest 会过滤用户 metadata 中 secret-looking key：

- `password`
- `token`
- `access_key`
- `api_key`
- `secret`

### 4.5 失败路径

以下情况都会抛错，且不会写 `_SUCCESS`：

1. index 已存在且未启用 overwrite
2. bulk item 失败
3. bulk 返回成功数量与 action 数不一致
4. count 校验失败
5. progress reporter 抛异常

## 5. 自检结果

### 5.1 必跑命令

```bash
pytest tests/indexes tests/chunking/test_artifact.py
ruff check .
mypy .
pytest
```

### 5.2 输出摘要

- `pytest tests/indexes tests/chunking/test_artifact.py`：
  - 通过，`37 passed`
- `ruff check .`：
  - 通过
- `mypy .`：
  - 通过，`Success: no issues found in 101 source files`
- `pytest`：
  - 通过，`396 passed`

## 6. 风险与未决项

- 已知风险：
  - 本轮没有真实 ES 集成测试，只使用 fake client 验证主库逻辑。
  - `HTTPElasticsearchClient` 是最小实现，只覆盖本轮 ingest 所需 API。
- 未覆盖场景：
  - 不覆盖正式 corpus build runner。
  - 不覆盖 Milvus ingest。
  - 不覆盖 retrieval / metrics。
- 需要验收者重点检查的点：
  - 是否满足只通过 `iter_chunk_shards(...)` 流式消费 chunked corpus。
  - 失败路径是否确实不会写 `_SUCCESS`。
  - manifest 是否足以审计 source artifact、index、mapping 和 shard 入库结果。

## 7. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无

## 8. 提交信息

- 是否已提交：`yes`
- commit subject：`Add Elasticsearch ingest artifact`
- 验收者确认的最终 commit：
