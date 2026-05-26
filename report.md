# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`live retrieval adapters`
- 当前分支：`feat/live-retrieval-adapters`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-27
- 完成时间：2026-05-27

## 2. 本次改动

- 新增 Elasticsearch retrieval adapter：
  - `HTTPElasticsearchRetrievalClientConfig`
  - `HTTPElasticsearchRetrievalClient`
  - `ElasticsearchRetrievalAdapterError`
  - `elasticsearch_retrieval_client_from_config(...)`
- 新增 Milvus retrieval adapter：
  - `PymilvusRetrievalClientConfig`
  - `PymilvusRetrievalClient`
  - `MilvusRetrievalAdapterError`
  - `milvus_retrieval_client_from_config(...)`
- 更新公共导出：
  - `src/eval_platform/retrieval/__init__.py`
- 新增 ADR：
  - `docs/decisions/0021-live-retrieval-adapters.md`
- 更新：
  - `docs/ai/current_status.md`
  - `report.md`
- 新增测试：
  - `tests/retrieval/test_elasticsearch_adapter.py`
  - `tests/retrieval/test_milvus_adapter.py`

## 3. Elasticsearch Adapter

`search_bm25(index_name, query, top_k)` 使用标准库 HTTP，测试通过 fake transport 注入。

请求路径：

```text
POST /<index_name>/_search
```

请求 body：

```json
{
  "size": 10,
  "query": {
    "multi_match": {
      "query": "...",
      "fields": ["title^1.5", "text"]
    }
  },
  "sort": [
    {"_score": {"order": "desc"}},
    {"chunk_id": {"order": "asc"}}
  ],
  "_source": [
    "chunk_id",
    "doc_id",
    "title",
    "text",
    "chunk_index",
    "start_offset",
    "end_offset",
    "metadata"
  ]
}
```

返回解析：

- `chunk_id` 优先用 `_source.chunk_id`，缺失时 fallback 到 `_id`。
- `doc_id` / `title` / `text` 来自 `_source`。
- `score` 和 `origin_es_score` 来自 `_score`。
- `recall_source = "es"`。
- `metadata` 保留 `_source.metadata` 以及 `chunk_index` / `start_offset` / `end_offset`。

`enrich_by_chunk_ids(...)`：

- 使用 `POST /<index_name>/_mget`。
- 保持输入 hits 顺序。
- 保留原 hit 的 `score`、`recall_source`、`origin_es_score`、`origin_milvus_score`。
- 补齐 `doc_id` / `title` / `text` / `metadata`。
- 缺失 chunk 不重排，保留原 hit，并设置 `metadata["enrich_missing"] = True`。

错误处理：

- HTTP 非 2xx 抛 `ElasticsearchRetrievalAdapterError`。
- invalid JSON 抛 `ElasticsearchRetrievalAdapterError`。
- 错误信息不包含 password / Authorization header。

## 4. Milvus Adapter

`PymilvusRetrievalClient` lazy-import `pymilvus.MilvusClient`。

构造行为：

- 测试可传入 fake `client`，不会 import 或访问真实 Milvus。
- 未安装 `pymilvus` 且未传入 client 时，抛 `MilvusRetrievalAdapterError`，提示安装 `milvus` extra。

`search(collection_name, vector, top_k)` 调用：

```python
client.search(
    collection_name=collection_name,
    data=[list(vector)],
    anns_field=vector_field,
    limit=top_k,
    output_fields=output_fields,
    search_params={"metric_type": metric_type, "params": search_params},
)
```

返回解析：

- 支持 pymilvus 常见 `list[list[dict]]` 结构。
- `chunk_id` 来自 `entity[primary_key_field]`，缺失时 fallback 到 hit `id`。
- `score` 和 `origin_milvus_score` 来自 `distance` 或 `score`。
- `doc_id` / `title` / `text` / `metadata` 来自 `entity`。
- `recall_source = "milvus"`。
- 不把 `vector` 或配置的 `vector_field` 写入 `RetrievalHit.metadata`。

## 5. Config Factory

新增：

```python
elasticsearch_retrieval_client_from_config(config: ElasticsearchConfig)
milvus_retrieval_client_from_config(config: MilvusConfig)
```

行为：

- `elasticsearch.url` 缺失时报明确错误。
- `milvus.address` 缺失时报明确错误。
- 不读取环境变量。
- 不把 secret 写入 report / manifest / error message。

## 6. 测试策略

- Elasticsearch adapter 测试全部使用 fake HTTP transport。
- Milvus adapter 测试全部使用 fake Milvus client 或 monkeypatch import。
- 集成级测试把真实 adapter 类型和 fake transport/client 注入 `run_retrieval(...)`。
- 未访问真实 ES / Milvus / embedding / rewrite / rerank 服务。

## 7. 范围自检

- 是否实现 benchmark runner：`no`
- 是否修改 metrics 逻辑：`no`
- 是否实现 HTTP rewrite / rerank adapter：`no`
- 是否实现 CLI / HTTP server：`no`
- 是否访问真实外部服务：`no`
- 是否提交 `.local_artifacts` 或真实 config / 密钥：`no`
- 是否修改流程控制文档：`no`

## 8. 自检结果

### 8.1 已运行命令

```bash
pytest tests/retrieval
ruff check .
mypy .
pytest
```

### 8.2 输出摘要

- `pytest tests/retrieval`
  - 通过，`31 passed`
- `ruff check .`
  - 通过，`All checks passed!`
- `mypy .`
  - 通过，`Success: no issues found in 132 source files`
- `pytest`
  - 通过，`513 passed`

## 9. 风险与未决项

- 本轮没有真实 ES / Milvus connectivity smoke，真实环境连通性仍需后续由验收或 smoke 任务执行。
- 本轮不实现 HTTP rewrite / rerank adapter。
- 本轮不实现 benchmark runner / CLI。
- Milvus 返回结构在不同 pymilvus 版本可能有差异；当前覆盖常见 `list[list[dict]]` 形态。

## 10. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无

## 11. 提交信息

- 是否已提交：`yes`
- commit subject：`Add live retrieval adapters`
- 验收者确认的最终 commit：由验收者用 `git log -1 --oneline` 确认
