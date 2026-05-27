# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`http rerank adapter`
- 当前分支：`feat/http-rerank-adapter`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-27
- 完成时间：2026-05-27

## 2. 本次改动

- 新增正式 HTTP rerank adapter：
  - `HTTPRerankClientConfig`
  - `HTTPRerankClient`
  - `RerankAdapterError`
  - `rerank_client_from_config(...)`
- 新增 rerank endpoint 一致性检查：
  - `RerankConsistencyCheckResult`
  - `run_rerank_consistency_check(...)`
- 更新公共导出：
  - `src/eval_platform/retrieval/__init__.py`
- 新增测试：
  - `tests/retrieval/test_rerank_adapter.py`
- 更新 runner 集成测试：
  - `tests/retrieval/test_runner.py`
- 更新：
  - `report.md`

## 3. 为什么固定单 endpoint

实测显示多个 rerank endpoint 都可用，但不同模型 / endpoint 之间排序不一定一致：

- 3886 与 3887 排序一致。
- 3885、3888 与 3886/3887 排序不一致。

如果默认轮询多个 endpoint，同一组 retrieval candidates 可能因为 endpoint 选择不同而得到不同排序，导致评测不可复现。

因此本轮实现固定单 endpoint：

- `rerank_client_from_config(config, endpoint_index=0)` 只选择一个 endpoint。
- 默认不轮询。
- 多 endpoint 使用前需要显式调用 `run_rerank_consistency_check(...)`。

## 4. Adapter 请求 / 响应格式

请求使用 JSON POST：

```json
{
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "query text",
  "documents": ["doc text 1", "doc text 2"],
  "top_n": 2,
  "return_documents": false
}
```

行为：

- `model_name=None` 时不发送 `model` 字段。
- 有 `api_key` 时发送 `Authorization: Bearer <api_key>`。
- 空文本 hit 使用单空格 `" "`。
- 请求 `top_n` 发送 `len(hits)`，本地再截断到调用入参 `top_n`。

支持响应格式：

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.9}
  ]
}
```

以及：

```json
{
  "data": [
    {"document_index": 0, "score": 0.9}
  ]
}
```

解析行为：

- 按 rerank score 降序、index 升序排序。
- 跳过越界 index、重复 index、缺 score、非有限 score。
- 返回 hit 保留原 `chunk_id/doc_id/title/text/metadata/origin_*` 字段。
- 返回 hit 的 `score` 替换为 rerank score。
- adapter 不重新设置 rank，最终 rank 仍由 `run_retrieval(...)` 统一设置。

## 5. 一致性检查函数

`run_rerank_consistency_check(...)`：

- 不读 config，不读环境变量。
- 接收已构造好的 `RerankClient` 列表和 `endpoint_ids`。
- 用 synthetic `RetrievalHit` 调用每个 client。
- 比较输出 `chunk_id` 顺序。
- 排序一致时 `passed=True`。
- 任一 client 报错或排序不一致时 `passed=False`，并记录 `failure_reason` 和各 endpoint ranking。

该检查不是 `run_retrieval(...)` 的默认前置步骤，只供真实实验脚本 / 验收显式调用。

## 6. 测试覆盖

新增 / 更新覆盖：

- config 校验：
  - `endpoint_url` 非空。
  - `endpoint_id/model_name` 如提供必须非空。
  - `timeout_seconds > 0`。
  - `max_retries >= 0`。
- 请求构造：
  - POST 到 endpoint URL。
  - `Content-Type: application/json`。
  - 有 api key 时带 Authorization。
  - payload 包含 `query/documents/top_n/return_documents=false`。
  - 空文本 hit 使用 `" "`。
- 响应解析：
  - 支持 `results/index/relevance_score`。
  - 支持 `data/document_index/score`。
  - 按 score 降序、index 升序排序。
  - 跳过越界、重复、缺字段、非有限 score row。
  - 本地截断到调用入参 `top_n`。
- hit 保真：
  - 保留 `chunk_id/doc_id/title/text/metadata/origin scores`。
  - score 替换为 rerank score。
  - adapter 不设置新 rank。
- 错误处理：
  - 空 hits 或 `top_n <= 0` 不发请求。
  - HTTP error / invalid JSON / empty parsed results 抛 `RerankAdapterError`。
  - 错误信息不泄漏 api key。
- config factory：
  - 固定选择一个 endpoint。
  - endpoint index 越界报错。
- consistency check：
  - 排序一致通过。
  - 排序不一致失败并记录 rankings。
  - client 报错失败。
- runner 集成：
  - `run_retrieval(..., rerank_enabled=True)` 可注入 `HTTPRerankClient` fake transport 跑通。

## 7. 自检结果

### 7.1 已运行命令

```bash
pytest tests/retrieval/test_rerank_adapter.py
pytest tests/retrieval
pytest tests/benchmark tests/retrieval tests/metrics
ruff check .
mypy .
pytest
```

### 7.2 输出摘要

- `pytest tests/retrieval/test_rerank_adapter.py`
  - 通过，`18 passed`
- `pytest tests/retrieval`
  - 通过，`51 passed`
- `pytest tests/benchmark tests/retrieval tests/metrics`
  - 通过，`74 passed`
- `ruff check .`
  - 通过，`All checks passed!`
- `mypy .`
  - 通过，`Success: no issues found in 141 source files`
- `pytest`
  - 通过，`543 passed`

## 8. 范围自检

- 是否开发 `benchmark_suite`：`no`
- 是否跑真实 4×5 实验：`no`
- 是否实现 rewrite adapter：`no`
- 是否修改 retrieval ranking / fusion / metrics 公式：`no`
- 是否修改 corpus build 主链路：`no`
- 是否访问真实 S3 / ES / Milvus / embedding / rerank：`no`
- 是否提交 `.local_artifacts` 或真实 config / 密钥：`no`
- 是否修改流程控制文档：`no`

## 9. 风险与未决项

- 本轮只提供 HTTP rerank adapter 和 fake transport 单测，未访问真实 rerank endpoint。
- 多 endpoint 一致性检查已提供，但不会在 `run_retrieval(...)` 中自动执行。
- 真实 E4 评测仍应固定 endpoint 为已验证一致的 3886 或 3887。
- rewrite adapter、benchmark suite 和真实 4×5 实验仍是后续任务。

## 10. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无

## 11. 提交信息

- 是否已提交：`yes`
- commit subject：`Add HTTP rerank adapter`
- 验收者确认的最终 commit：由验收者用 `git log -1 --oneline` 确认
