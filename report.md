# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`retrieval runner internal split`
- 当前分支：`feat/retrieval-runner-split`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-27
- 完成时间：2026-05-27
- 实现提交 SHA：`d1bb287b3e90096db6cb824d99e140856f05eaac`
- 报告提交 SHA：本报告单独提交后由 `git log -1 --oneline` 确认；提交内容无法自引用自身 SHA。

## 2. 新增模块和职责

新增 retrieval 内部模块：

- `src/eval_platform/retrieval/query_paths.py`
  - `resolve_query_paths`
  - `dedupe_queries`
  - `embed_query_paths`
- `src/eval_platform/retrieval/recall.py`
  - `recall_one`
  - ES / Milvus / hybrid recall path
  - RRF fusion 调用
- `src/eval_platform/retrieval/rerank_flow.py`
  - `maybe_rerank`
  - `rank_hits`
  - 说明：已有 `retrieval/rerank.py` 是 HTTP rerank adapter，本轮没有混改 adapter，因此流程 helper 使用 `rerank_flow.py`。
- `src/eval_platform/retrieval/trace.py`
  - `new_live_trace`
  - `append_recall_trace`
  - `hits_trace`
  - `build_error_trace`
- `src/eval_platform/retrieval/replay.py`
  - `run_retrieval_replay`
- `src/eval_platform/retrieval/errors.py`
  - `RetrievalRunError`
  - `runner.py` 继续导入并保留该 public symbol。

## 3. runner.py 保留职责

`src/eval_platform/retrieval/runner.py` 仍保留：

- `RetrievalRunConfig`
- `RetrievalRunError` public symbol
- `run_retrieval`
- runtime dependency validation
- live run 主循环
- 单 query 的高层编排 `_retrieve_one_query`
- manifest metadata 构造
- manifest dependencies 构造

未拆分的逻辑和理由：

- manifest metadata / dependencies 保留在 `runner.py`，因为它们直接依赖 `RetrievalRunConfig` 的完整字段和 artifact contract。
- runtime dependency validation 保留在 `runner.py`，因为它是 public run 入口前置检查。
- `_retrieve_one_query` 保留在 `runner.py` 作为薄编排层，具体 query paths / recall / rerank / trace 已拆出。

## 4. 对外 API

对外 API 不变：

- `eval_platform.retrieval.run_retrieval`
- `eval_platform.retrieval.RetrievalRunConfig`
- `eval_platform.retrieval.RetrievalRunError`
- `eval_platform.retrieval.read_retrieval_run_artifact`
- `eval_platform.retrieval.write_retrieval_run_artifact`

未修改：

- `RetrievalRunConfig` 字段名、默认值和校验语义。
- retrieval artifact JSON schema。
- manifest metadata 字段和值。
- manifest dependencies。
- trace 字段和值。
- ES / Milvus / hybrid recall 结果顺序。
- rerank input、rerank hits、final hits 顺序。
- replay 模式行为。
- HTTP rerank / embedding client。
- ES / Milvus adapter 协议。

## 5. 行为一致性证明

开发前在最新 `main` 上运行：

```bash
pytest tests/retrieval
```

结果：`51 passed in 0.19s`。

开发后新增并通过模块级测试：

- `tests/retrieval/test_query_paths.py`
- `tests/retrieval/test_recall.py`
- `tests/retrieval/test_rerank_flow.py`
- `tests/retrieval/test_replay.py`

固定 fake-client 合约测试：

- `tests/retrieval/test_runner.py::test_run_retrieval_fixed_live_artifact_can_be_replayed_without_behavior_changes`

该测试覆盖指定配置：

- `retrieval_mode="hybrid"`
- `top_k=2`
- `query_limit=1`
- `rewrite_enabled=True`
- `sub_queries=2`
- `rerank_enabled=True`
- `rerank_candidate_cap=2`
- `rerank_cross_path_topk=2`
- `trace_mode="replay"`
- `execution_mode="live"`

并验证：

- replay records 的 `model_dump(mode="json")` 与 source live records 完全一致。
- live / replay manifest metadata 关键字段保持预期。
- live / replay manifest dependencies 保持预期。
- live / replay complete marker 均存在。
- replay 不调用 live clients 的行为仍由既有测试覆盖。

## 6. 测试结果

已运行：

```bash
pytest tests/retrieval
pytest
ruff check .
mypy .
```

结果：

- `pytest tests/retrieval`
  - `64 passed in 0.20s`
- `pytest`
  - `586 passed in 2.23s`
- `ruff check .`
  - `All checks passed!`
- `mypy .`
  - `Success: no issues found in 165 source files`

## 7. 外部服务访问

- 是否访问真实 S3：`no`
- 是否访问真实 ES：`no`
- 是否访问真实 Milvus：`no`
- 是否访问真实 embedding：`no`
- 是否访问真实 rerank：`no`
- 是否访问真实 rewrite：`no`

本轮只使用本地 fake client 和本地 artifact store 测试。

## 8. 风险与未决项

- `RetrievalRunError` 的定义移入 `retrieval/errors.py`，`runner.py` 和 package public import 仍保留同名入口；常规 import API 不变。
- `retrieval/rerank.py` 仍专注 HTTP adapter，rerank flow helper 使用 `rerank_flow.py`，避免混改 adapter。
- 后续如果继续拆 benchmark setting 或 trace schema，应另起任务；本轮未改变 schema 或配置字段。

## 9. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无
