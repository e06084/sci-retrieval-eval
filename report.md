# 开发报告

本文件由开发 session 维护。

验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：修复 embedding 一致性门禁
- 当前分支：`feat/embedding-consistency-hardening`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-26
- 完成时间：2026-05-26

## 2. 本次改动

- 改了什么：
  - 给 `run_embedding(...)` 增加了一致性门禁：
    - 多 endpoint 但没有 `consistency_check` 时拒绝写 artifact
    - `consistency_check.passed is False` 时拒绝写 artifact
  - 给 `EmbeddingConsistencyCheckResult` 增加语义自洽校验：
    - `passed=True` 时不允许 `failure_reason`
    - `passed=False` 时必须提供非空 `failure_reason`
  - 把 `run_embedding_consistency_check(...)` 中 endpoint client 的异常、空向量、非数字、非有限值都收敛为结构化失败结果，而不是直接向外抛异常。
  - 为以上行为补了 schema / client / runner 测试。
- 为什么这样改：
  - 上一轮实现虽然补了 endpoint 配置和 consistency check schema，但还没有真正把 consistency check 变成 artifact 写入前的硬门禁。
  - 这会留下“预检查失败但仍能写出成功 artifact”的漏洞，不符合本项目的一致性目标。
- 没改什么：
  - 没有实现 ES / Milvus builder。
  - 没有实现 retrieval pipeline。
  - 没有实现 metrics / report 生成逻辑。
  - 没有访问真实外部服务。

## 3. 涉及文件

- `src/eval_platform/embeddings/__init__.py`
- `src/eval_platform/embeddings/client.py`
- `src/eval_platform/embeddings/runner.py`
- `src/eval_platform/embeddings/schema.py`
- `tests/embeddings/test_client.py`
- `tests/embeddings/test_runner.py`
- `tests/embeddings/test_schema.py`
- `report.md`

### 3.1 范围自检

- 是否改动了流程控制文档：`no`
- 如果是，改动理由：无

## 4. 实现说明

### 4.1 关键决策

- 决策 1：
  - 不重做上一轮设计，只在现有 `EmbeddingRunConfig`、`EmbeddingProvenance` 和 `run_embedding_consistency_check(...)` 上做门禁收紧。
- 决策 2：
  - `run_embedding(...)` 只有在单 endpoint，或多 endpoint 且提供通过的 `consistency_check` 时，才允许写 artifact。
- 决策 3：
  - endpoint client 运行异常属于“预检查失败结果”，不是调用方错误；但 `clients` 数量不匹配、`input_text` 为空仍视为调用方错误并抛 `EmbeddingClientError`。

### 4.2 关键行为

- 行为 1：
  - 多 endpoint 且缺少 `consistency_check` 时，`run_embedding(...)` 抛 `EmbeddingRunError`，并且不会写出 `embeddings.jsonl` 或 `_SUCCESS`。
- 行为 2：
  - `run_embedding_consistency_check(...)` 在 endpoint 异常、空向量、非数字、非有限值时返回 `passed=False` 的结构化结果。
- 行为 3：
  - `EmbeddingConsistencyCheckResult` 现在会强制 `passed` 与 `failure_reason` 的语义一致。

## 5. 自检结果

### 5.1 必跑命令

```bash
git status --short
git diff --name-only origin/main...HEAD
pytest tests/embeddings
ruff check .
mypy .
```

### 5.2 输出摘要

- `git status --short`：
  - 开发完成前仅包含 `src/eval_platform/embeddings/`、`tests/embeddings/` 与 `report.md` 改动。
- `git diff --name-only origin/main...HEAD`：
  - 本轮修补应只涉及 `src/eval_platform/embeddings/`、`tests/embeddings/`、`report.md`。
- `pytest tests/embeddings`：
  - 通过，`103 passed`
- `pytest`：
  - 本轮未重跑；上一轮通过，`313 passed`
- `ruff check .`：
  - 通过
- `mypy .`：
  - 通过，`Success: no issues found in 78 source files`

### 5.3 提交信息

- 是否已提交：`yes`
- commit subject：`Enforce embedding consistency gate`
- 验收者确认的最终 commit：
- 相关 commit 列表：
  - `Harden embedding consistency provenance`
  - `Enforce embedding consistency gate`

## 6. 风险与未决项

- 已知风险：
  - 当前一致性预检查仍是显式传入 `consistency_check` 结果，不会自动替开发者触发真实 endpoint 探测。
- 未覆盖场景：
  - 没有覆盖真实外部 endpoint 的数值漂移统计，只覆盖 fake client / fake transport 场景。
- 需要验收者重点检查的点：
  - `run_embedding(...)` 的门禁位置是否足够早，能否保证 artifact 完全无副作用。
  - `EmbeddingConsistencyCheckResult` 的语义约束是否足够清晰。

## 7. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无
