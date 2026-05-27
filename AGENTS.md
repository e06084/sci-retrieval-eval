# AGENTS.md

本文件是 `sci-retrieval-eval` 仓库内 agent 的长期工作规则，也是双 session 协作的唯一入口页。所有回复使用简体中文。

## 1. 工作定位

本仓库是面向科研检索系统的离线评测基座，核心目标是让 corpus 构建、检索实验、指标计算都可复现、可审计、可追溯。

开发 session 的工作目录应设置为本仓库根目录：

```text
/home/qiujiuantao/codex_project/sci-base/sci-retrieval-eval
```

默认协作方式是双 session：

- 验收 / 调度 session：决定下一步任务，维护 `TASK.md`，验收 commit 和 `report.md`。
- 开发 session：读取 `TASK.md`，实现、自测、提交 commit，并更新 `report.md`。

不要依赖聊天记录交接，任务与结果以仓库文件为准。

## 2. 文件职责

### 开发 session 必读

- `TASK.md`：当前任务单，本地文件，不进 git。开发 session 只读，不要修改。
- `report.md`：开发报告，开发 session 必须更新并随 commit 提交。
- `docs/architecture.md`：项目背景、架构、阶段目标和工程原则。

### 建议保留的长期文档

- `README.md`：仓库入口和基本使用说明。
- `docs/decisions/*.md`：ADR，记录已确认的关键设计决定。
- `docs/operations/*.md`：真实环境操作说明和运行手册。

### 维护归属

由验收 / 调度 session 维护：

- `TASK.md`
- `AGENTS.md`
- `docs/architecture.md` 中的项目背景、目标、原则部分

由开发 session 维护：

- `report.md`
- 与当前开发任务直接相关的 ADR / 设计说明，例如 `docs/decisions/*.md`
- 与当前开发任务直接相关的 operation runbook，例如 `docs/operations/*.md`

默认不应改动：

- `README.md`
- 既有 ADR，除非任务要求修订设计历史或补充新 ADR

如果 `TASK.md` 没有明确要求，开发 session 不应修改由验收 / 调度 session 维护的文件。

### 不应提交

- `TASK.md`
- 真实 `config.yaml`
- 密钥、token、AK/SK、真实密码
- `.local_artifacts/`
- 临时 inventory / dry-run 输出，除非任务明确要求且内容已脱敏

## 3. 开发边界

- 只做 `TASK.md` 指定的工作，不顺手扩展范围。
- 不修改 `TASK.md`，如发现任务有问题，写入 `report.md` 并等待验收 / 调度 session 调整。
- 不修改与任务无关的文件。
- 如果任务变更，先由验收 / 调度 session 更新 `TASK.md`，再继续开发。
- 如果开发中发现新问题，不要直接扩做，先写入 `report.md`。
- 不回滚用户或其他 session 的改动，除非明确要求。
- 不提交真实外部依赖配置或任何敏感信息。
- 不在单元测试中访问真实 S3、ES、Milvus、embedding、rerank、rewrite 服务。
- 真实外部服务访问只能作为显式的手工检查，并写入 `report.md`。

## 4. 最简工作流

1. 验收 / 调度 session 更新 `TASK.md`。
2. 开发 session 读取 `TASK.md` 并实现。
3. 开发 session 更新 `report.md`。
4. 开发 session 运行测试并提交 commit。
5. 验收 / 调度 session 读取 `report.md`，检查 `git log`、`git diff` 和必要的真实只读结果。
6. 验收 / 调度 session 给出结论；通过后再 push / PR / merge。

适合这个流程的场景：

- 单个 PR
- 单个阶段性任务
- 需要开发和验收分开的协作

不适合把本文件扩展成复杂项目管理看板。

## 5. 工程原则

- 评测流程以 artifact 为中心。
- 每个可复现阶段都应有 manifest 和完成标记。
- 下游步骤不能消费未完成 artifact。
- 配置、输入、外部服务身份、关键参数必须可记录和可回放。
- retrieval 执行和 metrics 计算保持分离。
- 默认保留完整 trace；如果要省空间，必须由任务明确指定。
- 优先写 schema、artifact、fake client 和测试，再接真实服务。
- 小 PR、单一目标、可验收。
- 如果开发 session 的 commit 改动了“由验收 / 调度 session 维护”的文件，默认视为越界，除非 `TASK.md` 明确允许。
- 如果验收 / 调度 session 的 commit 改动了业务代码，必须在 `TASK.md` 或验收结论里说明原因。

## 6. 模块边界

- `src/eval_platform/artifacts/`：artifact store、manifest、本地/S3 后端。
- `src/eval_platform/config/`：全局配置加载和脱敏输出。
- `src/eval_platform/datasets/`：raw / normalized dataset schema 和转换。
- `src/eval_platform/chunking/`：chunk schema、runner、外部 chunker provenance。
- `src/eval_platform/embeddings/`：embedding client、runner、artifact 读写。
- `src/eval_platform/indexes/`：ES / Milvus ingest 和 index artifact。
- `src/eval_platform/retrieval/`：检索 adapter、融合、rerank、trace、replay。
- `src/eval_platform/metrics/`：指标计算和 metrics artifact。
- `src/eval_platform/benchmark/`：benchmark run 编排。
- `src/eval_platform/mteb_adapter/`：MTEB 数据和接口适配。
- `scripts/`：真实环境辅助脚本或临时 orchestration，必须安全、可审计、默认不破坏外部状态。
- `tests/`：与模块对应的单元测试和 fake client 测试。

## 7. 测试要求

开发完成前至少运行 `TASK.md` 要求的命令。若任务没有特殊要求，默认运行：

```bash
pytest
ruff check .
mypy .
```

如果因为环境原因无法运行，必须在 `report.md` 中写清楚：

- 未运行的命令
- 原因
- 已完成的替代检查
- 剩余风险

## 8. 完成标准

开发 session 完成任务前必须满足：

- 代码和文档改动与 `TASK.md` 对齐。
- 新功能有测试，测试不访问真实外部服务。
- `report.md` 写清楚实现内容、验证命令、结果、风险和最终 commit。
- 已提交 commit。
- 工作区没有意外 tracked 改动。

验收 / 调度 session 通过 `report.md`、`git diff`、`git log` 和必要的真实只读检查决定是否通过。
