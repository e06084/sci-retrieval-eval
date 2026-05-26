# Session Board

这是双 session 协作的唯一入口页。

## 1. 当前路径

```text
/home/qiujiuantao/codex_project/sci-base/sci-retrieval-eval
```

## 2. 文件职责

- `TASK.md`
  - 验收 / 调度 session 写任务
  - 开发 session 只按这里开发

- `report.md`
  - 开发 session 写结果
  - 验收 session 只按这里验收

- `docs/ai/development_background.md`
  - 说明项目为什么优先做一致性、复现性与审计性

- `docs/ai/development_plan.md`
  - 说明当前阶段开发顺序和下一任务选择逻辑

- `validator_handoff.md`
  - 验收原则和长期边界
  - 不随单次任务频繁改动

## 2.1 文档维护归属

### 由验收 / 调度 session 维护并提交

- `TASK.md`
- `SESSION_BOARD.md`
- `validator_handoff.md`
- `docs/ai/development_background.md`
- `docs/ai/development_plan.md`

这些文件属于流程控制面，不应由开发 session 自行改动，除非任务单明确允许。

### 由开发 session 维护并提交

- `report.md`
- `docs/ai/current_status.md`
- 与当前开发任务直接相关的 ADR / 设计说明
  - 例如 `docs/decisions/*.md`

这些文件属于任务交付面，应跟随本次功能改动一起提交。

### 默认不应改动

- `docs/ai/project_brief.md`
- `docs/ai/open_questions.md`
- `docs/ai/handoff_template.md`

除非本轮任务明确要求，否则开发 session 与验收 session 都不应顺手修改这些文件。

## 3. 最简工作流

1. 验收 / 调度 session 更新 `TASK.md`
2. 开发 session 读取 `TASK.md` 并实现
3. 开发 session 更新 `report.md`
4. 验收 session 读取 `report.md` 并执行检查
5. 验收 session 给出结论

## 4. 沟通规则

- 不靠聊天记录交接，以文件为准。
- `TASK.md` 只写“要做什么”。
- `report.md` 只写“做了什么、怎么验证、还有什么风险”。
- 如果任务变更，先改 `TASK.md`，再继续开发。
- 如果开发中发现新问题，不要直接扩做，先写入 `report.md`。
- 开发 session 完成后必须提交 commit，并把 commit hash 写进 `report.md`。
- 验收 session 通过 `git log`、`git diff` 和 `report.md` 同时核对结果。
- 如果开发 session 的 commit 改动了“由验收 / 调度 session 维护并提交”的文件，默认视为越界。
- 如果验收 / 调度 session 的 commit 改动了业务代码，默认需要在 `TASK.md` 或验收结论里说明原因。

## 5. 适用场景

适合：

- 单个 PR
- 单个阶段性任务
- 需要开发和验收分开的并行协作

不适合：

- 多人长期排期管理
- 复杂项目管理看板

## 6. 当前建议

- 开发 session 先读：
  - `TASK.md`
  - `docs/ai/development_background.md`
  - `docs/ai/development_plan.md`
- 开发完成后：
  - 提交 commit
  - 更新 `report.md`
- 验收 session 再读 `report.md` 并检查 commit 与 diff。
