# 开发任务单

本文件由验收 / 调度 session 维护。

开发 session 进入仓库后，先读本文件，再开始开发。

## 0. 工作目录

```text
/home/qiujiuantao/codex_project/sci-base/sci-retrieval-eval
```

## 1. 角色约束

你是开发 session，不是验收 session。

你要做的事：

1. 只实现本文件指定任务。
2. 只修改必要文件。
3. 补测试。
4. 把结果写进 `report.md`。
5. 提交 commit，但不要自行 merge。

你不要做的事：

1. 不主动扩需求。
2. 不顺手做下一阶段功能。
3. 不改无关模块。
4. 不把临时脚本放到正式包外。
5. 不修改流程控制文档。

## 2. 当前任务

- 任务标题：embedding 一致性与 provenance 补强第一版
- 目标：
  1. 把 embedding 阶段的运行配置和 provenance 做完整。
  2. 支持多 embedding API endpoint 的一致性预检查。
  3. 为后续 ES / Milvus ingest artifact 提供更可靠前提。
- 非目标：
  1. 不实现 ES / Milvus builder。
  2. 不实现 retrieval pipeline。
  3. 不实现 metrics / report。
  4. 不访问真实外部服务。
- 期望输出：
  1. 在 `src/eval_platform/embeddings/` 下补强配置 / provenance / endpoint 检查能力。
  2. 新增多 endpoint 一致性预检查相关对象或函数。
  3. 新增测试。
  4. 必要时补一篇 ADR 或 docs 说明。

## 3. 范围边界

- 允许修改：
  - `src/eval_platform/embeddings/`
  - `src/eval_platform/__init__.py`
  - `tests/embeddings/`
  - `docs/decisions/`
  - `docs/ai/current_status.md`
  - `report.md`
- 明确禁止修改的流程控制文档：
  - `TASK.md`
  - `SESSION_BOARD.md`
  - `validator_handoff.md`
  - `docs/ai/development_background.md`
  - `docs/ai/development_plan.md`
- 禁止修改：
  - `src/eval_platform/chunking/`
  - `src/eval_platform/indexes/`
  - `src/eval_platform/retrieval/`
  - `src/eval_platform/metrics/`
  - `src/eval_platform/mteb_adapter/`
- 可以新增的测试：
  - 多 endpoint 配置校验
  - 一致性预检查结果校验
  - provenance / metadata 字段校验
  - 同输入向量一致 / 不一致场景
- 不应引入的依赖：
  - `pymilvus`
  - `elasticsearch`
  - `mteb`
  - 任何新的网络客户端依赖

## 4. 具体要求

建议补强的内容至少包括：

1. 多 endpoint 配置模型
   - 能表达一个 endpoint 列表
   - 能表达是否要求预检查
   - 能表达一致性容差规则

2. endpoint 一致性预检查结果
   - 至少记录：
     - 检查输入文本
     - 参与检查的 endpoint 标识
     - 是否通过
     - 如失败，失败原因或差异摘要

3. embedding provenance 补强
   - 在现有 `EmbeddingProvenance` 或相关 metadata 中，能够明确：
     - 用了哪个 endpoint 或 endpoint 集合
     - 是否做过一致性预检查
     - 关键运行参数

4. runner 或辅助函数层面的最小支持
   - 不要求接真实网络
   - 但要支持通过 fake transport / fake client 模拟多 endpoint 一致性检查

要求：

1. 尽量复用现有 `HTTPEmbeddingClient`、`EmbeddingProvenance`、`EmbeddingRunConfig`。
2. 字段命名保持直接、稳定，不要过度抽象。
3. 只做 embedding 阶段补强，不扩到 ES / Milvus。
4. 如果你认为“一致”需要容差而不是完全相等，可以调整，但必须在 `report.md` 说明规则。

## 5. 开发步骤

按顺序执行，不要跳步：

1. 读取：
   - `AGENTS.md`
   - `docs/ai/project_brief.md`
   - `docs/ai/current_status.md`
   - `docs/ai/development_background.md`
   - `docs/ai/development_plan.md`
2. 运行以下命令，确认当前分支状态：

```bash
git status
git branch --show-current
git log --oneline -5
git diff --name-only origin/main...HEAD
```

3. 阅读现有：
   - `src/eval_platform/artifacts/manifest.py`
   - `src/eval_platform/embeddings/schema.py`
   - `src/eval_platform/embeddings/client.py`
   - `src/eval_platform/embeddings/runner.py`
4. 实现最小可行补强。
5. 为新增行为补测试。
6. 运行验证命令：

```bash
pytest tests/embeddings
ruff check .
mypy .
```

7. 提交一个清晰 commit。
8. 把结果完整写入 `report.md`。
9. 停止，不要继续做下一个任务。

## 6. 完成标准

同时满足以下条件才算完成：

1. 标准离线实验 schema 可以稳定构造。
2. 能表达多 endpoint embedding 配置与一致性检查。
3. 测试已补齐。
4. `report.md` 已更新。
5. 没有越界改动。
6. 没有引入真实外部依赖。
7. 没有改动流程控制文档。

## 7. `report.md` 填写要求

必须写清楚：

1. 改了哪些文件。
2. endpoint 一致性检查如何定义。
3. 复用了哪些现有对象。
4. 跑了哪些命令。
5. 哪些命令通过，哪些没通过。
6. 提交了哪个 commit。
7. 还有什么风险。

不要只写一句“已完成”。

## 8. 给验收者的提示

如果你遇到以下情况，必须写进 `report.md`：

- 你认为补强位置不应放在 `embeddings/`
- 你删改了我建议的字段
- 测试失败但不是本次改动导致
- 你怀疑需求本身有歧义
- 你没有跑某条命令
