# 开发背景与原则

## 1. 项目背景

`sci-retrieval-eval` 不是一个新的检索系统，而是一套面向研发团队的离线评测基座。

团队已经能通过 `sciverse_benchmark` 跑出部分评测结果，也已有 `sciverse_clean/agentic-search` 作为主线服务与 chunk / search API 口径来源。

当前真正的问题不是“无法评测”，而是：

1. 不同同事跑出的结果不一致。
2. 差异来源不清楚，可能来自数据处理、参数、服务状态或脚本口径。
3. 结果缺乏统一证据链，团队无法稳定复现彼此实验。

因此，本项目的第一目标不是扩展更多检索功能，而是把离线评测重构为统一、可复现、可审计的团队流程。

## 2. 项目目标

本项目当前阶段的目标是：

1. 统一离线评测流程。
2. 固化实验输入、过程和结果身份。
3. 在外部 API 行为一致的前提下，使任何人都能复现他人的实验。
4. 为后续 ES / Milvus / retrieval / metrics 接入提供标准基座。

一句话概括：

> 先建立团队共享的评测真相来源，再逐步补齐完整检索评测闭环。

## 3. 关键现实约束

本项目不是从零开始，必须承认并吸收以下现实：

1. `sciverse_benchmark` 已有可运行的 benchmark 流程。
2. `sciverse_clean/agentic-search` 定义了主线 chunk 口径与 `/v1/search` API 语义。
3. 外部依赖很多，包括 chunk 逻辑、embedding API、rerank API、ES、Milvus、S3。
4. 外部 API 的稳定性与一致性无法完全由本系统保证，但其配置与调用身份必须被显式记录。

## 4. 开发原则

### 4.1 一致性优先于功能扩展

如果一个新功能会扩大“看起来能跑、但无法复现”的空间，就不应该优先做。

### 4.2 实验身份优先于结果分数

在结果可信之前，先回答清楚：

1. 这次实验的输入是什么。
2. 这次实验评测的对象是什么。
3. 这次结果由哪些 artifact 和配置生成。

### 4.3 主线口径显式引用

凡是引用 `sciverse_clean` 的 chunk / search 行为，必须把 repo、commit、dirty 状态或等效身份显式记录下来，不能只靠“默认用主线代码”。

### 4.4 配置必须可声明、可回放

配置可以来自文件、环境变量、CLI，但最终生效配置必须能被记录和回放。

### 4.5 先规范 schema，再接真实服务

开发顺序应当是：

1. 定义 schema
2. 定义 artifact / record
3. 写 fake / local 实现
4. 写测试
5. 最后再接真实外部服务

### 4.6 小 PR，单一目标

每次开发只解决一个清晰问题，例如：

- 定义标准实验记录
- 定义 index metadata artifact
- 定义 retrieval trace schema

不要在同一个 PR 里同时推进多个阶段。

## 5. 当前主线判断

截至当前主线，`normalized_dataset -> chunked_corpus -> embeddings` 已具备较好的 artifact 与 provenance 记录能力。

主线还缺少的是：

1. embedding 阶段更完整的配置与运行身份
2. 多 endpoint embedding API 的一致性校验机制
3. ES ingest artifact
4. Milvus ingest artifact

因此，下一阶段开发应先把 embedding 阶段做扎实，再进入 ES / Milvus ingest artifact。

## 6. 当前 embedding 原则

embedding 阶段当前有一个额外现实约束：

1. 可能同时提供多个 embedding API endpoint。
2. 这些 endpoint 在工程上被视为同一能力池。
3. 在把它们用于正式 artifact 生产前，需要先验证“同一输入字符串是否得到相同结果”。

这意味着本项目不只要记录“用了哪个 embedding API”，还要回答：

1. 是否存在多个 endpoint。
2. 一致性预检查是否通过。
3. 如果存在容差，容差规则是什么。
4. 正式产物最终使用了哪个 endpoint 或 endpoint 集合。
