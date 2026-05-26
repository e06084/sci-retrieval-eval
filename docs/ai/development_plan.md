# 开发计划

## 1. 规划目标

本计划服务于当前阶段目标：

1. 统一离线评测流程。
2. 建立实验身份与结果身份。
3. 在此基础上逐步闭合完整评测链路。

## 2. 阶段划分

### 阶段 A：Embedding 阶段补强

目标：把 `chunked_corpus -> embeddings` 这一段做扎实。

本阶段应完成：

1. embedding 运行配置补强
2. embedding provenance 补强
3. 多 endpoint 一致性预检查 schema / 结果记录
4. 基础测试

本阶段不做：

1. 真实 ES / Milvus 连接
2. retrieval pipeline
3. metrics

这是当前最高优先级。

### 阶段 B：索引级 artifact 身份

目标：让 embeddings 之后的索引阶段也具备可审计身份。

本阶段应完成：

1. Milvus index metadata artifact
2. ES index metadata artifact
3. fake builder / contract test

### 阶段 C：标准检索运行记录

目标：统一“评测时到底调用了谁、用了哪些参数”。

本阶段应完成：

1. retrieval run config schema
2. retrieval trace schema
3. predictions artifact schema

### 阶段 D：指标与结果身份

目标：让最终分数成为可追溯结果，而不是孤立输出。

本阶段应完成：

1. metrics artifact
2. report artifact
3. run summary schema

## 3. 当前推荐顺序

建议按下面顺序推进：

1. embedding 阶段补强
2. Milvus index metadata
3. ES index metadata
4. retrieval run / trace
5. metrics / report

原因很简单：

1. 当前前处理 artifact 已经有基础。
2. 当前最现实的不一致风险集中在 embedding endpoint 口径。
3. 如果不先锁住 embedding 阶段，后面 ES / Milvus ingest 只会继续放大不一致。

## 4. 当前建议的下一个开发任务

下一个开发任务建议是：

`feat/embedding-consistency-hardening`

任务目标：

1. 明确 embedding 运行配置与 provenance。
2. 支持多 endpoint embedding API 的一致性预检查。
3. 为后续 ES / Milvus ingest 提供更可靠的 embedding artifact 前提。

## 5. 完成标准

当以下条件满足时，才算本阶段真正完成：

1. 别人可以从 artifact / record 看懂 embedding 是怎么生成的。
2. 多 endpoint 是否可混用可以通过显式检查判断。
3. 不需要阅读聊天记录也能复现 embedding 前提。
4. 测试中不依赖真实外部服务。
