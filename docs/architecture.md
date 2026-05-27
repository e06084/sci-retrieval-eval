# Architecture

## 1. 背景

`sci-retrieval-eval` 是面向研发团队的离线评测基座，不是新的线上检索系统。

团队已经能通过历史脚本跑出部分 benchmark 结果，但不同同事的结果不一致，且差异来源不容易审计。主要风险来自：

- 原始数据处理口径不一致。
- chunk / embedding / ES / Milvus / rerank 参数不一致。
- 外部 API 状态不可完全由本系统控制。
- 实验过程缺少统一 artifact 证据链。

本项目的目标是建立统一、可复现、可审计的评测流程。在外部 API 行为一致的前提下，任何人都应能复现他人的 corpus 构建、检索运行和指标计算。

## 2. 目标与非目标

当前目标：

- 固化 corpus 构建链路。
- 固化 retrieval run 与 metrics run 的输入、参数、输出和 trace。
- 支持不同 retrieval setting 的可比对 benchmark run。
- 把配置、外部服务身份、代码版本和 artifact 依赖写入可审计记录。

当前非目标：

- 不引入 SQL、Redis、Airflow、MLflow、DVC、Celery。
- 不做复杂任务调度系统。
- 不把真实密钥或真实本地配置提交进仓库。
- 不在单元测试中访问真实 S3、ES、Milvus、embedding、rerank、rewrite 服务。

## 3. Artifact-Driven Design

系统以 artifact 为中心。每个阶段只通过 artifact 传递输入输出，不依赖隐藏的可变状态。

标准 corpus 链路：

```text
raw_dataset
  -> normalized_dataset
  -> chunked_corpus
  -> embeddings
  -> elasticsearch_index
  -> milvus_collection
```

标准评测链路：

```text
normalized_dataset + index artifacts
  -> retrieval_run
  -> metrics_run
  -> benchmark_run
```

每个完成的 artifact 必须包含：

- `_MANIFEST.json`
- `_SUCCESS`
- 必要的数据文件或外部资源元数据

下游阶段必须校验上游 artifact 的 manifest 和 `_SUCCESS`。

## 4. 关键 Artifact

- `raw_dataset`：原始数据快照，记录 raw source、文件列表、hash、fingerprint。
- `normalized_dataset`：统一 corpus / queries / qrels schema。
- `chunked_corpus`：chunk 文本、chunk_id、doc_id、外部 chunker provenance。
- `embeddings`：与 chunk shard 对齐的 embedding 产物，记录模型、endpoint、一致性检查和运行参数。
- `elasticsearch_index`：ES 入库结果、index name、mapping hash、数量校验。
- `milvus_collection`：Milvus 入库结果、collection name、schema hash、chunk/embedding 对齐校验。
- `retrieval_run`：每个 query 的检索结果、完整 trace 或 replay 所需记录。
- `metrics_run`：从 retrieval_run 和 qrels 计算出的指标，不调用检索服务。
- `benchmark_run`：绑定 retrieval_run 与 metrics_run，记录 setting 和主指标。

## 5. 模块边界

- `src/eval_platform/artifacts/`：artifact manifest、本地/S3 store。
- `src/eval_platform/config/`：全局配置加载和脱敏输出。
- `src/eval_platform/datasets/`：raw / normalized dataset schema 和转换。
- `src/eval_platform/chunking/`：chunk schema、runner、外部 chunker 版本记录。
- `src/eval_platform/embeddings/`：embedding client、runner、artifact IO。
- `src/eval_platform/indexes/`：ES / Milvus ingest 和 index artifact。
- `src/eval_platform/retrieval/`：ES、Milvus、hybrid、RRF、rerank、trace、replay。
- `src/eval_platform/metrics/`：doc-level metrics 和 query-level metrics。
- `src/eval_platform/benchmark/`：retrieval + metrics 的最小编排。
- `src/eval_platform/mteb_adapter/`：MTEB 数据和接口适配。
- `scripts/`：真实环境辅助脚本，必须默认安全、可审计。

## 6. 评测语义

MTEB 兼容评测默认使用 doc-level qrels。检索阶段可以返回 chunk-level hits，但 metrics 阶段必须把 chunk hits 投影回 doc ranking。

retrieval 执行和 metrics 计算必须分离：

- `retrieval_run` 可以调用外部服务，并记录完整 trace。
- `metrics_run` 只消费 `normalized_dataset` 和 `retrieval_run`，不调用 ES、Milvus、embedding、rerank 或 rewrite 服务。

默认 trace 策略是完整记录，便于 replay 和排查随机性。若要省空间，必须显式选择 `trace_mode=none`。

## 7. 当前阶段

当前主线已经覆盖：

- raw / normalized / chunk / embedding artifacts。
- shard-aware chunk 与 embedding 对齐。
- ES / Milvus ingest artifacts。
- retrieval_run、metrics_run、benchmark_run artifacts。
- live ES / Milvus retrieval adapters。
- HTTP embedding / rerank 相关能力的基础接入。

当前仍需推进：

- 五个目标数据集的完整 corpus/index artifact 准备。
- 真实 E1-E4 setting 的批量运行和结果比对。
- rewrite / rerank 等可能存在随机性的完整记录与 replay。
- benchmark CLI 或轻量 runner 的批处理体验。
- 多 setting 对比报告。

## 8. 工程原则

- 一致性优先于功能扩展。
- 实验身份优先于结果分数。
- 主线 chunk / search 口径必须显式记录外部 repo 版本或等效身份。
- 配置来源为默认值、配置文件、CLI 参数，优先级依次提高。
- 真实配置文件不进 git，只提交 `config.example.yaml`。
- 先定义 schema、artifact、fake client 和测试，再接真实服务。
- 每个 PR 只解决一个清晰问题。

## 9. 文档结构

- `AGENTS.md`：agent 协作规则和文件职责。
- `README.md`：仓库入口。
- `docs/architecture.md`：本文件，维护项目背景、架构、阶段目标和原则。
- `docs/decisions/*.md`：ADR，保留关键设计历史，不作为开发 session 必读材料。
- `docs/operations/*.md`：真实环境运行手册，按需要阅读。
- `TASK.md`：当前任务单，本地 ignored，不进 git。
- `report.md`：开发 session 的交付报告，随 PR 提交。
