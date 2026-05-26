# 0010. Embedding Consistency Hardening

## Status

Accepted

## Context

当前主线已经支持：

- `embeddings` artifact schema
- `HTTPEmbeddingClient`
- `run_embedding(...)`

但 embedding 阶段仍存在一个现实问题：

1. 同一个能力池可能由多个 embedding API endpoint 提供。
2. 在正式产出 artifact 之前，需要先验证这些 endpoint 对同一输入是否可混用。
3. 现有 provenance 只记录模型名和维度，无法完整回答：
   - 实际用了哪个 endpoint
   - 是否有多个 endpoint
   - 是否做过一致性预检查
   - 关键运行参数是什么

如果这些信息不显式记录，后续 ES / Milvus ingest 阶段会继续放大不一致。

## Decision

本次在 `src/eval_platform/embeddings/` 中补强以下能力：

1. 新增多 endpoint 配置模型：
   - `MultiEndpointEmbeddingConfig`
   - `EmbeddingConsistencyTolerance`
2. 新增一致性预检查结果模型：
   - `EmbeddingConsistencyCheckResult`
3. 新增一致性预检查辅助函数：
   - `run_embedding_consistency_check(...)`
4. 扩展 `EmbeddingProvenance`，显式记录：
   - `endpoint_id`
   - `endpoint_ids`
   - `consistency_check`
   - `runtime_parameters`
5. 扩展 `EmbeddingRunConfig`，允许 runner 在写 artifact 时把上述信息写入 provenance。

一致性规则采用：

- 同一输入文本
- 每个 endpoint 返回一个向量
- 向量维度必须一致
- 默认按完全一致处理，即 `max_abs_diff == 0.0`
- 如需容差，通过 `EmbeddingConsistencyTolerance.max_abs_diff` 显式声明

## Consequences

1. 后续可以在不接真实网络的测试里，稳定模拟多 endpoint 一致 / 不一致场景。
2. embeddings artifact 的 provenance 能明确表达 endpoint 集合与预检查结果。
3. 后续 ES / Milvus ingest artifact 可以直接消费更完整的 embedding 身份信息。
4. 本阶段仍不实现：
   - 真实 ES / Milvus builder
   - retrieval pipeline
   - metrics
