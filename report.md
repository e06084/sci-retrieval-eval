# 进度报告

## 当前阶段

Embedding runner to S3 PR ready for merge review.

输入链路：

- `chunked_corpus artifact`

输出链路：

- `embeddings artifact`

本阶段只覆盖：

- embedding schema
- embeddings artifact read/write
- fake embedding client
- local source -> local output
- local source -> S3 output

## 本次开发

- 新增 `src/eval_platform/embeddings/schema.py`
  - `EmbeddingProvenance`
  - `EmbeddingRecord`
  - `EmbeddedCorpus`
- 新增 `src/eval_platform/embeddings/jsonl.py`
  - `dump_embeddings_jsonl(...)`
  - `load_embeddings_jsonl(...)`
- 新增 `src/eval_platform/embeddings/artifact.py`
  - `write_embeddings_artifact(...)`
  - `read_embeddings_artifact(...)`
- 新增 `src/eval_platform/embeddings/client.py`
  - `EmbeddingClient`
  - `FakeEmbeddingClient`
- 新增 `src/eval_platform/embeddings/runner.py`
  - `EmbeddingRunConfig`
  - `run_embedding(...)`

## Runner 语义

`run_embedding(source_store, output_store, config, client)` 会：

1. 从 `source_store` 读取 `chunked_corpus`
2. 提取所有 chunk text
3. 调用注入式 client 计算向量
4. 构造 `EmbeddingRecord`
5. 写出 `embeddings` artifact 到 `output_store`

这允许：

- 从本地读 `chunked_corpus`
- 向 S3 写 `embeddings`

## 测试

已覆盖：

- schema 校验
- JSONL round-trip
- embeddings artifact read/write
- runner 本地到本地输出
- runner 本地到 S3 输出
- fake embedding client 的 deterministic 行为
- client 返回数量不匹配时报错
- client 返回维度不匹配时报错

## 范围约束

本 PR：

- 不实现真实 embedding API
- 不实现 HTTP client
- 不实现 batching / retry / rate limit
- 不实现 Milvus 入库
- 不实现 ES / retrieval / metrics

## 建议后续方向

- 合并 `feat/embedding-runner-s3`
- 然后开始 `feat/embedding-api-client`
