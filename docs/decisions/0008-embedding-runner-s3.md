# ADR 0008: Embedding Runner with S3 Artifact Output

## Status

Accepted

## Context

- 后续 Milvus indexing 需要消费 embeddings artifact。
- 当前阶段只需要计算 embedding 并上传 S3，不入 Milvus。
- embedding artifact 必须保留 `chunk_id` / `doc_id`，以便后续索引和检索层回连 chunk/document。
- embedding provenance 需要记录模型名、provider、api_version、维度等，保证后续可追溯性。

## Decision

- 使用 `embeddings.jsonl` 保存 embedding records。
- 每条记录包含：
  - `chunk_id`
  - `doc_id`
  - `vector`
  - `metadata`
- manifest metadata 记录：
  - `embedding_count`
  - `unique_chunk_count`
  - `unique_doc_count`
  - `embedding_dim`
  - `provenance`
- `run_embedding(source_store, output_store, ...)` 支持 source 和 output 不同 backend。
- 因此可以从本地读 `chunked_corpus`，向 S3 写 `embeddings`。
- 本阶段不实现 Milvus / ES indexing。

## Consequences

- 后续 Milvus builder 只需要读取 embeddings artifact。
- 真实 embedding API client 可在后续 PR 实现。
- 当前测试使用 fake embedding client 和 fake S3。
- 不会污染检索 / indexing 逻辑。
