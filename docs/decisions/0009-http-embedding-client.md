## Status

Accepted

## Context

- 当前已有 embedding runner 和 artifact。
- 需要真实 embedding API client。
- 但仍不做 Milvus / ES / retrieval。

## Decision

- 新增 `HTTPEmbeddingClient`。
- client 只负责文本到向量。
- runner 继续负责 artifact 读写。
- API key 从环境变量读取。
- 单元测试使用 fake transport，不访问网络。

## Consequences

- 后续可以用真实 embedding API 产出 embeddings artifact。
- Milvus builder 后续只消费 embeddings artifact。
