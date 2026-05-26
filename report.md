# 进度报告

## 当前阶段

Embedding API client ready for merge review.

## 本次开发

- 在现有 `EmbeddingClient` / `FakeEmbeddingClient` 基础上新增：
  - `EmbeddingClientError`
  - `HTTPEmbeddingClientConfig`
  - `HTTPEmbeddingClient`
  - `http_embedding_client_from_env(...)`
- 保持 runner 语义不变：
  - client 只负责文本到向量
  - runner 继续负责 artifact 读写

## HTTP client 语义

- 默认使用标准库 `urllib.request`
- 支持 batching
- 支持两种响应格式：
  - `{"embeddings": [[...], [...]]}`
  - `{"data": [{"embedding": [...]}, {"embedding": [...]}]}`
- API key 只从环境变量读取
- 单元测试全部使用 fake transport，不访问网络

## 测试

已覆盖：

- config 校验
- headers 默认值不共享
- response format A
- response format B
- batching 调用次数
- model 字段有无
- HTTP 非 2xx
- 非法 JSON
- 向量数量不匹配
- 空 vector
- NaN / inf / 非数字
- 从环境变量构造 client

## 范围约束

本 PR：

- 不访问真实 embedding API
- 不实现 Milvus
- 不实现 ES
- 不实现 retrieval / metrics / frontend

## 建议后续方向

- 合并 `feat/embedding-api-client`
- 然后开始 `feat/milvus-index-schema`
