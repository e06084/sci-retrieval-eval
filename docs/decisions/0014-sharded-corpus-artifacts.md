# 0014. Chunked Corpus / Embeddings Artifact 支持 Shard-aware 对齐

- Status: Accepted
- Date: 2026-05-26

## Context

当前主线已经有：

- `chunked_corpus` artifact
- `embeddings` artifact
- `run_chunking(...)`
- `run_embedding(...)`

但两类 artifact 仍以单文件为主：

- `chunked_corpus/<artifact_id>/chunks.jsonl`
- `embeddings/<artifact_id>/embeddings.jsonl`

这对小数据集足够，但在大 corpus 下会带来两个问题：

1. 后续 Milvus ingest 如果需要同时读取 chunk 和 embedding，再按 `chunk_id` 对齐，会倾向于一次性加载两份大文件。
2. norm / chunk / embedding 都是长耗时阶段，需要统一的进度回调接口，而不是临时 CLI 逻辑。

## Decision

### 1. Chunk shard 语义

`chunked_corpus` 新增可选 `file_record_num`。

其含义固定为：

- 每个 chunk shard 包含多少个切分前 source doc

不是：

- 每个 shard 包含多少个 chunk

因此：

- 同一个 source doc 产生的多个 chunk 必须落在同一个 shard
- shard 内 chunk 顺序保持 chunker 输出顺序

启用 sharding 后，文件布局改为：

```text
chunked_corpus/<artifact_id>/
  chunks/part-00000.jsonl
  chunks/part-00001.jsonl
  _MANIFEST.json
  _SUCCESS
```

未启用 sharding 时，保持旧布局：

```text
chunked_corpus/<artifact_id>/chunks.jsonl
```

### 2. Embedding shard 对齐

当 source `chunked_corpus` 是 sharded 时，`run_embedding(...)` 必须按 source chunk shard 逐 shard 产出 embedding shard。

文件布局：

```text
embeddings/<artifact_id>/
  embeddings/part-00000.jsonl
  embeddings/part-00001.jsonl
  _MANIFEST.json
  _SUCCESS
```

约束：

- `embeddings/part-00000.jsonl` 对应 `chunks/part-00000.jsonl`
- shard 内 `chunk_id` 顺序必须一致
- shard 内 `doc_id` 顺序必须一致
- 单文件 source 继续保持旧单文件行为

### 3. Manifest 元数据

chunk 和 embedding manifest 都新增：

- `sharding`
- `shards`
- shard 级 `sha256`
- 对齐信息

其中：

- chunk shard 记录 `source_doc_count` / `chunk_count`
- embedding shard 记录 `source_chunk_file` / `source_chunk_count` / `embedding_count`

### 4. 流式 shard 读取

新增 helper：

- `iter_chunk_shards(...)`
- `iter_embedding_shards(...)`

后续 Milvus ingest 可以按 shard 逐个读取，再做流式 zip join，而不需要全量加载两份大文件。

### 5. Progress reporter

新增通用接口：

```python
ProgressEvent
ProgressReporter
```

并接入：

- raw-to-normalized
- chunking
- embedding

本阶段只提供可复用接口，不实现正式 CLI 进度条。

## Consequences

正面影响：

- 后续 ingest 可以按 shard 对齐处理，减少内存峰值
- 新旧 artifact 布局兼容，旧 reader 仍可通过同名函数读取
- 进度汇报从临时脚本下沉到主库 API

代价：

- manifest 元数据更复杂
- writer / reader 需要同时维护单文件和 sharded 两套布局

非目标：

- 本阶段不实现 Milvus ingest
- 不实现 ES ingest
- 不实现正式 corpus build runner
