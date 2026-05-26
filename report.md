# 开发报告

本文件由开发 session 维护。

验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：chunked_corpus / embeddings artifact 支持 shard-aware 对齐
- 当前分支：`feat/sharded-corpus-artifacts`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-26
- 完成时间：2026-05-26

## 2. 本次改动

- 改了什么：
  - 返工 `src/eval_platform/chunking/artifact.py`
    - `iter_chunk_shards(...)` 从返回 `list[...]` 改成真正的惰性 iterator
    - 每次只读取一个 chunk shard 文件
  - 返工 `src/eval_platform/embeddings/artifact.py`
    - `iter_embedding_shards(...)` 从返回 `list[...]` 改成真正的惰性 iterator
    - 新增 `write_embedding_shards_artifact(...)`
    - embedding shard 可以逐 shard 写入，不再要求先构造全量 `EmbeddedCorpus`
  - 返工 `src/eval_platform/embeddings/runner.py`
    - 移除对 `read_chunked_corpus_artifact(...)` 的全量依赖
    - 移除累计全部 embedding records 后再写 artifact 的路径
    - 改为逐 shard 读取 source chunk、逐 shard 生成 embedding、逐 shard 落盘
  - 新增 `src/eval_platform/chunking/progress.py`
    - 定义 `ProgressEvent`
    - 定义 `ProgressReporter`
    - 定义 `report_progress(...)`
  - 更新 `src/eval_platform/chunking/artifact.py`
    - `write_chunked_corpus_artifact(...)` 支持 `file_record_num`
    - 新增 `ChunkShardDescriptor`
    - 新增 `ChunkShard`
    - 新增 `build_chunk_shards(...)`
    - 新增 `iter_chunk_shards(...)`
    - chunk manifest / files 补 `sha256`
  - 更新 `src/eval_platform/chunking/runner.py`
    - `ChunkingRunConfig` 新增 `file_record_num`
    - `run_chunking(...)` 支持 progress reporter
  - 更新 `src/eval_platform/chunking/__init__.py`
    - 导出 shard / progress 相关公共接口
  - 更新 `src/eval_platform/embeddings/artifact.py`
    - `write_embeddings_artifact(...)` 支持 shard-aware 输出
    - 新增 `EmbeddingShardDescriptor`
    - 新增 `EmbeddingShard`
    - 新增 `iter_embedding_shards(...)`
    - embedding manifest / files 补 `sha256`
  - 更新 `src/eval_platform/embeddings/runner.py`
    - `run_embedding(...)` 按 source chunk shard 对齐输出
    - 支持 batch 级和 shard 级 progress reporter
  - 更新 `src/eval_platform/embeddings/__init__.py`
    - 导出 shard 相关公共接口
  - 更新 `src/eval_platform/datasets/raw_normalize.py`
    - `normalize_raw_dataset_artifact(...)` 支持 progress reporter
  - 扩展测试：
    - `tests/chunking/test_artifact.py`
    - `tests/chunking/test_runner.py`
    - `tests/embeddings/test_artifact.py`
    - `tests/embeddings/test_runner.py`
    - `tests/datasets/test_raw_normalize.py`
  - 新增 ADR：
    - `docs/decisions/0014-sharded-corpus-artifacts.md`
  - 更新：
    - `docs/ai/current_status.md`
    - `report.md`
- 为什么这样改：
  - 当前 `chunked_corpus` 和 `embeddings` 默认都是单文件。
  - 大 corpus 下，后续 Milvus ingest 如果需要按 `chunk_id` 对齐 chunk 和 embedding，会被迫全量加载两份大文件，内存压力很大。
  - 用户要求 chunk shard 按 source doc 数切分，并让 embedding shard 与之逐 shard 对齐，为后续流式 zip join 做准备。
  - 上一版已经完成 shard 文件布局，但 `iter_*_shards(...)` 和 `run_embedding(...)` 仍然会全量加载；这轮返工的目标是把对下游关键的读取与写入路径改成真正流式。
- 没改什么：
  - 没有实现 Milvus ingest
  - 没有实现 ES ingest
  - 没有实现正式 corpus build runner
  - 没有接真实 embedding API
  - 没有改 raw_dataset / normalized_dataset artifact 格式

## 3. 涉及文件

- `src/eval_platform/chunking/__init__.py`
- `src/eval_platform/chunking/artifact.py`
- `src/eval_platform/chunking/progress.py`
- `src/eval_platform/chunking/runner.py`
- `src/eval_platform/embeddings/__init__.py`
- `src/eval_platform/embeddings/artifact.py`
- `src/eval_platform/embeddings/runner.py`
- `src/eval_platform/datasets/raw_normalize.py`
- `tests/chunking/test_artifact.py`
- `tests/chunking/test_runner.py`
- `tests/embeddings/test_artifact.py`
- `tests/embeddings/test_runner.py`
- `tests/datasets/test_raw_normalize.py`
- `docs/decisions/0014-sharded-corpus-artifacts.md`
- `docs/ai/current_status.md`
- `report.md`

### 3.1 范围自检

- 是否改动了流程控制文档：`no`
- 如果是，改动理由：无

## 4. 实现说明

### 4.1 sharding 语义

本轮新增配置：

```text
file_record_num
```

语义固定为：

```text
每个 chunk shard 包含多少个切分前 source doc
```

不是：

```text
每个 shard 包含多少个 chunk
```

具体行为：

1. 先按 `doc_id` 的连续输出段聚合 chunk。
2. 每 `file_record_num` 个 source doc 聚成一个 shard。
3. 同一个 source doc 产生的多个 chunk 不会跨 shard。
4. shard 内 chunk 顺序保持 chunker 原输出顺序。
5. 不传 `file_record_num` 时，保持旧单文件行为。

### 4.2 新旧 artifact 格式兼容

`chunked_corpus`：

- 旧格式：

```text
chunked_corpus/<artifact_id>/chunks.jsonl
```

- 新格式：

```text
chunked_corpus/<artifact_id>/chunks/part-00000.jsonl
chunked_corpus/<artifact_id>/chunks/part-00001.jsonl
...
```

`embeddings`：

- 旧格式：

```text
embeddings/<artifact_id>/embeddings.jsonl
```

- 新格式：

```text
embeddings/<artifact_id>/embeddings/part-00000.jsonl
embeddings/<artifact_id>/embeddings/part-00001.jsonl
...
```

兼容策略：

1. writer 在未启用 sharding 时继续写旧单文件。
2. reader 通过同名函数兼容读取新旧两种布局：
   - `read_chunked_corpus_artifact(...)`
   - `read_embeddings_artifact(...)`
3. 新增流式 shard 读取 helper：
   - `iter_chunk_shards(...)`
   - `iter_embedding_shards(...)`

需要特别说明：

1. `read_chunked_corpus_artifact(...)`
2. `read_embeddings_artifact(...)`

为了兼容旧调用方，仍然是**全量 in-memory 读取 API**。

真正给后续 ingest 用的流式接口是：

1. `iter_chunk_shards(...)`
2. `iter_embedding_shards(...)`

### 4.3 embedding 如何和 chunk shard 对齐

`run_embedding(...)` 在返工后不再走“先全量读 source，再全量攒 embedding”的路径。

新的对齐策略：

1. 先按 `iter_chunk_shards(...)` 惰性读取 source chunk shard。
2. 每个 chunk shard 内按 batch 调 `client.embed_texts(...)`。
3. 当前 shard 生成的 embedding 只写入对应 embedding shard。
4. 同一 shard 内逐行保持：
   - `chunk_id` 一致
   - `doc_id` 一致
   - 顺序一致
5. 每个 embedding shard 在当前 shard 完成后立即写入 store。
6. 最后只基于计数和 shard descriptor 汇总 manifest，不再保留全量向量列表。

这保证后续 ingest 可以逐 shard 做流式 zip join。

### 4.4 manifest 新增字段

chunk manifest 新增：

1. `sharding`
2. `shards`
3. `files[].sha256`

其中每个 chunk shard 记录：

1. `path`
2. `source_doc_count`
3. `chunk_count`
4. `first_chunk_id`
5. `last_chunk_id`
6. `sha256`

embedding manifest 新增：

1. `source_chunked_corpus_artifact_id`
2. `alignment_key`
3. `alignment_order`
4. `sharding`
5. `shards`
6. `files[].sha256`

其中每个 embedding shard 记录：

1. `source_chunk_file`
2. `embedding_file`
3. `source_chunk_count`
4. `embedding_count`
5. `first_chunk_id`
6. `last_chunk_id`
7. `sha256`

### 4.5 sha256 如何计算和校验

本轮对 shard 文件都补了 `sha256`：

1. 先把 JSONL 文本编码成 `bytes`
2. 对完整字节流做 `sha256`
3. 同时写入：
   - `manifest.files[].sha256`
   - `manifest.metadata.shards[].sha256`

单文件旧布局也一并补了 `sha256`，不是只在新 shard 文件上支持。

### 4.6 progress reporter 设计与支持范围

新增通用接口：

```python
class ProgressEvent(BaseModel):
    stage: str
    current: int
    total: int | None = None
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
```

以及：

```python
ProgressReporter = Callable[[ProgressEvent], None]
```

当前支持阶段：

1. raw-to-normalized
   - `stage="raw_to_normalized"`
   - 至少汇报：
     - corpus
     - queries
     - instructions
     - qrels
2. chunking
   - `stage="chunking"`
   - 汇报：
     - source doc 完成事件
     - shard 预写入事件
3. embedding
   - `stage="embedding"`
   - 汇报：
     - batch 完成事件
     - shard 完成事件

事件 `metadata.kind`：

1. `corpus`
2. `queries`
3. `instructions`
4. `qrels`
5. `source_doc`
6. `shard`
7. `batch`

后续 CLI runner 接入方式：

1. 在 runner 外层传入 `progress_reporter`
2. CLI 只需把 `ProgressEvent` 映射成日志 / 进度条
3. artifact 内容和 manifest 不需要关心 CLI 展示逻辑

### 4.7 本轮返工后哪些 API 是流式

真正流式：

1. `iter_chunk_shards(...)`
2. `iter_embedding_shards(...)`
3. `run_embedding(...)` 的 source chunk 消费路径
4. `write_embedding_shards_artifact(...)`

仍然是兼容性全量 API：

1. `read_chunked_corpus_artifact(...)`
2. `read_embeddings_artifact(...)`
3. `write_embeddings_artifact(...)`

### 4.8 reporter 异常时的行为

progress callback 默认不传时行为不变。

如果 callback 抛异常：

1. raw-to-normalized 会直接失败
2. chunking 会直接失败
3. embedding 会直接失败

并且测试已覆盖：

```text
不会写出 _SUCCESS
```

这满足任务单要求的“进度回调不能制造伪成功 artifact”。

## 5. 自检结果

### 5.1 必跑命令

```bash
pytest tests/datasets/test_raw_normalize.py
pytest tests/chunking tests/embeddings
ruff check .
mypy .
pytest
```

### 5.2 输出摘要

- `pytest tests/datasets/test_raw_normalize.py`：
  - 通过，`7 passed`
- `pytest tests/chunking tests/embeddings`：
  - 通过，`224 passed`
- `ruff check .`：
  - 通过
- `mypy .`：
  - 通过，`Success: no issues found in 98 source files`
- `pytest`：
  - 通过，`376 passed`

## 6. 风险与未决项

- 已知风险：
  - `run_chunking(...)` 仍然会先把所有 chunk 收到内存里，再统一写 shard；这已经足够支撑 shard-aware artifact，但还不是最终的流式 chunk writer
- 未覆盖场景：
  - 本轮没有实现正式 Milvus ingest，因此还没有真实验证 shard zip join 的下游消费
- 需要验收者重点检查的点：
  - `file_record_num` 的语义是否足够清晰，是否完全按 source doc 数而不是 chunk 数切分
  - shard manifest 字段是否已足够支撑后续 Milvus 流式 ingest

## 7. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无

## 8. 提交信息

- 是否已提交：`yes`
- commit subject：`Stream shard readers and embedding writer`
- 验收者确认的最终 commit：
