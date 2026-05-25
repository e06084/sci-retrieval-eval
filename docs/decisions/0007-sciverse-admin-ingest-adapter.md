# ADR 0007: Sciverse admin-ingest adapter

## Status

Accepted

## Context

当前系统已经具备：

- `normalized_dataset` artifact
- `chunked_corpus` artifact
- 外部 repo clean-state / remote URL / commit SHA 校验
- 通用 `ExternalChunker` protocol
- `run_chunking(...)` 统一写 artifact 和 provenance

但真实 chunk 逻辑位于外部仓库 `sciverse_clean/agentic-search` 的
`python_services/admin-ingest` 包内。我们需要在不复制外部代码、不写死本地路径、
不绕过现有 provenance 机制的前提下，把这套真实 chunk 逻辑接入当前项目。

## Decision

新增一个 repo-specific 但仍参数化的薄 adapter：

- 外部 repo 版本校验继续使用：
  - `ExternalChunkerRepoSpec`
  - `verify_external_chunker_repo(...)`
- 新增：
  - `SciverseAdminIngestChunkerConfig`
  - `SciverseAdminIngestExternalChunker`
  - `run_version_pinned_sciverse_chunking(...)`

适配方式：

1. 用户显式提供外部 repo 本地路径、remote URL、commit SHA。
2. 系统先验证当前 checkout：
   - repo clean
   - remote URL 匹配
   - commit SHA 匹配
3. adapter 运行时将：
   - `<repo>/python_services/admin-ingest` 临时加入 `sys.path`
   - 动态导入 `admin_ingest.chunk.recursive_split`
   - 动态导入 `admin_ingest.pipeline.steps`
4. adapter 将内部 `NormalizedDataset` 文档编码为 admin-ingest 可消费的 NDJSON 行。
5. adapter 调用 `chunk_ndjson_records(...)`，并把返回值规范化为 `ChunkRecord`。
6. 最终仍通过 `run_chunking(...)` 写出 `chunked_corpus` artifact。

## Consequences

优点：

- 不复制 `sciverse_clean` 代码。
- 不把真实本地路径写死进主仓库。
- 代码版本管理语义统一为：
  - remote URL
  - commit SHA
  - clean state
  - chunk params
- 输出 artifact 仍保留统一的 `ChunkerProvenance` 和 source dependency。

代价：

- 运行前必须准备好正确的外部 repo checkout。
- 当前只支持 Python 本地导入式 adapter。
- 若未来外部 chunker 变为 subprocess / RPC / service 模式，需要后续单独扩展。
