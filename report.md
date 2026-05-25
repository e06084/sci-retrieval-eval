# 进度报告

## 当前阶段

MTEB dataset adapter 已合并到 `main`。`feat/chunking-schema` 已完成 pre-merge 修补（ChunkRecord validation、artifact IO 测试），待 PR / merge review。

## 已完成事项（main）

- Local + S3 artifact store
- normalized dataset schema + JSONL artifact 读写
- MTEB dataset adapter

## 已完成事项（feat/chunking-schema）

- `ChunkRecord` / `ChunkedCorpus` / `ChunkerProvenance`
- `ChunkRecord` offset 与 index 校验
- `dump_chunks_jsonl` / `load_chunks_jsonl`
- `write_chunked_corpus_artifact()` manifest metadata（chunker / chunk_params / counts）
- `write_chunked_corpus_artifact()` source artifact dependency
- `read_chunked_corpus_artifact()` round-trip 与 incomplete artifact 行为测试
- ADR：`docs/decisions/0004-chunked-corpus-schema.md`
- `tests/chunking/` 单元测试

## 本 PR 范围

- 只定义 chunking schema 与 artifact metadata
- 不实现真实 chunking runner、外部 chunk 库调用或 git 检查

## 已验证事项

- 测试不访问网络、真实 S3 或 git 命令
- `pytest` / `ruff check .` 通过

## 当前结论

- chunking schema 已完成 pre-merge 修补，可进入 PR / merge review
- 合并后下一步：`feat/chunking-runner`

## 建议下一阶段目标

- 检查外部 chunker repo clean 状态并调用真实 chunker
- 继续保持小 PR、不引入 Redis / SQL / Airflow
