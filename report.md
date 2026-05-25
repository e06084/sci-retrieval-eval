# 进度报告

## 当前阶段

`main` 已具备 artifact store、S3 backend、dataset schema、MTEB adapter、chunking schema 和 chunking runner。当前分支上的 version-pinned external chunker adapter / Sciverse admin-ingest adapter 已完成本地实现，并已通过真实 `MTEB -> normalized_dataset -> real sciverse chunk -> S3` smoke，待 PR / merge review。

## 已完成事项（main）

- Local + S3 artifact store
- normalized dataset schema + JSONL artifact 读写
- MTEB dataset adapter
- chunked corpus schema + `ChunkerProvenance` + artifact IO
- `inspect_git_repo` / `ensure_git_repo_clean`
- `ChunkingRunConfig` / `run_chunking` + injectable `ExternalChunker`
- MTEB 新 layout 兼容修复
- 真实 `IFIRNFCorpus` MTEB -> normalized_dataset -> fake chunk 本地/S3 smoke 验证

## 本次开发

- 保留并复用现有 version-pinned external repo 校验：
  - remote URL
  - commit SHA
  - clean state
- 新增 `SciverseAdminIngestChunkerConfig`
- 新增 `SciverseAdminIngestExternalChunker`
- 新增 `run_version_pinned_sciverse_chunking(...)`
- 通过动态导入 `<repo>/python_services/admin-ingest`，把外部 repo 的 `chunk_ndjson_records(...)` 接入当前 `run_chunking(...)`
- 修复真实 `sciverse_clean` 兼容问题：
  - `source_type` / `file_ext` 不再传给 `RecursiveChunkOptions`
  - 这两个字段仍保留在 manifest `chunk_params` 中，作为运行语义记录
- 保持 provenance / dependency / manifest 统一收口
- 不实现 embedding / ES / Milvus / retrieval / metrics

## 已验证事项

- `pytest tests/chunking/test_external_repo.py tests/chunking/test_external_adapter.py tests/chunking/test_external_chunking_runner.py tests/chunking/test_sciverse_adapter.py` 通过
- `ruff check src/eval_platform/chunking tests/chunking` 通过
- `mypy src/eval_platform/chunking tests/chunking` 通过
- 新增 fake `sciverse` repo 测试覆盖：
  - 动态导入 `python_services/admin-ingest`
  - `NormalizedDataset -> NDJSON -> chunk_ndjson_records -> ChunkRecord`
  - version-pinned helper 写出 `chunked_corpus` artifact
  - dependency、repo provenance、chunk params 正确落入 manifest
- 通用 `PythonCallableExternalChunker` 已补 `sys.modules` 隔离：
  - 不同 repo 的同名 module 不再串用
  - 缺失 module 错误路径已覆盖

## 真实 smoke 结果

- 使用配置文件：
  - `/home/qiujiuantao/codex_project/sci-base/sciverse_benchmark/config.yaml`
- 使用真实外部 repo：
  - `/home/qiujiuantao/codex_project/sci-base/sciverse_clean/agentic-search`
- 外部 repo 版本：
  - remote URL: `git@gitlab.shlab.tech:sciverse/sciverse.git`
  - commit SHA: `e0b52937a59ce93162466e9178a5deda4c8800b5`
  - branch: `chunk_exp`
  - clean state: `true`
- 本次测试前缀：
  - `s3://scibase-service/sciverse_benchmark/test/sci-retrieval-eval/20260525_155519_sciverse_path_smoke/`

### SciFact

- normalized counts:
  - corpus: `5183`
  - queries: `300`
  - qrels: `339`
- chunked count:
  - `16120`
- artifact 完整性：
  - `normalized_dataset`: complete
  - `chunked_corpus`: complete
- 与 `sciverse_benchmark` 历史结果对比：
  - formatted doc_count: `5183`
  - chunk output_chunk_count: `16120`
  - 结论：完全一致
- 历史对比 manifest：
  - `sciverse_benchmark/corpus/scifact/formatted/manifest.json`
  - `sciverse_benchmark/corpus/scifact/chunks/e0b529_704f06/manifest.json`

### IFIRNFCorpus

- normalized counts:
  - corpus: `3633`
  - queries: `86`
  - qrels: `242`
- chunked count:
  - `11962`
- artifact 完整性：
  - `normalized_dataset`: complete
  - `chunked_corpus`: complete
- 与 `sciverse_benchmark` 历史结果对比：
  - formatted doc_count: `3633`
  - chunk output_chunk_count: `11958`
  - 结论：doc 数一致，chunk 数多 `4`
- 对比解释：
  - 历史 `sciverse_benchmark` chunk manifest 使用的外部 commit 是 `820da4242ee72c85bd8eb2d77ba11889a2b833ce`
  - 本次 smoke 使用的外部 commit 是 `e0b52937a59ce93162466e9178a5deda4c8800b5`
  - 因此这更像 external chunker 版本差异，不是当前 `sci-retrieval-eval` 流程错误
- 历史对比 manifest：
  - `sciverse_benchmark/corpus/nfcorpus__ifir_candidate/formatted/manifest.json`
  - `sciverse_benchmark/corpus/nfcorpus__ifir_candidate/chunks/820da4_704f06/manifest.json`

## 当前限制

- 仍未直接提供用户界面的 `SCIVERSE_PATH` 命令入口；当前是库级 adapter
- 不自动 `git fetch` 或 `git checkout`
- 用户必须事先准备好正确的外部 repo checkout

## 建议后续方向

- 合并这条 `sciverse` adapter 分支
- 然后开始 `feat/embedding-schema`
