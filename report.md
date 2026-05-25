# 进度报告

## 当前阶段

artifact store、S3 backend、dataset schema、MTEB adapter、chunking schema、chunking runner 已合并到 `main`。当前分支只收敛一个小型修复 PR：新版 MTEB task layout 兼容。

## 已完成事项（main）

- Local + S3 artifact store
- normalized dataset schema + JSONL artifact 读写
- MTEB dataset adapter
- chunked corpus schema + ChunkerProvenance + artifact IO
- `inspect_git_repo` / `ensure_git_repo_clean`
- `ChunkingRunConfig` / `run_chunking` + injectable `ExternalChunker`
- dirty repo 安全边界、round-trip 与 config validation 测试
- ADR：`docs/decisions/0005-chunking-runner.md`
- `tests/chunking/test_git.py` / `tests/chunking/test_runner.py`

## 本次修复

- 修复 `mteb_adapter` 对新版 MTEB task layout 的兼容问题
- 支持从 `task.dataset[subset][split]` 提取 `corpus / queries / relevant_docs / qrels`
- 支持将 dataset-like rows 转成内部 mapping
  - 支持 `id`
  - 支持 `_id`
  - 转换后移除 `id` / `_id`
- 保持原有 `task.corpus / task.queries / task.relevant_docs / task.qrels` 路径兼容
- 补充 `tests/mteb_adapter/test_load.py` 对新版数据布局的覆盖

## 已验证事项

- `pytest tests/mteb_adapter/test_load.py` 通过
- `ruff check src/eval_platform/mteb_adapter/load.py tests/mteb_adapter/test_load.py` 通过
- `mypy src/eval_platform/mteb_adapter/load.py tests/mteb_adapter/test_load.py` 通过
- 真实 `mteb.load_data()` 的 `IFIRNFCorpus` smoke 已跑通
- 兼容修复后，`normalized_dataset -> fake chunked_corpus` 的本地链路可继续跑通

## Smoke 结果

- run id: `smoke_retry_ifirnf2`
- normalized artifact:
  - `test_mteb_ifirnfcorpus_test_smoke_retry_ifirnf2`
  - `corpus=3633`
  - `queries=86`
  - `qrels=242`
- chunked artifact:
  - `test_mteb_ifirnfcorpus_test_smoke_retry_ifirnf2_fake_chunks`
  - `chunks=3633`
- chunked manifest 记录了：
  - source dependency -> normalized artifact
  - fake chunker git commit sha
  - `is_dirty=false`
  - `chunk_params`

## 未合并内容

- task-specific smoke CLI 未合并到主线
- 如需正式 smoke runner，应后续单独设计通用命令，例如 `evalctl smoke ...`

## 建议后续方向

- 定义 embedding schema 与 artifact 格式
- 下一步：`feat/embedding-schema`
