# 进度报告

## 当前阶段

artifact store、S3 backend、dataset schema、MTEB adapter、chunking schema 已合并到 `main`。`feat/chunking-runner` 已完成本地实现与 pre-merge 修补，待 PR / merge review。

## 已完成事项（main）

- Local + S3 artifact store
- normalized dataset schema + JSONL artifact 读写
- MTEB dataset adapter
- chunked corpus schema + ChunkerProvenance + artifact IO

## 已完成事项（feat/chunking-runner）

- `inspect_git_repo` / `ensure_git_repo_clean`
- `ChunkingRunConfig` / `run_chunking` + injectable `ExternalChunker`
- dirty repo 安全边界、round-trip 与 config validation 测试
- ADR：`docs/decisions/0005-chunking-runner.md`
- `tests/chunking/test_git.py` / `tests/chunking/test_runner.py`

## 本 PR 范围

- 实现 chunking runner 框架与 git clean-state 检查
- 记录 external chunker commit provenance 到 chunked corpus manifest
- 不实现真实 sciverse adapter、embedding、ES/Milvus、retrieval、metrics

## 已验证事项

- 测试不访问网络或真实 S3
- 使用临时 git repo 与 fake chunker 做离线测试
- `pytest` / `ruff check .` 通过

## 当前结论

- chunking runner 已完成 pre-merge 修补，可进入 PR / merge review
- 合并后下一步：`feat/embedding-schema`

## 建议后续方向

- 定义 embedding schema 与 artifact 格式
- 后续可按需接入真实 external chunker adapter
