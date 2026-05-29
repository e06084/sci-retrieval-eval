# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`PR2 / artifact writer 接入 asset_fingerprint`
- 当前分支：`feat/artifact-fingerprint-writers`
- 基线：`origin/main` / `268cfbc Add asset fingerprint foundations (#39)`
- 完成时间：2026-05-30

## 2. 本轮实现

本轮把 #39 中定义的 fingerprint schema/helper 接入 artifact manifest 写入路径。

新增：

- `src/eval_platform/assets/manifest.py`
  - `asset_fingerprint_metadata(...)`
  - `add_asset_fingerprint_metadata(...)`
  - `manifest_asset_fingerprint_sha256(...)`
  - `require_manifest_asset_fingerprint_sha256(...)`
  - `strip_asset_fingerprint_metadata(...)`
- manifest metadata key：
  - `asset_fingerprint`
  - `asset_fingerprint_sha256`

已接入的 artifact：

- `raw_dataset`
- `normalized_dataset`
- `chunked_corpus`
- `embeddings`
- `elasticsearch_index`
- `milvus_collection`
- `retrieval_run`
- `metrics_run`

语义边界：

- writer/runner 在语义字段足够时写入 `metadata.asset_fingerprint` 和
  `metadata.asset_fingerprint_sha256`。
- runner 能读上游 manifest 时，使用上游 `asset_fingerprint_sha256` 作为依赖身份。
- 低层 writer 如果缺少必要语义字段，不强造 fingerprint，避免写入错误身份。
- `index_name`、`collection_name`、真实服务 URL、时间戳、`artifact_id`、`run_id`
  不进入 fingerprint。

## 3. 测试覆盖

新增：

- `tests/assets/test_manifest_fingerprints.py`

覆盖：

- `raw_dataset` 的 fingerprint 不受 artifact id / created_at 影响。
- `normalized_dataset`、`chunked_corpus`、`embeddings` manifest 写入 fingerprint。
- `retrieval_run` fingerprint 使用 normalized / ES / Milvus 逻辑资产身份，不受
  `index_name` / `collection_name` 变化影响。
- `metrics_run` fingerprint 使用 normalized dataset fingerprint 与 retrieval run fingerprint。

## 4. 验证结果

已运行：

```bash
PYTHONPATH=src pytest tests/assets/test_manifest_fingerprints.py
PYTHONPATH=src pytest
ruff check .
mypy .
```

结果：

- `PYTHONPATH=src pytest tests/assets/test_manifest_fingerprints.py`
  - `3 passed`
- `PYTHONPATH=src pytest`
  - `686 passed`
- `ruff check .`
  - `All checks passed!`
- `mypy .`
  - `Success: no issues found in 177 source files`

## 5. 未实现项

按 PR2 范围，本轮未实现：

- planner/reuse 使用 fingerprint 做复用判断。
- minimal rebuild planning。
- stage override / pinned artifacts。
- benchmark_run / benchmark_suite_run fingerprint。
- 真实五数据集资产重建。

下一步是 PR3：让 corpus asset reuse planner 使用 `asset_fingerprint_sha256` 做复用校验。
