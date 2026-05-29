# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`PR3 / corpus asset reuse planner 使用 asset_fingerprint`
- 当前分支：`feat/corpus-asset-fingerprint-reuse`
- 基线：`feat/artifact-fingerprint-writers`，且 PR #40 已合入 `main`
- 完成时间：2026-05-30

## 2. 本轮实现

本轮让五数据集 corpus asset dry-run planner 能按逻辑资产 fingerprint 做复用过滤。

改动：

- `inventory_corpus_assets(...)`
  - manifest summary 增加 `asset_fingerprint_sha256`。
- `build_plan_for_datasets(...)`
  - 新增可选参数 `expected_asset_fingerprints_by_slug`。
  - 当提供 expected fingerprint 时，只复用对应 artifact type 中
    `metadata_summary.asset_fingerprint_sha256` 匹配的完整 artifact。
  - 未提供 expected fingerprint 时，保持原有 dependency-chain reuse 行为。
  - reused step 会暴露 `asset_fingerprint_sha256`，便于审计 dry-run 选择原因。

## 3. 行为语义

新增能力支持后续最小重建：

- 如果 raw / normalized / chunk / ES fingerprint 匹配，但 embedding fingerprint 变化：
  - 复用 raw / normalized / chunk / ES。
  - 重新创建 embeddings。
  - 重新创建 Milvus collection。
- 如果存在多个 embeddings 或 Milvus collection，planner 会优先选择 fingerprint 匹配且
  dependency chain 一致的那条链。
- 如果 expected fingerprint 指向的 artifact 不存在或旧 artifact 没有 fingerprint：
  - 不复用该 artifact。
  - planner 回退为 create 对应阶段和依赖它的下游阶段。

## 4. 测试覆盖

新增/更新：

- `tests/corpus_assets/test_planner.py`

覆盖：

- expected fingerprint 能从同一 chunk chain 中选择目标 embeddings / Milvus。
- embedding fingerprint 变化时，只复用 raw / normalized / chunk / ES，并重建
  embeddings / Milvus。
- 原有不传 expected fingerprint 的 reuse 行为保持不变。

## 5. 验证结果

已运行：

```bash
PYTHONPATH=src pytest tests/corpus_assets/test_planner.py tests/corpus_assets/test_inventory.py tests/scripts/test_build_real_corpus_assets.py
PYTHONPATH=src pytest tests/corpus_assets tests/scripts/test_build_real_corpus_assets.py tests/assets/test_manifest_fingerprints.py
ruff check .
mypy .
```

结果：

- `tests/corpus_assets/test_planner.py tests/corpus_assets/test_inventory.py tests/scripts/test_build_real_corpus_assets.py`
  - `16 passed`
- `tests/corpus_assets tests/scripts/test_build_real_corpus_assets.py tests/assets/test_manifest_fingerprints.py`
  - `25 passed`
- `ruff check .`
  - `All checks passed!`
- `mypy .`
  - `Success: no issues found in 177 source files`

## 6. 未实现项

按 PR3 范围，本轮未实现：

- expected fingerprint 的自动计算入口。
- minimal rebuild execute runner。
- stage override / pinned artifacts。
- 真实五数据集资产重建。

下一步是把 PR3 合入后，用最新 `main` 重新生成 5 个 benchmark 的 corpus assets。
