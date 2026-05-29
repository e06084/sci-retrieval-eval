# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`Track B1 验收返工 / asset fingerprint 语义修正`
- 当前分支：`feat/asset-fingerprint-spec`
- 对应指令文件：`TASK.md`
- 返工基线：`origin/feat/asset-fingerprint-spec` / `ecd6b33 Update asset fingerprint report`
- 返工完成时间：2026-05-30
- 返工实现提交 SHA：`6db94aee5d37fd2d23412c38198a9c21625cd98f`
- 报告提交 SHA：本报告提交后由 `git log -1 --oneline` 确认；提交内容无法自引用自身 SHA

## 2. 本轮返工内容

本轮仍只覆盖 B1 fingerprint schema/helper/spec/docs/tests，不接入 artifact writer，不改变
planner / runner 行为。

已修复：

- `src/eval_platform/assets/fingerprint.py`
  - 扩展运行实例字段 guard：`run_id`、`artifact_id`、`created_at`、`updated_at`、
    `started_at`、`completed_at`、`timestamp`、`created_time`、`updated_time`、
    `request_id`、`trace_file`、`trace_path` 等不允许进入 fingerprint payload。
  - 新增物理资源字段 guard，用于自由参数字典，拒绝 `index_name`、`collection_name`、
    `endpoint_url`、`url`、`uri`、`host`、`port`，以及 `_url` / `_uri` / `_host` /
    `_port` 后缀字段。
  - 保留稳定身份字段例外：`raw_source_uri`、`source_git_remote_url`、`endpoint_alias`。
  - `file_fingerprints` 在 raw dataset builder 中按 `path`、`sha256`、`size_bytes`
    canonical sort，避免对象存储 listing 顺序影响 raw dataset fingerprint。
- `tests/assets/test_fingerprint.py`
  - 新增 raw file listing 顺序不影响 fingerprint 的测试。
  - 新增 `path` / `sha256` / `size_bytes` 变化会改变 raw dataset fingerprint 的测试。
  - 新增自由参数字典拒绝物理资源名、真实 URL/URI、host/port、request id、trace path、
    timestamp 字段的测试。
  - 补齐 secret fragments：`api_key`、`access_key`、`secret`、`password`、`token`、
    `authorization`。
- `docs/decisions/0023-asset-fingerprint-spec.md`
  - 明确物理资源名、真实服务地址、request id、trace path、时间戳不进入 fingerprint。
  - 明确 `raw_source_uri`、`source_git_remote_url`、`endpoint_alias` 的允许边界。
  - 明确 `file_fingerprints` 是集合语义并 canonical sort。
- `docs/architecture.md`
  - 同步资产身份和等价性边界。
- `docs/operations/experiment_variants.md`
  - 同步实验变体和后续复用规划的参数边界。

## 3. Guard 语义

`canonical_json_hash(...)` 使用 canonical JSON：

```text
sort_keys=True
ensure_ascii=False
separators=(",", ":")
allow_nan=False
```

并拒绝：

- 非 JSON-serializable value。
- 非 string dict key。
- secret-like key。
- 运行实例身份和 timestamp 字段。

各 stage component builder 会在自由参数字典中额外拒绝物理连接信息。ES URL、Milvus URI、
ES index name、Milvus collection name 应记录在 artifact manifest metadata 或运行配置中，
不参与逻辑资产等价判断。

## 4. 验证结果

已运行：

```bash
PYTHONPATH=src pytest tests/assets/test_fingerprint.py
PYTHONPATH=src pytest
ruff check .
mypy .
```

结果：

- `PYTHONPATH=src pytest tests/assets/test_fingerprint.py`
  - `71 passed in 0.16s`
- `PYTHONPATH=src pytest`
  - `683 passed in 1.91s`
- `ruff check .`
  - `All checks passed!`
- `mypy .`
  - `Success: no issues found in 175 source files`

## 5. 外部服务访问

- 是否访问真实 S3：`no`
- 是否访问真实 ES：`no`
- 是否访问真实 Milvus：`no`
- 是否访问真实 embedding：`no`
- 是否访问真实 rerank：`no`
- 是否访问真实 rewrite：`no`

## 6. 未实现项

按 B1 / PR1 范围，本轮未实现：

- artifact writer 接入 `asset_fingerprint`。
- planner 行为变更。
- minimal rebuild planner。
- stage override。
- pinned artifacts。
- benchmark_run / benchmark_suite_run fingerprint。
- benchmark variant spec。
- 真实外部服务运行。

## 7. 后续建议

PR2：各 artifact writer / runner 将 `asset_fingerprint` 写入 manifest metadata。
PR3：reuse planner 增加 `complete + artifact_type + dependency-compatible chain + fingerprint match` 联合校验。
PR4：minimal rebuild planning、stage override、pinned artifacts 和 variant spec。
