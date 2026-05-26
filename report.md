# 开发报告

本文件由开发 session 维护。

验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：新增 raw_dataset artifact
- 当前分支：`feat/raw-dataset-artifact`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-26
- 完成时间：2026-05-26

## 2. 本次改动

- 改了什么：
  - 新增 `src/eval_platform/datasets/raw.py`，定义 `raw_dataset` artifact 的 schema、manifest 字段和读写逻辑。
  - 新增 `src/eval_platform/datasets/raw_import.py`，支持：
    - 从本地目录导入 raw 文件元数据快照
    - 从既有 S3 prefix 导入 raw 文件元数据快照
  - 更新 `src/eval_platform/datasets/__init__.py`，导出 `raw_dataset` 相关公共接口。
  - 根据验收反馈，把第一版实现收紧为 snapshot-only：
    - 不再复制 raw 文件内容到 artifact store
    - artifact 内只写 `_MANIFEST.json` 和 `_SUCCESS`
  - 新增 `tests/datasets/test_raw_dataset.py`，覆盖 snapshot-only、本地目录导入、fake S3 source 导入、fake S3 output 和流式 body 场景。
  - 新增 ADR `docs/decisions/0011-raw-dataset-artifact.md`。
  - 修复 `tests/chunking/test_external_chunking_runner.py` 中不稳定的错误 commit SHA 构造，保证 mismatch 测试一定使用不同 SHA。
- 为什么这样改：
  - 当前 `normalized_dataset` 直接从内存对象开始，缺少“原始输入先落盘”的可审计层。
  - `raw_dataset` artifact 先固定原始文件身份，后续才能稳定建立 `raw_dataset -> normalized_dataset` 依赖。
  - 验收反馈指出之前的 `dict[str, bytes]` 聚合方案在大文件场景下仍会把 raw 内容积到内存里，不满足本任务目标，所以改成 manifest snapshot 设计。
- 没改什么：
  - 没有实现 `raw_dataset -> normalized_dataset` 自动转换。
  - 没有实现 chunk / embedding / ES / Milvus / retrieval / metrics。
  - 没有访问真实外部服务或真实 S3。

## 3. 涉及文件

- `src/eval_platform/datasets/__init__.py`
- `src/eval_platform/datasets/raw.py`
- `src/eval_platform/datasets/raw_import.py`
- `tests/datasets/test_raw_dataset.py`
- `tests/chunking/test_external_chunking_runner.py`
- `docs/decisions/0011-raw-dataset-artifact.md`
- `docs/ai/current_status.md`
- `report.md`

### 3.1 范围自检

- 是否改动了流程控制文档：`no`
- 如果是，改动理由：无

## 4. 实现说明

### 4.1 关键决策

- 决策 1：
  - `raw_dataset` 放在 `datasets/` 下，而不是 `artifacts/` 下。
  - 理由是这层仍属于“数据输入身份”，和 `normalized_dataset` 属于同一数据域。
- 决策 2：
  - 第一版 `raw_dataset` 采用 snapshot-only 设计，不复制 raw 文件内容。
  - 理由是已知 raw source（尤其 `ifir_scifact/corpus.jsonl` 约 745MB）不适合通过 `dict[str, bytes]` 聚合后再写入。
- 决策 3：
  - manifest metadata 中的系统字段统一由写入逻辑最后覆盖，避免用户 metadata 覆盖 `stage / file_count / content_fingerprint_sha256` 等关键信息。

### 4.2 关键行为

- 行为 1：
  - `raw_dataset` manifest metadata 至少包含：
    - `stage`
    - `source_type`
    - `source_uri`
    - `dataset_name`
    - `dataset_revision`
    - `file_count`
    - `total_size_bytes`
    - `files`
    - `content_fingerprint_sha256`
    - `import_parameters`
- 行为 2：
  - `RawDatasetFile` 记录：
    - `path`
    - `uri`
    - `size_bytes`
    - `sha256`
- 行为 3：
  - 单文件 `sha256` 通过流式读取计算：
    - 按固定 chunk 大小逐块 `read(...)`
    - 每块增量更新 `hashlib.sha256()`
    - 不在 hash 阶段一次性整文件读入内存
- 行为 4：
  - dataset 级 `content_fingerprint_sha256` 按稳定排序后的 `(path, uri, size_bytes, sha256)` 序列计算：
    - `path<TAB>uri<TAB>size<TAB>sha256<LF>`
    - 再整体做 `sha256`
  - 这样文件内容不变且路径/顺序稳定时，fingerprint 可复现。
- 行为 5：
  - snapshot-only artifact 不会写 `files/...`，只写：
    - `_MANIFEST.json`
    - `_SUCCESS`

## 5. 自检结果

### 5.1 必跑命令

```bash
git status --short
git diff --name-only origin/main...HEAD
pytest tests/datasets tests/artifacts
ruff check .
mypy .
```

### 5.2 输出摘要

- `git status --short`：
  - 开发完成前仅包含 `src/eval_platform/datasets/`、`tests/datasets/`、ADR、`current_status.md` 与 `report.md` 改动。
- `git diff --name-only origin/main...HEAD`：
  - 本轮提交前工作区只涉及：
    - `src/eval_platform/datasets/`
    - `tests/datasets/`
    - `tests/chunking/test_external_chunking_runner.py`
    - `docs/decisions/0011-raw-dataset-artifact.md`
    - `docs/ai/current_status.md`
    - `report.md`
  - `tests/chunking/test_external_chunking_runner.py` 为基线测试稳定性修复，不改生产 `chunking/` 代码。
  - 不包含 `src/eval_platform/chunking/`、`embeddings/`、`retrieval/`、`metrics/` 等越界生产目录。
- `pytest tests/datasets tests/artifacts`：
  - 通过，`89 passed`
- `pytest`：
  - 通过，`336 passed`
- `ruff check .`：
  - 通过
- `mypy .`：
  - 通过，`Success: no issues found in 82 source files`

### 5.3 提交信息

- 是否已提交：`yes`
- commit subjects：
  - `Add raw dataset artifact`
  - `Switch raw dataset artifact to snapshot-only`
  - `Stabilize external chunker SHA mismatch test`
- 验收者确认的最终 commit：

## 6. 风险与未决项

- 已知风险：
  - 第一版是 snapshot-only，不提供 raw 副本 materialization；如果后续需要“导入时顺手复制 raw 数据”，需要单独扩展流式写入能力。
- 未覆盖场景：
  - 本轮没有实现后续 `raw_dataset -> normalized_dataset` 依赖连接。
- 需要验收者重点检查的点：
  - `content_fingerprint_sha256` 的定义是否满足后续复现实验要求。
  - snapshot-only 语义是否与后续 normalizer 设计一致。

## 7. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：
  - 当前全量 `pytest` 存在一条与本任务无关的 `chunking` 基线失败，需要验收方按项目当前基线判断是否阻塞
