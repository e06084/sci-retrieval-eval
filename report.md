# 开发报告

本文件由开发 session 维护。

验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：开发 raw_dataset -> normalized_dataset
- 当前分支：`feat/raw-to-normalized`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-26
- 完成时间：2026-05-26

## 2. 本次改动

- 改了什么：
  - 新增 `src/eval_platform/datasets/raw_normalize.py`
    - 定义 `RawToNormalizedConfig`
    - 定义 `RawFileOpener` protocol
    - 实现 `S3RawFileOpener`
    - 实现 `normalize_raw_dataset_artifact(...)`
    - 第一版支持 `IFIRNFCorpus`
  - 扩展 `src/eval_platform/datasets/normalized.py`
    - `write_normalized_dataset_artifact(...)` 新增可选 `dependencies`
    - 保持旧调用方兼容
  - 更新 `src/eval_platform/datasets/__init__.py`
    - 导出 raw-to-normalized 公共接口
  - 新增 `tests/datasets/test_raw_normalize.py`
    - 覆盖 raw snapshot -> normalized artifact
    - 覆盖 dependency
    - 覆盖 query instruction metadata
    - 覆盖流式 opener 约束
  - 扩展 `tests/datasets/test_normalized_dataset.py`
    - 验证 `write_normalized_dataset_artifact(...)` 传 dependency 时兼容
  - 新增 ADR `docs/decisions/0012-raw-to-normalized-dataset.md`
- 为什么这样改：
  - `raw_dataset` 已经是可信输入快照，但 `normalized_dataset` 还没有显式绑定 raw 上游身份。
  - 本轮把 `normalized_dataset` 明确变成 `raw_dataset` 的标准化产物，并保证 raw source 的打开方式可注入、可测试。
- 没改什么：
  - 没有实现 runner / orchestration
  - 没有实现 chunk / embedding / ES / Milvus / retrieval / metrics
  - 没有访问真实外部服务

## 3. 涉及文件

- `src/eval_platform/datasets/__init__.py`
- `src/eval_platform/datasets/normalized.py`
- `src/eval_platform/datasets/raw_normalize.py`
- `tests/datasets/test_normalized_dataset.py`
- `tests/datasets/test_raw_normalize.py`
- `docs/decisions/0012-raw-to-normalized-dataset.md`
- `docs/ai/current_status.md`
- `report.md`

### 3.1 范围自检

- 是否改动了流程控制文档：`no`
- 如果是，改动理由：无

## 4. 实现说明

### 4.1 raw-to-normalized API

新增入口：

```python
normalize_raw_dataset_artifact(
    source_store,
    output_store,
    RawToNormalizedConfig(...),
    opener=...,
)
```

含义：

- 输入：`raw_dataset` artifact
- 输出：`normalized_dataset` artifact
- `source_store` 与 `output_store` 可以不同
- `opener` 负责按 `RawDatasetFile.uri` 打开原始文件流
- 当前内置生产 opener：
  - `S3RawFileOpener`
  - 默认用 boto3 client 打开 `s3://bucket/key`
  - 测试使用 fake S3 client，不访问真实 S3

### 4.2 IFIRNFCorpus 字段映射

第一版只支持：

- `dataset_name == "IFIRNFCorpus"`
- 或 `normalizer_name == "ifir_nfcorpus_raw_jsonl_tsv_v1"`

映射规则：

- `corpus.jsonl`
  - `_id -> CorpusRecord.doc_id`
  - `title -> CorpusRecord.title`
  - `text -> CorpusRecord.text`
- `queries.jsonl`
  - `_id -> QueryRecord.query_id`
  - `text -> QueryRecord.text`
- `instructions.jsonl`
  - `query-id -> QueryRecord.metadata["instruction"]`
- `qrels/test.tsv`
  - `query-id -> QrelRecord.query_id`
  - `corpus-id -> QrelRecord.doc_id`
  - `score -> QrelRecord.relevance`

### 4.3 dependency 写入方式

`write_normalized_dataset_artifact(...)` 新增：

```python
dependencies: list[ArtifactDependency] | None = None
```

raw-to-normalized 调用时会写入：

- `artifact_type = "raw_dataset"`
- `artifact_id = config.source_artifact_id`

同时 normalized manifest metadata 至少写入：

- `source = "raw_dataset"`
- `task_name`
- `split`
- `normalizer_name`
- `raw_dataset_artifact_id`
- `raw_dataset_fingerprint`
- `raw_source_uri`
- `normalized_schema_version`

### 4.4 streaming 如何保证

本轮没有通过 `mteb.load_data()` 走内存对象。

`IFIRNFCorpus` 的 raw 文件读取方式：

- JSONL：
  - 通过 `for raw_line in stream` 逐行解析
  - 不做 `body.read().decode()`
- TSV：
  - 通过 `csv.DictReader(...)` 读取
  - 测试样本里规模很小

测试中使用 `FakeRawFileOpener` + `RecordingBinaryStream` 验证：

- corpus JSONL 确实通过迭代逐行读取
- 不会对 corpus stream 调用 `read(-1)` 一次性整文件读取

## 5. 自检结果

### 5.1 必跑命令

```bash
git status --short
git diff --name-only origin/main...HEAD
pytest tests/datasets tests/artifacts
ruff check .
mypy .
pytest
```

### 5.2 输出摘要

- `git status --short`：
  - 开发完成前仅包含 `src/eval_platform/datasets/`、`tests/datasets/`、ADR、`current_status.md` 与 `report.md` 改动
- `git diff --name-only origin/main...HEAD`：
  - 只涉及允许范围内文件
  - 不包含 `chunking/`、`embeddings/`、`indexes/`、`retrieval/`、`metrics/`
- `pytest tests/datasets tests/artifacts`：
  - 通过，`93 passed`
- `ruff check .`：
  - 通过
- `mypy .`：
  - 通过，`Success: no issues found in 84 source files`
- `pytest`：
  - 通过，`340 passed`

### 5.3 提交信息

- 是否已提交：`yes`
- commit subjects：
  - `Add raw to normalized dataset adapter`
  - `Add S3 raw file opener`
- 验收者确认的最终 commit：

## 6. 风险与未决项

- 已知风险：
  - 第一版只覆盖 `IFIRNFCorpus`，其余 raw asset 仍需补 dataset-specific raw normalizer
  - 当前 writer 仍要求先构造内存中的 `NormalizedDataset`；在支持 `ifir_scifact` 这类更大 corpus 前，还需要 streaming normalized writer 或分批 writer
- 未覆盖场景：
  - 还没有批量 runner / orchestration
- 需要验收者重点检查的点：
  - `raw_source_uri` 的定义是否足够稳定
  - `normalizer_name` 的命名是否满足后续扩展

## 7. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无
