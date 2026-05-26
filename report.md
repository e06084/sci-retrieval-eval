# 开发报告

本文件由开发 session 维护。

验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`corpus build runner v1`
- 当前分支：`feat/corpus-build-runner`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-26
- 完成时间：2026-05-26

## 2. 本次改动

- 新增 `src/eval_platform/corpus_build/runner.py`
  - 定义 `CORPUS_BUILD_ARTIFACT_TYPE = "corpus_build"`
  - 定义 `RawSourceSpec`
  - 定义 `CorpusBuildArtifactIds`
  - 定义 `CorpusBuildConfig`
  - 定义 `CorpusBuildError`
  - 实现 `default_corpus_build_artifact_ids(...)`
  - 实现 `run_corpus_build(...)`
- 新增 `src/eval_platform/corpus_build/__init__.py`
  - 导出 corpus build runner 公共 API
- 新增 `tests/corpus_build/test_runner.py`
  - 使用 local raw dir、fake chunker、fake embedding client、fake ES client、fake Milvus client 覆盖完整链路
  - 覆盖关闭 ES / Milvus、artifact id mismatch、阶段失败、progress failure 和 final manifest 不泄密
- 新增 ADR：
  - `docs/decisions/0017-corpus-build-runner.md`
- 更新：
  - `docs/ai/current_status.md`
  - `report.md`

## 3. 涉及文件

- `src/eval_platform/corpus_build/__init__.py`
- `src/eval_platform/corpus_build/runner.py`
- `tests/corpus_build/__init__.py`
- `tests/corpus_build/test_runner.py`
- `docs/decisions/0017-corpus-build-runner.md`
- `docs/ai/current_status.md`
- `report.md`

### 3.1 范围自检

- 是否改动了流程控制文档：`no`
- 如果是，改动理由：无
- 是否修改 ES / Milvus ingest 语义：`no`
- 是否实现 retrieval / metrics / frontend：`no`
- 是否扩展五个数据集 normalizer：`no`
- 单元测试是否访问真实 S3 / ES / Milvus / embedding API：`no`

## 4. 实现说明

### 4.1 阶段顺序

`run_corpus_build(...)` 固定串联以下阶段：

1. `raw_import`
2. `raw_to_normalized`
3. `chunking`
4. `embedding`
5. `elasticsearch_ingest`，可关闭
6. `milvus_ingest`，可关闭
7. 写 `corpus_build` final manifest

runner 不复制各阶段实现逻辑，只调用现有 API：

- `import_raw_dataset_from_local_dir(...)`
- `import_raw_dataset_from_s3_prefix(...)`
- `normalize_raw_dataset_artifact(...)`
- `run_chunking(...)`
- `run_embedding(...)`
- `run_elasticsearch_ingest(...)`
- `run_milvus_ingest(...)`

阶段之间只通过 artifact id 连接，不直接把内存对象传给下一阶段。

### 4.2 Artifact ID 规则

默认规则：

```text
raw_dataset:         <run_id>_raw
normalized_dataset:  <run_id>_normalized
chunked_corpus:      <run_id>_chunks
embeddings:          <run_id>_embeddings
elasticsearch_index: <run_id>_es_index
milvus_collection:   <run_id>_milvus_collection
corpus_build:        <run_id>
```

调用方可以显式传入 `CorpusBuildArtifactIds` 覆盖默认 id。

runner 会校验传入的 stage config artifact id 是否与最终 artifact id 一致。如果不一致，直接抛 `CorpusBuildError`，不做隐式覆盖。

### 4.3 Raw Source

V1 支持：

1. `local_dir`
2. `s3_prefix`

`local_dir` 使用：

```python
import_raw_dataset_from_local_dir(...)
```

`s3_prefix` 使用：

```python
import_raw_dataset_from_s3_prefix(...)
```

真实 S3 client 不由 runner 创建，必须由调用方注入。

### 4.4 Final Manifest

成功后写出：

```text
corpus_build/<run_id>/_MANIFEST.json
corpus_build/<run_id>/_SUCCESS
```

这类 artifact 没有 payload 文件，因此 `files=[]`。

Final manifest dependencies 包含所有成功且启用的 stage artifact：

1. `raw_dataset`
2. `normalized_dataset`
3. `chunked_corpus`
4. `embeddings`
5. `elasticsearch_index`，如果启用
6. `milvus_collection`，如果启用

Final manifest metadata 包含：

1. `run_id`
2. `dataset_name`
3. `raw_source`
4. `artifact_ids`
5. `enabled_stages`
6. `stage_manifests`

`stage_manifests` 只记录审计必要摘要，例如：

- 文件数量和 fingerprint
- normalized corpus / query / qrel count
- chunk / embedding count
- embedding model name / dim
- ES index name / mapping hash / indexed count
- Milvus collection name / schema hash / inserted count

不会把完整平台 config 塞进 final manifest。

### 4.5 进度汇报

runner 使用现有 `ProgressEvent`。

事件语义：

1. 每个阶段开始时：
   - `stage="corpus_build"`
   - `metadata.kind="stage_start"`
2. 每个阶段结束时：
   - `metadata.kind="stage_done"`
3. 子阶段 progress reporter 透传同一个 reporter。
4. final manifest 写入后、`_SUCCESS` 写入前：
   - `metadata.kind="run_done"`

这样如果 `run_done` reporter 抛异常，最终不会写 `_SUCCESS`。

### 4.6 失败路径

任一阶段失败时：

1. 抛 `CorpusBuildError`
2. 保留原始异常作为 `__cause__`
3. 不写最终 `corpus_build/_SUCCESS`
4. 不回滚已经成功写出的 stage artifact

已覆盖的失败路径：

1. raw import 失败
2. embedding 失败
3. ES 失败
4. Milvus 失败
5. progress reporter 失败
6. stage config artifact id mismatch

### 4.7 非目标

本轮不实现：

1. retrieval
2. metrics / reports
3. frontend
4. 五数据集 raw normalizer 扩展
5. CLI / scheduler
6. DAG / resume / retry
7. 真实外部服务测试
8. one-off scripts 迁移或删除

## 5. 自检结果

### 5.1 必跑命令

```bash
pytest tests/corpus_build tests/indexes tests/chunking/test_artifact.py tests/embeddings/test_artifact.py tests/datasets
ruff check .
mypy .
pytest
```

### 5.2 输出摘要

- `pytest tests/corpus_build tests/indexes tests/chunking/test_artifact.py tests/embeddings/test_artifact.py tests/datasets`：
  - 通过，`156 passed`
- `ruff check .`：
  - 通过
- `mypy .`：
  - 通过，`Success: no issues found in 107 source files`
- `pytest`：
  - 通过，`448 passed`

## 6. 风险与未决项

- 已知风险：
  - V1 只支持 IFIRNFCorpus。
  - runner 参数较多，后续 CLI / scheduler 接入时可能需要再包装一层 config builder。
  - 本轮没有真实 S3 / ES / Milvus / embedding API 集成测试。
- 未覆盖场景：
  - 不覆盖 retrieval / metrics。
  - 不覆盖五数据集 raw normalizer。
  - 不覆盖 resume / retry。
- 需要验收者重点检查的点：
  - 是否只通过 artifact id 连接阶段。
  - failure path 是否不会写最终 `corpus_build/_SUCCESS`。
  - final manifest 是否足以审计完整 corpus build。

## 7. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无

## 8. 提交信息

- 是否已提交：`yes`
- commit subject：`Add corpus build runner`
- 验收者确认的最终 commit：
