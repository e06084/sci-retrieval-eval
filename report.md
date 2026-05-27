# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`five dataset corpus assets`
- 当前分支：`feat/five-dataset-corpus-assets`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-27
- 完成时间：2026-05-27

## 2. 本次改动

- 合入最新 `main`，同步 #30 文档整理基线。
- 新增共享脚本模块：
  - `scripts/corpus_asset_common.py`
- 新增只读 inventory 入口：
  - `scripts/inventory_real_corpus_assets.py`
- 新增 dry-run build plan 入口：
  - `scripts/build_real_corpus_assets.py`
- 新增操作文档：
  - `docs/operations/five_dataset_corpus_assets.md`
- 新增测试：
  - `tests/scripts/test_inventory_real_corpus_assets.py`
  - `tests/scripts/test_build_real_corpus_assets.py`
- 更新：
  - `report.md`

### 2.1 返工修复

- 修复 `--reuse-existing` 时下游依赖不一致的问题。
- `build_plan_for_datasets(...)` 现在同时输出：
  - `generated_artifact_ids`：当前 `run_id` 会生成的 artifact ids。
  - `resolved_artifact_ids`：后续阶段实际应该消费的 artifact ids。
  - `artifact_ids`：兼容旧输出，语义等同 `generated_artifact_ids`。
- 当复用已有 complete `chunked_corpus` / `embeddings` artifact 时：
  - ES 阶段 `source_artifact_id` 指向复用的 chunks artifact。
  - Milvus 阶段 `chunked_corpus_artifact_id` 指向复用的 chunks artifact。
  - Milvus 阶段 `embeddings_artifact_id` 指向复用的 embeddings artifact。

### 2.2 二次返工修复

- 修复 `--reuse-existing` 每个 stage 独立贪心选择 complete artifact 导致混链的问题。
- 复用选择现在从 inventory manifest dependency / metadata 构建轻量依赖链，优先选择最下游自洽链：
  - `milvus_collection -> embeddings + chunked_corpus -> normalized_dataset -> raw_dataset`
  - `elasticsearch_index -> chunked_corpus -> normalized_dataset -> raw_dataset`
- 如果某个 artifact 无法证明属于同一条链，不会静默复用；对应 stage 保持 `action=create`。
- 复用 ES / Milvus artifact 时，step 中的依赖字段来自该 artifact manifest，而不是重新用其它 stage 的 resolved id 推导。

### 2.3 三次返工修复

- 修复 `--reuse-existing` 复用已有 ES / Milvus artifact 时资源名仍使用当前新 `run_id` 的问题。
- `build_plan_for_datasets(...)` 现在同时输出：
  - `generated_resource_names`：当前 `run_id` 创建新资源时使用的 ES / Milvus 名称。
  - `resolved_resource_names`：后续执行实际应该使用的 ES / Milvus 名称。
  - `elasticsearch_index_name` / `milvus_collection_name`：兼容旧输出，语义等同 resolved resource names。
- 当 `elasticsearch_index` step 为 `action=reuse` 时，`index_name` 必须来自 reused artifact manifest 的 `metadata_summary["index_name"]`。
- 当 `milvus_collection` step 为 `action=reuse` 时，`collection_name` 必须来自 reused artifact manifest 的 `metadata_summary["collection_name"]`。
- 如果 reused ES / Milvus artifact 缺少对应资源名，规划阶段抛出 `CorpusAssetError`，不再静默回退到新 `run_id` 生成名。

## 3. 真实 raw 数据格式依据

已参考：

- `src/eval_platform/datasets/raw_normalize.py`
- `src/eval_platform/mteb_adapter/normalizers/`
- `/home/qiujiuantao/codex_project/sci-base/sciverse_benchmark/format_scripts/`

五个目标数据集映射：

| Dataset | slug | raw format | expected raw files |
|---|---|---|---|
| IFIRNFCorpus | `ifir_nfcorpus` | `jsonl_tsv` | `corpus.jsonl`, `queries.jsonl`, `instructions.jsonl`, `qrels/test.tsv` |
| NFCorpus | `nfcorpus` | `jsonl_tsv` | `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` |
| IFIRScifact | `ifir_scifact` | `jsonl_tsv` | `corpus.jsonl`, `queries.jsonl`, `instructions.jsonl`, `qrels/test.tsv` |
| SciFact | `scifact` | `jsonl_tsv` | `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv` |
| LitSearchRetrieval | `litsearch` | `parquet_dir_shards` | `corpus/test-00000-of-00001.parquet`, `queries/test-00000-of-00001.parquet`, `qrels/test-00000-of-00001.parquet` |

真实只读 S3 inventory 确认这 5 个 immutable raw prefix 均存在：

```text
s3://scibase-service/sciverse_benchmark/raw/ifir_nfcorpus
s3://scibase-service/sciverse_benchmark/raw/nfcorpus
s3://scibase-service/sciverse_benchmark/raw/ifir_scifact
s3://scibase-service/sciverse_benchmark/raw/scifact
s3://scibase-service/sciverse_benchmark/raw/litsearch
```

## 4. Artifact 命名和构建计划

新增统一命名：

```text
<dataset_slug>_<run_id>_raw
<dataset_slug>_<run_id>_normalized
<dataset_slug>_<run_id>_chunks
<dataset_slug>_<run_id>_embeddings
<dataset_slug>_<run_id>_es_index
<dataset_slug>_<run_id>_milvus_collection
```

ES / Milvus 目标名：

```text
<dataset_slug>_<run_id>_es
<dataset_slug>_<run_id>_milvus
```

这些 generated resource names 只适用于 `action=create`。`action=reuse` 时，ES
`index_name` 和 Milvus `collection_name` 来自 reused artifact manifest；缺失资源名会明确失败。

构建计划阶段顺序：

```text
raw_dataset
normalized_dataset
chunked_corpus
embeddings
elasticsearch_index
milvus_collection
```

`scripts/build_real_corpus_assets.py` 默认 dry-run，只输出计划和 artifact ids，不写 S3，不调用 ES/Milvus/embedding。

`--execute` 当前显式拒绝执行。原因是现有真实运行还需要显式传入 chunker、embedding、ES、Milvus runtime clients；本轮先固定五数据集资产命名、依赖和 inventory，不引入第二套真实执行路径。

`--reuse-existing` 下，计划会区分 generated ids 和 resolved ids：

- `generated_artifact_ids` 始终表示当前 `run_id` 对应的新 artifact ids。
- `resolved_artifact_ids` 表示真实执行时后续阶段应消费的 ids。
- `generated_resource_names` 始终表示当前 `run_id` 创建新 ES/Milvus 资源时的名字。
- `resolved_resource_names` 表示真实执行时后续阶段应使用的 ES/Milvus 资源名。
- 只有 dependency / metadata 能证明属于同一条链的 complete artifact 才会被复用。
- ES/Milvus 复用步骤直接使用自身 manifest 记录的依赖字段和资源名，不再混用其它链路 artifact，也不再误指向新 `run_id` 资源名。

## 5. 当前 S3 Inventory 摘要

命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
python scripts/inventory_real_corpus_assets.py \
  --config /home/qiujiuantao/codex_project/sci-base/sciverse_benchmark/config.yaml \
  --s3-prefix test_sciverse_benchmark \
  --raw-prefix sciverse_benchmark/raw
```

结果摘要：

| Dataset | raw prefix | raw_dataset | normalized | chunks | embeddings | ES index | Milvus collection |
|---|---|---:|---:|---:|---:|---:|---:|
| IFIRNFCorpus | exists | complete | complete | complete | complete | complete | complete |
| NFCorpus | exists | missing | missing | missing | missing | missing | missing |
| IFIRScifact | exists | missing | missing | missing | missing | missing | missing |
| SciFact | exists | missing | missing | missing | missing | missing | missing |
| LitSearchRetrieval | exists | missing | missing | missing | missing | missing | missing |

IFIRNFCorpus 现有完整主链路 artifact：

```text
raw_dataset:         ifir_nfcorpus_full_20260526_1945_raw
normalized_dataset:  ifir_nfcorpus_full_20260526_1945_normalized
chunked_corpus:      ifir_nfcorpus_full_20260526_1945_chunks
embeddings:          ifir_nfcorpus_full_20260526_1945_embeddings
elasticsearch_index: ifir_nfcorpus_real_ingest_20260526_220102_es_index
milvus_collection:   ifir_nfcorpus_real_ingest_20260526_220102_milvus_collection
```

核心计数：

```text
IFIRNFCorpus normalized: corpus=3633, queries=86, qrels=242
IFIRNFCorpus chunks: chunk_count=11962, unique_doc_count=3633
IFIRNFCorpus embeddings: embedding_count=11962, dim=1024
IFIRNFCorpus ES: indexed_count=11962, failed_count=0, verified_document_count=11962
IFIRNFCorpus Milvus: inserted_count=11962, failed_count=0, verified_entity_count=11962
```

## 6. Dry-run Build Plan 检查

命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
python scripts/build_real_corpus_assets.py \
  --config /home/qiujiuantao/codex_project/sci-base/sciverse_benchmark/config.yaml \
  --dataset SciFact \
  --run-id five_ds_20260527_dryrun \
  --s3-prefix test_sciverse_benchmark \
  --raw-prefix sciverse_benchmark/raw \
  --dry-run
```

结果：

- 成功输出 dry-run plan。
- 未写 S3。
- 未调用 ES / Milvus / embedding。
- SciFact artifact ids 符合命名规则：
  - `scifact_five_ds_20260527_dryrun_raw`
  - `scifact_five_ds_20260527_dryrun_normalized`
  - `scifact_five_ds_20260527_dryrun_chunks`
  - `scifact_five_ds_20260527_dryrun_embeddings`
  - `scifact_five_ds_20260527_dryrun_es_index`
  - `scifact_five_ds_20260527_dryrun_milvus_collection`

## 7. 测试覆盖

新增测试覆盖：

- dataset name / slug 映射。
- artifact id 命名稳定。
- build plan 阶段顺序为 raw -> normalized -> chunks -> embeddings -> ES -> Milvus。
- dry-run plan 不包含外部 clients / secrets。
- reuse-existing 时 resolved ids 会传递到 ES / Milvus 依赖字段。
- reuse-existing 在多条 complete 链并存时选择依赖自洽链，不混用 depcheck 小链和 full 主链。
- reuse-existing 的 ES / Milvus 依赖字段来自被复用 artifact manifest。
- reuse-existing 的 ES `index_name` / Milvus `collection_name` 来自被复用 artifact manifest。
- reused ES artifact 缺失 `index_name` 时明确抛出 `CorpusAssetError`。
- reused Milvus artifact 缺失 `collection_name` 时明确抛出 `CorpusAssetError`。
- raw prefix 缺失时报错。
- inventory 识别完整 artifact。
- inventory 识别缺失 `_SUCCESS`。
- inventory 提取 manifest 关键字段。
- inventory 不把 `nfcorpus` 误匹配为 `ifir_nfcorpus`。
- JSON 输出 redacts secret / api_key / password / Authorization。

## 8. 自检结果

### 8.1 已运行命令

```bash
pytest tests/scripts
pytest tests/datasets tests/chunking tests/embeddings tests/indexes
ruff check .
mypy .
pytest
```

### 8.2 输出摘要

- `pytest tests/scripts`
  - `15 passed in 0.08s`
- `pytest tests/datasets tests/chunking tests/embeddings tests/indexes`
  - `347 passed in 1.18s`
- `ruff check .`
  - `All checks passed!`
- `mypy .`
  - `Success: no issues found in 140 source files`
- `pytest`
  - `558 passed in 1.81s`

## 9. 范围自检

- 是否开发 `benchmark_suite`：`no`
- 是否正式跑 E1-E4 × 5 datasets：`no`
- 是否修改 retrieval ranking / fusion / metrics 公式：`no`
- 是否修改 HTTP rerank adapter：`no`
- 是否实现 rewrite adapter：`no`
- 是否提交 `.local_artifacts` 或真实 config / 密钥：`no`
- 是否修改流程控制文档：`yes, only merged main/#30; no manual edits to AGENTS.md or architecture.md`
- 是否执行真实构建：`no`
- 是否访问真实 S3：`yes, read-only inventory / dry-run raw existence check`

## 10. 风险与未决项

- 当前 build 脚本只做 dry-run plan，`--execute` 显式拒绝。
- 真实构建仍需要使用现有 `corpus_build` runner，并显式注入真实 chunker、embedding、ES、Milvus clients。
- NFCorpus、IFIRScifact、SciFact、LitSearchRetrieval 仍缺完整 corpus/index artifacts。
- 本轮未创建任何真实 artifact，不覆盖已有 S3 路径。
- Redaction 采用保守 key 匹配；例如 `max_tokens` 会因为包含 `token` 被 redacted，属于安全优先。

## 11. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无

## 12. 提交信息

- 是否已提交：`yes`
- commit subject：`Use reused corpus asset resource names`
- 验收者确认的最终 commit：由验收者用 `git log -1 --oneline` 确认
