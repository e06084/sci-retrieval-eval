# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`corpus assets module refactor`
- 当前分支：`feat/corpus-assets-module`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-27
- 完成时间：2026-05-27

## 2. 本次改动

将五数据集 corpus/index asset planning 的核心逻辑从脚本 helper 迁移到正式包：

```text
src/eval_platform/corpus_assets/
```

新增模块：

- `registry.py`
  - `DatasetSpec`
  - `CorpusAssetError`
  - `TARGET_DATASETS`
  - `DATASETS_BY_NAME`
  - `DATASETS_BY_SLUG`
  - `dataset_specs_for_selection(...)`
- `naming.py`
  - `ARTIFACT_STAGE_ORDER`
  - `STAGE_SUFFIX`
  - artifact id / ES index / Milvus collection / raw prefix 命名函数
- `inventory.py`
  - `inventory_corpus_assets(...)`
  - manifest metadata summary
  - dataset/artifact matching
- `planner.py`
  - `build_plan_for_datasets(...)`
  - `--reuse-existing` 自洽 artifact chain 选择
  - generated/resolved artifact ids
  - generated/resolved resource names
- `s3.py`
  - S3 client / artifact store helper
  - raw prefix existence check
  - redacted JSON output
  - shared script args

新增 package public API：

- `src/eval_platform/corpus_assets/__init__.py`

## 3. Scripts 职责

保留两个真实环境入口：

- `scripts/inventory_real_corpus_assets.py`
- `scripts/build_real_corpus_assets.py`

它们现在只负责：

- argparse
- bootstrap `src/` import path
- load config
- create S3 client / artifact store
- call `eval_platform.corpus_assets.*`
- print redacted JSON

`scripts/corpus_asset_common.py` 已瘦身为兼容 re-export，不再承载核心实现。

## 4. 行为一致性

本轮是结构迁移，不改变 JSON 输出字段名和 dry-run 语义。

保持不变的行为包括：

- artifact id 生成。
- ES index name / Milvus collection name 生成。
- raw prefix key / URI 生成。
- inventory 输出结构。
- manifest metadata summary 字段。
- `--reuse-existing` 选择同一条依赖自洽 artifact chain。
- `generated_artifact_ids` / `resolved_artifact_ids`。
- `generated_resource_names` / `resolved_resource_names`。
- reused ES artifact 的 `index_name` 来自 manifest。
- reused Milvus artifact 的 `collection_name` 来自 manifest。
- reused ES/Milvus artifact 缺资源名时抛 `CorpusAssetError`。
- sensitive config / JSON redaction。

## 5. 测试迁移

新增：

- `tests/corpus_assets/test_registry_naming.py`
- `tests/corpus_assets/test_inventory.py`
- `tests/corpus_assets/test_planner.py`
- `tests/corpus_assets/test_s3.py`

保留轻量 wrapper 测试：

- `tests/scripts/test_inventory_real_corpus_assets.py`
- `tests/scripts/test_build_real_corpus_assets.py`

覆盖重点：

- dataset name / slug / all selection。
- unknown dataset 报错。
- empty `run_id` 报错。
- artifact id 和 resource name 命名。
- raw prefix key / URI。
- inventory manifest summary 和 `_SUCCESS` 识别。
- generated vs resolved artifact ids。
- generated vs resolved resource names。
- reuse-existing dependency-consistent chain resolution。
- reused ES / Milvus resource name 来自 manifest。
- reused ES / Milvus 缺资源名明确失败。
- redaction。
- 两个 script wrapper 会委托到正式模块。

## 6. 真实只读检查

本轮默认不访问真实 S3，未运行真实只读 inventory / dry-run。

未执行：

- 真实 corpus 构建。
- 真实 chunking。
- 真实 embedding。
- 真实 ES / Milvus 写入。

## 7. 自检结果

### 7.1 已运行命令

```bash
pytest tests/corpus_assets
pytest tests/scripts
pytest
ruff check .
mypy .
```

### 7.2 输出摘要

- `pytest tests/corpus_assets`
  - `18 passed in 0.09s`
- `pytest tests/scripts`
  - `3 passed in 0.10s`
- `pytest`
  - `564 passed in 1.86s`
- `ruff check .`
  - `All checks passed!`
- `mypy .`
  - `Success: no issues found in 151 source files`

## 8. 范围自检

- 是否实现 `--execute`：`no`
- 是否真实构建 corpus：`no`
- 是否真实调用 chunking：`no`
- 是否真实调用 embedding：`no`
- 是否真实写 ES / Milvus：`no`
- 是否修改 retrieval / metrics / benchmark：`no`
- 是否修改 indexes / chunking / embeddings 业务逻辑：`no`
- 是否新增正式 CLI：`no`
- 是否提交真实 config / 密钥 / `.local_artifacts` / inventory 输出：`no`
- 是否修改 `TASK.md` / `AGENTS.md`：`no`

## 9. 风险与未决项

- `scripts/corpus_asset_common.py` 暂时保留兼容 re-export，后续确认无外部引用后可删除。
- `scripts/build_real_corpus_assets.py --execute` 仍显式拒绝执行；真实构建仍需要后续 runner/client 接入。
- 本轮没有运行真实 S3 只读检查；行为一致性主要由 fake store / fake client 单元测试覆盖。

## 10. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无

## 11. 提交信息

- 是否已提交：`yes`
- commit subject：`Move corpus asset planning into package`
- 验收者确认的最终 commit：由验收者用 `git log -1 --oneline` 确认
