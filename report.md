# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`artifact type / metadata key registry`
- 当前分支：`feat/artifact-type-registry`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-27
- 完成时间：2026-05-27
- 实现提交 SHA：`9f7b3b44f18f473ce6f83c7112d150b84f78d3a1`
- 报告提交 SHA：本报告单独提交后由 `git log -1 --oneline` 确认；提交内容无法自引用自身 SHA。

## 2. 新增模块和职责

新增：

- `src/eval_platform/artifacts/types.py`
  - 集中定义 10 个 artifact type 字符串常量。
  - 定义 `CORPUS_ASSET_STAGE_ORDER`。
  - 定义 `ALL_ARTIFACT_TYPES`。
- `src/eval_platform/artifacts/metadata_keys.py`
  - 集中定义本轮涉及的 manifest metadata key 常量。
  - 定义 `DEPENDENCY_METADATA_KEYS_BY_ARTIFACT_TYPE`，供 corpus asset reuse 追踪依赖链。

`src/eval_platform/artifacts/__init__.py` 已导出新增 registry 常量。

## 3. 兼容 public import

以下旧入口已保留，且测试确认与新注册表值一致：

- `eval_platform.datasets.RAW_DATASET_ARTIFACT_TYPE`
- `eval_platform.datasets.NORMALIZED_DATASET_ARTIFACT_TYPE`
- `eval_platform.chunking.CHUNKED_CORPUS_ARTIFACT_TYPE`
- `eval_platform.embeddings.EMBEDDINGS_ARTIFACT_TYPE`
- `eval_platform.indexes.ELASTICSEARCH_INDEX_ARTIFACT_TYPE`
- `eval_platform.indexes.MILVUS_COLLECTION_ARTIFACT_TYPE`
- `eval_platform.retrieval.RETRIEVAL_RUN_ARTIFACT_TYPE`
- `eval_platform.metrics.METRICS_RUN_ARTIFACT_TYPE`
- `eval_platform.benchmark.BENCHMARK_RUN_ARTIFACT_TYPE`
- `eval_platform.corpus_build.CORPUS_BUILD_ARTIFACT_TYPE`

## 4. 替换范围

已替换：

- `src/eval_platform/corpus_assets/`
  - stage order 和 stage suffix 改为使用 artifact type 注册表。
  - inventory metadata summary key 改为使用 metadata key 注册表。
  - planner reuse dependency tracing 改为使用 `DEPENDENCY_METADATA_KEYS_BY_ARTIFACT_TYPE`。
  - planner ES/Milvus 资源名和依赖 artifact id 输出字段使用同值 metadata key 常量。
- `src/eval_platform/corpus_build/runner.py`
  - summary 中 artifact type 和 ES/Milvus resource metadata key 使用注册表常量。
  - corpus build run artifact type 使用注册表常量。
- 各 artifact writer 模块中本地定义的 `*_ARTIFACT_TYPE`
  - dataset raw / normalized
  - chunked corpus
  - embeddings
  - Elasticsearch / Milvus index artifact
  - retrieval / metrics / benchmark artifact
  - corpus build runner
- ES / Milvus / embeddings writer 中与本轮 registry 对应的高风险 dependency/resource metadata key。

## 5. 未替换范围和理由

- retrieval / metrics / benchmark runner 中的 dependency artifact type 引用属于 `TASK.md` 标注的“可以替换但不要过度扩张”，本轮未扩做。
- 测试中的 JSON 字面量断言保留字符串，用于验证序列化输出字段和值没有变化。
- 非本轮列出的业务字段名、Pydantic 字段名和用户可见输出字段未做全仓替换，避免改变 schema 语义或扩大重构面。

## 6. 测试结果

已运行：

```bash
pytest tests/artifacts tests/corpus_assets tests/corpus_build
pytest
ruff check .
mypy .
```

结果：

- `pytest tests/artifacts tests/corpus_assets tests/corpus_build`
  - `94 passed in 0.56s`
- `pytest`
  - `573 passed in 1.97s`
- `ruff check .`
  - `All checks passed!`
- `mypy .`
  - `Success: no issues found in 155 source files`

## 7. 真实只读一致性校验

基线在最新 `main` 上生成，分支在 `feat/artifact-type-registry` 上重新生成。三项结构化 JSON 对比均为 `equal=True`。

```text
inventory:
  equal=True
  baseline_sha256=a97825b9588234d1671ec82eb934678bd4aa088f9e334e5e47bf7b117e9822c0
  branch_sha256=a97825b9588234d1671ec82eb934678bd4aa088f9e334e5e47bf7b117e9822c0

ifir_reuse:
  equal=True
  baseline_sha256=d64d753b1184e94b1e44498bc6a9d0ccdc5ceff77a420e51248b78133772c82e
  branch_sha256=d64d753b1184e94b1e44498bc6a9d0ccdc5ceff77a420e51248b78133772c82e

all_create:
  equal=True
  baseline_sha256=11c53cd75e236d983af80f66e79c2797395b168241911c6f7055d7b44e6306da
  branch_sha256=11c53cd75e236d983af80f66e79c2797395b168241911c6f7055d7b44e6306da
```

访问真实外部服务情况：

- 仅执行 S3 read-only inventory / dry-run。
- 未写 S3。
- 未调用 ES / Milvus / embedding / rerank。
- 未提交真实 `config.yaml`、密钥或 `/tmp/artifact_registry_*.json` 输出文件。

## 8. 风险与未决项

- 本轮是字符串常量集中化，不改变 manifest JSON schema、artifact 路径布局或 corpus asset planning 行为。
- 后续如要继续降低拼写风险，可在独立任务中替换 retrieval / metrics / benchmark runner 的 optional dependency key 引用。

## 9. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无
