# 开发报告

本文件由开发 session 维护。验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：`Sciverse benchmark v1 默认参数固化`
- 当前分支：`feat/sciverse-benchmark-defaults`
- 基线：`main`
- 完成时间：2026-06-03

## 2. 本轮实现

本轮把复现实验确认后的 Sciverse benchmark v1 口径固化为代码默认值，避免后续运行时继续依赖脚本或人工显式传参。

改动：

- 新增 `eval_platform.defaults`
  - 集中定义 retrieval、hybrid、rerank 和 Milvus 默认参数。
  - `DEFAULT_RETRIEVAL_TOP_K = 100`。
  - `DEFAULT_RERANK_CANDIDATE_CAP = 0`。
  - Milvus 默认字段为 `chunk_id`、`vector`，默认 metric 为 `COSINE`。
  - Milvus 默认索引为 `HNSW`，参数为 `M=16`、`efConstruction=200`。
  - Milvus 默认 search params 为 `{"metric_type": "COSINE", "params": {}}`。
  - Milvus 默认 `title.max_length = 65535`，`text.max_length` 保持 `65535`。
- Benchmark / retrieval 默认参数
  - `BenchmarkSettingSpec.top_k` 默认改为 `100`。
  - `RetrievalRunConfig.top_k` 默认改为 `100`。
  - E1-E4 默认 setting 不显式传 `top_k` 时继承 `100`。
  - hybrid 默认保持 `hybrid_per_source_topk=50`、`rrf_path_topk=25`。
  - rerank 默认保持 `rerank_cross_path_topk=50`、`rerank_candidate_cap=0`。
- Milvus ingest 默认参数
  - 未显式配置 `index_params` 时，创建 HNSW/COSINE/M=16/efConstruction=200 索引。
  - manifest metadata 和 fingerprint 使用解析后的有效 index params。
  - 显式配置 `index_params` 时仍允许覆盖默认值。
- Milvus retrieval 默认参数
  - 查询时默认传完整 `search_params={"metric_type": "COSINE", "params": {}}`。
  - 显式 `search_params` 仍可覆盖 nested `params`。
- Schema 默认参数
  - `default_milvus_schema()` 的 `title.max_length` 默认改为 `65535`。
  - 不给 `vector_dim` 设置默认值，仍必须来自 embedding manifest 或显式配置。
- 文档
  - 更新 `README.md`、`config.example.yaml`、`docs/architecture.md`、`docs/operations/e1_e4_benchmark_suite.md`，记录 Sciverse benchmark v1 默认协议。

## 3. 行为语义

- 当前 main 默认 benchmark retrieval 深度从旧的 `top_k=10` 改为 `top_k=100`。
- Milvus 默认从未显式指定索引类型的行为收敛到 HNSW/COSINE/M=16/efConstruction=200。
- Milvus search params 默认使用完整结构，避免空 dict 在不同 pymilvus/Milvus 版本上的解释差异。
- `rerank_candidate_cap=0` 表示不额外截断 rerank 候选集，由前序 RRF 候选规模决定。
- `vector_dim` 不默认 1024，避免把 bge-m3 的维度误写成平台级默认。

## 4. 测试覆盖

新增/更新测试覆盖：

- Benchmark setting 默认值为 Sciverse benchmark v1 口径。
- Retrieval run config 默认值为 Sciverse benchmark v1 口径。
- E1-E4 默认 settings 继承 `top_k=100` 和 `rerank_candidate_cap=0`。
- Milvus ingest 默认创建 HNSW/COSINE/M=16/efConstruction=200 索引。
- Milvus ingest manifest/fingerprint 记录解析后的有效 index params。
- Milvus ingest 支持显式 index params 覆盖默认值。
- Milvus schema 默认字段和 max_length 符合新协议。
- Milvus retrieval 默认传完整 search params。
- Experiment runner 复用路径的默认 retrieval top_k 断言更新为 100。

## 5. 验证结果

开发 session 已运行：

```bash
PYTHONPATH=src python -m pytest tests/benchmark/test_settings.py tests/retrieval/test_runner.py tests/retrieval/test_milvus_adapter.py tests/indexes/test_milvus_ingest.py tests/config/test_load.py
PYTHONPATH=src python -m pytest
git ls-files '*.py' | xargs ruff check
mypy .
```

结果：

- `77 passed`
- `710 passed`
- `ruff`: `All checks passed!`
- `mypy`: `Success: no issues found in 194 source files`

说明：

- 本地工作区存在未提交实验脚本，`ruff check .` 会扫描这些非本轮文件并失败；本轮对 git 跟踪的 Python 文件执行了 ruff。
- 当前环境原始缺少 `ruff`、`mypy` 和 `types-PyYAML`，开发 session 已安装 dev 检查依赖后完成验证。

## 6. 未实现项

- 未把 `vector_dim=1024` 写入默认值；这是有意保持，维度仍来自 embedding manifest 或显式配置。
- 未合入此前复现实验脚本和实验结果文档；这些是实验资产，不属于本轮默认参数固化代码。
