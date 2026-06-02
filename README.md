# sci-retrieval-eval

`sci-retrieval-eval` 是面向科研文献检索研发团队的离线评测基座。项目目标不是替代线上检索服务，而是把 corpus 构建、检索实验、指标计算和结果比对做成可复现、可审计、可追溯的 artifact 流程。

当前主线已经支持：

- 五个目标数据集的 raw / normalized / chunk / embedding / ES / Milvus artifact 管理。
- live ES、Milvus、hybrid、rerank 检索。
- `retrieval_run -> metrics_run -> benchmark_run -> benchmark_suite_run` 评测链路。
- `experiment_run` 实验层，可按 fingerprint 复用或补跑 retrieval / metrics / benchmark stage。
- trace/replay 所需的检索记录。
- S3 artifact store、artifact catalog、真实服务配置脱敏、真实环境连通性检查。

## 核心设计

系统以 artifact 为中心。每个阶段只消费上游 artifact，不依赖隐藏的本地状态。

Corpus 构建链路：

```text
raw_dataset
  -> normalized_dataset
  -> chunked_corpus
  -> embeddings
  -> elasticsearch_index
  -> milvus_collection
```

评测链路：

```text
normalized_dataset + elasticsearch_index + milvus_collection
  -> retrieval_run
  -> metrics_run
  -> benchmark_run
  -> benchmark_suite_run 或 experiment_run
```

关键原则：

- 每个完成 artifact 都要有 manifest 和 `_SUCCESS`。
- 下游阶段必须校验上游 artifact 完成状态。
- 资产逻辑等价由 `asset_fingerprint` 判断，`run_id`、时间戳、物理资源名和真实服务地址不进入 fingerprint。
- 检索执行和指标计算分离。
- 检索可以返回 chunk-level hits，metrics 阶段投影回 doc-level ranking。
- 默认记录完整 trace，便于复现、replay 和排查随机性。
- 真实 `config.yaml` 不进 git，只提交 `config.example.yaml`。

## 模块结构

```text
src/eval_platform/
  artifacts/       artifact manifest、本地/S3 store、artifact type
  config/          全局配置 schema、加载、脱敏输出
  corpus_assets/   五数据集 registry、S3 inventory、asset planning
  corpus_build/    corpus 构建编排
  datasets/        raw / normalized dataset schema 和转换
  chunking/        chunk schema、外部 chunker adapter、provenance
  embeddings/      HTTP embedding client、embedding artifact
  indexes/         Elasticsearch / Milvus ingest artifact
  retrieval/       ES、Milvus、hybrid、RRF、rerank、trace、replay
  metrics/         MTEB 风格 doc-level metrics
  benchmark/       retrieval + metrics + suite 编排
  experiments/     实验计划、自动复用/补跑、experiment_run artifact
  mteb_adapter/    MTEB 数据和任务适配
```

辅助目录：

- `scripts/`：真实环境辅助脚本。当前 `build_real_corpus_assets.py` 只做 dry-run plan，不执行真实写入。
- `docs/architecture.md`：更详细的架构和原则。
- `docs/decisions/`：关键设计决策记录。
- `docs/operations/`：真实环境运行手册。
- `tests/`：单元测试和 fake client 测试，默认不访问真实外部服务。

## 外部依赖

运行单元测试所需依赖较少：

- Python `>=3.11`
- `pydantic`
- `typer`
- `pyyaml`
- `pytest` / `ruff` / `mypy`

真实 corpus 构建和评测还依赖：

- S3 兼容对象存储，用于保存所有 artifact。
- Elasticsearch，用于 BM25 检索和 Milvus 命中 enrich。
- Milvus，用于向量检索。
- HTTP embedding API，当前实验使用 `BAAI/bge-m3`，维度 1024。
- HTTP rerank API，当前实验使用 `BAAI/bge-reranker-v2-m3`。
- `sciverse_clean/agentic-search` 外部仓库中的 chunk 逻辑。
- 五个 benchmark 的 raw 数据前缀。

Python extras：

```bash
pip install -e ".[dev]"
pip install -e ".[s3,milvus]"
```

`mteb` 如需直接加载 MTEB 数据再安装：

```bash
pip install -e ".[mteb]"
```

Elasticsearch、embedding、rerank 当前使用标准库 HTTP adapter，不需要额外 Python SDK。S3 需要 `boto3`，Milvus 需要 `pymilvus`。

## 配置

复制示例配置后填写真实依赖：

```bash
cp config.example.yaml config.yaml
```

不要提交真实 `config.yaml`。其中可能包含 S3 AK/SK、ES 密码、embedding/rerank API key。

配置优先级：

```text
默认值 < YAML 配置文件 < CLI 参数
```

当前不使用环境变量作为配置层。若配置值本身引用了环境变量，只能由具体脚本显式支持。

查看脱敏后的合并配置：

```bash
evalctl config-show --config config.yaml
```

检查真实环境连通性：

```bash
python -m eval_platform.cli.check_connectivity --config config.yaml
```

输出 JSON：

```bash
python -m eval_platform.cli.check_connectivity --config config.yaml --json
```

## 开发和测试

安装开发环境：

```bash
pip install -e ".[dev]"
```

基础检查：

```bash
evalctl version
pytest
ruff check .
mypy .
```

真实外部服务不应出现在单元测试里。需要访问 S3、ES、Milvus、embedding、rerank 时，应作为显式手工检查，并记录配置、artifact id 和结果。

## Corpus Asset 操作

五个目标数据集：

| task | dataset key | raw layout |
|---|---|---|
| IFIRNFCorpus | `ifir_nfcorpus` | `jsonl_tsv` |
| NFCorpus | `nfcorpus` | `jsonl_tsv` |
| IFIRScifact | `ifir_scifact` | `jsonl_tsv` |
| SciFact | `scifact` | `jsonl_tsv` |
| LitSearchRetrieval | `litsearch` | `parquet_dir_shards` |

检查 S3 上已有 corpus/index 资产：

```bash
python scripts/inventory_real_corpus_assets.py \
  --config config.yaml \
  --s3-prefix sciverse_benchmark/assets \
  --raw-prefix sciverse_benchmark/raw \
  --output tmp/corpus_asset_inventory.json
```

生成五数据集资产规划：

```bash
python scripts/build_real_corpus_assets.py \
  --config config.yaml \
  --dataset all \
  --run-id e1_e4_bge_m3_YYYYMMDD \
  --reuse-existing \
  --s3-prefix sciverse_benchmark/assets \
  --raw-prefix sciverse_benchmark/raw \
  --output tmp/corpus_asset_plan.json
```

注意：`scripts/build_real_corpus_assets.py` 当前是规划脚本，`--execute` 会拒绝执行真实写入。真实构建应通过 `eval_platform.corpus_build` 和显式 client 编排，确保每一步产出 artifact 和 manifest。

## Experiment 运行层

main 分支当前的实验抽象是 Python API，不是正式 CLI。推荐流程：

```text
1. 用 corpus asset inventory / planning 确认五数据集 normalized、ES、Milvus 资产。
2. 构造 ExperimentRunConfig，可以显式传 BenchmarkDatasetSpec，也可以通过 ExperimentCorpusAssetConfig 从 corpus assets 自动解析。
3. plan_experiment 计算 dataset x setting 的 retrieval / metrics / benchmark 预期 fingerprint 和 reuse/create 决策。
4. run_experiment 只执行缺失 stage，并写入 experiment_run artifact。
5. 可选写入 artifact_catalog/default/records.jsonl，后续实验优先按 catalog 查找可复用 artifact。
```

核心 API：

- `ExperimentCorpusAssetConfig`
- `ExperimentRunConfig`
- `plan_experiment`
- `run_experiment`
- `read_experiment_run_artifact`

复用规则以 `asset_fingerprint_sha256` 为准。catalog 是查询加速索引，不是唯一真相；唯一真相仍是 artifact 目录中的 manifest 和 `_SUCCESS`。如果只改变 rerank 或 metric 配置，系统应复用 corpus / index 资产，只补跑受影响的 retrieval、metrics 或 benchmark stage。

## Benchmark 运行口径

当前评测 setting：

| setting | retrieval mode | 说明 |
|---|---|---|
| E1-milvus | `milvus` | 向量检索 |
| E2-es | `es` | BM25 检索 |
| E3-hybrid | `hybrid` | ES + Milvus + RRF |
| E4-hybrid-rerank | `hybrid` + rerank | E3 候选后 rerank |

与历史旧实验对齐时使用：

- `top_k=100`，对应旧实验 `search_limit=100`。
- `hybrid_per_source_topk=50`
- `rrf_path_topk=25`
- `rerank_cross_path_topk=50`
- `rerank_candidate_cap=0`
- `sub_queries=0`
- `rewrite_enabled=false`
- embedding endpoint index `1`，当前为 `3886`
- rerank endpoint index `1`，当前为 `3886`

当前 main 保留 `benchmark_suite` Python API：

- `BenchmarkDatasetSpec`
- `BenchmarkSettingSpec`
- `BenchmarkSuiteRunConfig`
- `run_benchmark_suite`
- `read_benchmark_suite_run_artifact`

完整 5bench x E1-E4 的批处理 CLI 仍需后续正式化。当前真实实验可通过 `benchmark_suite` API 直接编排，也可通过 `experiments` API 做计划、复用和补跑；结果分别写入 S3 `benchmark_suite_run` 或 `experiment_run` artifact。

## Baseline 实验结果

以下结果是 main README 记录的 2026-05-29 baseline，用于说明当前主线可复现的 E1-E4 全量评测口径；不包含实验分支上的后续 top-k / RRF / rerank sweep 结果。

实验时间：2026-05-29

Suite：

```text
e1_e4_top100_3886_5bench_20260529
```

Artifact：

```text
s3://scibase-service/sciverse_benchmark/assets/benchmark_suite_run/e1_e4_top100_3886_5bench_20260529/
```

配置：

```text
top_k=100
embedding endpoint index=1, 3886
rerank endpoint index=1, 3886
hybrid params=50/25
rerank params=50/0
```

验收：

```text
20/20 benchmark items complete
retrieval failed query count = 0
metrics failed retrieval query count = 0
```

### ndcg10

| setting | IFIRNFCorpus | NFCorpus | IFIRScifact | SciFact | LitSearch |
|---|---:|---:|---:|---:|---:|
| E1-milvus | 0.34309 | 0.30387 | 0.41194 | 0.62880 | 0.31389 |
| E2-es | 0.22833 | 0.28289 | 0.23974 | 0.58076 | 0.29673 |
| E3-hybrid | 0.30225 | 0.32946 | 0.37596 | 0.65288 | 0.37766 |
| E4-hybrid-rerank | 0.36683 | 0.33386 | 0.46157 | 0.70118 | 0.39945 |

### mrr10

| setting | IFIRNFCorpus | NFCorpus | IFIRScifact | SciFact | LitSearch |
|---|---:|---:|---:|---:|---:|
| E1-milvus | 0.39919 | 0.50479 | 0.65603 | 0.59422 | 0.28319 |
| E2-es | 0.27694 | 0.47715 | 0.36986 | 0.54895 | 0.26003 |
| E3-hybrid | 0.32460 | 0.52730 | 0.58551 | 0.61312 | 0.33802 |
| E4-hybrid-rerank | 0.43874 | 0.53626 | 0.69142 | 0.67203 | 0.35862 |

### r100

| setting | IFIRNFCorpus | NFCorpus | IFIRScifact | SciFact | LitSearch |
|---|---:|---:|---:|---:|---:|
| E1-milvus | 0.69514 | 0.23884 | 0.73189 | 0.89533 | 0.71265 |
| E2-es | 0.42591 | 0.19802 | 0.39833 | 0.83722 | 0.59354 |
| E3-hybrid | 0.43841 | 0.17356 | 0.47876 | 0.83689 | 0.57515 |
| E4-hybrid-rerank | 0.43841 | 0.17356 | 0.47876 | 0.83689 | 0.57515 |

## 重复性检查

复跑 suite：

```text
e1_e4_top100_3886_5bench_repeat1_20260529
```

复跑 artifact：

```text
s3://scibase-service/sciverse_benchmark/assets/benchmark_suite_run/e1_e4_top100_3886_5bench_repeat1_20260529/
```

复跑条件：

```text
不重建资产
asset_rebuild=false
同一批 normalized / ES / Milvus assets
同一 top_k=100 和 3886 embedding/rerank endpoint
```

重复性结果：

```text
20/20 benchmark items complete
failed query count = 0
ndcg10 / mrr10 / r100 共 60 个核心指标逐项差值 = 0
max absolute delta = 0
```

结论：在当前资产、当前 3886 embedding/rerank 服务和当前运行链路下，评测结果可重复。

## 与历史结果的关系

历史旧实验 `no3888-e1e4-20260522` 使用 `search_limit=100`。新系统默认 `top_k=10` 时不可直接比较；对齐为 `top_k=100` 后，大部分数据集和 setting 已接近旧结果。

当前仍存在部分差异，主要原因不是重复性问题，而是旧资产和旧服务状态不可完全复现：

- 旧 IFIRNFCorpus vector 资产 chunk 数为 11958，当前资产为 11962。
- 旧 chunker commit 与当前 chunker commit 不同。
- 旧 rerank 服务版本或 endpoint 行为没有被完整记录为 replay artifact。
- 当前 3888 rerank endpoint 在新系统 adapter 下不可用，3886 可稳定复跑。

因此，当前系统已能保证“同一资产 + 同一配置 + 同一外部 API 行为”下的结果一致；完全复现历史旧实验还需要历史资产和外部服务状态也可审计。
