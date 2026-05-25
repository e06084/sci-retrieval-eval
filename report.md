# 进度报告

## 当前阶段

`main` 已具备 artifact store、S3 backend、normalized dataset schema、MTEB adapter、chunking schema、chunking runner、version-pinned external chunker adapter 和 Sciverse admin-ingest adapter。当前分支 `feat/mteb-normalizer-registry` 在此基础上把 MTEB normalization 改成 per-dataset normalizer registry，并已完成 5 个目标任务的真实 normalization smoke，待 PR / merge review。

## 本次开发

- 新增 `mteb_adapter/base.py`
  - 放置 `MTEBTaskNormalizer`
  - 放置共享 task load / split select / dataset-field extract helpers
- 新增 `mteb_adapter/registry.py`
  - 明确注册 5 个目标任务的 normalizer
- 新增 `mteb_adapter/normalizers/`
  - `scifact.py`
  - `nfcorpus.py`
  - `ifir_scifact.py`
  - `ifir_nfcorpus.py`
  - `litsearch.py`
- 保持统一输出：
  - `CorpusRecord`
  - `QueryRecord`
  - `QrelRecord`
  - `NormalizedDataset`
- 保持统一 artifact writer：
  - `write_normalized_dataset_artifact(...)`
- 保持外部 API 基本稳定：
  - `load_mteb_retrieval_dataset(...)`
  - `export_mteb_retrieval_dataset_artifact(...)`
  - `extract_retrieval_data_from_mteb_task(...)`

## LitSearch 专用规则

`LitSearchRetrieval` 的真实 MTEB 数据中存在大量无效 corpus 行：

- 一类是：
  - `{"title": "IMPLICIT NORMALIZING FLOWS", "text": ""}`
- 另一类是：
  - `{"title": "", "text": ""}`

基于真实数据检查，当前分支对 `LitSearchRetrieval` 采用显式 normalizer 规则：

- 当 `text` / `abstract` 都为空但 `title` 非空时：
  - 允许回退到 `title` 作为 `CorpusRecord.text`
- 当 `text` / `abstract` / `title` 都为空时：
  - 丢弃该空文档
- 同时：
  - 从 qrels 中移除指向空文档的引用
  - 移除清洗后不再拥有任何 qrels 的 query

这部分规则只存在于 `LitSearchRetrievalNormalizer`，不污染其他数据集。

## 单元测试

已新增 / 更新：

- `tests/mteb_adapter/test_convert.py`
  - title-only corpus 行可被转成有效 `CorpusRecord`
- `tests/mteb_adapter/test_registry.py`
  - registry 包含 5 个显式 normalizer
  - `load_mteb_retrieval_dataset(...)` 会分发到注册 normalizer
- `tests/mteb_adapter/test_load.py`
  - `extract_retrieval_data_from_mteb_task(...)` 在 task name 已知时走 registry `extract_raw()`
- `tests/mteb_adapter/test_normalizers.py`
  - `LitSearchRetrievalNormalizer` 会移除空文档、空引用和 orphan queries

## 真实 5-task normalization smoke

本次真实 smoke 目录：

- `.local_artifacts/test/mteb_norm_all_20260525_164656_c67132a/`

报告文件：

- `.local_artifacts/test/mteb_norm_all_20260525_164656_c67132a/normalization_report.json`
- `.local_artifacts/test/mteb_norm_all_20260525_164656_c67132a/normalization_report.md`

### LitSearchRetrieval

- status: `success`
- corpus_count: `57986`
- query_count: `593`
- qrel_count: `634`
- title_non_empty_count: `57423`
- empty_doc_id_count: `0`
- empty_text_count: `0`
- duplicate_doc_id_count: `0`
- qrels_query_missing_from_queries_count: `0`
- qrels_doc_missing_from_corpus_count: `0`
- manifest metadata:
  - `normalizer_name = LitSearchRetrievalNormalizer`

### SciFact

- status: `success`
- corpus_count: `5183`
- query_count: `300`
- qrel_count: `339`
- qrels_query_missing_from_queries_count: `0`
- qrels_doc_missing_from_corpus_count: `0`
- manifest metadata:
  - `normalizer_name = SciFactNormalizer`

### IFIRScifact

- status: `success`
- corpus_count: `500000`
- query_count: `135`
- qrel_count: `735`
- qrels_query_missing_from_queries_count: `0`
- qrels_doc_missing_from_corpus_count: `0`
- manifest metadata:
  - `normalizer_name = IFIRScifactNormalizer`

### IFIRNFCorpus

- status: `success`
- corpus_count: `3633`
- query_count: `86`
- qrel_count: `242`
- qrels_query_missing_from_queries_count: `0`
- qrels_doc_missing_from_corpus_count: `0`
- manifest metadata:
  - `normalizer_name = IFIRNFCorpusNormalizer`

### NFCorpus

- status: `success`
- corpus_count: `3633`
- query_count: `323`
- qrel_count: `12334`
- qrels_query_missing_from_queries_count: `0`
- qrels_doc_missing_from_corpus_count: `0`
- manifest metadata:
  - `normalizer_name = NFCorpusNormalizer`

## 结论

- 5 个目标任务现在都能完成：
  - `load_mteb_task(...)`
  - `extract_retrieval_data_from_mteb_task(...)`
  - `convert_retrieval_data_to_normalized_dataset(...)`
  - `write_normalized_dataset_artifact(...)`
  - `read_normalized_dataset_artifact(...)`
- 当前 registry 设计比继续堆通用 fallback 更可解释：
  - 每个目标任务有显式 normalizer
  - 后续如果单个数据集 layout 演进，只改对应 normalizer
- 后续 chunk / embedding / index / retrieval 继续只消费统一 `NormalizedDataset` artifact

## 建议后续方向

- 合并 `feat/mteb-normalizer-registry`
- 然后开始 `feat/embedding-schema`
