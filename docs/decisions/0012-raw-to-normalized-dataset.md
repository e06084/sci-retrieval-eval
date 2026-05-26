# 0012. Raw Dataset To Normalized Dataset

- Status: Accepted
- Date: 2026-05-26

## Context

`raw_dataset` 第一版已经把不可变 raw source 固定成 snapshot-only artifact，但当前主线的 `normalized_dataset` 仍然没有显式依赖 raw snapshot。

这会带来两个问题：

1. `normalized_dataset` 的输入身份不明确，后续难以审计“当前标准化结果到底来自哪一份 raw source”。
2. 已知 raw source 里存在较大 JSONL 文件，例如 `ifir_scifact/corpus.jsonl` 约 745MB；标准化阶段如果直接复用一次性读完整文件的模式，会再次引入内存风险。

## Decision

新增显式 raw-to-normalized API，由 raw snapshot artifact 作为输入、normalized dataset artifact 作为输出。

第一版约束如下：

1. 入口 API 接受：
   - `source_store`
   - `output_store`
   - `RawToNormalizedConfig`
   - `RawFileOpener`
2. `RawFileOpener` 负责根据 `RawDatasetFile.uri` 打开原始文件流。
3. 提供内置 `S3RawFileOpener`，用于打开 `s3://bucket/key` raw source。
4. 第一版只实现：
   - `IFIRNFCorpus`
   - 默认 normalizer 名称：`ifir_nfcorpus_raw_jsonl_tsv_v1`
5. `normalized_dataset` manifest 必须显式写入对 `raw_dataset` 的 dependency。
6. `normalized_dataset` metadata 至少记录：
   - `source`
   - `task_name`
   - `split`
   - `normalizer_name`
   - `raw_dataset_artifact_id`
   - `raw_dataset_fingerprint`
   - `raw_source_uri`
   - `normalized_schema_version`
7. raw JSONL 解析必须按流式逐行读取，不能通过 `body.read().decode()` 一次性读取大文件。

## Consequences

1. `normalized_dataset` 现在有了明确的 raw 上游身份。
2. 标准化逻辑与 raw source 打开方式解耦，测试可以用 fake opener，生产可使用内置 S3 opener。
3. 第一版只覆盖 `IFIRNFCorpus`，但 API 保留了对其他 raw asset 扩展 normalizer 的入口。
4. 本轮仍然不实现全链路 runner / orchestration，后续如需批量执行，可在此 API 之上再封装。
