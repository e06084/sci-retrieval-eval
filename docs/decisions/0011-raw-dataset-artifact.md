# 0011. Raw Dataset Artifact

- Status: Accepted
- Date: 2026-05-26

## Context

当前主线已经具备 `normalized_dataset`、`chunked_corpus` 和 `embeddings` artifact，但原始输入文件在进入标准化前仍缺少一个显式、可审计的落盘层。

对于较大或外部依赖较多的数据集，直接从内存对象开始标准化有几个问题：

1. 原始数据版本、文件列表和文件内容缺少稳定快照。
2. 下载/cache/load 失败时，很难判断问题发生在“拉取原始数据”还是“标准化逻辑”。
3. 后续多人复现实验时，无法直接核对最初消费的原始文件集合是否一致。

## Decision

新增 `raw_dataset` artifact，作为数据链路的第一层显式快照。

其约束如下：

1. artifact 类型固定为 `raw_dataset`。
2. 第一版采用 snapshot-only 设计：
   - artifact 只写 `_MANIFEST.json` 和 `_SUCCESS`
   - 默认不复制 raw 文件内容进入 artifact store
3. manifest metadata 至少记录：
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
4. `files` 中每条记录至少包含：
   - `path`
   - `uri`
   - `size_bytes`
   - `sha256`
5. 单文件 `sha256` 与 dataset 级 `content_fingerprint_sha256` 都按稳定排序生成。
6. `content_fingerprint_sha256` 基于：
   - `path`
   - `uri`
   - `size_bytes`
   - `sha256`
7. hash 计算必须流式读取文件内容，不要求为了满足当前 `ArtifactStore` 接口而改造整个存储抽象。
6. 第一阶段支持两种导入方式：
   - 从本地目录导入
   - 从既有 S3 prefix 导入

## Consequences

1. 后续 `normalized_dataset` 应依赖 `raw_dataset` 作为上游输入身份。
2. raw 层与 normalized 层职责分离后，下载/拉取问题与标准化问题可以分开定位。
3. 第一版不会生成 raw 文件副本，因此可以直接把不可变 S3 raw prefix 作为可信起点。
4. 若后续需要 materialize raw 文件副本，应单独演进 `ArtifactStore` 的 streaming upload 能力，而不是回到 `dict[str, bytes]` 聚合方案。
