# 0013. Platform Config System

- Status: Accepted
- Date: 2026-05-26

## Context

当前主线中配置来源分散：

1. benchmark 侧脚本会直接读取 YAML。
2. embedding client 另有环境变量 helper。
3. 后续 raw opener、runner、CLI 都需要共享同一份配置结构。

如果继续让每个模块各自解析 YAML、环境变量或 CLI 参数，会带来两个问题：

1. 配置优先级不一致，实验难以复现。
2. 密钥与 endpoint 等敏感字段很难统一做 redaction。

## Decision

新增统一 config 系统，约束如下：

1. 配置优先级固定为：
   - 代码默认值
   - `config.yaml`
   - CLI 参数
2. 不支持环境变量隐式覆盖。
3. 配置输出使用强类型 `PlatformConfig`。
4. 提供统一 redacted dump，用于 CLI / manifest / report。
5. 真实本地配置文件不进 git，只提交 `config.example.yaml`。

第一版 schema 至少覆盖：

1. `s3`
2. `elasticsearch`
3. `milvus`
4. `embedding`
5. `rerank`
6. `search_runtime`
7. `raw_sources`
8. `chunking`

## Consequences

1. 后续 runner / CLI 都可以共享同一套配置加载入口。
2. 真实密钥不会通过 config dump 泄露。
3. 旧的 `http_embedding_client_from_env(...)` 暂时保留，但不再作为新配置系统的覆盖层。
4. 如未来需要机器私密配置，仍应通过本地 YAML 文件路径显式传入，而不是引入环境变量隐式覆盖。
