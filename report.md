# 开发报告

本文件由开发 session 维护。

验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：开发全局 config 系统
- 当前分支：`feat/platform-config-system`
- 对应指令文件：`TASK.md`
- 开始时间：2026-05-26
- 完成时间：2026-05-26

## 2. 本次改动

- 改了什么：
  - 新增 `src/eval_platform/config/`
    - `schema.py`
    - `load.py`
    - `redaction.py`
    - `__init__.py`
  - 更新 `src/eval_platform/cli/main.py`
    - 新增 `config-show`
  - 新增 `tests/config/`
    - `test_load.py`
    - `test_redaction.py`
  - 扩展 `tests/test_cli.py`
    - 验证 redacted JSON 输出
    - 验证 CLI override 优先级
  - 新增 `config.example.yaml`
  - 更新 `.gitignore`
    - 忽略 `config.yaml`
    - 忽略 `config.local.yaml`
    - 忽略 `*.secret.yaml`
  - 新增 ADR `docs/decisions/0013-platform-config-system.md`
- 为什么这样改：
  - 配置来源现在分散在 YAML、环境变量 helper 和后续 runner 需求之间，缺少统一入口。
  - 用户明确要求配置优先级可复现，并且不支持环境变量隐式覆盖。
- 没改什么：
  - 没有实现 corpus build runner
  - 没有改 raw-to-normalized / chunk / embedding 的业务逻辑
  - 没有接真实 S3 / ES / Milvus / embedding API

## 3. 涉及文件

- `.gitignore`
- `config.example.yaml`
- `src/eval_platform/config/__init__.py`
- `src/eval_platform/config/schema.py`
- `src/eval_platform/config/load.py`
- `src/eval_platform/config/redaction.py`
- `src/eval_platform/cli/main.py`
- `tests/config/__init__.py`
- `tests/config/test_load.py`
- `tests/config/test_redaction.py`
- `tests/test_cli.py`
- `docs/decisions/0013-platform-config-system.md`
- `docs/ai/current_status.md`
- `report.md`

### 3.1 范围自检

- 是否改动了流程控制文档：`no`
- 如果是，改动理由：无

## 4. 实现说明

### 4.1 配置优先级如何实现

`load_platform_config(...)` 固定实现：

```text
代码默认值 < config.yaml < CLI 参数
```

实现步骤：

1. 先构造 `PlatformConfig()` 默认值。
2. 如果提供 `config_path`，加载 YAML 并做 deep merge。
3. 如果提供 `cli_overrides`，再做一次 deep merge。
4. 最终用 `PlatformConfig.model_validate(...)` 输出强类型对象。

### 4.2 为什么不支持环境变量覆盖

本轮 loader 完全不读取 `os.environ`：

1. 不做环境变量 override
2. 不做 `${VAR}` expansion
3. 不把环境变量作为优先级层

原因：

1. 用户明确要求不要有环境变量隐式覆盖。
2. 只有 config 文件和 CLI 参数，实验配置才能被稳定还原。
3. 私密配置如果需要注入，也应通过本地 YAML 文件路径显式传入。

### 4.3 schema 覆盖的顶层配置块

第一版 `PlatformConfig` 覆盖：

1. `s3`
2. `elasticsearch`
3. `milvus`
4. `embedding`
5. `rerank`
6. `search_runtime`
7. `raw_sources`
8. `chunking`

其中：

1. `PlatformConfig()` 可直接构造
2. 外部依赖字段大多允许为 `None`
3. `raw_sources` 可表达 `IFIRNFCorpus -> s3://.../raw/ifir_nfcorpus/`

### 4.4 deep merge 规则

`deep_merge_config(...)` 规则：

1. dict：递归合并
2. list：整体替换
3. scalar：覆盖
4. `None`：显式覆盖为 `None`

这保证：

1. YAML 可以覆盖默认值
2. CLI 可以覆盖 YAML
3. nested dict 可局部覆盖
4. `embedding.endpoints` 这种 list 不会做逐项拼接

### 4.5 redaction 规则

`dump_redacted_config(...)` 会遮蔽字段名包含以下 token 的值：

1. `password`
2. `secret`
3. `access_key`
4. `api_key`
5. `token`

遮蔽值统一为：

```text
"***"
```

支持 nested dict / list，包括：

1. `embedding.endpoints[].api_key`
2. `search_runtime.rewrite.api_key`
3. `s3.secret_access_key`

### 4.6 CLI 是否接入

本轮已接入最小 CLI：

```bash
evalctl config-show --config path/to/config.yaml
```

行为：

1. 默认输出 redacted JSON
2. 支持示例 override：
   - `--s3-prefix`
   - `--embedding-batch-size`
3. 不读取环境变量

## 5. 自检结果

### 5.1 必跑命令

```bash
git status --short
git diff --name-only origin/main...HEAD
pytest tests/config tests/test_cli.py
ruff check .
mypy .
pytest
```

### 5.2 输出摘要

- `git status --short`：
  - 开发完成前只包含允许范围内文件改动
- `git diff --name-only origin/main...HEAD`：
  - 不包含 `chunking/`、`datasets/`、`embeddings/`、`indexes/`、`retrieval/`、`metrics/`
- `pytest tests/config tests/test_cli.py`：
  - 通过，`13 passed`
- `ruff check .`：
  - 通过
- `mypy .`：
  - 通过，`Success: no issues found in 91 source files`
- `pytest`：
  - 通过，`354 passed`

### 5.3 提交信息

- 是否已提交：`yes`
- commit subject：`Harden platform config validation`
- 验收者确认的最终 commit：

## 6. 风险与未决项

- 已知风险：
  - 旧的 `http_embedding_client_from_env(...)` 仍然存在于 embeddings 模块中；本轮没有删除，只是不把它接入新 config 系统
- 未覆盖场景：
  - 还没有把新 config 系统接到 corpus build runner / raw opener / embedding runner 的真实业务入口
- 需要验收者重点检查的点：
  - `config.example.yaml` 的字段覆盖范围是否足够支撑下一阶段
  - CLI 是否只做了最小配置展示，而没有越界做业务 runner

## 7. 交付结论

- 是否建议验收：`yes`
- 是否建议合并：`yes`
- 如果不能合并，卡点是什么：无
