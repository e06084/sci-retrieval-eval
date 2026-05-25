# 进度报告

## 当前阶段

Local + S3 artifact store 与 normalized dataset schema 已合并到 `main`。`feat/mteb-dataset-adapter` 上 MTEB dataset adapter 已完成本地实现，待 PR / merge review。

## 已完成事项（main）

- bootstrap 基线与项目协作规则
- Local artifact store + S3 artifact store
- normalized dataset schema + JSONL artifact 读写

## 已完成事项（feat/mteb-dataset-adapter）

- MTEB retrieval task → `NormalizedDataset` 转换
- MTEB task 加载与 artifact 导出
- defensive `load_data` fallback
- export manifest 系统字段防覆盖
- `[mteb]` optional dependency
- ADR：`docs/decisions/0003-mteb-dataset-adapter.md`
- `tests/mteb_adapter/` 单元测试（fake task，不下载真实数据）

## 本 PR 范围

- 只实现 MTEB retrieval task 到 normalized dataset artifact 的转换
- 不包含 chunking / embedding / ES / Milvus / retrieval / metrics

## 已验证事项

- 测试不访问真实 MTEB 数据或网络
- 无硬编码 endpoint / bucket / API key / token
- `pytest` / `ruff check .` 通过

## 当前结论

- MTEB adapter 功能完整，可进入 PR / merge review
- 合并后下一步：`feat/chunking-schema`

## 建议下一阶段目标

- 定义 chunking schema 与 chunk artifact 格式
- 继续保持小 PR、不引入 Redis / SQL / Airflow
