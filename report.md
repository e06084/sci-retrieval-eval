# 进度报告

## 当前阶段

目前已经按 `plan.md` 完成项目启动阶段，新的独立仓库已经建立并形成可继续开发的基线。

## 已完成事项

- 新建独立仓库：`sci-retrieval-eval`
- 初始化 git，并整理出 bootstrap 基线
- 写好项目协作规则：`AGENTS.md`
- 写好项目背景文档：`docs/ai/project_brief.md`
- 建立代码骨架：`src/eval_platform/`
  - `artifacts`
  - `datasets`
  - `chunking`
  - `embeddings`
  - `indexes`
  - `retrieval`
  - `mteb_adapter`
  - `metrics`
  - `frontend`
  - `cli`
- 建立测试骨架：`tests/unit`、`tests/integration`
- 补齐 CLI 入口：`evalctl version`
- 补齐基础文档：
  - `README.md`
  - `docs/architecture.md`
  - `docs/decisions/0001-artifact-driven-eval.md`
  - `docs/ai/current_status.md`
  - `docs/ai/handoff_template.md`
  - `docs/ai/open_questions.md`
- 补齐 CI：`.github/workflows/ci.yml`

## 已验证事项

- `pip install -e ".[dev]"` 可通过
- `evalctl version` 可执行
- `pytest` 通过
- `ruff check .` 通过
- 包布局符合要求：只有 `src/eval_platform/`，没有根目录 `eval_platform/`
- 当前未引入业务逻辑
- 未发现硬编码的 endpoint、bucket、API key、token、password

## 当前仓库状态

- 分支：`main`
- 工作区：干净
- 最新提交：`0750cf1 complete bootstrap baseline`

## 当前结论

- bootstrap 阶段已经完成
- 仓库已具备继续实现核心模块的条件
- 下一步建议进入 `artifact store` 实现，作为第一个真正的业务模块

## 建议下一阶段目标

- 实现 `eval_platform/artifacts/`
- 明确 artifact manifest schema
- 提供 local store / S3 store 抽象
- 增加对应单元测试
- 继续保持“不引入 Redis / SQL / Airflow”的约束
