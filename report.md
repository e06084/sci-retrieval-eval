# 开发报告

本文件由开发 session 维护。

验收 session 默认只检查这里，不追聊天记录。

## 1. 任务信息

- 任务名：
- 当前分支：
- 对应指令文件：`TASK.md`
- 开始时间：
- 完成时间：

## 2. 本次改动

- 改了什么：
- 为什么这样改：
- 没改什么：

## 3. 涉及文件

- `path/to/file`
- `path/to/test_file`

### 3.1 范围自检

- 是否改动了流程控制文档：`yes / no`
- 如果是，改动理由：

## 4. 实现说明

### 4.1 关键决策

- 决策 1：
- 决策 2：

### 4.2 关键行为

- 行为 1：
- 行为 2：

## 5. 自检结果

### 5.1 必跑命令

```bash
git status --short
git diff --name-only origin/main...HEAD
pytest
ruff check .
```

### 5.2 输出摘要

- `git status --short`：
- `git diff --name-only origin/main...HEAD`：
- `pytest`：
- `ruff check .`：
- `mypy .`：

### 5.3 提交信息

- 最新 commit：
- 相关 commit 列表：

## 6. 风险与未决项

- 已知风险：
- 未覆盖场景：
- 需要验收者重点检查的点：

## 7. 交付结论

- 是否建议验收：`yes / no`
- 是否建议合并：`yes / no`
- 如果不能合并，卡点是什么：
