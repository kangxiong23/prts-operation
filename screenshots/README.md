# Agent 截图工作区

本目录是 AI agent（负责本项目「测试样例扩充」）专属的截图工作区。

## 归属

- 使用者：负责测试样例扩充的 AI agent
- 人类一般只读，不在此目录直接编辑
- 位置：仓库根目录下的 `screenshots/`

## 用途

- 存放测试样例扩充过程中产生的界面截图、渲染结果与对比图
- 作为 agent 的中间产物区，不进入版本库

## 版本控制约定

`.gitignore` 中的规则：

```gitignore
screenshots/*
!screenshots/README.md
```

即：除本 `README.md` 外，`screenshots/` 下的所有文件都不入库。
（必须写成 `screenshots/*` 而非 `screenshots/`，否则 git 无法用 `!` 反向放行本文件。）

## 命名约定

- 按测试样例分子目录，例如 `screenshots/<case-name>/`
- 文件名避免空格与中文，使用 `case-step.png` 形式

## 清理

- 目录内截图属于临时产物，可随时删除，不保证长期保留
