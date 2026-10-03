# ark-ai-op

明日方舟排轴辅助工具。**目前处于设计阶段：有设计文档和代码骨架，还没有实现。**

目标：让 AI 通过 ADB 操作明日方舟真机，把时间花在"轴"的构思与优化上，而不是花在手工操作上。

## 目录

- `docs/` —— 设计文档。**从这里进：[docs/README.md](docs/README.md)**
- `src/ark/` —— 代码骨架。分层与规则见 [docs/10-规格/代码结构.md](docs/10-规格/代码结构.md)
  - `contracts/` 共享契约 · `engine/` 在线引擎 · `offline/` 离线工具 · `runtime/` 组装与入口
- `tests/contract/` —— 导入规则的强制检查
- `dsh/` —— DSH 侧的纯配置：预设、MCP 连接、本模式专属的 skill
- `assets/` —— 生成物。特征库与地形进版本库，游戏数据库不进
- `data/` —— 运行时数据，全部不进版本库
- `MaaAssistantArknights-dev-v2/` —— MAA 的本地检出，**只读参考**，仅用于借鉴设计取舍

## 一条硬规则

`contracts` 谁都许 import，它自己谁也不 import；`engine` 永远不 import `offline`；`engine` 的六个子包互不 import，由 `runtime/wiring.py` 接起来。

这不是口号 —— 跑一条命令就知道有没有破：

```powershell
python tests/contract/import_rules.py
```

它只用标准库，不需要装任何东西。

## 关于 MAA 检出

[MaaAssistantArknights](https://github.com/MaaAssistantArknights/MaaAssistantArknights) 采用 **AGPL-3.0**，本项目采用 **MIT**。

因此该目录只作为设计灵感的来源：**不复制它的任何代码、协议格式、字段命名或 schema 结构。** 只借鉴"哪些事情值得做、哪些坑它踩过"这类取舍层面的结论。
