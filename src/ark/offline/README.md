# ark.offline

**职责**：不跟游戏一起跑的那一半。资产生成、难例采集、基准集与离线验收。

**允许 import**：`ark.contracts`、`ark.engine`、`ark.offline`
**禁止 import**：`ark.runtime`

## 依赖方向只有这一边

`offline` 可以用 `engine`（重放要走线上那套识别链路），但**`engine` 永远不 import `offline`**。

这条是机器强制的。理由是：一旦线上代码开始依赖离线工具，线上就不再是能被独立重放的那一半，"离线重放验收线上"这个验收手段本身就失效了。

## 不在这里

- 线上识别链路 → `../engine/vision/`
- 依赖注入与入口 → `../runtime/`
