# ark.offline.hardcase

**职责**：在「模型自己不确定」的时候自动截图落盘，按类分目录。

触发时机、定量上限，以及"它同时是三样东西" → 见 `docs/10-规格/采集与验收.md`（规范不在本文件重复）。

落盘位置：`data/hardcases/<类>/`（不进版本库）。

**允许 import**：`ark.contracts`、`ark.engine.*`、`ark.offline.*`

## 不在这里

- 基准集怎么跑 → `../bench/`
