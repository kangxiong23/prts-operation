# ark.contracts

**职责**：只有数据定义与端口（Protocol）。它是唯一被所有人 import 的东西，也是唯一不 import 任何内部模块的东西。

放什么：几何与格坐标、一帧图像、视觉原语的结果（三态 + 置信度）、逻辑空间的类型、动作、时间片与轴、时间刻度、端口定义、异常层次。

**允许 import**：`ark.contracts`、标准库、第三方纯数据包
**禁止 import**：任何其它 `ark.*` 子包

## 为什么它必须这么干净

一旦契约层 import 了实现，所有人就都间接依赖了所有人，解耦就成了摆设。这条由 `tests/contract/import_rules.py` 机器强制，`python tests/contract/import_rules.py` 可单独跑。

## 不在这里

- 原语怎么算 → `../engine/vision/`
- 时间片怎么存 → `../engine/timeline/`
