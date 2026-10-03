# ark.engine.device

**职责**：取像、注入触摸。**这是整个项目里唯一接触真机的地方。**

它是「端口 + 适配器」结构：端口定义在 `ark.contracts.ports`，这里只放适配器。

| 适配器 | 用途 |
| --- | --- |
| `adb/` | 真机 |
| `replay/` | 离线重放：读录制的帧、只记动作不真的点 |

**离线重放不是特例代码，是换一个适配器。** 这是"视觉层必须能被离线驱动"那条地基约束的落地方式 —— 也正因为如此，设备层必须是端口结构。

**允许 import**：`ark.contracts`、`ark.engine.device`
**禁止 import**：其它 `ark.engine.*`、`ark.offline.*`

## 不在这里

- 像素怎么解读 → `../vision/`
- 什么时候取像 → `../control/`
