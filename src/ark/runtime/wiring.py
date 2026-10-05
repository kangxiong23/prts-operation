"""**唯一的组装点**：把适配器接到端口上（代码结构 规则一 · 规则二）。

四个端口各接一个实现：

| 端口 | 实现 | 住哪 |
| --- | --- | --- |
| 设备 | `AdbDevice` | `engine/device/adb.py` |
| 时钟 | `SystemClock` | `engine/device/clock.py` |
| 样本 | `EmptySamplePort` | `offline/hardcase/port.py` |
| 资产 | `EmptyAssetPort` | `offline/assets_gen/port.py` |

**只有这里许 import 别人**（导入规则里 `ark.runtime` 是唯一不设限的那条）。`engine`
的六个子包互不 import，它们之间靠契约层的端口连 —— 接到一起是这里的事。

**`runtime → offline` 这条边是真的用上了**，用的正是资产端口存在的那条理由：
`engine` 永远不 import `offline`，所以那些值只能经契约层到达、由这里接上。

## 视图是装配参数，不是端口的开关

`build_wiring` 多收一个 `view`（T5 裁定三）：设备层要像素时向 `assets.projection(view)`
要，而投影有**常速与慢放两套**。**切视图 = 重新装配** —— 那正是规则二那句
「换一套参数 = 换一个适配器」；给端口加一个「我在慢放」的开关会把它破掉。

## `PORT_GAPS`：没接的必须是写下来的

一个端口方法**这一轮故意不接**时，必须在这里留一条。它与
`tests/contract/port_completeness.py` 是一对：**测试问「接上了吗」，登记答
「没接的是哪几条、为什么」**。「还没接」必须是**写下来的**，不许是「忘了」
（代码结构 §尚未落地）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ark.contracts import AssetPort, ClockPort, DevicePort, SamplePort, View
from ark.engine.device.adb import AdbDevice, AdbTransport
from ark.engine.device.clock import SystemClock
from ark.offline.assets_gen.port import EmptyAssetPort
from ark.offline.hardcase.port import EmptySamplePort

__all__ = ["Wiring", "build_wiring", "PORT_GAPS"]


@dataclass(frozen=True)
class Wiring:
    """装好的四个端口。**一个不缺**，缺的写进 `PORT_GAPS`（不是写成 `None`）。"""

    device: DevicePort
    clock: ClockPort
    samples: SamplePort
    assets: AssetPort

    def __post_init__(self) -> None:
        for name in ("device", "clock", "samples", "assets"):
            if getattr(self, name) is None:
                raise TypeError(
                    f"{name} 那个端口没有实现者：缺的那一条要写进 PORT_GAPS，"
                    "不许用 None 蒙过去"
                )


def build_wiring(transport: AdbTransport, *, view: View = View.NORMAL) -> Wiring:
    """把四个端口接上，返回装好的 `Wiring`。

    **设备那条缝从外面进来**：`transport` 是参数，不是这里 new 出来的 —— 离线验收
    要能换掉它（换一个适配器，不是给端口加一个「我在重放」的开关）。

    **`view` 是「此刻的视图」**：同一个资产端口实例同时交给 `Wiring.assets` 与
    设备层，所以两边看到的投影是**同一套**。
    """
    assets = EmptyAssetPort()
    return Wiring(
        device=AdbDevice(transport, assets, view),
        clock=SystemClock(),
        samples=EmptySamplePort(),
        assets=assets,
    )


PORT_GAPS: tuple[str, ...] = (
    "AssetPort（资产端口）· projection 未接：投影的值住在关卡资产里，"
    "而标定是首轮探索编辑器的第一块 —— 这一轮没有真值，编一个出来就是假事实 · "
    "归段一 / 段四",
)
"""这一轮**故意没接**的端口方法，每条一句话：哪个端口 · 缺什么 · 归哪一轮。

**设备端口一条都没有**（T5 裁定六）：取一帧 · 四个动词 · 设备事实，六个方法全接上了。

资产端口那一条是**「方法不存在」**（T5 裁定四的新口径），不是「方法在、一调就炸」：
代码结构 §尚未落地 说「留空 = 方法不存在」，而 `port_completeness` 的判据是**方法级
的** —— 一个在但一调就炸的方法会让「接上了」变成假的。所以 `EmptyAssetPort` 一个
公开方法都没有，欠账写在这里。
"""
