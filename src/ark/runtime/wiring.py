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

**`runtime → offline` 这条边在这一轮第一次真的被用上**，用的正是资产端口存在的
那条理由：`engine` 永远不 import `offline`，所以那些值只能经契约层到达、
由这里接上。

## `PORT_GAPS`：没接的必须是写下来的

一个端口方法**这一轮故意不接**时，必须在这里留一条。它与
`tests/contract/port_completeness.py` 是一对：**测试问「接上了吗」，登记答
「没接的是哪几条、为什么」**。「还没接」必须是**写下来的**，不许是「忘了」
（代码结构 §尚未落地）。
"""

from __future__ import annotations

from dataclasses import dataclass

from ark.contracts import AssetPort, ClockPort, DevicePort, SamplePort
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


def build_wiring(transport: AdbTransport) -> Wiring:
    """把四个端口接上，返回装好的 `Wiring`。

    **设备那条缝从外面进来**：`transport` 是参数，不是这里 new 出来的 —— 离线验收
    要能换掉它（换一个适配器，不是给端口加一个「我在重放」的开关）。
    """
    return Wiring(
        device=AdbDevice(transport),
        clock=SystemClock(),
        samples=EmptySamplePort(),
        assets=EmptyAssetPort(),
    )


PORT_GAPS: tuple[str, ...] = (
    "DevicePort（设备端口）· inject_gesture 未接："
    "部署的起点是备战席那张卡，而「哪张卡」不在签名里；facing 又对撤退 / 开技能 / "
    "关技能不适用 —— 修法（加起点 / 拆方法 / 换形状）未裁定 · 归 T4.1",
    "DevicePort（设备端口）· device_facts 未接："
    "orientation 没有任何一处规定读哪条命令，时序层 §3 的前提头清单里也没有它 —— "
    "「方向从哪来」未裁定 · 归 T4.1",
)
"""这一轮**故意没接**的端口方法，每条一句话：哪个端口 · 缺什么 · 归哪一轮。

两条都是**卡在没裁定**，不是卡在没时间：`inject_gesture` 的签名不够（部署表达
不了），`device_facts` 缺一半（方向没有来源）。自己补一个出来就是发明规格。
"""
