"""契约层：只有数据定义与端口，没有任何实现。

本包是唯一被所有人 import、且自己不 import 任何内部模块的东西。这条一旦破，
解耦就是摆设 —— 由 tests/contract/import_rules.py 机器强制。

这一层唯一的验收判据：让"不知道"在类型上没法被伪装成"知道"。
可验的问法：拿一份快照问"这里面哪几个数其实是猜的"，答案必须能在类型里
直接看出来，不用读实现。

叶子按《契约层》的节切（见 docs/10-规格/代码结构.md 的目录树）：

- slot.py        §1 槽：三态 × 载荷 × 置信度。三态是三支不同的类型，
                 载荷的六种形态是六个类型；区间 / 下界 / 上界 共用同一套操作
                 （包含 / 相交）也在这里 —— §2 的标量区间不是第四个类型；
- scale.py       §3 刻度：逻辑帧与墙钟毫秒。跨刻度的构造 / 运算 / 比较一律拒绝；
- geometry.py    §2 几何：格坐标 / 像素 / 画布 / 像素矩形 / 二维区域（Region，
                 它是第六种载荷）/ 视图与投影 / UI 布局常量；
- containers.py  §4 容器：槽集合 / 前提头（Premise）/ 动作 / 轴 / 记录 /
                 观测迹条目 / 成分表 / 存疑清单条目 / 快照。

方向是 slot → geometry：二维区域的**形状**在 §2，词表在 §1 的 PAYLOAD_TYPES。
端口与错误（§5）那一节随自己落地时再建文件，**不预建**。

这个 __init__ 只做再导出：T1.1 定下的那 15 个名字原样排在 __all__ 最前面，
后面按节往上加 —— 名单只增不减，接口才稳。
"""

from .containers import (
    Action,
    Axis,
    Composition,
    Contribution,
    DoubtEntry,
    DoubtTarget,
    FieldRef,
    NamedSlot,
    Premise,
    Record,
    SlotSet,
    Snapshot,
    SnapshotHeader,
    SourceRef,
    TraceEntry,
)
from .geometry import (
    Canvas,
    GridCoord,
    Pixel,
    PixelRect,
    Projection,
    Region,
    UiLayout,
    View,
)
from .scale import LOGICAL_FRAMES_PER_GAME_SECOND, LogicalFrame, WallClockMs
from .slot import (
    PAYLOAD_TYPES,
    SLOT_STATES,
    Exact,
    Interval,
    LowerBound,
    Payload,
    SampledEmpty,
    SlotValue,
    Unknown,
    Unsampled,
    UpperBound,
    Valued,
)

# 前 15 项是拆分叶子之前（T1.1）那一份 __all__，顺序原样在最前面：
# tests/unit/test_contracts_package.py 盯着这个前缀。新名字一律往后面加。
__all__ = [
    "LOGICAL_FRAMES_PER_GAME_SECOND",
    "LogicalFrame",
    "WallClockMs",
    "Exact",
    "Interval",
    "LowerBound",
    "UpperBound",
    "Unknown",
    "Payload",
    "PAYLOAD_TYPES",
    "Unsampled",
    "SampledEmpty",
    "Valued",
    "SlotValue",
    "SLOT_STATES",
    # §2 几何
    "GridCoord",
    "Pixel",
    "Canvas",
    "PixelRect",
    "Region",
    "View",
    "Projection",
    "UiLayout",
    # §4 容器
    "FieldRef",
    "SourceRef",
    "DoubtTarget",
    "NamedSlot",
    "SlotSet",
    "Premise",
    "Action",
    "Axis",
    "TraceEntry",
    "Record",
    "Contribution",
    "Composition",
    "DoubtEntry",
    "SnapshotHeader",
    "Snapshot",
]
