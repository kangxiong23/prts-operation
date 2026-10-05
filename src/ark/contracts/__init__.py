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
- scale.py       §3 刻度与时间态：逻辑帧与墙钟毫秒（跨刻度的构造 / 运算 /
                 比较一律拒绝）、时间态（三值，与 §2 的视图是两个类型）；
- geometry.py    §2 几何：格坐标 / 像素 / 画布 / 像素矩形 / 二维区域（Region，
                 它是第六种载荷）/ 视图与投影 / UI 布局常量；
- containers.py  §4 容器：槽集合 / 前提头（Premise）/ 动作 / 轴 / 记录 /
                 观测迹条目 / 成分表 / 存疑清单条目 / 快照；
- ports.py       §5 端口与错误：设备 / 时钟 / 样本 / 资产 四个端口（Protocol），
                 与拒绝 / 降级 / 中止三种错误。

方向是 slot → geometry：二维区域的**形状**在 §2，词表在 §1 的 PAYLOAD_TYPES。
ports 是**汇点**：它 import geometry / scale / containers（格坐标与画布、墙钟与
逻辑帧、中止与降级要用的容器），别人不 import 它 —— 端口的实现（适配器）住
engine 与 offline，由 runtime/wiring.py 接上。

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
    UI_LAYOUT,
    Canvas,
    squad_card_face_rect,
    squad_card_rect,
    GridCoord,
    Pixel,
    PixelRect,
    Projection,
    Region,
    UiLayout,
    View,
)
from .ports import (
    Abort,
    AssetPort,
    ClockPort,
    Degradation,
    DeviceFacts,
    DevicePort,
    Refusal,
    SamplePort,
)
from .scale import LOGICAL_FRAMES_PER_GAME_SECOND, LogicalFrame, TimeState, WallClockMs
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
    "UI_LAYOUT",
    "squad_card_rect",
    "squad_card_face_rect",
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
    # §3 时间态（T1.2b 补：它属于 §3，但排在名单末尾 —— 前面那 15 项是
    # T1.1 定下的前缀，一字不许动，其余按节往上加）
    "TimeState",
    # §5 端口与错误（DeviceFacts 是设备事实那条的返回形状，不是规格点名的
    # 那七个名字之一 —— 它比 tuple[Canvas, str] 好：契约层不给无名字的元组）
    "DevicePort",
    "ClockPort",
    "SamplePort",
    "AssetPort",
    "DeviceFacts",
    "Refusal",
    "Degradation",
    "Abort",
]
