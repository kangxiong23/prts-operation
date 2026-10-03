"""§4 容器：定义"什么叫容器"，**不定义"里面有哪些名字"**。

九个容器的形状来自《契约层》§4：槽集合 · 前提头 · 动作 · 轴 · 记录 · 观测迹
条目 · 成分表 · 存疑清单条目 · 快照。

两条绷得最紧的线：

- **里面有哪些字段名、哪些块名不在这里**（状态层与时序层的事，见分界线表）：
  这一层只收**名字**（字符串），不收名字的**词表**。存疑清单的类别尤其如此 ——
  它是**一个字符串**，不是八个成员的枚举；严重度的取值同样不在这里发明。
- **槽不逐条带帧戳**：帧戳属于容器（快照页眉 / 观测迹条目），一片里几十个槽
  共用一个时刻。观测迹条目记的是**整个槽**，不是"数值" —— 三态与形态长在槽里，
  条目里取不出单独的一份。

帧戳一律是**逻辑帧**（§3：轴上只许出现逻辑帧）：动作、观测迹条目、存疑清单
的"涉及哪几帧"、快照页眉都是。这里连裸 int 都不收 —— 那不是刻度那边"裸数
不拦"的边界，那是"只许出现逻辑帧"这条规则的落点。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias, final

from ark.contracts.scale import LogicalFrame
from ark.contracts.slot import SLOT_STATES, SlotValue

__all__ = [
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


def _is_number(value: object) -> bool:
    """实数（int 或 float）。bool 不算。与 slot / geometry 里同一条判据。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_text(value: object, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{what}是非空字符串，收到 {value!r}")
    return value


def _check_tuple_of(value: object, item: type | tuple[type, ...], what: str) -> tuple:
    """容器里的有序成员：**只收 tuple**，不替调用方把 list 转过去。

    不转的理由：同一份数据在库里只该有一种可变性。要序列化/构造时自己转一次，
    别让"我手里的这个 list 改了它会不会跟着变"成为一个问题。
    """
    if not isinstance(value, tuple):
        raise TypeError(f"{what}要用 tuple，收到 {type(value).__name__}")
    for element in value:
        if not isinstance(element, item):
            raise TypeError(f"{what}里只能是 {item}，收到 {type(element).__name__}")
    return value


def _check_unique_names(slots: tuple) -> None:
    """同一个集合里重名就是自相矛盾：同片内两处字段不一致的口子。"""
    names = [slot.name.name for slot in slots]
    if len(names) != len(set(names)):
        duplicated = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"同一个集合里的槽不许重名，重复的是：{duplicated}")


# ---------------------------------------------------------------------------
# 名字：不透明的两类
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class FieldRef:
    """字段名。**不透明**：有哪些字段名归状态层（分界线表），这里只收名字。"""

    name: str

    def __post_init__(self) -> None:
        _check_text(self.name, "字段名")


@final
@dataclass(frozen=True)
class SourceRef:
    """来源标签。**不透明**：来源有哪些归状态层。"""

    label: str

    def __post_init__(self) -> None:
        _check_text(self.label, "来源标签")


DoubtTarget: TypeAlias = FieldRef | SourceRef
"""存疑清单的"涉及什么"：**要么是一个字段，要么是一个来源**（§4）。

两种用两个类型表达，而不是"一个字符串 + 一个类型字段"——后者正是这一层
一路在拆的那种并列判别字段。
"""


# ---------------------------------------------------------------------------
# 槽集合 · 前提头
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class NamedSlot:
    """具名槽：名字 + 槽本身（三态 × 载荷 × 置信度）。

    槽的取值只能是三态之一 —— 这里不做第二套三态。名字是**字段名**，不是
    "里面有哪些字段"。
    """

    name: FieldRef
    slot: SlotValue

    def __post_init__(self) -> None:
        if not isinstance(self.name, FieldRef):
            raise TypeError(f"具名槽的名字要是 FieldRef，收到 {type(self.name).__name__}")
        if not isinstance(self.slot, SLOT_STATES):
            raise TypeError(f"具名槽的取值只能是三态之一，收到 {type(self.slot).__name__}")


@final
@dataclass(frozen=True)
class SlotSet:
    """槽集合：具名槽的**有序**集合，**可带块名**（块名由状态层给）。

    块名是收进来的字符串，不是这里枚举出来的那几个块 —— "里面有哪些块名"
    不归这一层。同一个集合里重名即自相矛盾，构造时就拒。
    """

    block: str | None
    slots: tuple[NamedSlot, ...]

    def __post_init__(self) -> None:
        if self.block is not None:
            _check_text(self.block, "块名")
        _check_tuple_of(self.slots, NamedSlot, "槽集合的槽")
        _check_unique_names(self.slots)


@final
@dataclass(frozen=True)
class Premise:
    """前提头（Premise）：一局内不变的那几个槽（时序层 §3）。

    与槽集合分开：前提头**不分块**（块是快照的组织方式），所以这里没有块名。
    一局内不变也意味着它值得被快照"指"着而不是每片抄一份。
    """

    slots: tuple[NamedSlot, ...]

    def __post_init__(self) -> None:
        _check_tuple_of(self.slots, NamedSlot, "前提头的槽")
        _check_unique_names(self.slots)


# ---------------------------------------------------------------------------
# 动作 · 轴 · 观测迹 · 记录
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class Action:
    """动作：帧号 + 动词 + 参数…（动词表在时序层 §4）。

    - 帧号**只许是逻辑帧**：裸 int 与墙钟都在这里被拒（§3：轴上只许出现逻辑帧）；
    - 参数的具体形状由动词表定，这里只收一个有序的位置参数表，**不预设**。
    """

    frame: LogicalFrame
    verb: str
    params: tuple[object, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, LogicalFrame):
            raise TypeError(
                f"动作的帧号只许是逻辑帧，收到 {type(self.frame).__name__}："
                "墙钟只在记录里留痕"
            )
        _check_text(self.verb, "动词")
        _check_tuple_of(self.params, object, "动作的参数")


@final
@dataclass(frozen=True)
class TraceEntry:
    """观测迹条目：帧号 + 字段名 + 槽。

    **只有这三样。** 三态与形态取不出单独的一份 —— 它们长在槽里（§4：观测迹
    条目记的是整个槽，不是"数值"；只记数值会让重建出来的快照没有置信度）。
    时序层原来那个 (帧号, 字段, 新值, 取值形态) 因此自然是三元组。
    """

    frame: LogicalFrame
    field: FieldRef
    slot: SlotValue

    def __post_init__(self) -> None:
        if not isinstance(self.frame, LogicalFrame):
            raise TypeError(f"观测迹条目的帧号只许是逻辑帧，收到 {type(self.frame).__name__}")
        if not isinstance(self.field, FieldRef):
            raise TypeError(f"观测迹条目的字段名要是 FieldRef，收到 {type(self.field).__name__}")
        if not isinstance(self.slot, SLOT_STATES):
            raise TypeError(f"观测迹条目的槽只能是三态之一，收到 {type(self.slot).__name__}")


@final
@dataclass(frozen=True)
class Axis:
    """轴：前提头 + 动作表。"""

    header: Premise
    actions: tuple[Action, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.header, Premise):
            raise TypeError(f"轴的前提头要是 Premise，收到 {type(self.header).__name__}")
        _check_tuple_of(self.actions, Action, "轴的动作表")


@final
@dataclass(frozen=True)
class Record:
    """记录：前提头 + 实际动作表 + 观测迹。"""

    header: Premise
    actions: tuple[Action, ...]
    trace: tuple[TraceEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.header, Premise):
            raise TypeError(f"记录的前提头要是 Premise，收到 {type(self.header).__name__}")
        _check_tuple_of(self.actions, Action, "记录的动作表")
        _check_tuple_of(self.trace, TraceEntry, "记录的观测迹")


# ---------------------------------------------------------------------------
# 成分表 · 存疑清单条目
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class Contribution:
    """成分表里的一条：来源标签 + 贡献量。

    贡献量是数，**不为负没有规定**：规格没说惩罚不许为负，不发明。
    """

    source: SourceRef
    amount: int | float

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceRef):
            raise TypeError(f"贡献的来源要是 SourceRef，收到 {type(self.source).__name__}")
        if not _is_number(self.amount):
            raise TypeError(f"贡献量必须是数，收到 {type(self.amount).__name__}")


@final
@dataclass(frozen=True)
class Composition:
    """成分表：一个值的各来源贡献；**取最高时标出胜者**。

    winner 是胜者的**来源标签**（不是一份拷贝）：给了它就必须能在贡献里找到
    同一个标签 —— 造不出"胜者不在表里"的成分表。winner=None 表示没有取最高
    （还没定 / 不需要）。
    """

    contributions: tuple[Contribution, ...]
    winner: SourceRef | None

    def __post_init__(self) -> None:
        _check_tuple_of(self.contributions, Contribution, "成分表的贡献")
        if self.winner is not None:
            if not isinstance(self.winner, SourceRef):
                raise TypeError(f"胜者要是 SourceRef，收到 {type(self.winner).__name__}")
            if all(c.source != self.winner for c in self.contributions):
                raise ValueError(f"标出的胜者 {self.winner.label!r} 不在贡献表里")


@final
@dataclass(frozen=True)
class DoubtEntry:
    """存疑清单条目：**类别 + 涉及什么（字段 / 来源） + 严重度 + 涉及哪几帧**（§4）。

    **类别是一个字符串，不是八个成员的枚举** —— 八类是哪八类归状态层（分界线
    表这一层只管条目的形状，不管类别取哪八个值）。严重度的取值同样不在这里
    发明。涉及哪几帧用**逻辑帧**：页眉与观测迹都是逻辑帧。
    """

    category: str
    about: DoubtTarget
    severity: str
    frames: tuple[LogicalFrame, ...]

    def __post_init__(self) -> None:
        _check_text(self.category, "存疑清单的类别")
        if not isinstance(self.about, (FieldRef, SourceRef)):
            raise TypeError(
                "存疑清单的「涉及什么」要么是字段要么是来源，收到 "
                f"{type(self.about).__name__}"
            )
        _check_text(self.severity, "严重度")
        _check_tuple_of(self.frames, LogicalFrame, "存疑清单涉及的帧")


# ---------------------------------------------------------------------------
# 快照
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class SnapshotHeader:
    """快照页眉：**六项，一项不少**（状态层/快照 ⓪）。

    1. frame            逻辑帧戳
    2. time_state       当前时间态（常速 / 慢放 / 暂停）——**取值归控制层**，
                        这一层只收一个字符串，不在这里发明它的枚举
    3. sampled_fields   本片实际取了哪些字段
    4. end_reason       本段结束原因（取值归控制层 / 时序层）
    5. doubts           存疑清单（条目形状见 DoubtEntry）
    6. compositions     多来源字段的成分（字段 → 成分表）

    这六项是**形状**；每一块的字段名、块名、类别词表都不在这里（分界线表）。
    """

    frame: LogicalFrame
    time_state: str
    sampled_fields: tuple[FieldRef, ...]
    end_reason: str
    doubts: tuple[DoubtEntry, ...]
    compositions: Mapping[FieldRef, Composition]

    # 映射字段让它不可哈希（与 UiLayout 同一个理由）。
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, LogicalFrame):
            raise TypeError(f"页眉的逻辑帧戳只许是逻辑帧，收到 {type(self.frame).__name__}")
        _check_text(self.time_state, "页眉的当前时间态")
        _check_tuple_of(self.sampled_fields, FieldRef, "页眉的取样字段")
        _check_text(self.end_reason, "页眉的结束原因")
        _check_tuple_of(self.doubts, DoubtEntry, "页眉的存疑清单")
        if not isinstance(self.compositions, Mapping):
            raise TypeError(f"页眉的成分要是映射，收到 {type(self.compositions).__name__}")
        for field, composition in self.compositions.items():
            if not isinstance(field, FieldRef):
                raise TypeError(f"页眉成分的键要是 FieldRef，收到 {type(field).__name__}")
            if not isinstance(composition, Composition):
                raise TypeError(f"页眉成分的值要是 Composition，收到 {type(composition).__name__}")
        object.__setattr__(self, "compositions", MappingProxyType(dict(self.compositions)))


@final
@dataclass(frozen=True)
class Snapshot:
    """快照：**页眉 + 按块组织的槽集合 + 前提头指针**。

    快照原来只写了"槽集合 + 前提头指针"，**漏了页眉**（修正过一次）：帧戳、
    时间态、取样字段、结束原因、存疑清单、成分都在页眉里。

    "指针"在 Python 里就是**持有同一个前提头对象**（不做拷贝）——这一层不发明
    指针的编号方案。一块一份槽集合，块的顺序就是这里 tuple 的顺序。
    """

    header: SnapshotHeader
    blocks: tuple[SlotSet, ...]
    precondition: Premise

    def __post_init__(self) -> None:
        if not isinstance(self.header, SnapshotHeader):
            raise TypeError(f"快照的页眉要是 SnapshotHeader，收到 {type(self.header).__name__}")
        _check_tuple_of(self.blocks, SlotSet, "快照的块")
        if not isinstance(self.precondition, Premise):
            raise TypeError(
                f"快照的前提头指针指向 Premise，收到 {type(self.precondition).__name__}"
            )
