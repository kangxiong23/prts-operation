"""§4 容器：槽集合 / 前提头 / 动作 / 轴 / 记录 / 观测迹条目 / 成分表 / 存疑清单条目 / 快照。

以 test_c_ 开头的那组是任务里的验收 c：**观测迹条目里取不出单独的"三态"或
"取值形态"** —— 它们只能从槽里拿。
以 test_d_ 开头的那组是验收 d：**存疑清单条目四个字段都在，且类别没有写死成
八个成员**。
"""

from __future__ import annotations

import dataclasses
from enum import Enum

import pytest

import ark.contracts as contracts
import ark.contracts.containers as containers
from ark.contracts import (
    Action,
    Axis,
    Composition,
    Contribution,
    DoubtEntry,
    Exact,
    FieldRef,
    LogicalFrame,
    NamedSlot,
    Premise,
    Record,
    SampledEmpty,
    SlotSet,
    Snapshot,
    SnapshotHeader,
    SourceRef,
    TraceEntry,
    Unknown,
    Unsampled,
    Valued,
    WallClockMs,
)


def _slot() -> Valued:
    return Valued(Exact(12), 0.87)


def _header() -> Premise:
    return Premise(
        slots=(NamedSlot(FieldRef("关卡"), Valued(Exact("1-7"), None)),)
    )


# ---------------------------------------------------------------------------
# c. 观测迹条目：帧号 + 字段名 + 槽，没有第二份三态
# ---------------------------------------------------------------------------


def test_c_trace_entry_has_only_frame_field_slot() -> None:
    """c. 只有三样：帧号 / 字段名 / 槽。三态与形态取不出单独的一份。"""
    entry = TraceEntry(frame=LogicalFrame(30), field=FieldRef("费用"), slot=_slot())
    assert [f.name for f in dataclasses.fields(TraceEntry)] == ["frame", "field", "slot"]
    for absent in ("state", "form", "value", "confidence", "payload", "shape", "slot_state"):
        assert not hasattr(entry, absent), f"观测迹条目不该有 {absent}"


def test_c_the_state_and_form_come_from_the_slot() -> None:
    """c. 要三态与形态就去槽里拿 —— 条目本身把它当整体收着。"""
    entry = TraceEntry(
        frame=LogicalFrame(30), field=FieldRef("费用"), slot=SampledEmpty(confidence=0.4)
    )
    assert isinstance(entry.slot, (Unsampled, SampledEmpty, Valued))
    assert isinstance(entry.slot, SampledEmpty)
    assert entry.slot.confidence == 0.4

    valued = TraceEntry(
        frame=LogicalFrame(31), field=FieldRef("费用"), slot=Valued(Unknown(), None)
    )
    assert isinstance(valued.slot, Valued)
    assert isinstance(valued.slot.payload, Unknown)
    assert valued.slot.confidence is None


def test_c_the_trace_entry_needs_a_slot_not_a_bare_number() -> None:
    """c. 只记数值的那种写法在这里构造不出来：槽是必填，且只能是三态之一。"""
    with pytest.raises(TypeError):
        TraceEntry(frame=LogicalFrame(30), field=FieldRef("费用"), slot=12)
    with pytest.raises(TypeError):
        TraceEntry(frame=LogicalFrame(30), field=FieldRef("费用"), slot=Exact(12))


# ---------------------------------------------------------------------------
# d. 存疑清单条目：四个字段，类别是一个字符串
# ---------------------------------------------------------------------------


def test_d_doubt_entry_has_the_four_fields() -> None:
    """d. 类别 + 涉及什么 + 严重度 + 涉及哪几帧，一个不多一个不少。"""
    assert [f.name for f in dataclasses.fields(DoubtEntry)] == [
        "category",
        "about",
        "severity",
        "frames",
    ]
    field_doubt = DoubtEntry(
        category="识别降级", about=FieldRef("费用"), severity="低", frames=(LogicalFrame(30),)
    )
    source_doubt = DoubtEntry(
        category="账不平", about=SourceRef("识别"), severity="高", frames=()
    )
    assert field_doubt.category == "识别降级"
    assert isinstance(field_doubt.about, FieldRef)
    assert isinstance(source_doubt.about, SourceRef)
    assert field_doubt.frames == (LogicalFrame(30),)
    with pytest.raises(TypeError):
        DoubtEntry(category="识别降级", about="费用", severity="低", frames=())
    with pytest.raises(TypeError):
        DoubtEntry(category="", about=FieldRef("费用"), severity="低", frames=())


def test_d_the_category_is_not_written_down_as_eight_members() -> None:
    """d. 类别收任何字符串（八类归状态层），这一层里没有八个成员的枚举。"""
    for category in ("识别降级", "账不平", "特效遮挡", "随便什么第八类"):
        entry = DoubtEntry(category=category, about=FieldRef("费用"), severity="低", frames=())
        assert entry.category == category

    for module in (containers, contracts):
        for name, value in vars(module).items():
            if isinstance(value, type) and issubclass(value, Enum):
                assert len(value) < 8, f"{name} 把存疑清单八类写死成了枚举"


# ---------------------------------------------------------------------------
# 帧戳属于容器，而且只许是逻辑帧
# ---------------------------------------------------------------------------


def test_frames_are_logical_frames_only() -> None:
    """§3：轴上只许出现逻辑帧。裸 int 与墙钟在这里都被拒（这不是"裸数不拦"）。"""
    with pytest.raises(TypeError):
        Action(frame=30, verb="部署", params=())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Action(frame=WallClockMs(30), verb="部署", params=())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TraceEntry(frame=WallClockMs(30), field=FieldRef("费用"), slot=_slot())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        DoubtEntry(category="账不平", about=SourceRef("识别"), severity="低", frames=(30,))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SnapshotHeader(
            frame=30,  # type: ignore[arg-type]
            time_state="常速",
            sampled_fields=(),
            end_reason="一段跑完",
            doubts=(),
            compositions={},
        )


# ---------------------------------------------------------------------------
# 槽集合 · 前提头 · 动作 · 轴 · 记录
# ---------------------------------------------------------------------------


def test_named_slot_holds_a_state_not_a_raw_value() -> None:
    assert NamedSlot(FieldRef("费用"), _slot()).slot == _slot()
    with pytest.raises(TypeError):
        NamedSlot(FieldRef("费用"), 12)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NamedSlot("费用", _slot())  # type: ignore[arg-type]


def test_slot_set_is_ordered_may_carry_a_block_and_forbids_duplicates() -> None:
    first = NamedSlot(FieldRef("费用"), _slot())
    second = NamedSlot(FieldRef("生命"), Valued(Exact(3), None))
    slot_set = SlotSet(block="HUD", slots=(first, second))
    assert [slot.name.name for slot in slot_set.slots] == ["费用", "生命"]
    assert SlotSet(block=None, slots=(first,)).block is None  # 可带 = 也可以不带
    with pytest.raises(ValueError):
        SlotSet(block="HUD", slots=(first, first))
    with pytest.raises(TypeError):
        SlotSet(block="HUD", slots=[first])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        SlotSet(block="", slots=(first,))


def test_axis_and_record_compose_header_actions_and_trace() -> None:
    header = _header()
    actions = (Action(LogicalFrame(30), "部署", (Exact("能天使"),)),)
    axis = Axis(header=header, actions=actions)
    trace = (TraceEntry(LogicalFrame(31), FieldRef("费用"), _slot()),)
    record = Record(header=header, actions=actions, trace=trace)

    assert axis.header == header
    assert record.actions == axis.actions
    assert record.trace[0].slot == _slot()
    with pytest.raises(TypeError):
        Axis(header=header, actions=list(actions))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Record(header=header, actions=actions, trace=(_slot(),))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 成分表 · 快照
# ---------------------------------------------------------------------------


def test_composition_marks_a_winner_that_must_be_in_the_table() -> None:
    contributions = (
        Contribution(SourceRef("识别"), 0.7),
        Contribution(SourceRef("推算"), 0.3),
    )
    composition = Composition(contributions=contributions, winner=SourceRef("识别"))
    assert composition.winner == contributions[0].source
    assert Composition(contributions=contributions, winner=None).winner is None
    with pytest.raises(ValueError):
        Composition(contributions=contributions, winner=SourceRef("记类"))
    with pytest.raises(TypeError):
        Contribution(SourceRef("识别"), "0.7")
    with pytest.raises(TypeError):
        Composition(contributions=contributions, winner="识别")  # type: ignore[arg-type]


def test_snapshot_header_has_the_six_items() -> None:
    """页眉六项（状态层/快照 ⓪）：帧戳 / 时间态 / 取样字段 / 结束原因 / 存疑 / 成分。"""
    assert [f.name for f in dataclasses.fields(SnapshotHeader)] == [
        "frame",
        "time_state",
        "sampled_fields",
        "end_reason",
        "doubts",
        "compositions",
    ]
    header = SnapshotHeader(
        frame=LogicalFrame(30),
        time_state="慢放",
        sampled_fields=(FieldRef("费用"),),
        end_reason="一段跑完",
        doubts=(DoubtEntry("账不平", SourceRef("识别"), "低", ()),),
        compositions={FieldRef("费用"): Composition((Contribution(SourceRef("识别"), 1.0),), None)},
    )
    assert header.time_state == "慢放"
    assert list(header.compositions) == [FieldRef("费用")]
    with pytest.raises(TypeError):
        header.compositions[FieldRef("费用")] = Composition((), None)  # 页眉交出去就不能改


def test_snapshot_is_header_plus_blocks_plus_a_precondition_pointer() -> None:
    header = SnapshotHeader(LogicalFrame(30), "常速", (), "一段跑完", (), {})
    precondition = _header()
    blocks = (
        SlotSet(block="HUD", slots=(NamedSlot(FieldRef("费用"), _slot()),)),
        SlotSet(block="敌人块", slots=()),
    )
    snapshot = Snapshot(header=header, blocks=blocks, precondition=precondition)

    assert snapshot.header is header
    assert [block.block for block in snapshot.blocks] == ["HUD", "敌人块"]
    assert snapshot.precondition is precondition  # "指针"就是同一个对象，不拷贝
    with pytest.raises(TypeError):
        Snapshot(header=header, blocks=list(blocks), precondition=precondition)  # type: ignore[arg-type]
