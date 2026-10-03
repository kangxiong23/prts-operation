"""拆叶子之后的公开面：from ark.contracts import X 逐项对得上。

叶子切法见 docs/10-规格/代码结构.md 的目录树：contracts/ 按《契约层》的节开叶子
—— slot.py（§1）· geometry.py（§2）· scale.py（§3）· containers.py（§4）·
ports.py（§5 端口与错误）。五个节各一个叶子，再没有别的。

EXPECTED_ALL 是**拆分叶子之前（T1.1）那 15 项**，逐项抄下来的，顺序也是原样：
它必须仍然是 __all__ 的前缀 —— 新名字一律往后面加。
"""

from __future__ import annotations

import pathlib
import types

import ark.contracts as contracts
import ark.contracts.containers as containers
import ark.contracts.geometry as geometry
import ark.contracts.ports as ports
import ark.contracts.scale as scale
import ark.contracts.slot as slot

EXPECTED_ALL = [
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
]

GEOMETRY_NAMES = [
    "GridCoord",
    "Pixel",
    "Canvas",
    "PixelRect",
    "Region",
    "View",
    "Projection",
    "UiLayout",
]

CONTAINER_NAMES = [
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

# §3 后补的一支（T1.2b）：它排在名单末尾，不在上面两组的顺序里。
SCALE_EXTRA_NAMES = ["TimeState"]

# §5 端口与错误：四个端口 + 三种错误（DeviceFacts 是设备事实那条的返回形状，
# 它不在规格点名的那七个名字里 —— 见叶子的文档字符串）。
PORTS_NAMES = [
    "DevicePort",
    "ClockPort",
    "SamplePort",
    "AssetPort",
    "DeviceFacts",
    "Refusal",
    "Degradation",
    "Abort",
]

LEAVES = (slot, scale, geometry, containers, ports)


def _home(name: str):
    """这个名字住在哪个叶子里。

    看叶子的 __all__，不看命名空间：叶子会 import 别的东西（containers 就
    import 了 LogicalFrame），按命名空间找会找出两个家。
    """
    value = getattr(contracts, name)
    homes = [leaf for leaf in LEAVES if name in getattr(leaf, "__all__", ())]
    assert len(homes) == 1, f"{name} 不是恰好住在一个叶子的 __all__ 里"
    assert getattr(homes[0], name) is value, f"{name} 再导出的不是叶子里那一个对象"
    return homes[0]


def test_the_t11_surface_is_still_the_prefix() -> None:
    """老 15 项一处不少、顺序不变 —— 新名字只许往后面加。"""
    assert contracts.__all__[: len(EXPECTED_ALL)] == EXPECTED_ALL


def test_new_names_are_appended_after_the_old_ones() -> None:
    assert contracts.__all__[len(EXPECTED_ALL) :] == (
        GEOMETRY_NAMES + CONTAINER_NAMES + SCALE_EXTRA_NAMES + PORTS_NAMES
    )


def test_every_public_name_is_importable_from_the_package() -> None:
    for name in contracts.__all__:
        assert getattr(contracts, name) is getattr(_home(name), name)


def test_each_leaf_is_the_home_of_its_own_names() -> None:
    for name in GEOMETRY_NAMES:
        assert _home(name) is geometry
    for name in CONTAINER_NAMES:
        assert _home(name) is containers
    for name in ("LOGICAL_FRAMES_PER_GAME_SECOND", "LogicalFrame", "WallClockMs", "TimeState"):
        assert _home(name) is scale
    for name in ("Exact", "Payload", "Unsampled", "SlotValue", "SLOT_STATES"):
        assert _home(name) is slot
    for name in PORTS_NAMES:
        assert _home(name) is ports


def test_every_leaf_is_importable_by_name() -> None:
    from ark.contracts.containers import Snapshot
    from ark.contracts.geometry import GridCoord
    from ark.contracts.ports import DevicePort, Refusal
    from ark.contracts.scale import LogicalFrame, TimeState as LeafTimeState
    from ark.contracts.slot import Valued

    assert contracts.Snapshot is Snapshot
    assert contracts.GridCoord is GridCoord
    assert contracts.LogicalFrame is LogicalFrame
    assert contracts.TimeState is LeafTimeState
    assert contracts.Valued is Valued
    assert contracts.DevicePort is DevicePort
    assert contracts.Refusal is Refusal


def test_the_five_leaves_exist_and_nothing_else() -> None:
    """按节开出的五个叶子（§1 §2 §3 §4 §5）—— 多一个都是没跟着节走。"""
    leaves = sorted(
        path.stem
        for path in pathlib.Path(contracts.__file__).parent.glob("*.py")
        if path.stem != "__init__"
    )
    assert leaves == ["containers", "geometry", "ports", "scale", "slot"]


def test_all_has_no_duplicates() -> None:
    assert len(contracts.__all__) == len(set(contracts.__all__))
    assert len(contracts.__all__) == (
        len(EXPECTED_ALL)
        + len(GEOMETRY_NAMES)
        + len(CONTAINER_NAMES)
        + len(SCALE_EXTRA_NAMES)
        + len(PORTS_NAMES)
    )


def test_re_export_does_not_pull_in_other_layers() -> None:
    """契约层自己谁也不 import：这个包里出现的模块只能来自 ark.contracts。"""
    modules = [value for value in vars(contracts).values() if isinstance(value, types.ModuleType)]
    outside = sorted(
        module.__name__ for module in modules if not module.__name__.startswith("ark.contracts")
    )
    assert outside == [], f"契约层里出现了别的模块：{outside}"
    assert sorted(module.__name__ for module in modules) == [
        "ark.contracts.containers",
        "ark.contracts.geometry",
        "ark.contracts.ports",
        "ark.contracts.scale",
        "ark.contracts.slot",
    ]
