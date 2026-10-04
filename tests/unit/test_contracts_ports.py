"""§5 端口与错误：四个端口（Protocol）+ 三种错误。

规格出处：docs/10-规格/契约层/端口与错误.md（唯一权威）· 设备层 §5 ·
接口层/执行与失败 §4 · 代码结构（端口完备性测试还不许写：一个适配器都没有）。

这里盯的是**边界本身**，不是行为：四个端口各自有几个方法、签名收什么、错误三种
是不是三种数据、有没有哪一处把"谁在实现它"混进来。
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Protocol

import pytest

import ark.contracts as contracts
import ark.contracts.ports as ports
from ark.contracts import (
    Abort,
    Action,
    AssetPort,
    Canvas,
    ClockPort,
    Degradation,
    DeviceFacts,
    DevicePort,
    DoubtEntry,
    FieldRef,
    GridCoord,
    LogicalFrame,
    Refusal,
    SamplePort,
    WallClockMs,
)

PORTS = (DevicePort, ClockPort, SamplePort, AssetPort)
ERRORS = (Refusal, Degradation, Abort)


def _public_names(obj: object) -> list[str]:
    """一条协议对外露出来的方法名 —— 与规格验收用的是同一条口径（dir 去下划线）。"""
    return sorted(name for name in dir(obj) if not name.startswith("_"))


def _annotations(func: object) -> str:
    return str(getattr(func, "__annotations__", {}))


# ---------------------------------------------------------------------------
# 四个端口：都是 Protocol，各自方法数写死
# ---------------------------------------------------------------------------


def test_the_four_ports_are_protocols() -> None:
    """§5：端口有四个 —— 设备 · 时钟 · 样本 · 资产。"""
    assert [port.__name__ for port in PORTS] == [
        "DevicePort",
        "ClockPort",
        "SamplePort",
        "AssetPort",
    ]
    for port in PORTS:
        assert issubclass(port, Protocol), f"{port.__name__} 应当是 Protocol"
        assert getattr(port, "_is_protocol", False), f"{port.__name__} 应当是 Protocol"


def test_device_port_has_exactly_six_methods() -> None:
    """设备层 §1 + §5：**对外面就是轴的动词表** —— 取一帧 · 设备事实 · 四个动词。"""
    assert _public_names(DevicePort) == [
        "deploy",
        "device_facts",
        "grab_frame",
        "retreat",
        "skill_off",
        "skill_on",
    ]


def test_the_gesture_signature_is_gone() -> None:
    """设备层 §5：**曾经写作 `inject_gesture(at, facing)`，已改**。

    那个签名是**手势那一层**的，而且它表达不了部署（起点是备战席那张卡，
    "哪张卡"不在签名里），`facing` 又对撤退一类不适用。
    """
    assert not hasattr(DevicePort, "inject_gesture")


def test_clock_port_has_one_method_and_it_is_wall_clock() -> None:
    """契约层 §3：时钟端口只给**墙钟毫秒**；**逻辑帧不在这个端口里**。"""
    assert _public_names(ClockPort) == ["wall_clock_ms"]
    assert "frame" not in "wall_clock_ms"
    annotation = _annotations(ClockPort.wall_clock_ms)
    assert "WallClockMs" in annotation
    # 逻辑帧这个类型一提都不许提 —— 它由费用推算，不是从时钟读来的
    assert "LogicalFrame" not in annotation


def test_sample_port_has_exactly_one_method() -> None:
    """§5：样本端口方法集 = 一个方法（落一份样本）。"""
    assert _public_names(SamplePort) == ["submit"]


def test_asset_port_has_one_method_and_carries_the_rule() -> None:
    """§5：资产端口**方法集由第一个用户层补**，现在有第一个：`projection`。

    补法规则必须写在它的文档字符串里，否则下一个人会在这一层替别的层发明类型。
    """
    assert _public_names(AssetPort) == ["projection"]
    # 文档字符串会按行折行，所以把空白全抹掉再对句子 —— 折行不该让一条规则"查不到"
    doc = "".join((AssetPort.__doc__ or "").split())
    for phrase in ("第一个用户层", "只加方法", "不改端口本身"):
        assert phrase in doc, f"资产端口的文档字符串缺了补法规则的这句：{phrase}"


# ---------------------------------------------------------------------------
# 设备端口：格坐标与朝向，不是像素；取像返回原始帧
# ---------------------------------------------------------------------------


def test_deploy_takes_card_grid_and_facing_not_pixel() -> None:
    """设备层 §1 + §2：部署的参数是**第几张卡 · 格坐标 · 朝向**，不是像素。"""
    signature = inspect.signature(DevicePort.deploy)
    assert list(signature.parameters) == ["self", "card", "at", "facing"]
    assert signature.return_annotation == "None"
    annotation = _annotations(DevicePort.deploy)
    assert "GridCoord" in annotation
    assert "Pixel" not in annotation, f"部署的签名里出现了像素：{annotation}"


def test_deploy_parameters_are_all_required() -> None:
    """朝向**必填**（时序层 §4）：不选朝向 = 一次未完成的放置，所以没有默认值。

    卡下标与格坐标同理 —— 它们都是"这次动作到底做什么"，缺一个就不是同一个动作。
    """
    parameters = inspect.signature(DevicePort.deploy).parameters
    for name in ("card", "at", "facing"):
        assert parameters[name].default is inspect.Parameter.empty, name


def test_only_deploy_has_a_facing() -> None:
    """设备层 §5：**只有这一个方法有朝向** —— 撤退 / 开技能 / 关技能没有这回事。"""
    for name in ("retreat", "skill_on", "skill_off"):
        parameters = inspect.signature(getattr(DevicePort, name)).parameters
        assert list(parameters) == ["self", "at"], name


def test_grab_frame_returns_raw_frame() -> None:
    """设备层 §4：取像端口返回**原始帧**，不是归一化帧（所以这一层不提画布）。"""
    annotation = _annotations(DevicePort.grab_frame)
    assert "bytes" in annotation
    assert "Canvas" not in annotation
    assert "Pixel" not in annotation


def test_device_facts_is_resolution_and_orientation() -> None:
    """设备层 §5：设备事实 = 分辨率 · 方向。"""
    assert [field.name for field in dataclasses.fields(DeviceFacts)] == [
        "resolution",
        "orientation",
    ]
    facts = DeviceFacts(Canvas(1920, 864), "landscape")
    assert facts.resolution == Canvas(1920, 864)
    with pytest.raises(TypeError):
        DeviceFacts((1920, 864), "landscape")  # 裸元组不是画布
    with pytest.raises(TypeError):
        DeviceFacts(Canvas(1920, 864), "")  # 方向是非空字符串


# ---------------------------------------------------------------------------
# 端口必须"纯"到 replay 适配器能满足
# ---------------------------------------------------------------------------


def test_ports_do_not_name_who_implements_them() -> None:
    """§5：端口里**不许出现"现在真的是游戏在跑"这类信息**。

    判据落在这儿：签名与参数名里不许出现任何"实现侧"的词。文档字符串里可以提
    replay（"离线重放不是特例代码"正是在说它），但**类型上**不许有一丝。
    """
    leaky = ("adb", "screencap", "input", "engine", "offline", "replay", "game", "live")
    for port in PORTS:
        for name in _public_names(port):
            member = getattr(port, name)
            haystack = (_annotations(member) + " " + str(inspect.signature(member))).lower()
            for word in leaky:
                assert word not in haystack, f"{port.__name__}.{name} 签名里漏进了 {word!r}"


def test_ports_do_not_leak_into_each_other() -> None:
    """四个端口互不引用：一个端口的方法签名里不许出现另一个端口的名字。"""
    names = {port.__name__ for port in PORTS}
    for port in PORTS:
        for name in _public_names(port):
            annotation = _annotations(getattr(port, name))
            for other in sorted(names - {port.__name__}):
                assert other not in annotation, f"{port.__name__}.{name} 提到了 {other}"


# ---------------------------------------------------------------------------
# 三种错误：三种数据，各自带自己那份
# ---------------------------------------------------------------------------


def test_errors_are_three_distinct_frozen_data_types() -> None:
    """§5：错误三种（拒绝 / 降级 / 中止）—— 三种**类型**，不许合成一个。

    它们不是异常：降级本来就"照常返回"，拒绝与中止是**回答的形状**。没有共同
    基类 —— 有基类就会有人写一个 except 把它们一口吞掉。
    """
    assert len({Refusal, Degradation, Abort}) == 3
    for error in ERRORS:
        assert dataclasses.is_dataclass(error), f"{error.__name__} 应当是数据类"
        assert not issubclass(error, BaseException), f"{error.__name__} 不该是异常"
        assert error.__bases__ == (object,), f"{error.__name__} 不该有共同基类"
        instance = _make(error)
        with pytest.raises(dataclasses.FrozenInstanceError):
            instance.__setattr__(list(dataclasses.fields(error))[0].name, "改一下")


def _make(error: type) -> object:
    """每种错误给一份最小合法值。"""
    if error is Refusal:
        return Refusal("画布尺寸", "轴要 1920×864，设备给 1280×720")
    if error is Degradation:
        return Degradation(DoubtEntry("账目", FieldRef("费用"), "高", (LogicalFrame(3),)))
    return Abort(LogicalFrame(12), ())


def test_refusal_says_which_item_mismatched() -> None:
    """执行与失败 §4：拒绝要**附具体哪一项不吻合**（不降级、不"尽力执行"）。"""
    refusal = Refusal("画布尺寸", "轴要 1920×864，设备给 1280×720")
    assert refusal.item == "画布尺寸"
    assert "1920" in refusal.detail
    with pytest.raises(TypeError):
        Refusal("", "理由是空的话等于没说卡在哪一项")
    with pytest.raises(TypeError):
        Refusal("画布尺寸", "")


def test_degradation_carries_the_doubt_entry() -> None:
    """§5：降级**照常返回**，但进快照页眉的存疑清单 —— 它带的就是那一条条目。"""
    entry = DoubtEntry("账目", FieldRef("费用"), "高", (LogicalFrame(3),))
    degradation = Degradation(entry)
    assert degradation.doubt is entry
    with pytest.raises(TypeError):
        Degradation("账目")  # 一句人话不是一条存疑清单条目


def test_abort_says_where_it_stopped_and_what_was_confirmed() -> None:
    """§5：中止返回**跑到哪一帧** + **哪些已确认生效**，不假装跑完了。"""
    done = Action(LogicalFrame(3), "部署", ())
    abort = Abort(LogicalFrame(12), (done,))
    assert abort.frame == LogicalFrame(12)
    assert abort.confirmed == (done,)
    assert Abort(LogicalFrame(12), ()).confirmed == ()  # 空 tuple 是有意义的一种


def test_abort_frame_refuses_bare_numbers() -> None:
    """轴上只许出现逻辑帧（§3）：裸 int 与墙钟在这一栏一样收不了。"""
    for bad in (12, 12.0, "12", WallClockMs(12)):
        with pytest.raises(TypeError):
            Abort(bad, ())
    with pytest.raises(TypeError):
        Abort(LogicalFrame(1), [Action(LogicalFrame(3), "部署", ())])  # 只收 tuple
    with pytest.raises(TypeError):
        Abort(LogicalFrame(1), ("部署",))  # 已确认生效的只能是动作


def test_no_partial_success_anywhere_on_the_surface() -> None:
    """铁律：错误集里**不许有"部分成功"这一支**（工具不报"我尽力了"）。"""
    banned = [
        name
        for name in contracts.__all__
        if any(word in name.lower() for word in ("partial", "success", "succeed"))
    ]
    assert banned == []
    assert ports.__all__ == [
        "DevicePort",
        "ClockPort",
        "SamplePort",
        "AssetPort",
        "DeviceFacts",
        "Refusal",
        "Degradation",
        "Abort",
    ]


def test_grid_coord_is_not_a_pixel_in_the_device_port() -> None:
    """顺手确认真设备的入口只认格：拿像素喂部署，类型上就过不去。"""
    annotation = _annotations(DevicePort.deploy)
    assert "GridCoord" in annotation
    assert "Pixel" not in annotation
    assert not isinstance(GridCoord(5, 4), contracts.Pixel)
