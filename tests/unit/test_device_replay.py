"""设备端口的第二个适配器（离线重放）的单元测试。

这一批**不 import tests/spec**（D-全局-17：那是规划者的验收套件），也**不抄**
任何真值 —— 这里测的是"它是不是那个端口 · 流水记不记得住 · 帧是不是依次吐 ·
前提头是不是只收 DeviceFacts"。

**前提头那一组的形状是 T6 补定的**：这一层**只收 `DeviceFacts | None`**，
折清单里那个映射是**加载器**的事（基准集 §2 · 设备层 §5）—— 所以"传映射进来"
现在是一条**要报错**的路，不再是一条要折的路。
"""

from __future__ import annotations

import inspect

import pytest

from ark.contracts import Canvas, DeviceFacts, DevicePort, GridCoord
from ark.engine.device.adb import AdbDevice
from ark.engine.device.replay import (
    ReplayAction,
    ReplayDevice,
    ReplayExhausted,
    ReplayFactsMissing,
)

FACING = "右"
FACTS = DeviceFacts(resolution=Canvas(width=2800, height=1260), orientation="1")


def _port_methods() -> set[str]:
    """端口自己声明的那几个方法名（从契约层取，不在这里抄一份）。"""
    return {
        name
        for name, value in vars(DevicePort).items()
        if not name.startswith("_") and callable(value)
    }


def _public_methods(impl: object) -> set[str]:
    return {
        name
        for name in dir(impl)
        if not name.startswith("_") and callable(getattr(impl, name, None))
    }


def _shape(function: object) -> list[tuple[str, object, object]]:
    """一个方法的形状：参数名 · 种类 · 默认值。"""
    return [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(function).parameters.values()
    ]


# ---------------------------------------------------------------------------
# 它是那个端口
# ---------------------------------------------------------------------------


def test_the_six_port_methods_are_all_there_and_nothing_else_is_public() -> None:
    device = ReplayDevice([])
    assert _public_methods(device) == _port_methods()
    assert _port_methods() == {
        "grab_frame",
        "deploy",
        "retreat",
        "skill_on",
        "skill_off",
        "device_facts",
    }
    # ledger 是属性，不是公开方法 —— 端口方法集之外的公开方法要先把契约层改了。
    assert "ledger" not in _public_methods(device)
    assert isinstance(device.ledger, tuple)


def test_the_signatures_are_the_same_shape_as_the_adb_adapter() -> None:
    for name in sorted(_port_methods()):
        assert _shape(getattr(ReplayDevice, name)) == _shape(
            getattr(AdbDevice, name)
        ), name


def test_the_constructor_takes_the_frame_source_and_optional_facts_only() -> None:
    """它**不收资产端口与视图**（设备层 §5 · T6 定）—— 那是它与 adb 适配器唯一
    的结构差别，不是破例。"""
    parameters = inspect.signature(ReplayDevice.__init__).parameters
    assert set(parameters) - {"self"} == {"frames", "facts"}
    assert parameters["facts"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["facts"].default is None


# ---------------------------------------------------------------------------
# 帧依次吐
# ---------------------------------------------------------------------------


def test_frames_come_out_in_order() -> None:
    device = ReplayDevice([b"one", b"two", b"three"])
    assert [device.grab_frame() for _ in range(3)] == [b"one", b"two", b"three"]


def test_asking_again_after_the_last_frame_is_refused_not_rewound() -> None:
    device = ReplayDevice([b"only"])
    assert device.grab_frame() == b"only"
    with pytest.raises(ReplayExhausted):
        device.grab_frame()
    # 再要一次也不回卷：报的是同一件事，不是从头再来。
    with pytest.raises(ReplayExhausted):
        device.grab_frame()


def test_an_empty_frame_source_is_exhausted_from_the_first_ask() -> None:
    with pytest.raises(ReplayExhausted):
        ReplayDevice([]).grab_frame()


def test_a_bare_frame_is_not_a_frame_source() -> None:
    """把一帧当成一批喂进来，第一次取帧就会吐出第一个字节 —— 这里直接拦下来。"""
    with pytest.raises(TypeError):
        ReplayDevice(b"x" * 32)


def test_the_frame_source_is_not_read_at_construction() -> None:
    """**构造时不预读**（设备层 §5 · T6 补）：第一帧该由第一次 grab_frame() 取。

    这一层"一个像素都不算"，所以它没有理由在构造时就把帧源拉一格 —— 拉了就会
    多握一帧 14 MB，而且"没取帧"与"取了一帧"这两种状态就分不开了。
    """
    taken: list[int] = []

    def frames():
        for index in range(2):
            taken.append(index)
            yield b"frame-%d" % index

    device = ReplayDevice(frames())
    assert taken == []
    assert device.grab_frame() == b"frame-0"
    assert taken == [0]
    assert device.grab_frame() == b"frame-1"
    assert taken == [0, 1]
    with pytest.raises(ReplayExhausted):
        device.grab_frame()


# ---------------------------------------------------------------------------
# 四个动词只记一笔
# ---------------------------------------------------------------------------


def test_the_four_verbs_write_one_line_each_in_signature_order() -> None:
    device = ReplayDevice([])
    at = GridCoord(5, 4)
    device.deploy(0, at, FACING)
    device.retreat(at)
    device.skill_on(at)
    device.skill_off(at)
    assert device.ledger == (
        ReplayAction(verb="deploy", params=(0, at, FACING)),
        ReplayAction(verb="retreat", params=(at,)),
        ReplayAction(verb="skill_on", params=(at,)),
        ReplayAction(verb="skill_off", params=(at,)),
    )


def test_the_ledger_is_a_snapshot_not_the_inner_list() -> None:
    device = ReplayDevice([])
    before = device.ledger
    device.retreat(GridCoord(1, 1))
    assert before == ()
    assert [action.verb for action in device.ledger] == ["retreat"]


def test_the_verbs_send_nothing_and_need_no_projection() -> None:
    """没有传输、没有资产端口、没有视图，四个动词照样只记一笔：它算不出像素，
    因为它根本不算像素（离线验收 §2）。"""
    device = ReplayDevice([])
    device.deploy(3, GridCoord(1.5, 2.5), "上")
    assert [action.verb for action in device.ledger] == ["deploy"]
    for attribute in ("transport", "assets", "view"):
        assert not hasattr(device, attribute)


def test_a_ledger_line_needs_a_verb_and_a_tuple_of_params() -> None:
    with pytest.raises(TypeError):
        ReplayAction(verb="", params=())
    with pytest.raises(TypeError):
        ReplayAction(verb="deploy", params=[0, 1, 2])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 前提头：只收 DeviceFacts | None
# ---------------------------------------------------------------------------


def test_recorded_facts_are_replayed_verbatim() -> None:
    device = ReplayDevice([], facts=FACTS)
    assert device.device_facts() is FACTS


def test_a_recording_without_facts_is_refused_not_invented() -> None:
    device = ReplayDevice([b"one"])
    assert device.grab_frame() == b"one"
    with pytest.raises(ReplayFactsMissing):
        device.device_facts()


def test_a_manifest_facts_mapping_is_refused() -> None:
    """折清单里那个映射是**加载器**的事（基准集 §2）—— 引擎不认离线那套键名
    （代码结构 规则三）。传进来要报错，不许悄悄收下、更不许自己解帧头。"""
    with pytest.raises(TypeError):
        ReplayDevice([b"one"], facts={"orientation": "3"})
    with pytest.raises(TypeError):
        ReplayDevice([b"one"], facts={"resolution": [2800, 1260], "orientation": "3"})


def test_facts_of_another_shape_are_refused() -> None:
    for wrong in ("1", 1, [2800, 1260], object()):
        with pytest.raises(TypeError):
            ReplayDevice([], facts=wrong)  # type: ignore[arg-type]
