"""组装点：四个端口一个不缺 · 欠账写下来了 · 空实现真的是空的。"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

from ark.engine.device.adb import AdbTransport
from ark.offline.assets_gen.port import EmptyAssetPort
from ark.offline.hardcase.port import EmptySamplePort
from ark.runtime.wiring import PORT_GAPS, Wiring, build_wiring


class Silent:
    """假的「跑一条命令」：装配不许发命令，所以它只记，不答。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv, timeout_ms):
        self.calls.append(list(argv))
        return b""


def test_build_wiring_supplies_all_four_ports():
    wiring = build_wiring(AdbTransport(run=Silent()))
    assert isinstance(wiring, Wiring)
    for name in ("device", "clock", "samples", "assets"):
        assert getattr(wiring, name) is not None


def test_wiring_is_frozen_and_refuses_none():
    wiring = build_wiring(AdbTransport(run=Silent()))
    with pytest.raises(dataclasses.FrozenInstanceError):
        wiring.device = None  # type: ignore[misc]
    with pytest.raises(TypeError):
        Wiring(device=None, clock=wiring.clock, samples=wiring.samples, assets=wiring.assets)  # type: ignore[arg-type]


def test_build_wiring_sends_no_command_by_itself():
    recorder = Silent()
    build_wiring(AdbTransport(run=recorder))
    assert recorder.calls == []


def test_port_gaps_is_a_tuple_of_sentences():
    """登记的形状不变：一个元组，每条一句话 —— 每条点名一个端口方法。"""
    assert isinstance(PORT_GAPS, tuple)
    for entry in PORT_GAPS:
        assert isinstance(entry, str) and entry.strip()
        assert "DevicePort" in entry or "AssetPort" in entry


def test_port_gaps_names_no_device_method_any_more():
    """T5 裁定六：这一轮做完，`PORT_GAPS` 里**不该再有任何 DevicePort 条目**。"""
    assert [entry for entry in PORT_GAPS if "DevicePort" in entry] == []


def test_device_adapter_has_exactly_the_ports_methods():
    """设备端口的六个方法一个不缺，也**一个不多**（多了要先改契约层）。"""
    device = build_wiring(AdbTransport(run=Silent())).device
    for name in ("grab_frame", "device_facts", "deploy", "retreat", "skill_on", "skill_off"):
        assert callable(getattr(device, name, None)), name
    assert not hasattr(device, "inject_gesture")


def test_build_wiring_hands_the_view_to_the_device_adapter():
    """T5 裁定三：`build_wiring(transport, view=...)` 把视图交给设备层。"""
    from ark.contracts import View

    wiring = build_wiring(AdbTransport(run=Silent()), view=View.SLOW)
    assert wiring.device._view is View.SLOW  # noqa: SLF001
    assert build_wiring(AdbTransport(run=Silent())).device._view is View.NORMAL  # noqa: SLF001


def test_empty_sample_port_does_nothing():
    port = EmptySamplePort()
    assert port.submit({"anything": object()}) is None
    assert vars(port) == {}


def test_empty_asset_port_has_no_projection_and_the_gap_is_registered():
    """T5 裁定四（新口径）：**留空 = 方法不存在** —— 空实现一个公开方法都没有。

    欠账必须**写下来**（`PORT_GAPS`），不许摆一个「在但一调就炸」的方法：那会让
    「接上了」这个判断变成**假的**（`port_completeness` 的判据是方法级的）。
    """
    assert [name for name in dir(EmptyAssetPort) if not name.startswith("_")] == []
    assert any(
        "AssetPort" in entry and "projection" in entry for entry in PORT_GAPS
    ), "资产端口那条欠账没有登记进 PORT_GAPS"


def test_port_completeness_check_passes():
    """端口完备性测试自己也过一遍（它平时是直接跑的，不进 pytest 收集）。"""
    contract = Path(__file__).resolve().parents[1] / "contract"
    if str(contract) not in sys.path:
        sys.path.insert(0, str(contract))
    import port_completeness

    assert port_completeness.check() == []
