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


def test_port_gaps_is_a_tuple_of_sentences_naming_a_round():
    assert isinstance(PORT_GAPS, tuple)
    assert PORT_GAPS
    for entry in PORT_GAPS:
        assert isinstance(entry, str) and entry.strip()
        assert "T4.1" in entry
        assert "DevicePort" in entry


def test_port_gaps_names_the_two_methods_left_open():
    text = "\n".join(PORT_GAPS)
    assert "inject_gesture" in text
    assert "device_facts" in text


def test_device_adapter_does_not_invent_the_unsettled_methods():
    device = build_wiring(AdbTransport(run=Silent())).device
    assert callable(getattr(device, "grab_frame", None))
    assert not callable(getattr(device, "inject_gesture", None))
    assert not callable(getattr(device, "device_facts", None))


def test_empty_sample_port_does_nothing():
    port = EmptySamplePort()
    assert port.submit({"anything": object()}) is None
    assert vars(port) == {}


def test_empty_asset_port_has_no_public_method():
    assert [name for name in dir(EmptyAssetPort) if not name.startswith("_")] == []


def test_port_completeness_check_passes():
    """端口完备性测试自己也过一遍（它平时是直接跑的，不进 pytest 收集）。"""
    contract = Path(__file__).resolve().parents[1] / "contract"
    if str(contract) not in sys.path:
        sys.path.insert(0, str(contract))
    import port_completeness

    assert port_completeness.check() == []
