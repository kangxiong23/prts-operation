"""时钟适配器：只给墙钟毫秒，且单调不减。"""

from __future__ import annotations

import time

from ark.contracts import LogicalFrame, WallClockMs
from ark.engine.device.clock import SystemClock


def test_wall_clock_ms_is_a_wall_clock_not_a_logical_frame():
    value = SystemClock().wall_clock_ms()
    assert isinstance(value, WallClockMs)
    assert not isinstance(value, LogicalFrame)


def test_wall_clock_ms_never_goes_backwards():
    clock = SystemClock()
    samples = [int(clock.wall_clock_ms()) for _ in range(200)]
    assert all(later >= earlier for earlier, later in zip(samples, samples[1:]))


def test_wall_clock_ms_advances_over_a_real_wait():
    clock = SystemClock()
    before = int(clock.wall_clock_ms())
    time.sleep(0.05)
    after = int(clock.wall_clock_ms())
    assert after > before


def test_clock_has_no_other_public_method():
    public = [name for name in dir(SystemClock()) if not name.startswith("_")]
    assert public == ["wall_clock_ms"]
