"""设备层的时钟适配器：**只给墙钟毫秒**（契约层 §3 · 设备层 §3）。

时钟端口里**没有逻辑帧**：逻辑帧由费用推算（时间基），不是从时钟读来的。两种刻度
混起来会产生一个**静默的错** —— 同一次 200 ms 的 ADB 延迟，常速下是 6 个游戏帧、
慢放下只有 0.6 帧 —— 所以这里返回的是 `WallClockMs`，不是裸数。

## 用哪个时钟

本机实测（CPython 3.12 / Windows）：`time.monotonic_ns()` 的粒度是 **15,625,000 ns**
（背后是 `GetTickCount64()`，分辨率 0.015625 s），而 `time.perf_counter_ns()` 是
**100 ns**（`QueryPerformanceCounter()`）。设备层 §3 要用这个量算「取像往返」
（中位 1.02–1.07 s、极差 170–210 ms），16 ms 的台阶会把极差量成一堆 0 ——
所以这里取 `perf_counter`。

两者都是 `monotonic=True`、`adjustable=False`：**单调不减是平台的保证**，
不是这里补的一层保护。用挂钟时间（`time.time()`）就不行 —— 系统校时可以把
「两次取帧隔了多久」往回拨，而那是测量，不是日历。
"""

from __future__ import annotations

import time

from ark.contracts import WallClockMs

__all__ = ["SystemClock"]


class SystemClock:
    """时钟端口的实现：`perf_counter_ns()` 折成毫秒。

    纳秒整数整除 `1_000_000` 得到毫秒 —— 只做一次整除，不引入浮点，
    免得「同一个时刻量两次」给出两个不同的数。
    """

    def wall_clock_ms(self) -> WallClockMs:
        """此刻的墙钟毫秒（自某个不指定的起点）。折成帧是控制层的事。"""
        return WallClockMs(time.perf_counter_ns() // 1_000_000)
