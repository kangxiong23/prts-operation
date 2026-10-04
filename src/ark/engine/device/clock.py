"""设备层的时钟适配器：**只给墙钟毫秒**（契约层 §3 · 设备层 §3）。

时钟端口里**没有逻辑帧**：逻辑帧由费用推算（时间基），不是从时钟读来的。两种刻度
混起来会产生一个**静默的错** —— 同一次 200 ms 的 ADB 延迟，常速下是 6 个游戏帧、
慢放下只有 0.6 帧 —— 所以这里返回的是 `WallClockMs`，不是裸数。

## 墙钟读哪把时钟

**要 `perf_counter` 那一类、不要 `monotonic`** —— 规定与实测数在「契约层/刻度 §3」
（T4 定）。这里**只引用，不重述**：同一条规定出现两次，改的时候就会漏掉一处。
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
