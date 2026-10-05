"""把一批基准集**经重放适配器**逐帧吐出来（采集与验收/基准集 §3）。

    BenchBatch → ReplayDevice → 逐帧的原始帧字节 → 识别链

**帧走的是设备端口那条路，不是"读文件"那条路**：识别链拿到的帧与真机上拿到的是
同一个形状（设备层 §4 的原始帧字节），而"离线重放不是特例代码"这句话就落在这一
层 —— 换一个适配器，识别链一个字不用改（代码结构 规则二）。

**它是生成器，一次只握一帧**：一帧 14,112,016 字节，31 帧全留在内存里是 437 MB。
逐帧吐是这条路本来就该有的形态（设备层 §4：只有"取一帧"，没有"取像流"）。

**跑一遍是一次性的**：帧依次吐、吐完再要就是 `ReplayExhausted`，**不许回卷**
（基准集 §3）—— 离线验收 §1 要的正是"这一批跑过了"这句话。
"""

from __future__ import annotations

from collections.abc import Iterator

from ark.engine.device.replay import ReplayDevice
from ark.offline.bench.batch import BenchBatch, BenchCase

__all__ = ["replay_frames"]


def replay_frames(batch: BenchBatch) -> Iterator[tuple[BenchCase, bytes]]:
    """一批帧经重放适配器逐帧吐出来：`(这一条, 它的原始帧)`。

    顺序就是清单里 `crops` 的顺序 —— 重放适配器不回卷、不重排。
    """
    if not isinstance(batch, BenchBatch):
        raise TypeError(f"要喂的是一批基准集，收到 {type(batch).__name__}")
    device = ReplayDevice((case.raw() for case in batch.cases), facts=batch.facts)
    for case in batch.cases:
        yield case, device.grab_frame()
