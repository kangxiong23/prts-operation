"""端到端：取一帧 → 解成 Frame → 读费用 → 打印。

**T4 只写不跑**（真机那一步是 T4.1）。跑法：

    .venv\\Scripts\\python.exe scripts\\e2e_device_read.py

## 它走 `runtime/wiring.py`

**自己不拼任何设备命令** —— 这是「离线重放不是特例代码」的落地形态（代码结构
规则二 · 设备层 §1）：换一条通道就是换一个适配器，这个脚本一个字不用改。

## 判据（T4.1 跑它时按这个判）

- 帧长 == `16 + 宽 × 高 × 4`，宽高来自**帧自己的头**；
- 解出来的像素形状 == 帧的画布；
- 帧的画布 == `CANVAS_2800x1260`（**线上取像保持原生**，用户裁定）；
- **同一帧连读两次，费用读数逐位相同**（确定性）；
- 打印：取帧墙钟 ms · 帧长 · 画布 · 槽的原始形态。

**exit 1 只给链路坏了**：连不上 · 取帧失败 · 画布不符 · 解帧失败。
**读不出费用不算失败**（可能不在战斗界面）—— 那是「看不清」，不是「看错了」。
"""

from __future__ import annotations

import sys
import time

from ark.engine.vision.cost import read_cost_bar, read_cost_digits
from ark.engine.vision.frame import (
    CANVAS_2800x1260,
    RAW_HEADER_BYTES,
    frame_from_raw,
)
from ark.engine.vision.glyphs import DEFAULT_GLYPHS_PATH, GlyphTable
from ark.runtime.wiring import AdbTransport, build_wiring


def _say(message: str) -> None:
    print(message)


def _same(first: object, second: object) -> str:
    """两次读数逐位相同吗。**不同就是判据不确定**，如实打出来，不咽下去。"""
    return "是" if repr(first) == repr(second) else "否 —— 判据不确定，这是个 bug"


def main() -> int:
    wiring = build_wiring(AdbTransport())

    started = time.perf_counter_ns()
    try:
        raw = wiring.device.grab_frame()
    except Exception as bad:  # noqa: BLE001 —— 连不上 / 取帧失败都算链路坏了
        print(f"取帧失败：{type(bad).__name__}: {bad}", file=sys.stderr)
        return 1
    grab_ms = (time.perf_counter_ns() - started) // 1_000_000

    try:
        frame = frame_from_raw(raw)
    except Exception as bad:  # noqa: BLE001
        print(f"解帧失败：{type(bad).__name__}: {bad}", file=sys.stderr)
        return 1

    expected = RAW_HEADER_BYTES + frame.canvas.width * frame.canvas.height * 4
    if len(raw) != expected:
        print(
            f"帧长与帧自己的头对不上：头说 {frame.canvas.width}×{frame.canvas.height}，"
            f"该是 {expected} 字节，收到 {len(raw)} 字节",
            file=sys.stderr,
        )
        return 1
    if frame.canvas != CANVAS_2800x1260:
        print(
            f"画布不符：拿到 {frame.canvas.width}×{frame.canvas.height}，"
            f"线上取像要的是 {CANVAS_2800x1260.width}×{CANVAS_2800x1260.height}",
            file=sys.stderr,
        )
        return 1

    _say(
        f"取帧 {grab_ms} ms ｜ 帧长 {len(raw)} 字节 ｜ "
        f"画布 {frame.canvas.width}×{frame.canvas.height}"
    )
    _say(f"像素形状 {frame.pixels.shape} ｜ dtype {frame.pixels.dtype}")

    bar_first = read_cost_bar(frame)
    bar_second = read_cost_bar(frame)
    _say(f"费用条　　　：{bar_first!r}")
    _say(f"费用条再看　：{bar_second!r}")
    _say(f"两次逐位相同：{_same(bar_first, bar_second)}")

    glyphs = None
    try:
        glyphs = GlyphTable.load(DEFAULT_GLYPHS_PATH)
    except Exception as bad:  # noqa: BLE001 —— 资产端口这一轮是空的
        _say(f"字形表读不出来（{type(bad).__name__}: {bad}）—— 费用数字这一项跳过")
    if glyphs is not None:
        digits_first = read_cost_digits(frame, glyphs)
        digits_second = read_cost_digits(frame, glyphs)
        _say(f"费用数字　　：{digits_first!r}")
        _say(f"费用数字再看：{digits_second!r}")
        _say(f"两次逐位相同：{_same(digits_first, digits_second)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
