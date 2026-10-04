"""端到端：取一帧 → 解成 Frame → 读费用 → 打印。

**T4.1 第一次真跑它**。跑法：

    .venv\\Scripts\\python.exe scripts\\e2e_device_read.py

## 它走 `runtime/wiring.py`

**自己不拼任何设备命令** —— 这是「离线重放不是特例代码」的落地形态（代码结构
规则二 · 设备层 §1）：换一条通道就是换一个适配器，这个脚本一个字不用改。

## 判据

- 帧长 == `16 + 宽 × 高 × 4`，宽高来自**帧自己的头**；
- 解出来的像素形状 == 帧的画布；
- 帧的画布 == `CANVAS_2800x1260`（**线上取像保持原生**，用户裁定）；
- **同一帧连读两次，费用读数逐位相同**（确定性）；
- 打印：取帧墙钟 ms · 帧长 · 画布 · 槽的原始形态。

**exit 1 只给"这条链坏了"**：连不上 · 取帧失败 · 画布不符 · 解帧失败 ·
**同一帧连读两次不同**（那是判据不确定，是 bug）。

**读不出费用不算失败**（可能不在战斗界面）—— 那是「看不清」，不是「看错了」。
而「读出与画面不符的数」**不是这里的判据**：脚本没有真值，判不了 —— 那是人拿
打印出来的读数对着画面看的事。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
"""仓库根。**包还没装**（`pyproject.toml` 的 `[project.scripts]` 还是注释状态），
所以这个脚本得自己把 `src/` 放进 `sys.path` —— 与 `data/probe/` 里那些脚本
一致。少了这三行，文档里那条跑法会死在
`ModuleNotFoundError: No module named 'ark'`（T4.1 真跑时撞到：`py_compile` 不
执行 import，所以它躲过了上一轮的验收）。
"""
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.engine.device.adb import AdbTransport  # noqa: E402
from ark.engine.vision.cost import read_cost_bar, read_cost_digits  # noqa: E402
from ark.engine.vision.frame import (  # noqa: E402
    CANVAS_2800x1260,
    RAW_HEADER_BYTES,
    frame_from_raw,
)
from ark.engine.vision.glyphs import DEFAULT_GLYPHS_PATH, GlyphTable  # noqa: E402
from ark.runtime.wiring import build_wiring  # noqa: E402

LOCAL_DIR = REPO / "data" / "e2e"
"""本地落点：**工作区里一个位置，显式给，不靠默认值**（T4.1 定）。

默认值落在**共享的平台临时区**，而文件名是固定的（设备端那个槽位要求同名覆盖）
—— 与别的进程撞名就会读到别人的帧。放进工作区就没人抢（`data/` 是 gitignored
的运行时数据）。目录不存在时由 `grab_frame` 建。
"""


def _same(first: object, second: object) -> bool:
    """两次读数逐位相同吗。"""
    return repr(first) == repr(second)


def main() -> int:
    wiring = build_wiring(AdbTransport(work_dir=LOCAL_DIR))

    started = time.perf_counter_ns()
    try:
        raw = wiring.device.grab_frame()
    except Exception as bad:  # noqa: BLE001 —— 连不上 / 取帧失败都算这条链坏了
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

    print(
        f"取帧 {grab_ms} ms ｜ 帧长 {len(raw)} 字节 ｜ "
        f"画布 {frame.canvas.width}×{frame.canvas.height}"
    )
    print(f"像素形状 {frame.pixels.shape} ｜ dtype {frame.pixels.dtype}")

    bar_first = read_cost_bar(frame)
    bar_second = read_cost_bar(frame)
    print(f"费用条　　　：{bar_first!r}")
    print(f"费用条再看　：{bar_second!r}")
    if not _same(bar_first, bar_second):
        print(
            f"同一帧连读两次读数不同：{bar_first!r} vs {bar_second!r} —— "
            "判据不确定，这是 bug",
            file=sys.stderr,
        )
        return 1

    glyphs = None
    try:
        glyphs = GlyphTable.load(DEFAULT_GLYPHS_PATH)
    except Exception as bad:  # noqa: BLE001 —— 资产端口这一轮是空的
        print(f"字形表读不出来（{type(bad).__name__}: {bad}）—— 费用数字这一项跳过")
    if glyphs is not None:
        digits_first = read_cost_digits(frame, glyphs)
        digits_second = read_cost_digits(frame, glyphs)
        print(f"费用数字　　：{digits_first!r}")
        print(f"费用数字再看：{digits_second!r}")
        if not _same(digits_first, digits_second):
            print(
                f"同一帧连读两次数字不同：{digits_first!r} vs {digits_second!r} —— "
                "判据不确定，这是 bug",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
