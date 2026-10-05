"""视觉层的 UI 常量：8 个具名矩形。

值**逐字抄**规格那两张表，一个数没改：`cost_*` / `deploy_limit_row` /
`pause_button` 抄《视觉层/顶部HUD与数字》§区域与常量；`enemy_count` /
`life_digits` 抄《视觉层/顶部两个数》§两个窗口（T7 复量之后从那两份里拆开的，
原来那两行是**比例换算**来的，实测把它们推翻了）。

表里写的是 `(x0, y0, x1, y1)`、**含两端**；契约层 §2 的 `PixelRect` 要的是
`x / y / 宽 / 高`，所以这里做一次换算：`宽 = x1 - x0 + 1`、`高 = y1 - y0 + 1`。

**换算不是自由发挥**：文档自己给了 `cost_bar` **宽 153 px**，而
2741 - 2589 + 1 = 153 —— 含两端就是这个口径。

**`squad_row` 不在这张表里**：它是「上缘 + 下缘 + 右缘 + 卡距」，
卡距是**标量**，而 `UiLayout` 的值只能是像素矩形（契约层 §2）。
给它硬造一个退化矩形就是假事实 —— 所以是 8 个键，不是 9 个。

**常量是「够用的窗口」，不是「精确边界」**（文档原话）：ROI 多切几像素无害，
窗口之间允许重叠。这里只做「抄下来」，不替文档决定窗口该多大。

`DOCUMENTED_WINDOWS` 是换算**之前**那张表（原样、含两端）。留着它是为了
让「抄得对不对」能被机器比一遍，不用人去数。
"""

from __future__ import annotations

from ark.contracts import PixelRect, UiLayout

__all__ = ["UI", "DOCUMENTED_WINDOWS", "RECT_NAMES"]


# 名字 → 文档原样的 (x0, y0, x1, y1)，**含两端**，原生 2800×1260。
DOCUMENTED_WINDOWS: dict[str, tuple[int, int, int, int]] = {
    "cost_panel": (2585, 845, 2799, 972),
    "cost_icon": (2595, 873, 2658, 936),
    "cost_digits": (2666, 868, 2776, 936),
    "cost_bar": (2589, 945, 2741, 954),
    "deploy_limit_row": (2462, 968, 2755, 998),
    "enemy_count": (1200, 33, 1385, 72),
    "life_digits": (1574, 33, 1643, 72),
    "pause_button": (2594, 54, 2710, 130),
}
"""上面那两张表，去掉不是矩形的 `squad_row` 之后剩下的 8 行。"""

RECT_NAMES: tuple[str, ...] = tuple(DOCUMENTED_WINDOWS)
"""8 个具名矩形，顺序就是文档表里的顺序。"""


def _as_rect(window: tuple[int, int, int, int]) -> PixelRect:
    """(x0, y0, x1, y1) 含两端 → PixelRect 的 x / y / 宽 / 高。"""
    x0, y0, x1, y1 = window
    return PixelRect(x=x0, y=y0, width=x1 - x0 + 1, height=y1 - y0 + 1)


UI = UiLayout(rects={name: _as_rect(window) for name, window in DOCUMENTED_WINDOWS.items()})
"""这台设备的 UI 常量：名字 → 像素矩形。**换一台设备，全部 ROI 与投影一起重来。**"""
