"""费用读数链：**一帧 → 费用整数 + 进度条填充**。

两个读数，各自只回答一件事（《视觉层/顶部HUD与数字》§每个区域必须知道的事实）：

| 读数 | 读什么 | 载荷 |
| --- | --- | --- |
| `read_cost_digits` | 费用数字（1–2 位，字形模板） | 精确 / **区间** / 未知 |
| `read_cost_bar` | 进度条**自左端起亮白填充的列数** | 精确 / **未知** |

**这一层不许造新载荷**（契约层 §1 只有六种）：逐位歧义往**区间**上落
（`1?` → `[10, 19]`），满态往**未知**上落 —— 契约层的词表里塞不下一个
「满」字，而相位在满态本来就没有意义。所以这一层的职责是
「**不许把满报成 0**」，不是「报出满」；触顶那个信息由费用数字 `99` 给出。

**两个读数的降级方向是相反的**，这是它们最容易写错的地方：

- 数字：**没笔画 → 未采样**（那里没有东西，或我们没能看到）；**有笔画但对不上
  任何字形 → 未知**（那里确实有东西，我不知道它是什么）；
- 条：**整条暗 → 精确 0**；**整条中间灰 + 白 MAX → 未知**。后两种样子在
  「有没有亮白填充」这个判据下长得一模一样，**读相位的人必须把它们分开** ——
  把满态读成 0 就是假事实。
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ark.contracts import (
    Exact,
    Interval,
    PixelRect,
    SlotValue,
    UiLayout,
    Unknown,
    Unsampled,
    Valued,
)
from ark.engine.vision.frame import Frame
from ark.engine.vision.glyphs import GlyphTable, glyph_boxes, ink_of, normalize
from ark.engine.vision.layout import UI

__all__ = [
    "read_cost_digits",
    "read_cost_bar",
    "SOLID_MATCH_SCORE",
    "AMBIGUITY_MARGIN",
    "CENTER_TOLERANCE",
]


SOLID_MATCH_SCORE: Final[float] = 0.90
"""一位字形至少要像到什么程度，才配被「认出来」成一个确定候选。

低于它，这一位就是「**对不上任何字形**」—— 那时**不许**从最像的那个里挑一个，
也不许报未知（未知说的是「那里有东西我不知道是什么」，而这里我们**知道**
那里有一位数字，只是认不出是哪一位）。正确的落法是**逐位歧义传播**：
这一位的候选是**全部十个数字**，于是整块降成区间。

**这个数是量出来的，不是拍的。** 实测两簇：正确的那一位是 0.938–0.958，
最像的错字是 0.774 —— 0.774 与 0.938 之间是一条空档，0.90 落在里面，
并且**偏向「正确」那一侧**：把一位读成精确值的代价（假事实）比读成区间的
代价大得多（契约层「让不知道在类型上没法被伪装成知道」），所以宁可多报区间。
"""

AMBIGUITY_MARGIN: Final[float] = 0.03
"""次像的那个离最像的有多近才算「分不开」。

进了这个范围的候选一起往下传：**逐位歧义往区间上落**，而不是替它挑一个。
实测正确那一位与最像的错字差 0.167 以上，所以 0.03 只会抓住真正并排的两个。
"""

BAR_DARK_MAX: Final[int] = 94
"""「整条暗」的上界 —— **取在空档里**（文档：空态整条中位数 ≤94）。

**这个数是空档的中点，不是「刚过暗簇上界」**。实测三簇是暗 **47–71**、
灰 **117–189**、白 **197–255**，所以暗与灰之间那条空档是 **72–116**，
中点是 **94**，两侧各留 **23**。

**它是被两次量出来的：**

| 版本 | 值 | 空条余量 | 为什么不够 |
| --- | --- | --- | --- |
| 最初 | 67 | −1 … +0 | 录像通道把暗簇上界抬到 70/71，空条大量读成未知 |
| 上一版 | 70 | **+0 / −1** | 只比暗簇上界（71）低 1 格，等于压在观测值上 |
| 这一版 | **94** | **+27 / +27** | 空档中点，两侧各 23 |

**它和 `BAR_FULL_MIN = 117` 是两个不同的统计量**（这里是**整条中位数**，
那里是**整条最暗列**），所以**两个阈值不许取成同一个数** —— 取成一样会让
「满」那一级的极值条件把「空」这一级的中位数条件吃掉。
"""

BAR_FULL_MIN: Final[int] = 117
"""「整条中间灰 + 白 MAX」的下界（文档：满态每列最亮 ≥117）。"""

BAR_BRIGHT_MIN: Final[int] = 190
"""「这一列有亮白填充」的判据。**它不是文档那个 110。**

文档的 110 是给「整条暗 ↔ 整条中间灰」这一对用的（见上两个常量），拿它来数填充
会错：满态轨道每一列的峰值是 117–146，全都 ≥110，于是**满态会被数成 153 列填充**。
实测 31 帧的「每列最亮」只有三个簇，中间有两条空档：

| 簇 | 值 | 是什么 |
| --- | --- | --- |
| 暗 | 47–68 | 轨道没走到的那一截（面板底色渐变） |
| 灰 | 117–189 | **满态**的轨道灰（146）加 MAX 那几个字的边缘 |
| 白 | 197–255 | 亮白填充 |

所以 190–196 这一段是**一个观测值都没有的空档**，判据落在这里面，
换 190 还是 196 结果完全一样（31 帧逐一验过）。取 190。
"""

FILL_START_MIN: Final[int] = 160
"""「填充**有没有从最左端开始**」的线 —— **只判窗口的第 0 列**（x = 2589）。

**它和 `BAR_BRIGHT_MIN = 190` 是两条线，不许合并。** 190 判的是「**这一列算不算
填充的一部分**」（白簇的下界），160 判的是「**填充有没有从最左端开始**」。
合并成一条会二选一地坏掉：

- 取 190 → 被污染到 **188–189** 的第 0 列会让**整段填充消失**（`probe_m15b` 实测
  48 帧，条体 247–255）；
- 取 160 当亮白线 → 会把**满态的灰轨道（117–189）**数成填充（`probe_m15` 的
  第 1..152 列有 **80543** 格落在 132–187）。

**这条线自己落在一条空档里，而且那条空档是第 0 列专用的**：非填充的第 0 列实测到
**131**（压暗的满条），被污染过的填充第 0 列低到 **188**，所以空档是 **132–187**、
**160** 是它的中点。三条录像 3354 帧里，**第 0 列落在 132–187 的一帧都没有**；
而第 1..152 列落在那一段的有 `m15` 80543 格、`m15b` 1672 格（**那是别的列的事**）。

**判据的混淆矩阵（`col0 ≥ 160` 对真值「有填充」）**：

| 通道 | 真值有填充 且 col0≥160 | 真值没填充 且 col0≥160 | 真值有填充 但 col0<160 | 有填充帧的 col0−160 |
| --- | --- | --- | --- | --- |
| `m15c` | 1389 | **0** | 0 | min **+30** / 中位 +37 |
| `m15b` | 1153 | **0** | **11**（压暗的满条） | min +28 / 中位 +37 |
| `probe_m15` | 0 | **0** | 0 | — |

没填充那一侧：`160 − col0` 最小 **+87**（`m15c`）· **+90**（`m15b`）· **+35**（`m15`）。

**为什么不用「多列投票」补它**：投票会让「最左列暗、旁边两列亮」被判成填充
（实测报出 `Exact(3)`），而那不是填充 —— 填充是**自左端起**的。
**这道门的余量来自把线放进 132–187 那条空档，不是来自多看几列。**
"""

FILL_GAP_TOLERANCE: Final[int] = 1
"""「自左端起连续亮白」里允许夹几个**连续**坏列，才算到了填充的边。

第一级判据原本是「**一列**够不够亮」。改成「要 `FILL_GAP_TOLERANCE + 1` 列
**连续**不亮才算到了边」之后，填充内部有一列被压暗不会把整段填充吃掉；而真正的
边后面跟着的是**一整条暗轨道**，永远满足「连续两列不亮」，所以边不会被推后。

**取 1 不取更大**：容错同时会让**边后面那一列**的杂散亮点把 `n` 推后一格，
`k` 越大这个风险越大。实测两条录像 2636 帧上，`k = 1` 与 `k = 0` 给出的
`n` **一帧都不差** —— 所以 1 是保险，不是修 bug。
"""

MAX_DIGITS: Final[int] = 2
"""费用是 1–2 位（文档：位数范围 1–2 位，触顶时是 `99`）。"""

CENTER_TOLERANCE: Final[int] = 8
"""「这一位落没落在中心附近」的判据（±px）。

**位数是「数有几个字形」数出来的**，所以**缺了一位的两位数长得就像一个更短的
一位数** —— 把 `10` 读成 `1`。费用是所有部署的闸门，这是这条链最危险的一种
假事实，而且它比「认不出」更糟：认不出会报未知，这个会报一个**确切但更短的数**。

文档（《顶部HUD与数字》§每个区域必须知道的事实）给了判据：**读到 1 位时，回头
验那一个字形落没落在中心附近**；中心取 `cost_digits` **窗口自己的中心**
（`rect.x + rect.width / 2`），不另立常数。

**这个数是从两侧的极值挑的，不是拍的。** 31 帧逐帧量「块心离窗口中心多远」，
真正卡住判据的不是缺得最狠的那帧，而是**缺位后离中心最近**的那帧：

| | 偏移 |
| --- | --- |
| 31 帧**真**的读数（1 位 8 帧 + 2 位 23 帧） | **−1.5 … +1.0** |
| 只剩第一块（末位缺失）里离中心**最近**的那帧 | **16.0** |
| 只剩第二块（首位缺失）里离中心**最近**的那帧 | **15.0** |

后两行是 `shot05` / `shot19` / `shot23`（读数都是 `11`，两个 `1` 都窄，
去掉一个，剩下的那个离中心最近）。所以真读数与缺位之间那条空档是
**1.0 … 15.0**，判据取**中段 8**，两边各留 **6.5 px**。

> 这两个数是规划者复量的；我这一版的框算法读到 **15.5 / 17.5**（差 0.5 px，
> 就是框宽算不含端点的差别）。无论按哪一组，8 都留了 6.5 px 以上。

**不能取 15**：那正好压在观测值上 —— 能过只是因为比较写的是 `>`，
换台设备、换一帧、或换个笔画阈值，那一路就会**静默塌成一个更短的确切数**，
而它正是这条链最危险的那种假事实。
"""


def _rect(ui: UiLayout, name: str):
    """从 UI 常量里取一个矩形。**ROI 一律从表里取**，不许在代码里另抄一份坐标。"""
    if not isinstance(ui, UiLayout):
        raise TypeError(f"ui 要是 UiLayout，收到 {type(ui).__name__}")
    try:
        return ui.rects[name]
    except KeyError:
        raise KeyError(
            f"UI 常量里没有 {name!r}：这一版认识 {sorted(ui.rects)}"
        ) from None


def _number_center(rect: PixelRect, boxes: list[tuple[int, int, int, int]]) -> float:
    """这些字形在**画布**上的水平中心：取所有紧外框并集的中心。

    框是 ROI 坐标（右/下开），所以中心 = `rect.x + (最左 + 最右) / 2`。
    """
    left = min(box[0] for box in boxes)
    right = max(box[2] for box in boxes)
    return rect.x + (left + right) / 2.0


def read_cost_digits(frame: Frame, glyphs: GlyphTable, *, ui: UiLayout = UI) -> SlotValue:
    """读费用整数。

    - 逐位都读出来、而且**位数没缺** → `Valued(Exact(n), 置信度)`；
    - 有一位分不开（并排两个候选，或整个对不上任何字形）→
      `Valued(Interval(lo, hi), 置信度)` —— **逐位歧义传播**：每一位各自留一组
      候选，最小值是「每位取最小」拼起来、最大值是「每位取最大」拼起来
      （十进制是逐位单调的，所以这样就够了）。对不上的那一位，候选是**全部
      十个数字**：`1?` → `[10, 19]`；
    - **只读到 1 位、而它不在中心附近**（`CENTER_TOLERANCE`）→ 那就是**缺了一位**：
      偏左说明缺的在它右边（`1` 缺末位 = `1?` → `[10, 19]`），偏右说明缺的在它
      左边（`?1`）。**不许把它报成一个更短的确切数**；
    - 整块没笔画 → `Unsampled()`（「那里没有东西」与「我们没能看到」在这一层
      信息上等价，契约层 §1 把这两件事合成同一支）；
    - 切出的块数超过 2（费用只有 1–2 位）→ `Valued(Unknown(), None)`：
      那不是「一位对不上」，是**这块 ROI 里的东西我根本没法按位读**。

    **绝不猜一个整数。** 认不出来就是认不出来 —— 而且认不出来要落成**区间**，
    不是落成「未知」：这里我们确实知道那里有一位数字。
    """
    if not isinstance(frame, Frame):
        raise TypeError(f"要读的是一帧，收到 {type(frame).__name__}")
    if not isinstance(glyphs, GlyphTable):
        raise TypeError(f"要一张字形表，收到 {type(glyphs).__name__}")
    rect = _rect(ui, "cost_digits")
    roi = frame.crop(rect)
    ink = ink_of(roi)
    boxes = glyph_boxes(roi)
    if not boxes:
        return Unsampled()
    if len(boxes) > MAX_DIGITS:
        return Valued(Unknown(), None)

    positions: list[tuple[tuple[int, ...], float]] = []
    for (x0, y0, x1, y1) in boxes:
        scores = glyphs.scores(normalize(ink[y0:y1, x0:x1]))
        best_score = max(scores.values())
        if best_score >= SOLID_MATCH_SCORE:
            plausible = tuple(
                sorted(
                    digit
                    for digit, score in scores.items()
                    if score >= best_score - AMBIGUITY_MARGIN
                )
            )
        else:
            # 对不上任何字形：这一位的候选是全部十个数字 —— 区间会因此变宽，
            # 而变宽是**安全的**（区间越宽越不像假事实）。不许在这里挑一个最像的。
            plausible = tuple(sorted(scores))
        positions.append((plausible, best_score))

    if len(boxes) == 1:
        offset = _number_center(rect, boxes) - (rect.x + rect.width / 2.0)
        if abs(offset) > CENTER_TOLERANCE:
            # 只读到一位、而它不在中心附近 —— 位数是「数有几个字形」数出来的，
            # 所以**缺了一位的两位数长得就像一个更短的一位数**。缺的那一位在
            # 哪边，由偏移的**符号**给出：偏左 → 缺的在右边；偏右 → 缺的在左边。
            every = tuple(range(10))
            weakest = min(score for _, score in positions)
            if offset < 0:
                positions.append((every, weakest))
            else:
                positions.insert(0, (every, weakest))

    lowest = int("".join(str(candidates[0]) for candidates, _ in positions))
    highest = int("".join(str(candidates[-1]) for candidates, _ in positions))
    confidence = min(score for _, score in positions)
    if lowest == highest:
        return Valued(Exact(lowest), confidence)
    return Valued(Interval(lowest, highest), confidence)


def _column_peaks(roi: np.ndarray) -> np.ndarray:
    """「每列最亮」= 该列里 `min(R,G,B)` 的最大值（文档给的口径）。

    先把每个像素压成它自己的三通道最小值（有多白），再逐列取最大 ——
    顺序反过来会量到「某个通道很亮的彩色像素」，那不是白。
    """
    if not isinstance(roi, np.ndarray) or roi.ndim != 3 or roi.shape[2] < 3:
        raise TypeError(f"条要从 (高, 宽, ≥3) 的像素里取，收到 {getattr(roi, 'shape', roi)}")
    if roi.shape[1] == 0:
        return np.zeros(0, dtype=np.uint8)
    return roi[:, :, :3].min(axis=2).max(axis=0)


def _fill_confidence(peaks: np.ndarray, filled: int) -> float:
    """这条相位读数有多稳：**定下这次读数的那个判据两侧，隔了多远**（归一到 0..1）。

    置信度是「原语算出的连续量」（契约层 §1），这一层自己定义它：**哪几个条件
    一起定下了这次读数，余量就取它们里面小的那个**（「最弱的一环」）。归一只用
    255（通道的上限），不掺别的常数。

    - **有填充**（`filled > 0`）：定下它的是**两个判断** ——
      ① 第一级的「最左那一列过 `FILL_START_MIN`」，② **计数边界**那两列
      （「末亮列离 `BAR_BRIGHT_MIN` 多远」与「边界列离 `BAR_BRIGHT_MIN` 多远」
      里小的那个）。余量取 ① 与 ② 里小的那个 —— **两条线不同，但都是灰阶**，
      所以可以直接比。
    - **空**（`filled == 0`，只会在这个分支被调用）：定下它的是**两个条件** ——
      整条**中位数 ≤ `BAR_DARK_MAX`**、且最亮列 < `BAR_FULL_MIN`（安全网）。
      所以余量是这两个里小的那个。**实测空帧是 27/255 ≈ 0.106** —— 它如实说出
      「这次读数离翻面（被读成满 / 读成未知）只有二十几格」。**注意这不再是
      `BAR_BRIGHT_MIN − max`**：那是「离亮白线多远」，而亮白线**没有**定下这次
      读数（它是在第一级已经被否掉的那条线）。

    于是满态读数的置信度**天生很低**（实测最暗的那一列是 197，只比判据高 7）——
    这不是坏事：它如实说出「满条与满态只差 8 个灰阶」这件事（灰簇最高 189）。
    """
    if peaks.size == 0:
        return 0.0
    if filled == 0:
        # 「空」是**两个条件**一起定下来的：整条中位数 ≤ BAR_DARK_MAX，且最亮列
        # < BAR_FULL_MIN（安全网）。所以余量取这两个里小的那个 —— 那才是
        # 「定下这次读数」的判据。原来的 `BAR_BRIGHT_MIN − max` 算的是**另一条**
        # 线（离「亮白」多远），它没有定下这次读数。
        margin = min(
            BAR_DARK_MAX - float(np.median(peaks)),
            BAR_FULL_MIN - int(peaks.max()),
        )
    else:
        # 有填充是**两个判断**一起定下来的：第一级那道门（最左列过 FILL_START_MIN）
        # 与计数边界。余量取两者里小的那个。
        started = int(peaks[0]) - FILL_START_MIN
        if filled >= peaks.size:
            boundary = int(peaks.min()) - BAR_BRIGHT_MIN
        else:
            boundary = min(
                int(peaks[filled - 1]) - BAR_BRIGHT_MIN,
                BAR_BRIGHT_MIN - int(peaks[filled]),
            )
        margin = min(started, boundary)
    return min(1.0, max(0.0, margin / 255.0))


def read_cost_bar(frame: Frame, *, ui: UiLayout = UI) -> SlotValue:
    """读进度条：**自左端起亮白填充了多少列**。

    - 自左端起有亮白填充 → `Valued(Exact(n), 置信度)`，`0 < n ≤ 153`；
      **整条填满也是这一支**（n = 153），不是「满态」；
    - 没有亮白填充、且整条暗 → `Valued(Exact(0), 置信度)`；
    - 没有亮白填充、且整条中间灰 + 白 MAX → `Valued(Unknown(), None)` ——
      **不许报成 0**，那是假事实；
    - 其余（没有亮白填充，又既不全暗也不全灰）→ 未知：那是**没见过**的样子，
      报 0 等于说「我看清了它是空的」，而我们没有。

    **三个判据各自建立在什么统计量上**（那就是「形状」）：

    | 判据 | 线 | 形状 |
    | --- | --- | --- |
    | 有没有填充 | **`FILL_START_MIN = 160`** | **只判窗口第 0 列** —— 填充是**自左端起**的。**不许用多列投票补它**：那会让「最左列暗、旁边两列亮」变成填充（实测报出 `Exact(3)`） |
    | 数 `n` | `BAR_BRIGHT_MIN = 190` | 自左端起连续亮白，按 `FILL_GAP_TOLERANCE` 容错 |
    | 空 | `BAR_DARK_MAX = 94` | 整条**中位数** ≤ 94 **且** 最亮列 < `BAR_FULL_MIN`（安全网） |
    | 满 | `BAR_FULL_MIN = 117` | 整条**中位数** ≥ 117 **且** 最暗列 > `BAR_DARK_MAX`（安全网） |

    「空 / 满」用中位数看**整条的水平**，用另一半当**安全网**：中位数判「绝大多数
    列在哪一簇」，安全网判「有没有**任何一列**进了对面那一簇」。两道都过才落。
    安全网是必须的 —— 填充可以只有几列宽（实测最窄的一帧**只有 1 列**），
    光看中位数会把小填充读成空。而这一支只在**第一级已经判定没有填充**
    之后才走到，所以安全网只需要拦住「有列进了灰簇 / 白簇」。

    **为什么第一级是「最左那一列 ≥160」而不是「≥190」**：第 0 列落在条左沿的
    4 px 渐变沿上，同一个轨道状态它读 **199 / 118 / 88**（右边第一条全强度列读
    255 / 146 / 146）—— 所以它**自己需要一条更低的线**。取 190 会让被污染到
    **188–189** 的第 0 列（`probe_m15b` 48 帧）把**整段填充**吃掉；而 160 落在那条
    空档 **132–187** 的中点（第 0 列在这段里一帧都没有），实测**零假事实**。

    **「多列投票」在这一版被拆掉了**：它会把「最左列暗、旁边两列亮」判成填充。
    """
    if not isinstance(frame, Frame):
        raise TypeError(f"要读的是一帧，收到 {type(frame).__name__}")
    rect = _rect(ui, "cost_bar")
    peaks = _column_peaks(frame.crop(rect))
    if peaks.size == 0:
        return Unsampled()
    judged = peaks
    # 第一级：**填充是自左端起的**，所以这道门就锚在窗口**最左那一列**上。
    # 用的是 FILL_START_MIN（160），不是 BAR_BRIGHT_MIN（190）—— 第 0 列在渐变沿上，
    # 它自己那条线要低一档；用 190 会让被污染到 188–189 的第 0 列吃掉整段填充。
    # **不许用多列投票补它**：那会把「最左列暗、旁边两列亮」判成填充。
    started = int(judged[0]) >= FILL_START_MIN
    # 再数填充，最后分「暗 / 灰」—— 顺序不能反。反了的话，**整条填满**那种样子
    # （每列最亮 197–255，全都 ≥117）会先落进「满态」，于是全长 153 永远读不出来。
    filled = 0
    if started:
        gap = 0
        for index in range(judged.size):
            if int(judged[index]) >= BAR_BRIGHT_MIN:
                filled = index + 1
                gap = 0
            else:
                gap += 1
                if gap > FILL_GAP_TOLERANCE:
                    break
    if filled > 0:
        return Valued(Exact(filled), _fill_confidence(judged, filled))
    bulk = float(np.median(judged))
    if bulk <= BAR_DARK_MAX and int(judged.max()) < BAR_FULL_MIN:
        return Valued(Exact(0), _fill_confidence(judged, 0))
    if bulk >= BAR_FULL_MIN and int(judged.min()) > BAR_DARK_MAX:
        return Valued(Unknown(), None)
    return Valued(Unknown(), None)
