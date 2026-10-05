"""顶部两个数：**已击杀数 / 敌人总数** 与 **蓝门血量**。

一帧 → 两条读数，各自只回答一件事（《视觉层/顶部两个数》§每个区域必须知道的事实）：

| 读数 | 读什么 | 载荷 |
| --- | --- | --- |
| `read_enemy_count` | 顶部条带上并排的「甲 / 乙」（字形 + 一个分隔符） | 两侧各自：精确 / 未知 / 未采样 |
| `read_life_digits` | 蓝门血量（1–2 位，字形模板） | 精确 / 未知 / 未采样 |

**取墨口径是这一条链自己的事**（D-全局-24）：同一块条带上敌数是纯白
`(255,255,255)`（`min(R,G,B) = 255`），生命是纯粉 `(255,120,120)`
（`min(R,G,B) = 120`）—— 视觉层原来那套共用的「`min(R,G,B) ≥ 128`」**对生命
整个是瞎的**（原生 35 帧上一块字形都切不出来）。所以这里两条链各带自己的
`take` 与墨线，**不去动共用的那一个**：费用面板的轨道灰是 117–189，把线降下来
就会把它数成笔画。**字形表不用换** —— 模板归一化之后比的是形状，与通道无关：
同一张白色表对粉字给 0.956–0.976（实测），与它对白字给的一样。

**敌数那个窗口里住的是三个东西**：甲、一个分隔符、乙。所以「认分隔符」与
「认数字」是**两件事**（文档 §每个区域必须知道的事实 那条约束）：

- 分隔符**有它自己的判据** —— **形状**：它是一条细斜笔画（每一列的墨迹只有
  7 px 高、质心纵移 26.5–27.0 px），而**每一个真数字**都有一列是 28–30 px 高。
  判据**不许**是「哪个框不像数字」：那样一来，一位被挡住会让两个框都不像数字，
  于是要么把能读的那一半也丢掉，要么替其中一个挑一个（假事实）；
- **恰好一个**分隔符才切；否则两个数**都报未知**（不许替它挑一个，也不许把
  分隔符当一位数字读进去 —— 那会把 `0/8` 读成一个三位数）。

**位数是数出来的，所以缺一位长得像一个更短的一位。** 这个复合体有三处锚点，
每一处都钉住一段位置；**任何一处判出缺了一位，这一侧就读不全 → 未知**：

| 锚点 | 钉住什么 | 真读数 | 缺一位 |
| --- | --- | --- | --- |
| **贴着分隔符的内端** | 甲右对齐、乙左对齐 | 1–4 px | ≥ 15 px |
| **块与块之间** | 同侧的数字是紧邻的 | 1–5 px | ≥ 15 px |
| **整个复合体在窗口里居中** | 锚点 x = 1292.5 ≈ 窗口中心 1293 | 两端差 ≤ 1 px | ≥ 13 px |

**三处都是「位置」给的，不是「分数」给的** —— 分数说不出「少了一位」这件事，
它只会让剩下的那一位看起来很像它自己。

**这一层只产三种东西：精确 / 未知 / 未采样。它不产区间**（《顶部两个数》§降级）：
整块没笔画 → 未采样（那里没有东西，或我们没能看到）；有笔画但**读不全** → 未知；
逐位都读出确切值、位数也对 → 精确。**「读不全」是三种情形**（某一位认不出 ·
一块都认不出 · 缺一位），落法一样、理由不一样。

> **为什么不产区间**：这三个字段的「允许的形态」里**没有区间**（生命 = 精确／上界／
> 未知 · 已击杀 = 精确／下界／未知 · 总数 = 精确／未知），而**形态是字段级的决定**，
> 读数链没有资格替它加一个。**`cost.py` 照样产区间** —— 费用的字段里有区间；
> 两条链在这件事上不同，不是不一致，是**它们的字段不一样**。

**两个数各自带自己的三态**：一边读不全，只让那一边落未知，不许把另一边一起丢掉。
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ark.contracts import (
    Exact,
    PixelRect,
    SlotValue,
    UiLayout,
    Unknown,
    Unsampled,
    Valued,
)
from ark.engine.vision.cost import AMBIGUITY_MARGIN, CENTER_TOLERANCE, SOLID_MATCH_SCORE
from ark.engine.vision.frame import Frame
from ark.engine.vision.glyphs import (
    INK_MAX,
    INK_MIN,
    GlyphTable,
    glyph_boxes,
    ink_of,
    normalize,
)
from ark.engine.vision.layout import UI

__all__ = [
    "read_life_digits",
    "read_enemy_count",
    "ENEMY_INK_TAKE",
    "ENEMY_INK_THRESHOLD",
    "LIFE_INK_TAKE",
    "LIFE_INK_THRESHOLD",
    "SEPARATOR_STROKE_MAX",
    "SEPARATOR_TRAVEL_MIN",
    "SEPARATOR_GAP_MAX",
    "COMPOSITE_ASYM_MAX",
    "MAX_ENEMY_DIGITS",
    "MAX_LIFE_DIGITS",
]


ENEMY_INK_TAKE: Final[str] = INK_MIN
"""敌数走**白色**那条口径：笔画 `(255,255,255)`，`min(R,G,B) = 255`。"""

ENEMY_INK_THRESHOLD: Final[int] = 128
"""敌数的墨线。**它和字形表建表用的那个 128 数值相同，但名字是自己的**。

实测 35 帧：笔画是 255；把笔画所在的那几列（各外扩 2 列）遮掉之后，整窗
`min(R,G,B)` 的最大值是 **99** —— 所以 128 落在 99…255 之间。

**为什么不由这条链直接借用 `glyphs.INK_THRESHOLD`**：取墨口径是每条读数链
自己的事（D-全局-24）。数值撞在一起是巧合（敌数本来就是白字），**口径不是共用的**。
"""

LIFE_INK_TAKE: Final[str] = INK_MAX
"""生命走**粉色**那条口径：笔画 `(255,120,120)`，`max(R,G,B) = 255`。"""

LIFE_INK_THRESHOLD: Final[int] = 150
"""生命的墨线（文档给的那个数）。

实测 35 帧：把笔画所在的那几列（各外扩 2 列）遮掉之后，整窗 `max(R,G,B)` 的
最大值是 **118**（shot01 是 113），而笔画核心是 255 —— 线取 150，**距背景 32 格**。

**线上方那一段是一条连续谱**（笔画的抗锯齿边从背景一路爬到 255），所以这里
没有第二个空档：真正空着的是「背景 ≤118 ↔ 笔画核心 255」这一段，150 落在里面
偏背景那一侧。**它同时是那条「共用口径对生命是瞎的」的落点**：用
`min(R,G,B) ≥ 128` 在这 35 帧上切得出笔画的帧数是 **0**。
"""

SEPARATOR_STROKE_MAX: Final[int] = 17
"""分隔符**每一列**的墨迹高度上限（px）。

实测 35 帧：分隔符的每一列只有 **7** px 高（它是一条细斜笔画），而**每一个真
数字**都有一列是 **28–30** px 高（数字有竖笔画）。空档是 **7…28**，取中点 17 ——
两侧各留 10 / 11 px。

**它不是「刚过 7」**：压在 7 上的话，多一个像素的抗锯齿就会把真分隔符判掉。

**这一条就是「分隔符有它自己的判据」的落点**：一个被挡住的数字是一团
（每一列都有整个字高），永远过不了这一条 —— 所以挡住一位不会把分隔符也一起
判错。
"""

SEPARATOR_TRAVEL_MIN: Final[int] = 21
"""分隔符的墨迹质心**纵向**要走多远（px）才算一条斜笔画。

实测分隔符 26.5–27.0，而真数字最多 16.1（`7` 的斜笔）—— 空档 16.1…26.5，
取中点 21。

**它是安全网**：只判「每一列都细」的话，一条细横线（每列 1 px）也会被当成
分隔符，而它的质心根本不走。
"""

SEPARATOR_GAP_MAX: Final[int] = 10
"""「贴着分隔符」与「块与块紧邻」的判据（px）。

实测 35 帧：甲右缘到分隔符左缘 **1–4** px、乙左缘到分隔符右缘 **1–4** px、
同侧两个数字之间 **1–5** px。而**缺一位**之后那个缝是 **15 以上**（缺的那位
自己 11–26 px 宽，再加上它前面那条缝 —— 最窄的一位 `1` 加最窄的缝也给 15）。
所以空档是 5…15，取 **10**，两侧各留 5。
"""

COMPOSITE_ASYM_MAX: Final[int] = 7
"""整个复合体**居不居中**的判据（px）：两端到窗口中心的距离差。

复合体的锚点是**它的中心**（实测 x = 1292.5，窗口中心 1293）—— 所以「左端离
中心多远」与「右端离中心多远」必须相等。这个**距离差是中心偏移的两倍**，而
缺一位只会把中心挪动「缺掉的那一位的半宽」，所以用中心偏移本身当判据会漏掉
窄的那一位（最窄的一位 `1` 只把中心挪 **6.5** px，落在 `CENTER_TOLERANCE`
的 8 里面）。

实测 35 帧：真读数两端差 **≤ 1** px；缺一位（在复合体的外端）时 **≥ 13** px。
空档 1…13，取中点 **7**，两侧各留 6。
"""

MAX_ENEMY_DIGITS: Final[int] = 3
"""敌数两侧各 1–3 位（文档：实测只见过 1–2 位，窗口按 3 位取）。"""

MAX_LIFE_DIGITS: Final[int] = 2
"""生命 1–2 位（与费用同一个理由：缺位会长成一个更短的确切数）。"""


def _rect(ui: UiLayout, name: str) -> PixelRect:
    """从 UI 常量里取一个矩形。**ROI 一律从表里取**，不许在代码里另抄一份坐标。"""
    if not isinstance(ui, UiLayout):
        raise TypeError(f"ui 要是 UiLayout，收到 {type(ui).__name__}")
    try:
        return ui.rects[name]
    except KeyError:
        raise KeyError(
            f"UI 常量里没有 {name!r}：这一版认识 {sorted(ui.rects)}"
        ) from None


def _guard(frame: object, glyphs: object) -> None:
    """两个入口共用的类型闸门（照 `cost.py` 的样子）。"""
    if not isinstance(frame, Frame):
        raise TypeError(f"要读的是一帧，收到 {type(frame).__name__}")
    if not isinstance(glyphs, GlyphTable):
        raise TypeError(f"要一张字形表，收到 {type(glyphs).__name__}")


def _number_center(rect: PixelRect, boxes: list[tuple[int, int, int, int]]) -> float:
    """这些字形在**画布**上的水平中心：取所有紧外框并集的中心。

    框是 ROI 坐标（右/下开），所以中心 = `rect.x + (最左 + 最右) / 2`。
    """
    left = min(box[0] for box in boxes)
    right = max(box[2] for box in boxes)
    return rect.x + (left + right) / 2.0


def _one_digit(
    glyphs: GlyphTable, ink: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[int | None, float]:
    """这一块墨迹能不能读成**一位确切数字**：读得出给数字，读不出给 `None`。

    读不出的两种情形都归 `None`（这一层不产区间，见模块文档）：

    - **对不上任何字形**（最像的那个低于 `SOLID_MATCH_SCORE`）—— 我们知道那里有
      一位数字，但不知道它是几。**不许挑一个最像的**；
    - **并排候选**（次像的离最像的不到 `AMBIGUITY_MARGIN`）—— 两个都中等分，
      挑一个就是假事实。实测真数字与次像的差 0.083 以上，所以这一条只会抓住
      真正并排的那两个。

    第二个返回值是最像的那个的分数（置信度取这一串里最弱的那个）。
    """
    x0, y0, x1, y1 = box
    scores = glyphs.scores(normalize(ink[y0:y1, x0:x1]))
    best = max(scores.values())
    if best < SOLID_MATCH_SCORE:
        return None, best
    rivals = sorted(d for d, s in scores.items() if s >= best - AMBIGUITY_MARGIN)
    if len(rivals) != 1:
        return None, best
    return rivals[0], best


def _is_separator(ink: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """这一块墨迹是不是那个分隔符 —— **只看形状，不看它像不像数字**。

    一条细斜笔画：每一列的墨迹都不高（`SEPARATOR_STROKE_MAX`），而整条的质心
    纵向走得很远（`SEPARATOR_TRAVEL_MIN`）。**判据必须与「认数字」分开**：
    靠「哪个框不像数字」认它的话，一位被挡住就会同时废掉两个框（文档那条约束）。
    """
    x0, y0, x1, y1 = box
    mask = ink[y0:y1, x0:x1] >= ENEMY_INK_THRESHOLD
    centroids: list[float] = []
    for column in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, column])
        if rows.size == 0:
            continue
        if int(rows[-1] - rows[0]) + 1 > SEPARATOR_STROKE_MAX:
            return False
        centroids.append(float(rows.mean()))
    if len(centroids) < 2:
        return False
    return max(centroids) - min(centroids) >= SEPARATOR_TRAVEL_MIN


def _has_inner_gap(boxes: list[tuple[int, int, int, int]]) -> bool:
    """同一侧两块之间**有没有**缝太大的地方 —— 缺的那一位在这两块**之间**。"""
    return any(
        boxes[index + 1][0] - boxes[index][2] > SEPARATOR_GAP_MAX
        for index in range(len(boxes) - 1)
    )


def _read_digits(
    glyphs: GlyphTable,
    ink: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    *,
    max_digits: int,
    missing: bool,
) -> SlotValue:
    """一侧（或一个单独的窗口）里的那些字形 → 一个槽值。**只有两种落法。**

    - **逐位都读出确切值、位数也对** → `Valued(Exact(n), 置信度)`；
    - **其余一切「读不全」** → `Valued(Unknown(), None)`：某一位认不出 ·
      一块都认不出 · **缺一位**（`missing`）· 块数超过上限。

    `missing` 是**三处缝里任何一处判出缺了一位**。它**只用来举手**，不用来拼区间
    —— 缺位由**位置**给出，不由分数给出（分数说不出「少了一位」这件事）。

    **绝不猜一个更短的确切数**：缺一位时报的是未知，不是把剩下的那一位报成一个
    整数 —— `10` 读成 `1` 是这条链最危险的一种假事实。**也绝不落未采样**：
    那里有东西，只是读不出来（混掉就是假事实）。
    """
    if not boxes:
        return Unsampled()
    if len(boxes) > max_digits:
        return Valued(Unknown(), None)
    if missing:
        # 缺一位：位数是数出来的，所以这时剩下的那些块长得就像一个更短的确切数。
        return Valued(Unknown(), None)
    read = [_one_digit(glyphs, ink, box) for box in boxes]
    if any(digit is None for digit, _ in read):
        return Valued(Unknown(), None)
    digits = [digit for digit, _ in read if digit is not None]
    return Valued(
        Exact(int("".join(str(digit) for digit in digits))),
        min(score for _, score in read),
    )


def read_life_digits(frame: Frame, glyphs: GlyphTable, *, ui: UiLayout = UI) -> SlotValue:
    """读蓝门血量（1–2 位）。ROI 是 `life_digits`，取墨走**粉色**那条口径。

    - 逐位都读出确切值、位数也没缺 → `Valued(Exact(n), 置信度)`；
    - **整块没笔画** → `Unsampled()`（「那里没有东西」与「我们没能看到」在这一层
      信息上等价）；
    - **其余一切「读不全」→ `Valued(Unknown(), None)`**：某一位对不上任何字形 ·
      并排候选分不开 · 块数超过 2（生命只有 1–2 位）· **只读到 1 位而它不在窗口
      中心附近**（`CENTER_TOLERANCE` —— 那就是**缺了一位**：偏左说明缺的在它
      右边、偏右说明缺的在它左边）。

    **缺一位这一条必须判出来**：不判就会报一个更短的确切数（`10` 读成 `1`），
    那才是这条链最危险的假事实。**这一层不产区间**（见模块文档）。

    **窗口中心就是锚点**（文档 §每个区域必须知道的事实）：实测生命 `1` 的中心
    落在 1609.5、`3` 落在 1608.5，而窗口中心是 1609 —— 判据直接用窗口中心，
    不另立常数。
    """
    _guard(frame, glyphs)
    rect = _rect(ui, "life_digits")
    roi = frame.crop(rect)
    boxes = glyph_boxes(roi, take=LIFE_INK_TAKE, threshold=LIFE_INK_THRESHOLD)
    ink = ink_of(roi, take=LIFE_INK_TAKE)
    missing = False
    if len(boxes) == 1:
        offset = _number_center(rect, boxes) - (rect.x + rect.width / 2.0)
        missing = abs(offset) > CENTER_TOLERANCE
    return _read_digits(
        glyphs, ink, boxes, max_digits=MAX_LIFE_DIGITS, missing=missing
    )


def read_enemy_count(
    frame: Frame, glyphs: GlyphTable, *, ui: UiLayout = UI
) -> tuple[SlotValue, SlotValue]:
    """读顶部条带上并排的那两个数，**顺序是 (已击杀, 敌人总数)**。

    ROI 是 `enemy_count`，取墨走**白色**那条口径。窗口里切出来的块 =
    甲的位数 + **1 个分隔符** + 乙的位数：

    - **整窗没笔画** → 两个都 `Unsampled()`（先判这一条：涂黑之后连分隔符也
      没有，那时报「未知」等于说「那里有东西」）；
    - **认不出恰好一个分隔符** → 两个都 `Valued(Unknown(), None)` —— 不许把
      分隔符当一位数字读进去（那会把 `0/8` 读成一个三位数），也不许替它挑一个；
    - 否则切开，**两个数各自带自己的三态**：一边的某一位认不出，只让那一边降级。

    缺一位的判据有三处锚点（见模块文档那张表）：贴着分隔符的内端、块与块之间、
    整个复合体在窗口里居中。**任何一处判出缺了一位，这一侧就落未知** —— 位置给
    证据，分数给不出「少了一位」这件事。
    """
    _guard(frame, glyphs)
    rect = _rect(ui, "enemy_count")
    roi = frame.crop(rect)
    boxes = glyph_boxes(roi, take=ENEMY_INK_TAKE, threshold=ENEMY_INK_THRESHOLD)
    if not boxes:
        return Unsampled(), Unsampled()
    ink = ink_of(roi, take=ENEMY_INK_TAKE)
    separator_at = [index for index, box in enumerate(boxes) if _is_separator(ink, box)]
    if len(separator_at) != 1:
        return Valued(Unknown(), None), Valued(Unknown(), None)
    cut = separator_at[0]
    separator = boxes[cut]
    kills = boxes[:cut]
    total = boxes[cut + 1:]

    kills_missing = False
    total_missing = False
    # 内端：甲右对齐、乙左对齐（实测间距 1–4 px）。缝一大，缺的那一位就在这一侧
    # 的**内端**（甲是末位、乙是首位）。
    if kills and separator[0] - kills[-1][2] > SEPARATOR_GAP_MAX:
        kills_missing = True
    if total and total[0][0] - separator[2] > SEPARATOR_GAP_MAX:
        total_missing = True
    # 块与块之间：同侧的数字是紧邻的（实测 1–5 px）。
    kills_missing = kills_missing or _has_inner_gap(kills)
    total_missing = total_missing or _has_inner_gap(total)
    # 外端：整个复合体在窗口里居中，所以它偏了就说明缺的那一位在复合体的外端。
    # 判的是**两端到窗口中心的距离差**（中心偏移的两倍，见 COMPOSITE_ASYM_MAX）：
    # 右端伸得更远 → 复合体整体偏右 → 它**左边**短了一截 → 甲的首位缺；
    # 左端伸得更远 → 偏左 → 它**右边**短了一截 → 乙的末位缺。
    if kills and total:
        left = rect.x + min(box[0] for box in boxes)
        right = rect.x + max(box[2] for box in boxes)
        middle = rect.x + rect.width / 2.0
        if (right - middle) - (middle - left) > COMPOSITE_ASYM_MAX:
            kills_missing = True
        elif (middle - left) - (right - middle) > COMPOSITE_ASYM_MAX:
            total_missing = True
    return (
        _read_digits(
            glyphs, ink, kills, max_digits=MAX_ENEMY_DIGITS, missing=kills_missing
        ),
        _read_digits(
            glyphs, ink, total, max_digits=MAX_ENEMY_DIGITS, missing=total_missing
        ),
    )
