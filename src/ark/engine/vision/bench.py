"""备战席那一栏：**卡位 · 每张卡的当前部署费用 · 再部署状态与剩余秒数**。

一帧 → 这一栏的读数（《视觉层/备战席》）。这一栏**右对齐**、**只有还没上场的卡才显示**，
所以卡数在一局之内会变；**卡数是数出来的，不是参数**（下标不设上限）。

| 读数 | 判据 | 载荷 |
| --- | --- | --- |
| 卡位 | 卡上「职业标识牌 → 费用暗盒」那道黑↔白跳变（= 卡的水平正中心） | 一个中心 x |
| 每张卡的费用 | 卡中心右边那个暗盒里的数字（1–2 位） | 精确 / 未知 / 未采样 |
| 是否再部署中 | 卡片正中**那一圈环在不在**（环带最暗的那一度还是亮的） | 是 / 否 / 判不了 |
| 再部署剩余秒数 | 圆环中央那个数字（一位小数） | 精确 / 未知 / 未采样 |
| 环形进度 | 同一圈上白色弧占整圈的比例 = **已过比例** | 0…1，或判不了 |

**身份不归这一层**（《备战席》§点名）：这一栏的顺序是**进场时的顺序** —— 进场那一刻
从右往左按费用降序，之后**相对位置不动**，费用再怎么变卡也不换位（《备战席》§卡片怎么排）。
所以：**位次记的是进场顺序**，它与「这张卡是谁」没有关系。身份住 `roster.py`。

**「降序」连启发式都不是**（用户裁定）：我们那批帧一直严格降序，是因为队里**那个干员的
技能在场上时会把这一栏（连关卡机制卡一起）强制按费用重排**；**他不在场时这一栏就是
「进场顺序 + 位置冻住」，根本不降序**。所以读数链**不许拿「降序」当判据或捷径** ——
我这一版一个地方都没用它（唯一用到的顺序是「最右那张是锚点」，那是槽位制的几何、
不是费用）。

**格子不一定是干员，而且这一层不判它是谁**（《备战席》§点名）：已知**两类**不是干员
—— **装置**（跟着某个干员来）与**关卡机制卡**（跟着这一关来，实例 `<阿勒黛的护卫>`），
两者外观**几乎一样**（同一个「光圈」职业图标、同样的卡框与费用暗盒）。顶上那个数：
干员与关卡机制卡都是「这张卡自己的部署费用」（**同一个语义，照读**）；**装置那一格是不是
同一个语义还没实测**，所以照读、但**不解释它是谁的费用**。右下角那个 `XN`（一共可以部署
几个）**这一轮不读**（还没登记成字段）。

**卡位首选读跳变，网格是它的抽验与备选**（《备战席》§卡位怎么读）。判据是**形状**
—— 「左边一大块亮、右边一大块暗」，不是「哪条边最陡」：卡面上别处也有黑白边
（费用牌自己的边、职业图标自己的边、卡与卡之间那道缝），只认「最陡的那一条」会认错。
所以每一格**先用网格定出该往哪儿看，再问像素那儿有没有那道跳变** —— 有就采信像素读出的
中心（实测与网格差 ≤ 2 px），没有就**不认这一格有卡**。

**卡距不进这一层**：它是**标量**，`UiLayout` 装不下（契约层 §2），所以它住**用它的那
一层**自己的常量（设备层的 `SQUAD_CARD_PITCH`），**由调用方传进来** —— 导入规则不许
`ark.engine.vision` import `ark.engine.device`（`tests/contract/import_rules.py`）。
下标 → 矩形只有一份实现：`ark.contracts.squad_card_rect`。

**再部署中的判据是「环在不在」，不是「白弧多长」**（《备战席》§判死与再部署）：那道白弧
量的是**已过比例**（从 12 点起顺时针长的角度 ÷ 360）—— 剩 12.9 s 的那张白弧 278°，
而剩 68.6 s 的那张只有约 20°。**刚撤退的卡白弧接近 0°**，所以拿弧长当判据会把
「正在再部署」说成「不在再部署」，一条假事实。这一层因此只问**环带最暗的那一度还亮不亮**
（`_ring_floor`）：环是画满一整圈的，它在，整圈就都在。

**红罩不单独当判据**：实测它分不开 —— `shot26` 下标 1 那张**普通卡**的卡面红度
（`R − max(G,B)` 的均值 **+64.8**）比再部署那张（**+48.5**）还高，普通卡里整片红的
角色有的是。红罩是**外观**的一部分，不是这一层能用的信号。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from ark.contracts import (
    Exact,
    PixelRect,
    SlotValue,
    Unknown,
    Unsampled,
    Valued,
    UI_LAYOUT,
    squad_card_rect,
)
from ark.engine.vision.frame import Frame
from ark.engine.vision.glyphs import (
    INK_MIN,
    GlyphTable,
    glyph_boxes,
    ink_of,
    normalize,
)

__all__ = [
    "read_squad_row",
    "SquadCard",
    "JUMP_LIGHT_MIN",
    "JUMP_DARK_MAX",
    "CENTER_MATCH_TOL",
    "COST_BOX_WIDTH",
    "COST_DIGITS_TOP",
    "COST_DIGITS_HEIGHT",
    "COST_INK_TAKE",
    "COST_INK_THRESHOLD",
    "MIN_DIGIT_WIDTH",
    "MAX_COST_DIGITS",
    "COST_SOLID_MATCH_SCORE",
    "COST_AMBIGUITY_MARGIN",
    "RING_CENTER_DY",
    "RING_INNER",
    "RING_OUTER",
    "RING_BRIGHT_MIN",
    "RING_PRESENT_OUTER",
    "RING_PRESENT_MIN",
    "RING_PRESENT_ABSENT_MAX",
    "COUNTDOWN_INK_THRESHOLD",
    "COUNTDOWN_MIN_DIGIT_WIDTH",
    "COUNTDOWN_MIN_DIGIT_HEIGHT",
    "COUNTDOWN_DOT_MIN_WIDTH",
    "COUNTDOWN_DOT_MAX_WIDTH",
    "COUNTDOWN_DOT_MIN_HEIGHT",
    "COUNTDOWN_DOT_MAX_HEIGHT",
    "COUNTDOWN_DOT_BASELINE_TOL",
    "MAX_COUNTDOWN_WHOLE_DIGITS",
]


JUMP_LIGHT_MIN: Final[int] = 80
"""跳变**亮的那一侧**要有多亮（每列 `min(R,G,B)` 的中位数）。

实测 6 帧里**每一张卡**的那一列（职业标识牌最右一列）是 **89–116**，而它右边那一列
（费用暗盒）是 **6–29**。缝、卡面别处的黑白边也落在这个区间附近（实测 46–81），
所以**光靠这一对线分不开**，还得靠网格（见 `CENTER_MATCH_TOL`）。
"""

JUMP_DARK_MAX: Final[int] = 45
"""跳变**暗的那一侧**要有多暗。与 `JUMP_LIGHT_MIN` 配对使用，见上。"""

CENTER_MATCH_TOL: Final[int] = 5
"""像素读出的中心离网格那一格最多差多少像素，还算**同一格**。

实测 6 帧 56 张卡：像素中心与 `squad_card_rect` 给的中心**最多差 2 px**；
而**没有卡的那些格**上，最近的跳变候选离网格 **131 px 以上**。空档 2…131，
取 5 —— 两侧各留 3 / 126。
"""

COST_BOX_WIDTH: Final[int] = 54
"""费用暗盒的宽度（从卡的水平正中心往右，px）。

实测 6 帧：暗盒从中心那一列起、到中心 +54 那一列止（再往右就出了暗盒、进卡面）。
"""

COST_DIGITS_TOP: Final[int] = 12
"""费用数字**上缘**离卡片上缘多少像素 —— **它跳过的正是费用图标**。

卡的上缘（`squad_card_rightmost.y` = 1045）到数字上缘（1057）之间那 12 行住着
**部署费用那个图标**（实测它在暗盒顶部、横跨 x 2713–2733）。不跳过它，`glyph_boxes`
会把它和数字**按列并成一块**（图标横跨两个数字的列），整串就读不出来了。
"""

COST_DIGITS_HEIGHT: Final[int] = 29
"""费用数字那一条的高度（px）。实测 6 帧 56 张卡：数字都是 **29 px 高**（1057–1085）。

**上面不能再宽**：再往上 2 行就吃到图标；**下面也不能再宽**：再往下 2 行就吃到卡面
（暗盒下缘之外那一片亮起来，数字块被拉长，匹配分数从 0.94 掉到 0.50）。
"""

COST_INK_TAKE: Final[str] = INK_MIN
"""费用数字走**亮**那一侧（白/灰字压在近黑的暗盒上）—— 与费用面板那个数字同一类。

实测数字笔画核心是 `(128,128,128)`（`min = 128`），暗盒底色 `min = 5–29`。
"""

COST_INK_THRESHOLD: Final[int] = 64
"""费用的墨线。**它与 `glyphs.INK_THRESHOLD = 128` 不是同一个数**：那个是给
「白字压在 47–67 的渐变底上」用的，这一格的暗盒底色更低、笔画也更暗。

**实测这一栏的卡面是两档，不是一档**（6 帧 56 张卡）：

| 档 | 职业标识牌那一片的中位亮度 | 费用数字的峰值 | 是哪几张 |
| --- | --- | --- | --- |
| 压暗 | **86–111** | **128** | `shot01` 全部 12 张 · `shot02` 全部 10 张 · `shot07` 全部 8 张 · `shot21` 全部 11 张 · `shot05` 的 0–6 · `shot26` 的 0（再部署中那张） |
| 正常 | **198–207** | **255** | `shot05` 的 7–8 · `shot26` 的 1–5 |

（《备战席》§每张卡要多少费 说「**灰掉那一档要单独判**……这一档的取墨阈值不能照搬」
—— 这两档就是它说的那件事。**哪一档是"灰掉"我从像素上断定不了**，如实报上来。）

**64 对两档都成立**：压暗那一档的数字峰值是 128、正常那一档是 255，而两档的暗盒底色都
≤29、两档的卡面那一侧都 ≥100。所以 64 落在 **29…100** 那条空档里，**两档共用一条线**
—— 实测 56 张卡的费用**全部读成精确值**，没有一张掉进未知或未采样。
"""

MIN_DIGIT_WIDTH: Final[int] = 8
"""一位数字至少要多宽（px）才算**一位数字**。

实测 6 帧：真数字 **11–27 px**（最窄的是 `1`），而暗盒右缘 / 卡面漏进来的碎块
**1–4 px**。空档 4…11，取 8，两侧各留 4 / 3。**它是「形状」判断，不是「挑一个最像的」**
—— 一块 1 px 宽的墨迹不是一位数字，丢掉它不会丢掉任何一位。
"""

MAX_COST_DIGITS: Final[int] = 2
"""费用是 1–2 位（《备战席》§每张卡要多少费：实测跨 1–25）。"""

COST_SOLID_MATCH_SCORE: Final[float] = 0.86
"""这一位至少要像到什么程度，才配被认出来。

实测 6 帧 56 张卡 109 位：**正确那一位最低 0.8827**（`shot05` 第 8 张的费用 `10`），
而**不是数字的碎块最高 0.7527**。空档 0.7527…0.8827，取 0.86，两侧各留 0.107 / 0.023。

**它比 `cost.py` 的 `SOLID_MATCH_SCORE = 0.90` 低**：那一条是拿费用面板那一档量的
（笔画是 255 的纯白），而备战席这一档的笔画核心只有 128、卡面是半透明的，同一套模板
打出来的分整体低一档。**两条链各带各的线**（D-全局-24）。
"""

COST_AMBIGUITY_MARGIN: Final[float] = 0.03
"""次像的离最像的有多近才算「分不开」（照 `cost.py` 那个数）。

实测 6 帧 109 位：正确那一位与次像的差**最小 0.106**，所以 0.03 只会抓住真正并排的两个
—— 这一版在 6 帧上一次都没触发（**纸上的防线**，见报告）。
"""

RING_CENTER_DY: Final[float] = 116.5
"""再部署圆环的圆心离**卡片上缘**多少像素。

实测 `shot26` 那张：圆心 `(2695.5, 1161.5)`、外径 58、白色弧在半径 **51–58**
那一条环带上（内径 50 以内是数字、59 以外什么都没有）。
"""

RING_INNER: Final[int] = 50
"""环带的内半径（px）。**它同时是「读中央数字时要把环挖掉」的那把刀**：
不挖掉，`glyph_boxes` 会把整圈白环和数字**按列并成一块**（实测并出一块 116×116）。"""

RING_OUTER: Final[int] = 60
"""环带的外半径（px）。"""

RING_BRIGHT_MIN: Final[int] = 200
"""环上「这一度是白的」的判据（`min(R,G,B)`）。

实测 `shot26` 那张：白弧的像素 `min = 250–255`，环带里其余地方 `min ≤ 40`。
"""

RING_PRESENT_OUTER: Final[int] = 57
"""问「环在不在」时，环带**到哪一行为止**（px）—— 比 `RING_OUTER` 窄。

实测 `shot26` 那张：环的**实心部分**到 r ≈ 57 为止，r = 58 起就是环外的抗锯齿边与卡面
（那一张在 r = 58 上已经是卡面的 93/34/31）。读白弧时多含两圈无妨（白弧是 255，卡面比不过
它），但问「环带最暗的那一度」时**多含的两圈就是卡面** —— 实测把上界从 57 放到 60，
普通卡里最高那个读数从 54 抬到 65，空档白窄一截。
"""

RING_PRESENT_MIN: Final[int] = 95
"""环带上**最暗的那一度**至少这么亮，才算「环在」。

实测 6 帧 56 张真卡（环带 = 半径 50…57 上逐度取最亮的那个像素的 `min(R,G,B)`）：

| 卡 | 环带最暗的那一度 |
| --- | --- |
| `shot26` 下标 0（再部署中：红罩 + 灰环 + 白弧 + 中央数字） | **110** |
| 其余 55 张**普通卡** | **≤ 54** |

空档 54…110，取 95，两侧各留 41 / 15。**它问的是形状**：环是画满一整圈的，所以「最暗的
那一度还亮着」= 有一圈东西在；普通卡上那 360° 落在卡面各处，总有一度是暗的。
"""

RING_PRESENT_ABSENT_MAX: Final[int] = 70
"""环带上最暗的那一度**暗到这个程度**，才算「环不在」。见 `RING_PRESENT_MIN`。

70…95 之间**判不了** —— 那里有一圈不够亮的东西，报「是」或「不是」都是假事实。
**这一档在 6 帧 56 张卡上一次都没触发**（纸上的防线，见报告）。
"""

COUNTDOWN_MASK_RADIUS: Final[int] = 45
"""读中央秒数时**把环挖掉**的那把刀：半径小于它的像素才算数（px）。

实测 `shot26` 那张：数字最远只到 **r ≈ 40**（`1` 的左下角、`9` 的右下角），
而环的内缘在 **r ≈ 49.5**（环的笔画占 50–59，抗锯齿往里吃到 49 多一点）。
空档 40…49.5，取 45，两侧各留 5 / 4.5。

**不能就取 `RING_INNER = 50`**：`Frame.crop` 会把矩形的 y 取整（1101.5 → 1101），
圆心因此与环心差半像素 —— 那半像素正好让环的内缘抗锯齿漏进来，实测会把 `9` 那一块
从 29 px 高拉到 **57 px**（顶上多出一截环边），整串就读不出来了。
"""

COUNTDOWN_INK_THRESHOLD: Final[int] = 200
"""中央秒数的墨线。实测 `shot26` 那张：数字是**纯白 255**，环也是 250–255，
所以这一条线分不开数字与环 —— **分开它们的是形状（把环挖掉），不是线**。
"""

COUNTDOWN_MIN_DIGIT_WIDTH: Final[int] = 8
"""中央秒数里**一位数字**至少多宽（px）。实测 `shot26` 那张的 `1` 是 16、`2` 是 19、
`9` 是 18，而碎块 1–4。"""

COUNTDOWN_MIN_DIGIT_HEIGHT: Final[int] = 20
"""中央秒数里**一位数字**至少多高（px）。实测 29；小数点只有 5。"""

COUNTDOWN_DOT_MIN_WIDTH: Final[int] = 2
COUNTDOWN_DOT_MAX_WIDTH: Final[int] = 12
COUNTDOWN_DOT_MIN_HEIGHT: Final[int] = 3
COUNTDOWN_DOT_MAX_HEIGHT: Final[int] = 12
"""**小数点**那一块的尺寸范围（px）。实测 `shot26` 那张是 **6×5**；
碎块最小到 **1×2**（落在下界之外）。《识别方法》§已定 说这个游戏的数字
「只有 0-9 与小数点」，而**字形表里没有小数点** —— 所以它按**形状**认，不按模板比。
"""

COUNTDOWN_DOT_BASELINE_TOL: Final[int] = 4
"""小数点必须**贴着数字的基线**（它的下缘离数字块下缘不超过这么多 px）。

实测是齐的（都在 y 1176）。加这一条是为了把「浮在上面的小碎块」挡掉。
"""

MAX_COUNTDOWN_WHOLE_DIGITS: Final[int] = 3
"""秒数的整数部分最多 3 位（《备战席》：实测到小数位，见过的最大是 `68.6`）。"""


@dataclass(frozen=True)
class SquadCard:
    """备战席上**一张卡**的读数。`index` 从 0 起、**0 = 最右那张**（T5 定）。

    - `center_x` 是这张卡的**水平正中心**（画布像素）；
    - `center_from_jump` 说这个中心是**像素读出来的**还是**网格备选给的**
      —— 两者差 ≤ 2 px，但「谁给的」是证据，不能吞掉；
    - `cost` / `seconds` 是槽值（精确 / 未知 / 未采样）；
    - `redeploying` 是三值：`True` / `False` / `None`（判不了）——
      判据是**环在不在**，不是白弧多长（见模块文档）；
    - `ring_fraction` 是环上白弧占整圈的比例 —— **已过比例**（0…1），不是进度、
      更不是剩余；不在再部署中就是 `None`。**它和 `seconds` 不同单位**，见模块文档。
    """

    index: int
    center_x: float
    center_from_jump: bool
    cost: SlotValue
    redeploying: bool | None
    seconds: SlotValue
    ring_fraction: float | None


def _guard(frame: object, glyphs: object, pitch: object) -> None:
    """入口的类型闸门（照 `cost.py` / `topbar.py` 的样子）。"""
    if not isinstance(frame, Frame):
        raise TypeError(f"要读的是一帧，收到 {type(frame).__name__}")
    if not isinstance(glyphs, GlyphTable):
        raise TypeError(f"要一张字形表，收到 {type(glyphs).__name__}")
    if isinstance(pitch, bool) or not isinstance(pitch, (int, float)):
        raise TypeError(f"卡距要是数，收到 {type(pitch).__name__}")
    if pitch <= 0:
        raise ValueError(f"卡距必须为正，收到 {pitch}")


def _anchors() -> tuple[PixelRect, PixelRect]:
    """最右那张卡的**卡框**与**身份区**（`"squad_card_rightmost"` 那两个键）。

    名字是契约层 `UI_LAYOUT` 里定的，这一层只管取 —— 取不到就报出来，
    **不另抄一份坐标**。

    **只有这一份表**：卡框那两个键住契约层，`squad_card_rect` 也写死读它。所以这一层
    **不收 `ui` 参数** —— 一个只能换掉锚点、换不掉位移的参数比没有参数更危险。
    """
    names = ("squad_card_rightmost", "squad_card_rightmost_face")
    missing = [name for name in names if name not in UI_LAYOUT.rects]
    if missing:
        raise KeyError(
            f"UI 常量里没有 {missing}：这一版认识 {sorted(UI_LAYOUT.rects)}"
        )
    return UI_LAYOUT.rects[names[0]], UI_LAYOUT.rects[names[1]]


def _row_strip(frame: Frame) -> PixelRect:
    """整条卡行的**窗口**：左右铺满画布，上下取「卡片上缘 → 身份区上缘」。

    上下那两条边**走 `squad_card_rect` / `squad_card_face_rect`**：卡上缘 1045、
    身份区上缘 1092，那中间那 47 行住着职业标识牌与费用暗盒（`"顶上那一条"`）。
    左右铺满是因为**卡数会变、左缘随卡数变**，而这一层不许拿卡数去推左缘。
    """
    card, face = _anchors()
    return PixelRect(0, card.y, frame.canvas.width, face.y - card.y)


def _jump_columns(frame: Frame, strip: PixelRect) -> list[int]:
    """整条卡行上，**每一道「左边一大块亮、右边一大块暗」的跳变**在哪一列。

    返回的是**亮的那一侧的最后一列** —— 中心是它的**下一列**（实测：最右那张卡的
    跳变落在 2695/2696 之间，而 `squad_card_rect` 给的中心是 2696）。

    判据用的是**每列的 `min(R,G,B)` 中位数**（不是均值）：职业标识牌里压着一个
    **深色的职业图标**、费用暗盒里压着**亮数字**，逐列取中位数才不会被它们带偏
    —— 图标最多占一列的一半高，中位数还在亮那一侧。
    """
    ink = ink_of(frame.crop(strip), take=INK_MIN)
    column = np.median(ink, axis=0)
    return [
        int(strip.x) + x
        for x in range(column.size - 1)
        if column[x] >= JUMP_LIGHT_MIN and column[x + 1] <= JUMP_DARK_MAX
    ]


def _one_digit(
    glyphs: GlyphTable,
    ink: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    solid: float,
    margin: float,
) -> tuple[int | None, float]:
    """这一块墨迹能不能读成**一位确切数字**：读得出给数字，读不出给 `None`。

    读不出的两种情形都归 `None`：

    - **对不上任何字形**（最像的那个低于 `solid`）—— 我们知道那里有一位数字，
      但不知道它是几。**不许挑一个最像的**；
    - **并排候选**（次像的离最像的不到 `margin`）—— 两个都中等分，挑一个就是假事实。

    第二个返回值是最像的那个的分数（置信度取这一串里最弱的那个）。
    """
    x0, y0, x1, y1 = box
    scores = glyphs.scores(normalize(ink[y0:y1, x0:x1]))
    best = max(scores.values())
    if best < solid:
        return None, best
    rivals = sorted(d for d, s in scores.items() if s >= best - margin)
    if len(rivals) != 1:
        return None, best
    return rivals[0], best


def _read_cost(
    frame: Frame, glyphs: GlyphTable, window: PixelRect
) -> SlotValue:
    """费用暗盒里那个数字 → 一个槽值。**只有两种落法**（《备战席》§每张卡要多少费）：

    - **没笔画** → `Unsampled()`（那里没有东西）；
    - **有笔画但读不全** → `Valued(Unknown(), None)`：某一位认不出 · 并排候选 ·
      块数超过 2（费用只有 1–2 位）—— **不许拿"挑最像的"补上**；
    - 逐位都读出确切值、位数也对 → `Valued(Exact(n), 置信度)`。

    **绝不像 `cost.py` 那样落区间**：费用的字段里**有**区间，备战席这一行**没有**
    （《状态层/我方干员》「当前部署费用」：精确 / 未知 / 未采样）—— 形态是字段级的决定。
    """
    roi = frame.crop(window)
    ink = ink_of(roi, take=COST_INK_TAKE)
    boxes = [
        box
        for box in glyph_boxes(roi, take=COST_INK_TAKE, threshold=COST_INK_THRESHOLD)
        if box[2] - box[0] >= MIN_DIGIT_WIDTH
    ]
    if not boxes:
        return Unsampled()
    if len(boxes) > MAX_COST_DIGITS:
        return Valued(Unknown(), None)
    read = [
        _one_digit(
            glyphs,
            ink,
            box,
            solid=COST_SOLID_MATCH_SCORE,
            margin=COST_AMBIGUITY_MARGIN,
        )
        for box in boxes
    ]
    if any(digit is None for digit, _ in read):
        return Valued(Unknown(), None)
    digits = [digit for digit, _ in read if digit is not None]
    return Valued(
        Exact(int("".join(str(digit) for digit in digits))),
        min(score for _, score in read),
    )


def _ring_profile(frame: Frame, centre_x: float, card_top: float) -> np.ndarray:
    """卡片正中那个圆环的**径向剖面**：`(360, RING_OUTER - RING_INNER + 1)`。

    整圈按 1° 采样（0° = 12 点、顺时针增大）；每一度上，半径
    `RING_INNER`…`RING_OUTER` 的每一个像素取 `min(R,G,B)`，存成一列。

    **只采一次样**：白弧（要 `RING_OUTER` 那么宽）与「环在不在」（只要
    `RING_PRESENT_OUTER` 那么宽）是同一批像素上的两个不同问题；分开采样会让同一个
    圆环被读两遍，还会让两次读的圆心各自算一遍。
    """
    radius = RING_OUTER
    centre_y = card_top + RING_CENTER_DY
    rect = PixelRect(
        centre_x - radius, centre_y - radius, radius * 2, radius * 2
    )
    roi = frame.crop(rect)
    profile = np.zeros((360, RING_OUTER - RING_INNER + 1), dtype=int)
    for degree in range(360):
        theta = np.radians(90.0 - degree)
        cos, sin = float(np.cos(theta)), float(np.sin(theta))
        for step, r in enumerate(range(RING_INNER, RING_OUTER + 1)):
            x = int(round(radius + r * cos))
            y = int(round(radius - r * sin))
            if 0 <= y < roi.shape[0] and 0 <= x < roi.shape[1]:
                profile[degree, step] = int(roi[y, x].min())
    return profile


def _ring_arc(profile: np.ndarray) -> int:
    """白弧**最长连续多少度**（0° = 12 点、顺时针）。

    每一度取环带里**最亮**的那个像素，它过 `RING_BRIGHT_MIN` 就算这一度是白的。
    然后取**环形最长连续段**。

    **为什么按度数而不是按像素个数**：卡面本身也可能有大片白（白发角色），按个数会被
    它带偏；「连续多少度」问的是**形状**——环是圆的，卡面不是。

    **它给的是「已过比例」，不是「是不是再部署中」**（判据是 `_ring_floor`）：
    刚撤退的卡这一路接近 0°，拿它当判据会把「正在再部署」说成「不在再部署」。
    """
    hit = profile.max(axis=1) >= RING_BRIGHT_MIN
    doubled = np.concatenate([hit, hit])
    longest = run = 0
    for value in doubled:
        run = run + 1 if value else 0
        if run > longest:
            longest = run
    return min(longest, 360)


def _ring_floor(profile: np.ndarray) -> int:
    """环带上**最暗的那一度**有多亮 —— 「环在不在」的判据。

    只看 `RING_INNER`…`RING_PRESENT_OUTER` 那几列（再往外就是卡面）。环是画满一整圈
    的，所以「最暗的那一度还亮着」= 有一圈东西在。两条线见 `RING_PRESENT_MIN`。
    """
    width = RING_PRESENT_OUTER - RING_INNER + 1
    return int(profile[:, :width].max(axis=1).min())


def _read_countdown(
    frame: Frame, glyphs: GlyphTable, centre_x: float, card_top: float
) -> SlotValue:
    """圆环中央那个秒数（一位小数）→ 一个槽值。

    环**先挖掉**（半径 `RING_INNER` 以内才算数），否则 `glyph_boxes` 会把整圈
    白环和数字按列并成一块。然后：

    - **一块墨迹都没有** → `Unsampled()`；
    - **认不出小数点**（不是恰好一块贴着基线的点状块）→ `Valued(Unknown(), None)`：
      秒数**必须**有小数位（实测 `12.9` / `68.6`），认不出它就没法把整数位与
      小数位分开，报一个整数就是假事实；
    - **某一位认不出 / 整数位超过 3 位** → `Valued(Unknown(), None)`；
    - 否则 → `Valued(Exact(秒), 置信度)`。

    **小数点按形状认，不按模板比**：字形表里只有 0-9（《识别方法》§已定 说这个游戏的
    数字「只有 0-9 与小数点」），所以它靠尺寸与基线定位。
    """
    radius = RING_OUTER
    centre_y = card_top + RING_CENTER_DY
    rect = PixelRect(
        centre_x - radius, centre_y - radius, radius * 2, radius * 2
    )
    roi = np.array(frame.crop(rect))
    rows, columns = np.mgrid[0 : roi.shape[0], 0 : roi.shape[1]]
    # 圆心在 ROI 里的**精确**位置：`crop` 把矩形原点取了整，所以这里也照那个原点算。
    middle_x = centre_x - int(rect.x)
    middle_y = centre_y - int(rect.y)
    inside = (columns - middle_x) ** 2 + (rows - middle_y) ** 2 < (
        COUNTDOWN_MASK_RADIUS**2
    )
    roi[~inside] = 0
    ink = ink_of(roi, take=INK_MIN)
    boxes = glyph_boxes(roi, take=INK_MIN, threshold=COUNTDOWN_INK_THRESHOLD)
    digits = [
        box
        for box in boxes
        if box[2] - box[0] >= COUNTDOWN_MIN_DIGIT_WIDTH
        and box[3] - box[1] >= COUNTDOWN_MIN_DIGIT_HEIGHT
    ]
    if not digits:
        return Unsampled() if not boxes else Valued(Unknown(), None)
    baseline = max(box[3] for box in digits)
    dots = [
        box
        for box in boxes
        if COUNTDOWN_DOT_MIN_WIDTH <= box[2] - box[0] <= COUNTDOWN_DOT_MAX_WIDTH
        and COUNTDOWN_DOT_MIN_HEIGHT <= box[3] - box[1] <= COUNTDOWN_DOT_MAX_HEIGHT
        and abs(box[3] - baseline) <= COUNTDOWN_DOT_BASELINE_TOL
    ]
    if len(dots) != 1:
        return Valued(Unknown(), None)
    dot = dots[0]
    whole = [box for box in digits if box[2] <= dot[0]]
    fraction = [box for box in digits if box[0] >= dot[2]]
    if not whole or len(fraction) != 1:
        return Valued(Unknown(), None)
    if len(whole) > MAX_COUNTDOWN_WHOLE_DIGITS:
        return Valued(Unknown(), None)
    read = [
        _one_digit(
            glyphs,
            ink,
            box,
            solid=COST_SOLID_MATCH_SCORE,
            margin=COST_AMBIGUITY_MARGIN,
        )
        for box in whole + fraction
    ]
    if any(digit is None for digit, _ in read):
        return Valued(Unknown(), None)
    tenths = int("".join(str(digit) for digit, _ in read))
    return Valued(Exact(tenths / 10.0), min(score for _, score in read))


def read_squad_row(
    frame: Frame,
    glyphs: GlyphTable,
    *,
    pitch: float,
) -> tuple[SquadCard, ...]:
    """读备战席那一栏：**一帧 → 每张卡一条读数**（下标 0 = 最右那张）。

    `pitch` 是**卡距**（标量）。它住设备层的 `SQUAD_CARD_PITCH`，而导入规则不许
    视觉层 import 设备层，所以**由调用方传进来** —— 契约层 `squad_card_rect` 的
    docstring 写的就是这个口径。

    **不收 `ui` 参数**：卡框那两个键只有一份表（契约层 `UI_LAYOUT`），而
    `squad_card_rect` 也写死读它 —— 传一张别的表进来只能换掉锚点、换不掉位移，
    **一个只能换一半的参数比没有参数更危险**（调用方会以为换了表）。要换表就换契约层
    那一份，这里不开口子。

    **卡数是数出来的**：从最右那张起，一格一格往左问「这一格有没有跳变证据」，
    问到没有为止。右对齐 + 只有还没上场的才显示 ⇒ 有卡的那些格**一定是连着的一片**
    （中间缺一格说明那一格的跳变没读出来，那时**停在缺口处**、报出已经读到的那些）。

    **再部署中的判据是环在不在**（`_ring_floor`），不是白弧多长 —— 白弧量的是
    **已过比例**，刚撤退的卡白弧接近 0°，拿它当判据会把「正在再部署」说成「不在再部署」。
    **红罩也不单独当判据**：普通卡里整片红的角色比红罩还红。

    **不在再部署中的卡，秒数一律 `Unsampled()`**（那里没有倒计时可读）——
    与「有笔画但认不出 → 未知」是两支，不许混。**判不了那一档照读**：那里有没有倒计时
    是**肯定证据**，读到了就是读到了；读不到就按秒数自己的三档落。
    """
    _guard(frame, glyphs, pitch)
    anchor, _ = _anchors()
    strip = _row_strip(frame)
    jumps = _jump_columns(frame, strip)
    top = float(anchor.y)
    anchor_centre = float(squad_card_rect(0, pitch).x) + float(anchor.width) / 2.0

    # 一格一格问过去：这一格的网格中心附近有没有跳变证据。
    found: dict[int, int] = {}
    index = 0
    while True:
        card = squad_card_rect(index, pitch)
        if float(card.x) + float(card.width) <= 0.0:
            break
        expected = anchor_centre - index * pitch
        near = [x for x in jumps if abs((x + 1) - expected) <= CENTER_MATCH_TOL]
        if near:
            found[index] = min(near, key=lambda x: abs((x + 1) - expected))
        index += 1
    if not found:
        return ()
    # 右对齐 + 只有还没上场的才显示 ⇒ 有卡的那些格是**连着的一片**：
    # 最远那一格有卡，它右边每一格就都有卡。中间缺一格只说明那一格的跳变没读出来。
    count = max(found) + 1

    cards: list[SquadCard] = []
    for index in range(count):
        expected = anchor_centre - index * pitch
        if index in found:
            centre = float(found[index]) + 1.0
            from_jump = True
        else:
            centre = expected  # 备选：锚点 + 卡距（这一格的跳变没读出来）
            from_jump = False
        cost = _read_cost(
            frame,
            glyphs,
            PixelRect(
                centre,
                top + COST_DIGITS_TOP,
                COST_BOX_WIDTH,
                COST_DIGITS_HEIGHT,
            ),
        )
        profile = _ring_profile(frame, centre, top)
        floor = _ring_floor(profile)
        if floor >= RING_PRESENT_MIN:
            redeploying: bool | None = True
        elif floor <= RING_PRESENT_ABSENT_MAX:
            redeploying = False
        else:
            redeploying = None
        ring_fraction = float(_ring_arc(profile)) / 360.0
        if redeploying is False:
            seconds: SlotValue = Unsampled()
            ring_fraction = None
        else:
            seconds = _read_countdown(frame, glyphs, centre, top)
        cards.append(
            SquadCard(
                index=index,
                center_x=centre,
                center_from_jump=from_jump,
                cost=cost,
                redeploying=redeploying,
                seconds=seconds,
                ring_fraction=ring_fraction,
            )
        )
    return tuple(cards)
