"""场上我方干员：三条读数链 —— **血条 · 技力条 · 可释放菱形**。

《视觉层/场上我方干员》§1–§3 管这一层。三条链各自只回答一件事：

| 链 | 读什么 | 载荷 |
| --- | --- | --- |
| `read_health` | 蓝条**自左端起亮了多少**（比例） | 精确 / 区间 / 下界 / 未采样 |
| `read_skill` | 技力条**填了多少** + **它是什么含义** | 精确 / 区间 / 下界 / 未采样（+ 含义） |
| `find_ready_diamond` | 那个黄菱形**在不在** | 布尔 / 未采样 |

## 三条元素画在同一条横带上

一帧上，一个干员的那三样是**上下一串**（原生 2800×1260 实测）：

- **血条**：146×8 的横条，填充**自左端起**，颜色实测 (110, 190, 228)；
- **技力条**：**紧贴在血条下面那一行**（再往下 8 行），同样 146×8；充能中是黄绿、
  生效中是橙、弹药技那条上有一列**竖线分格**；
- **黄菱形**：挂在干员**头顶**，它的中心在血条左上角 **+(72, −233)** 处 —— 这一条
  实测过 **7 个干员 / 5 个不同来源的帧**，偏差 **0 / −0.5 px**。

所以三条链共用**一个取区**（OPERATOR_UI_WIDTH × OPERATOR_UI_HEIGHT），血条左上角
落在取区里的 (40, 320)。

## 取区从哪来：**它是参数，这一层不产生它**

干员在屏幕上的位置来自**部署记录 + 投影**（状态层/我方干员 §位置）—— 那个
「在哪」**不在这一单**，而且 AssetPort.projection 的值现在还不存在。所以三条链
都把**取区**收成一个 PixelRect 参数：**调用方开窗，这一层量窗里的东西**。

**这不是把 ROI 预先替识别链做好**（[基准集] §3 那条）：这一批的裁剪比取区大一圈，
开窗这一步仍是真的在被测 —— 只是被测的那一段是"**窗里的哪一块是条**"。

## 宽容窗口：窗的位置允许差，链自己找

取区的原点是**记录推出来的**，它一定带像素级的误差。所以三条链都**不钉死**在
(40, 320) 上：

- **行方向**：在 (40, 320) 上下各 ROW_TOL 行里找那条最像的；
- **列方向**：在血条应然左缘左右各 COL_TOL 列里找那条**真正的左缘**；
- **菱形**：在应然位置周围 ±DIAMOND_SEARCH_RADIUS 的**邻域**里滑着搜，不是钉在
  一个 64×64 的小框里分类（《视觉层/场上我方干员》§3 与 MAA 的取舍差异就在这一条）。

## 「看不到」与「认不出」**分得开**

这是这一族最要紧的一条纪律（《状态层/我方干员》§容易吞掉的语义边界 · 《识别方法》
§已定 末尾）：

| 情形 | 这一层怎么落 | 为什么 |
| --- | --- | --- |
| **看不到 / 没笔画** | Unsampled() | 那里没有东西，或我们没能看到；**血条是常驻的，看不到就是识别失败**，绝不是"满血" |
| **认得出有东西、但读不准** | Interval / LowerBound / Unknown | 那里确实有东西 —— 报未采样就是把"看错了"伪装成"没看到" |
| **能力缺失**（模板 / 取区这个前提本身不成立） | **抛异常** | 与「表没建好 ≠ 这张卡没有倒计时」同一条：静默降级成"没有"是假缺席 |

**每一条链只产它那个字段「允许的形态」里的东西**（状态层那张表）：血量是
精确 / 区间 / 下界 / 未采样，所以它**不产 Unknown**；可手动触发是布尔 / 未采样，
所以它**不产区间** —— 那条链的"认不出"落 Unsampled，而且**绝不落 Exact(False)**：
把"认不出"读成"不在"正是这一族最危险的**漏检方向**（《识别方法》存在性搜索那行的 ⚠）。

## 与线上那条链的关系

这一层**不产状态层字段**：它只把三样读数交出去，「技力生效期间这个字段落未采样」
是**字段**的决定，由调用方按 含义 那一支去落（`sp_of` 把它写成了一个函数，
省得每个调用点各抄一遍）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import cv2
import numpy as np

from ark.contracts import (
    Exact,
    Interval,
    LowerBound,
    PixelRect,
    SlotValue,
    Unknown,
    Unsampled,
    Valued,
)
from ark.engine.vision.frame import Frame

__all__ = [
    "read_health",
    "read_skill",
    "sp_of",
    "find_ready_diamond",
    "diamond_score",
    "load_ready_diamond",
    "save_ready_diamond",
    "operator_ui_window",
    "READY_DIAMOND_PATH",
    "DIAMOND_ICON_FILENAME",
    "HP_BLUE",
    "HP_COLOUR_TOL",
    "BAR_WIDTH",
    "BAR_HEIGHT",
    "SKILL_ROW_OFFSET",
    "OPERATOR_UI_WIDTH",
    "OPERATOR_UI_HEIGHT",
    "BARS_IN_WINDOW",
    "DIAMOND_IN_WINDOW",
    "ROW_TOL",
    "COL_TOL",
    "MIN_FILL_COLUMNS",
    "GAP_TOLERANCE",
    "COVER_GAP_TOLERANCE",
    "GREEN_MAX_RG",
    "ORANGE_MIN_RG",
    "SEPARATOR_DROP",
    "SKILL_FILL_TOL",
    "DIAMOND_SEARCH_RADIUS",
    "DIAMOND_SOLID_SCORE",
    "DIAMOND_ABSENT_SCORE",
]


HP_BLUE: Final[tuple[int, int, int]] = (110, 190, 228)
"""血条填充的颜色（实测，原生帧上逐点一致）。"""

HP_COLOUR_TOL: Final[int] = 24
"""一列算不算"血条填着"：它的**列中位数**颜色的逐通道最大差 ≤ 这个数。

**这个数是量出来的**：cd 那一批 61 帧里，血条填充列的实测差是 **0–1**；而同一根条
**空着**的那一截是 **146–188**（cd01 的 y=476 那根：第 129 列之后整段跳到 185 以上）。
中间那条空档宽得离谱，所以 24 只是"取在空档里靠填充那一侧"的保守值 —— 填充那一侧实测
差 **0–1**，空轨道那一侧是 **146–188**，24 离两侧都很远。
"""

BAR_WIDTH: Final[int] = 146
"""一根条的**整长**（含空着的那一截），原生像素。

它在一帧上是**常量**：cd 那一批 61 帧里，同一根条的左缘固定在 x = 1597，
而整长从来没有变过（满填充 = 146，实测 206 根满条全是 146）。
"""

BAR_HEIGHT: Final[int] = 8
"""一根条的**高**（血条 8 行，技力条也 8 行，两者只差 8 行）。"""

SKILL_ROW_OFFSET: Final[int] = 8
"""技力条在血条**下面第几行**：血条占 y..y+7，技力条那一带就是 y+8..y+15。

**这 8 行是"填充那 8 行"** —— 技能关那条灰轨道、技能开那条亮金填充、弹药那条橙板，
画的都是这 8 行；条自己还要再往下多 2 行，见 `SKILL_TRACK_TAIL_ROWS`。

**±1 行会飘**（run-02 实测）：`S3-1-4` 的 **+8 整行是血条蓝**（逐行拟合出 A = 0.007、
B = (108.4, 188.8, 226.4)，就是 HP_BLUE 自己）；弹药帧的填充有时画到 **+16** 才结束。
所以"边界行归谁"不固定 —— 这一带**只当一条 8 行的带子用**（取逐列中位），
**不许**把其中某一行当成"一定是轨道"。
"""

SKILL_TRACK_TAIL_ROWS: Final[tuple[int, int]] = (17, 18)
"""技力条**最后那一行**（相对血条顶行）—— 板的下沿。`(17, 18)` 是半开区间、只有一个行号。

**为什么只取 +17、不取 +16…+17**（run-02 实测，逐列中位口径，底取同场景干净帧）：

- **两行合起来会坏**：`S8-3-4` 上 A = 0.251、残差 sd **13.6**（+16 那一行是**橙板** A = 0.444，
  不是灰轨道）；`S4-1-5` / `S4-2-3` 上 A 掉到 0.289 / 0.274；
- **单取 +17 就好**：五帧弹药上 A = **0.58…0.72**、残差 sd ≤ **0.54**；灰轨道那 8 行是
  A = 0.64…0.71（同一批帧）—— 两者对得上，所以 +17 是**同一根轨道的最后一行**。

**它的用处只有一个**：弹药那一档里，+8…+15 是**橙板**（A = 0.43…0.49），
灰轨道只剩这一行可以当证人。**它不是"第二根条"**，只是同一根条露出来的一行。
"""

_TRACK_ROW_GAP: Final[int] = 10
"""量「条内比条外」时，「外」取**本行往下第几行**（相对本行，不是相对条顶）。

**恒定间隔 10**，不是「固定取某一绝对行」。条下面那块地形**逐行都在变**
（横差 60…130 px、竖差 10…15 行）：固定取一行时，行带里越靠下的行借到的底越远、
偏差越大，而且**每一帧偏多少都不一样**。恒定间隔让八行各自借到**同样竖距**的底，
把「地形竖着差多少」这件事在八行之间拉平 —— 这就是借底这条路的天花板所在。

**为什么是 10**：运行时取区只有 360 行、条顶落在取区第 320 行，判据最深要用到
「320 + 22 + 10 = 352 < 360」（+22 是自控行的最后一行）。Δ=15 已到 357（没有余量）、
Δ=20 / Δ=23 出界。批次里裁剪是 300×420、能用 Δ=23，但**判据按运行时的预算定**。

**和阴影的关系**：本行或它往下第十行只要落在 +18…+22 里，量到的数里就混着那道阴影
—— 要按 _TRACK_SHADOW 除回去（见 _track_ratio）。
"""

_TRACK_SHADOW: Final[dict[int, float]] = {
    18: 0.81,
    19: 0.90,
    20: 0.94,
    21: 0.97,
    22: 0.98,
}
"""条下面那道**阴影**在每一行上把地形压到原来的几分之几（键 = 相对血条顶行的偏移）。

逐行实测（run-02 八帧，活帧 ÷ 干净帧）：**+18 0.74…0.88 / +19 0.87…0.93 /
+20 0.92…0.97 / +21 0.96…1.00 / +22 0.98…1.03**，**+23 起 1.00**（残差 sd 0.0…1.1）。
表里取的是那一段的中值；**≤ +17 与 ≥ +23 都取 1.00**（那两段没有阴影）。

**为什么要它**：判据拿「本行 ÷ 往下第十行」当比值，只要**两边里有一边**落在 +18…+22，
量到的就是被阴影压过的数 —— 不修正的话，行带里靠上的那几行会凭空多出 0.02…0.19 的偏置
（那正是这几行在干净底上本该有的值）。查表一律按**相对条顶的偏移**，不许按画面绝对行号。

**它同时是「底变了吗」的试纸，逐行容差 ±0.10**（用户定的）：按 ±0.15 放，「S5-2-4」
那种「条底下又多盖了一层半透明」会被吞进去（它 +18 量到 0.889，干净帧那一档 0.74…0.88）。
"""

_TRACK_CONTROL_ROWS: Final[tuple[int, int]] = (18, 23)
"""**帧内自控**用哪几行（相对血条顶行）：**+18…+22**。

这 5 行**只有条自己的阴影、没有条** —— 所以它们是**活帧自带的一把尺**。自控量的就是
_track_ratio（阴影已除回去）：干净底上这五个数应当回到 **0** 附近；它们要是离 0 很远，
说明这一格的地形**竖着变太快**，「借条下面那块地形当底」这件事在这个窗上就不成立 ——
那几块一律落未采样，**不许硬给**。
"""

_TRACK_CONTROL_MAX: Final[float] = 0.30
"""自控那条线的门槛：+18…+22 五个比值（**已除阴影**）的**绝对值都要 ≤ 0.30**。

**它管的是"允不允许否决"**：不过 ⇒ 判据**弃权**（照旧给值），因为那五个数说的是
「这一格的地形竖着变太快、借底在这个窗上不成立」—— 那是**这件仪器不适用**，不是
「断口之后那一截不是轨道」。用户点名的两块亮底真轨道就落在这里（对照 0.45 / 0.51）。

**0.30 先按它走**（用户：「落 0.3 即可」）：未修正的对照在难例批 52 块上的最大绝对值，
从小到大排到 **0.288 与 0.327 之间有一处明显空隙**（差 0.039），门槛最初落在缝里；
把阴影除回去之后这五个数被拉回 0 附近，0.30 就变成「**地形的竖向变化不许超过 0.30**」。

**这是「先按它走」的数，不是量到底的数**：修正之后的门限**还没有一次干净的扫描**
（探针 r99 那次口径作废）。它现在只决定「否决线允不允许开火」，落错方向的代价是
「该拦的没拦」或「不该拦的拦了」，不是「报假值」。
"""

_TRACK_VETO: Final[float] = 0.17
"""断口之后那一截**还是不是轨道**的否决线：行带中位**取绝对值**，**< 0.17 ⇒ 不是轨道**。

**为什么看绝对值（两头的线）**：「明显变暗」是半透明轨道在压暗底；「**明显变亮**」是
亮底上借底那点地形差**翻了个号** —— 用户点名确认真轨道的两块都在变亮那一侧。
这一族的统计**分不开**下面三块（探针 r101 实测）：

    r01_01_06       比值 +0.111   ← 用户裁定：不是轨道
    亮底上一块真轨道  比值 −0.170   ← 用户裁定：真轨道（未填充轨道）
    橙色那一块真轨道  比值 −0.570   ← 用户裁定：真轨道（橙色轨道）

要拦的那一块（+0.111）**比两块真轨道都高**，所以「只拦正的一侧」必然把真轨道一起
拦掉 —— **两个方向都不明显，才是"没落差"**。0.17 取在 +0.111 与「干净底上真轨道
0.26 以上」之间（难例批里判「是轨道」那几块的中位是 0.259…0.494），
另一侧同一根线（|−0.170| 刚好在线上）。

**两头夹的带子（0.25…0.55）试过，作废**：它在难例批上会冤枉一大片真轨道。
"""

_TRACK_MIN_COLUMNS: Final[int] = 20
"""断口之后那一小截**至少要有几列**才有资格下判。

不足就给「窗太短」（照旧给值）。**20 这一条是「先按它走」**：探针 r101 用的就是这一档，
而更早那次窗宽扫描（8 列 / 12 列 / …）因为口径 bug **作废** —— 所以「为什么不是 12 或 30」
**还没有一次干净的测量**。

难例批里填充画到条右端的那十几块**没有断口**，判据根本不该触发 —— 调用方在
「tolerant >= BAR_WIDTH」时直接给 "short"。
"""

_TRACK_WINDOW_COLUMNS: Final[int] = 60
"""断口之后**最远扫到多少列**。

扫区从**断口 + 2 列**起（跳掉断口那一两列本身的抗锯齿混色），右端取
「min(条右端 − 9 = 137, 起 + 60)」 —— 退 9 列是避开与右邻叠着画的那 8 列。
窗宽上限只影响「凑不凑得够列数」（见 _TRACK_MIN_COLUMNS），不影响结论方向。
"""


def _track_ratio(
    region: np.ndarray, row: int, base: int, c0: int, c1: int, gap: int = _TRACK_ROW_GAP
) -> float:
    """**某一行**的「内」比它**隔开 gap 行的那一行**（外）暗了多少 —— 整行的中位。

    gap 是**相对本行**的偏移：默认 +_TRACK_ROW_GAP（**往下**借底，技力条那一条用）；
    **血条那一条用负数**（往上借底，也就是"上沿"）—— 血条紧贴技力条上方，它那一侧
    只剩上面能用（下面被技力条占着）。负偏移时 _TRACK_SHADOW 查不到表、按 1.00 走，
    这是对的：条**上面**没有阴影。

    **每一行各自算一个数**（先逐列相除、再对这一行的列取中位），返回这一个；
    调用方拿行带那八个数**取中位** —— 不是「八行先平均、再逐列相除」。两种口径在同一批
    像素上差得很远：回归批 s00 逐行算是 **0.273**（是轨道），八行先平均是 **0.086**
    （被冤枉成「不是轨道」）—— 那 60 列上底的亮度从 65 走到 217，先平均等于让最亮的
    那几列把整条压平。

    **阴影修正**：本行（base 是本图那条血条的顶行 → 键 = 相对条顶的偏移）或它往下第十行
    落在 +18…+22 里时，量到的数里混着那道阴影，按 _TRACK_SHADOW 除回去；别处取 1.00。
    **查表一律按相对条顶的偏移**，不许按画面绝对行号（犯过一次，修正空转）。

    **为什么要借底**：条上那一截的底**拿不到**，只能借条下面那一块地形。两块地形的
    逐列乘性差在相除里消掉，而在一元回归里会把结果拽死（实测回归的 A 在 −1.3…0.39
    之间乱跳，比值却稳定）。取不到底（越出取区 / 底上有一列 ≤ 0）给 nan。
    """
    outer_row = row + gap
    if outer_row >= region.shape[0] or outer_row < 0:
        return float("nan")
    inner = region[row, c0:c1].astype(np.float64).mean(axis=1)
    outer = region[outer_row, c0:c1].astype(np.float64).mean(axis=1)
    if inner.size == 0 or (outer <= 0).any():
        return float("nan")
    s_in = _TRACK_SHADOW.get(row - base, 1.0)
    s_out = _TRACK_SHADOW.get(outer_row - base, 1.0)
    return float(np.median(1.0 - inner * s_out / (outer * s_in)))


def _track_verdict(region: np.ndarray, row: int, first: int, gap: int) -> str:
    """**断口之后那一小截是不是半透明的轨道** —— 只用活帧自己。

    返回四选一（调用方按它落法）：

    | 返回 | 意思 | 怎么落 |
    | --- | --- | --- |
    | track | 是轨道（或者这件仪器在这块地形上不适用、弃权） | 照旧给值 |
    | covered | 不是轨道（有东西压在 UI 上 / 压根不是条） | 下界 + 含义未采样 |
    | unclear | 一个比值都取不到（底上有一列 ≤ 0） | 下界 + 含义未采样 |
    | short | 断口离条右端太近、或取区不够深 | 照旧给值 |

    **顺序**：先看自控、再看否决线。自控（+18…+22 那五行的 _track_ratio）是「这个窗上
    地形竖着变没变太快」的尺；**它不过时判据弃权**（不否决、照旧给值）—— 见
    _TRACK_CONTROL_MAX 的文档。真正会改读数的只有两档：否决线（covered）与
    比值取不到（unclear）。

    **否决线是两头的**（看绝对值）：明显变暗是轨道在压暗底；**明显变亮是亮底上借底
    那点地形差翻了个号**（用户点名的两块真轨道都在这一侧）；**两个方向都不明显**才是
    "没落差"。为什么不能只拦正的一侧：见 _TRACK_VETO 的文档。

    **用哪一个「断口」**：gap 是**不夹容错**数出来的那一截（调用方传 strict）——
    夹了 GAP_TOLERANCE 的那个数在缝的另一侧，拿它当断口会把被压住的那几列当成没填。
    扫区从**断口 + 2 列**起（跳掉断口那一两列本身的抗锯齿混色）。

    **为什么不用一元回归**：条外那块地形不是条下那一块，回归解出来的 A 会被**衰减**
    （实测 A 在 −1.3…0.39），而比值是尺度无关的。有**干净底图**时再用回归 ——
    那时底是真值，A、色调都能解出来。
    """
    c0 = first + gap + 2
    c1 = min(first + BAR_WIDTH - 9, c0 + _TRACK_WINDOW_COLUMNS, region.shape[1])
    if c1 - c0 < _TRACK_MIN_COLUMNS:
        return "short"
    if row + _TRACK_CONTROL_ROWS[1] - 1 + _TRACK_ROW_GAP >= region.shape[0]:
        return "short"
    band = [
        _track_ratio(region, row + r, row, c0, c1)
        for r in range(SKILL_ROW_OFFSET, SKILL_ROW_OFFSET + BAR_HEIGHT)
    ]
    controls = [
        _track_ratio(region, row + r, row, c0, c1)
        for r in range(_TRACK_CONTROL_ROWS[0], _TRACK_CONTROL_ROWS[1])
    ]
    if not all(np.isfinite(value) for value in band + controls):
        return "unclear"
    if max(abs(value) for value in controls) > _TRACK_CONTROL_MAX:
        # **自控不过 ⇒ 判据弃权**（用户裁定）：+18…+22 那五个数说的是"这一格的地形竖着
        # 变太快、借底在这个窗上不成立" —— 那是**这件仪器在这块地形上不适用**，
        # 不是"断口之后那一截不是轨道"。所以这里不否决、**照旧给值**。
        # （用户点名的两块亮底真轨道就落在这里：一块的对照 0.45、一块 0.51。）
        return "track"
    return "covered" if abs(float(np.median(band))) < _TRACK_VETO else "track"


def _track_verdict_at(
    region: np.ndarray, row: int, first: int, strict: int, tolerant: int
) -> str:
    """`_track_verdict` 的**按需入口**：填充画到条右端（没有断口）时直接给 "short"。

    **为什么要有这个入口**（用户裁定，探针 r113）：这一问的落点只在**颜色认得出族**的块上
    —— 颜色认不出族时，读数在**更早的支**就定了（走的是横向那套
    `_looks_like_an_empty_track`），这一问的返回值**没人读**。早先的写法是
    「只要有断口就一律先算」，于是难例批 `r01_01_05` 那种块会留下一个"算出来是 track"
    的值 —— 而它链上读的是**未采样**（082/084 那一次就是这么把它误当成"那一块的结论"的）。

    所以：**用到了才算，没用到就不产生这个值。**
    """
    if tolerant >= BAR_WIDTH:
        return "short"
    return _track_verdict(region, row, first, strict)


OPERATOR_UI_WIDTH: Final[int] = 240
OPERATOR_UI_HEIGHT: Final[int] = 360
"""**取区**的宽高。

它装得下三样：血条左上角在取区里的 (40, 320)，血条 146×8 到 (186, 328)；
菱形中心在 (112, 87)，64×64 的模板落在 (80..144, 55..119)。上下各留 40 px 以上，
左右各留 40 px 以上 —— 留给调用方的锚点误差。
"""

BARS_IN_WINDOW: Final[tuple[int, int]] = (40, 320)
"""**血条左上角**在取区里的位置（取区坐标）。

于是"取区该摆在哪"这件事由调用方一减就得到：取区原点 = 血条左上角 − (40, 320)。
**投影给出的是干员所在格，不是血条左上角** —— 那一次换算（格 → 血条左上角）
住投影里，不在这一层。
"""

DIAMOND_IN_WINDOW: Final[tuple[int, int]] = (112, 87)
"""**菱形的中心**在取区里的位置 = (40 + 72, 320 − 233)。

那个 (72, −233) 是**实测的**：五批不同来源的帧、七个干员上，菱形模板的最高分都落在
血条左上角 +(72, −232.5) —— 换算到取区就是 (112, 87.5)，这里取整成 (112, 87)。
"""

ROW_TOL: Final[int] = 12
COL_TOL: Final[int] = 12
"""**宽容窗口**：条的位置允许差多少行 / 多少列（±px）。

取区的原点是记录推出来的，一定带像素级的误差。行方向在应然位置上下各 12 行里挑
"最像的那一条"，列方向在应然左缘左右各 12 列里找"真正的左缘"。

**为什么是 12**：一格的屏幕尺寸远大于 12 px（cd 那一批同一个干员的条跨 61 帧
**一个像素都没动**），所以 12 只吸收**像素级**偏移；**整格错位吸收不了** ——
那正是《视觉层/场上我方干员》§4 给位置自检留的那条边界。
"""

MIN_FILL_COLUMNS: Final[int] = 4
"""少于这几列填着，就算**看不到血条**（落未采样）。

**这是一条保守的线**：血条常驻，所以"看不到"只能是识别失败 —— 落未采样绝不会产假事实。
代价是**极低填充那一档读不出来**（约 3% 以下），那一档这一批**没有覆盖到**，
如实写在 tests/帧样本/场上干员/README.md 的「没覆盖到什么」里。
"""

GAP_TOLERANCE: Final[int] = 4
"""数"自左端起连续填着"时，允许夹几列**连续**不填的。

**它专门为弹药技那条分格橙条留的**：实测分格的竖线占 **3–4 列**（cd01 的
y=476 那根：150 150 … 76 76 76 149 …，每 25 列掉一次）。没有这个容错，
亮区比例会在**第一道分格线上就断掉**（读成 22/146 而不是 129/146）。

**取 4 不取更大**：容错同时会让"边后面那一列"的杂散亮点把 n 推后。分格线是 3–4 列，
取 4 正好盖住它、又不给真正的空轨道留缝（空轨道是一整条暗的，远超 4 列）。
"""

COVER_GAP_TOLERANCE: Final[int] = 10
"""**被压住**那一档的缝宽容度：压在条上的东西（**别的条** —— 用户给的图层事实：
**特效与干员贴图都在条下面**，能挡住条的只有别的条与黄色菱形）把填充色断掉几列时，
容许多宽的缝**跨过去接着算**。**它只管技力条**，`GAP_TOLERANCE` 一个字不动。

**为什么要另开一条**：`GAP_TOLERANCE = 4` 是给**分格橙条**的竖线定的（实测 3–4 列），
弹药那一档正靠它工作；而 `shot04_a` 上那条缝是 **5 / 7 / 6 / 6 / 8 / 6 / 8** 列
（逐列距离在那些列上跳到 90 / 108 / 109 / 101 / 118）—— 那是**另一根条**压在它上面。
把 `GAP_TOLERANCE` 抬到 8 会**顺手改掉弹药那一档的读数**（它和"竖线 3–4 列"这个事实绑着），
所以两条缝要分开。

**10 取在量出来的空档里**：被压住那一档实测**最宽 8 列**（上面那一排），**轨道**那一档
实测**最窄 19 列**（`sp_series/s02` 那条绿条：第 127 列到第 145 列**一列都不是填充色**）。
10 落在 8 与 19 之间，取的是**靠保守那一侧** —— 两者代价不对称：取大了会把**轨道**算进填充
（那正是这一层刚修掉的那个现象），取小了只是"缝没跨过去、读数落成一条更弱的下界"。
所以宁小不大。

**它是"跨过去接着算"，不是"把缝里那几列当填着"**：跨过去之后，填充的**终点**仍然由
**填充色在缝后面出现到哪一列**决定（见 read_skill 那一支）。
"""


GREEN_MAX_RG: Final[int] = 0
ORANGE_MIN_RG: Final[int] = 40
"""**色相判定**的两条线（逐列中位数颜色的 R − G）。

- R − G ≤ 0 → **绿**（充能中）；实测绿条那一簇是 **−47 … −5**；
- R − G ≥ 40 → **橙**（生效中）；实测橙条那一簇是 **+65 … +77**（分格的暗线降到 38–40）；
- 中间那一段（也就是**正的那一段**）→ **认不出**（落未知，不挑一个）。

**这条线原来是 −8，是错的。** 原来那句「实测绿条那一簇是 −36 … −1」与实测对不上：
绿条**不止到 −1**，这一层实测到 **−5**（`sp_series/s04`：整条 32 列 `R − G = −5`），
别处还有 −3 的。−8 落在**绿条自己那一簇的里面**，于是货真价实的绿条被判成
**认不出** —— 而《状态层/我方干员》那条纪律是「绿条就是充能中」，把绿条读成未知
是**丢事实**（字段表允许未知，不等于可以把绿读成未知）。

**两侧的余量都指得出出处**（三批 34 帧上逐帧量的，量的是**填着那一截**的中位 `R − G`）：

- **绿那一侧**：绿条那一簇到 **−5** 为止 —— 出处是**回归批 `s04`**（整条 32 列
  `R − G = −5`；规划者那边同一条量到 −4，差在参考列怎么取）。0 到那一簇是 **5 格**。
- **橙那一侧**：橙条那一簇从 **+65** 起 —— 出处是**第一批 `cd01_c` / `cd12_a` / `shot26_a`**
  三条（都是 +65；同一批的 `cd01_b` 是 +71、回归批 `s00` 是 +77）。0 到那一簇是 **65 格**。

**但这 65 格里一列样本都没有。** 绿的到 −5 为止、橙的从 +65 起 —— 中间那一段（连
「认不出」那一档 `0 … 40` 的整段）**从来没有被样本压过**：那个"空"**不是量出来的，
是没有样本**。

**这条线的风险记在这里（别当成新现象）**：绿的边界是**一条帧**定的（回归批 `s04`），
橙那一侧**从没被样本压过**。**以后哪个批里冒出 `R − G` 落在 **(0, 65)** 的一根条，
先回头查这一条线**，别急着说"新的颜色档出现了"。
"""

SEPARATOR_DROP: Final[float] = 0.8
"""一条条里，"这一列比整条的中位亮度暗到几成"算**分格的竖线**。

**取 0.8 而不是 0.5**：实测分格线掉到中位的 **0.44–0.51 倍**（150 → 66..76），
而**空轨道**掉得更狠。0.8 落在"分格线肯定算、抗锯齿的边不算"之间 —— 它只用来**判
有没有分格**（弹药那一档），不用来数格子（《视觉层/场上我方干员》§2：不数格子）。
"""

SKILL_FILL_TOL: Final[int] = 48
"""技力条里"这一列填没填着"：这一列的**逐通道中位颜色**离**这一条自己的填充色**的
逐通道最大差 ≤ 这个数（与血条那条 `_blue_distance` 同一个统计量，只是参考色不是
写死的常量、而是从这一条上量出来的）。

**判据是"颜色像不像填充色"，不是"亮不亮"。** 原来那条是「亮度 ≥ 这一条自己逐列
中位亮度 × 0.55」，它在**亮底**上会把整条判成填满：轨道是**半透明**的，亮地板 /
亮花丛透出来的那一截在亮度上够得上那条线；而**中位**又是由轨道自己撑起来的
（填得少的时候轨道占多数），于是"≥ 0.55 × 自己"对轨道也成立。

    帧（原生，取区 = 血条左上角 − BARS_IN_WINDOW）  自左起实际填着  旧判据    新判据
    sp_series/s00（橙条纹地板）                        11 / 146     1.0       11
    sp_series/s01（花丛）                              76 / 146     0.6507↓   76
    sp_series/s02、s03（青地板）                      127 / 146     1.0      127
    sp_series/s04（橙条纹地板）                        32 / 146     1.0       32
    另有一条同形状的（青地板）                93 / 146（实绿）  0.788     93

（↓ = 报的是**下界**、而且比实际的还高 —— 反向的假事实。）这几帧的**轨道**在亮度上是
填充的 **0.51–0.92 倍**，0.55 那条线一条都拦不住；而颜色上，轨道离填充色的逐通道
最大差是 **87 / 93 / 94 / 106 / 149**，填充自己沿条的漂移最多 **40**。

**48 取在量出来的空档里**：填充自己的漂移 ≤ **40**（sp_series/s02 那条从
(131,183,85) 漂到 (140,163,55)），最近的一个非填充（sp_series/s01 那条轨道，
(121,126,139) 对填充 (157,178,71)）是 **93**。48 离两边各留一截。

**它盖住弹药那条分格橙条**：分格竖线是**暗线**，离填充色 **106–123**（cd01_c 那条
橙条上的竖线），落在 48 外面 → 算"不填"，再由 `GAP_TOLERANCE` 把 3–4 列的竖线跨过去
（不跨过去，亮区比例会在第一道分格线上断掉）。

**它的失效方向**（两个，都记着）：

1. 轨道上**正好**贴着填充出现一段与填充同色的东西会被算成填着 —— 与原来那条判据同类
   （把轨道算进填充），只是把"亮"换成了"同色"。要跨过 `GAP_TOLERANCE`（4 列）才算，
   而"被压住"那一档的跨缝上限另有一条（`COVER_GAP_TOLERANCE`）。实测第一批 16 帧 +
   真值 13 帧 + 回归 5 帧共 34 帧上没出现。
2. **参考色可能不是这一条条的填充色**（见 `_skill_fill_reference`）—— 按用户给的图层事实，
   那意味着这一格**不是在读一根条**（取区里有好几根条、或者条只截了一部分）；
   那时整条读数只会偏成**一条下界**。**真帧上这一档现在一帧都没有。**

**它管不到"填充被盖住"**：别的条压在条上是**另一种颜色**，按这条判据就是
"不填"，于是那一截被切断 —— 那一支由 `COVER_GAP_TOLERANCE` 决定是"跨过去接着算"
还是落**下界**（见 read_skill），不是精确值。
"""

DIAMOND_SEARCH_RADIUS: Final[int] = 60
"""菱形在**邻域**里搜：应然中心周围 ±这么多像素。

模板 64×64 在这个 120×120 的邻域里滑，于是它**不是**"钉在格心上方的小框里分类"
（《视觉层/场上我方干员》§3）。实测菱形的位置在 7 个干员上偏差 ≤ 1 px，所以这 ±60
里绝大部分是余量 —— 余量的用途是**图标自己在动**（暂停态下它会动）与锚点误差。
"""

DIAMOND_SOLID_SCORE: Final[float] = 0.45
DIAMOND_ABSENT_SCORE: Final[float] = 0.34
"""菱形的两条线：≥ 0.45 算**在**、≤ 0.34 算**不在**、中间落**未采样**。

**两条线都是从空档里取的，而且空档是量出来的**（五个来源的帧上）：

| 簇 | 分数 | 个数 |
| --- | --- | --- |
| **在** | **0.543 … 1.000** | 12 个干员 |
| **不在** | **0.087 … 0.280** | 25 个干员 |

空档是 **0.280 … 0.543**；两条线把这个空档切成三段，中间那一段（0.34 … 0.45）
就是**"认不出"**。它**绝不落 Exact(False)** —— 可手动触发这个字段只有布尔 / 未采样
两种形态，而把"认不出"读成"不在"正是《识别方法》「存在性搜索」那行标了 ⚠ 的
**漏检方向**（说没有其实有）。
"""

DIAMOND_ICON_FILENAME: Final[str] = "diamond.npz"
"""菱形模板的文件名（与字形表的 digits.npz 同一个做法）。"""


def _assets_path(feature: str) -> Path:
    """一个特征资产的默认落点：assets/features/<feature>/diamond.npz。

    src/ark/engine/vision/ 往上四级就是仓库根 —— 与 glyphs._assets_path 算的是同一个根。
    """
    return (
        Path(__file__).resolve().parents[4]
        / "assets"
        / "features"
        / feature
        / DIAMOND_ICON_FILENAME
    )


READY_DIAMOND_PATH: Final[Path] = _assets_path("ready_diamond")
"""可释放菱形那个**图标模板**的出厂落点（进版本库，clone 下来即用）。

来源与裁法写在资产自己的 source / box 字段里（load_ready_diamond 会给回来）——
**不许取自验收批的帧**（《识别方法》§已定 那条纪律）。
"""


# ---------------------------------------------------------------------------
# 取区
# ---------------------------------------------------------------------------


def operator_ui_window(bars: PixelRect) -> PixelRect:
    """由**血条那块矩形**折出取区：取区原点 = 血条左上角 − BARS_IN_WINDOW。

    它是给调用方用的一个方便式子，**不是**"这一层自己会去猜干员在哪"：血条那块矩形
    仍然是调用方给的（来自部署记录 + 投影）。**取区怎么开在这一层只有这一份实现**，
    省得调用方各抄一遍偏移。
    """
    if not isinstance(bars, PixelRect):
        raise TypeError(f"要的是血条那块矩形，收到 {type(bars).__name__}")
    return PixelRect(
        float(bars.x) - BARS_IN_WINDOW[0],
        float(bars.y) - BARS_IN_WINDOW[1],
        OPERATOR_UI_WIDTH,
        OPERATOR_UI_HEIGHT,
    )


def _region(frame: Frame, window: PixelRect) -> tuple[np.ndarray, int, int]:
    """取区的像素 + 它在画布上的原点。

    **取区越出画布时报错**：那多半是锚点算错了，而"把越出的那一块裁到边上"会静默
    把三样元素挪到别处 —— 与"裁剪要落在画布内"（[基准集] §2）是同一条。
    """
    if not isinstance(frame, Frame):
        raise TypeError(f"要读的是一帧，收到 {type(frame).__name__}")
    if not isinstance(window, PixelRect):
        raise TypeError(f"取区要是像素矩形，收到 {type(window).__name__}")
    x0 = int(round(float(window.x)))
    y0 = int(round(float(window.y)))
    width = int(round(float(window.width)))
    height = int(round(float(window.height)))
    if x0 < 0 or y0 < 0 or x0 + width > frame.canvas.width or y0 + height > frame.canvas.height:
        raise ValueError(
            f"取区 {window} 落在画布 {frame.canvas.width}×{frame.canvas.height} 之外："
            "锚点多半算错了，而裁到边上会把三样元素静默挪走"
        )
    if width < OPERATOR_UI_WIDTH or height < OPERATOR_UI_HEIGHT:
        raise ValueError(
            f"取区 {window} 比 {OPERATOR_UI_WIDTH}×{OPERATOR_UI_HEIGHT} 还小："
            "它装不下血条与菱形两样，而少了哪一样都是静默的"
        )
    return frame.pixels[y0 : y0 + height, x0 : x0 + width], x0, y0


# ---------------------------------------------------------------------------
# 条
# ---------------------------------------------------------------------------


def _column_profile(region: np.ndarray, row: int, start: int, width: int) -> np.ndarray:
    """一横条逐列的**中位颜色**：(列数, 3) 的 int16。

    中位数而不是均值：条上常常压着别的条，均值会被它拽走，
    中位数只要求"八行里多数是条自己"。
    """
    band = region[row : row + BAR_HEIGHT, start : start + width]
    return np.median(band.astype(np.int16), axis=0)


def _blue_distance(profile: np.ndarray) -> np.ndarray:
    """逐列离血条蓝多远（逐通道最大差）。"""
    return np.abs(profile - np.asarray(HP_BLUE, dtype=np.int16)).max(axis=1)


_SKILL_HALF_LIT_CEILING: Final[int] = 56
"""`_half_lit` 那条"半填"带子的上沿：离填充色的逐通道最大差落在
`SKILL_FILL_TOL`（48）与这个数之间的那一列算**半填**。

**带子为什么这么窄**：实测**干净的边界是一步跳过去的** —— 填着那一侧最多差 **40**，
紧接着那一列就跳到 **68 / 87 / 88 / 93 / 94 / 106 / 149**。带子只能取在
**40 与 68** 中间那一段；56 取在它靠中间的地方。

**先试过 48 的两倍（96）**：那会把 68/87/88/93 这几种**误判成"半填"**，于是本该精确的
读数被摊成区间 —— sp_series 那三条本来是 11 / 76 / 32 列，被摊成 [11,24] / [76,93] /
[32,36]。半填说的是"**边界落在这一列里面**"（这一列是填充与轨道的混色），不是
"这一列不像填充"，所以带子只能贴着那条线。

**代价**：这么窄的带子在真帧上基本不触发 —— 区间那一支靠**合成用例**钉
（与血量那两支同一个处境，写在 README 的「没覆盖到什么」里）。
"""

_SKILL_FILL_REF_SKIP: Final[int] = 2
_SKILL_FILL_REF_COLUMNS: Final[int] = 8
"""取"填充色"那点样子时**跳过开头几列、再取几列**。

**跳 2 列**：条最左边那一两列常带抗锯齿的半像素，取进去会把参考色拉偏。
**取 8 列**：够中位数稳住，又短到"填充只有 11 列"时（`sp_series/s00`）整段都还在
填充里 —— 填充是**自左端起**的（见本模块开头那张表），所以最左那几列**按定义**就是
填充色，不需要先知道比例再回头取参考。
"""


def _skill_fill_reference(span: np.ndarray) -> np.ndarray | None:
    """这一条自己的**填充色**：自左缘起跳过抗锯齿那两列、取 8 列的逐列中位颜色。

    条太短、连这 8 列都取不满时给 None —— 那说明这一条根本没读到东西。

    **参考取的是"条最左那 8 列"，前提是那 8 列真的是这一条。** 用户给的图层事实：
    **特效与干员贴图都在条下面** —— 能挡住条的只有**别的条**（比如怪的）与**黄色菱形**。
    所以"参考色不是填充色"**不可能是特效染出来的**，它只可能是
    **这一格压根不是在读一根条**（格里有好几根条、或者条只截了一部分）。

    碰到"参考色绿橙两族都不是"的帧，**先查取区里到底有几根条**；
    那种帧的读数按用户对"很难判准"那一档的要求，应当报**未采样**。
    """
    hi = min(_SKILL_FILL_REF_SKIP + _SKILL_FILL_REF_COLUMNS, span.shape[0])
    if hi <= _SKILL_FILL_REF_SKIP:
        return None
    return np.median(span[_SKILL_FILL_REF_SKIP : hi].astype(np.int16), axis=0)


def _skill_fill_flags(span: np.ndarray) -> np.ndarray:
    """一列算不算"技力条填着"：**它是不是这一条自己的填充色**。

    判据是"这一列的逐通道中位颜色离**这一条自己的填充色**的逐通道最大差 ≤
    `SKILL_FILL_TOL`"。**为什么是相对这一条自己、而不是一个写死的颜色**：条右边的
    背景实测能比条本身亮一倍（shot30：条 145–157、右边背景 164–213），而填充色本身
    还随底下的场景漂（sp_series 那批四条绿条分别是 (157,178,71) / (131,183,85) /
    (171,176,53)）；写死一个颜色两边都会错。

    **为什么不是"比这一条自己亮"**：轨道是半透明的，亮底上透出来的那一截在亮度上
    够得上填充；而中位数由轨道自己撑起来（填得少的时候轨道占多数），于是
    "≥ 0.55 × 自己"对轨道也成立 —— 整条判成填满。详见 `SKILL_FILL_TOL`。

    **它同时盖住弹药那条分格橙条**：分格竖线是暗线，离填充色 106–123，落在
    `SKILL_FILL_TOL` 外面，于是它们被算作"不填"—— 再由 `GAP_TOLERANCE` 把 3–4 列的
    竖线跨过去（不跨过去，亮区比例会在第一道分格线上断掉）。
    """
    if span.size == 0:
        return np.zeros(0, dtype=bool)
    reference = _skill_fill_reference(span)
    if reference is None:
        return np.zeros(span.shape[0], dtype=bool)
    distances = np.abs(span.astype(np.int16) - reference).max(axis=1)
    return distances <= SKILL_FILL_TOL


def _leading_run(flags: np.ndarray, gap: int) -> tuple[int, int]:
    """自下标 0 起"填着的那一截"到哪儿为止 —— 返回 **(不留余量的, 夹了 gap 个假的)**。

    两个数各有各的用处，**不许合成一个**：

    - **第一个（strict）**：一格缝都不跨。它是"**至少这么多**"那个下界的依据 ——
      缝底下藏着什么，谁也看不见，所以只能说"缝之前那一截是填着的"；
    - **第二个（tolerant）**：允许夹 gap 个连续假。它是**比例**的依据 —— 弹药条那列
      分格竖线占 3–4 列，不跨过去比例会在第一道分格线上断成 22/146。

    两个数不一样就说明**中间断过**：那是"被盖住 / 被切碎"，也就是《识别方法》
    「亮区比例」那行的 ⚠（**单边**：比例读低）。"
    """
    size = flags.size
    strict = 0
    while strict < size and flags[strict]:
        strict += 1
    tolerant = strict
    miss = 0
    index = strict
    while index < size:
        if flags[index]:
            tolerant = index + 1
            miss = 0
        else:
            miss += 1
            if miss > gap:
                break
        index += 1
    return strict, tolerant


def _right_run(lit: np.ndarray) -> tuple[int, int]:
    """条**最右那一截**"填着"的列：[起点, 终点) —— 自最后一列往回、连续为真的一段。

    最右那一截就是"**填充画到条的右端了吗**"这句话要问的地方（见 `read_skill` 那几支）。
    整条都亮时它就是整条；条右边压根没亮（右边是空轨道）时它是空的。
    """
    end = int(lit.size)
    while end > 0 and not lit[end - 1]:
        end -= 1
    start = end
    while start > 0 and lit[start - 1]:
        start -= 1
    return start, end


def _reach(flags: np.ndarray) -> int:
    """这一段里**最远**的一个亮列的下标 + 1（不管中间断了没有）。"""
    lit = np.flatnonzero(flags)
    return int(lit[-1]) + 1 if lit.size else 0


def _scan_span(region: np.ndarray) -> tuple[int, int]:
    """横条那一行的**扫区**：(起点列, 宽)。

    起点 = 应然左缘 − COL_TOL、宽 = BAR_WIDTH + 2 × COL_TOL：**两头都留一截**，
    这样取区沿 x 摆到 ±COL_TOL 都还在扫区里。

    右边多留的那一截**不会**被数进填充：填着的那一截是从"找到的左缘"起、**截到
    BAR_WIDTH 为止**（见 read_health / read_skill 里那个切片）。所以条右边的背景再亮
    （实测 shot30 那条：条 145–157、右边背景 164–213）也进不来。
    """
    start = max(0, BARS_IN_WINDOW[0] - COL_TOL)
    width = min(BAR_WIDTH + 2 * COL_TOL, region.shape[1] - start)
    return start, width


def _locate_health(region: np.ndarray) -> tuple[int, int, np.ndarray] | None:
    """在取区里找**血条**：行方向挑最像的那一行，列方向留一截容错。

    返回 (行号, 扫区起点列, 那一行的逐列颜色) —— 找不到给 None。

    判据是"**这一行里有多少列是血条蓝**"。**行方向在应然位置上下各 ROW_TOL 行里挑**
    （宽容窗口），挑中的那一行是 argmax，所以锚点差两三个像素不会改变结论。

    **技力条不另开一次定位**：它与血条是**同一条 UI 的两截**（实测两截的左缘一个像素
    都不差），所以它的行就是 `这一行 + SKILL_ROW_OFFSET`、左缘就是这一条的左缘。
    另开一次定位反而会锁到别的东西上 —— 实测条右边的背景常常比条本身还亮，
    用"哪一行最亮"去挑会挑到背景上。
    """
    start, width = _scan_span(region)
    if width <= 0:
        return None
    _, expected_row = BARS_IN_WINDOW
    best: tuple[int, int, np.ndarray] | None = None
    best_count = -1
    for row in range(
        max(0, expected_row - ROW_TOL),
        min(region.shape[0] - BAR_HEIGHT, expected_row + ROW_TOL) + 1,
    ):
        profile = _column_profile(region, row, start, width)
        count = int((_blue_distance(profile) <= HP_COLOUR_TOL).sum())
        closer = best is None or abs(row - expected_row) < abs(best[0] - expected_row)
        if count > best_count or (count == best_count and closer):
            best_count = count
            best = (row, start, profile)
    if best is None or best_count < MIN_FILL_COLUMNS:
        return None
    return best


def _left_edge(flags: np.ndarray, start: int) -> int:
    """这一行里**填充从哪一列开始** —— 找不到给 −1。

    候选只在应然左缘的 ±`COL_TOL` 列里挑（再远说明那不是这根条 —— 可能是旁边
    那个干员的，或者锚点整格错了）。挑哪一列，判据是：

    > **从左缘起、往后 `BAR_WIDTH` 那一整段里，亮列最多的那个起点。**

    另外两种看起来更自然的判据都不行，而且各错在一边：

    - **取"最靠应然位置"的那一列**：取区一偏右，应然那一列就落进填充**内部**了，于是
      左缘被读在填充里面，比例**静默偏短**；
    - **取最左边那一列**：条左边那一截背景实测也能有几列是蓝的（shot06 那条：条左边
      6 列落在线内、中间隔 6 列、然后才是条），于是那 6 列被当成填充的头，比例**偏长**。

    "整段里亮列最多"这个判据两边都躲开了：背景那几列只带 6 列亮，条那一整段带 146 列；
    而"填着的那一截中间被切断"时，两个碎片合起来仍然比别的起点多，所以切断不会把左缘
    推到后面去。

    **候选只认"亮段的起点"（轴 I）**，而且**扫区最左那一列还多一道门**：从它起必须能连出
    至少 BAR_WIDTH − COL_TOL 列 —— 见下面那段注释（左邻的尾巴 vs 锚点往右挪了 12 px）。
    """
    expected = BARS_IN_WINDOW[0] - start
    low = max(0, expected - COL_TOL)
    high = min(flags.size - 1, expected + COL_TOL)
    if high < low:
        return -1
    # **判据不是"哪一列最靠应然位置"，也不是"哪一列最靠左"**：
    # - 取"最靠应然位置"会在取区偏右时把左缘读进填充**内部**（比例静默偏短）；
    # - 取"最靠左"会把条左边那一截背景的 Blue 当成填充的头（shot06 那条：条左边
    #   6 列落在线内、中间隔 6 列、然后才是条）。
    #
    # 所以判据是：**从左缘起、往后 BAR_WIDTH 那一整段里亮列最多的那个起点**。
    # 背景那几列只带 6 列亮，条那一整段带 146 列 —— 一比就分开了；而"填着的那一截
    # 中间被切断"时两个碎片合起来仍然比别的起点多，所以切断不会把左缘推到后面去。
    # **候选只认"亮段的起点"**（axis I）：左邻那一根条会探进这一格 —— 相邻两条的左缘
    # 只差 138 列、而条宽 146，所以左邻的尾巴会盖住本条的 +0…+7 列。只按"亮列最多"挑，
    # 候选会落到**左邻那根条**身上（它的 146 窗口里亮列一样多，甚至更多）。真条分的
    # 亮段起点只有一个：**这一列亮、它左邻那一列不亮**。
    best = -1
    best_count = -1
    for index in range(low, high + 1):
        if not flags[index]:
            continue
        if index > 0 and flags[index - 1]:
            continue
        if index == 0 and start > 0:
            # **扫区最左那一列亮着**时它的起点看不见 —— 有两种形状要分开：
            # ① **左邻的尾巴**（真左缘在 12 那一列，尾巴从画面左面伸进来）；
            # ② **取区被锚点往右挪了**（COL_TOL 以内），真左缘正好落在扫区最左列。
            # 只看"亮不亮"分不开这两样；**亮段有多长**分得开：左邻的尾巴最多伸进
            # (条宽 146 − 间距 + COL_TOL 12) 列 —— 实测间距 138/139，也就是最多 20 列。
            # 所以要求「从这一列起能连出至少 BAR_WIDTH − COL_TOL = 134 列」才认它是左缘：
            # 锚点偏 +12 px 的满条/接近满条救得回来，左邻的尾巴仍然不认
            # （§5 那条「左邻顶掉存在性」照旧拦着）。
            if _leading_run(flags, GAP_TOLERANCE)[0] < BAR_WIDTH - COL_TOL:
                continue
        count = int(flags[index : index + BAR_WIDTH].sum())
        if count > best_count:
            best_count = count
            best = index
    return best


_TRACK_DROP_MIN: Final[float] = 0.01
_TRACK_DROP_MAX: Final[float] = 0.51
"""「条内比条外暗」这件事的**下沿与上沿**（`_looks_like_an_empty_track` 那条判据的两头）。

空的技力条是**半透明**的：它把底下的画压暗一点点，所以"条内比条外暗"这件事
**有上下两头** —— 太弱就成了没落差，太狠就不是半透明该有的样子。

- **下沿 0.01**：落差要看得出来。真帧上空的技力条（难例批 `r01_06_04`）两端的落差是
  **0.159 / 0.300**；而"边缘压根没对上条"的那两块（`shot30_a` 沿 x 挪 −24 / −16）
  是 **0.0011 / 0.0006**（几乎是零）。0.01 取在这两簇的**几何中项**上
  （√(0.159 × 0.0006) ≈ 0.010）：离噪声 9–17 倍，离空条 16 倍。
- **上沿 0.51**：半透明的条**压不了那么狠**。被一个近黑的盒子压住时，条内比条外
  暗到 **0.860**（难例批 `r01_01_05` 的左缘）。0.51 同样取几何中项
  （√(0.300 × 0.860) ≈ 0.508）：离空条 1.7 倍，离被压住那一块 1.7 倍。

**两头都要**：只有下沿会把 `r01_01_05` 认成空条（它暗得**过头**了）；只有上沿会把
"边缘没对上条"那两块认成空条。**这不是绝对亮度线** —— 量的是"条内 ÷ 条外"这一个比值，
两头的常数也是从"两簇测出来的数之间取中项"来的，不是拍的。

**血条那一条也用它**（_empty_health_track）：血条一个蓝像素都没有时，判「这一格是不是
一根空轨道」用的就是这一对上下沿 —— 只是量的是**上沿**那一侧（条上面那块地形当底）。
实测正样本落在 **+0.30…+0.44**。
"""


def _track_drop(lum: np.ndarray, inside: slice, outside: slice) -> float | None:
    """**条内那一段比条外那一段暗多少**：（外 − 内）÷ 外。取不到给 None。

    `lum` 是一整行的**逐列亮度**（逐列中位色再按通道平均，与 `_has_separator` 同一个口径）。
    外面那一段越出取区时给 **None** —— 那是"这一端没有证据"，**不许**当成"没落差"。
    """
    if inside.start < 0 or outside.start < 0:
        return None
    if inside.stop > lum.size or outside.stop > lum.size:
        return None
    outer = float(np.median(lum[outside]))
    if outer <= 0:
        return None
    return (outer - float(np.median(lum[inside]))) / outer


def _edge_drops(skill: np.ndarray, first: int) -> tuple[float | None, float | None]:
    """技力条**两条长边**各量一次条内/条外：(左缘的落差, 右缘的落差)。

    内 / 外 都是 `_SKILL_FILL_REF_COLUMNS`（8）列宽的窗口，**紧贴边缘的
    `_SKILL_FILL_REF_SKIP`（2）列两头都跳掉** —— 用户提醒过：抗锯齿下边缘那一两列是
    **混色**的，算进"内"或"外"都会把结论带偏（跳 2 列这条口径本来就是为它定的，
    见那条常量的文档）。

    `first` 是**条在 `skill` 里的下标**（不是取区列号）—— 与 `_left_edge` 的返回值同源。

    **这里的"外"是横向的**（同一行的左右两侧），与"条下面"无关。若以后谁要加一条
    **纵向**的条内/条外落差：条下面还拖着一道**阴影**，逐行实测（run-02 八帧，
    活帧 ÷ 干净帧 的 A）**+18 0.74…0.88 / +19 0.87…0.93 / +20 0.92…0.97 /
    +21 0.96…1.00 / +22 0.98…1.03**，**+23 起 1.00**（残差 sd 0.0…1.1）——
    纵向的"外"**要么退到 +23 之后，要么按 _TRACK_SHADOW 把阴影除回去**（取 +18 又不修正
    会被那道阴影稀释）；纵向那条判据（_TRACK_ROW_GAP / _track_ratio）走的是
    **恒定间隔 + 阴影修正**这一条路。

    **拿这道阴影当"底变了吗"的试纸时，逐行容差取 ±0.10**（不是 ±0.15：按 0.15 放，
    `S5-2-4` 那种"条底下又多盖了一层半透明"会被吞进去 —— 它 +18 量到 0.889，
    而干净帧那一档是 0.74…0.88）。**这是个"先按它走"的数**，不是量到底的数：
    以后若发现该拦的没拦住、或好帧被冤枉，就动这一条 —— 它只影响"理由写被挡了
    还是写底变了"，两边都落未采样。
    """
    lum = skill.mean(axis=1)
    skip = _SKILL_FILL_REF_SKIP
    span = _SKILL_FILL_REF_COLUMNS
    left = _track_drop(
        lum,
        slice(first + skip, first + skip + span),
        slice(first - skip - span, first - skip),
    )
    right = _track_drop(
        lum,
        slice(first + BAR_WIDTH - skip - span, first + BAR_WIDTH - skip),
        slice(first + BAR_WIDTH + skip, first + BAR_WIDTH + skip + span),
    )
    return left, right


def _looks_like_an_empty_track(skill: np.ndarray, first: int) -> bool:
    """技力条那一行**是不是一根空的轨道**（而不是被东西压住）。

    用户给的画面事实（《视觉层/场上我方干员》§1）：**空的**条会在那一格留下一条
    **条形变暗区** → **条的内外亮度有落差**；**被完全压住的**没有那道落差。
    条的位置与形状是**已知**的（紧贴在血条下面、横向范围与血条一样，而血条不可能为空），
    所以两端的边缘能**精准标出**。

    判据（**两端的比值，不是亮度绝对值**）：

    > **两条长边各自量一次「条内比条外暗了多少」；
    > 两次都要落在 `_TRACK_DROP_MIN` … `_TRACK_DROP_MAX` 之间。**

    **为什么两头都要，而且只有"有落差"不够**（真帧上的三块，见难例批）：

    - `r01_06_04`（空轨道）：左 **0.159** / 右 **0.300** —— 落在带子里 ✓；
    - `r01_01_05`（被一个近黑的盒子压住）：左 **0.860** —— 落差**过头**了，上沿挡住；
    - `r01_01_06`（被压住）：右 **−0.155** —— 条内比条外还**亮**，下沿挡住；
    - `shot30_a` 沿 x 挪 −24 / −16（边缘压根没对上条）：右缘 **0.0011 / 0.0006** —— 几乎
      是零，下沿挡住。

    **一端量不出来**（那一侧的窗口越出取区）时给 **False** —— 那是"没有证据"，
    与"没有落差"一样落未采样（保守那一侧）。
    """
    left, right = _edge_drops(skill, first)
    for drop in (left, right):
        if drop is None or not _TRACK_DROP_MIN <= drop <= _TRACK_DROP_MAX:
            return False
    return True


def _fill_confidence(distances: np.ndarray, filled: int) -> float:
    """这条比例读数有多稳：**定下这次读数的那个判据两侧隔了多远** ÷ 255。

    定下这次读数的是两个判断 —— **填着的那一截里最弱的一列**（离那条线最近的，
    距离最大）与**右边那一列**（它必须干脆地在线外）。余量取这两者里小的那个；
    再除以 255（通道的上限）归一。与 cost._fill_confidence 同一个口径。
    """
    if filled <= 0:
        return 0.0
    # 填着的那一截里**最弱的一环**（离那条线最近的一列）—— 这就是"定下这次读数"的余量。
    inside = float(HP_COLOUR_TOL - distances[:filled].max())
    if filled >= distances.size:
        # 满条：右边那一列**不在条上**（看它是多的那一列），所以"到底了"这件事是
        # 由 BAR_WIDTH 那个常量给的，不是由像素给的 —— 余量只剩填着那一截自己的。
        return min(1.0, max(0.0, inside / 255.0))
    outside = float(distances[filled] - HP_COLOUR_TOL)
    return min(1.0, max(0.0, min(inside, outside) / 255.0))


def _empty_health_track(region: np.ndarray, row: int, first: int) -> bool:
    """血条**一个蓝像素都没有**时：那一行**是不是一根空的半透明轨道**（**上沿**那一侧量）。

    血条紧贴技力条**上方**，所以它那一侧只剩**上面**能用（下面被技力条自己占着）——
    量的方向与技力条那一支相反：内 = 血条那 8 行（+0…+7），外 = **往上**第
    _TRACK_ROW_GAP 行的地形。

    返回 True ⇒ 那是**这一格自己的空轨道**（人还在、生命极低）→ 读 **0**；
    False ⇒ 量不出落差 / 落差过头 / 反过来 → 这一帧**没有证据** → 未采样。

    **为什么只当"是不是空轨道"用、不拿它给任何比例**：条**上方**那块地形跟条背后
    那一块不是一块地形。实测 31 块没填满的血条（填充右边那一截就是血条自己的轨道）：
    整条都空的那些块量到 **+0.30…+0.44**（与技力轨道那 0.35 一档对得上），
    而**一半以上**的块量成负数（−0.25…−3.4）—— 所以这一支只回答一个二值问题，
    落错方向也只丢读数、不产假值。

    门槛用**同一对上下沿**（_TRACK_DROP_MIN…_TRACK_DROP_MAX）—— 那就是
    《视觉层/场上我方干员》§1 说的「半透明该有的那一段」。

    first 是**记下来的左缘**（BARS_IN_WINDOW[0] − 扫区起点）：这一支的前提正是
    「填充一个像素都没有」，左缘只能由记录给（§6.0 规矩 1）。
    """
    c0 = first + 2
    c1 = min(first + BAR_WIDTH - 9, c0 + _TRACK_WINDOW_COLUMNS, region.shape[1])
    if c1 - c0 < _TRACK_MIN_COLUMNS or row - _TRACK_ROW_GAP < 0:
        return False
    band = [
        _track_ratio(region, row + r, row, c0, c1, gap=-_TRACK_ROW_GAP)
        for r in range(BAR_HEIGHT)
    ]
    if not all(np.isfinite(value) for value in band):
        return False
    return _TRACK_DROP_MIN <= float(np.median(band)) <= _TRACK_DROP_MAX


def read_health(frame: Frame, window: PixelRect) -> SlotValue:
    """读**血条比例**：自左端起填着的那一截占整长（146）的多少。

    落法（这一层**不产 Unknown** —— 血量那个字段的形态里没有它）：

    | 情形 | 落法 |
    | --- | --- |
    | 找得到条、填满 | Valued(Exact(1.0), 余量) |
    | 找得到条、边界干净 | Valued(Exact(比例), 余量) |
    | 找得到条、**填着的那一截被缝切断过**（缝两边都亮着） | Valued(LowerBound(下沿), 余量) |
    | 找得到条、**边界是渐变的**（右边那一截半亮半不亮） | Valued(Interval(下沿, 上沿), 余量) |
    | 找得到条、**右端被压暗**（半亮那一截一直顶到条的右端） | Valued(Exact(1.0), 余量) |
    | **一个蓝像素都没有、但量得出空轨道**（生命极低，人还在） | Valued(Exact(0.0), 余量) |
    | **一个蓝像素都没有、量不出空轨道** | Unsampled() |
    | **找不到条** | Unsampled() |

    五支的理由各自不同、**不许混**：

    - LowerBound 说的是"**至少这么多**" —— 缝底下藏着什么看不见，所以只能报"缝之前
      那一截"。**被别的条压住只会让比例读低（单边）**，所以下界正是那件事
      的诚实读数，也就是《识别方法》「亮区比例」那一行的天然失效模式；
    - Interval 说的是"那里有东西，我不知道边界落在哪一列"；
    - **满**（右端被压暗那一支）说的是"半亮那一截**一直顶到条的右端**（条外那一列
      已经不蓝了）" —— 那说明**没有边界落在条里面**，是整条都在、右端被压暗。
      **这一支不看"血条那一带对不对得上条"**：那条判据（按血条蓝的列数看带子齐不齐）
      已经**撤掉**了 —— 它在全批 86 块里走到这一支的 3 块（`r01_06_04` / `r01_05_04` /
      `r01_07_02`）上**一次都没对过**（其中两块的带子蓝列数只差 1–3 列，那是抗锯齿噪声，
      不是带子错位），而用户裁定这三块都是**满**。"这一行是不是空的轨道"那件事现在由
      `_looks_like_an_empty_track` 直接在技力条那一行上量（见 `read_skill`）；
    - Unsampled 说的是"**那里没有条**"。血条是常驻的，所以这一支**只能是识别失败**
      —— **绝不许读成"满血"**（《状态层/我方干员》§血量：看不到 ≠ 满血）；
    - **0**（新开的一支）说的是「**条在、只是空的**」：一个蓝像素都没有时再问一句
      「这一行是不是一根空的半透明轨道」（_empty_health_track，**上沿**那一侧量）。
      是 → 人还在、生命极低 → **0**；不是 → 回到 Unsampled。**死活这一层不下结论**。

    **下界取的是"一格缝都不跨"的那一截**（strict），不是夹了容错的那一截 —— 夹了容错
    的那个数在缝的**另一侧**，把它当"至少"就把看不见的那一段当成亮着的了。

    **取区越出画布 / 比最低要求还小 → 抛异常**（能力缺失，不是"看不到"）。
    """
    region, _, _ = _region(frame, window)
    found = _locate_health(region)
    if found is None:
        return Unsampled()
    row, start, profile = found
    distances = _blue_distance(profile)
    flags = distances <= HP_COLOUR_TOL
    first = _left_edge(flags, start)
    if first < 0:
        # **一个蓝像素都没有** —— 「看不到」与「看到且为空」要分开的就是这一处。
        # 左缘由记录给（§6.0 规矩 1），再问一句「这一行是不是**空轨道**」（**上沿**那一侧量）：
        #   是 → 条在、只是空的 → 人还在、生命极低 → **0**（用户裁定）；
        #   不是 → 这一帧没有证据（阵亡那一格、或者压根不是这根条）→ 未采样。
        # **死活这一层不下结论**（《视觉层/场上我方干员》§6.1 最后一行）。
        if _empty_health_track(region, row, BARS_IN_WINDOW[0] - start):
            return Valued(Exact(0.0), None)
        return Unsampled()
    # 条只有 BAR_WIDTH 那么长：**右边一列都不多留**。条右边的背景实测能比条本身亮
    # （shot30 那条的右边背景 164–213），多留就会把它数进填充里。
    # 条只有 BAR_WIDTH 那么长，但**多看右边那一列**：满条时"边界"就是条的右端，
    # 那一列是唯一能说"这里到底了"的证据（少了它，满条的余量永远是 0）。
    flags = flags[first : first + BAR_WIDTH + 1]
    distances = distances[first : first + BAR_WIDTH + 1]
    strict, tolerant = _leading_run(flags, GAP_TOLERANCE)
    tolerant = min(tolerant, BAR_WIDTH)
    strict = min(strict, BAR_WIDTH)
    if tolerant < MIN_FILL_COLUMNS:
        return Unsampled()
    if tolerant >= BAR_WIDTH:
        return Valued(Exact(1.0), _fill_confidence(distances, tolerant))
    # 「被切断」：缝两边都亮着（跨过了缝，或者夹了余量停下之后还有亮列）→ 缝底下
    # 藏着什么看不见，所以只能说"**至少**缝之前那一截"。
    if strict < tolerant or _reach(flags) > tolerant:
        return Valued(
            LowerBound(strict / BAR_WIDTH), _fill_confidence(distances, tolerant)
        )
    # 「边界渐变」：右边那一列并没有干脆地跳开（半亮）→ 只能给区间。
    if tolerant < distances.size and distances[tolerant] < 3 * HP_COLOUR_TOL:
        blurred = tolerant
        while blurred < distances.size and distances[blurred] <= 2 * HP_COLOUR_TOL:
            blurred += 1
        upper = min(BAR_WIDTH, blurred)
        if upper > tolerant:
            if upper >= BAR_WIDTH:
                # 半亮的那一截**一直顶到条的右端**（条外那一列已经不蓝了）—— 那不是「边界
                # 落在条里面」，是**整条都在、右端被压暗**：`shot_20261007-203049-920` 的
                # `r01_06_04` 最后 6 列离血条蓝 25–29、条外那一列 186；难例批 `r01_05_04`
                # （24 / 25 / 26 / 27 / 27、条外 191）与 `r01_07_02`（32 / 36 / 40 / 41 / 41、
                # 条外 203）是**同一个形状**。
                #
                # **这一支不看"血条那一带对不对得上条"**：那条判据已经撤掉了（它在走到
                # 这一支的 3 块上一次都没对过 —— `r01_05_04` / `r01_07_02` 的带子蓝列数
                # 只差 1–3 列，那是抗锯齿的噪声，不是带子错位）。
                return Valued(Exact(1.0), _fill_confidence(distances, tolerant))
            return Valued(
                Interval(tolerant / BAR_WIDTH, upper / BAR_WIDTH),
                _fill_confidence(distances, tolerant),
            )
    return Valued(Exact(tolerant / BAR_WIDTH), _fill_confidence(distances, tolerant))


def _has_separator(span: np.ndarray) -> bool:
    """这一段里有没有**分格的竖线**：有哪一列比整段的中位亮度暗到 SEPARATOR_DROP。

    **不许数格子**（《视觉层/场上我方干员》§2）：这里只回答"有没有分格"。
    """
    if span.shape[0] < 3:
        return False
    luminance = span.mean(axis=1)
    base = float(np.median(luminance))
    if base <= 0:
        return False
    return bool((luminance < base * SEPARATOR_DROP).any()) and float(luminance.min()) > 8


def _meaning_of(profile: np.ndarray, filled: int) -> SlotValue:
    """技力条的**含义**：充能中 / 生效中 / 弹药分格。

    - R − G ≤ GREEN_MAX_RG（绿）→ Exact("充能中")；
    - R − G ≥ ORANGE_MIN_RG（橙）→ Exact("生效中")，而条里有分格竖线时改判
      **弹药分格** —— 判据是**形状**（有没有一列比整条中位暗到 SEPARATOR_DROP），
      不是"它像不像弹药"（与顶部敌数那个分隔符同一条纪律：形状自己带判据）；
    - 落在两条线**中间** → Valued(Unknown(), None)：**不许挑一个** —— 那是"颜色对不上
      任何一族"，而按用户给的图层事实（见 `_skill_fill_reference`），一根**条**的颜色只会落在
      绿或橙两族里；碰到它先查**这一格是不是在读一根条**（取区里是不是有好几根条）。
      这一档真帧上现在有了（难例批 `r01_01_05` / `r01_01_06` / `r01_06_04`）；**这个函数只负责**
      "颜色两族都不是"这件事，**怎么落**在 `read_skill` 里（空轨道 → 填充 0、含义就是本
      函数交出的 Unknown；被盖住 → 两个量都未采样）—— 不许拿别的帧去凑；
    - **「一列彩度都没有 → Unsampled()」这一支没实现**（现状，不许当它存在）：本函数里那一个
      `Unsampled()` 只在 `span` 为空时给，而 `read_skill` 保证 `filled ≥ MIN_FILL_COLUMNS`，
      所以**从链上到不了**。真帧上也没有"一列彩度都没有"的技力条 —— 要真做，得先有量出来的
      彩度出处，不许顺手加一条没出处的线。
    """
    span = profile[:filled]
    if span.size == 0:
        return Unsampled()
    rg = float(np.median(span[:, 0] - span[:, 1]))
    if rg <= GREEN_MAX_RG:
        return Valued(Exact("充能中"), min(1.0, (GREEN_MAX_RG - rg) / 255.0))
    if rg >= ORANGE_MIN_RG:
        margin = min(1.0, (rg - ORANGE_MIN_RG) / 255.0)
        if _has_separator(span):
            return Valued(Exact("弹药分格"), margin)
        return Valued(Exact("生效中"), margin)
    return Valued(Unknown(), None)


def _hue_family(meaning: SlotValue) -> str | None:
    """含义 -> **颜色族**（"green" / "orange"）；含义**认不出**时给 None。

    与 `_meaning_of` 用**同一把尺子**（`GREEN_MAX_RG` / `ORANGE_MIN_RG`）："族"就是那两条
    线把那一段正数分开之后的两段，这里**不新开阈值**。
    """
    if isinstance(meaning, Valued) and isinstance(meaning.payload, Exact):
        if meaning.payload.value == "充能中":
            return "green"
        if meaning.payload.value in ("生效中", "弹药分格"):
            return "orange"
    return None


def _right_end_is_fill(span: np.ndarray, lit: np.ndarray, meaning: SlotValue) -> bool:
    """条**最右那一截**里，有没有一列真的是**本条填充色所在族**的颜色。

    这是"这一条**满了**"那句话的**前件**（`read_skill` 里那两支：满 = 填充一直画到条的
    右端）。原来只看"最右那一列离参考色 <= `SKILL_FILL_TOL`" —— **那一条挡不住别的东西
    压在最右那一截上**：

    - 难例批 `r01_04_03`（源帧 `shot_20261007-202959-919.png`）那条绿条：最右 **8 列**
      （第 138…145 列）离参考色 **30-44**，而这条**自己的填充体**（第 0…137 列）**<= 17**
      —— 按距离算"像填充"，按**色族**算**两族都不是**（逐列中位 `R - G` 是 **+28 … +31**，
      正落在 `GREEN_MAX_RG … ORANGE_MIN_RG` 那一档里）。那 8 列占了最右那一截的**全部**，
      于是"填充画到右端"这句话**没有证据**了 —— 技能按钮上的 当前/上限 也证实它**不是满**。

    **含义认不出时不替它判**（与 `sp_of` 同一条纪律）：那时给 True，落法一个字不变。
    """
    family = _hue_family(meaning)
    if family is None:
        return True
    start, end = _right_run(lit)
    if end <= start:
        return False
    rg = span[start:end, 0] - span[start:end, 1]
    if family == "green":
        return bool((rg <= GREEN_MAX_RG).any())
    return bool((rg >= ORANGE_MIN_RG).any())


def read_skill(frame: Frame, window: PixelRect) -> tuple[SlotValue, SlotValue]:
    """读**技力条**：(填充比例, 含义) 两个量。

    **比例**（自左端起**填着**的那一截 / 146）的落法。什么叫"填着"见
    `SKILL_FILL_TOL`：这一列的颜色像不像**这一条自己的填充色**（不是"亮不亮"——
    轨道是半透明的，亮底上"亮"判不出填没填）：

    - 填满 → Valued(Exact(1.0), None) —— **但"满"要自证**：最右那一截里得看得到
      **本条填充色族的颜色**（`_right_end_is_fill`）。尾巴上全是外来颜色时
      **不许默认读满** —— 宁可报一条下界（难例批 `r01_04_03` 就是这个形状）；
    - 含义**认不出**（颜色两族都不是）→ 再问**这一行是不是一根空的轨道**
      （`_looks_like_an_empty_track`：两条长边各自量"条内比条外暗了多少"，两端都要有
      落差、而且落差要落在半透明该有的那一段里）：
      是 → 那是**这一条自己的空轨道**（半透明，底下的东西透出来）→ 填充
      **Valued(Exact(0.0), None)**、含义**照实交 `Valued(Unknown(), None)`**（认得出有
      东西、只是读不准 —— **不是**未采样）；不是 → 有东西压在 UI 上 → 两个量都 Unsampled()；
    - **弹药分格**且没填满 → Valued(LowerBound(整格那一截 / 146), None)：分格把
      "最后一格是半格还是空格"抹平了（半格剩的那一截与空格一样暗），精确值给不出来；
    - **被压住**（压住的东西是另一种颜色，缝两边都还是填充色）：缝 ≤
      `COVER_GAP_TOLERANCE` 列 → **跨过去接着算**（跨过去之后**最右那一截**里看得到
      本条填充色 → 满条；最右那一截里一列都不是 → `LowerBound(缝之前那一截)`；
      否则给 `LowerBound(填充色在缝后面出现到哪一列)`）；缝比它还宽 → `LowerBound(缝之前那一截)`；
    - 边界**渐变** → Valued(Interval(下沿, 上沿), None)；
    - 干净 → Valued(Exact(比例), None)；
    - **找不到条** → 两个量都 Unsampled()。

    **弹药那一档只报亮区比例**，总格数不在这里乘 —— 那是数据库的事（《视觉层/
    场上我方干员》§2：总格数是已知量，这里读的只是比例）。而**分界模糊时给下界**
    （状态层那张表：弹药剩余 精确 / 区间 / **下界**，分界模糊 → 下界，不猜精确）。

    **含义**见 _meaning_of。两个量**各自带自己的三态**。

    **方向**：《视觉层/场上我方干员》§1 —— 绿条充能中（比例 = 充了多少），橙条生效中
    （比例 = **还剩多久**）。**方向相反**这件事在这一层只是"含义"里那句话，比例本身
    仍然照实报（它是同一个统计量）；**怎么用由调用方按「含义」决定**。
    """
    region, _, _ = _region(frame, window)
    found = _locate_health(region)
    if found is None:
        return Unsampled(), Unsampled()
    row, start, profile = found
    flags = _blue_distance(profile) <= HP_COLOUR_TOL
    first = _left_edge(flags, start)
    if first < 0:
        return Unsampled(), Unsampled()
    skill = _column_profile(region, row + SKILL_ROW_OFFSET, start, profile.shape[0])
    span = np.ascontiguousarray(skill[first : first + BAR_WIDTH])
    if span.shape[0] < MIN_FILL_COLUMNS:
        return Unsampled(), Unsampled()
    lit = _skill_fill_flags(span)
    strict, tolerant = _leading_run(lit, GAP_TOLERANCE)
    tolerant = min(tolerant, BAR_WIDTH)
    strict = min(strict, BAR_WIDTH)
    # **被压住那一档**用的是另一条缝宽容度（COVER_GAP_TOLERANCE，不是 GAP_TOLERANCE）：
    # 分格竖线 3–4 列那一条是弹药那一档的命根子，shot04_a 上那条缝是 5–8 列。
    _, covered = _leading_run(lit, COVER_GAP_TOLERANCE)
    covered = min(covered, BAR_WIDTH)
    # **断口之后的自证**：只在"填充没画到条右端"时有意义，而且**用到的那两处才算**
    # （`_track_verdict_at`）—— 颜色认不出族、填充画满、被压住跨过去那几支都在它前面
    # 就定下了读数，提前算出来的那个值对它们没有落点（用户裁定，见 `_track_verdict_at`）。
    if tolerant < MIN_FILL_COLUMNS:
        return Unsampled(), Unsampled()
    meaning = _meaning_of(span, tolerant)
    if isinstance(meaning, Valued) and isinstance(meaning.payload, Unknown):
        # **认不出**（颜色两族都不是）：按《视觉层/场上我方干员》§1 本来该报未采样。
        # 再问一句"这一行**是不是一根空的轨道**"（`_looks_like_an_empty_track` —— 它量的
        # 是"条内比条外暗了多少"，两端都要有落差、而且落差要落在半透明该有的那一段里）：
        #   是 → 那是**这一条自己的空轨道**（半透明，底下的东西透出来，所以颜色两族
        #        都不是），里面没东西 → 填充 **0**；
        #   不是 → 有东西压在 UI 上（详情界面那种图标）→ 这一行不是这根条 → 未采样。
        if _looks_like_an_empty_track(skill, first):
            # 含义**照实交出去**（Valued(Unknown)），**不许拿未采样顶**：「看不到」与
            # 「看得到、只是空的」是两件事（用户裁定：技力未知与技力为 0 是两种行为）。
            # 那一族 = 状态层「技能是否生效中 / 剩余时长」形态表里的**未知**
            # （精确 / 下界 / 未知）—— 就是《状态层/我方干员》§容易吞掉的语义边界
            # 那条「认得出有东西、但读不准」。难例批 `r01_06_04` 就是这一块。
            return Valued(Exact(0.0), None), meaning
        return Unsampled(), Unsampled()
    ends_in_fill = _right_end_is_fill(span, lit, meaning)
    if tolerant >= BAR_WIDTH and ends_in_fill:
        return Valued(Exact(1.0), None), meaning
    # 弹药那一档：填充没到整长时**只给下界** —— 分格把"最后一格是半格还是空格"这件事
    # 抹平了（半格剩下的那一截与空格一样暗），所以精确值给不出来，只能报"至少到上一条
    # 分格线"（状态层那张表：弹药剩余 精确 / 区间 / **下界**）。
    if _is_ammo(meaning):
        ammo = Valued(LowerBound(_whole_segments(span, lit, tolerant) / BAR_WIDTH), None)
        if _track_verdict_at(region, row, first, strict, tolerant) in ("covered", "unclear"):
            # 断口之后**不是轨道**（或被盖住）：填充就算到断口也只能算下界，
            # 而"还剩多少"这句话没有证据 → 含义照实交未采样。
            return ammo, Unsampled()
        return ammo, meaning
    if strict < tolerant or _reach(lit) > tolerant:
        # **被压住**（缝两边都还是填充色）：压在条上的东西是另一种颜色，所以缝里那几列
        # 不算填着 —— 但**缝后面接着填**说明只是被压住了一小截。用更宽的缝宽容度
        # （COVER_GAP_TOLERANCE）跨过去再看一眼。
        if covered >= BAR_WIDTH:
            if ends_in_fill:
                # 跨过去之后**最右那一截**里看得到这一条自己的填充色。填充是**自左端起**的，
                # 所以那一列只能是同一条填充的一部分 —— 这一条是满的，这不是"猜"。
                return Valued(Exact(1.0), None), meaning
            # 最右那一截里**一列都不是**本条填充色族的颜色（例如被另一根条压住）：
            # "填充画到右端"这句话就没有证据了 —— 只能报到**缝之前**那一截。
            return Valued(LowerBound(strict / BAR_WIDTH), None), meaning
        if covered > tolerant:
            # 缝后面确实是同一个填充色，一直填到 covered 那一列 —— "至少这么多"。
            return Valued(LowerBound(covered / BAR_WIDTH), None), meaning
        # 缝比 COVER_GAP_TOLERANCE 还宽（或者后面压根没有填充色）：只能说"缝之前那一截"。
        return Valued(LowerBound(strict / BAR_WIDTH), None), meaning
    if tolerant < lit.size and _half_lit(span, tolerant):
        blurred = tolerant
        while blurred < lit.size and _half_lit(span, blurred):
            blurred += 1
        return (
            Valued(Interval(tolerant / BAR_WIDTH, min(BAR_WIDTH, blurred) / BAR_WIDTH), None),
            meaning,
        )
    if _track_verdict_at(region, row, first, strict, tolerant) in ("covered", "unclear"):
        # **断口之后那一截自证不是轨道**（有东西压在 UI 上 / 压根不是条）：填充到底停在
        # 哪一列就没有证据了 —— 比例只能报到**断口**（strict 那一列），含义交未采样。
        # 「未采样」这一档的**证据必须来自技力条那一段自己的画面**（用户裁定）：
        # 难例批 r01_01_06 走的就是这一支 —— 断口在 46 列，之后那 60 列是 (16,15,15) 的黑，
        # 行带（+8…+15）对往下第十行的比值中位 **0.111 < 0.17** ⇒ 那一截不是半透明轨道。
        # 它的自控（+18…+22，阴影已除回）是 **0.12** —— **自控是过的**，定案的是否决线。
        return Valued(LowerBound(strict / BAR_WIDTH), None), Unsampled()
    return Valued(Exact(tolerant / BAR_WIDTH), None), meaning


def _is_ammo(meaning: SlotValue) -> bool:
    """这个含义是不是弹药分格。"""
    return (
        isinstance(meaning, Valued)
        and isinstance(meaning.payload, Exact)
        and meaning.payload.value == "弹药分格"
    )


def _half_lit(span: np.ndarray, index: int) -> bool:
    """第 index 列是不是"半填"：离填充色差着一截，但也没走远。

    它是"边界渐变"那一支的判据 —— 半填说明边界落在这一列**里面**，于是只能给区间。

    落在 **`SKILL_FILL_TOL`（48）与 `_SKILL_HALF_LIT_CEILING`（56）之间**的那一档算半填。
    原来这一档是按亮度量的（0.55 倍再乘 0.55），现在与"填没填"那一侧**共用同一把尺子**
    （离填充色的逐通道最大差），两支才对得上：一根条里"填着 / 半填 / 不填"三档应该是
    同一个量上的三段。

    **带子为什么这么窄**（真帧上这一支基本不触发）：见 `_SKILL_HALF_LIT_CEILING`。
    """
    if index < 0 or index >= span.shape[0]:
        return False
    reference = _skill_fill_reference(span)
    if reference is None:
        return False
    distance = float(np.abs(span[index].astype(np.int16) - reference).max())
    return _SKILL_HALF_LIT_CEILING > distance > SKILL_FILL_TOL


def _whole_segments(span: np.ndarray, lit: np.ndarray, filled: int) -> int:
    """弹药条里**整格填满**的那一截有多长 —— 上一条分格线之前那一段。

    "整格"在这里不是"数格子"（《视觉层/场上我方干员》§2 明说不数格子），而是
    "自左端起、到最后一个**不亮**的列为止"：分格线本身就不亮，所以这条线正好落在
    格与格之间。找不到这样的列（一路亮到 filled）就给 0 —— 那说明这一截是从头亮
    起来的，下界只能是 0，那是保守的那一侧。
    """
    if filled <= 1:
        return 0
    dark = np.flatnonzero(~lit[:filled])
    if dark.size == 0:
        return 0
    return int(dark[-1])


def sp_of(fill: SlotValue, meaning: SlotValue) -> SlotValue:
    """**技力那个字段**的读数：这是一道**形态闸门**，不只是含义那一支。

    《状态层/我方干员》§技力 那条硬规则：场上只有绿条（充能中）能读技力；**技能生效
    期间场上没有技力条可读** → 未采样，或由轴数据算。所以这个函数把它写成一个地方，
    省得每个调用点各抄一遍、抄漏一处就是"编一个比例"。

    两件事，**顺序反了就会漏**：

    1. **含义 = 生效中 / 弹药分格 → 未采样**（技能生效期间没有技力条可读）；
    2. **读数里的「下界」加宽成「区间」**。下界是**弹药剩余**那一行的形态，
       **技力那个字段的形态只有 精确 / 区间 / 未采样**（《状态层/我方干员》那张字段表）。
       下界说的是"**至少这么多**"（值落在 [下沿, 1] 里），所以加宽成
       `Interval(下沿, 1.0)` 是**同一个事实的另一种写法**，一个字都没有多、也没有少。
       **不许原样透出去** —— 那等于替那个字段加了一个它没收的形态。
    3. **含义未采样 → 未采样**（这一支是**新判据**开出来的路）。技力条那一段自己的画面
       自证「断口之后那一截不是轨道」时，比例只报到断口、含义交未采样。那时**含义不知道
       是什么**，比例也就不能当技力交出去 —— 技力只能从绿条来，不知道含义就落未采样。

    **为什么不落未采样**：那时那里确实有东西、下沿也读到了，落未采样就是把"读不准"
    伪装成"没看到"（《状态层/我方干员》§容易吞掉的语义边界）。

    含义**认不出**（Valued(Unknown)）时**不替它判**：读数自己是什么形态就交什么形态。
    **这一支现在到得了了**（上一版到不了 —— 那时 `read_skill` 把"认不出"一律落成未采样）：
    难例批 `r01_06_04` 就是它 —— 含义认不出 + 填充精确 0 → sp **精确 0**；这正是用户裁的
    「技力未知与技力为 0 是两个完全不同的行为」落在这一层的地方。

    **只有下界需要加宽**：fill 来自 read_skill，它只产 精确 / 区间 / 下界 / 未采样
    （那四种写在它的文档里），所以上界 / 未知这两支到不了这里。
    """
    if isinstance(meaning, Valued) and isinstance(meaning.payload, Exact):
        if meaning.payload.value in ("生效中", "弹药分格"):
            return Unsampled()
    if isinstance(meaning, Unsampled):
        # **含义不知道是什么时，比例不许当技力交出去。** 这一支由**新判据**开出来：
        # 技力条那一段自己的画面自证「断口之后那一截不是轨道」时，比例报到断口、
        # 含义交未采样；若在这里把那条下界加宽成区间，就等于拿一个**不知道含义**的比例
        # 替技力编一个值（用户裁定的是「未采样」，不是「区间」）。
        return Unsampled()
    if isinstance(fill, Valued) and isinstance(fill.payload, LowerBound):
        return Valued(Interval(fill.payload.lo, 1.0), fill.confidence)
    return fill


# ---------------------------------------------------------------------------
# 可释放菱形
# ---------------------------------------------------------------------------


def load_ready_diamond(path: Path | str = READY_DIAMOND_PATH) -> np.ndarray:
    """读出厂的可释放菱形模板 → (高, 宽, 3) 的 uint8 RGB。

    **读不出来就抛**：这是能力缺失，静默给一个空模板等于"这张卡没有倒计时"
    那种假缺席（《资产生成器/特征库》§3 那一行的"人工框一次"）。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"可释放菱形的模板不在 {path}：它是识别链的前提资产，缺了要响亮地报出来，"
            "不许静默降级成一个「没有菱形」"
        )
    with np.load(path) as data:
        if "template" not in data:
            raise KeyError(f"{path} 里没有 template 这一项：找到 {list(data)}")
        template = np.array(data["template"])
    if template.dtype != np.uint8 or template.ndim != 3 or template.shape[2] != 3:
        raise TypeError(
            f"{path} 里的模板要是 (高, 宽, 3) 的 uint8 RGB，"
            f"收到 {template.dtype} 的 {template.shape}"
        )
    return template


def save_ready_diamond(
    template: np.ndarray,
    path: Path | str = READY_DIAMOND_PATH,
    *,
    source: str,
    box: tuple[int, int, int, int],
) -> Path:
    """把模板写成一个资产：模板本身 + **它从哪来**。

    source 是那一帧的仓库相对路径、box 是它在那一帧上的 (x, y, 宽, 高) ——
    来源必须跟着资产走，否则"这个模板是不是取自验收帧"就无从回答
    （《视觉层/识别方法》§已定 那条纪律）。
    """
    if not isinstance(template, np.ndarray) or template.dtype != np.uint8:
        raise TypeError("模板要是 uint8 的 ndarray")
    if template.ndim != 3 or template.shape[2] != 3:
        raise TypeError(f"模板要是 (高, 宽, 3) 的 RGB，收到 {template.shape}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        template=template,
        source=np.asarray(source),
        box=np.asarray(box, dtype=np.int64),
    )
    return path


def _search_box(window: PixelRect, x0: int, y0: int) -> PixelRect:
    """菱形那个**邻域**在画布上的矩形：应然中心 ± DIAMOND_SEARCH_RADIUS。"""
    centre_x = x0 + DIAMOND_IN_WINDOW[0]
    centre_y = y0 + DIAMOND_IN_WINDOW[1]
    radius = DIAMOND_SEARCH_RADIUS
    return PixelRect(centre_x - radius, centre_y - radius, 2 * radius, 2 * radius)


def _guard_template(template: object) -> np.ndarray:
    """模板的形状闸门。**空模板 / 尺寸不对一律抛** —— 那是能力缺失。"""
    if not isinstance(template, np.ndarray):
        raise TypeError(f"模板要是 ndarray，收到 {type(template).__name__}")
    if template.dtype != np.uint8 or template.ndim != 3 or template.shape[2] != 3:
        raise TypeError(
            f"模板要是 (高, 宽, 3) 的 uint8 RGB，收到 {template.dtype} 的 "
            f"{getattr(template, 'shape', template)}"
        )
    if min(template.shape[:2]) < 8:
        raise ValueError(f"模板太小了：{template.shape[:2]}，那是拿一块噪声当图标")
    return template


def diamond_score(
    frame: Frame, window: PixelRect, template: np.ndarray
) -> tuple[float, int, int]:
    """在取区的**菱形邻域**里搜模板 → (最高分, 中心 x, 中心 y)。

    **这是判据本身**（find_ready_diamond 用的就是它），所以它是公开的 ——
    与 roster_scores 同一个理由：要让别人拿**同一条链的分数**去核，
    就不能让他另写一份匹配（另写一份，量的就是另一条链了）。

    分数是 cv2.matchTemplate(..., TM_CCOEFF_NORMED)（零均值归一化相关，RGB 三通道），
    与备战席那条链是**同一个统计量**。位置**是滑出来的**，不是钉死的。

    **邻域越出取区 → 抛异常**：那是取区给错了。
    """
    _guard_template(template)
    region, x0, y0 = _region(frame, window)
    box = _search_box(window, x0, y0)
    bx = int(round(float(box.x))) - x0
    by = int(round(float(box.y))) - y0
    width = int(round(float(box.width)))
    height = int(round(float(box.height)))
    if bx < 0 or by < 0 or bx + width > region.shape[1] or by + height > region.shape[0]:
        raise ValueError(
            f"菱形邻域 {box} 越出取区 {window}：取区容不下 "
            f"±{DIAMOND_SEARCH_RADIUS} 的搜索邻域，常量与取区对不上了"
        )
    if template.shape[0] > height or template.shape[1] > width:
        raise ValueError(f"模板 {template.shape[:2]} 比搜索邻域 {height}×{width} 还大")
    area = region[by : by + height, bx : bx + width]
    result = cv2.matchTemplate(
        np.ascontiguousarray(area),
        np.ascontiguousarray(template),
        cv2.TM_CCOEFF_NORMED,
    )
    _, best, _, location = cv2.minMaxLoc(result)
    half_h, half_w = template.shape[0] // 2, template.shape[1] // 2
    return (
        float(best),
        x0 + bx + int(location[0]) + half_w,
        y0 + by + int(location[1]) + half_h,
    )


def find_ready_diamond(
    frame: Frame, window: PixelRect, template: np.ndarray
) -> SlotValue:
    """搜那个黄菱形**在不在**（可手动释放的那个图标）。

    - 最高分 ≥ DIAMOND_SOLID_SCORE → Valued(Exact(True), 最高分)；
    - ≤ DIAMOND_ABSENT_SCORE → Valued(Exact(False), 最高分)；
    - **中间那一段 → Unsampled()** —— 那是"认不出"。

    **中间那一段为什么不是 Exact(False)**：可手动触发这个字段只有**布尔 / 未采样**
    两种形态（[状态层/我方干员]），而"认不出"被读成"不在"正是《识别方法》
    「存在性搜索」那一行标了 ⚠ 的**漏检方向**（图标被盖住 → 说"没有"其实有）。
    所以这一支落未采样，**它和"不在"在类型上分得开**。

    **它不是"钉在格心上方的小框里分类"**：位置是在取区的邻域里**滑出来的**，
    取区本身由调用方从部署记录开（这一层不产生它）。
    """
    score, _, _ = diamond_score(frame, window, template)
    if score >= DIAMOND_SOLID_SCORE:
        return Valued(Exact(True), score)
    if score <= DIAMOND_ABSENT_SCORE:
        return Valued(Exact(False), score)
    return Unsampled()
