
"""敌人血条：**整张战场上一次筛完**，逐根报填充比例。

《视觉层/敌方单位》管这一层。它只回答一件事：**这一帧画面上看得见几根敌人血条、
各自填了多少**。

**它不做的两件事**：「哪一根条属于哪一只」（归属）与「那一只还剩多少血」
（逐个体血量）—— 那两件都等「候选区 / 位置」那条前端。

## 用途：反查我们自己的伤害账

伤害涉及的因素多，叠加索敌机制之后更容易漏，所以「账对不对」只能靠画面反查：
账说没漏，而画面说这只还剩一点血时，AI 还来得及补一刀。

## 语义：血条显示 当且仅当 未满血

所以「看不到条」= **满血** —— 这是**读出来**的，不是推断，不需要「敌人不回血」
这个前提；反过来**读不出**「它到底受没受过伤」（回满之后与从未受伤一模一样）。

## 读法：按色相 + 形状一次筛完，不做滑窗模板匹配

为一条纯色水平长条把整张战场滑一遍没有道理。三层：

1. **色相**：逐像素离 ENEMY_FILL 的逐通道最大差 <= SCREEN_TOL 的那些像素；
2. **形状**：只留**竖着连成 8 行**的像素（血条高 8 行，见 BAR_ROWS）；再按连通块取
   宽 >= MIN_FILL_COLUMNS、高 <= MAX_BAR_ROWS 的块；
3. **逐列比色**（与我方血条那条链**同一个口径**，只是参考色换了）：块里那一 8 行
   逐列取中位色，离 ENEMY_FILL 多远 —— 这就是这一块的颜色分。

**逐列中位而不是均值**：条上常常压着别的东西，中位数只要求「八行里多数是条自己」。

## 三个出口（与可释放菱形那条链同款）

| 情形 | 落法 |
| --- | --- |
| 分数（= 1 - 颜色距 / 255）>= SOLID_SCORE | Valued(Exact(True), 分数) |
| 每一块都 <= ABSENT_SCORE（含一块都没有） | Valued(Exact(False), 分数) |
| 有块、但都落在中间带 | **Unsampled()** |

**中间那一带绝不并进「不在」**：把「认不出」读成「不在」正是《识别方法》
「存在性搜索」那一行标了警告的**漏检方向**（条被盖住 -> 说「没有」其实有）。
所以这一支落未采样，**它和「不在」在类型上分得开**。

## 我方那两根条要排除

它们的位置是**我们自己算的**（部署记录 + 投影折出来的血条矩形），调用方从
`our_bars` 传进来。排除的判据是**几何重合**、不是颜色：敌人条可以**压在我方条
上面**（raw_a 里那根敌条就盖掉了一条我方血条的右端 27 列），所以「按颜色筛掉
蓝条」是不够的，得按位置认。

## 两根参考色：橙族与**红族**

条有两族颜色：**橙**（小怪那一档，`ENEMY_FILL`）与**红**
（**只有 boss**，`ENEMY_FILL_RED` —— 用户裁定：红条的敌人只有 boss 那一档）。
红族**做窄的**，不是「所有偏红的都收」—— 所以它除了那一支参考色，还多一道
**宽度门** `RED_MIN_COLUMNS`：boss 是**宽**的（实测 116 列），而红族最像条的
假阳性（`boss.png` 那一帧红门上的红色碎块）只有 18…32 列。

## 形状门：候选必须是**矩形 + 内部均匀**

色相只回答「这一块的颜色像不像条」，答不了「这一块是不是条」—— 地形美术里
就有整块都是条色的。所以色相之后还有一道形状门，两条：

- **矩形**：**每一列都亮着**的那几行（条身）要装下整块 `RECT_FILL_MIN` 的像素
  —— 边界平、长宽比规整，跑出条身去的毛刺不算这一块（见 `RECT_FILL_MIN`）；
- **内部均匀**：那 8 行取**逐行**中位色，通道极差 <= `RECT_ROW_SPREAD_MAX`
  （条是平的；「颜色从上到下渐变」的地形木料过不去）。

**逐行而不是逐列**：条被别的东西压过一截时逐列中位色的极差能到 34（实测
`敌条互叠2` 那一帧两块），而逐行的极差真条最多 13 —— 逐列那一支的空档太窄，
当不了门（见 `RECT_ROW_SPREAD_MAX`）。**这道门只收紧、不放宽**：它收掉的正是
`raw_b` 那块 9 列、色距 8 的假阳性（它以前是被宽度门顺带挡掉的）。

## 宽度：进门松、出口严

`MIN_CANDIDATE_COLUMNS`（7）是**候选**的门：连不满 7 列的块连候选都不是 ——
7 是**现有最短的真帧**（`快死的敌人` 那一根填充只有 7 列）。`MIN_FILL_COLUMNS`
（12）是**够不够量比例**的门：填充段短于它的候选**照样报「在」**（条确实在），
但**比例落未采样** —— 短填充给不出「填了多少」这句话该有的证据。

**「在门以下」不是通往「不在」的一条路**：颜色说它是条就报「在」，颜色说不清
是不是条就落**未采样**，只有颜色说它**不是**条才是「不在」。

## 填充比例：**整根条** = 填充段 + 空轨，逐根量

与我方血条同源：**逐列比色 + 空轨判断**。区别只有一条 —— 我方血条的整长是常量
（146），而敌人条的整长**要看敌人**（实测小怪那一档量到 122…123，raw_b 那根精英
量到 `154），所以整长只能**量**：

- **填充段**：自左缘起连续算作填充的那些列（容 `TRACK_GAP` 列，与我方条同一支）；
- **条身**：**按量到的行数**，不写死 8（见 `_body_rows`）—— 小怪与红门那些整齐的块是
  8 行，boss 那根红条是 **10 行**，`raw_b` 上那根是 9 行；
- **空轨**：填充段右边那一段**与它上下那两块地形都不一样**的列 —— 有**偏暗**的
  （raw_a：带内 43 而上下 65），也有**偏亮**的（raw_b：一段纯灰 177 而上下 74）。
  所以判据取「上下两侧**都**差得够远」，不是「更暗」；量那两段地形时**与条身之间
  空 TRACK_EDGE_GAP 行**（条的上下边是软的，贴着的那一行还是条自己）；
  量到的那一列**自己就暗**（亮度 <= TRACK_DARK_LINE）时换一支：**逐通道**比、
  上下**任一侧**够远就算 —— 暗底那一族里轨色会与它一侧的地形长得一样，
  两侧都要会系统性答成「没有空轨」（见 TRACK_DARK_LINE）；
- **整长 = 填充段 + 空轨**。填满时没有空轨，整长就等于填充段 —— 实测
  `screenshots/_probe.png` 那根满条就是这样（123 列，右邻一列回到地形）。

**遮挡按「下界」降级**：填充段中间被切断（缝两边都还亮着）时，缝底下藏着什么
看不见，所以只能说「**至少**缝之前那一截」—— 与我方血条那条链的 LowerBound
同一套纪律，不另立一套。

**整长量得不干净时，比例也不许报「精确」**：量空轨的那一趟容得下跨过
`TRACK_GAP` 列回落（顶着一条半透明的边时那是常态），可**只要跨过过**，
那个「整长」就不是**连着**量出来的 —— 它可能多吃了一截地形，于是分母偏大、
比例偏小。这时给的是 **`LowerBound(填充 / 整长)`**（真整长只可能更短，真比例
只可能更大，所以量到的这一个正是**下界**），不是 `Exact`。**理由**：分母是
**量**出来的；量得不干净还说「精确」，就是把「不知道」写成了「知道」。
软尾那一支（边界落在末尾几列弱差里）给的是**区间**，信息比下界多，所以先走它。

## 条数：分不开时给**下界**，不给精确

叠在一起的条**不要求数清**（用户 10-14 裁定：**尽力而为，而不是一定要全识别
出来**）—— 那是**信息本身不在画面上**。所以这一层的输出是两件：`bars`（逐根，
每根带 `separated`）与 `count`。

**分不开的一堆只算 1 根**：`LowerBound(k)` 里的 `k` 只数确实互相分得开的那些，
所以这一支**永远不高报**；分得开才给 `Exact`。**把分不开的报成精确正是这一支
要挡的那件事**（条数报多了，下游会以为还有更多的敌人；报精确了，下游就没法
知道这个数只是个下限）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import cv2
import numpy as np

from ark.contracts import (
    Exact,
    Interval,
    LowerBound,
    PixelRect,
    SlotValue,
    Unsampled,
    Valued,
)
from ark.engine.vision.field import SKILL_ROW_OFFSET
from ark.engine.vision.frame import Frame

__all__ = [
    "read_enemy_bars",
    "screened_cores",
    "ScreenedCore",
    "EnemyBar",
    "EnemyBars",
    "ENEMY_FILL",
    "ENEMY_FILL_RED",
    "SCREEN_TOL",
    "RED_MIN_COLUMNS",
    "MIN_CANDIDATE_COLUMNS",
    "RECT_FILL_MIN",
    "RECT_ROW_SPREAD_MAX",
    "SOLID_TOL",
    "ABSENT_TOL",
    "SOLID_SCORE",
    "ABSENT_SCORE",
    "BAR_ROWS",
    "MAX_BAR_ROWS",
    "MIN_FILL_COLUMNS",
    "TRACK_CONTRAST",
    "TRACK_EDGE_GAP",
    "TRACK_DARK_LINE",
    "TRACK_GAP",
    "MAX_BAR_COLUMNS",
    "FIELD_TOP",
    "FIELD_BOTTOM",
    "ROW_SLACK",
    "COL_SLACK",
    "MERGE_GAP",
]


ENEMY_FILL: Final[tuple[int, int, int]] = (218, 106, 53)
"""敌人血条**填充段**的颜色。**这是量出来的**，不是给的常量。

原生帧上逐列取中位色，两根条（`raw_a` 的 y480 与 y517）都是
(218, 106, 53) / (217, 107, 53)；`raw_b` 那一根是 (216, 103, 54)；
全库 176 帧里筛出来的每一块「8 行高的橙块」，中位色都落在离它 1…12 以内。
**它与我方血条那支蓝 (110,190,228) 是两族颜色**，没有任何重合。
"""

ENEMY_FILL_RED: Final[tuple[int, int, int]] = (197, 41, 40)
"""**红族**（boss 那一档）的填充色。**红族只有这一档**。

用户裁定：**红条的敌人只有 boss 那一档**，所以红族不做成「一族宽的红」。
**它是量出来的**：`boss.png` 那根红宽条 116 列 × 10 行，逐列中位色就是
(197, 41, 40)、逐列色距 0…13；整帧里唯一另一块「像条的红」是**红门**那几块
地形美术（18…32 列）。

**它与我方那支蓝 (110,190,228)、与橙族 (218,106,53) 都分得开**：与橙族相距 65
（两侧半径 34 与 34 几乎不搭界；真落进重叠区的那几列由更近的那一族管）。
"""

RED_MIN_COLUMNS: Final[int] = 60
"""红族的宽度门：红块的连通块宽不过它就不当候选。

**这是「红族做窄的」那一步**：boss 是**宽**的（实测 116 列），而 `boss.png`
红门上那些红色碎块只有 18…32 列、`raw_b` 里那两块红色平片 26 与 58 列。
60 取在 32 与 116 之间（下留 28、上留 56）。

**代价**：一根**被压掉大半**的 boss 条（可见填充短于 60 列）会被漏掉 ——
这一档现在只有那一个真值样本，量不出这条代价有多大（见夹具 README 的缺口）。
"""

SCREEN_TOL: Final[int] = 34
"""逐像素筛的颜色门：离 ENEMY_FILL 的逐通道最大差 <= 它。

**这个数取在空档里**：全库 176 帧按 60 的松门扫一遍，把所有「8 行高的橙块」的
中位色距做成分布，得到两簇 ——

- **1…12**（56 块）：真条。8 行高、宽 16…130、中位色就是那个橙；
- **34…59**（几百块）：地形与贴图（红罩、危险图标、地貌边），从 34 起**密起来**；
- 中间 13…33：**稀**（21 块）—— 见 SOLID_TOL / ABSENT_TOL。

34 取在密堆的下沿：它上面是那条稀疏带，下面是那几百块。
"""

SOLID_TOL: Final[int] = 16
"""**够高 -> 在**：这一块的中位色距 <= 它就当它是条。

真条实测 1…12，所以线取 16 是**往「不在」那一侧留了 4**。
"""

ABSENT_TOL: Final[int] = 28
"""**够低 -> 不在**：这一块的中位色距 >= 它就不当它是条。

非条那一堆从 34 起密起来，所以线取 28 是**往「在」那一侧留了 6**，
同时也是**把 13…27 那一带整段留给「认不出」** —— 那一带里既有几块像是
阴影里的真条（197, 97, 53），也有贴图色（236, 83, 66），**这一层不替它们拍**。
"""

SOLID_SCORE: Final[float] = 1.0 - SOLID_TOL / 255.0
"""分数 = 1 - 颜色距 / 255 的那条上线（与 ABSENT_SCORE 一起组成中间带）。"""

ABSENT_SCORE: Final[float] = 1.0 - ABSENT_TOL / 255.0
"""分数 = 1 - 颜色距 / 255 的那条下线。"""

BAR_ROWS: Final[int] = 8
"""一根敌人条的高（行）。**实测**：全库筛出来的每一块都是 8 行。

（`raw_b` 那一带有两块是 9 行，那是抗锯齿多出来的一行，不是「一根更高的条」。）
"""

MAX_BAR_ROWS: Final[int] = 12
"""形状门的高上限：筛出来的块高过它就不当候选。

真条实测 8…9 行；高过 12 的那些是**贴图与地形**（`raw_b` 里那块红罩 121 行、
`cd01` 那个红色警戒框 150 行、`S5-0-0` 那根栏杆 25…81 行），离 12 很远。
"""

MIN_CANDIDATE_COLUMNS: Final[int] = 7
"""形状门的宽下限（**候选**那一档）：连不满 7 列的块连候选都不是。

7 是**现有的最短真帧**：`快死的敌人` 那一根填充只有 **7 列**（右边一长截空轨）。
它以前低于 `MIN_FILL_COLUMNS`，在 `_screen` 里就被丢掉 —— 连「认不出」都进不去，
于是那一帧报出**精确的「不在」**（假事实）。这一支就是修那件事的：
**颜色说它是条就是条**，宽窄只决定「比例能不能量」。

**为什么下沿就是 7 而不是更低**：更窄的块（1…6 列）连「一根条」的证据都不够 ——
门总得落在某一列上，这一列取的是**画面给的最短真条**。
"""

MIN_FILL_COLUMNS: Final[int] = 12
"""**够不够量比例**的宽下限：填充段短于它的候选**照样报「在」**，比例落**未采样**。

真条实测最窄 16 列（`special_黄色菱形...png` 那一块），线取 12 是留了 4。
**它不再是一道「丢块」的门**：短于它的候选留在 `bars` 里，只是比例不敢给 ——
短填充给不出「填了多少」这句话该有的证据（它也不让这一帧落成「不在」）。
`raw_b` 里那块 9 列、色距 8 的假阳性**不是这条线挡的，是上面那道矩形门挡的**。
"""

RECT_FILL_MIN: Final[float] = 0.95
"""**矩形**那道门：**每一列都亮着**的那几行（这一块的「条身」）要装下整块
`RECT_FILL_MIN` 的像素。

一根条是**一条平的带子**：横着看每一列一样高、竖着看上下两条边是直的 —— 所以
「每一列都亮着」的那几行，正好就是这根条，别的像素不该跑出那几行去。
**量出来的空档**（两个批次加 `boss.png` 整帧，取「条身像素 / 整块像素」）：

- 真条这一侧：**0.990…1.0**（最松的是 `敌条互叠2` 那根 87 列 —— 尾巴上有 5 列
  被旁边那根条压出 1…2 行；其余都是 1.0）；
- 假阳性这一侧：**0.878**（`raw_b` 那块 9 列、色距 8 的）、0.910（`boss.png`
  红门边上那块 28 列的）、0.40 / 0.51（`S5-0-0` 那两段栈道边）。

线取 0.95：下留 0.04、上留 0.04。**它是这道门唯一的承重处**（`raw_b` 那一块）：
没有它，那一块会因为色距 8 直接变成一根「条」。

**不是「占满外接框」那种量法**：`boss.png` 那根红宽条有 **10 行**（不是 8），
按外接框算是 0.83，会把它自己挡掉 —— 条身那几行才是一根条的形状。
"""

RECT_ROW_SPREAD_MAX: Final[int] = 20
"""**内部均匀**那道门：那 8 行的**逐行**中位色，通道极差不得超过它。

条是**平的**（从上到下同一种颜色）；地形木料是**渐变的**。量出来的空档：

- 真条这一侧：**0…13**（最松的是 `敌条互叠2` 那块被压过一截的 120 列）；
- 渐变这一侧：`全满血` 那块橙黄警戒线 **38**、`raw_b` 那块深红平片 20、
  红门那块木料 30 以上。

线取 20：下留 7、上留 0（`raw_b` 那块 20 刚好压线）。**它今天不承重**
（那一块已经被矩形门收掉了），但它量的是「这一块是不是一根**平**的条」——
所以留在门里，不为了「数清」把它拿掉。

**为什么不按逐列极差设门**：那一支真条能到 **34**（被压过的那些），而假阳性
41 起 —— 空档只有 7，当不了门。
"""
TRACK_CONTRAST: Final[int] = 12
"""空轨那道门：**条身那几行**（按量到的，见 `_body_rows`）的中位色均值，离**它上面
那几行**与**它下面那几行**的均值，**两侧都要**差过这个数，才算「这一段不是地形」。

**这个数也取在空档里**（量出来的）：

- **不是空轨**这一侧：平地上两侧差 **0…5**；条自己那道**很软**的尾巴差 **6…8**
  （`raw_a` 那根条的 x1799…1822，两侧都差 6…8）—— 最大 8；
- **是空轨**这一侧：偏暗的那一段差 **20…32**（raw_a x1761…1798），
  偏亮的那一段差 **100 以上**（raw_b x1676…1684 那段纯灰）—— 最小 20。

所以空档是 **8…20**，线取 12，两侧余量 **4 与 8**。取 12 而不是取 8 的用处，
恰恰是**不把那道软尾巴算进整长**：算进去整长会多 24 列，比例就从 0.69 掉到 0.57。
"""

TRACK_EDGE_GAP: Final[int] = 1
"""量空轨时，条身与「借来当地形的那几行」之间**空一行**。

一根条的上下边是**软的**（抗锯齿 + 半透明）：贴着条身的那一行常常还是条自己的颜色。
实测最要紧的一处是 boss 那根红条 —— 它 **10 行**（y1002…1011），而 y1001 与 y1012
**还是那条轨**（(45,43,41)），真正的地形从 y1013 才开始。不留缝的时候，`_off_terrain`
的「上面」那一窗（y992…1001）把 y1001 那道轨算了进去、中位色被拉低到 (48,46,43)，
与轨色 (45,44,42) 只差 **2** —— 空轨当场量不到，整长 = 填充段。

**1 行就够**（量出来的）：条的软边实测就是 1 行 —— `在_压住干员` 那根的 y700 与
y709、`raw_b` 上那根的 y489 都还是条自己，再往外才是地形。留 2 行会把
`在_压住干员` 那根的软尾从 4 列算成 12 列，比例形态从**区间**掉成**精确**
（那一支本来不敢给精确数）。
"""

TRACK_DARK_LINE: Final[int] = 55
"""**暗底地形**上量空轨的开关：这一列自己的亮度 <= 55 就换一支量法。

**为什么非有它不可**（量出来的）：空轨是**半透明压暗** —— 它压在地形上，地形暗它就暗。
所以**地形本身很暗**那一族里，轨色会与它一侧的地形**长得几乎一样**：

| 那一根（画布坐标） | 轨色 | 上面地形 | 下面地形 | 逐通道最大差 |
| --- | --- | --- | --- | --- |
| `在_boss_红宽条` x1677（红条） | (45,44,42) | (48,46,43) | (69,67,63) | 上面只有 **3**、下面 24 |
| `在_boss_红宽条` x1632（24 列） | (25,18,20) | (92,24,26) | (30,28,30) | 下面只有 **10**、上面 67 |
| `在_boss_红宽条` x1690（24 列） | (12,10,9) | (35,10,8) | (15,14,11) | 上面 23、下面 **4** |

「两侧都要差够远」在这一族上会**系统性答成「没有空轨」**（整长 = 填充段），
而那正是那两根读成 1.0 的成因。所以**这一列自己就暗**时换一支：

1. **逐通道**比（`|差|.max()`）—— 上面 x1690 那一行，逐通道是 23，而三个通道的
   **均值**只差 7.3（红的差 23、绿蓝几乎不变），均值那一支会把轨道漏掉；
2. 上面、下面**任一侧**够远就算 —— 另一侧与轨色重合是这一族的常态。

**55 取在量出来的空档里、并往暗那一侧靠**：

- **必须走这一支**的那些列，亮度实测 **10…45**（最松的是那根 24 列的 x1682 / x1684，
  两个都是 44.7）；
- **必须不走**这一支的最近样本是 `在_压住干员` 里那根绿技力条，亮度 **100**
  —— 它离下面那块地形只差 1，当场停下，那一帧的读数与单 5 **逐位相同**
  （区间 0.7143…0.8824）。

所以空档是 **45…100**，线取 55：下留 **10**、上留 **45**。**故意往暗那一侧靠** ——
亮底走的 min 那一支是**严**的那一支（地形与自己差 0，天然停得住），暗底走的 max 那一支
是**宽**的那一支，线压低一点，宽的那一支就少盖几列。

**亮底那一族走的还是原来那一支量法**（带内均值 vs 上下均值、取小的那个）；
`raw_a` 左那根从 125 涨到 129 是**取哪几行**改了（条身量到的 + 空 1 行），不是这一支改了。

**暗底那一支只有一个真值条件**：这一批里「轨色与一侧地形重合」只被 boss 那两根验过，
而且是同一帧、同一块机甲脚 —— 别的暗底（暗地板、夜里、阴影里）**零**。这一条
如实写在夹具 README 的缺口里。
"""

TRACK_GAP: Final[int] = 2
"""跨过「看起来像地形」的那几列时，最多连续放几列（填充段与空轨**共用**这一个数）。

与我方条那一支同一个用处：填充段里的抗锯齿列、空轨里的 1 列杂色，不该把一整段
判成「到此为止」。`raw_b` 那根条正中间就有**1 列**被压暗（x1577 上那一条竖线）——
1 这一档刚好跨不过去，整长会从 130 掉到 98。**它跨不过缝**：真正被压断的那一档
由 LowerBound 接。

**为什么是 2 而不是 3**：3 起，地形自己的杂色（一根条右邻那片地上零星的「不是地形」
列）会把整长一路推出去 —— 实测 `raw_a` 那两根会从 124/125 推到 131/131，
而它们画面上的整长是 122…123（同批 20 根满条量到的众数就是 123）。
"""

MAX_BAR_COLUMNS: Final[int] = 200
"""一根条整长的上限：量到这个宽度还没到头，就当「到头了也不知道」。

实测最长的整长是 raw_b 那根 `154。200 是它的余量，也是搜索窗的上限 ——
没有它，一次量错就会把整张战场扫穿。
"""

FIELD_TOP: Final[int] = 100
FIELD_BOTTOM: Final[int] = 1040
"""战场那一条：**顶部 HUD 之下、底部备战席之上**。筛只在这一段里做。

**为什么要有这一段**：整张画布上也有橙色的 UI（顶部那个橙靶心图标在 y18…68、
右下那个费用菱形在 y1187 起、左下的技能范围框图），它们**本来就不该进这条链**。
实测那几块的**高**都在 25 行以上或 6 行以下，形状门也挡得住，但那是**顺带**
挡住的；把战场这一段划出来，理由只有一条：**敌人条只可能出现在敌人站的地方**。

实测全库筛出来的真条，行都落在 263…934 之间；战场下沿取 1040（备战席那一条
从 1045 起），上沿取 100（顶部 HUD 那一条到 68 止）。
"""

ROW_SLACK: Final[int] = 1
COL_SLACK: Final[int] = 4
"""判「这块是不是我们自己那两根条」的两个容差：**行差 1 行以内、列差 4 列以内**。

**为什么不是「跟我方条那 16 行有交叠」**：敌人条可以**压在**我方条上面，而且能压到
**整段行带都交叠**。实测两例：

- `raw_a`：敌条 y481（8 行 481…488），我方血条 y486（486…493）—— 交叠 3 行；
- `raw_b`：敌条 y482（482…489），我方血条 y476（476…483）—— 交叠 **8 行**，
  两边列范围还重了 17 列（敌条 x1546…1675、我方条 x1417…1562）。

按「交叠」判，第二例那根**真敌人条会被当成我方条剔掉**。所以要判的是**重合**：
一块要**正好落在**我方血条那 8 行、或者**正好落在**紧贴它下面第 8 行的那条技力条
的 8 行上（行差 <= `ROW_SLACK`），**并且**列范围夹在我方那一格的跨度里
（列差 <= `COL_SLACK`）。我方那两根条是**同一格 UI 的两截**（共用左缘、只差 8 行），
敌人条是**另一格** —— 判重合就把它们分开了。
"""


@dataclass(frozen=True)
class ScreenedCore:
    """筛出来的一块**填充段**（还没量空轨）。

    - `rect`：这一块的 8 行矩形（左上角 + 宽 8 高）；
    - `distance`：它的**中位色距**（逐列中位色离 ENEMY_FILL 的逐通道最大差，
      取这一段的中位）；
    - `masked`：它是不是**被我方条的位置挡掉**的那一块（调用方给的 `our_bars`）。

    `masked` 的块留在读数里、不进 `EnemyBars.bars`：**「那里有东西但那是我们自己的」
    与「那里什么都没有」是两件事**，混掉就是把排除做成了否认。
    """

    rect: PixelRect
    distance: int
    masked: bool

    @property
    def score(self) -> float:
        """这一块的分数：1 - 颜色距 / 255（与那两条线同一个量纲）。"""
        return max(0.0, min(1.0, 1.0 - self.distance / 255.0))


@dataclass(frozen=True)
class EnemyBar:
    """一根敌人条：**整根**的矩形 + 填充比例 + 它跟旁边分不分得开。

    `rect` 是**整根条**（填充段 + 空轨），不是填充段 —— 因为「填了多少」这个
    比例的分母是整根。它的**高是量到的条身行数**（`_body_rows`），不是常量
    `BAR_ROWS`：一根量到 10 行的条报成 8 行，就是拿写死的数去描述一个量出来的
    东西（`BAR_ROWS` 只当默认与上限）。`fill` 是那个比例（0…1），形态见模块
    开头那张表。

    `separated` = **这一根能确认跟旁边那些分得开**（它就是一根完整的条）。
    `False` 时 `fill` 落 `Unsampled()`：**整长量不准，比例就没有分母** ——
    它可能只是某一根条被压出来的一截（分不开的判据见 `_inseparable`）。
    """

    rect: PixelRect
    fill: SlotValue
    distance: int
    separated: bool


@dataclass(frozen=True)
class EnemyBars:
    """这一帧的读数：三个出口 + **条数** + 逐根 + 被我方条挡掉的那几块。

    `presence` 是**这一帧画面上看不看得见敌人血条**那个布尔（三态）：
    `Valued(Exact(True))` 在 / `Valued(Exact(False))` 不在 / `Unsampled()` 认不出。
    `bars` 非空当且仅当它在。

    `count` 是**这一帧有几根**，与 `bars` 分开报 —— 叠在一起的条不要求数清
    （用户 10-14 裁定），所以这一支要能降级：

    | 情形 | 落法 |
    | --- | --- |
    | 每一根都分得开（`separated` 全 True） | `Valued(Exact(len(bars)))` |
    | 有分不开的 | `Valued(LowerBound(k))`，k = 分不开的那一堆各自只算一根 |
    | 这一帧认不出 | `Unsampled()`（与 `presence` 同一个出口） |

    **`LowerBound` 这一支永远不高报**：它说的是「**至少** k 根」，所以 k 只数
    确实互相分得开的那些，分不开的一堆只贡献 1。`len(bars)` 与它不总相等 ——
    差值正是这一支存在的理由（分不开时 `len(bars)` 会把一截数成一根）。
    """

    presence: SlotValue
    count: SlotValue
    bars: tuple[EnemyBar, ...]
    cores: tuple[ScreenedCore, ...]


def screened_cores(
    frame: Frame, *, our_bars: Sequence[PixelRect] = ()
) -> tuple[ScreenedCore, ...]:
    """把这张战场**一次筛完**，给回每一块候选（不量空轨、不落三态）。

    它是 `read_enemy_bars` 的那一半公开件：夹具要让别人拿**同一条链的中间产物**
    去核对（哪些块被筛出来了、哪几块被我方条挡掉了），就不能让人另写一份筛法 ——
    另写一份，量的就是另一条链了。
    """
    region, top = _field(frame)
    return tuple(
        _screened(core, top, our_bars) for core in _screen(region, top)
    )


def read_enemy_bars(
    frame: Frame, *, our_bars: Sequence[PixelRect] = ()
) -> EnemyBars:
    """读**这一帧画面上有几根敌人血条、各自填了多少**。

    落法（三个出口与可释放菱形那条链同款）：

    | 情形 | 落法 |
    | --- | --- |
    | 有一块的分数 >= `SOLID_SCORE` | `Valued(Exact(True), 最高分)` |
    | 每一块都 <= `ABSENT_SCORE`，或者一块都没有 | `Valued(Exact(False), 最高分)` |
    | 有块、但最高的那块的分数落在中间带 | `Unsampled()` |

    `count` 与 `bars` 一起给（见模块开头「条数」那一节）：**分得开才给
    `Exact`，分不开就降成 `LowerBound`**，分不开的那几根 `fill` 落 `Unsampled()`。

    `our_bars` 是**我们自己那两根条的位置**（每条给血条那块矩形，144/146 宽、
    8 行高都行 —— 这一层按「血条 + 紧贴它下面第 8 行的技力条」那 16 行解释它）。
    被我方条挡掉的块**不进 `bars`、也不参与这个布尔**，但留在 `cores` 里。

    **取区越出画布 -> 抛异常**（能力缺失，不是「看不到」）。
    """
    region, top = _field(frame)
    read: list[tuple[_Core, int, int, int, SlotValue]] = []
    screened: list[ScreenedCore] = []
    best = 0.0
    middle = False
    for core in _screen(region, top):
        entry = _screened(core, top, our_bars)
        screened.append(entry)
        if entry.masked:
            continue
        best = max(best, entry.score)
        if core.distance <= SOLID_TOL:
            total, row, rows, fill = _read_bar(region, core)
            # 填充段短于 MIN_FILL_COLUMNS：**存在性照报，比例不给**（见模块开头
            # 「宽度：进门松、出口严」）—— 短填充给不出「填了多少」该有的证据。
            read.append(
                (
                    core,
                    total,
                    row,
                    rows,
                    Unsampled() if core.width < MIN_FILL_COLUMNS else fill,
                )
            )
        elif core.distance < ABSENT_TOL:
            middle = True
    read.sort(key=lambda item: (item[0].y, item[0].x))
    grouped = _inseparable(region, read)
    bars = tuple(
        EnemyBar(
            # 高是**量到的条身行数**（`_body_rows`）：写死 BAR_ROWS 去描述一根量到
            # 10 行的条，就是拿常量冒充事实。y 也跟着量到的那一行走。
            rect=PixelRect(core.x, row + top, total, rows),
            fill=fill if grouped[index] == index else Unsampled(),
            distance=core.distance,
            separated=grouped[index] == index,
        )
        for index, (core, total, row, rows, fill) in enumerate(read)
    )
    if bars:
        presence: SlotValue = Valued(Exact(True), best)
    elif middle:
        presence = Unsampled()
    else:
        presence = Valued(Exact(False), best)
    # 条数是**几何分组**的结论，不是颜色挣来的 —— 所以置信度写 None（§1：None =
    # 不是读来的），不把颜色那一头的分数挪过来冒充它。
    if not bars:
        count: SlotValue = Unsampled() if middle else Valued(Exact(0), best)
    elif all(bar.separated for bar in bars):
        count = Valued(Exact(len(bars)), None)
    else:
        count = Valued(LowerBound(len(set(grouped))), None)
    return EnemyBars(
        presence=presence, count=count, bars=bars, cores=tuple(screened)
    )


# ---------------------------------------------------------------------------
# 战场那一条
# ---------------------------------------------------------------------------


def _field(frame: Frame) -> tuple[np.ndarray, int]:
    """战场那一条的像素 + 它在画布上的原点行。

    上下各多留 8 行（`BAR_ROWS`）：量空轨时要拿「带子上方那 8 行」当底，
    贴着战场边的那一根条不该因为借不到底而读不出来。**判据只认战场那一段
    里的块**（见 `_inside_field`），多留的那几行只是给量法用的底。
    """
    if not isinstance(frame, Frame):
        raise TypeError(f"要读的是一帧，收到 {type(frame).__name__}")
    height = frame.canvas.height
    if FIELD_TOP - BAR_ROWS < 0 or FIELD_BOTTOM + BAR_ROWS > height:
        raise ValueError(
            f"战场那一段 {FIELD_TOP}…{FIELD_BOTTOM} 加上上下各 {BAR_ROWS} 行借底，"
            f"越出画布 {frame.canvas.height}：常量与画布对不上了"
        )
    top = FIELD_TOP - BAR_ROWS
    bottom = FIELD_BOTTOM + BAR_ROWS
    return frame.pixels[top:bottom], top


def _inside_field(row: int, top: int) -> bool:
    """这一块是不是落在**战场那一段**里（不是借底那几行）。"""
    absolute = top + row
    return FIELD_TOP <= absolute and absolute + BAR_ROWS <= FIELD_BOTTOM


def _screened(
    core: _Core, top: int, our_bars: Sequence[PixelRect]
) -> ScreenedCore:
    """一块私有候选 -> 对外的读数（补上画布坐标与「是不是被我方条挡掉」）。"""
    return ScreenedCore(
        rect=PixelRect(core.x, core.y + top, core.width, BAR_ROWS),
        distance=core.distance,
        masked=_is_ours(core, our_bars, top),
    )


# ---------------------------------------------------------------------------
# 一次筛完
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Core:
    """私有的那一块：**帧内坐标**（行相对战场那一条的顶，列就是画布列）。"""

    x: int
    y: int
    width: int
    distance: int
    reference: tuple[int, int, int]


def _screen(region: np.ndarray, top: int) -> tuple[_Core, ...]:
    """整张战场筛一遍 -> 每一块「8 行高的条块」。

    三步（顺序不能换）：**逐像素比色** -> **竖着 8 行** -> **连通块 + 形状门 + 宽度门**。

    第一步比的是**两族参考色**（橙 `ENEMY_FILL` 与红 `ENEMY_FILL_RED`，见模块开头）；
    第三步有两道形状门（矩形 / 内部均匀）与**按族分的**宽度门（红族 `RED_MIN_COLUMNS`、
    橙族 `MIN_CANDIDATE_COLUMNS`）。

    第二步是 `_long_runs`：**竖着连不满 8 行的像素整片消失**。它挡掉的是那一类
    最要紧的假阳性 —— 地貌的「红扁条」（`S5-0-0` 那堆土红栈道边在颜色门里还剩
    4…5 行、`shot26` 那根发光栏杆只有 5…6 行，都过不了 8 行）。

    **这一步不能用 cv2.morphologyEx 的开运算**：8 这种偶数尺寸的核，OpenCV 取
    `BAR_ROWS // 2` 当锚点，开运算会把结果**整块往下挪一行** —— 量出来的每一根条
    都会比画面上低一行，而且贴着战场下沿的那一根会被挪出界、整个漏掉。
    所以这里自己数竖向的连续长度（前景与背景各扫一遍），**位置一个像素都不动**。

    最后一步 `_merge` 把「被一条窄东西压过去」的两块接回一根条。
    """
    found = list(_screen_family(region, top, ENEMY_FILL, MIN_CANDIDATE_COLUMNS))
    found.extend(_screen_family(region, top, ENEMY_FILL_RED, RED_MIN_COLUMNS))
    found.sort(key=lambda core: (core.y, core.x))
    return _merge(found, region)


def _screen_family(
    region: np.ndarray,
    top: int,
    reference: tuple[int, int, int],
    min_columns: int,
) -> list[_Core]:
    """**一族颜色**单独筛一遍 -> 这一族里够格当候选的块。

    **两族各筛一遍、不并成一个掩膜**：并起来之后，两族在画面上挨着的像素会把
    本来分开的块连成一块 —— 实测 `敌条互叠2` 那一帧会因此把 7 根条并进别的块里、
    那些块的高度涨过 `MAX_BAR_ROWS` 而**整批丢掉**。那是**在改量法**，不是加一族
    颜色；分成两遍，橙族那一边的量法与加红族之前**逐块一样**。
    """
    pixels = region.astype(np.int16)
    family = np.asarray(reference, np.int16)
    mask = np.abs(pixels - family).max(axis=2) <= SCREEN_TOL
    tall = _long_runs(mask, BAR_ROWS).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(tall, connectivity=8)
    found: list[_Core] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if height > MAX_BAR_ROWS or width < min_columns:
            continue
        row = _band_row(mask, y, height, x, width)
        if row < 0 or not _inside_field(row, top):
            continue
        block = mask[y : y + height, x : x + width]
        band = region[row : row + BAR_ROWS, x : x + width]
        if not _rectangular(band, block, area, width):
            continue
        profile = np.median(band, axis=0)
        distance = np.abs(profile.astype(np.int16) - family).max(axis=1)
        lit = distance <= SCREEN_TOL
        tolerant = _leading_run(lit, TRACK_GAP)[1]
        if tolerant < MIN_CANDIDATE_COLUMNS:
            continue
        found.append(
            _Core(
                x=x,
                y=row,
                width=tolerant,
                distance=int(np.median(distance[:tolerant])),
                reference=tuple(int(v) for v in family),
            )
        )
    return found


def _rectangular(
    band: np.ndarray, block: np.ndarray, area: int, width: int
) -> bool:
    """这一块是不是**一个矩形、而且内部平**（形状门，见模块开头）。

    - **矩形**：**每一列都亮着**的那几行（这块的「条身」）要装下整块
      `RECT_FILL_MIN` 的像素（见 `RECT_FILL_MIN`）—— 剩下的像素是散在条身外的
      毛刺，它们多了，这一块就不是一条平的带子。
    - **内部平**：那 8 行取**逐行**中位色，通道极差 <= `RECT_ROW_SPREAD_MAX` ——
      条从上到下同一种颜色，地形木料是渐变的。

    `band` 是已经挑好的那 8 行像素（`_band_row` 的结果），`block` 是**整块**
    （外接框那么大）的掩膜，`area` 是整块的像素数，`width` 是外接框的宽。
    """
    body = int(block.all(axis=1).sum())
    if body * width < RECT_FILL_MIN * area:
        return False
    rows = np.median(band, axis=1).astype(np.int16)
    return int((rows.max(axis=0) - rows.min(axis=0)).max()) <= RECT_ROW_SPREAD_MAX


MERGE_GAP: Final[int] = 8
"""接两根的时候，中间最多跨几列。

「同一根条被一条窄东西压过去」实测只差 **1 列**（`raw_b` 那根条在 x1577 上有一条
1 列宽的竖线）—— 8 是它的余量，不是量出来的上沿。**判据不是这个数**：中间那几列
还得**每一列都更靠近填充色**（见 `_blended`），所以这个数只是搜索的上界。
"""


def _merge(cores: list[_Core], region: np.ndarray) -> tuple[_Core, ...]:
    """把「被一条窄东西压过去」的两块接回一根条。

    **为什么非接不可**：一根条被压出 1 列缝时，筛出来是**两块**；不接的话
    `len(bars)` 会把一根数成两根 —— 而「有几根条」正是这一单要回答的第一件事。

    **判据**（`_attached`）：两块的行带相差不超过 1 行、中间那几列**每一列都更靠近
    填充色**（半透明遮挡压出来的混色）、且间距 <= `MERGE_GAP`。
    中间是**纯地形**时不接：那是两根挨着的条。
    """
    merged: list[_Core] = []
    for core in cores:
        if merged and _attached(merged[-1], core, region):
            head = merged.pop()
            right = max(head.x + head.width, core.x + core.width)
            merged.append(
                _Core(
                    x=head.x,
                    y=head.y,
                    width=right - head.x,
                    distance=min(head.distance, core.distance),
                    reference=head.reference,
                )
            )
        else:
            merged.append(core)
    return tuple(merged)


def _attached(left: _Core, right: _Core, region: np.ndarray) -> bool:
    """右边那一块是不是**左边那一根条被压过去之后接着露出来的那一截**。"""
    if right.x < left.x + left.width:
        return False
    if abs(right.y - left.y) > 1:
        return False
    if right.x - (left.x + left.width) > MERGE_GAP:
        return False
    for column in range(left.x + left.width, right.x):
        if not _blended(region, left.y, column):
            return False
    return True


def _blended(region: np.ndarray, row: int, column: int) -> bool:
    """这一列是不是**条被半透明的东西压过去**压出来的混色。

    判据：这一列的颜色**离填充色更近**，而不是离它上下那两块地形更近。

    **为什么不用 `_off_terrain`**：那一支比的是「与地形的差」，而**压在一条填充色
    上面的半透明遮挡**压出来的混色，亮度可能正好与它下面那块地形接近 ——
    实测 `raw_b` 那条 1 列宽的缝（x1577）就是：`_off_terrain` 只有 **4.2**
    （低于 `TRACK_CONTRAST`），可它离填充色 **64**、离上下地形 **99 / 112**。
    所以「更靠近填充色」才是这一列压的是条、不是空地的证据。

    **参考色写死橙族那一支**：红族的核心要走到这里（`_merge` / `_inseparable` 都要
    这一支）时，比的是橙族的填充色 —— 红条混色那一头会答成「不是混色」，于是
    红核之间**不接、也不并成一组**（保守：各算一根，不高报）。这一版红族只有 boss
    一根，没撞上这条；要让红条也吃这一支，得把参考色按族传进来。
    """
    inside = np.median(region[row : row + BAR_ROWS, column], axis=0).astype(np.int16)
    above = np.median(region[row - BAR_ROWS : row, column], axis=0).astype(np.int16)
    below = np.median(
        region[row + BAR_ROWS : row + 2 * BAR_ROWS, column], axis=0
    ).astype(np.int16)
    reference = np.asarray(ENEMY_FILL, dtype=np.int16)
    to_fill = int(np.abs(inside - reference).max())
    to_ground = min(
        int(np.abs(inside - above).max()), int(np.abs(inside - below).max())
    )
    return to_fill < to_ground


def _long_runs(mask: np.ndarray, rows: int) -> np.ndarray:
    """布尔掩膜 -> **属于某一条竖着连满 `rows` 行的那些像素**（位置一个都不动）。

    逐列数「经过这一行的那一段有多长」（往上连了多少 + 往下连了多少 − 1），
    够长就留下。前景与背景各扫一遍，都是纯 numpy。

    **为什么不用 cv2.morphologyEx**（见 `_screen`）：偶数尺寸的核会把结果整块挪
    一行，而「一根条在哪一行」正是这一条链要报的东西之一。
    """
    height = mask.shape[0]
    forward = np.zeros_like(mask, np.int32)
    backward = np.zeros_like(mask, np.int32)
    for row in range(height):
        forward[row] = np.where(mask[row], forward[row - 1] + 1 if row else 1, 0)
    for row in range(height - 1, -1, -1):
        backward[row] = np.where(
            mask[row], backward[row + 1] + 1 if row + 1 < height else 1, 0
        )
    return mask & ((forward + backward - 1) >= rows)


def _band_row(mask: np.ndarray, y: int, height: int, x: int, width: int) -> int:
    """一块里**那 8 行**从哪一行起：滑一个 8 行的窗，取「里面的像素最多」那一窗。

    干净的一根条只有一种选择（它就是那 8 行）。`height` 是 9…12 时（抗锯齿多出来
    的几行、或者上下贴着别的东西）才有得选 —— 那一窗里逐列中位色最像填充色的那 8
    行，才是这一块的「条身」。
    """
    best_top = -1
    best_hits = -1
    for top in range(y, y + height - BAR_ROWS + 1):
        hits = int(mask[top : top + BAR_ROWS, x : x + width].sum())
        if hits > best_hits:
            best_hits, best_top = hits, top
    return best_top


def _leading_run(flags: np.ndarray, gap: int) -> tuple[int, int]:
    """自下标 0 起连续为真的那一截 -> (不留余量的, 夹了 gap 个假的)。

    **两个数各有各的用处，不许合成一个**（与我方血条那条链同一支）：

    - 第一个（strict）：一格缝都不跨。它是「**至少这么多**」那个下界的依据；
    - 第二个（tolerant）：允许夹 gap 个连续假。它是**比例**的依据 ——
      填充段里的抗锯齿列不该把比例断在第一道上。
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


def _resumes(lit: np.ndarray, filled: int) -> bool:
    """`filled` 之后**还接着亮**吗 —— 也就是「填着的那一截被切断过」。

    `_leading_run` 连着碰到 `TRACK_GAP + 1` 个假列就停下了，所以它停下的**那一侧**
    只要还有亮列，就是缝底下压着东西、条并没有到此为止。
    """
    return bool(lit[filled : filled + MERGE_GAP].any())


# ---------------------------------------------------------------------------
# 空轨与比例
# ---------------------------------------------------------------------------


def _body_rows(region: np.ndarray, core: _Core) -> tuple[int, int]:
    """这一根条**量到的**条身 -> (条身顶行, 条身高)。**不许写死 8**。

    自 `core.y` 起上下扩：**填充段那几列**上，条身以内是填充色、以外不是 ——
    走到不亮为止。封顶 `MAX_BAR_ROWS`（形状门已经保证那一块不高于它）。

    实测两档：小怪与红门那些整齐的块是 **8 行**，boss 那根红条是 **10 行**
    （`在_boss_红宽条` x1562，y1002…1011），`raw_b` 上那根是 **9 行**。
    「条身取几行按量到的」是用户这一单的裁定 3 —— 写死 8 的代价见 `TRACK_EDGE_GAP`。
    """
    reference = np.asarray(core.reference, np.int16)
    columns = region[:, core.x + 2 : core.x + max(3, min(core.width, 12))]
    columns = columns.astype(np.int16)

    def lit(row: int) -> bool:
        colour = np.median(columns[row], axis=0)
        return int(np.abs(colour - reference).max()) <= SCREEN_TOL

    top = core.y
    while top - 1 >= 0 and lit(top - 1):
        top -= 1
    bottom = core.y
    while bottom + 1 < columns.shape[0] and lit(bottom + 1):
        if bottom - top + 2 > MAX_BAR_ROWS:
            break
        bottom += 1
    return top, max(BAR_ROWS, bottom - top + 1)


def _band_median(
    region: np.ndarray, start: int, rows: int, column: int
) -> np.ndarray | None:
    """某一列上自 `start` 起 `rows` 行的中位色；这一段跑到区外时给 `None`。

    **要夹一下**：条身最多 12 行、上下再各让出 `TRACK_EDGE_GAP` 行，战场那一条
    只多借了 8 行底 —— 贴着战场边的那一根会伸出去。伸出去的那一侧就当「借不到」，
    另一侧照样算（只影响贴着战场上下沿的那几根）。
    """
    lo = max(0, start)
    hi = min(region.shape[0], start + rows)
    if hi - lo < 2:
        return None
    return np.median(region[lo:hi, column], axis=0)


def _off_terrain(
    region: np.ndarray, row: int, rows: int, column: int
) -> float:
    """这一列**离地形多远**：条身那 `rows` 行与它**上面**、**下面**那两段的差。

    取**两侧都要差**是必要的：`raw_b` 那一段空轨是**纯灰 177**，而它的上下都是
    74 左右 —— 「更暗才算轨道」这条判据在那里会当场失效（那段灰正是那根条的
    没填到的那一截）。反过来 `raw_a` 那一段是偏暗的（43 对 65）。
    **两个方向都算，才盖得住这两种。**

    **暗底那一族改取「任一侧」**（见 `TRACK_DARK_LINE`）：那一族里轨色与它一侧的
    地形长得一样，「两侧都要」会系统性答成「没有空轨」。闸门是**这一列自己暗**，
    不是「哪一族地形」—— 亮底那一族走的还是**上面那一支量法**
    （带内均值 vs 上下均值、取小的那个），改的只是「取哪几行」。
    """
    inside = _band_median(region, row, rows, column)
    if inside is None:
        return 0.0
    sides = []
    for start in (row - TRACK_EDGE_GAP - rows, row + rows + TRACK_EDGE_GAP):
        side = _band_median(region, start, rows, column)
        if side is not None:
            sides.append(side)
    if not sides:
        return 0.0
    middle = float(inside.mean())
    if middle <= TRACK_DARK_LINE:
        # 暗底那一族：**逐通道**量（只差一个通道也算），取两侧**任一**
        return float(
            max(
                int(abs(inside.astype(np.int16) - side.astype(np.int16)).max())
                for side in sides
            )
        )
    # 亮底那一族：与今天同一支（带内均值 vs 上下均值，取小的那个）
    return min(abs(middle - float(side.mean())) for side in sides)


def _track_end(
    region: np.ndarray, row: int, rows: int, start: int
) -> tuple[int, int, bool]:
    """自 `start` 起量空轨 -> (最后一个「不是地形」的列 + 1, 那一段软尾的列数)。

    **终止条件**：连着 `TRACK_GAP + 1` 列都「离地形不够远」就停。它停得住的理由
    是**量的就是「离这一列上下那两段地形多远」** —— 地形自己与自己的差是 0，
    所以它不会一路吃到地里去；`raw_a` 那两根量到 129 / 124 列、`raw_b` 那根 154，
    画面上的众数是 122…123 与 154。

    软尾 = 末尾那几列**差得过线、但差不过两倍线**的列。边界落在它们里面时，
    「整根到哪儿为止」是一个**区间**而不是一个数（与「边界渐变」那一支同源）。

    第三个给回的是**这一趟有没有跨过回落**（bridged）：前面已经有过「是轨道」的
    列，后面出现「不是轨道」的列，再后面**又接上**「是轨道」—— 说明这一趟是
    **跨过**几列才走下来的。那几列到底是轨道里的杂色、还是条本来就到头了，画面
    答不了；于是整长就不是**连着**量出来的，比例只能给下界的形态（见 _read_bar）。
    紧贴填充段的那几列不算：那几列出现的时候，还没有出现过「是轨道」的列。
    """
    width = region.shape[1]
    limit = min(width, start + MAX_BAR_COLUMNS)
    column = start
    last = start
    miss = 0
    tail = 0
    hit = False
    bridged = False
    while column < limit:
        amount = _off_terrain(region, row, rows, column)
        if amount > TRACK_CONTRAST:
            if hit and miss:
                # 这一列是**跨过**前面那几列回落才接上的：整长不是连着量出来的。
                bridged = True
            hit = True
            last = column + 1
            tail = tail + 1 if amount <= 2 * TRACK_CONTRAST else 0
            miss = 0
        else:
            miss += 1
            if miss > TRACK_GAP:
                break
        column += 1
    return last, tail, bridged


def _read_bar(
    region: np.ndarray, core: _Core
) -> tuple[int, int, int, SlotValue]:
    """一根条 -> (整长, 条身顶行, 条身行数, 填充比例)。**整长 = 填充段 + 空轨**。

    比色用的是**这一根自己那一族的参考色**（`core.reference`）：红族那根比的是
    `ENEMY_FILL_RED`，不是橙族那一支 —— 比错了它整根都不亮、填充段会量成 0。

    填充比例（0…1）按「下界 / 区间 / 精确」三支落：

    | 情形 | 落法 |
    | --- | --- |
    | 填充段中间被切断（缝两边都还亮着） | `LowerBound(严那一段 / 整长)` |
    | 空轨的尾巴是软的（边界落在几列里） | `Interval(填充 / 整长, 填充 / 短整长)` |
    | 空轨量到了上限还没到头 | `Interval(填充 / 上限, 填充 / 整长)` |
    | 空轨量得不干净（跨过过回落） | `LowerBound(填充 / 整长)` |
    | 其余（含填满：没有空轨） | `Exact(填充 / 整长)` |

    **填满那一支不需要另开一条路**：没有空轨时整长就等于填充段，比例就是 1.0 ——
    实测 `screenshots/_probe.png` 那根满条就是这样（填充段 123 列，右邻第一列
    就回到地形，差 0…3；`raw_a` 那两根则各自留了 40 与 47 列空轨）。
    """
    reference = np.asarray(core.reference, dtype=np.int16)
    row, rows = _body_rows(region, core)
    span = core.width + MAX_BAR_COLUMNS
    profile = np.median(
        region[row : row + rows, core.x : core.x + span], axis=0
    )
    distance = np.abs(profile.astype(np.int16) - reference).max(axis=1)
    lit = distance <= SCREEN_TOL
    strict, tolerant = _leading_run(lit, TRACK_GAP)
    end, tail, bridged = _track_end(region, row, rows, core.x + tolerant)
    total = max(tolerant, end - core.x)
    if strict < tolerant or _resumes(lit, tolerant):
        return total, row, rows, Valued(
            LowerBound(strict / total), _fill_confidence(distance, strict)
        )
    if tail and tail < total - tolerant:
        return total, row, rows, Valued(
            Interval(tolerant / total, tolerant / (total - tail)),
            _fill_confidence(distance, tolerant),
        )
    if total - tolerant >= MAX_BAR_COLUMNS:
        return total, row, rows, Valued(
            Interval(tolerant / (tolerant + MAX_BAR_COLUMNS), tolerant / total),
            _fill_confidence(distance, tolerant),
        )
    if bridged:
        # 空轨是**跨过**回落才量完的：那个分母不是连着量出来的，可能多算了一截
        # 地形 —— 真整长只可能更短、真比例只可能更大，所以这里给的是下界。
        return total, row, rows, Valued(
            LowerBound(tolerant / total), _fill_confidence(distance, tolerant)
        )
    return total, row, rows, Valued(
        Exact(tolerant / total), _fill_confidence(distance, tolerant)
    )


def _fill_confidence(distance: np.ndarray, filled: int) -> float:
    """这一根比例读数有多稳：**定下这次读数的两个判据两侧隔了多远** / 255。

    与我方血条那条链同一个口径：里头最弱的一列（离色线最近的）与右边那一列
    （它必须干脆地在线外）里，小的那个。**另一边（空轨那一头）不进来** ——
    空轨量错只改分母，改不了「填充段到哪儿为止」这个事实。
    """
    if filled <= 0 or filled > distance.size:
        return 0.0
    inside = float(SCREEN_TOL - distance[:filled].max())
    if filled >= distance.size:
        return min(1.0, max(0.0, inside / 255.0))
    outside = float(distance[filled] - SCREEN_TOL)
    return min(1.0, max(0.0, min(inside, outside) / 255.0))


# ---------------------------------------------------------------------------
# 分不开的那几根：条数只能给下界
# ---------------------------------------------------------------------------


def _inseparable(
    region: np.ndarray, read: Sequence[tuple[_Core, int, int, int, SlotValue]]
) -> list[int]:
    """每一根条归到哪一组：自己一组的返回自己的下标，分不开的返回这一组的最小下标。

    **判据只有一条**：两根条**踩在同一条行带上**（行差 <= `ROW_SLACK`），而且中间
    **分不开**（见 `_touching`）。分不开的那一堆只贡献 1 根 —— 这就是 `count`
    那一支下界的来源。

    **这里不合并、也不动任何一条线**：把两块接回一根是 `_merge` 的事（间距
    `MERGE_GAP` 以内、且中间**每一列**都混色）。这里只是**承认分不开**，
    把条数降成下界 —— 精细的活儿在别处，这一支只负责**不假装数清了**。
    """
    parent = list(range(len(read)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for band in _bands(read):
        # **只比同一条带里相邻的两根**：隔着一根去比，中间那一根的列本身就会让
        # 「中间有没有混色」答成有 —— 三根同带的条会因此串成一组。
        for left, right in zip(band, band[1:]):
            if not _touching(region, read[left], read[right]):
                continue
            first, second = root(left), root(right)
            if first != second:
                parent[max(first, second)] = min(first, second)
    return [root(index) for index in range(len(read))]


def _bands(
    read: Sequence[tuple[_Core, int, int, int, SlotValue]]
) -> list[list[int]]:
    """把条按**行带**切开，每一条带里按列排好。

    「行带」是**连着**算的：行差 <= `ROW_SLACK`（与判我方条同一个容差）就算同一条，
    所以 451 与 452 同带、452 与 453 也同带 —— 一列 1 行一行的阶梯本来就分不开，
    把它切成两条带才是假装分开了。
    """
    order = sorted(range(len(read)), key=lambda i: (read[i][0].y, read[i][0].x))
    bands: list[list[int]] = []
    current: list[int] = []
    for index in order:
        if current and read[index][0].y - read[current[-1]][0].y > ROW_SLACK:
            bands.append(current)
            current = []
        current.append(index)
    if current:
        bands.append(current)
    return [sorted(band, key=lambda i: read[i][0].x) for band in bands]


def _touching(
    region: np.ndarray,
    first: tuple[_Core, int, int, int, SlotValue],
    second: tuple[_Core, int, int, int, SlotValue],
) -> bool:
    """同一条带上的两块**分不分得开**：分不开才 True。

    两种情形都不给「这是两根」这句话作保：

    - **整长交叠或相接**：同一条 8 行带上，两根整条不可能这样并排放 ——
      至少有一根是被截出来的一截；
    - **中间至少有一列是「混色」**（`_blended`：那一列比它上下两块地形更靠近
      填充色）—— 那正是「条被半透明的东西压过去」的像素证据，于是这两块
      可能本来就是同一根条的两截。

    **只找「混色」，不找「条色」**：中间那一截又亮起来的（同一条带上被我们自己的
    技能光压断）在 `_screen` 里已经接回去了（`_merge`）；走到这里还分开的，
    中间那几列要么是地形（那就真的是两根），要么是压在条上的半透明东西。
    """
    left, right = (first, second) if first[0].x <= second[0].x else (second, first)
    core, total = left[0], left[1]
    gap = right[0].x - (core.x + total)
    if gap <= 0:
        return True
    row = max(core.y, right[0].y)
    return any(
        _blended(region, row, column)
        for column in range(core.x + total, right[0].x)
    )


# ---------------------------------------------------------------------------
# 排除我方那两根条
# ---------------------------------------------------------------------------


def _is_ours(core: _Core, our_bars: Sequence[PixelRect], top: int) -> bool:
    """这一块是不是**我们自己那两根条**之一（血条，或者紧贴它下面的技力条）。

    判据是**几何重合**，不是颜色 —— 颜色挡不住它：我方技力条**生效中也是橙的**，
    它离 ENEMY_FILL 只有 20 上下，跟敌人条同一个色族。

    两根我方条的位置都由调用方的血条矩形折出来：血条 = 这一格的第 0…7 行，
    技力条 = 再往下第 `SKILL_ROW_OFFSET` 行起的那 8 行。一块要**踩在**其中一根的
    行上（行差 <= `ROW_SLACK`）**并且**夹在我方那一格的列跨度里（列差 <=
    `COL_SLACK`）才算我们的。**判重合、不判交叠**的理由见 `ROW_SLACK`。
    """
    row = top + core.y
    for bar in our_bars:
        if not isinstance(bar, PixelRect):
            raise TypeError(
                f"我方条的位置要给像素矩形，收到 {type(bar).__name__}："
                "这一层不自己猜干员在哪"
            )
        bar_left = int(round(float(bar.x)))
        bar_right = bar_left + int(round(float(bar.width)))
        if core.x < bar_left - COL_SLACK or core.x + core.width > bar_right + COL_SLACK:
            continue
        bar_top = int(round(float(bar.y)))
        for ours in (bar_top, bar_top + SKILL_ROW_OFFSET):
            if abs(row - ours) <= ROW_SLACK:
                return True
    return False
