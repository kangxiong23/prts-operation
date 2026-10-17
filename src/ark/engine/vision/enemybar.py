
"""敌人血条：**整张战场上一次筛完**，报**逐根的填充比例**、**拆不开的那一堆**与**未采样**。

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
   宽 >= MIN_CANDIDATE_COLUMNS、高 <= MAX_BAR_ROWS 的块；
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
**宽度门** `RED_MIN_COLUMNS`：boss 是**宽**的（那一档满条长 `BOSS_FILL_COLUMNS`
是 154 列），而红族最像条的假阳性（`boss.png` 那一帧红门上的红色碎块）只有
18…32 列。

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
（12）是**「这一块小得可疑」**的门：填充段短于它的候选**照样进逐根那一栏**
（列数与比例都给），只是多一个**小尺寸标记**（`EnemyBar.short_fill`）——
**不再抹掉任何一条读数**（用户 10-15 的裁定）。

**「在门以下」不是通往「不在」的一条路**：颜色说它是条就报「在」，颜色说不清
是不是条就落**未采样**，只有颜色说它**不是**条才是「不在」。

## 比例：**分母是常量，不是画面上量出来的那一截**

**血条长度只有两档**（用户 10-15 的裁定）：

| 档 | 谁 | 满条长（列） | 条身（行） |
| --- | --- | --- | --- |
| **小怪档** | **所有小怪**（含精英、经验怪）—— 橙族 | `MOB_FILL_COLUMNS = 123` | `MOB_BODY_ROWS = 8` |
| **boss 档** | **只有 boss** —— 红族 | `BOSS_FILL_COLUMNS = 154` | `BOSS_BODY_ROWS = 10` |

**用哪一档看族、不看类型**：橙族走小怪档、红族走 boss 档（见 `_Family`）。比例 =
**填充段列数 ÷ 那一档的满条长**。

**分母不来自画面** —— 这是这一版最大的一处改动，理由是一条实证：没填到的那一截
（半透明暗带）压在**任意背景**上，背景暗它就量不到，于是「拿它当分母」会读出一堆
互不相容的数 —— 实测同一屏里量到过 **73 / 117 / 122 / 127 / 129** 五种宽度、
互叠时量到 **9 行**、`S8-3-4` 沿集装箱一路走满 200 列上限。**那条路整个退役**
（原来量空轨的那几个常量、连同 `_track_end` / `_off_terrain` / `_band_median`
三个私有件、以及「量到上限」那一支都已删掉）。而「填了多少」**不需要**那个分母：两档长度是反推
出来的**常量** —— 各自的依据写在它们自己的 docstring 里。

## 填充段：怎么起、怎么止（可复核）

- **左端**：候选块的左缘 —— 条只可能从自己的左缘开始填；
- **右端**：在**量到的条身那几行**上逐列取中位色，离这一族参考色的**逐通道最大差**
  <= `SCREEN_TOL` 的**最后一列**；中间容 `FILL_GAP` 列不亮（抗锯齿、一根窄遮挡）；
- **那道容差跨过过**（`strict < columns`）或填充段右边**还接着亮**（`_resumes`）
  ⇒ 填充段**被切断**，缝底下藏着什么都看不见 ⇒ **两条读数都**落
  `LowerBound`，值取**严格那一段**（与我方血条那条链同一套纪律，不另立一套）；
- **填充段短于 `MIN_FILL_COLUMNS`** ⇒ 条照报「在」，**列数与比例都照给** —— 这一根
  进**逐根**那一栏；它**够不够格当一根条**这一件事交给那个**小尺寸标记**
  （`EnemyBar.short_fill`），**一列读数都不抹掉**（见「三样读法」那一节）；
- **填充段长过这一档的满条长** ⇒ 这一根（连同它那一组）落成**一堆**（见「三样
  读法」那一节），**不给单根比例**；
- **条身行数到了更高那一档的行数** ⇒ 这一根落 `unsampled`（防「橙条 boss」那一道
  便宜的跨档闸门）。门**对着 boss 那一档设**：小怪那一档量到 **>= 10 行**
  （= `BOSS_BODY_ROWS`）才拦，**9 行及以下放过** —— 9 是抗锯齿与叠条压出来的，
  9 不等于 10。

**可复核的两个点**（都在夹具里）：`raw_a` 那两根的填充段都是 **84 列**
（x1320…x1403 的中位色 (220,109,57)；x1404 是 (87,102,83)，逐通道差 132 > 34）
⇒ 84/123 = **0.6829**，与画面上量出来的约 0.683 对上；`在_raw_b_下` 那根 **57 列**
⇒ 57/123 = 0.4634 —— 它右边那 28 列**白条**是「这一下掉了多少血」的打击感展示，
**一列都不算进去**（用户 10-15 的裁定）。

## 分辨率：两档是**画布宽 2800 下的绝对像素**

两档常量都写成**原生画布 2800×1260 下的列数**，与 `field.BAR_WIDTH = 146` 同一个
规矩（都是常量分母，都不是从画面上量的）。**不写折算公式**：现有素材**全是
2800×1260**，别的分辨率**一个样本都没有** ⇒ 折算系数**未验**。
若将来出现别的分辨率，按 **`canvas.width / 2800`** 折算 —— 理由是**画布就是屏幕**
（《契约层》§2：画布 = 捕获分辨率本身，不缩放），游戏 UI 按屏幕排；**不拿战场区
宽当基准**：那是**这条链自己划的窗**，游戏不知道它存在。

## 三样读法：逐根 / 一堆 / 未采样（用户 10-15 的裁定）

这一层的输出是**三栏并列**，一眼分得开：

| 栏 | 是什么 | 带什么 |
| --- | --- | --- |
| `bars` | **逐根**读得出来的 | 填充段列数 + 比例（各自带**形态**与**余量**；短填充还带一个**小尺寸标记**） |
| `clumps` | **一堆**：挤在一起、拆不开的那一坨 | **区域** + **至少几根** + **这一坨的填充总列数** + **来路** |
| `unsampled` | 那一根确实有东西、但**填了多少读不出来** | 位置 + 那两项读数（都落未采样） |

三栏**互斥**，合起来正好是颜色说「是条」的那些块。**「一堆」与「未采样」是两件
事**：未采样 = 那里有东西、连填充都读不出来；一堆 = 知道那一坨在哪、多大、至少
几只。**一堆里没有单根比例**（硬给就是编）。

**「未采样」这一版只剩一条来路**：条身行数到了**更高那一档**的行数（跨档校验）。
**短填充不再是未采样**（用户 10-15 的裁定）：那 7 列的列数是**读出来了**的，抹掉它
就是用降级盖住已经知道的东西；那一块**够不够格当一根条**由**小尺寸标记**带出去
（残余风险如实说：它**仍可能是假阳性** —— 但「那里有东西」这个主张在它被筛成候选
的那一刻就下了，给值不新增假事实）。

**「一堆」多带一栏「来路」**（`EnemyClump.route`）：**分不开** · **长过一根的上限** ·
**两者**。两条证据的强度不同 ——「长过一根的上限」是**算术**（总列数长过满条长 ⇒
至少不止一根），「分不开」是**判断**（几块挨着 / 交叠 / 中间夹着混色列）；读的
人有权知道这一堆是哪一条判出来的。

**同一根上「列数」与「比例」用同一支口径**（用户 10-15 的裁定）：有缝（或者右边
没读到）⇒ **两个都是下界**；没缝 ⇒ **两个都精确**。**不许一个精确一个下界** ——
那两个数说的是同一件事（缝那几列到底算不算填充），口径不一致会让读的人以为
「列数我知道了、比例不知道」。**两条读数的余量因此也合成一个**：它们量的是同一个
门（严格那一段的右端）。

**判「一堆」用的是已经有的那两条**（不新立一套）：

1. **分不开**：同一条行带里相邻两块，填充段交叠 / 相接，或者中间夹着**混色**列
   （`_inseparable` —— 就是原来把条数降成下界的那件事）；
2. **填充总列数长过这一档的满条长**：一根条的填充不可能长过它那一档的满条长
   （`raw_b_上` 那 130 列就是这一支）。

**按族归组，不许跨族**：一堆里的成员**同一族**，否则「至少几根」要除以哪一档的
满条长就说不清。所以每一堆都带 `tier`（小怪 / boss）。

**「至少几根」= 算术下界**：`ceil(这一坨的填充总列数 / 那一档的满条长)`。
**条数的上界不从像素来** —— 像素只能给下界；上界由状态层拿**账**配（这一关还该有
在场几只）。**别在像素上找上界。**

**「这一坨的填充总列数」是那个读得准的数**（它换算过来就是「这一堆怪**加起来**
还剩多少血」，止损就靠它）。它**自己带形态**：干净 ⇒ **精确** · 右边被盖住 ⇒
**下界**。**「至少几根」跟着它算**，所以两边不打架：`count` 的下界**等于**逐根
那几根 + 各堆的「至少几根」之和。

**它的形态只看右端、不看缝**（与单根那两条读数**不一样**，这条差别是刻意的）：
这一坨的总列数是**止损要用的那个数**，它由 `FILL_GAP` 那道容差**把缝桥过去**、
把这一坨的填充**连起来**数；「右边还被盖住」才是它读不到头的那种不确定。单根那
两条读数**不许跨缝**（缝底下藏着什么看不见）—— 一个答「这一坨一共填了多少」、
一个答「这一根填了多少」，问的不是同一件事。

**一条读数各自一份形态与余量，但没有总分。** 每条读数（单根的列数 / 比例、一堆的
总列数 / 至少几根）各自落 精确 / 下界 / 区间 / 未采样 里合适的那一档，各自带一个
**余量**（「定下这次读数的两侧离判据的门有多远」，量出来的、单调的）。**这一层
没有「置信度 / 可信度 / 相似度」这种 0–1 总分**：叠条这一档全库只有两三帧，标不出
一个有意义的分数，而一个分数还会把「往哪个方向不确定」丢掉（「**至少** 90」与
「90 **上下差 3**」是两件事）。下游若只要一个标量，**让它自己从「形态 + 余量」算**。

**「至少几根」的余量取总列数那个读数的余量**（会动的东西只有总列数）；**一堆的
余量取组内各根 `columns` 余量的最小值**（口径与单根那一支同一个，见 `_clump`）。
**「至少几根」是拿一个下界去 `ceil` 得来的 ⇒ 它是「下界的下界」**：总列数自己可能
已经是个下界（`LowerBound`），再除、再向上取整，只会更保守。

**这一版产不出「区间」，这是缺口**：单根只产 精确 与 下界 两支（干净 / 有缝或右边
没到头），求和之后还是这两支 ⇒ `Interval` 那一档**没有来路** —— 单 7 把「软尾」那
一支连同量轨道整个退役了。**「两头模糊」这一档现在没有量法**（边界落在几列渐变里
那种真帧全库零样本）；将来真出现那样的帧再补，**不为了让它出现而新造一个判据**。

**分不开的一堆只算起 1 根**：`LowerBound(k)` 里的 `k` 是一堆一堆加起来的下界，
所以这一支**永远不高报**；**把分不开的报成精确正是这一支要挡的那件事**（条数报多
了，下游会以为还有更多的敌人；报精确了，下游就没法知道这个数只是个下限）。
**那道便宜的算术门的代价如实说**：填满那一档的条右邻若正好有几列「看起来像填充
色」的地形，那几列会把填充段读长，于是这一支也会报出 >= 2 —— 它只在**填充段本身
读长了**的时候误报，而那种时候画面自己也说不清那几列是不是填充（用户 10-15 认了
这个代价，不做更贵的形式）。

**「至少几根」的样本只到过 1 与 2**（全库：`在_raw_b_上` 2、互叠那两帧各 2、合成
用例 1）—— **`ceil` 到 >= 3 的样本零**，这是个缺口。而它的**上界不在这层写**：
像素只给下界，上界由**状态层拿账配**（这一关还该在场几只）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import cv2
import numpy as np

from ark.contracts import (
    Exact,
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
    "ClumpRoute",
    "EnemyClump",
    "EnemyBars",
    "Reading",
    "BarTier",
    "MOB_TIER",
    "BOSS_TIER",
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
    "MOB_FILL_COLUMNS",
    "BOSS_FILL_COLUMNS",
    "MOB_BODY_ROWS",
    "BOSS_BODY_ROWS",
    "CROSS_TIER_ROWS",
    "FILL_GAP",
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
"""形状门那一步「竖着连成几行」：**8**。它同时是**小怪档**的条身行数。

**为什么门取 8**：小怪那一档的条实测就是 8 行（`MOB_BODY_ROWS`），而地貌那些
「红扁条」在颜色门里只剩 4…6 行 —— 8 正好掐在这一段空档上。**boss 那一档是 10 行**
（`BOSS_BODY_ROWS`），它照样过得了这道门（连通块高 10 <= `MAX_BAR_ROWS`，
`_band_row` 在那 10 行里滑一个 8 行的窗）。所以这一道门**不是**「一根条有多高」
的判据，只是「竖着连不够长的整片消失」。
"""
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
"""**「这一块小得可疑」那道门**：填充段短于它的候选**照样进逐根那一栏**，
列数与比例**都照给**；只是**多一个「小尺寸标记」**（`EnemyBar.short_fill`）。

真条实测最窄 16 列（`special_黄色菱形...png` 那一块），线取 12 是留了 4。
**它不是一道读数的门**（用户 10-15 的裁定）：从前它把短填充的**比例**抹成
未采样，那等于用降级盖住已经知道的东西 —— 7 列是**读出来了**的
（`在_快死的敌人` 那一根真帧：7 列 ⇒ 7/123 = 0.0569 精确）。**够不够格当一根条**
这件事本来就已经被主张过了（它过了颜色门、形状门、候选宽度门），交给那个标记
带出去；**残余风险如实说**：这一块**仍可能是假阳性**，标记就是为它留的。
`raw_b` 里那块 9 列、色距 8 的假阳性**不是这条线挡的，是上面那道矩形门挡的**。
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
MOB_FILL_COLUMNS: Final[int] = 123
"""**小怪那一档**的满条长（列）。比例的分母：填充段列数 / 123。

**谁算小怪**：**所有小怪** —— 含精英、经验怪（用户 10-15 的括注）。用哪一档
**看族**（橙族 = 小怪档），**不看类型**。

**它是自己反推出来的**（规划者没把数发下来），三条独立的手：

1. **一根满条**：`screenshots/_probe.png` x648 y395 那根填了 **123 列**，而紧挨着的
   下一列 **x771 就是背景本身** —— (43,47,50)，与它正下方那段地形 (41,44,46)
   逐通道差 <= 4。**这根条没有「没填到的那一段」**；
2. **全库的天花板**：`screenshots/**` 与 `data/precheck/**` 里 **167 张 2800x1260**
   的帧整张筛一遍，橙族一共 **131 块**，填充列数的顶上是 **123 列 × 11 块**，横跨
   5 帧：`_probe.png` x648/x1790/x2088/x1630、`run-01` 那三帧 shot_...202939-929
   x1989/x1039、shot_...202959-919 x2262/x2373、shot_...203049-920 x1069/x2101、
   以及 `special_带弹药机制的有竖线间隔的橙色技能条.png` x2083。比它长的只有两块，
   **都不是条**：154 列（`敌条互叠1` x1628，两根的填充接在一起）与 125 列
   （`hard_多个技能条紧挨着` x2019，餐厅界面里的橙条）；
3. **同一档「没填到的那一段」**：`run-06/快死的敌人` x1506 那根填充只有 **7 列**，
   它后面那段带子（比上下两块地形都暗 >= 8）**到 x+122 为止** —— 正好 123 列。

**第 2 条里那 11 块「到头了」怎么判**：9 块的**下一列就是背景**（与上下地形差
<= 10，多半 <= 5）；剩下 2 块（`_probe.png` x2088 / x1790）的下一列确实更暗，但把
那一段的纵向剖面拉出来看，**暗区不是只占条身那 8 行**（x2088 是 12 行、往条身下面
多出 4 行，混进地形自己的暗区），x1790 是地形自己的亮/暗交界 —— **都不是「没填到的
那一段」**。若这一档真长过 123，那 11 根填到 123 的条**每一根**都该在 x+123 起露
出一段暗带；实际只在 2 根上看到，而那两处都有别的解释。

**交叉验证**：`raw_a` 那两根的填充段都是 **84 列** ⇒ 84/123 = **0.6829**，与画面上
量出来的约 0.683 对上。
"""

BOSS_FILL_COLUMNS: Final[int] = 154
"""**boss 那一档**的满条长（列）。红族比例的分母：填充段列数 / 154。

**用哪一档看族**：红族 ⇒ boss 档（用户 10-15 的裁定 —— 也就是 10-14 那条「红条
只有 boss」的另一面）。**这个数按用户 (c) 记账** —— 认，但把来路记清楚：

1. **红族全库只有这一根真值**：167 帧里红族的真条就 `screenshots/run-06/boss.png`
   x1562 那一根（另有 4 块是红门上的地形碎块、一块是界面的红条，都不算）；
2. **那次标定借了「量轨道」这条已经退役的路**：那根红条**不是满的**（填充 115 列），
   所以要拿它右边那段暗带 (45,44,42) 的右端当满条长 —— 那段带子到 x1715 为止、
   条从 x1561 起（填充从 x1562 起）⇒ **154**；
3. **`123 × 1.25 = 153.75` 与 `条身 8 行 × 1.25 = 10 行` 是同一条旁证**、而且落在
   同一根条上，**不是两次独立验证**。

⇒ **将来出现第二根 boss 条、或者一根满的红条时复核这个数。运行时不量轨道**：
红族比例 = 填充列数 / 154。
"""

MOB_BODY_ROWS: Final[int] = 8
BOSS_BODY_ROWS: Final[int] = 10
"""两档的**条身行数**：小怪 8 行、boss 10 行（`_body_rows` 量出来的）。
**小怪 8 行**：全库筛出来的每一块小怪条都是 8 行（`raw_b` 那一带有两块量到 **9 行**，
那是抗锯齿多出来的一行 —— 它在新的门下面**过得了**，见下）。

**boss 10 行**：`boss.png` 那根红宽条 y1002…1011 —— 10 行。它与 `BOSS_FILL_COLUMNS`
的 **1.25 倍**是**同一条旁证**（都在同一根条上），别当成两次独立验证。

**它们是一道便宜的跨档校验**（用户 10-15 给的查法）：**选档仍按族**，可某一根
**量到的条身行数够得着更高那一档**时，那一根**不给比例**（落未采样）—— 防的是
「橙条 boss」那一类：一根橙色的条若其实是 boss 档（10 行 / 154 列），拿橙族的
123 当分母就会系统性报高。

**门对着 boss 那一档设**（用户 10-15 的裁定）：小怪那一档**量到 >= `BOSS_BODY_ROWS`
（10 行）才拦**，**9 行及以下放过** —— 定档靠**族**（橙 ⇒ 小怪 · 红 ⇒ boss），
行数只是护栏；拿「比 8 大」当门会把抗锯齿与叠条压出来的 9 行全拦掉，而 9 不等于
10。**这道门只设在小怪那一档** —— 红族**不设**（用户 10-15 的裁定）：它本来就是最高
那一档、没有更高的档可比，而这道门只为防「橙条 boss」。

**代价记账**（写进夹具 README）：万一真有一根 **9 行的橙条其实是 boss**（10 行被
量少了一行），我们会按 123 算 ⇒ **比例偏高约 25%**；要消除它，得有**第二根 boss
条样本**来看「行数会不会被量少」。这一支**只降不升**，不会把「不知道」写成「知道」。
**只降不升**，不会把「不知道」写成「知道」。
"""

FILL_GAP: Final[int] = 2
"""填充段里最多连续放几列不亮（抗锯齿列、一根窄遮挡）。

**它现在只服务「填充段到哪儿为止」这一件事** —— 原来这个数与空轨共用，空轨退役
之后只剩填充段这一头。`raw_b` 上那根条正中间就有 **1 列**被压暗（x1577 上那条
竖线）—— 所以容差至少要 1。

**为什么是 2 而不是更大**：3 起，地形自己的杂色（一根条右邻那片地上零星的「看
起来像条色」的列）会把填充段一路推出去 —— 原来量整长的时候实测 `raw_a` 那两根
会从 124/125 推到 131，而它们画面上的填充段只有 84 列。**那两个数都是从旧那条路
上量出来的**，这里如实记着（空轨那一头已经退役）。
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
class BarTier:
    """血条长度的那一档。**一堆要能看出它是哪一档**，靠的就是这个。

    | 档 | 谁 | 满条长（列） | 条身（行） |
    | --- | --- | --- | --- |
    | **小怪档** | **所有小怪**（含精英、经验怪）—— 橙族 | `MOB_FILL_COLUMNS` | `MOB_BODY_ROWS` |
    | **boss 档** | **只有 boss** —— 红族 | `BOSS_FILL_COLUMNS` | `BOSS_BODY_ROWS` |

    `full_columns` 是比例的分母（按族查表，**不来自画面**）；`body_rows` 是这一档
    条身的行数（跨档校验用）。**用哪一档看族、不看类型**（见 `_Family`）。
    两档常量各自的来路写在它们自己的 docstring 里。
    """

    name: str
    full_columns: int
    body_rows: int


MOB_TIER: Final[BarTier] = BarTier("小怪", MOB_FILL_COLUMNS, MOB_BODY_ROWS)
BOSS_TIER: Final[BarTier] = BarTier("boss", BOSS_FILL_COLUMNS, BOSS_BODY_ROWS)
"""两档常量（用户 10-15 定稿：执行者反推 + 规划者独立复核）。

**它们的来路**在 `MOB_FILL_COLUMNS` 与 `BOSS_FILL_COLUMNS` 的 docstring 里 ——
boss 那一档按用户 (c) 记账（红族全库只有一根真值、且那根不满）。
"""


CROSS_TIER_ROWS: Final[int] = BOSS_BODY_ROWS
"""跨档校验那道门：**对着 boss 那一档的条身行数设**（用户 10-15 的裁定）。

**选档仍按族**（橙 ⇒ 小怪 · 红 ⇒ boss）；这一道门只防「**橙条 boss**」那一类。
小怪那一档**量到 >= 10 行（= `BOSS_BODY_ROWS`）才拦** —— **9 行及以下放过**：
9 是抗锯齿与叠条压出来的，9 不等于 10。**红族不设这道门**（用户 10-15 的裁定）：
它本来就是最高那一档、没有更高的档可比，而这道门只为防「橙条 boss」。见 `_over_tier`。
"""


@dataclass(frozen=True)
class Reading:
    """一条读数：**值（形态就长在值自己身上）** + **余量**。

    `value` 是 `SlotValue`：`Valued(Exact / LowerBound / Interval)`，或者
    `Unsampled()`（这一条读不出来）。

    `margin` 是「**定下这次读数的两侧离判据的门有多远**」（0…1，量出来的、单调的，
    口径与 `_fill_confidence` 一模一样）—— **不是概率、不是置信度、不是总分**：
    `Valued.confidence` 里带的是同一个数（那是原语算出的连续量），这里再摆一份是
    因为下游要拿它算自己的阈值。`None` = 这一条不是读出来的（比如「至少几根」是
    除法算的）。
    **一条读数各自一份余量**（用户 10-15 的裁定 3）：单根上「填充列数」与「比例」
    各有各的一份。**单 9 起两者的口径合成一支**（同一根上一个精确一个下界被用户
    否掉了），所以同一根上那两份余量**是同一个数** —— 它们量的是同一个门（严格
    那一段的右端）；`_clump` 取的那一份也是它。
    各有各的余量 —— 有缝的那一根两者不一样（见 `_measure`）。

    **这一层没有 0–1 的总分**（理由见模块开头「三样读法」那一节）。
    """

    value: SlotValue
    margin: float | None


@dataclass(frozen=True)
class ScreenedCore:
    """筛出来的一块**填充段**（候选那一步量到的）。

    - `fill_rect`：这一块**填充段**的矩形（左上角 + 填充列数 + `BAR_ROWS` 高）。
      **名字里那个 fill 是承重的**：空轨退役之后，这一块的宽就是**填充段列数**，
      把它叫 `rect` 会让人以为它是「整根条」；
    - `distance`：它的**中位色距**（逐列中位色离这一族参考色的逐通道最大差，
      取这一段的中位）；
    - `masked`：它是不是**被我方条的位置挡掉**的那一块（调用方给的 `our_bars`）。

    `masked` 的块留在读数里、不进 `EnemyBars.bars`：**「那里有东西但那是我们自己的」
    与「那里什么都没有」是两件事**，混掉就是把排除做成了否认。
    """

    fill_rect: PixelRect
    distance: int
    masked: bool

    @property
    def score(self) -> float:
        """这一块的分数：1 - 颜色距 / 255（与那两条线同一个量纲）。"""
        return max(0.0, min(1.0, 1.0 - self.distance / 255.0))


@dataclass(frozen=True)
class EnemyBar:
    """**逐根**读出来的一根：填充段本身 + 填充列数 + 由它算出的比例。

    `fill_rect` 是**填充段**（原点 + 填充列数 + 量到的条身行数）—— **不是整根条**。
    这一版的比例是「填充段列数 ÷ 那一档的满条长」，**分母是常量、不来自画面**（见
    模块开头「比例」那一节），所以这里没有「整长」这个东西可报。

    `columns` 与 `fill` **各是一条读数**（各自带形态与余量，见 `Reading`），
    **两条用同一支口径**（用户 10-15 的裁定）：

    - 干净（没有缝、右边也读到了）⇒ `columns = Exact(填充段列数)`、
      `fill = Exact(填充段列数 / 满条长)`；
    - 有缝、或者右边被盖住 ⇒ **两条都落 `LowerBound`**，值取**严格那一段**
      （一格缝都不跨的那一截）。

    **`short_fill`** 是那个**小尺寸标记**（用户 10-15 的裁定）：填充段短于
    `MIN_FILL_COLUMNS` 时为 `True` —— 读数是**精确的**（那几列真的读出来了），
    标记说的是**另外一件事**：「这一块够不够格当一根条」没把握。它**不是一条读数**：
    不带形态、不带余量，也不改这一根落在哪一栏。

    `fill_rect` 是**量出来的那一整段填充**（跨得过 `FILL_GAP` 列缝），所以有缝的
    那一根它的宽比 `columns` 那条读数**大**：框画的是**量到的那一段**，读数给的是
    **下界**（严格那一段）—— 两个数问的不是同一件事。

    `fill_rect.height` 是**量到的条身行数**（`_body_rows`），不是常量 `BAR_ROWS`：
    一根量到 10 行的条报成 8 行，就是拿写死的数去描述一个量出来的东西。

    `distance` 是这一块的中位色距（读数与核对都要它）。

    **`fill` 落 `Unsampled()` 的那一根不进 `bars`、进 `unsampled`**（见 `EnemyBars`）：
    「这一根有东西、但填了多少读不出来」与「这一根读出来了」是两件事。这一版只剩
    **一条**来路：条身行数到了更高那一档的行数（跨档校验）。
    """

    fill_rect: PixelRect
    columns: Reading
    fill: Reading
    distance: int
    short_fill: bool


@dataclass(frozen=True)
class ClumpRoute:
    """这一堆是**哪一条**判出来的（用户 10-15 的裁定）：两条证据强度不同。

    - **长过一根的上限**：填充总列数长过这一档的满条长 —— 这是**算术**
      （一根条不可能填得比自己长，所以那里至少不止一根）；
    - **分不开**：同一条行带里相邻几块的填充交叠 / 相接 / 中间夹着混色列 ——
      这是**判断**（`_inseparable` / `_touching`）。

    两条可以**同时**成立（`两者`）：比如两块挨着的填充加起来长过满条长。
    读的人有权知道哪一条在承重，所以这一栏落进 `判据读数.json`。
    """

    inseparable: bool
    over_full: bool

    @property
    def name(self) -> str:
        """这一堆的来路：`分不开` / `长过一根的上限` / `两者`。"""
        if self.inseparable and self.over_full:
            return "两者"
        if self.inseparable:
            return "分不开"
        return "长过一根的上限"


@dataclass(frozen=True)
class EnemyClump:
    """**一堆**：挤在一起、拆不开的那一坨 —— 当成**一件事**报出来（用户 10-15 的裁定）。

    - `region`：这一坨**占画面哪一块**（组里每一根**填充段**的外接框，填着的那一块，
      不含空轨）；
    - `tier`：**哪一档**（小怪 / boss）—— 「至少几根」要除以哪一档的满条长，所以这
      一栏不能省。**按族归组、不许跨族**正是为了这一栏说得清；
    - `columns`：这一坨的**填充总列数** —— 这是**读得准**的那个数，它换算过来就是
      「这一堆怪**加起来**还剩多少血」（止损就靠它）。它**自己带形态**：干净 ⇒
      `Exact` · 右边被盖住 ⇒ `LowerBound`（见模块开头「三样读法」）。**它的形态
      只看右端**：这一坨的总列数由 `FILL_GAP` 那道容差**把缝桥过去**连起来数 ——
      与单根那两条读数**不许跨缝**是两件事（一个答「这一坨一共填了多少」、一个答
      「这一根填了多少」）；
    - `least`：**至少几根** = `ceil(填充总列数 / 这一档的满条长)`，落 `LowerBound`
      —— 它**只是下界**，而且是**下界的下界**（总列数自己可能已经是 `LowerBound`）。
      **上界不从像素来**：上界由状态层拿**账**配（这一关还该有在场几只）；
    - `route`：**来路**（`ClumpRoute`）—— 分不开 / 长过一根的上限 / 两者。
    - **余量**取**组内各根 `columns` 余量的最小值**（最弱的那一环就是这一堆的
      余量），见 `_clump`。

    **拆不开就不给单根比例**（硬给就是编）：一堆里没有「逐根」那一栏。
    「一堆」与「未采样」是两件事：未采样 = 那里有东西、连填充都读不出来；
    一堆 = 知道那一坨在哪、多大、至少几只。
    """

    region: PixelRect
    tier: BarTier
    columns: Reading
    least: Reading
    route: ClumpRoute


@dataclass(frozen=True)
class EnemyBars:
    """这一帧的读数：**三个出口** + **条数** + **三样读法** + 被我方条挡掉的那些块。

    `presence` 是**这一帧画面上看不看得见敌人血条**那个布尔（三态）：
    `Valued(Exact(True))` 在 / `Valued(Exact(False))` 不在 / `Unsampled()` 认不出。
    **三栏里只要有一栏非空就是「在」**。

    **三样一眼分得开**（用户 10-15 的裁定 1）：

    | 栏 | 是什么 | 带什么 |
    | --- | --- | --- |
    | `bars` | **逐根**读得出来的 | 填充段列数 + 比例（各自带形态与余量） |
    | `clumps` | **一堆**（挤在一起、拆不开） | 区域 + 至少几根 + 这一坨的填充总列数 |
    | `unsampled` | 那一根确实有东西、但**填了多少读不出来** | 位置 + 那两项读数 |

    三栏**互斥**、合起来正好是颜色说「是条」的那些块：分不开的整坨进 `clumps`，
    其余逐根进 `bars`（比例读出来了）或 `unsampled`（没读出来）。**一堆里没有
    单根比例。**

    `count` 是**这一帧有几根的下界**：

    | 情形 | 落法 |
    | --- | --- |
    | 一堆都没有 | `Valued(Exact(len(bars) + len(unsampled)))` |
    | 有一堆（或更多） | `Valued(LowerBound(逐根 + 各堆的「至少几根」之和))` |
    | 这一帧认不出 | `Unsampled()`（与 `presence` 同一个出口） |

    **两边不打架**（用户 10-15 的裁定 4）：那个下界**等于**逐根那几根 + 各堆的
    「至少几根」之和。**它永远不高报**：分不开的一堆只按
    `ceil(填充总列数 / 满条长)` 贡献，而那个除法只用得上「总列数」这一个数。
    """

    presence: SlotValue
    count: SlotValue
    bars: tuple[EnemyBar, ...]
    clumps: tuple[EnemyClump, ...]
    unsampled: tuple[EnemyBar, ...]
    cores: tuple[ScreenedCore, ...]


def screened_cores(
    frame: Frame, *, our_bars: Sequence[PixelRect] = ()
) -> tuple[ScreenedCore, ...]:
    """把这张战场**一次筛完**，给回每一块候选（不量比例、不落三态）。

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

    **三样读法**（见模块开头「三样读法」那一节）：颜色说「是条」的每一块，最后要么
    进 `bars`（逐根读出来了）、要么进 `clumps`（拆不开的那一坨）、要么进
    `unsampled`（有东西、但填了多少读不出来）。分不开的那几根和长过满条长的
    那一根**整坨进 `clumps`**（`_is_clump`）；其余逐根进 `_bar`。三栏互斥。

    `count` 是**条数的下界**（见 `EnemyBars`）：一堆都没有时才给 `Exact`，
    否则 `LowerBound(逐根 + 各堆的「至少几根」之和)`；这一帧认不出时落
    `Unsampled()`（与 `presence` 同一个出口）。

    `our_bars` 是**我们自己那两根条的位置**（每条给血条那块矩形，144/146 宽、
    8 行高都行 —— 这一层按「血条 + 紧贴它下面第 8 行的技力条」那 16 行解释它）。
    被我方条挡掉的块**不进这三栏、也不参与这个布尔**，但留在 `cores` 里。

    **取区越出画布 -> 抛异常**（能力缺失，不是「看不到」）。
    """
    region, top = _field(frame)
    measured: list[_Measured] = []
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
            # 量的是**填充段**（怎么起、怎么止见 `_measure`）：「短填充 / 长过满条长 /
            # 条身比这一档更高」三支分别在 `_bar` 与 `_clump` 里落定 —— 塞在这一层
            # 做，就得把它量到的那几行再算一遍。
            measured.append(_measure(region, core))
        elif core.distance < ABSENT_TOL:
            middle = True
    measured.sort(key=lambda item: (item.core.y, item.core.x))
    grouped = _inseparable(region, measured)
    groups: dict[int, list[_Measured]] = {}
    for index, item in enumerate(measured):
        groups.setdefault(grouped[index], []).append(item)
    bars: list[EnemyBar] = []
    clumps: list[EnemyClump] = []
    unsampled: list[EnemyBar] = []
    counted = 0
    for root in sorted(groups):
        members = groups[root]
        if _is_clump(members):
            # 拆不开／长过满条长的一坨：**整坨进一堆**，里面那几根不单独报比例。
            clump = _clump(top, members)
            clumps.append(clump)
            counted += int(clump.least.value.payload.lo)  # type: ignore[union-attr]
            continue
        # 这一组只有一根（不止一根的都在上面那一支整坨进了一堆）：比例读得出来就进
        # `bars`，读不出来就进 `unsampled` ——「有东西、但填了多少读不出来」与
        # 「读出来了」是两件事。**按成员走而不是只取第一个**：只取第一个的话，
        # 「分组说它们分不开、`_is_clump` 却说不是一堆」这种自相矛盾的状态会**静悄悄
        # 吞掉几根**（红证那条用例把这种状态摆出来过）。
        for item in members:
            bar = _bar(top, item)
            counted += 1
            if isinstance(bar.fill.value, Unsampled):
                unsampled.append(bar)
            else:
                bars.append(bar)
    if bars or clumps or unsampled:
        presence: SlotValue = Valued(Exact(True), best)
    elif middle:
        presence = Unsampled()
    else:
        presence = Valued(Exact(False), best)
    # 条数是**几何分组 + 一道除法**的结论，不是颜色挣来的 —— 所以置信度写 None
    # （§1：None = 不是读来的），不把颜色那一头的分数挪过来冒充它。
    if not bars and not clumps and not unsampled:
        count: SlotValue = Unsampled() if middle else Valued(Exact(0), best)
    elif clumps:
        count = Valued(LowerBound(counted), None)
    else:
        count = Valued(Exact(counted), None)
    return EnemyBars(
        presence=presence,
        count=count,
        bars=tuple(bars),
        clumps=tuple(clumps),
        unsampled=tuple(unsampled),
        cores=tuple(screened),
    )


# ---------------------------------------------------------------------------
# 战场那一条
# ---------------------------------------------------------------------------


def _field(frame: Frame) -> tuple[np.ndarray, int]:
    """战场那一条的像素 + 它在画布上的原点行。

    上下各多留 8 行（`BAR_ROWS`）：量条身时得拿「带子上方那几行」当底，
    贴着战场边的那一根条不该因为**量不到头**而读不出来。**判据只认战场那一段
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
        fill_rect=PixelRect(core.x, core.y + top, core.width, BAR_ROWS),
        distance=core.distance,
        masked=_is_ours(core, our_bars, top),
    )


# ---------------------------------------------------------------------------
# 一次筛完
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Family:
    """一条**色族**。族同时定下它那一档的三个数（用户 10-15：**看族、不看类型**）。

    - `reference`：这一族的填充色（橙 `ENEMY_FILL` / 红 `ENEMY_FILL_RED`）；
    - `min_columns`：这一族候选的**宽度门**（橙 7 / 红 `RED_MIN_COLUMNS`）；
    - `tier`：这一族**是哪一档**（`MOB_TIER` / `BOSS_TIER`）—— 满条长（比例的分母）
      与条身行数（跨档校验）都在它里面。**一堆报出去的那一档就是它。**
    """

    reference: tuple[int, int, int]
    min_columns: int
    tier: BarTier

    @property
    def full_columns(self) -> int:
        """这一族满条长的列数（比例的分母）。"""
        return self.tier.full_columns

    @property
    def body_rows(self) -> int:
        """这一族条身的行数（跨档校验用）。"""
        return self.tier.body_rows


_ENEMY_FAMILY: Final[_Family] = _Family(ENEMY_FILL, MIN_CANDIDATE_COLUMNS, MOB_TIER)
_BOSS_FAMILY: Final[_Family] = _Family(ENEMY_FILL_RED, RED_MIN_COLUMNS, BOSS_TIER)
_FAMILIES: Final[tuple[_Family, ...]] = (_ENEMY_FAMILY, _BOSS_FAMILY)
"""两族各筛一遍（不并掩膜的理由见 `_screen_family`）。**族到档的对应只有这一处。**"""


@dataclass(frozen=True)
class _Core:
    """私有的那一块：**帧内坐标**（行相对战场那一条的顶，列就是画布列）。

    `width` 是在**那 8 行**上量到的**填充段列数**（容 `FILL_GAP` 列不亮的连续亮列）
    —— 它只做两件事：过**候选宽度门**、给 `_measure` 的扫描窗垫底；**报出去的那个数
    由 `_measure` 在量到的条身那几行上重新量**（见 `_Measured.columns`）。
    `family` 是这一块走的族，分母与条身行数都在它里面。
    """

    x: int
    y: int
    width: int
    distance: int
    family: _Family


@dataclass(frozen=True)
class _Measured:
    """一根候选**量出来的全部东西**（`_measure` 的产物，私有的）。

    它是「一堆」与「逐根」两支共用的原料：分组（`_inseparable`）只看 `core`
    与 `columns`，报出去的那些数由 `_bar` / `_clump` 从这里取。

    - `columns`：量到的**填充段列数**（跨得过 `FILL_GAP` 列不亮的那一段）——
      它**只是量出来的那个数**；报出去的那条读数落精确还是下界，由 `_bar` 定；
    - `strict`：**缝前那一截**（一格缝都不跨的那一段）—— 缝底下藏着什么看不见，
      所以**两条读数**（列数与比例）落 `LowerBound` 时用的都是它；
    - `covered`：这一段填充的**右端没读到**（右边还接着亮）；
    - `margin`：**这一根的那个余量**（用户 10-15 的裁定）。**两条读数合成一个**：
      口径统一之后，决定列数与比例的是**同一个门**（严格那一段的右端）—— 从前
      那个「列数按容忍那一段算余量」的口径会量出一个已经不是这次读数的边界，
      所以删掉了；
    - `row` / `rows`：**量到的**条身（顶行 + 行数），不写死 `BAR_ROWS`。
    """

    core: _Core
    columns: int
    strict: int
    covered: bool
    margin: float
    row: int
    rows: int

    @property
    def clean(self) -> bool:
        """这一段填充两头都干脆：没有缝、右边也没接着亮。"""
        return self.strict == self.columns and not self.covered


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
    found: list[_Core] = []
    for family in _FAMILIES:
        found.extend(_screen_family(region, top, family))
    found.sort(key=lambda core: (core.y, core.x))
    return _merge(found, region)


def _screen_family(
    region: np.ndarray,
    top: int,
    family: _Family,
) -> list[_Core]:
    """**一族颜色**单独筛一遍 -> 这一族里够格当候选的块。

    **两族各筛一遍、不并成一个掩膜**：并起来之后，两族在画面上挨着的像素会把
    本来分开的块连成一块 —— 实测 `敌条互叠2` 那一帧会因此把 7 根条并进别的块里、
    那些块的高度涨过 `MAX_BAR_ROWS` 而**整批丢掉**。那是**在改量法**，不是加一族
    颜色；分成两遍，橙族那一边的量法与加红族之前**逐块一样**。

    这一族走**哪一档**由传进来的 `family` 定：参考色、候选宽度门、满条长（分母）、
    条身行数都在它里面（见 `_Family`）。
    """
    pixels = region.astype(np.int16)
    reference = np.asarray(family.reference, np.int16)
    mask = np.abs(pixels - reference).max(axis=2) <= SCREEN_TOL
    tall = _long_runs(mask, BAR_ROWS).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(tall, connectivity=8)
    found: list[_Core] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if height > MAX_BAR_ROWS or width < family.min_columns:
            continue
        row = _band_row(mask, y, height, x, width)
        if row < 0 or not _inside_field(row, top):
            continue
        block = mask[y : y + height, x : x + width]
        band = region[row : row + BAR_ROWS, x : x + width]
        if not _rectangular(band, block, area, width):
            continue
        profile = np.median(band, axis=0)
        distance = np.abs(profile.astype(np.int16) - reference).max(axis=1)
        lit = distance <= SCREEN_TOL
        tolerant = _leading_run(lit, FILL_GAP)[1]
        # **这里原来还有一道 `tolerant < family.min_columns` 的宽度检查 —— 删掉了**
        # （用户 10-15 的裁定 6：恒真就删）。实测它是**恒真的**：全库 167 张
        # 2800x1260 整张筛一遍，过得了形状门的候选 **140 块**，量到的 `tolerant`
        # 最小 **7**（= `MIN_CANDIDATE_COLUMNS`，橙族那一支），那一道**一次都没触发**。
        # 它的内容由上面 `width < family.min_columns` 那道门管住了：连通块的宽与这
        # 一段填充的宽是同一个量级（`tolerant` 只会更长，因为它容得下 `FILL_GAP`
        # 列不亮）。
        found.append(
            _Core(
                x=x,
                y=row,
                width=tolerant,
                distance=int(np.median(distance[:tolerant])),
                family=family,
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
                    family=head.family,
                )
            )
        else:
            merged.append(core)
    return tuple(merged)


def _attached(left: _Core, right: _Core, region: np.ndarray) -> bool:
    """右边那一块是不是**左边那一根条被压过去之后接着露出来的那一截**。

    **跨族不接**：红的与橙的不可能是一根条的两截（`_screen` 把两族筛出来的块放在
    一张表里 `_merge`，所以这一道门得有）。`_blended` 那一支比的是橙族的参考色，
    红条混色那一头会答成「不是混色」—— 两道加起来，红核之间既不接也不并组
    （保守：各算一根，不高报）。
    """
    if left.family != right.family:
        return False
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

    **为什么不比「与地形的差」**：**压在一条填充色上面的半透明遮挡**压出来的混色，
    亮度可能正好与它下面那块地形接近 —— 实测 `raw_b` 那条 1 列宽的缝（x1577）就是：
    它与上下地形差不了多少（**4.2**），可它离填充色 **64**、离上下地形 **99 / 112**。
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

    `_leading_run` 连着碰到 `FILL_GAP + 1` 个假列就停下了，所以它停下的**那一侧**
    只要还有亮列，就是缝底下压着东西、条并没有到此为止。
    """
    return bool(lit[filled : filled + MERGE_GAP].any())


# ---------------------------------------------------------------------------
# 填充段与比例
# ---------------------------------------------------------------------------


def _body_rows(region: np.ndarray, core: _Core) -> tuple[int, int]:
    """这一根条**量到的**条身 -> (条身顶行, 条身高)。**不许写死 8**。

    自 `core.y` 起上下扩：**填充段那几列**上，条身以内是这一族的填充色、以外不是
    —— 走到不亮为止。封顶 `MAX_BAR_ROWS`（形状门已经保证那一块不高于它）。

    实测两档：小怪与红门那些整齐的块是 **8 行**（`MOB_BODY_ROWS`），boss 那根红条是
    **10 行**（`BOSS_BODY_ROWS`；`在_boss_红宽条` x1562 的 y1002…1011），`raw_b` 上
    那根是 **9 行**（抗锯齿多出来的一行）。

    **它同时是「取哪几行」与「跨档校验」的输入**：量到的条身决定逐列比色取哪几行、
    也决定 `fill_rect.height` 报几行；它**比这一档更高**时 `_bar` 就把这一根落成
    `unsampled`（用户 10-15 的裁定 6：**只拦更高的那一头**，矮一点的不拦）——
    防的是「一根橙色的条其实是 boss 档（10 行 / 154 列）」，拿 123 当分母会系统性
    报高。**它只降不升。**
    """
    reference = np.asarray(core.family.reference, np.int16)
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


def _measure(region: np.ndarray, core: _Core) -> _Measured:
    """一根候选 -> **它量出来的全部东西**（见 `_Measured`）。

    **量法一个字节都没变**（与我方血条那条链同一套逐列比色 + 容差）：

    - 条身取**量到的行数**（`_body_rows`，不写死 8）；
    - 扫描窗取「这一档的满条长 + `FILL_GAP` + 1」：填充段**不可能**长过满条长，
      所以看到窗底还亮着就已经不止一根了（那一支由 `_is_clump` 接走）；
    - `columns`（容得下 `FILL_GAP` 列不亮的那一段）是**这一段的填充总列数** ——
      缝底下是半透明遮挡压出来的，那几列算在里头；
    - `strict`（一格缝都不跨的那一段）是**两条读数**落 `LowerBound` 时用的值
      （列数与比例同口径，见 `_bar`）；
    - `covered` 是「这一段填充的右边**还接着亮**」⇒ 没读到右端。

    **余量只有一个**（用户 10-15 的裁定）：两条读数统一到同一支口径之后，决定列数
    与比例的是**同一个门** —— 严格那一段的右端（`_fill_confidence` 量的是「里头
    最弱的一列」与「右边那一列」哪个离门更近）。从前那两支（列数按容忍那一段、
    比例按缝前那一段）在**有缝的那一根上会分开**，而列数报的已经不是容忍那一段
    了 —— 那个口径删掉了。

    比色用的是**这一根自己那一族的参考色**（`core.family.reference`）：红族那根比的
    是 `ENEMY_FILL_RED`，不是橙族那一支 —— 比错了它整根都不亮、填充段会量成 0。
    """
    family = core.family
    row, rows = _body_rows(region, core)
    span = max(core.width, family.full_columns) + FILL_GAP + 1
    profile = np.median(region[row : row + rows, core.x : core.x + span], axis=0)
    distance = np.abs(
        profile.astype(np.int16) - np.asarray(family.reference, np.int16)
    ).max(axis=1)
    lit = distance <= SCREEN_TOL
    strict, columns = _leading_run(lit, FILL_GAP)
    return _Measured(
        core=core,
        columns=columns,
        strict=strict,
        covered=bool(_resumes(lit, columns)),
        # 两条读数共用这一个余量（口径见 `_Measured.margin`）：决定列数与比例的
        # 是同一个门 —— 严格那一段的右端（`min(strict, columns)` 就是 `strict`）。
        margin=_fill_confidence(distance, min(strict, columns)),
        row=row,
        rows=rows,
    )


def _bar(top: int, item: _Measured) -> EnemyBar:
    """**单出来的一根** -> 逐根那一栏的一个读数。

    | 情形 | 两条读数 |
    | --- | --- |
    | 条身行数到了更高那一档的行数 | `fill = Unsampled()`（跨档校验；**列数照量**） |
    | 干净 | 列数与比例**都** `Exact` |
    | 有缝、或者右边没到头 | 列数与比例**都** `LowerBound(严格那一段 ...)` |

    **同一支口径**（用户 10-15 的裁定）：有缝 ⇒ 两个都是下界、没缝 ⇒ 两个都精确。
    **不许一个精确一个下界** —— 那两个数说的是同一件事（缝那几列到底算不算填充）。
    **「长过 123 ⇒ 不止一根」那道门照旧可以用「把缝桥过去之后的总列数」**（见
    `_is_clump`）：它是**症状探测**，不是读数，所以不受这一支影响。

    **短填充照样进这一栏**（用户 10-15 的裁定）：列数与比例**都给精确值**，风险
    由 `short_fill` 那个标记带出去（填充段短于 `MIN_FILL_COLUMNS` ⇒ True）。

    **切断了为什么只给下界**：与我方血条那条链的 `LowerBound` 同一套纪律，
    不另立一套。
    """
    core = item.core
    family = core.family
    if item.clean:
        columns_value: SlotValue = Valued(Exact(item.columns), item.margin)
        fill_value: SlotValue = Valued(
            Exact(item.columns / family.full_columns), item.margin
        )
    else:
        columns_value = Valued(LowerBound(item.strict), item.margin)
        fill_value = Valued(
            LowerBound(item.strict / family.full_columns), item.margin
        )
    if _over_tier(item):
        # 跨档校验没过（量到的条身够得着更高那一档，见 `_over_tier`）：
        # 存在性照报，**填了多少不给** —— 这一根落 `unsampled`（由 `read_enemy_bars`
        # 分栏）。**列数照量**：那件事读出来了。
        fill_value = Unsampled()
    return EnemyBar(
        # 填充段本身：原点 + 填充段列数（在**量到的条身那几行**上量的）+ 量到的
        # 条身行数。写死 BAR_ROWS 去描述一根量到 10 行的条，就是拿常量冒充事实；
        # y 也跟着量到的那一行走。
        fill_rect=PixelRect(core.x, item.row + top, item.columns, item.rows),
        columns=Reading(columns_value, item.margin),
        fill=Reading(
            fill_value,
            item.margin if isinstance(fill_value, Valued) else None,
        ),
        distance=core.distance,
        short_fill=item.columns < MIN_FILL_COLUMNS,
    )


def _over_tier(item: _Measured) -> bool:
    """这一根量到的条身**够得着更高那一档**吗（跨档校验，防「橙条 boss」）。

    **门对着 boss 那一档设**（用户 10-15 的裁定）：小怪那一档量到
    **>= `BOSS_BODY_ROWS`（10 行）才拦** —— 9 行及以下放过（抗锯齿与叠条都会在
    8 行上多出 1 行，9 不等于 10）。**红族不设这道门**（用户 10-15 的裁定）：它本来
    就是最高那一档、没有更高的档可比，而这道门只为防「橙条 boss」。

    **代价**（写进夹具 README）：万一有一根 9 行的橙条其实是 boss（10 行被量少
    一行），我们会按 123 算 ⇒ 比例**偏高约 25%**。要消除它，得有第二根 boss 条
    样本来定「行数会不会被量少」。
    """
    if item.core.family is _BOSS_FAMILY:
        # **红族不设这道门**：没有更高的档可比，这里没有任何要防的东西。
        return False
    return item.rows >= CROSS_TIER_ROWS


def _is_clump(members: Sequence[_Measured]) -> bool:
    """这一组是不是**一堆**（用户 10-15 的裁定 2：用已经有的那两条，不新立一套）。

    - **分不开**：这一组不止一块（`_inseparable` 把它们并成一组了）—— 同一条带上
      相邻两根的填充交叠 / 相接，或者中间夹着混色列；
    - **填充总列数长过这一档的满条长**：一根条的填充不可能长过它那一档的满条长
      （`raw_b_上` 那 130 列就是这一支）。**这里用的是跨得过 `FILL_GAP` 的那个
      总列数**（把缝桥过去之后量的）—— 它是**症状探测**、不是读数，所以与单根那
      两条读数不许跨缝**不矛盾**（用户 10-15 的裁定：那一道门照旧用总列数）。

    两条可以同时成立（`ClumpRoute` 那一栏把来路报出来）。
    """
    if len(members) > 1:
        return True
    return members[0].columns > members[0].core.family.full_columns


def _clump(top: int, members: Sequence[_Measured]) -> EnemyClump:
    """一组 -> **一堆**那一个读数（区域 + 至少几根 + 填充总列数 + 来路）。

    **填充总列数** = 组里每一根的填充段列数**相加**（含缝那几列，见 `_measure`）。
    它的形态只分两支：**右边被盖住**（组里有一条没读到右端）-> `LowerBound`；
    否则 -> `Exact`。**缝不降它的形态**（与单根那两条读数口径不同，见模块开头
    「三样读法」）：它是**止损要用的那个数**，由 `FILL_GAP` 把缝桥过去连起来数。
    **为什么只有两支**：`Interval` 要有「两头都模糊」的来路，而单根只产 精确 /
    下界 两支。

    **至少几根** = `ceil(总列数 / 这一档的满条长)`，落 `LowerBound` —— 它**只是
    下界**，而且是**下界的下界**（总列数自己可能就是个 `LowerBound`）。**上界不
    从像素来**：上界由状态层拿**账**配。

    **余量 = 组内各根 `columns` 余量的最小值**（口径见 `_Measured.margin`）：最弱
    的那一环就是这一堆的余量。

    **按族归组**（`_inseparable` 已经保证）：所以 `tier` 只有一个，取第一个成员
    的就对。**来路**看两条各自成不成立：组里不止一块 -> 分不开；总列数长过满条
    长 -> 长过一根的上限；两条都成立 -> 两者。
    """
    tier = members[0].core.family.tier
    total = sum(item.columns for item in members)
    margin = min(item.margin for item in members)
    if any(item.covered for item in members):
        columns_value: SlotValue = Valued(LowerBound(total), margin)
    else:
        columns_value = Valued(Exact(total), margin)
    least = max(1, -(-total // tier.full_columns))
    return EnemyClump(
        region=_group_rect(top, members),
        tier=tier,
        columns=Reading(columns_value, margin),
        least=Reading(Valued(LowerBound(least), None), margin),
        route=ClumpRoute(
            inseparable=len(members) > 1,
            over_full=total > tier.full_columns,
        ),
    )


def _group_rect(top: int, members: Sequence[_Measured]) -> PixelRect:
    """这一组的**区域**：组里每一根**填充段**的外接框（填着的那一块，不含空轨）。

    坐标补上战场那一段的原点 `top`（与 `EnemyBar.fill_rect` 同一套画布坐标）。
    """
    left = min(item.core.x for item in members)
    right = max(item.core.x + item.columns for item in members)
    first = min(item.row for item in members)
    last = max(item.row + item.rows for item in members)
    return PixelRect(left, first + top, right - left, last - first)


def _fill_confidence(distance: np.ndarray, filled: int) -> float:
    """这一根比例读数有多稳：**定下这次读数的两个判据两侧隔了多远** / 255。

    与我方血条那条链同一个口径：里头最弱的一列（离色线最近的）与右边那一列
    （它必须干脆地在线外）里，小的那个。**分母那一头不进来** —— 分母是常量，
    它错不了；能错的只有「填充段到哪儿为止」。
    """
    if filled <= 0 or filled > distance.size:
        return 0.0
    inside = float(SCREEN_TOL - distance[:filled].max())
    if filled >= distance.size:
        return min(1.0, max(0.0, inside / 255.0))
    outside = float(distance[filled] - SCREEN_TOL)
    return min(1.0, max(0.0, min(inside, outside) / 255.0))


# ---------------------------------------------------------------------------
# 分不开的那几根：整坨算一堆，条数只能给下界
# ---------------------------------------------------------------------------


def _inseparable(
    region: np.ndarray, read: Sequence[_Measured]
) -> list[int]:
    """每一根条归到哪一组：自己一组的返回自己的下标，分不开的返回这一组的最小下标。

    **判据只有一条**：两根条**踩在同一条行带上**（行差 <= `ROW_SLACK`），而且中间
    **分不开**（见 `_touching`）。这一组就是「一堆」的候选（`_is_clump` 再看那一
    组是一根还是一堆）：分不开的可能本来就是一根被压出来的一截，所以条数只能给
    下界，而那几根也不单独报比例。

    **按族归组、不许跨族**（用户 10-15 的裁定 2）：一堆里的成员同一族 —— 否则
    「至少几根」要除以哪一档的满条长就说不清，而红条与橙条本来也不可能是一根。

    **这里不合并、也不动任何一条线**：把两块接回一根是 `_merge` 的事（间距
    `MERGE_GAP` 以内、且中间**每一列**都混色）。这里只是**承认分不开**，
    把那一组送进 `_clump` —— 精细的活儿在别处，这一支只负责**不假装数清了**。
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
            if read[left].core.family != read[right].core.family:
                continue
            if not _touching(region, read[left], read[right]):
                continue
            first, second = root(left), root(right)
            if first != second:
                parent[max(first, second)] = min(first, second)
    return [root(index) for index in range(len(read))]


def _bands(read: Sequence[_Measured]) -> list[list[int]]:
    """把条按**行带**切开，每一条带里按列排好。

    「行带」是**连着**算的：行差 <= `ROW_SLACK`（与判我方条同一个容差）就算同一条，
    所以 451 与 452 同带、452 与 453 也同带 —— 一列 1 行一行的阶梯本来就分不开，
    把它切成两条带才是假装分开了。
    """
    order = sorted(
        range(len(read)), key=lambda i: (read[i].core.y, read[i].core.x)
    )
    bands: list[list[int]] = []
    current: list[int] = []
    for index in order:
        if current and read[index].core.y - read[current[-1]].core.y > ROW_SLACK:
            bands.append(current)
            current = []
        current.append(index)
    if current:
        bands.append(current)
    return [sorted(band, key=lambda i: read[i].core.x) for band in bands]


def _touching(region: np.ndarray, first: _Measured, second: _Measured) -> bool:
    """同一条带上的两块**分不开**吗：分不开才 True。

    **判据只用填充段自己的像素**（空轨退役之后，这里再没有「整长」可用）：

    - **填充段交叠或相接**：填着的那两段在列上压着/贴着 —— 一根条只从自己的左缘
      开始填，两段填充这样并排不可能各自都从头填起，所以至少有一根是被截出来的
      一截（也可能是同一根被压断、`_merge` 没敢接）；
    - **中间至少有一列是「混色」**（`_blended`：那一列比它上下两块地形更靠近填充
      色）—— 那正是「条被半透明的东西压过去」的像素证据，于是这两块可能本来就是
      同一根条的两截。

    **只找「混色」，不找「条色」**：中间那一截又亮起来的（同一条带上被我们自己的
    技能光压断）在 `_screen` 里已经接回去了（`_merge`）；走到这里还分开的，中间
    那几列要么是地形（那就真的是两根），要么是压在条上的半透明东西。

    **它只把条数降成下界、把这一组送进「一堆」**（见 `_inseparable` 与 `_clump`）：
    判不准的两块当然也可能是两根，所以这一支永远不高报。

    **跨族不比**：`_inseparable` 先按族分过了 —— 红条与橙条不可能是一根条的两截。
    """
    left, right = (first, second) if first.core.x <= second.core.x else (second, first)
    gap = right.core.x - (left.core.x + left.columns)
    if gap <= 0:
        return True
    row = max(left.core.y, right.core.y)
    return any(
        _blended(region, row, column)
        for column in range(left.core.x + left.columns, right.core.x)
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
