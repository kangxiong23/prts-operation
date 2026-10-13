"""备战席点名：**一帧 → 每个卡位一个身份**。

《视觉层/备战席点名》管这一层。它只回答一件事：备战席上第 k 张卡是哪一位干员
—— 或者，我们认不出来。

**它不产状态层字段。** 状态层的「身份」是**记**出来的（部署记录），这一层只是
「把干员名换成第几张卡」的查找，给 `部署(能天使, (5, 4), 右)` 用。

## 库是一个参数，而且可以不同源

「库」= `faces`，**一串 `(名字, 立绘块)`**，按编队页的切割顺序排好。它由产卡面
那一侧（`ark.offline.assets_gen.cardface`）造，而这一层**收它当参数**：不 import
`assets_gen`、不认编队页、不知道库存在哪张盘上（导入规则也不许 —— `vision` 只许
import `contracts` 与 `vision` 自己）。

**库与战斗帧可以不同源**（库按设计是跨局累积的，《资产生成器/卡面与指纹》§3），
所以「备战席上这个人，库里根本没有」是**常态**，不是失败 —— 落**未知**。

**空库不是「库里没有这个人」**：那是**没建库**，报错。与「表没建好 ≠ 这张卡没有
倒计时」是同一条 —— 静默吐出来的**假缺席**与假事实一样坏。

## 判据是三步

1. 备战席卡里取一块**固定窄框**（`identity_region`，卡内 `(52, 76, 100, 100)`）；
2. 在库里每一条的**立绘**里**滑着找位置**，尺度在 `IDENTITY_SCALE_MIN`–`MAX`
   里按 `IDENTITY_SCALE_STEP` 扫（**区间是拿真峰的分布定的**，见那两个常量）；
3. **最高分过线，而且次高退到自己那条线以下**：
   `最高 >= ROSTER_SOLID_MATCH_SCORE` 并且 `次高 <= required_runner_up(最高)`。

**第 3 条里的余量是"随分数走"的**（用户裁定）：要求的分差 = `1.09 − 最高`，不是从前那条
固定的 `ROSTER_AMBIGUITY_MARGIN = 0.25`。最高 >= 0.85 时它 <= **0.24**，所以换到这条
**判定账只可能松**。旧那条常量仍然定义着 —— 它现在只作为**历史夹具的对照**
（`测试/余量/余量.json` 那批是拿它量的），**线上不再用它**。

**为什么框要窄**：宽的框搜不了尺度 —— 备战席卡 204 宽、编队页立绘 214 宽，
204 宽的模板一放大就放不下，尺度被卡死在 1.00，而真匹配的尺度在 **1.01–1.16**
（实测真峰分布，见 `IDENTITY_SCALE_MIN`）。**取窄了，尺度就自由了。**

## 扫法：先粗后细，判据满足即停

尺度那一步既不是"整段逐个试到底"，也不是"只在粗找那一档附近试"（用户裁定）：
**试的档扩到整段**（`IDENTITY_SCALES`），**次序 = 离粗找那一档越近越先试**，
**判据满足即停**。

| 层 | 图缩到 | 干什么 | 报不报分 |
| --- | --- | --- | --- |
| 锚点层 | 25×25 | 范围铺满整段，给每个条目每一档一个分；**锚点 = 分最高那个条目的并列档组的中点** | 不报 |
| 位置层 | 50×50 | 给**每个条目每一档**一个位置 | 不报 |
| 精算层 | 全分辨率 | 从锚点那档起按远近挨个算，**判据一满足就停** | **报** |

**报出来的分是第一个满足判据的那一档上的分**，不是全局最大分（用户裁定）。一段走完
仍然不满足的（库里没有这个人），报的是**每个条目在整段上的最大分**。

**为什么锚点必须可信**：12 像素那一层上 `round(12 × 档)` 只有 2 个不同值（25 像素 7 个、
50 像素 13 个、全分辨率 18 个），模板尺寸一样的那几档 `resize` 出来逐位相同、分数逐位相同、
整片并列，而 `argmax` 只能取一个 —— 那种"粗找那一档"**基本只知道下沿**，拿去排序等于乱序。
`_tie_midpoint` 并列时取组中点，锚点就不带偏向。实测（`停即扫.json`），对**认得出**的
210 张卡：锚点与穷举选中那一档之差中位 **2 档**、<=2 档 143 张、<=3 档 175 张。

**位置窗贴边就往里挪**（不退回整张）：库里没有这个人时，粗层给的位置本来就是噪声峰、
常常落在立绘边上；从前那条碰上就把整张塞回去（搜索面大 20 倍以上），于是"认不准的图"
反而最贵。现在把窗**平移**回立绘里 —— 窗的大小不变、模板一定装得下。

**代价**：调用次数**不再**是 `卡位 × 库长 × 档数` —— "与库长无关"那条**死了**，
因为**停不停取决于库里有什么**。还真的有两件：次数**有上界** `卡位 × 档数 × 库长`，
而且**每个条目被同样对待**（同一张卡上，所有条目在同一个档上各算一次，没有谁被跳过）。

**为什么"滑着找位置"就够了**：两侧是**同一款美术**，所以判据退化成"找位置" ——
"每个干员脸的位置不一样"这件事被滑动吸收了，**不需要先找脸**。

## 几何资产：每卡位 × 每条 · 粗层直接接受 · 画面没变就跳过

**找几何与打分是两件事**（用户裁定）：每个卡位**第一次**跑满那一趟，把**每条库自己的对位**
留成资产（`SlotGeometry`：`立绘坐标 = 偏移 + 档 × 卡内坐标`）；之后每一帧**只在那份
几何上打分**（`_geometry_scores` / `_masked_at_geometry`），不再重新找位置、找档。
资产按 **(库指纹, 卡距, 画布尺寸) ＋ 卡位下标** 分本，键一变整本清空并记一次重建 ——
**不许拿旧几何去量新局面**。身份**不进资产**：它每帧在那份对应关系上**重新**逐条算。

**每条各存各的**（用户裁定 3）：跑满那一趟里每个条目自己的对位本来就已经算出来了，只存
赢家那条会让错条目全部塌到赢家的对位上 —— 那样"次高"根本不是次高。所以复用那条路报的
那个次高叫 `runner_at_geometry`（**同几何次高**），与全扫那条路的次高 `runner`
**不是同一个词**：前者是"这一条在**它自己**的对位上有多像"，后者是"整段上第二像的那一条"。

**复用那条路有自己的分线**（用户裁定 2）：`ROSTER_REUSE_MATCH_SCORE` —— 那条路上的分与
全扫的分不是同一批（只在一个档 + ±`_GEOMETRY_JITTER` 上比），所以线要坐在**它自己的
空档**里（两簇与空档见 `tests/帧样本/先粗后细/README.md`）。**同一条余量公式仍然成立**
（用户裁定 3 之后余量在那条路上重新有意义），公式一个字没动。

**粗层直接接受**（用户裁定）：锚点层一过 `ROSTER_COARSE_ACCEPT_SCORE`（0.9）**而且**判据
成立，就在这一层判精确 —— 位置层与全分辨率那一步一次都不跑。**只有锚点层能直接接受**，
因为位置层要裁 ±`_ANCHOR_POSITION_HALF` 的窗，而裁窗压的是"只压次高"，那是把余量线弄弱的
唯一方向。**红罩那条路同一套**（用户裁定 7）：`_masked_layered_scores` 的三层与这里
**同一批常数**，锚点层同样**不裁窗**。

**红罩那条路也吃到锚点带来的收窄**（用户裁定 1）：`_masked_layered_scores` 把"找档"
搬到 25 像素那一层（`MaskedRuler.rows(pixels=…)`），全分辨率那一层只在一个档、±6 px
的窗里算 —— 从前那条路整张滑 18 档，一卡 **2.758 s**，是干净那条路（0.238 s）的 11.6 倍。

**这一格的画面没变就整格跳过**（用户裁定 4）：按卡位记住上一次**打分时**那块取区的每一个
字节（`blake2b`），下一帧逐位相同就沿用上次的判定 —— 判据是**画面事实**，**不是分数**
（分数低正是"几何坏了 / 换了人"的表现，按它免打分等于把唯一能自愈的机制关掉）。
跳过的格照样出判定、而且计数（`counts["跳过"]`）。

## 分数就是 Pearson

`roster_scores` 走 `cv2.matchTemplate(..., TM_CCOEFF_NORMED)`：零均值归一化相关，
与 `glyphs.similarity` 是**同一个统计量**（那边还多一个 ±2 px 的平移搜索，这里的
位置是滑出来的）。判据用这个数，所以它必须与读字形那条链同源 ——
`tests/unit/test_vision_roster.py` 里有一条拿手算的 Pearson 把它逐位对上。

**匹配在 RGB 三通道上做**（三个通道当一个向量），不是灰度：卡面是彩色的，颜色
本身就是辨识度的一部分。而 `TM_CCOEFF_NORMED` 减均值、除能量 —— **它比的是归一化
之后的东西，所以对「整片乘一个标量」天生免疫**（用户裁定，第九单）：费用不足那一档
的压暗不影响它的判定与名字。实测同一张卡跨帧的相关：灰罩 **0.9967** / 乘回去
**0.9968** / 逐位相同 **1.0000**；两张横带与差图（「压暗 = ×0.502」那一段读数）留在
`tests/帧样本/缺费/`。

## 落法：只有两种形态

| 情形 | 落法 |
| --- | --- |
| 最高分过线**而且**余量拉开 | `Valued(Exact(名字), 最高分)` |
| 比不中（库里没有这个人 / 装置卡） | `Valued(Unknown(), None)` |
| **红罩盖住**（再部署中） | **读得出 ⇒ 精确；读不出 ⇒ 未知** |

**这一层只产两种形态：精确 / 未知，没有「未采样」**（用户裁定）。"这一格没卡"是
**卡位**那一层的事 —— `squad_card_centres` 数出几个卡位就有几个值，不是身份的事。

**红罩卡不再一律未知**（D-全局-34 推翻原来那条）：它换**另一块取区**，而判据、形态
与那两条线**一个都没变** —— 读得出就是精确（分数就是定下它的那个），读不出就是未知，
而 `Unknown` **不带分数**。

## 两块取区：干净卡走原来那条，红罩卡走圆盘那条

| 这一格 | 取区 | 谁给 | 用什么比 |
| --- | --- | --- | --- |
| 干净卡（`redeploying` 不是 `True`） | 身份区那个 100×100 的方框 `(52, 76, 100, 100)` | `identity_region` | `roster_scores`（`TM_CCOEFF_NORMED`） |
| **红罩卡**（`redeploying` 是 `True`） | **环带以内的圆盘 − 中央秒数那一条** | `redcap_region` / `margin.IDENTITY_DISC_*` | `margin.MaskedRuler`（**逐通道减均值的掩码相关**） |

**两块取区分开记、不互相顶替**（《备战席点名》§判据的形状），但那两条线**两块共用**
—— 换的只是取区，不是判据。

**这一层不为「缺费」开分支**（用户裁定 1，第九单）：曾经接过一条「缺费就按每格裁决把
取区乘回去」的路，**已撤**（那个分支、它的模块与它的 14 条单测一起退了）。有没有缺费是
**状态层**的事，形状是**一次比较**（卡费 vs 当前费）；**视觉层不养分支**，取区仍然只有上面那两块。

**它们凭什么算同一条统计量**：`MaskedRuler` 掩码全开时**逐位复现 `roster_scores`**
（`tests/unit/test_vision_redcap_margin.py` 钉着）；两者的分子都是逐通道减均值之后的
内积、分母都是两边能量的几何平均 —— 掩码只是把「哪些像素算数」那一项换掉。所以
`ROSTER_SOLID_MATCH_SCORE` / `ROSTER_AMBIGUITY_MARGIN` **不为红罩卡另开一条**。

**红罩卡那条路不反解**：罩子对这条判据**天生透明**（逐通道仿射整类被约掉，D-全局-28），
反解换的是**单位**、不是修复 —— 所以线上**不读模型**，那张卡**原样**进取区、原样比。
（模型仍住 `redcap.py`，是**判据侧的测量工具**。）

**`redeploying` 判不了（`None`）按"不在再部署中"处理**：宁可让一张红罩卡留在原来那条
路上（落未知），也不把一张可能干净的卡换成新口径 —— 「干净卡逐位不变」那条才守得住。

**这一层不读倒计时那条链**：判"是不是再部署中"只要**环比环外亮多少**那一条
（`bench.redeploying`），不用字形表、也不读秒数。两件事各自独立。

**"比不中"不许判成"装置"**：库里缺一张（没注册过的新组合、新皮肤、新精英化）时
**同样比不中**，两种原因在这一层分不开 —— **人点名是唯一的消歧手段**。

**再部署中的卡点得动，但部署不了**（用户确认）：所以"点得动"**不等于**"可以部署"，
生效反馈不能建立在它上面 —— 那是**状态**那一侧的事，与这一格读不读得出身份无关
（红罩卡现在读得出来，见 §两块取区）。

## 两路：全扫一帧 / 单条查一位

| 路 | 入口 | 什么时候 | 判据 |
| --- | --- | --- | --- |
| **全扫** | `read_squad_identities` | 每次要用之前，把**整栏**读一遍 | 最高分过线**而且**余量拉开 |
| **单条** | `find_squad_card` | 只问**某一位**在第几张卡 | 恰有一张卡过 `ROSTER_SOLID_MATCH_SCORE` |

**单条那条路不扫全库**：它只拿「名字对上的那一条」去比每一张卡，所以
`matchTemplate` 的次数是 `卡位 × 尺度档数`，**与库长无关**（实测库 2 条与 11 条同值）。

**为什么一条线就够**（实测，`shot01` + `cd60` 两帧 23 张卡、253 个 (卡, 条目) 对；
口径是"**第一次跨过线**那个分"，见 `tests/unit/test_vision_roster.py` 那张表）：
真匹配 **0.924280–0.974215**（16 对）· 非匹配 **≤ 0.562917**（237 对）—— 空档 **0.361363**，
而 `ROSTER_SOLID_MATCH_SCORE = 0.85` 落在里面。所以「像到过了 0.85」这一件事本身
就把两簇分开了，**不必再比次高分**。**这两簇的间距被一条单测钉住**（用户裁定 6）：
线一动、或者两簇一挤，那条就红 —— 这条路没有次高可比，判据退化成只看分线，
所以它自己的网只能是这一条。

**跳掉的是「库条目之间那一比」，它改不了结论**：那一比只有两个用处 ——
① 决定全扫会不会认成**别人**；② 决定余量够不够。①发生不了：别的条目在那张卡上是
**非匹配**，实测 ≤ 0.5622 < 0.85 ≤ 那一位自己的分；②只会把「全扫说 n」变成
「全扫说未知」，而那条路上**未知是保守的那一侧**，不是假事实。所以跳掉它，
单条**不会认错人、也不会产假事实** —— 至多比全扫多指出一张卡。

（②的取证就是上面那两簇。它同时是 `read_squad_identities` 里那句「库里只有一条时
判据退化成只看最高分」的读数：只留一个分数时，两簇就是这 16 对与那 237 对。）

**这条路不产槽值，也不产否定存在断言**：它回答的是一个**卡下标**，所以它既不是
第三种形态，也不是「未知」的另一种写法。`Unknown` 那句话是「那里确实有东西、但
不知道是什么」，用在这里等于替「读不出来」**断言了存在** —— 而这条路证不了存在。
它只回答两件事：**在第 k 张卡** / **没能指出他在哪**。
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from ark.contracts import (
    Exact,
    PixelRect,
    SlotValue,
    Unknown,
    Valued,
    squad_card_rect,
)
from ark.engine.vision.bench import redeploying, squad_card_centres
from ark.engine.vision.frame import Frame

__all__ = [
    "read_squad_identities",
    "find_squad_card",
    "roster_scores",
    "squad_card_scores",
    "identity_region",
    "redcap_region",
    "IDENTITY_BOX_X",
    "IDENTITY_BOX_Y",
    "IDENTITY_BOX_WIDTH",
    "IDENTITY_BOX_HEIGHT",
    "IDENTITY_SCALE_MIN",
    "IDENTITY_SCALE_MAX",
    "IDENTITY_SCALE_STEP",
    "IDENTITY_SCALES",
    "ROSTER_SOLID_MATCH_SCORE",
    "ROSTER_AMBIGUITY_MARGIN",
    "ROSTER_COARSE_ACCEPT_SCORE",
    "ROSTER_REUSE_MATCH_SCORE",
    "SlotGeometry",
    "SquadGeometry",
]


IDENTITY_BOX_X: Final[int] = 52
IDENTITY_BOX_Y: Final[int] = 76
IDENTITY_BOX_WIDTH: Final[int] = 100
IDENTITY_BOX_HEIGHT: Final[int] = 100
"""备战席卡内的**身份区**，以**卡片左上角**为原点（实测值 `(52, 76, 100, 100)`）。

**上下位置不挑**：实测 y0 取 56…96 都是同一个平台，所以这个值不敏感。
**宽窄很挑**：几乎整块立绘（204×180）时分不开（空档 **−0.038**），140×140 空档
0.235、**100×100 空档 0.265**、60×60 空档 0.301 —— 取 100 是因为再窄就没有余量
可赚，而 100 已经够。

**它不进 `UiLayout`**（用户裁定）：`UiLayout` 是**战斗界面**的地址表，而这一层
**不收 `ui` 参数**（与 `bench.py` 同一个口径）—— 一个只能换掉锚点、换不掉位移的
参数比没有参数更危险。这个框住在**用它的这一层**。
"""


IDENTITY_SCALE_MIN: Final[float] = 1.00
IDENTITY_SCALE_MAX: Final[float] = 1.17
IDENTITY_SCALE_STEP: Final[float] = 0.01
"""尺度搜索的上下限与步长。

**区间是拿真峰的分布定的，不是拍出来的**（用户裁定：一个边缘上坐着近四成峰的区间
就是设错的范围）。实测（`tests/帧样本/先粗后细/曲线.json`，26 档 0.95…1.20 的全分辨率
取值域，45 张有逐位真值的卡 —— 真值那一条的峰）：

    1.03 x1 · 1.05 x1 · 1.06 x9 · 1.07 x1 · 1.08 x2 · 1.09 x6 · 1.10 x3
    1.12 x8 · 1.13 x7 · 1.14 x1 · 1.15 x3 · 1.16 x1

另有 2 张"峰"落在 0.95，但那两张真值条目自己的分只有 0.22 / 0.28 —— 那是噪声坐在
网格边上，不是峰。所以**真峰的实际跨度是 1.01…1.16**，两边各留一档 → **1.00…1.17（18 档）**。

**为什么下沿一定要往下**：`立绘两组` 的 `组2_备战席 #4`（真值那一组是 `组2·塞雷娅`）
峰在 **1.01 @ 0.9758** —— 旧区间从 1.05 起，它在区间里的最大分只有 **0.8309**，
差一点过不了 0.85 那条线。峰在范围外的卡，不是"认不出"，是**没给它机会**。

**为什么上沿只到 1.17**：26 档网格一直铺到 1.20，**1.20 上一个峰都没有**，1.17–1.19 也没有 ——
数据说 1.17 够了。（旧说明里那句「至少有一张的峰在 1.16」在数据里对得上：1.16 有 1 张。）

**步长 0.01 是量出来的**：同一批 16 张上，步长 0.005 与 0.01 给出的最高分一位都不差；
而粗到 1.00/1.10/1.20 三档时最差那张会从 0.94 掉到 0.83。所以"细"是必须的，0.01 已经够细。

**报出的分不是全局最大分**（用户裁定）：扫法在**第一个满足判据的档**上就停，
所以报出来的是那一档上的分。见模块文档 §扫法。
"""

IDENTITY_SCALES: Final[tuple[float, ...]] = tuple(
    round(IDENTITY_SCALE_MIN + index * IDENTITY_SCALE_STEP, 3)
    for index in range(
        int(round((IDENTITY_SCALE_MAX - IDENTITY_SCALE_MIN) / IDENTITY_SCALE_STEP)) + 1
    )
)
"""实际扫的那些尺度：`1.00, 1.01, …, 1.17`（**18 档**）。"""


ROSTER_SOLID_MATCH_SCORE: Final[float] = 0.85
"""最高分要像到什么程度，才配被"认出来"成一个确切的名字。

**这个数是量出来的，而且要说清用在哪几帧上。** 规格那条实测是 **6 帧 56 张卡
全量**：

| | 分数 | 余量 |
| --- | --- | --- |
| 真匹配（26 张） | **0.940–0.987** | **≥ 0.392** |
| 不是它（含装置，16 张） | **≤ 0.575** | **≤ 0.084** |

**我这一版只拿得到 `shot01` + `cd60` 两帧的逐位真值**（`tests/帧样本/点名/取材.json`
给的就是这两帧），在那 23 张卡上复量：真匹配 **0.9618–0.9921**（余量
0.4261–0.6118）、不是它 **≤ 0.5622**（余量 ≤ 0.1052）—— 与规格那条对得上。
**验收那几帧没有参与定线。**

空档是 **0.575 … 0.940**，**0.85 落在里面而且偏向"正确"那一侧**（离真匹配那簇
更近）。理由是不对称：把一张卡认成确切的名字是**假事实**（验收里"任何一个确切值
认错就是没过"），而认不出来只是**未知** —— 契约层那句「让不知道在类型上没法被
伪装成知道」说的就是这个方向。
"""

ROSTER_AMBIGUITY_MARGIN: Final[float] = 0.25
"""次像的那个离最像的有多近才算"分不开"。

空档是 **0.084 … 0.392**（同上那两张表），**0.25 落在里面**，仍然偏向"正确"那一侧。

**这个数是从 0.30 往下挪到空档下缘的**（用户裁定，[D-全局-39]）：0.30 是**干净卡识别
开发阶段**设的；强化测试（同一位两套立绘同时在库）量到的「库里有相似条目时余量被压到
多少」是 **0.19–0.20**，而**线上库 = 本局一帧**，那一档的压缩线上不发生 —— 压在新旧两条
线之间的那一段，正是**线上会发生、而开发阶段没覆盖到**的那一段。

**它是放低一条线，不是加固**：这次只把「分数够、余量差一点」的卡放回来（召回上去），
而**分数线一个字没动** —— 实测里「错的最像那条」从没到过 **0.73**（干净框 ≤ 0.57），真匹配
落在 **0.83–0.99**，所以**放不进一个错名字**。认不出照样落**未知**。

**余量才是真正分开两簇的那条**：不是它的那些卡，最高分与次高分本来就差不多
（≤ 0.1052）—— 库里没有这个人时，"最像的那个"是**挑出来的**，不是匹配出来的。
所以**只看最高分会把没认出来的那些也挑一个**，两条线必须**同时**过。
"""


ROSTER_AMBIGUITY_FACTOR: Final[float] = 2.0
ROSTER_AMBIGUITY_PAD: Final[float] = 0.09
"""余量那条线的两个数（**用户裁定**）：

    最高 >= 0.85   并且   (1 - 次高) / (1 - 最高) >= 2 + 0.09 / (1 - 最高)

**它是"分差随分数走"的写法**：两边同乘 `(1 - 最高)`（它是正的）等价于

    次高 <= 2 x 最高 - 1.09

也就是**要求的分差 = 1.09 - 最高**。最高 >= 0.85 时它 <= **0.24**，永远小于旧那条固定的
0.25 —— 所以换到这条，**判定账只可能松、不可能紧**（丢 0）。

**为什么要把余量写成随分数走的**：0.25 是拿**开发阶段那批画面**的空档（0.084…0.392）定的，
而空档自己随分数走 —— 分越高，"次像的那个"本来就越该远一点。实测：

|| 最高分那一带 | 旧线 0.25 挡住、新线放行的 | 新线要求的分差 | 富余 |
|| --- | --- | --- | --- |
|| 0.9882–0.9897（红罩那 6 张） | **6 张** | 0.1003–0.1018 | 0.0964–0.1037 |
|| 0.85–0.95（最紧的那一段） | 0 张 | 0.14–0.24 | — |

**它是放低一条线，不是加固**（与 0.25 当初那次同一个方向）：分数那条线一个字没动，
放回来的那 6 张分差 0.198–0.204，而它们**错的那条**（次高）在 0.786–0.790 —— 离 0.85 很远，
放不进一个错名字。
"""


def required_runner_up(top: float) -> float:
    """判据那半条线的**等价形式**：次高不超过这个数才算"分得开"。

    `次高 <= 2 x 最高 - 1.09`。**这一处是判据的唯一定义** —— 扫法的停止条件与
    最后的判定都调它，所以"哪一档停"与"算不算认出"永远是同一件事。
    """
    return ROSTER_AMBIGUITY_FACTOR * top - (
        ROSTER_AMBIGUITY_FACTOR + ROSTER_AMBIGUITY_PAD - 1.0
    )


def identity_is_exact(top: float, runner_up: float) -> bool:
    """最高过线**而且**次高退到 `required_runner_up` 以下 —— 就是"认出来了"。

    `runner_up` 用 `-2.0` 表示"没有次高"（库里只有一条）：那一档判据退化成只看最高分。
    """
    return top >= ROSTER_SOLID_MATCH_SCORE and runner_up <= required_runner_up(top)
def identity_region(centre_x: float, *, pitch: float) -> PixelRect:
    """某个卡位的**身份区**在画布上的矩形。

    只收**中心与卡距**：中心由 `squad_card_centres` 给（`bench.py` 已经有的那个
    公共入口，这一层**不许另开一套定位**），卡框的宽与 y 由契约层
    `squad_card_rect` 给 —— 这一层**不另抄一份卡距、也不另抄一份卡宽**。

    框以**卡片左上角**为原点，所以这里要把中心折回左缘：`左缘 = 中心 − 卡宽 / 2`。
    """
    card = squad_card_rect(0, pitch)
    left = float(centre_x) - float(card.width) / 2.0
    return PixelRect(
        left + IDENTITY_BOX_X,
        float(card.y) + IDENTITY_BOX_Y,
        IDENTITY_BOX_WIDTH,
        IDENTITY_BOX_HEIGHT,
    )


def redcap_region(centre_x: float, *, pitch: float) -> PixelRect:
    """**红罩卡那一格**的取区在画布上的矩形 —— 环带以内的圆盘、挖掉中央秒数那一条。

    折法与 `identity_region` **一模一样**（中心折回左缘、卡框的 y 走契约层），**不另抄
    一份卡宽、也不另开一套定位**；不同的只有那个框：形状住在 `margin.IDENTITY_DISC_*`
    （一处定义），这里只把它摆到这一格上。

    **它与 `identity_region` 是两个框、不互相顶替**（《备战席点名》§判据的形状）：
    哪一格用哪块由 `redeploying` 说（见模块文档 §两块取区）。**整块落在卡片以内**由
    `margin` 那边把住（半径超过圆心到任一边的距离就报错），所以这里不裁边。
    """
    box = _margin().IDENTITY_DISC_BOX
    card = squad_card_rect(0, pitch)
    left = float(centre_x) - float(card.width) / 2.0
    return PixelRect(left + box[0], float(card.y) + box[1], box[2], box[3])


def _margin():
    """延迟拿到 `margin` —— **模块级 import 会成环**。

    `margin.MaskedRuler` 默认的尺度网格就是本模块的 `IDENTITY_SCALES`，所以那边
    import 本模块；本模块再在顶层 import 它，先被 import 的那一个会在自己还没定义完
    时被对方读到。**只有这一处 import 它**，理也只讲一遍。
    """
    from ark.engine.vision import margin

    return margin


def _redcap_scorer(
    faces: Sequence[tuple[str, np.ndarray]],
) -> Callable[[np.ndarray], tuple[tuple[float, str, float], ...]]:
    """→ 一个「一块**红罩卡取区** → 读数」的函数（`((分数, 名字, 尺度), ...)` 降序）。

    **尺子在第一次要它的时候才建**（调用方只在真遇到红罩卡时才调本函数）：建一把要
    ~1 s，而一帧里建一次就够 —— 没有红罩卡的帧一分钱不花（见模块文档 §扫描量）。

    **它与 `roster_scores` 是同一条统计量**：`MaskedRuler` 掩码全开时逐位复现
    `roster_scores`（单测钉着），两者都是「逐通道减均值的内积 ÷ 两边能量的几何平均」，
    掩码只是把「哪些像素算数」那一项换掉。所以那条线（`ROSTER_SOLID_MATCH_SCORE`）
    **不用另开**。
    """
    margin = _margin()
    _box, mask = margin.identity_disc_region()
    ruler = margin.MaskedRuler(mask, faces)

    def score(region: np.ndarray) -> tuple[tuple[float, str, float], ...]:
        return tuple(ruler.scores(region))

    return score


ROSTER_COARSE_ACCEPT_SCORE: Final[float] = 0.9
"""**粗层直接接受**那条分线（用户裁定）—— 只有锚点层用它。

比全分辨率那条（`0.85`）**严**，两个理由：

1. 粗层的分数是"往里挤满格"的：25 像素那一层上每个像素都在赢家身上，**次高抬得比赢家
   猛得多**，所以同一条余量公式在粗层上**更严、不是更松**（方向是安全侧，已在实测上确认）；
2. `0.9` 是**用户当场定的**，不是量出来的 —— 所以它必须被那张网盖住：
   `tests/帧样本/先粗后细/粗层接受.json` 把每一张"是在哪一层被接受的"逐张列出来，
   凡是**粗层接受、而全分辨率跑满给出不同答案**的也逐张列出来（那是"0.9 太松"的取证）。

**前提是粗层不裁位置窗**：裁窗压的是"只压次高"，那才是把余量线弄弱的方向（实测 +0.13，
带来假精确）。所以**只有锚点层**（在整块立绘上滑、不裁窗）能直接接受；位置层要裁
`±_ANCHOR_POSITION_HALF` 的窗，不在那里接受 —— 见 `_layered_scores`。
"""

ROSTER_REUSE_MATCH_SCORE: Final[float] = 0.85
"""**复用那条路自己的分线**（用户裁定 2）—— 只在这一条路上用。

那条路上的分与全扫的分**不是同一批**：它只在**每条库自己那一档**、±`_GEOMETRY_JITTER` 的
窗里比。所以线要坐在**那条路自己的空档**里 —— 两簇（真匹配的最低 / 非匹配的最高）与空档
是量出来的（`tests/帧样本/先粗后细/README.md` §第六单），单测钉在
`tests/unit/test_vision_roster_geometry.py`。

**它不许为了过线往下挪**（用户裁定）：这条路是"验证"那条路，宁可多升级。
判据是"过这条线**而且**那条余量公式成立"——公式一个字没动。
"""

_GEOMETRY_JITTER: Final[int] = 3
"""复用几何时那次**小抖动搜索**的半宽（全分辨率像素）。

首帧留下的几何可能是从**粗层的位置**来的（锚点层的像素比全分辨率粗 4 倍），落点带几个
像素的量化误差；±3 的抖动把那点误差自己吃掉，并把**修正过的位置**写回资产（用户裁定：
几何自己越用越准，别把首帧当成"一次要做对"）。
"""


def _library_fingerprint(faces: Sequence[tuple[str, np.ndarray]]) -> str:
    """库的指纹：**名字 + 立绘块的每一个字节**（短哈希）。

    几何资产的键要能把「局 / 库 / 分辨率 / 卡位布局」区分开（用户裁定），库这一项靠它。
    只拿名字与形状不够 —— 同一套名字换一套皮肤（同形状、不同像素）必须算**另一把库**，
    不然就是拿旧几何去量新局面。11 条立绘 3.4 MB，blake2b 一次 ~3 ms，而它只在库那个
    对象换掉时才算一次（同一个 `faces` 对象逐帧传进来时走缓存）。
    """
    digest = hashlib.blake2b(digest_size=10)
    for name, art in faces:
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(np.ascontiguousarray(art).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SlotGeometry:
    """**一个卡位自己的对应关系**：把卡内任意一块映到立绘上那一块。

    **它不是 `(档, 位置)` 那两个数**（用户裁定）：那两个数只属于当时那块取区
    （干净卡那个 100×100 的方框），搬到另一块取区上（红罩卡那个圆盘）就不成立了 ——
    换块取区两个数都要重算。这里存的是**对应关系本身**：

        立绘坐标 = (offset_x, offset_y) + 档 × 卡内坐标

    所以**从它出发，换哪块取区、用哪个档都算得出来**（`art_top_left`）。

    **卡内坐标是"以卡片左上角为原点"，不是画布坐标**：这样它就不含卡位在画布上的位置，
    也不含"卡是几号位" —— 位号住在 `SquadGeometry` 的键上。

    **它里面没有身份**（用户裁定）：它说的是"这张卡上的画怎么摆"，与"画的是谁"无关。
    谁能被读出来，每一帧都在这份对应关系上**重新**逐条目算 —— 卡位换人、卡变红罩，
    都不会把身份带错。
    """

    scale: float
    offset_x: float
    offset_y: float

    @classmethod
    def from_position(
        cls, scale: float, position: tuple[float, float], box: tuple[float, float]
    ) -> "SlotGeometry":
        """从「某一档上、某块卡内框的模板左上角落在立绘的哪一点」**反推**对应关系。

        `box` 是那块取区在**卡内**的位置（干净卡 `(52, 76)`、红罩卡
        `(58, 70)`）—— 反推出来的对应关系与用的是哪块取区无关，这正是它可复用的原因。
        """
        return cls(
            scale=float(scale),
            offset_x=float(position[0]) - float(scale) * float(box[0]),
            offset_y=float(position[1]) - float(scale) * float(box[1]),
        )

    def art_top_left(self, box: tuple[float, float]) -> tuple[float, float]:
        """卡内框 `(bx, by)` 在立绘上对应的**模板左上角**。"""
        return (
            self.offset_x + self.scale * float(box[0]),
            self.offset_y + self.scale * float(box[1]),
        )


class SquadGeometry:
    """**一局里的可复用资产**：每个卡位 × **每条库**自己的对应关系 ＋ 那把握码尺子。

    用法（调用方持有一个，逐帧传进来）：

        assets = SquadGeometry()
        for frame in frames:
            identities = read_squad_identities(frame, faces, pitch=pitch, assets=assets)

    **第一次是每个卡位各自的第一次**（用户裁定）：开局有卡位是空的（干员上阵了），那一格
    没有卡可认、也就没有几何；等它回来时**那一格**才跑满。所以账是按**卡位下标**存的。

    **资产是「每卡位 × 每条」**（用户裁定 3）：跑满那一趟里，**每个条目自己的对位本来就已经
    算出来了**（锚点层/位置层对每个条目每一档都给一个落点），存下来几乎不要钱。只存赢家那
    一条的对应关系是错的 —— 那样"复用"那一步里**错条目会全部塌到赢家的对位上**，"同几何
    次高"也就没有意义、余量公式在那条路上不成立。每条各存各的，次高才是"这一条在**它自己**
    的对位上有多像"。

    **键 = (库指纹, 卡距, 画布尺寸) ＋ 卡位下标**（用户裁定）：任何一样变了，旧几何
    **不许**再拿来量新局面 —— `ensure` 一发现键变了就把整本账**清空**并记一次重建。
    卡位用**下标**（从最右那张数起）而不是中心点：同一局里在左头添一张卡时，右边那几张的
    中心一动都不动（实测 cd01→cd20：中心序列只差 ±1.5 像素的抖动），而从左边数起会全体错位。

    **尺子也在这里**（用户裁定：每局建一次）：`MaskedRuler` 的构造量只取决于
    (掩码, 库, 尺度网格)，与卡、帧都无关 —— 从前它挂在**帧**上（每帧建一次），现在挂在
    **局**上。没有红罩卡的局一次都不建。

    **"这一格的画面没变就整格跳过"**（用户裁定 4）：按卡位记住**上一次打分时那块取区的
    每一个字节**（`blake2b`），下一帧逐位相同就**沿用上次的判定**，不重新打分。
    **判据是画面事实（逐位相同），不是分数** —— 分数低正是"几何坏了 / 换了人"的表现，
    按它免打分等于把唯一能自愈的机制关掉。跳过的格**照样出判定**（沿用的那一个），
    而且**计数**（`counts["跳过"]`）—— 跳了几格是要报出来的。

    `counts` / `layers` / `jitter` / `upgrades` / `rebuilds` 是**量代价的口径**
    （复用了几格 · 跑满了几格 · 在哪一层被接受 · 抖动把位置修正了多少 ·
    每次升级的"前名/后名/分数" · 键变了几次），
    读数的人在这里取，不在 `read_squad_identities` 的返回值里。
    """

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._faces_id: int | None = None
        self._library: str | None = None
        self._slots: dict[int, dict[str, SlotGeometry]] = {}
        self._ruler: object | None = None
        self._seen: dict[int, tuple[str, str, SlotValue]] = {}
        self.rebuilds = 0
        self.counts: dict[str, int] = {
            "复用": 0, "升级": 0, "首帧": 0, "红罩首帧": 0, "跳过": 0,
        }
        self.layers: dict[str, int] = {
            "复用": 0, "锚点层": 0, "锚点层(掩码)": 0, "全分辨率": 0,
            "全分辨率(掩码尺子)": 0,
        }
        self.jitter: list[tuple[float, float]] = []
        self.upgrades: list[dict] = []

    def ensure(self, faces: Sequence[tuple[str, np.ndarray]], *,
               pitch: float, frame: Frame) -> None:
        """键不对就把整本账清掉 —— **不许拿旧几何去量新局面**。"""
        if id(faces) == self._faces_id:
            library = self._library
        else:
            library = _library_fingerprint(faces)
            self._faces_id = id(faces)
            self._library = library
        key = (library, float(pitch), int(frame.canvas.width), int(frame.canvas.height))
        if self._key == key:
            return
        if self._key is not None:
            self.rebuilds += 1
        self._key = key
        self._slots.clear()
        self._ruler = None
        self._seen.clear()

    def slot_of(self, index: int) -> dict[str, SlotGeometry] | None:
        """这个卡位**每一条库**的对应关系 —— 没有就是**那一格还没跑满过**。"""
        return self._slots.get(int(index))

    def geometry_of(self, index: int, name: str) -> SlotGeometry | None:
        """这个卡位、**这一条库**自己的对位 —— 没有就是这一条在这个卡位上还没量过。"""
        slot = self._slots.get(int(index))
        if slot is None:
            return None
        return slot.get(name)

    def record(self, index: int, geometries: Mapping[str, SlotGeometry]) -> None:
        """把这一格**每一条**的对位写进资产（每卡位 × 每条，用户裁定 3）。"""
        self._slots[int(index)] = dict(geometries)

    def seen(self, index: int, kind: str, digest: str) -> SlotValue | None:
        """这一格的画面与**上一次打分时**逐位相同 → 沿用那个判定（用户裁定 4）。

        判据只有"逐位相同"这一条：**不看分数**。取区种类（干净/红罩）也进键 ——
        换了一块取区，画面就不是同一块了。
        """
        remembered = self._seen.get(int(index))
        if remembered is None:
            return None
        was_kind, was_digest, value = remembered
        if was_kind != kind or was_digest != digest:
            return None
        return value

    def remember(self, index: int, kind: str, digest: str, value: SlotValue) -> None:
        """记下**这一帧实际打分时**那块取区的指纹与判定 —— 跳过的格不重记。"""
        self._seen[int(index)] = (kind, digest, value)

    def ruler(self, faces: Sequence[tuple[str, np.ndarray]]) -> object:
        """那把握码尺子 —— **一局建一次**（用户裁定），没有红罩卡的局一次都不建。"""
        if self._ruler is None:
            margin = _margin()
            _box, mask = margin.identity_disc_region()
            self._ruler = margin.MaskedRuler(mask, faces)
        return self._ruler

    def summary(self) -> dict:
        """量代价用的快照。"""
        absolute = [abs(dx) + abs(dy) for dx, dy in self.jitter]
        return {
            "counts": dict(self.counts),
            "layers": dict(self.layers),
            "rebuilds": self.rebuilds,
            "slots": len(self._slots),
            "entries": sum(len(slot) for slot in self._slots.values()),
            "jitter_n": len(self.jitter),
            "jitter_mean": float(np.mean(absolute)) if absolute else None,
            "jitter_max": float(np.max(absolute)) if absolute else None,
            "upgrades": len(self.upgrades),
            "upgrade_rows": self.upgrades[:200],
        }


def _guard_faces(faces: object) -> None:
    """库的形状闸门：**一串 `(名字, 立绘块)`**，而且**不许是空的**。

    空库不是「库里没有这个人」，是**没建库** —— 报错，不静默吐一列"未知"。
    """
    if isinstance(faces, (str, bytes)) or not isinstance(faces, Sequence):
        raise TypeError(f"库要是一串 (名字, 立绘块)，收到 {type(faces).__name__}")
    if not faces:
        raise ValueError(
            "库是空的：空库不是「库里没有这个人」，是没建库 —— "
            "静默吐一列未知与假事实一样坏"
        )
    for position, entry in enumerate(faces):
        if (
            isinstance(entry, (str, bytes))
            or not isinstance(entry, Sequence)
            or len(entry) != 2
        ):
            raise TypeError(f"库里第 {position} 条要是 (名字, 立绘块)，收到 {entry!r}")
        name, art = entry
        if not isinstance(name, str) or not name:
            raise TypeError(f"库里第 {position} 条的名字要是非空字符串，收到 {name!r}")
        if not isinstance(art, np.ndarray):
            raise TypeError(
                f"库里第 {position} 条的立绘块要是 ndarray，收到 {type(art).__name__}"
            )
        if art.dtype != np.uint8 or art.ndim != 3 or art.shape[2] != 3:
            raise TypeError(
                f"库里第 {position} 条的立绘块要是 (高, 宽, 3) 的 uint8 RGB，"
                f"收到 {art.dtype} 的 {art.shape}"
            )



def _guard_region(region: object) -> None:
    """要比的那块身份区的形状闸门（与库块的通道序必须一致：都是 RGB）。"""
    if not isinstance(region, np.ndarray):
        raise TypeError(f"要比的是一块身份区，收到 {type(region).__name__}")
    if region.dtype != np.uint8 or region.ndim != 3 or region.shape[2] != 3:
        raise TypeError(
            f"身份区要是 (高, 宽, 3) 的 uint8 RGB，收到 {region.dtype} 的 "
            f"{getattr(region, 'shape', region)}"
        )


_ANCHOR_PIXELS: Final[int] = 25
"""锚点层把身份区缩到多大。**它比 12 像素那一层大是有原因的**（用户裁定：排序锚点
不许不可信）：

    层像素   18 档里不同的模板尺寸
    12       2   （1.00–1.05 → 12、1.06–1.11 → 13 …）
    25       7
    50       13
    100      18

模板尺寸一样的那几档，`cv2.resize` 出来的模板**逐位相同**，分数也就逐位相同、整片并列；
而 `argmax` 在并列时只能取一个（从前那条取的是最小的那一档）—— 于是"粗找那一档"
**基本只知道下沿**，拿它当排序锚点等于乱序。25 像素上还有 7 个不同尺寸，够排一个序；
再细就贵了（见模块文档 §扫法的账）。
"""
_ANCHOR_POSITION_HALF: Final[int] = 12
"""位置层用的位置窗半宽（**全分辨率像素**）—— 只影响位置层找得准不准，不影响锚点档。"""
_POSITION_PIXELS: Final[int] = 50
"""位置层把身份区缩到多大 —— 它只负责给出**每个条目每一档**的位置。"""
_FINE_POSITION_HALF: Final[int] = 6
"""全分辨率那一层的位置窗半宽（**全分辨率像素**）—— 位置由上面两层给。"""


def _template_shape(region: np.ndarray, scale: float) -> tuple[int, int]:
    """某一档上模板的高与宽 —— 与旧扫法用的是同一个算式（`round(边 × 档)`）。"""
    return (
        int(round(region.shape[0] * scale)),
        int(round(region.shape[1] * scale)),
    )


def _resize_by(art: np.ndarray, factor: float) -> np.ndarray:
    """把一块图按同一个因子缩下去（粗层用它做"先粗后细"里"粗"的那一半）。"""
    return cv2.resize(
        art,
        (max(1, int(round(art.shape[1] * factor))),
         max(1, int(round(art.shape[0] * factor)))),
        interpolation=cv2.INTER_AREA,
    )


def _anchor_position(
    positions: dict[float, tuple[float, float]], scale: float
) -> tuple[float, float]:
    """上一层在**同一档**上给的位置；那一档缺了就用**最近的档**。"""
    for key, value in positions.items():
        if abs(key - scale) < 1e-9:
            return value
    return positions[min(positions, key=lambda key: abs(key - scale))]


def _window_block(
    image: np.ndarray,
    positions: dict[float, tuple[float, float]],
    scale: float,
    position_half: int | None,
    factor: float,
    width: int,
    height: int,
) -> tuple[np.ndarray, int, int]:
    """按上一层在**同一档**上给的位置裁出来的块 —— **窗贴边就往里挪，不退回整张**。

    **窗贴边是常事**：库里没有这个人时，粗层给的位置本来就是噪声峰，常常落在立绘边上。
    从前那条碰上就把整张立绘塞回去（搜索面从 `模板+12` 变成 `214×480`，一次调用贵 **20 倍**
    以上），于是"认不准的图"反而最贵。现在把窗**平移**到还在立绘里那一格 —— 窗的大小不变、
    模板一定装得下，代价也就封在 `模板 + 2×半宽` 上。

    （整张比模板还小这种配置错仍然由调用方在扫之前响亮报错，不在这里兜。）
    """
    if position_half is None or not positions:
        return image, 0, 0
    anchor = _anchor_position(positions, scale)
    half = float(position_half) * factor
    limit_x = int(image.shape[1]) - width
    limit_y = int(image.shape[0]) - height
    if limit_x < 0 or limit_y < 0:
        return image, 0, 0
    x0 = min(max(0, int(round(anchor[0] * factor - half))), limit_x)
    y0 = min(max(0, int(round(anchor[1] * factor - half))), limit_y)
    x1 = min(limit_x, max(0, int(round(anchor[0] * factor + half))))
    y1 = min(limit_y, max(0, int(round(anchor[1] * factor + half))))
    x1 = max(x0, x1)
    y1 = max(y0, y1)
    block = np.ascontiguousarray(image[y0 : y1 + height, x0 : x1 + width])
    return block, x0, y0


def _layer_rows(
    small_query: np.ndarray,
    small_art: np.ndarray,
    scales: tuple[float, ...],
    positions: dict[float, tuple[float, float]],
    position_half: int | None,
    factor: float,
) -> tuple[dict[float, tuple[float, float, float]], list[float]]:
    """**一层**的读数：`({档: (分, 全分辨率坐标 x, 全分辨率坐标 y)}, 被跳过的档)`。

    位置按**档各自带**（不是全层一个）；`position_half` 是 `None` 或还没有位置可依时搜整张。
    放不下这一层模板的档**记进跳过的名单**，不静默丢 —— 这个名单是要报出来的。

    **同一 (模板尺寸, 落点) 的档只算一次**（用户裁定 1）：模板逐位相同、块逐位相同时，
    `matchTemplate` 与 `minMaxLoc` 的结果逐位相同。这一层因此不重复付钱，**读数一个字节
    不变**（去重前后逐位相同，见 `tests/unit/test_vision_roster_geometry.py`）。
    """
    rows: dict[float, tuple[float, float, float]] = {}
    skipped: list[float] = []
    # **同一模板尺寸的档只算一次**（用户裁定 1）。模板尺寸一样时 `cv2.resize` 出来的模板
    # 逐位相同；`_window_block` 的落点也由 (尺寸, 位置, 档) 定，块一样时 `matchTemplate`
    # 与 `minMaxLoc` 的结果**逐位相同** —— 再算一遍是白算。锚点层（`positions={}`、不裁窗、
    # 落点一律 (0, 0)）上 18 个档只落在 5 个尺寸上（25/26/27/28/29，分布 2/5/3/5/3），
    # 所以那一层**13 次是白算的**。位置层上窗口随档走，所以键里带上 (x0, y0)：
    # 只有「尺寸与落点都一样」才复用，任何一处不同都照算（那两条路本来就该各算各的）。
    cache: dict[tuple[int, int, int, int], tuple[float, float, float]] = {}
    for scale in scales:
        height, width = _template_shape(small_query, scale)
        if height > small_art.shape[0] or width > small_art.shape[1]:
            skipped.append(scale)
            continue
        block, x0, y0 = _window_block(
            small_art, positions, scale, position_half, factor, width, height
        )
        key = (height, width, x0, y0)
        hit = cache.get(key)
        if hit is None:
            template = cv2.resize(
                small_query, (width, height), interpolation=cv2.INTER_LINEAR
            )
            result = cv2.matchTemplate(block, template, cv2.TM_CCOEFF_NORMED)
            _low, high, _low_loc, high_loc = cv2.minMaxLoc(result)
            hit = (
                float(high),
                (high_loc[0] + x0) / factor,
                (high_loc[1] + y0) / factor,
            )
            cache[key] = hit
        rows[scale] = hit
    return rows, skipped


def _tie_midpoint(rows: dict[float, tuple[float, float, float]]) -> float:
    """最高分那一档所在的**并列组的中点**。

    并列是粗层的常态（模板尺寸一样就逐位相同）。从前那条 `argmax` 在并列时取最小的
    那一档，于是锚点被系统性地拉到**下沿**；取组中点，锚点就**没有偏向**。
    """
    top = max(value[0] for value in rows.values())
    group = [scale for scale, value in rows.items() if value[0] >= top - 1e-12]
    return sum(group) / float(len(group))


def _order_from(anchor: float) -> tuple[float, ...]:
    """离锚点越近越先试；同样远近取小的那一档（保证次序唯一、可复现）。"""
    return tuple(sorted(IDENTITY_SCALES, key=lambda scale: (abs(scale - anchor), scale)))


def _fine_cell(
    query: np.ndarray,
    art: np.ndarray,
    scale: float,
    positions: dict[float, tuple[float, float]],
) -> tuple[float, tuple[float, float]]:
    """**全分辨率**上某一档的那个分与位置（位置裁在上一层给的窗里）—— 报出来的分就是它。"""
    height, width = _template_shape(query, scale)
    if height > art.shape[0] or width > art.shape[1]:
        return -2.0, (0.0, 0.0)
    template = cv2.resize(query, (width, height), interpolation=cv2.INTER_LINEAR)
    block, x0, y0 = _window_block(
        art, positions, scale, _FINE_POSITION_HALF, 1.0, width, height
    )
    result = cv2.matchTemplate(block, template, cv2.TM_CCOEFF_NORMED)
    _low, high, _low_loc, high_loc = cv2.minMaxLoc(result)
    return float(high), (float(high_loc[0] + x0), float(high_loc[1] + y0))


def _geometry_scores(
    query: np.ndarray,
    faces: Sequence[tuple[str, np.ndarray]],
    geometries: Mapping[str, SlotGeometry],
    box: tuple[float, float],
    *,
    jitter: int = _GEOMETRY_JITTER,
) -> tuple[tuple[float, str, float, tuple[float, float]], ...]:
    """**在已知几何上打分** —— 每条库用**它自己**那份对应关系，不再重新找位置、也不再重新找档。

    对每一条库：模板 = 取区按**这一条**资产里那一档 resize（`_template_shape`，与全分辨率
    那条路同一个算式），落点 = **这一条**的对应关系算出来的那一点（`art_top_left`），
    **只在那一点附近 ±`jitter` 的小窗里滑**。返回 `((分, 名字, 档, 位置), ...)` 降序。

    **它是"每卡位 × 每条"的**（用户裁定 3）：从前这里只收一份几何，于是每条库都被摆在
    **赢家那个位置**上 —— 错条目全塌下去，"同几何次高"因此根本不是次高。现在每条各用自己的
    对位，那个次高才是"这一条在**它自己**的对位上有多像"，余量公式在那条路上重新成立。

    **它不是另算一条统计量**：同一条库、同一个位置上，这里给的数与 `_fine_cell` 在那个窗里
    给的最大**逐位相同** —— 同一个 `TM_CCOEFF_NORMED`、同一对像素。变的只是**搜哪儿**：
    档由资产给死，位置由对应关系算出来。

    **窗贴边照样往里挪**（`_window_block`）：资产里那一档万一比某条立绘大，这里**响亮报错**，
    不静默跳过 —— 与 `roster_scores` 的形状闸门同一个口径。**缺一条的对位也响亮报错**：
    调用方要先问清楚（`_read_slot` 缺就升级），不在这里静默少算一条。
    """
    out: list[tuple[float, str, float, tuple[float, float]]] = []
    for name, art in faces:
        geometry = geometries.get(name)
        if geometry is None:
            raise ValueError(
                f"资产里没有「{name}」在这个卡位上的对位：每卡位 × 每条都要有，"
                "少一条就等于把这一条从比较里静默删掉"
            )
        scale = geometry.scale
        height, width = _template_shape(query, scale)
        if height > art.shape[0] or width > art.shape[1]:
            raise ValueError(
                f"库里「{name}」那块立绘是 {art.shape[0]}×{art.shape[1]}，"
                f"装不下 {height}×{width} 的模板（资产里那一档是 {scale}）"
            )
        template = cv2.resize(query, (width, height), interpolation=cv2.INTER_LINEAR)
        block, x0, y0 = _window_block(
            art, {scale: geometry.art_top_left(box)}, scale, jitter, 1.0, width, height
        )
        result = cv2.matchTemplate(block, template, cv2.TM_CCOEFF_NORMED)
        _low, high, _low_loc, high_loc = cv2.minMaxLoc(result)
        out.append(
            (float(high), name, scale, (float(high_loc[0] + x0), float(high_loc[1] + y0)))
        )
    out.sort(key=lambda item: (-item[0], item[1]))
    return tuple(out)


def _masked_at_geometry(
    region: np.ndarray,
    faces: Sequence[tuple[str, np.ndarray]],
    ruler: object,
    geometries: Mapping[str, SlotGeometry],
    box: tuple[float, float],
    *,
    jitter: int = _GEOMETRY_JITTER,
) -> tuple[tuple[float, str, float, tuple[float, float]], ...]:
    """**在已知几何上打分**（红罩那块取区）—— 与 `_geometry_scores` 同一个形状、同一件事。

    差别只有取区与统计量（掩码那条）；**"每条用自己的对位、只在那一点附近 ±`jitter` 里滑"
    这一件一模一样** —— 红罩那一格因此不再是"另开一条路"，它就是同一件事换一块取区。
    返回 `((分, 名字, 档, 位置), ...)` 降序。
    """
    windows = {
        name: {geometry.scale: geometry.art_top_left(box)}
        for name, geometry in geometries.items()
    }
    rows = ruler.rows(region, windows=windows, half=jitter)
    out: list[tuple[float, str, float, tuple[float, float]]] = []
    for name, _art in faces:
        geometry = geometries.get(name)
        if geometry is None:
            raise ValueError(
                f"资产里没有「{name}」在这个卡位上的对位：每卡位 × 每条都要有，"
                "少一条就等于把这一条从比较里静默删掉"
            )
        row = rows.get(name, {}).get(geometry.scale)
        if row is None:
            raise ValueError(
                f"「{name}」那一档（{geometry.scale}）在红罩那块取区上装不下：资产与取区配不上"
            )
        out.append((row[0], name, geometry.scale, (row[1], row[2])))
    out.sort(key=lambda item: (-item[0], item[1]))
    return tuple(out)


def _per_entry_best(
    rows_by_entry: dict[str, dict[float, tuple[float, float, float]]],
) -> dict[str, tuple[float, float, float]]:
    """**每个条目自己**最像的那一档 → `{名字: (档, 落点 x, 落点 y)}`。

    并列时取小的那一档（按 `(分, -档)` 取最大），与 `_layer_reading` 同一个取法。
    **资产的原料就是这个**（用户裁定 3：跑满那一趟里每个条目自己的对位本来就已经算出来了）。
    一档都放不下的条目**不进这份账** —— 它在取区上根本没有位置，硬塞一个就是编。
    """
    out: dict[str, tuple[float, float, float]] = {}
    for name, rows in rows_by_entry.items():
        if not rows:
            continue
        scale = max(rows, key=lambda item: (rows[item][0], -item))
        score, x, y = rows[scale]
        out[name] = (scale, x, y)
    return out


def _layer_reading(
    rows_by_entry: dict[str, dict[float, tuple[float, float, float]]],
) -> tuple[tuple[float, str, float, tuple[float, float]], ...]:
    """**一层**的读数：每个条目在最像的那一档上的 `(分, 名字, 档, 位置)`，降序。

    粗层直接接受（用户裁定）要的就是这个 —— 每个条目在这一层上"最像到什么程度"，
    与全分辨率那一步**同一个形状**（那边是每个条目在全分辨率上滑出来的最大）。
    """
    out: list[tuple[float, str, float, tuple[float, float]]] = []
    for name, (scale, x, y) in _per_entry_best(rows_by_entry).items():
        out.append((rows_by_entry[name][scale][0], name, scale, (x, y)))
    out.sort(key=lambda item: (-item[0], item[1]))
    return tuple(out)


def _ranked(
    reading: tuple[tuple[float, str, float, tuple[float, float]], ...],
) -> tuple[tuple[float, str, float], ...]:
    """把一层/一步的读数削成公开那个形状：`((分, 名字, 档), ...)`。"""
    return tuple((score, name, scale) for score, name, scale, _position in reading)


def _winner(
    ranked: tuple[tuple[float, str, float], ...],
    where: dict[str, tuple[float, float]],
) -> dict:
    """定下这次读数的那个条目：名字 · 档 · 位置 · 分 · 次高 —— **位置是几何资产的原料**。"""
    top, name, scale = ranked[0]
    runner = ranked[1][0] if len(ranked) > 1 else -2.0
    return {"name": name, "scale": scale, "position": where[name], "top": top,
            "runner": runner}


def _layered_scores(
    region: np.ndarray,
    faces: Sequence[tuple[str, np.ndarray]],
    *,
    coarse_accept: bool = True,
) -> tuple[tuple[tuple[float, str, float], ...], dict]:
    """按裁定那几条扫一遍 —— → `(((分, 名字, 档), ...), 代价足迹)`。

    **三条**（用户裁定）：

    1. 试的档**扩到整段**（`IDENTITY_SCALES`）—— 没有再"只试 ±2 档"这回事；
    2. 次序 = **离粗找那一档越近越先试**（`_order_from`）；
    3. **判据满足即停**（`identity_is_exact`）—— 好认的图头一两档就过，认不准的才花到底。

    **粗层直接接受**（用户裁定，`coarse_accept=True`）：锚点层算完之后，只要最高分过了
    `ROSTER_COARSE_ACCEPT_SCORE`（**0.9，比全分辨率那条 0.85 严**）**而且**判据成立，
    就在这一层判精确 —— 位置层与全分辨率那一步**一次都不跑**。

    **只有锚点层能直接接受**：位置层要拿锚点给的位置裁一个 ±12 像素的窗，而**裁窗压的是
    "只压次高"** —— 那才是把余量线弄弱的方向（实测 +0.13，带来假精确）。锚点层在整块立绘上
    滑、不裁窗，所以它的次高是"往里挤满格"挤出来的、只会更高，同一条余量公式在那里
    **更严、不是更松**。

    **报出的分是第一个满足判据的那一档（或那一层）上的分**，不是全局最大分。整段走完仍然
    不满足的（库里没有这个人），报的是**每一档走下来每个条目的最大分**。

    第二部分是代价足迹（锚点 · 每层试了哪些档 / 跳了哪些档 / 挑了哪一档 · 全分辨率试了几档 ·
    定下读数的那个条目的档与位置）—— **只给量代价的人用**，`roster_scores` 不碰它。

    `coarse_accept=False` 是**那张网的参照**："全分辨率跑满"那条路 —— 粗层不接，一路走到
    全分辨率那一步。逐张对比就是拿它比的（见 `tests/帧样本/先粗后细/README.md` §第五单）。
    """
    query = np.ascontiguousarray(region)
    trace: dict = {}
    trace["coarse_accept"] = bool(coarse_accept)
    # **按层拆解**（用户裁定 3）：每一层的墙钟落进 `trace["cost"]`。它是「还能不能再压」
    # 唯一的依据 —— 逐条对比只回答「答案变没变」，不回答「钱花在哪儿」。
    # 计时的开销是每格几次 `perf_counter`，相对一格 0.2 s 是可以忽略的那一档。
    cost: dict[str, float] = {}
    trace["cost"] = cost
    _t = time.perf_counter()

    # ---- 1) 锚点层：范围铺满整段，不裁位置；锚点取并列组的中点 ----
    anchor_factor = _ANCHOR_PIXELS / float(query.shape[0])
    anchor_query = _resize_by(query, anchor_factor)
    anchor_rows: dict[str, dict[float, tuple[float, float, float]]] = {}
    anchor_skip: list[float] = []
    for name, art in faces:
        rows, skipped = _layer_rows(
            anchor_query, _resize_by(art, anchor_factor), IDENTITY_SCALES, {}, None,
            anchor_factor,
        )
        anchor_rows[name] = rows
        if skipped and not anchor_skip:
            anchor_skip = list(skipped)
    leader = max(
        faces,
        key=lambda entry: max(value[0] for value in anchor_rows[entry[0]].values()),
    )[0]
    anchor = _tie_midpoint(anchor_rows[leader])
    cost["锚点层"] = time.perf_counter() - _t
    trace["anchor"] = {"pixels": _ANCHOR_PIXELS, "leader": leader, "scale": anchor,
                       "skipped": anchor_skip, "row_scales": sorted(anchor_rows[leader])}

    # ---- 1b) 粗层直接接受：锚点层就满足判据（分线 0.9）的话，一步都不再往下走 ----
    trace["coarse"] = None
    if coarse_accept:
        reading = _layer_reading(anchor_rows)
        ranked = _ranked(reading)
        top, name, scale = ranked[0]
        runner = ranked[1][0] if len(ranked) > 1 else -2.0
        if top >= ROSTER_COARSE_ACCEPT_SCORE and identity_is_exact(top, runner):
            trace["coarse"] = {"layer": "锚点层", "pixels": _ANCHOR_PIXELS,
                               "top": top, "runner": runner, "scale": scale}
            trace["fine"] = {"tried": [], "stopped_at": None, "satisfied": True,
                             "layer": "锚点层", "position_half": None}
            trace["winner"] = _winner(
                ranked, {entry[1]: entry[3] for entry in reading}
            )
            trace["geometry"] = _per_entry_best(anchor_rows)
            return ranked, trace

    # ---- 2) 位置层：每个条目每一档一个位置 ----
    _t = time.perf_counter()
    position_factor = _POSITION_PIXELS / float(query.shape[0])
    position_query = _resize_by(query, position_factor)
    positions: dict[str, dict[float, tuple[float, float]]] = {}
    position_rows: dict[str, dict[float, tuple[float, float, float]]] = {}
    position_skip: list[float] = []
    for name, art in faces:
        rows, skipped = _layer_rows(
            position_query, _resize_by(art, position_factor), IDENTITY_SCALES,
            {scale: (value[1], value[2]) for scale, value in anchor_rows[name].items()},
            _ANCHOR_POSITION_HALF, position_factor,
        )
        positions[name] = {scale: (value[1], value[2]) for scale, value in rows.items()}
        position_rows[name] = rows
        if skipped and not position_skip:
            position_skip = list(skipped)
    cost["位置层"] = time.perf_counter() - _t
    trace["position"] = {"pixels": _POSITION_PIXELS, "position_half": _ANCHOR_POSITION_HALF,
                         "skipped": position_skip,
                         "covered": sorted(positions[leader])}

    # ---- 3) 全分辨率：按距离试档，判据满足即停 ----
    _t = time.perf_counter()
    order = _order_from(anchor)
    best: dict[str, tuple[float, float, tuple[float, float]]] = {
        name: (-2.0, IDENTITY_SCALES[0], (-1.0, -1.0)) for name, _art in faces
    }
    tried: list[float] = []
    for scale in order:
        tried.append(scale)
        cells = {
            name: _fine_cell(query, art, scale, positions[name])
            for name, art in faces
        }
        for name, (value, where) in cells.items():
            if value > best[name][0]:
                best[name] = (value, scale, where)
        step = tuple(
            (value, name, scale)
            for name, (value, _where) in sorted(
                cells.items(), key=lambda item: (-item[1][0], item[0])
            )
        )
        top = step[0][0]
        runner = step[1][0] if len(step) > 1 else -2.0
        if identity_is_exact(top, runner):
            cost["精算层"] = time.perf_counter() - _t
            trace["fine"] = {"tried": tried, "stopped_at": scale, "satisfied": True,
                             "layer": "全分辨率", "position_half": _FINE_POSITION_HALF}
            trace["winner"] = _winner(
                step, {name: where for name, (_value, where) in cells.items()}
            )
            geometry = _per_entry_best(position_rows)
            winner_name = step[0][1]
            geometry[winner_name] = (scale, cells[winner_name][1][0], cells[winner_name][1][1])
            trace["geometry"] = geometry
            return step, trace
    cost["精算层"] = time.perf_counter() - _t
    trace["fine"] = {"tried": tried, "stopped_at": None, "satisfied": False,
                     "layer": "全分辨率", "position_half": _FINE_POSITION_HALF}
    ranked = tuple(
        (value, name, scale)
        for name, (value, scale, _where) in sorted(
            best.items(), key=lambda item: (-item[1][0], item[0])
        )
    )
    trace["winner"] = _winner(
        ranked, {name: where for name, (_value, _scale, where) in best.items()}
    )
    geometry = _per_entry_best(position_rows)
    winner_name = ranked[0][1]
    _value, winner_scale, winner_where = best[winner_name]
    geometry[winner_name] = (winner_scale, winner_where[0], winner_where[1])
    trace["geometry"] = geometry
    return ranked, trace

def _masked_layered_scores(
    region: np.ndarray,
    ruler: object,
    *,
    coarse_accept: bool = True,
) -> tuple[tuple[tuple[float, str, float], ...], dict]:
    """红罩那块取区上的**同一条扫法** —— 锚点层 → 位置层 → 全分辨率，判据满足即停。

    **这一刀就是"掩码那条路也吃到锚点带来的搜索收窄"**（用户裁定 1）：掩码那条路从前
    **整张滑**（18 档 × 每条库、全分辨率），一卡 2.758 s；干净那条路靠锚点把范围收窄了，
    它没有。这里把干净那条路那三层**原样搬过来**，只是每一层都在掩码上算
    （`MaskedRuler.rows(pixels=…)`）—— 层像素、位置窗半宽、判据、停法**一个数都没改**。

    **粗层不裁位置窗**（用户裁定 7）：锚点那一层在**整张立绘**上滑（`windows=None`）；
    位置窗只从位置层起才有。粗层直接接受也照旧**只发生在锚点层** —— 裁窗压的是"只压次高"，
    那是把余量线弄弱的唯一方向，红罩这条路也不例外。

    `trace` 与 `_layered_scores` **同一个形状**（`anchor` / `coarse` /
    `position` / `fine` / `winner` / `geometry` / `cost`），多一个 `"masked": True`
    说明取区是圆盘那块。**`geometry` 是每条库自己那一份**（每卡位 × 每条，裁定 3）。
    `trace["cost"]` 是**按层拆解**（用户裁定 3）：`锚点层` / `位置层` / `精算层` 三段的墙钟，
    与干净那条路同一套键 —— 它是「还能不能再压」唯一的依据。

    **报出来的分是第一个满足判据的那一档上的分，不是全局最大分**（用户裁定 9）——
    与干净那条路同一条口径。它进了槽值的 `confidence`，所以那句话在
    `read_squad_identities` 的文档里也写着：**不许当下游的"全局极值"用**。
    """
    query = np.ascontiguousarray(region)
    trace: dict = {"coarse_accept": bool(coarse_accept), "masked": True}
    # **按层拆解**（用户裁定 3）—— 与干净那条路同一个形状，见 `_layered_scores`。
    cost: dict[str, float] = {}
    trace["cost"] = cost
    _t = time.perf_counter()

    # ---- 1) 锚点层：整张滑，不裁位置窗 ----
    anchor_rows = ruler.rows(query, pixels=_ANCHOR_PIXELS)
    leader = max(
        anchor_rows,
        key=lambda name: max(
            (row[0] for row in anchor_rows[name].values()), default=-2.0
        ),
    )
    anchor = _tie_midpoint(anchor_rows[leader])
    cost["锚点层"] = time.perf_counter() - _t
    trace["anchor"] = {"pixels": _ANCHOR_PIXELS, "leader": leader, "scale": anchor,
                       "skipped": [], "row_scales": sorted(anchor_rows[leader])}

    # ---- 1b) 粗层直接接受：锚点层就满足判据（分线 0.9）的话，一步都不再往下走 ----
    trace["coarse"] = None
    if coarse_accept:
        reading = _layer_reading(anchor_rows)
        ranked = _ranked(reading)
        top, name, scale = ranked[0]
        runner = ranked[1][0] if len(ranked) > 1 else -2.0
        if top >= ROSTER_COARSE_ACCEPT_SCORE and identity_is_exact(top, runner):
            trace["coarse"] = {"layer": "锚点层(掩码)", "pixels": _ANCHOR_PIXELS,
                               "top": top, "runner": runner, "scale": scale}
            trace["fine"] = {"tried": [], "stopped_at": None, "satisfied": True,
                             "layer": "锚点层(掩码)", "position_half": None}
            trace["winner"] = _winner(
                ranked, {entry[1]: entry[3] for entry in reading}
            )
            trace["geometry"] = _per_entry_best(anchor_rows)
            return ranked, trace

    # ---- 2) 位置层：每个条目每一档一个位置（窗由锚点层给） ----
    _t = time.perf_counter()
    position_rows = ruler.rows(
        query,
        pixels=_POSITION_PIXELS,
        windows={name: {scale: (row[1], row[2]) for scale, row in rows.items()}
                 for name, rows in anchor_rows.items()},
        half=_ANCHOR_POSITION_HALF,
    )
    cost["位置层"] = time.perf_counter() - _t
    trace["position"] = {"pixels": _POSITION_PIXELS, "position_half": _ANCHOR_POSITION_HALF,
                         "skipped": [], "covered": sorted(position_rows[leader])}

    # ---- 3) 全分辨率：按距离试档，判据满足即停 ----
    _t = time.perf_counter()
    order = _order_from(anchor)
    best: dict[str, tuple[float, float, tuple[float, float]]] = {
        name: (-2.0, IDENTITY_SCALES[0], (-1.0, -1.0)) for name in anchor_rows
    }
    tried: list[float] = []
    for scale in order:
        tried.append(scale)
        windows = {
            name: {scale: position_rows[name][scale][1:]}
            for name in position_rows
            if scale in position_rows[name]
        }
        rows = ruler.rows(query, only=scale, windows=windows, half=_FINE_POSITION_HALF)
        cells: dict[str, tuple[float, tuple[float, float]]] = {}
        for name in anchor_rows:
            row = rows.get(name, {}).get(scale)
            cells[name] = (-2.0, (0.0, 0.0)) if row is None else (row[0], (row[1], row[2]))
        for name, (value, where) in cells.items():
            if value > best[name][0]:
                best[name] = (value, scale, where)
        step = tuple(
            (value, name, scale)
            for name, (value, _where) in sorted(
                cells.items(), key=lambda item: (-item[1][0], item[0])
            )
        )
        top = step[0][0]
        runner = step[1][0] if len(step) > 1 else -2.0
        if identity_is_exact(top, runner):
            cost["精算层"] = time.perf_counter() - _t
            trace["fine"] = {"tried": tried, "stopped_at": scale, "satisfied": True,
                             "layer": "全分辨率(掩码尺子)",
                             "position_half": _FINE_POSITION_HALF}
            trace["winner"] = _winner(
                step, {name: where for name, (_value, where) in cells.items()}
            )
            geometry = _per_entry_best(position_rows)
            winner_name = step[0][1]
            geometry[winner_name] = (
                scale, cells[winner_name][1][0], cells[winner_name][1][1]
            )
            trace["geometry"] = geometry
            return step, trace
    cost["精算层"] = time.perf_counter() - _t
    trace["fine"] = {"tried": tried, "stopped_at": None, "satisfied": False,
                     "layer": "全分辨率(掩码尺子)", "position_half": _FINE_POSITION_HALF}
    ranked = tuple(
        (value, name, scale)
        for name, (value, scale, _where) in sorted(
            best.items(), key=lambda item: (-item[1][0], item[0])
        )
    )
    trace["winner"] = _winner(
        ranked, {name: where for name, (_value, _scale, where) in best.items()}
    )
    geometry = _per_entry_best(position_rows)
    winner_name = ranked[0][1]
    _value, winner_scale, winner_where = best[winner_name]
    geometry[winner_name] = (winner_scale, winner_where[0], winner_where[1])
    trace["geometry"] = geometry
    return ranked, trace


def _guard_a_scale_fits(
    region: np.ndarray,
    faces: Sequence[tuple[str, np.ndarray]],
    scales: Sequence[float] = IDENTITY_SCALES,
) -> None:
    """形状闸门：**每一条库至少有一档装得下** —— 一档都装不下才是配置错。

    **不是"每一档都要装得下"**（区间扩到 `1.00…1.17` 之后改的口径）：一块比卡面窄的
    取区（`余量` 那一份里那条 **186 宽**的横带就是）装不下最高那几档 —— 那是**取区窄**，
    不是库坏。装不下的档在扫的时候**跳过并记进名单**（`_layer_rows` 的 `skipped`、
    `_fine_cell` 的 `-2.0`），不静默；只有"连最小那一档都装不下"才是
    "库块比一整块备战席卡还小"那种配置错。
    """
    for name, art in faces:
        for scale in scales:
            height, width = _template_shape(region, scale)
            if height <= art.shape[0] and width <= art.shape[1]:
                break
        else:
            height, width = _template_shape(region, scales[0])
            raise ValueError(
                f"库里「{name}」那块立绘是 {art.shape[0]}×{art.shape[1]}，"
                f"一档都装不下（最小那一档就要 {height}×{width} 的模板）："
                "库块至少要有一整块备战席卡那么大"
            )


def roster_scores(
    region: np.ndarray,
    faces: Sequence[tuple[str, np.ndarray]],
) -> tuple[tuple[float, str, float], ...]:
    """一块身份区对库里**每一条**的最高分 —— `((分数, 名字, 定下它的尺度), ...)`，按分数降序。

    **这是判据本身**（**干净卡**那一格用的就是它；红罩卡那一格用的是
    `margin.MaskedRuler` —— **同一条统计量、换一块取区**，见模块文档 §两块取区），
    所以它是公开的：「红罩反解后剩多少信息」那件实测要拿**同一条链的分数**当客观量，
    不能自己另写一份匹配 —— 另写一份，量的就是另一条链了。

    **分数是 `TM_CCOEFF_NORMED`**（零均值归一化相关，在 RGB 三通道上算），
    **在尺度网格上滑着找位置**取最大：一个库条目给一条，尺度的选择与位置的选择
    是**同一个 max**。

    **降序是"最像的在前"**：判据要的是"最像的那个有多像、以及它与次像的差多少"，
    所以次序不是装饰 —— `scores[0]` 与 `scores[1]` 就是判据的两半。

    **报出来的分是第一个满足判据的那一档上的分**，不是全局最大分（用户裁定）：扫法按
    "离粗找那一档越近越先试"挨个档算，**判据一满足就停**（见模块文档 §扫法）。
    所以同一个 (卡, 库条目) 对，**报出来的分数会随库里还有谁而变** —— 判据是**这一张卡**上
    最高与次高之间的关系，不是每个条目各自的全局最大。

    **"逐条目独立"这件事被有意换掉了**：换来的是"好认的图头一两档就过、认不准的才花到底"。
    实测（`停即扫.json`，350 张卡）：一次就停 209 张、走满 18 档 140 张。

    **分与穷举不等**（用户裁定：旧那条"逐对分数不许低于穷举"作废），但**判定与名字一个都没变**
    —— 对全语料 350 张卡，读出来的名字变了 **0 张**、丢 **0** 张、新错 **0** 张。

    **它走的是线上那条扫法，含"粗层直接接受"**：锚点层上分线过了 `0.9`
    （`ROSTER_COARSE_ACCEPT_SCORE`，比全分辨率那条 0.85 严）而且判据成立，就在那一层
    判精确、位置层与全分辨率那一步一次都不跑。所以"报出来的分"**有时候是锚点层那个分** ——
    它比全分辨率那个数更严（粗层的次高是挤出来的、只会更高）。全语料 **265 张干净卡**实测：
    锚点层接受 **210 张** · 全分辨率 **55 张** · 两者给出的**（判定, 名字）一张都不同**
    （`tests/帧样本/先粗后细/几何复用.json` §coarse）。
    """
    _guard_faces(faces)
    _guard_region(region)
    _guard_a_scale_fits(region, faces)
    query = np.ascontiguousarray(region)
    scores, _trace = _layered_scores(query, faces)
    return scores



def _redcap_ruler(faces: Sequence[tuple[str, np.ndarray]]):
    """那把掩码尺子 —— 不走资产时的一把（`squad_card_scores` 那条路用）。见 `SquadGeometry.ruler`。"""
    margin = _margin()
    _box, mask = margin.identity_disc_region()
    return margin.MaskedRuler(mask, faces)


def _pair(ranked: tuple[tuple[float, str, float], ...]) -> tuple[float, float]:
    """判据的两半：最高分与次高分（库里只有一条时次高是 `-2.0`）。"""
    return ranked[0][0], (ranked[1][0] if len(ranked) > 1 else -2.0)


def _geometry_fits(region: np.ndarray, art: np.ndarray, scale: float) -> bool:
    """这一档的模板**装得下**这条立绘吗 —— 与 `_template_shape` / 形状闸门同一口径。

    复用那条路要拿它先问一句：资产里那一档是从**另一块取区**（红罩那个圆盘比干净那个方框小）
    反推出来的时候，换回大方框可能就装不下了。**装不下就升级**（退回跑满），不在这里报错 ——
    报错会把整帧带走，而那只是"这一格这一次不比"。
    """
    height, width = _template_shape(region, scale)
    return height <= art.shape[0] and width <= art.shape[1]


def _exact_name(value: SlotValue) -> str | None:
    """槽值里那个名字 —— 未知就是 `None`（**不产否定存在断言**：只回答"这次读出的是什么"）。"""
    payload = getattr(value, "payload", None)
    if isinstance(payload, Exact):
        return str(payload.value)
    return None


def _reuse_scores(
    region: np.ndarray,
    faces: Sequence[tuple[str, np.ndarray]],
    stored: Mapping[str, SlotGeometry],
    box: tuple[float, float],
    *,
    red: bool,
    ruler: object | None,
) -> tuple[tuple[float, str, float, tuple[float, float]], ...]:
    """复用那条路的读数：**每条库在它自己的对位上**那一个分与落点。

    干净卡与红罩卡的区别只有取区与统计量（`_geometry_scores` / `_masked_at_geometry`）；
    "每条用自己的对位、只在那一点附近 ±`_GEOMETRY_JITTER` 里滑"这一件**两处一模一样**。
    """
    if red:
        if ruler is None:
            raise ValueError("红罩卡复用几何要那把掩码尺子：调用方没给")
        return _masked_at_geometry(region, faces, ruler, stored, box)
    return _geometry_scores(region, faces, stored, box)


def _read_slot(
    region: np.ndarray,
    faces: Sequence[tuple[str, np.ndarray]],
    *,
    index: int,
    red: bool,
    box: tuple[float, float],
    stored: Mapping[str, SlotGeometry] | None,
    assets: SquadGeometry | None,
    ruler: object | None = None,
) -> tuple[SlotValue, dict]:
    """**一个卡位的一次读数** —— 复用几何 / 跑满在这里分岔，**判据只有一处**。

    分岔只决定"这块取区上每一条库拿多少分、在哪一档、落在哪"，**判定那一步两条路共用**
    （`identity_is_exact`）—— 所以"复用会不会把名字带错"这件事只有一处可错。

    **复用那条路不自信就升级**（用户裁定）：分数没到线、或者余量没拉开，就把**那一格**
    退回跑满，并把新几何写回资产。升级是唯一的兜底，所以它被计数（`assets.counts`），
    而且**每一次都留账**（`assets.upgrades`：升级前的名字或未知 · 升级后的名字 ·
    当时的最高分）—— 那是"卡位换人"这件事唯一的账，也是"复用路会不会答案发霉"唯一的证据。

    **复用那条路有自己的分线**（用户裁定 2）：`ROSTER_REUSE_MATCH_SCORE` 坐在那条路
    **自己的空档**里（它只在一个档 + ±`_GEOMETRY_JITTER` 上比，错条目塌不下去），
    再叠上用户裁定那条余量公式 —— 公式一个字没动。

    **资产不全就升级**：某一条库在这个卡位上没有对位（它一档都放不下），那就不在这里比 ——
    少一条等于把这一条从比较里静默删掉，宁可退回跑满（`trace["missing"]` 记着是谁）。
    """
    trace: dict = {"index": index, "red": bool(red), "how": None}
    reuse_top: float | None = None
    reuse_name: str | None = None
    if stored is not None:
        missing = [
            name for name, art in faces
            if name not in stored or not _geometry_fits(region, art, stored[name].scale)
        ]
        if missing:
            trace["missing"] = missing
        else:
            _t = time.perf_counter()
            reading = _reuse_scores(region, faces, stored, box, red=red, ruler=ruler)
            trace["cost"] = {"复用": time.perf_counter() - _t}
            ranked = _ranked(reading)
            where = {name: position for _score, name, _scale, position in reading}
            trace["how"] = "复用"
            if ranked:
                top, runner = _pair(ranked)
                reuse_top, reuse_name = top, ranked[0][1]
                trace["scale"] = stored[ranked[0][1]].scale
                trace["top"] = top
                # **次高改名**（用户裁定 2）：这是"同几何次高"，不是全扫那条路的次高。
                trace["runner_at_geometry"] = runner
                if top >= ROSTER_REUSE_MATCH_SCORE and identity_is_exact(top, runner):
                    fresh = {
                        name: SlotGeometry.from_position(scale, where[name], box)
                        for _score, name, scale, _p in reading
                    }
                    winner = ranked[0][1]
                    winner_d = (
                        fresh[winner].offset_x - stored[winner].offset_x,
                        fresh[winner].offset_y - stored[winner].offset_y,
                    )
                    trace["geometry"] = {
                        "offset": (fresh[winner].offset_x, fresh[winner].offset_y),
                        "d_scale": 0.0,
                        "d_offset": winner_d,
                        "d_offset_all": {
                            name: (fresh[name].offset_x - stored[name].offset_x,
                                   fresh[name].offset_y - stored[name].offset_y)
                            for name in fresh
                        },
                    }
                    if assets is not None:
                        assets.counts["复用"] += 1
                        assets.layers["复用"] += 1
                        assets.record(index, fresh)
                        assets.jitter.append(winner_d)
                    return Valued(Exact(ranked[0][1]), top), trace
        if assets is not None:
            assets.counts["升级"] += 1
        trace["upgrade"] = True

    # ---- 跑满：首帧（这一格还没有几何）或升级兜底 ----
    if red:
        if ruler is None:
            raise ValueError("红罩卡跑满要那把掩码尺子：调用方没给")
        ranked, walk = _masked_layered_scores(region, ruler)
    else:
        ranked, walk = _layered_scores(region, faces)
    if not ranked:
        raise ValueError(
            "库里的立绘一个都装不下这块取区：这块取区与库配不上"
        )
    layer = walk["fine"]["layer"]
    trace["how"] = "跑满" if stored is not None else ("红罩首帧" if red else "首帧")
    trace["layer"] = layer
    # **跑满那一趟的按层墙钟并进这一格的足迹**（用户裁定 3）：复用那一段的账在上面已经
    # 记过（键是 `复用`），这里的键是 `锚点层` / `位置层` / `精算层`，两套键不重名。
    # 不并的话，升级的格子只剩"复用那一段"的账、首帧的格子一条账都没有 —— 拆解就断了。
    cost = dict(trace.get("cost") or {})
    cost.update(walk.get("cost") or {})
    trace["cost"] = cost
    top, runner = _pair(ranked)
    trace["top"], trace["runner"] = top, runner
    exact = identity_is_exact(top, runner)
    if assets is not None:
        if stored is None:
            assets.counts["红罩首帧" if red else "首帧"] += 1
        if exact:
            # **每条库自己的对位**都写进资产（每卡位 × 每条，用户裁定 3）。
            # **只在这一格真读出来的时候写**：对位要是一份**验过**的对应关系才敢复用 ——
            # 判成未知时那几条的对位是"最像的巧合"，拿它去当几何，下一帧就会拿巧合当证据。
            assets.record(
                index,
                {
                    name: SlotGeometry.from_position(scale, (x, y), box)
                    for name, (scale, x, y) in walk["geometry"].items()
                },
            )
            assets.layers[layer] += 1
        if stored is not None:
            assets.upgrades.append({
                "index": int(index), "red": bool(red),
                "before": reuse_name, "before_top": reuse_top,
                "after": ranked[0][1] if exact else None,
                "after_top": top, "layer": layer,
                "missing": trace.get("missing"),
            })
    if exact:
        return Valued(Exact(ranked[0][1]), top), trace
    return Valued(Unknown(), None), trace


def read_squad_identities(
    frame: Frame,
    faces: Sequence[tuple[str, np.ndarray]],
    *,
    pitch: float,
    assets: SquadGeometry | None = None,
    trace: list | None = None,
) -> tuple[SlotValue, ...]:
    """读备战席那一栏：**一帧 → 每个卡位一个身份**（下标 0 = 最右那张）。

    `faces` 是**库**（一串 `(名字, 立绘块)`，按编队页的切割顺序）；`pitch` 是
    **卡距**（标量）—— 它住设备层的 `SQUAD_CARD_PITCH`，而导入规则不许视觉层
    import 设备层，所以**由调用方传进来**。

    **卡位的个数与顺序跟 `squad_card_centres` 一致**，不许另开一套定位：有卡的那些
    格**一定是连着的一片**（中间缺一格说明那一格的跳变没读出来、中心落回网格），
    所以数出几个卡位就吐几个值。

    **只有两种形态**：

    - 认出来 → `Valued(Exact(名字), 最高分)`（置信度就是**定下这次认出的那个分数**）；
    - 认不出 → `Valued(Unknown(), None)`。**红罩卡也走这一支**：它换一块取区重比
      （§两块取区），读得出就是精确、读不出就是未知，**不为它另开形态**。`None`
      不是"零分"，是"这次读数不是一个分数定下来的" —— 未知本来就没有分数。

    **`confidence` 的含义（用户裁定 9，红罩那条路上尤其要看清）**：它是**第一个满足
    判据的那一档（或那一层）上的分**，**不是全局最大分**。干净卡与红罩卡都是这一条；
    红罩卡从前是"整张滑 18 档、报全局最大"，现在是"锚点 → 位置 → 全分辨率，判据一满足
    就停"——**同一个名字上的那个数会变小**（实测 `cd14` 那一格：0.991018 → 0.943155，
    名字仍是斩业星熊）。所以**下游不许把它当"这一格像到什么程度"的全局极值用**：
    它回答的是"判据在哪儿被满足"，不是"最像能有多像"。要全局极值请自己另量一条，
    那不是这一层的事。

    **红罩卡那一格换的是取区，不是判据**：判"是不是再部署中"走 `bench.redeploying`
    （判据只有那一处），是 `True` 的才走 `redcap_region` + `margin.MaskedRuler`；
    **其余一律照旧**（含 `redeploying` 判不出来的那一档）。


    **`assets` 是"一局里的可复用资产"**（用户裁定，见 `SquadGeometry`）：给了它，
    每个卡位**第一次**跑满之后就把**每一条库自己的对位**留成资产，之后每帧**只在那块几何上
    打分** —— 不再重新找位置、找档。不给（`None`）就是每一帧都跑满，也就是从前那条路。
    几何自己越用越准：复用那一趟会做一次 ±`_GEOMETRY_JITTER` 的小抖动，把修正过的
    位置写回资产；**不自信就升级**（那一格退回跑满），升级被计数、而且每次留账
    （`assets.upgrades`：前名 · 后名 · 当时的最高分）。

    **这一格的画面与上一次打分时逐位相同就整格跳过**（用户裁定 4）：沿用上次的判定，
    不重新打分 —— **判据是画面事实，不是分数**。跳过的格照样出判定（`trace` 里记着
    沿用的是哪一个），而且计数（`counts["跳过"]`）。

    **`trace` 只给量代价的人用**：给一个 list 就把每个卡位那一次的足迹追加进去
    （复用还是跑满 · 在哪一层被接受 · 分与次高 · 几何动了多少）。`None` 时一个都不记。

    **没有「未采样」**："这一格没卡"由**卡位**那一层回答（一个卡位一个值），不是
    身份的事。

    **库里只有一条时没有次高分可比**，判据退化成只看最高分 —— 那一条自己就是
    "最像的"，余量这个量不存在。库是**本局队伍**，实际总是好几条。
    """
    if not isinstance(frame, Frame):
        raise TypeError(f"要读的是一帧，收到 {type(frame).__name__}")
    _guard_faces(faces)
    if isinstance(pitch, bool) or not isinstance(pitch, (int, float)):
        raise TypeError(f"卡距要是数，收到 {type(pitch).__name__}")
    if pitch <= 0:
        raise ValueError(f"卡距必须为正，收到 {pitch}")
    if assets is not None:
        assets.ensure(faces, pitch=pitch, frame=frame)

    identity_box = (float(IDENTITY_BOX_X), float(IDENTITY_BOX_Y))
    disc_box = _margin().IDENTITY_DISC_BOX
    redcap_box = (float(disc_box[0]), float(disc_box[1]))
    identities: list[SlotValue] = []
    card_top = float(squad_card_rect(0, pitch).y)
    redcap_ruler = None
    centres = squad_card_centres(frame, pitch=pitch)
    for index, (centre, _) in enumerate(centres):
        red = redeploying(frame, centre, card_top=card_top) is True
        box = redcap_box if red else identity_box
        rect = (
            redcap_region(centre, pitch=pitch)
            if red
            else identity_region(centre, pitch=pitch)
        )
        region = np.ascontiguousarray(frame.crop(rect))

        kind = "红罩" if red else "干净"
        digest = hashlib.blake2b(region.tobytes(), digest_size=10).hexdigest()
        if assets is not None:
            remembered = assets.seen(index, kind, digest)
            if remembered is not None:
                # **画面没变就整格跳过**（用户裁定 4）：判据是逐位相同，不是分数。
                assets.counts["跳过"] += 1
                if trace is not None:
                    trace.append({"index": index, "red": bool(red), "how": "跳过",
                                  "cost": {"跳过": 0.0},
                                  "reused": _exact_name(remembered),
                                  "verdict": ("精确" if _exact_name(remembered)
                                              is not None else "未知")})
                identities.append(remembered)
                continue
        stored = assets.slot_of(index) if assets is not None else None
        if red and redcap_ruler is None:
            redcap_ruler = (
                assets.ruler(faces) if assets is not None else _redcap_ruler(faces)
            )
        value, slot_trace = _read_slot(
            region, faces, index=index, red=red, box=box, stored=stored, assets=assets,
            ruler=redcap_ruler,
        )
        if assets is not None:
            assets.remember(index, kind, digest, value)
        if trace is not None:
            trace.append(slot_trace)
        identities.append(value)
    return tuple(identities)


def _entry_for(
    faces: Sequence[tuple[str, np.ndarray]], name: str
) -> tuple[str, np.ndarray]:
    """库里名字对上的那**一条** —— 对不上就响亮报错。

    **「库里没有这个名字」是配置错，不是「他不在备战席上」**（与「空库要响亮」
    同一条）：静默吐一个「没能指出」就是把一次拿错库伪装成一次正常读数。所以这里
    是 `KeyError` —— 与 `bench._anchors` 取不到 UI 常量时同一个做法，报错时把库里
    有哪几个名字一起打出来。

    **同名两条也报错**：一位干员一条，两条同名会让「他在第几张卡」没有唯一答案 ——
    那同样是库本身的问题，不是读数的问题。
    """
    matched = [entry for entry in faces if entry[0] == name]
    if not matched:
        raise KeyError(
            f"库里没有「{name}」这一条：库里是 {[entry[0] for entry in faces]}"
        )
    if len(matched) > 1:
        raise ValueError(
            f"库里「{name}」有 {len(matched)} 条：一位干员一条，"
            "两条同名会让「他在第几张卡」没有唯一答案"
        )
    return matched[0]


def squad_card_scores(
    frame: Frame,
    name: str,
    faces: Sequence[tuple[str, np.ndarray]],
    *,
    pitch: float,
    assets: SquadGeometry | None = None,
) -> tuple[tuple[float, int], ...]:
    """一位干员对**每个卡位**的分数 —— `((分数, 卡下标), ...)`，按分数降序。

    **这是这条路判据本身**（`find_squad_card` 用的就是它），所以它是公开的 ——
    与 `roster_scores` 同一个理由：要让别人拿**同一条链的分数**去核，就不能让他
    另写一份匹配（另写一份，量的就是另一条链了）。

    **只拿一条库**：名字对上的那一条，去比每一张卡。所以 `matchTemplate` 的次数是
    `卡位 × 尺度档数`，**与库长无关**。

    **红罩卡那一格与 `read_squad_identities` 同一套**：`redeploying` 是 `True` 就走
    `redcap_region` + `margin.MaskedRuler`（那把尺子也**只建这一条库**），其余照旧 ——
    两块取区、同一条统计量，而且在**同一块取区上**逐条目比（§两块取区）。

    **线是 `ROSTER_SOLID_MATCH_SCORE`** —— 就是全扫那条路「最高分要过线」的那一条，
    **不是另开的一条**：两处量的是同一个统计量、同一批画面，线本来就该是同一条。
    这条路**不看余量**（余量是**库条目之间**的比较，这里只有一条库可比）：
    「像到过了线」这一件事已经把两簇分开了，「至多一张」由 `find_squad_card`
    从结构上把住。两簇的读数与论证见模块文档 §两路。

    下标与 `squad_card_centres` **同一套**（0 = 最右那张）—— 它走的就是它，
    不另开一套定位、也不另开一套次序。

    **`assets` 给了就接几何**（用户裁定 6）：这一条在某个卡位上已经有对位时，就**在那份
    对位上打分**（`_geometry_scores` / `_masked_at_geometry`），不再整段扫 —— 省时间，
    而且与全扫那条路**同一份几何**。没有对位（或者没给资产）时照旧整段扫，
    **粗层直接接受保留**：这条路**没有次高可比**，判据退化成只看分线，所以它自己的网是
    "两簇的间距"那一条单测（见模块文档 §两路）。
    """
    if not isinstance(frame, Frame):
        raise TypeError(f"要读的是一帧，收到 {type(frame).__name__}")
    _guard_faces(faces)
    if not isinstance(name, str) or not name:
        raise TypeError(f"要查的名字要是非空字符串，收到 {name!r}")
    if isinstance(pitch, bool) or not isinstance(pitch, (int, float)):
        raise TypeError(f"卡距要是数，收到 {type(pitch).__name__}")
    if pitch <= 0:
        raise ValueError(f"卡距必须为正，收到 {pitch}")

    entry = _entry_for(faces, name)
    card_top = float(squad_card_rect(0, pitch).y)
    disc_box = _margin().IDENTITY_DISC_BOX
    redcap: Callable[[np.ndarray], tuple[tuple[float, str, float], ...]] | None = None
    redcap_ruler: object | None = None
    scored: list[tuple[float, int]] = []
    centres = squad_card_centres(frame, pitch=pitch)
    for index, (centre, _) in enumerate(centres):
        red = redeploying(frame, centre, card_top=card_top) is True
        region = np.ascontiguousarray(frame.crop(
            redcap_region(centre, pitch=pitch) if red
            else identity_region(centre, pitch=pitch)
        ))

        geometry = assets.geometry_of(index, name) if assets is not None else None
        if geometry is not None and not _geometry_fits(region, entry[1], geometry.scale):
            geometry = None
        if red:
            if geometry is not None:
                if redcap_ruler is None:
                    redcap_ruler = _redcap_ruler((entry,))
                score = _masked_at_geometry(
                    region, (entry,), redcap_ruler, {name: geometry},
                    (float(disc_box[0]), float(disc_box[1])),
                )[0][0]
            else:
                if redcap is None:
                    redcap = _redcap_scorer((entry,))
                score, _, _ = redcap(region)[0]
        else:
            if geometry is not None:
                score = _geometry_scores(
                    region, (entry,), {name: geometry},
                    (float(IDENTITY_BOX_X), float(IDENTITY_BOX_Y)),
                )[0][0]
            else:
                score, _, _ = roster_scores(region, (entry,))[0]
        scored.append((score, index))
    scored.sort(key=lambda item: item[0], reverse=True)
    return tuple(scored)


def find_squad_card(
    frame: Frame,
    name: str,
    faces: Sequence[tuple[str, np.ndarray]],
    *,
    pitch: float,
    assets: SquadGeometry | None = None,
) -> int | None:
    """一位干员**在第几张卡** —— 指得出给下标，指不出给 `None`。

    下标与 `squad_card_centres` / `read_squad_identities` **同一套**（0 = 最右那张）。

    **判据**：拿那一条库去比每一张卡，**恰有一张**过 `ROSTER_SOLID_MATCH_SCORE`
    才给下标；**零张**（不像）与**两张以上**（分不开）都落 `None`。线是哪一条、
    为什么一条就够，见 `squad_card_scores` 与模块文档 §两路。

    **它只回答两件事**：在第 k 张卡 / 没能指出他在哪。**不产否定存在断言** ——
    不许回答「他不在备战席上」：他可能就在，只是**这一层没读出来**（库缺一条、
    余量不够，或者那一格根本不是一张已注册的卡面）。所以返回值
    **不是槽值**（不是 `Valued(Unknown(), None)`）：那等于替「读不出来」断言了存在。
    调用方若要更强的说法，那是**另一条链的事**，不在这一层。

    **定下这次读数的那个分数**在 `squad_card_scores` 的第一项 —— 本层不另开一份：
    它就是这条路判据本身，`find_squad_card` 是它的一个薄壳。
    """
    scored = squad_card_scores(frame, name, faces, pitch=pitch, assets=assets)
    hits = [index for score, index in scored if score >= ROSTER_SOLID_MATCH_SCORE]
    if len(hits) == 1:
        return hits[0]
    return None
