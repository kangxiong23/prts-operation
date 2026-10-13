"""红罩那条路的**「余量从哪来」** —— 一把**只用来量**的尺子，不接线上那条链。

《视觉层/备战席点名》§判据的形状 定的判据是「取一块窄框 → 在库立绘里滑着找位置 →
**同时看最高分与余量**」。红罩那条路要救，缺的是**余量**（分数到了线、余量过不去）。
这个模块只回答一件事：**余量能从哪来** —— 它把判据的**取区**换掉，别的都不动。

## 它不是 `roster_scores` 的替代品

线上那条链（`roster_scores`）**一个字节没动** —— 这个模块做的事只有一件：
**同一个算式、换一块取区**（报出来的数不同，见下一节）。**线上只有红罩卡那几格会走到它**：`roster` 的两条入口
在判到"这一格在再部署中"之后改用这里的取区与 `MaskedRuler`，干净卡照旧走
`roster_scores`。那两处**在函数里 import**（`margin` 反过来 import
`roster.IDENTITY_SCALES` —— 那把尺子默认的尺度网格，模块级 import 会成环）。

## 两个算式同源 —— 但**报出来的数不是同一个量**

**这一条是量出来的，不是照抄文档**：`roster_scores` 的分数是
`cv2.matchTemplate(..., TM_CCOEFF_NORMED)`，而 OpenCV 那个量**不是**「把三通道拉平之后
的 Pearson」（`roster.py` 的文档与 `tests/unit/test_vision_roster.py` 那条用例都这么写，
但那条用例用的是**随机图**，随机图三通道的均值几乎相同，两种口径在那里对得上）。
实测（`cd59` 位 4 那块身份区，同一个位置）：

| 口径 | 分 |
| --- | --- |
| 三通道拉平、只减一个全局均值的 Pearson | **0.356033** |
| **逐通道各减自己的均值、分母把三通道加起来** | **0.293475** |
| OpenCV `TM_CCOEFF_NORMED` | **0.293475** |

所以链上的分数是**逐通道**那个。这个模块照的是**逐通道**那一份 —— **同一个档、同一个位置上，
两个算式逐位同值**（`tests/unit/test_vision_redcap_margin.py` 钉着这一条）。

**但它不复现 `roster_scores` 报出来的数，也不再拿「复现」当自证方式** —— 两把尺子
**不是同一个量**，两处不一样：

1. **取区与掩码不同**：线上红罩那一格取的是「圆盘 − 中央秒数那一条」（`identity_disc_mask`），
   这个模块里还有别的形状（`free_mask` 那一族）；掩码决定「哪些像素算数」；
2. **就算把掩码开到全 True、取区换回老身份框，两个入口报出来的数也不等**：`roster_scores`
   报的是「定下判据的那一档上的分」—— 它**判据一满足就停**，还会**在粗层直接接受**
   （`ROSTER_COARSE_ACCEPT_SCORE`）；而 `MaskedRuler.scores` 是**每一档整张滑、取全局最大**。
   差别全在**报哪一个数**，不在算式。

实测（`tests/帧样本/余量` 那 14 帧贴回画布之后的 **148 张卡**；同为老身份框
`(52, 76, 100, 100)`、同为 18 档、同一份库）：

| 新尺子（掩码全开） − `roster_scores` | 分 |
| --- | --- |
| 最小 | **−0.000002**（浮点噪声） |
| 最大 | **+0.049858** |
| 平均 | **+0.020272** |

**`roster_scores` 走满 18 档的卡上两把逐位相同**（判据一处都不成立时，两条都取全局最大）；
差只出在它**早停**的那些卡上，方向一律是**新尺子更高** —— 高出来的正是「全局最大 − 早停
那一档」那一段。**所以「掩码全开时它必须把 `roster_scores` 逐位复现」这句话不成立。**

**新尺子的立足点是它自己那 44 对的实测**：罩着端真值最低 **0.973** · 次高最高 **0.668** ·
两条规则 **44/44** —— 两半都比 `0.85` 与「次高 ≤ 2 × 最高 − 1.09」宽裕，所以
**线不动、判据不动**。
**线上那条链与它的文档这一版都不动**：那是另一件事，记在报告的 ④ 里。

## 换掉的是取区

`MaskedRuler` 收一块**形状 + 掩码**：掩码是「这一块里哪些像素算数」。它把**与卡无关的
那半边**（库立绘那侧的量）在构造时算一次，然后一张卡一张卡地打分 —— 所以同一块取区上
量几百张卡是可行的。

**为什么要掩码**：红罩卡上那条倒计时图形（环 + 中央数字）**逐帧在变**，它不是罩子的一部分
（罩子是静态的，见 `redcap.py`），所以它只能**从取区里排掉**。排掉之后剩下的像素是
**不规则**的形状 —— 而矩形是被判据自己逼出来的：框一宽，放大就放不下，尺度搜索就死了。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from ark.engine.vision.bench import RING_CENTER_DY, RING_INNER, RING_OUTER
from ark.engine.vision.redcap import TOP_STRIP_HEIGHT
from ark.engine.vision.roster import IDENTITY_SCALES

__all__ = [
    "MaskedRuler",
    "card_distance",
    "free_mask",
    "identity_disc_mask",
    "identity_disc_box",
    "identity_disc_region",
    "CARD_HEIGHT",
    "CARD_WIDTH",
    "IDENTITY_DISC_RADIUS",
    "IDENTITY_DISC_STRIP_HALF",
    "IDENTITY_DISC_BOX",
]


CARD_HEIGHT: int = 214
CARD_WIDTH: int = 204
"""一张备战席卡的尺寸（px）。取区一律以**卡片左上角**为原点，与 `roster.identity_region` 同源。"""

MASK_RADIUS: int = RING_OUTER
"""从取区里排掉的那个圆盘用哪个半径 —— **与读倒计时那条链同一个数**（`bench.RING_OUTER`）。

不另开一份半径：环带外缘就是 60，`redcap._countdown_mask` 用的也是它。
"""


def card_distance(shape: tuple[int, int] = (CARD_HEIGHT, CARD_WIDTH)) -> np.ndarray:
    """整张卡上每个像素离**环心**多远（px）。环心在卡内 `(卡宽/2, RING_CENTER_DY)`。

    圆心按**卡片自己的几何**取（与 `redcap._countdown_mask` 同一口径），所以这一层也
    假定喂进来的是整张卡。
    """
    height, width = int(shape[0]), int(shape[1])
    rows = np.arange(height, dtype=np.float64)[:, None]
    cols = np.arange(width, dtype=np.float64)[None, :]
    return np.sqrt((cols - width / 2.0) ** 2 + (rows - RING_CENTER_DY) ** 2)


def free_mask(
    box: tuple[int, int, int, int],
    *,
    radius: int = MASK_RADIUS,
    card: tuple[int, int] = (CARD_HEIGHT, CARD_WIDTH),
) -> np.ndarray:
    """卡内一块矩形 `(x0, y0, 宽, 高)` 里，**可以用**的像素（`True` 就是可用）。

    两把刀与 `redcap._countdown_mask` 一模一样：**倒计时圆盘**（半径 `radius`）与
    **顶上那一条**（职业图标 + 费用画在罩子上面，不属于那条仿射）。顶上那一条有多高由
    `redcap.TOP_STRIP_HEIGHT` 定，这里用**同样的口径**。
    """
    x0, y0, width, height = (int(value) for value in box)
    distance = card_distance(card)[y0 : y0 + height, x0 : x0 + width]
    rows = np.arange(y0, y0 + height, dtype=np.float64)[:, None]
    return (distance > float(radius)) & (rows >= float(TOP_STRIP_HEIGHT))


IDENTITY_DISC_STRIP_HALF: int = 16
"""圆盘正中被挖掉的**那一条**的半个高度（px）—— 中央那个秒数所占的一条。

**它是量出来的，不是画的**：44 张罩着的备战席卡上（19 张互不相同），把圆盘
（半径 `bench.COUNTDOWN_MASK_RADIUS = 45`）里那串秒数按 `min(R,G,B) > 阈值` 扫一遍：

| 阈值 | 扫出来的行 |
| --- | --- |
| 200…240（笔画**核心**） | **102…130** |
| 80…150 | 101…131 |
| 40…60 | **101…132** |

阈值再往下就没有边了 —— 卡面自己的暗底（圆盘里、条之外）p99 只有 **38**。所以这串
秒数的**可见范围**是卡内行 **101…132**（环心 `RING_CENTER_DY = 116.5`，即
−15.5…+15.5 行）。取**整数**半高 **16**：`|行 − 116.5| < 16` 恰好挖掉 **101…132** ——
整整那一段，核心（102…130）每边各多留一行给抗锯齿。

**再小一档（15）挖的是 102…131**：第 101 行与第 132 行那两圈边留在取区里，实测那一档
掩码里最亮的像素从 **38 跳到 159**（半径 47 那一档上量的）。读数梯子（半高 13…17）在
`tests/帧样本/预处理/圆盘取区.json` 的 `margin` 那一栏，**它给的同一条线**。

**它不随半径变**：条的高低由秒数自己的字形定，与圆盘取到哪儿无关。
"""

IDENTITY_DISC_RADIUS: int = 47
"""这块取区取到**多大的圆盘**（px）—— 圆心在卡内 `(卡宽 / 2, RING_CENTER_DY)`。

**它不等于 `bench.RING_INNER`（50）**：50 是环带的名义内缘，而**画出来的**那圈白环
往内多压了约 1.5 px。实测（这一批 19 张互不相同的罩着卡，只看中央那一条之外，
逐环取 `min(R,G,B)` 的最大值）：

| 离环心那**一环** | 环里最亮的 `min(R,G,B)` | 是什么 |
| --- | --- | --- |
| 47 及以内 | **38** | 卡面自己的暗底 |
| 48 ≤ 距离 < 49 | **87** | 白环的**半亮内边** |
| 49 ≤ 距离 < 50 | **226** | 白环本体 |
| 50 ≤ 距离 < 58 | **255** | 白环本体 |
| 58 ≤ 距离 < 60 | 151–250 | 白环外缘 |
| 60 及以外 | 39 | 环外 |

逐张量「环形亮像素（`min > 200`）离环心最近多远」是 **49.85…50.50**。

**取到哪一档，看掩码里最亮的那个像素**（中央条半高 16 那一档上量的）：

| 半径 | 掩码里的像素数 | 掩码里最亮的 `min(R,G,B)` |
| --- | --- | --- |
| 44…**48** | 3328…4228 | **38** —— 就是卡面自己的暗底，**一点都没漏** |
| 49 | 4462 | **87** —— 环的半亮内边进来了 |
| 50 | 4720 | **226** —— 环本体 |
| 51…52 | 4962…5216 | **255** |

所以 **48 是最后一个「一点都没漏」的档，49 一进去就是 87**。取 **47**：它的 +1 邻居
（48）**仍然全干净**，而 48 的 +1 邻居（49）已经开始了 —— 47 是「决策挪一档不摆」的
**最大**那一档。

**这个判断只用画面事实（环有多亮），不是拿分数挑的**（用户裁定：不许为了比分好看挑取区）
—— 读数梯子（半径 44…52）在 `tests/帧样本/预处理/圆盘取区.json` 的 `margin` 那一栏。
"""


def identity_disc_mask(
    *,
    radius: int = IDENTITY_DISC_RADIUS,
    half: int = IDENTITY_DISC_STRIP_HALF,
    card: tuple[int, int] = (CARD_HEIGHT, CARD_WIDTH),
) -> np.ndarray:
    """整张卡上**这块取区**的掩码：**环带以内的圆盘，挖掉中央秒数那一条**。

    `True` 就是取区取的那些像素。圆心按**卡片自己的几何**取（横在卡片正中 `卡宽 / 2`、
    纵在离卡片上缘 `RING_CENTER_DY`），所以这一层与 `card_distance`、
    `redcap._countdown_mask` 同一个口径：喂进来的也得是整张备战席卡。

    **两把刀**：

    - **外圈**：离圆心**超过** `radius` 的像素不要 —— `radius` 落在环带内缘**以内**
      （见 `IDENTITY_DISC_RADIUS`），所以环带与它外面的卡框都不进这块取区；
    - **中央那一条**：离环心**不足** `half` 行的像素不要 —— 那是秒数的地盘，它
      **逐帧在变**，不归罩子管。

    **罩子只有一层**（用户裁定）：盘内盘外是同一层，所以这里不按「哪一层」分两套 ——
    罩着端与干净端用**同一块**取区。

    **它整块落在卡片以内**：`radius` 超过「圆心到上下左右任一边」就直接报错，不裁边 ——
    裁边会让「取区」看起来还在、实际少一块。
    """
    height, width = int(card[0]), int(card[1])
    if isinstance(radius, bool) or not isinstance(radius, int) or radius <= 0:
        raise ValueError(f"半径要是正整数像素，收到 {radius!r}")
    if isinstance(half, bool) or not isinstance(half, int) or half < 0:
        raise ValueError(f"半高要是非负整数像素，收到 {half!r}")
    if half >= radius:
        raise ValueError(f"半高 {half} 不小于半径 {radius}：那一条能把整个圆盘吃掉")
    reach = min(width / 2.0, RING_CENTER_DY, height - RING_CENTER_DY)
    if radius > reach:
        raise ValueError(
            f"半径 {radius} 在 {height}×{width} 的卡上装不下：圆心到最近那条边只有"
            f" {reach} px。取区要整块落在卡片以内，裁边不算"
        )
    rows = np.arange(height, dtype=np.float64)[:, None]
    columns = np.arange(width, dtype=np.float64)[None, :]
    distance = np.sqrt((columns - width / 2.0) ** 2 + (rows - RING_CENTER_DY) ** 2)
    return (distance <= float(radius)) & (np.abs(rows - RING_CENTER_DY) >= float(half))


def identity_disc_box(
    *,
    radius: int = IDENTITY_DISC_RADIUS,
    half: int = IDENTITY_DISC_STRIP_HALF,
) -> tuple[int, int, int, int]:
    """这块取区的**框**（卡内 `(x0, y0, 宽, 高)`）—— 上面那把掩码自己的紧包围盒。

    **它不是第二个形状**：形状只在 `identity_disc_mask` 一处定义，这个框回答的是
    「那块掩码落在卡内的哪一格」。取区 = 这个框裁出来的画面 + 这个掩码。

    **紧到边**（而不是钉一个像 `(52, 76, 100, 100)` 那样的常数）：半径挪一档，框跟着
    挪，所以「在半径上挪一档」这件事在读数上摆多少，是**端到端**摆多少。
    """
    mask = identity_disc_mask(radius=radius, half=half)
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError(f"半径 {radius} / 半高 {half} 这一档一个像素都不剩")
    return (
        int(columns[0]),
        int(rows[0]),
        int(columns[-1] - columns[0] + 1),
        int(rows[-1] - rows[0] + 1),
    )


IDENTITY_DISC_BOX: tuple[int, int, int, int] = identity_disc_box()
"""这块取区的框在卡内的位置 —— 由 `identity_disc_mask` 与上面的半径一处推出来。"""


def identity_disc_region(
    *,
    radius: int = IDENTITY_DISC_RADIUS,
    half: int = IDENTITY_DISC_STRIP_HALF,
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """→ `(框, 框里的掩码)` —— **取区就是这一对**，调用方不要再自己裁一次。

    用法：`box, mask = identity_disc_region()`；`x0, y0, w, h = box`；
    卡上那块画面就是 `card[y0 : y0 + h, x0 : x0 + w]`（形状与掩码一致，可以直接喂
    `MaskedRuler`）。
    """
    box = identity_disc_box(radius=radius, half=half)
    x0, y0, width, height = box
    mask = identity_disc_mask(radius=radius, half=half)[y0 : y0 + height, x0 : x0 + width]
    return box, np.ascontiguousarray(mask)


def _guard(region: object, what: str) -> np.ndarray:
    if not isinstance(region, np.ndarray):
        raise TypeError(f"{what} 要是一块 (高, 宽, 3) 的 uint8 RGB，收到 {type(region).__name__}")
    if region.dtype != np.uint8 or region.ndim != 3 or region.shape[2] != 3:
        raise TypeError(
            f"{what} 要是 (高, 宽, 3) 的 uint8 RGB，收到 {region.dtype} 的 "
            f"{getattr(region, 'shape', region)}"
        )
    return region


class MaskedRuler:
    """一块 `(取区形状, 掩码)` 上的分数尺子：**逐通道减均值**的掩码相关，滑着找位置。

    **库里立绘那一侧**的量（每个库条目、每个尺度、每个通道的 `sum(t)` 与 `sum(t²)` 在
    窗口上的滑动值）**只取决于掩码与库，不取决于卡**，所以一张一张卡打分时只剩「查询那三个
    通道的相关」要算。

    **它们是懒建的**（用户裁定 2）：构造**一层都不建**（实测 0.011 s），某个
    (层, 条目, 模板尺寸) 第一次被走到时才现算，算过留在 `_built` 里。所以这把尺子的
    建造成本**砸在"第一次出现红罩卡那一帧"**上，而不是构造那一瞬间。要看它建了多少：
    `built_cells`。

    **同一 (层, 条目, 模板尺寸) 只建一次**（用户裁定 1）：模板尺寸一样 ⇒ `resize(mask)`
    出来的 `layer_f` 逐位相同、`a` 是同一份、那六个 `TM_CCORR` 的结果**逐位相同**。
    掩码这一侧尺寸分组很粗 —— 25 像素那一层 18 档只落在 **9** 个尺寸上、50 像素那一层
    **12** 个、全分辨率 18 个 —— 所以粗层那一半里有一半是白算的。`rows` 里还有第二道
    去重（同 (模板尺寸, 落点) 的档共用一次算式），两道合起来**读数一个字节不变**
    （见 `tests/unit/test_vision_roster_geometry.py` 与 `data/probe/dedup_equiv.py`）。

    `scores(region)` 的返回形状与 `roster_scores` **一模一样**：
    `((分数, 名字, 尺度), ...)` 按分数降序。掩码全开、尺度网格取 `IDENTITY_SCALES` 时，
    它必须把 `roster_scores` 复现到浮点误差内。

    **整张滑（`scores` / `scores_and_positions`）报的是每一档上的最大值**，与
    `roster_scores` 同一条；而 `rows` 只是把每一档都交出来，报不报分由调用方定。
    **分层那条路（`roster._masked_layered_scores`）报出来的是第一个满足判据的那一档的分，
    不是全局最大分**（用户裁定 9）—— 那个数进了槽值的 `confidence`，所以下游不许当
    "全局极值"用。
    """

    def __init__(
        self,
        mask: np.ndarray,
        faces: Sequence[tuple[str, np.ndarray]],
        *,
        scales: tuple[float, ...] = IDENTITY_SCALES,
    ) -> None:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2:
            raise TypeError(f"掩码要是 (高, 宽) 的布尔数组，收到 {getattr(mask, 'shape', mask)}")
        if int(mask.sum()) < 16:
            raise ValueError(f"掩码只有 {int(mask.sum())} 个像素：太少，量不出余量")
        if not faces:
            raise ValueError("库是空的：空库不是「库里没有这个人」，是没建库")
        self.mask_shape = (int(mask.shape[0]), int(mask.shape[1]))
        self.mask_pixels = int(mask.sum())
        self.faces = tuple(faces)
        self.scales = tuple(scales)
        self._mask_u8 = np.ascontiguousarray(mask.astype(np.uint8) * 255)
        for name, art in self.faces:
            if art.ndim != 3 or art.shape[2] != 3 or art.dtype != np.uint8:
                raise TypeError(f"库里 {name!r} 的立绘要是 (高, 宽, 3) 的 uint8 RGB")
        # **懒建**（用户裁定 2）：构造时**一层都不建**。库里那一侧的量只在某个
        # (条目, 档) 真被走到时才现算，算过就留在 `_cells` 里。
        self._built: dict[tuple[float, str, int, int], tuple | None] = {}
        self._art: dict[tuple[float, str], np.ndarray] = {}
        self._levels: dict[float, list[tuple[str, np.ndarray, list[tuple]]]] = {}
        if not self.usable_scales:
            raise ValueError("一个尺度都放不下：这块取区与库里的立绘配不上")

    @property
    def built_cells(self) -> dict:
        """懒建到目前为止**建了什么** —— 「第一次出现红罩卡那一帧凸起多少」靠它解释。

        `queried` 是问过的 (层, 条目, 模板尺寸) 个数，`built` 是其中真建出来了的
        （装不下的那些也进 `queried`、不进 `built`）。用户裁定 2：不报这个数，
        那个凸起就只能说「不知道哪儿来的」。
        """
        return {
            "queried": len(self._built),
            "built": sum(1 for value in self._built.values() if value is not None),
            "art": len(self._art),
            "full_layers": len(self._levels),
        }

    @property
    def _prepared(self) -> list[tuple[str, np.ndarray, list[tuple]]]:
        """**全分辨率那一整套层级** —— 从前构造时就算好的那一份，现在按需才建。

        它是 `_levels_at(1.0)` 的别名（懒建的入口），所以整张滑那条路（`scores` /
        `scores_and_positions` / `layers_for`）**行为一个字节没变**：它们本来就要每一档。
        """
        return self._levels_at(1.0)

    @property
    def usable_scales(self) -> tuple[float, ...]:
        """这把尺子在**全分辨率**上用得上哪几档 —— **不建层级**就能回答。

        判据与 `_level_at` 里那两处 `continue` **同源**（尺寸装不下 / 掩码像素不足 16），
        所以它报的就是全分辨率那一套里真正会出现的档。**这是一次形状检查，不是一次打分**：
        它不碰 `matchTemplate`，所以构造完就能问，不会把懒建变成「一建就建满」。
        """
        out: set[float] = set()
        for name, art in self.faces:
            for scale in self.scales:
                height = int(round(self.mask_shape[0] * scale))
                width = int(round(self.mask_shape[1] * scale))
                if height > art.shape[0] or width > art.shape[1]:
                    continue
                small = cv2.resize(
                    self._mask_u8, (width, height), interpolation=cv2.INTER_NEAREST
                )
                if int((small > 127).sum()) < 16:
                    continue
                out.add(float(scale))
        return tuple(sorted(out))

    def _art_at(self, factor: float, name: str, art: np.ndarray) -> np.ndarray:
        """库里这一条缩到 `factor` 倍、转成 float32 的那一份（按 (factor, 条目) 缓存）。

        **它是这一层最贵的一步**（整块立绘的 `astype` / 除法），所以它必须与 (条目, 档)
        分开缓存：18 个档共用同一份，不能每档重算一遍。
        """
        key = (float(factor), name)
        cached = self._art.get(key)
        if cached is not None:
            return cached
        if factor == 1.0:
            small_art = art
        else:
            small_art = cv2.resize(
                art,
                (max(1, int(round(art.shape[1] * factor))),
                 max(1, int(round(art.shape[0] * factor)))),
                interpolation=cv2.INTER_AREA,
            )
        cached = np.ascontiguousarray(small_art.astype(np.float32) / 255.0)
        self._art[key] = cached
        return cached

    def _level_at(
        self, factor: float, name: str, art: np.ndarray, scale: float
    ) -> tuple | None:
        """**一个 (条目, 档)** 在这一层上的那一套量 —— `(档, 高, 宽, 立绘, 掩码, 像素数, 通道)`。

        装不下（尺寸 / 掩码像素不足）给 `None`，与 `_levels_at` 里那两处 `continue` 同源。
        """
        a = self._art_at(factor, name, art)
        height = int(round(self.mask_shape[0] * scale * factor))
        width = int(round(self.mask_shape[1] * scale * factor))
        built = self._built_at(factor, name, a, height, width)
        if built is None:
            return None
        return (scale,) + built

    def _built_at(
        self, factor: float, name: str, a: np.ndarray, height: int, width: int
    ) -> tuple | None:
        """**同一 (层, 条目, 模板尺寸) 只建一次**（用户裁定 1，与干净那条路同一条）。

        模板尺寸一样 ⇒ `resize(mask)` 出来的 `layer_f` 逐位相同、`a` 是同一份、
        那六个 `TM_CCORR` 的结果**逐位相同** —— 再建一遍是白算。掩码这一侧的尺寸分组
        比干净那条路粗：25 像素那一层 18 个档只落在 **9** 个尺寸上（50 像素那一层 12 个、
        全分辨率 18 个），所以粗层那一半里有一半是白算的。

        返回的那一串里**不含档** —— 档由调用方贴上去（`_level_at`），所以同一份量能被
        几个档共用，而每个档仍然各自出现在读数里。
        """
        key = (float(factor), name, int(height), int(width))
        if key in self._built:
            return self._built[key]
        built: tuple | None = None
        if height <= a.shape[0] and width <= a.shape[1]:
            small = cv2.resize(
                self._mask_u8, (width, height), interpolation=cv2.INTER_NEAREST
            )
            layer = small > 127
            pixels = float(layer.sum())
            if pixels >= 16:
                layer_f = np.ascontiguousarray(layer.astype(np.float32))
                channels: list[tuple[np.ndarray, np.ndarray]] = []
                for index in range(3):
                    channel = np.ascontiguousarray(a[:, :, index])
                    channels.append(
                        (
                            cv2.matchTemplate(channel, layer_f, cv2.TM_CCORR),
                            cv2.matchTemplate(channel * channel, layer_f, cv2.TM_CCORR),
                        )
                    )
                built = (height, width, a, layer_f, pixels, channels)
        self._built[key] = built
        return built

    def _levels_at(self, factor: float) -> list[tuple[str, np.ndarray, list[tuple]]]:
        """**立绘那一侧缩到 `factor` 倍**之后的那一套层级 —— 按 `factor` 缓存。

        `factor=1.0` 就是**全分辨率**那一套（从前那套，一个字节没动）。粗层拿它把
        "找档"这一步搬到小图上 —— 与干净那条路 `_ANCHOR_PIXELS` 做的是**同一类**事
        （用户裁定 1：掩码那条路也吃到锚点带来的搜索收窄）。

        掩码那一侧跟着一起缩：`layer = resize(mask, (宽 × 档 × factor)) > 127`，
        所以 `factor=1.0` 时它逐位等于原来那个 `small`。

        **它是「整套都建」那条路**（整张滑、粗层的整层找档都要每一档）；懒建那条路走
        `_level_at`，两边共用同一份 `_cells` —— 所以先懒建再整套建，**读数一个字节不变**。
        """
        cached = self._levels.get(float(factor))
        if cached is not None:
            return cached
        out: list[tuple[str, np.ndarray, list[tuple]]] = []
        for name, art in self.faces:
            levels: list[tuple] = []
            for scale in self.scales:
                level = self._level_at(factor, name, art, scale)
                if level is not None:
                    levels.append(level)
            out.append((name, self._art_at(factor, name, art), levels))
        self._levels[float(factor)] = out
        return out

    def layers_for(self, name: str) -> int:
        """这个库条目在这块取区上**用了几档尺度** —— 放不下的那几档被跳过了。"""
        for entry_name, _art, levels in self._prepared:
            if entry_name == name:
                return len(levels)
        raise KeyError(name)

    def scores(self, region: np.ndarray) -> tuple[tuple[float, str, float], ...]:
        """一块取区（`uint8` RGB，形状必须与掩码一致）→ `((分数, 名字, 尺度), ...)` 降序。"""
        return tuple(
            (score, name, scale)
            for score, name, scale, _position in self.scores_and_positions(region)
        )

    def _levels_for(
        self,
        factor: float,
        windows: Mapping[str, Mapping[float, tuple[float, float]]] | None,
        only: float | None,
    ) -> list[tuple[str, np.ndarray, list[tuple]]]:
        """**这一次调用真会走到的那些 (条目, 档)** —— 给了 `windows` 就只建这些。

        `windows is None` 是「每一档都要」（整张滑、粗层的整层找档）—— 走整套那条路
        （`_prepared` / `_levels_at`），行为与从前一个字节不差。
        给了 `windows` 就是「只算列出来的那些 (名字, 档)」（复用那条路、全分辨率那一层
        的按档试）—— 这时**只建那些**：一把尺子不再一建就建满 11 条 × 18 档（用户裁定 2）。
        没被点到的条目照旧在结果里留一个空字典（调用方按名字取，缺键会炸）。
        """
        if windows is None:
            prepared = self._prepared if factor == 1.0 else self._levels_at(factor)
            if only is None:
                return prepared
            return [
                (name, art, [lv for lv in levels if abs(lv[0] - only) <= 1e-9])
                for name, art, levels in prepared
            ]
        out: list[tuple[str, np.ndarray, list[tuple]]] = []
        for name, art in self.faces:
            by_scale = windows.get(name) or {}
            levels: list[tuple] = []
            for scale in self.scales:
                if only is not None and abs(scale - only) > 1e-9:
                    continue
                if scale not in by_scale:
                    continue
                level = self._level_at(factor, name, art, scale)
                if level is not None:
                    levels.append(level)
            out.append((name, art, levels))
        return out

    def rows(
        self,
        region: np.ndarray,
        *,
        only: float | None = None,
        windows: Mapping[str, Mapping[float, tuple[float, float]]] | None = None,
        half: int = 0,
        pixels: int | None = None,
    ) -> dict[str, dict[float, tuple[float, float, float]]]:
        """**每个条目、每一档**的读数：`{名字: {档: (分, 落点 x, 落点 y)}}`。

        分与 `scores` 是**同一个统计量、同一个算式**（逐通道减均值那条），只是这里不
        只挑最好那一档，而是把每一档都交出来 —— 干净那条路的 `_layer_rows` 交的是
        同一个形状，两边的"层"因此可以一层一层对着做。

        **两个筛选参数还是那两件事，公式一个字节没改**：

        - `only` 只留这一档；
        - `windows` 只在这些小块里取最大，`{名字: {档: (x, y)}}`，半宽是 `half`
          （**全分辨率像素**；粗层按 `factor` 折下去）。

        **给了 `windows` 就是"只算列出来的那些 (名字, 档)"**：没列出来的档**不算**（不退回
        整张）—— 复用那条路靠这一条把"每条库只在它自己那一档上比"落到实处。

        `pixels` 是**粗层**：把立绘那一侧缩到这么高再滑（`None` = 全分辨率）。
        这就是掩码那条路吃到的"锚点带来的收窄"（用户裁定 1）—— 找档那一步搬到小图上，
        与干净那条路 `_ANCHOR_PIXELS` 做的是同一类事。**粗层不裁位置窗**（裁定 7）：
        位置窗只由调用方给，锚点那一层传 `windows=None`。

        落点一律是**全分辨率**坐标（粗层算出来的位置除回 `factor`）。

        **库里那一侧的量是懒建的**（用户裁定 2）：这一次调用只会走到
        `_levels_for` 挑出来的那几个 (条目, 档)，**只有它们**会被现算（`_level_at`），
        算过就留在尺子上。所以一把尺子不再「一建就建满 11 条 × 18 档」—— 它在
        **第一次出现红罩卡那一帧**上按需要多少建多少。读数一个字节不变：建与不建
        走的是同一段算式，只是早晚。
        """
        _guard(region, "要打分的取区")
        if (int(region.shape[0]), int(region.shape[1])) != self.mask_shape:
            raise ValueError(
                f"取区形状与掩码不一致：取区 {region.shape[:2]} 对掩码 {self.mask_shape}"
            )
        factor = 1.0 if pixels is None else float(pixels) / float(self.mask_shape[0])
        source = np.ascontiguousarray(region)
        out: dict[str, dict[float, tuple[float, float, float]]] = {}
        for name, art, levels in self._levels_for(factor, windows, only):
            here: dict[float, tuple[float, float, float]] = {}
            # **同一 (模板尺寸, 落点) 的档只算一次**（用户裁定 1）：档只改 `here` 的键，
            # 不改下面那一串算式里的任何一个数 —— 尺寸一样、窗一样、层级一样，读数就逐位一样。
            # 掩码这一侧尺寸分组很粗（25 像素那一层 18 档 → 9 个尺寸），所以这一条省得不少；
            # `None` 也进缓存（"这一档算不出分"同样只该判一次）。
            cells: dict[tuple[int, int, int, int, int], tuple[float, float, float] | None] = {}
            for scale, height, width, art_rgb, layer_f, pixels_in_layer, channels in levels:
                window = None
                if windows is not None:
                    by_scale = windows.get(name)
                    if by_scale is None or scale not in by_scale:
                        # **给了 `windows` 就是"只算列出来的那些 (名字, 档)"**：不在里面的档
                        # 直接不算，**不退回整张**。退回整张会让"每条在它自己的档上比"这件事
                        # 静默变成"每条把 18 档都滑一遍" —— 那正是复用那条路贵且分数虚高的原因。
                        continue
                    window = by_scale[scale]
                half_px = int(round(float(half) * factor))
                rows_n = art_rgb.shape[0] - height + 1
                columns_n = art_rgb.shape[1] - width + 1
                if window is not None and rows_n > 0 and columns_n > 0:
                    window_x, window_y = window
                    y0 = max(0, min(int(round(float(window_y) * factor)) - half_px, rows_n - 1))
                    x0 = max(0, min(int(round(float(window_x) * factor)) - half_px, columns_n - 1))
                    y1 = min(rows_n, y0 + 2 * half_px + 1)
                    x1 = min(columns_n, x0 + 2 * half_px + 1)
                else:
                    x0, y0 = 0, 0
                    y1, x1 = rows_n, columns_n
                key = (height, width, x0, y0, half_px)
                if key in cells:
                    hit = cells[key]
                    if hit is not None:
                        here[scale] = hit
                    continue
                hit = None
                # **窗是先把立绘那一侧裁掉再算**，不是算完再裁分数面（分数面裁了也白裁：
                # `matchTemplate` 那一趟已经按整张立绘付过钱了）。裁出来的那一块与整张上的
                # 同一块**逐位同源**：相关面对平移不变，所以窗口里每个位置的值一个字节没变。
                if x0 == 0 and y0 == 0 and x1 == rows_n and y1 == columns_n:
                    art_block = art_rgb
                else:
                    art_block = np.ascontiguousarray(
                        art_rgb[y0 : y1 + height - 1, x0 : x1 + width - 1]
                    )
                query = cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)
                query = np.ascontiguousarray(query.astype(np.float32) / 255.0)
                masked = np.ascontiguousarray(query * layer_f[:, :, None])
                # **一次三通道 `TM_CCORR`** 就够：它把三个通道加在一起，而那正是分子要的
                # `sum_c sum_p art_c * q_c`。每次调用都按 (库条目 x 尺度) 走一遍的话，
                # 186 宽的掩码会掉出 OpenCV 的 FFT 通路、慢一个数量级。
                cross = cv2.matchTemplate(art_block, masked, cv2.TM_CCORR)
                numerator = np.array(cross, dtype=np.float64)
                template_energy = np.zeros_like(numerator)
                query_energy = 0.0
                for index in range(3):
                    channel = np.ascontiguousarray(masked[:, :, index])
                    mean = float(channel.sum()) / pixels_in_layer
                    image_sum, image_square = channels[index]
                    piece_sum = image_sum[y0:y1, x0:x1]
                    piece_square = image_square[y0:y1, x0:x1]
                    numerator -= mean * piece_sum
                    template_energy += piece_square - piece_sum * piece_sum / pixels_in_layer
                    query_energy += float((channel * channel).sum()) - pixels_in_layer * mean * mean
                if query_energy > 0.0:
                    denominator = (np.sqrt(query_energy)
                                   * np.sqrt(np.maximum(template_energy, 0.0)))
                    normalised = numerator / (denominator + 1e-12)
                    flat = int(np.argmax(normalised))
                    hit = (
                        float(normalised.ravel()[flat]),
                        (x0 + flat % normalised.shape[1]) / factor,
                        (y0 + flat // normalised.shape[1]) / factor,
                    )
                    here[scale] = hit
                cells[key] = hit
            out[name] = here
        return out

    def scores_and_positions(
        self,
        region: np.ndarray,
        *,
        only: float | None = None,
        window: tuple[tuple[float, float], int] | None = None,
    ) -> tuple[tuple[float, str, float, tuple[int, int]], ...]:
        """`scores` 那一份读数，外加**定下它的那个位置**（模板左上角在立绘上的落点）。

        **两个筛选参数是给"几何复用"那条路用的，公式一个字节没改**：

        - `only` 只留这一档（"档"这件事由资产给，不再滑一遍）；
        - `window` 只在这一小块里取最大，`((x, y), half)`（"位置"这件事也由资产给）。

        所以 `scores_and_positions(...)` 与 `scores_and_positions(..., only=档,
        window=(位置, 0))` 在同一个位置上的值是**逐位相同**的 —— 这只是不滑了，
        不是另算一条。整张滑的时候位置是全图 argmax，一处不漏。

        **它是 `rows` 的一个薄壳**（一处实现）：`rows` 交"每个条目每一档"，
        这里按库顺序把每个条目最好的那一档挑出来，再按分数降序排。
        """
        if window is None:
            windows = None
            half = 0
        else:
            (window_x, window_y), half = window
            windows = {
                name: {scale: (float(window_x), float(window_y)) for scale in self.scales}
                for name, _art, _levels in self._prepared
            }
        rows = self.rows(region, only=only, windows=windows, half=half)
        out: list[tuple[float, str, float, tuple[int, int]]] = []
        for name, _art, _levels in self._prepared:
            best = -2.0
            best_scale = self.scales[0]
            best_position = (0, 0)
            for scale in self.scales:
                row = rows[name].get(scale)
                if row is None:
                    continue
                if row[0] > best:
                    best = row[0]
                    best_scale = scale
                    best_position = (int(round(row[1])), int(round(row[2])))
            out.append((best, name, best_scale, best_position))
        out.sort(key=lambda item: item[0], reverse=True)
        return tuple(out)
