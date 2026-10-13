"""备战席那一栏的**第三态：缺费（灰卡）** —— 判据与像素还原。

一帧上，某一张卡会不会被蒙上一层灰，由**这张卡自己的部署费用**与**当前费用**决定
（`卡费 > 当前费` ⇒ 缺费）。这一层只回答两件事，**各住一处**（与 `bench.redeploying`
同待遇）：

| 问题 | 判据 | 住哪 |
| --- | --- | --- |
| 这一格**缺费**吗 | **代数** `卡费 > 当前费`；两个读数有一个不是精确时退到**自比**兜底 | `insufficient_cost` |
| 缺费那格的**像素怎么还原** | 乘回去（`× 2`，逐像素、**不要库**） | `restore` |

调用方怎么把两样凑齐（都在 `ark.engine.vision` 里，导入规则不受影响）：

    cards = bench.read_squad_row(frame, glyphs, pitch=pitch)      # 每格卡费
    current = cost.read_cost_digits(frame, glyphs)                # 当前费
    verdicts = [gray.insufficient_cost(card.cost, current) for card in cards]
    identities = roster.read_squad_identities(
        frame, faces, pitch=pitch, insufficient=verdicts
    )

## 灰罩是什么：**整片乘 0.5**（差图定的）

`tests/帧样本/缺费/`（run-04 同一局的两次暂停，卡位 / 干员 / 顺序 / 剩余可放置角色全一样，
唯一变的是当前费用 **20 → 28**）上：**1–5 号卡两帧逐位相同**（零假设，噪声地板 **0**），
**6–10 号**（费用 21/21/22/23/25）五张同时从缺费变够费。五对的身份区（100×100）逐像素
`G/C`（G = 灰罩帧、C = 对照帧）、取每张卡的中位：

    序号 6 idx4 费21  0.5024 ｜ 7 idx3 费21  0.5021 ｜ 8 idx2 费22  0.5021
    序号 9 idx1 费23  0.5022 ｜ 10 idx0 费25  0.5025

逐通道最小二乘 `G = a·C + b` 给同一件事：`a = (0.4952, 0.4957, 0.4974)`、
`b ≈ (2.1, 1.7, 1.1)`，rms **2.36 / 255**。三个模型排到只剩一个：

| 模型 | 判 | 凭据 |
| --- | --- | --- |
| ① 色度**被乘小** | **是** | 中位比 0.50；而灰罩卡身份区里 `chr>20` 的像素还剩 **23…2786** 个 |
| ② 色度被整片**抹平** | **否** | 抹平的话灰罩卡的色度该塌到 0，实测它只是被乘小 |
| ③ **连亮度一起变** | **是，而且就是同一个标量** | 亮度比 0.507×5，与色度同一条 —— 所以**一个算式**就能逆，不需要两趟流程 |

**它是「乘小了」，不是「抹平了」**：还原就是**逐像素乘回去**，**不需要库**（②那条
「先认出是谁、再拿库里的颜色回填」的两趟流程**没有它的位置**）。同一个标量在**另一局**
上复现：`shot05` 与 `shot07` 是同局的两次暂停（当前费用 11 → 8），费用 11 与 10 那两张
是同一个干员、一净一灰，身份区中位比 **0.5026 / 0.5024**。

## 还原到什么程度（实测）

五对的身份区，**还原前**与对照帧逐像素平均差 **89.589**（最大 127、>8 的像素 29993），
**乘回去之后 2.042**（最大 52、>8 的像素 2210）。

**剩下的 2 是罩子里那个加法项**：`G = 0.5·C + K` 里的 K 在身份区里不是常数
（上多半块 0.3–0.4、最下 16 行到 5–9），但它**与是哪张卡无关**（五张卡的 K 均值
1.001 / 1.044 / 1.076 / 1.052 / 1.045），而且**只有这一对的差图能看见它** ——
一个帧里没有第二份「没蒙罩的同一张卡」去解它。所以这一层给的是**全局标量**那一步，
不是「没还原干净」。

**判据不吃这个还原**：`roster_scores` 走 `TM_CCOEFF_NORMED`，那一条对**整片乘一个
标量**天生透明（实测同一张卡跨帧的相关：灰罩 **0.9967** / 乘回去 **0.9968** /
逐位相同 **1.0000**）。所以还原**不改身份那条链的判定与名字**；它改的是**像素本身**
（平均差 89.6 → 2.0）—— 让缺费卡在**任何读绝对像素的人**眼里（取墨阈值、环增益、
将来别的读数）与干净卡一样。**「同一条线」因此是字面意义上的**：还原之后那一格走的就是
`read_squad_identities` 原来那条路（`ROSTER_SOLID_MATCH_SCORE` + 余量公式），
一个数都没动。

## 自比：兜底那一条怎么算（**按卡自比**，不吃配色）

**兜底只在自己比自己时成立**：同一个卡位、同一个游戏局里的前后两帧 —— 只有那时
`卡费` 或 `当前费` 读不出来才用它。统计量是那一格**身份区墨迹（`min(R,G,B)`）的中位**
之比（`own_share`）：同一张卡的配色在比里被约掉。

**拿绝对值当线在这一批上就是错的**（用户裁定 2 的现成反例，实测）。**在这一帧上它们
"分得开"**：亮度缺费 79.63…105.35 / 够费 148.77…204.92（空档 43.42）、色度
3.01…13.21 / 16.66…33.06（空档 3.45）—— 但那是**这一帧的运气**（五张缺费卡恰好都是
亮卡面）。把语料摊开（**68 张**卡：备战席 6 帧 + 这一对，卡费与当前费都读得出的那些）：

| 统计量 | 缺费 n=53 | 干净/够费 n=15 | 穿不穿 |
| --- | --- | --- | --- |
| 身份区亮度 | 47.75…**105.35** | **89.88**…208.42 | **穿** |
| 身份区色度 | 3.01…66.73 | 4.98…132.10 | **穿** |
| 身份区饱和度 | 8.42…157.74 | 7.18…158.52 | **穿** |

（干净那一簇最低的 89.88 是 `shot26` idx0 —— 那是**红罩**那张；**纯干净**最低是
`shot26` idx5 的 **92.14**，仍然低于缺费最高的 105.35。）

实测两簇（`shot05` ↔ `shot07`，同局两次暂停）：

| 那一格 | 自比 |
| --- | --- |
| 状态没变（费 18/17/14/14/13 那五格，两帧逐位相同） | **1.0000 ×5** |
| 同一个干员一净一灰（费 11 / 费 10） | **0.5055 / 0.5164** |

空档 **0.52…1.00**，两条线取 `GRAY_SHARE_MAX = 0.70` 与 `CLEAR_SHARE_MIN = 0.85`，
中间是**判不了**。**这是纸上的防线**：这一批上一次都没落在里面（与
`COST_AMBIGUITY_MARGIN` 同一个处境），如实报上来。

**它只在「同一个卡位、同一个局」里成立**：卡位在一局之内冻结（《备战席》§卡片怎么排），
所以「上一次那一格」就是「同一个干员」。跨局、跨卡位都不可比 —— 那时这一支给
**判不了**，不是给一个错答案。

## 它不做什么

- **不给缺费卡另开一条线**（用户裁定）：还原之后走的是原来那一条，判据一个数没动；
- **不反解红罩**：那是 `bench.redeploying` 那一格的事，两块取区各判各的；
- **不碰整帧**：费用数字那条链读的是**原帧**，而它的取墨线
  （`bench.COST_INK_THRESHOLD = 64`）是**绝对**的 —— 实测这一批上灰罩反而**帮**了它：
  灰罩帧 idx0–4（费 25/23/22/21/21）都读成精确（0.95–0.98），而对照帧那五格的第一个
  字形块只打 0.14–0.30、落成未知（卡面漏进取区、与数字并成一块）。所以还原**只加在
  身份那一块取区上**，不替别的链改画面。
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ark.contracts import Exact, SlotValue, Valued
from ark.engine.vision.glyphs import INK_MIN, ink_of

__all__ = [
    "insufficient_cost",
    "own_share",
    "restore",
    "GRAY_SHARE",
    "GRAY_LIFT",
    "GRAY_SHARE_MAX",
    "CLEAR_SHARE_MIN",
]


GRAY_SHARE: Final[float] = 0.5
"""灰罩把卡面**乘掉**的那个标量（缺费那一档）。

实测逐像素 `G/C` 的中位：run-04 五对 **0.5021–0.5025** · 另一局的孪生对
**0.5024 / 0.5026** · 逐通道最小二乘的斜率 **0.4952 / 0.4957 / 0.4974**。
四个数与 **0.50** 差 **0.4%** 以内，所以取 **0.50** —— 那是这个罩子在游戏里的整数，
比「量出来的 0.5021」更可能是它本来的样子。
"""

GRAY_LIFT: Final[float] = 2.0
"""还原 = 乘回去：`1 / GRAY_SHARE`。**不是** `1 / 0.5021`：那 0.4% 不是模型、
是估值的抖动，把它写死只会让「还原」看起来比实际精确。
"""

GRAY_SHARE_MAX: Final[float] = 0.70
"""自比**低到**这个程度才算「这一格被乘过」（缺费）。

两簇实测 0.5055–0.5164（缺费）与 1.0000（没变），空档 **0.52…1.00**。
0.70 与 0.85 这两条线都坐在空档里，但**靠哪一边不是随手取的**：兜底那条路
**只在代数判不了时才被问到**，而「判不了」的代价是**这一格按干净卡处理**
（走原来那一条、可能读成未知）—— 保守的那一侧是**多认成缺费**（去还原），
所以 0.70 比中点（0.76）**更靠近缺费那一簇**。
"""

CLEAR_SHARE_MIN: Final[float] = 0.85
"""自比**高到**这个程度才算「这一格没被乘过」。见 `GRAY_SHARE_MAX`。"""


def own_share(now: np.ndarray, before: np.ndarray) -> float:
    """**同一个卡位**前后两帧那一块取区之比 —— 兜底那条路的统计量。

    两个数：两块取区各自**墨迹（`min(R,G,B)`）的中位**，然后**后除前**：
    `1.0` = 两次一样亮（状态没变）· `≈ GRAY_SHARE` = 这一格被乘过。

    口径是这一层通行的**墨迹**（`glyphs.ink_of` 的默认那一支）。用**中位**不用均值：
    立绘里总有一小片很亮的头发或很暗的阴影，中位数才代表「这一块整体多亮」。

    **配色在比里被约掉**：比的是**同一张卡**，所以「这个干员偏浅色」这件事在两边同时
    出现，除掉了。**跨卡位、跨局不可比**（那不是同一张卡）—— 调用方要保证这一点。
    """
    _guard_region(now, "自比里的这一帧")
    _guard_region(before, "自比里的上一帧")
    if now.shape != before.shape:
        raise ValueError(
            f"自比的两块取区要同形，收到 {now.shape} 与 {before.shape}："
            "两块不是同一个卡位的同一块取区，就不是「这张卡自己比自己」"
        )
    first = float(np.median(ink_of(before, take=INK_MIN)))
    second = float(np.median(ink_of(now, take=INK_MIN)))
    if first <= 0.0:
        raise ValueError(
            f"上一帧那块取区的墨迹中位是 {first}：全是黑的，比不出「被乘过多少」"
        )
    return second / first


def restore(pixels: np.ndarray) -> np.ndarray:
    """把**缺费那一格取区**的像素乘回去（逐像素、**不要库**）。

    灰罩是**整片乘 `GRAY_SHARE`**（见模块文档 §灰罩是什么），所以逆就是乘
    `GRAY_LIFT`、然后夹到 0…255：`(128,128,128)` 那一档的字形笔画乘回去正好回到
    255，而罩子那个加法项（K）**不在这一层能解的范围里**（一个帧里没有第二份
    「没蒙罩的同一张卡」）。

    **它只吃一块取区、不碰整帧**（用户裁定：还原的落点是那一格的像素，不是画面）：
    费用数字那条链读的是原帧，而它的取墨线是绝对的 —— 实测这一批上灰罩反而帮了它
    （见模块文档 §它不做什么）。所以调用方把**取区**交给它，别把帧交给它。
    """
    _guard_region(pixels, "要还原的取区")
    lifted = pixels.astype(np.uint16) * int(round(GRAY_LIFT))
    return np.clip(lifted, 0, 255).astype(np.uint8)


def insufficient_cost(
    card_cost: SlotValue,
    current_cost: SlotValue,
    *,
    share: float | None = None,
) -> bool | None:
    """这一格**缺费**吗 —— `True` / `False` / `None`（判不了）。

    **口径是代数**（用户裁定 1）：`卡费 > 当前费`，两个数都是**精确**时直接给答案。
    它比像素可靠得多 —— 两张数都是从画面里读出来的**事实**，而像素那一边要拿
    「这张卡自己上一次多亮」当参照。

    **像素只做验证与兜底**：

    - 两个读数**有一个不是精确**（未知 / 未采样）时，`card_cost` 与 `current_cost`
      给不出裁决 —— 那时才看 `share`（`own_share` 给的那个自比）：
      `<= GRAY_SHARE_MAX` ⇒ 缺费 · `>= CLEAR_SHARE_MIN` ⇒ 不缺 · 中间 ⇒ **判不了**；
    - 两个读数都有时**不理会 `share`**：它改成「验证」的角色 —— 调用方要核就自己拿
      `own_share` 去对这条裁决（`tests/unit/test_vision_gray.py` 就是这么核的）。
      **验证不改裁决**：像素那一边没有资格推翻一个读出来的数。

    **判不了就是判不了**（`None`）：调用方按「没缺费」处理（不还原、走原来那条路），
    与 `redeploying` 判不了时按「不在再部署中」处理是同一个方向 —— **保守的那一侧是
    不动画面**。
    """
    card = _exact_int(card_cost)
    current = _exact_int(current_cost)
    if card is not None and current is not None:
        return card > current
    if share is None:
        return None
    if share <= GRAY_SHARE_MAX:
        return True
    if share >= CLEAR_SHARE_MIN:
        return False
    return None


def _exact_int(value: object) -> int | None:
    """槽值里的**精确整数**；不是「精确的整数」（未知 / 未采样 / 别的数）就是 `None`。

    **只放行精确**：区间、未知、未采样都不该被这一层当成一个数去比 ——
    「卡费 15 到 20」与「当前费 17」比出来的是**一个猜测**，而这里的裁决会被用来
    改画面（还原），猜测改画面就是假事实。
    """
    if not isinstance(value, Valued):
        return None
    if not isinstance(value.payload, Exact):
        return None
    number = value.payload.value
    if isinstance(number, bool) or not isinstance(number, int):
        return None
    return number


def _guard_region(region: object, what: str) -> None:
    """取区的形状闸门（照 `roster._guard_region`：`(高, 宽, 3)` 的 uint8 RGB）。"""
    if not isinstance(region, np.ndarray):
        raise TypeError(f"{what}要是 numpy 数组，收到 {type(region).__name__}")
    if region.dtype != np.uint8 or region.ndim != 3 or region.shape[2] != 3:
        raise TypeError(
            f"{what}要是 (高, 宽, 3) 的 uint8 RGB，收到 {region.dtype} 的 "
            f"{getattr(region, 'shape', region)}"
        )
