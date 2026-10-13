"""红罩反解：**那层罩子底下还剩多少信息** —— 判据，不是接线。

《资产生成器/特征库》§1 那条分界线问的是两件事：**那层红罩是不是一个可解的线性
叠加**、**解掉之后剩下的信息够不够认出是谁**。这个模块只回答这两问，**不接进线上
那条链**：线上红罩卡那一格走的是 `margin` 那块取区与那把带掩码的尺子，**不用这里的
模型** —— 判据对逐通道仿射天生免疫（D-全局-28：反解换的是**单位**、不是修复），
所以 `fit_redcap` / `undo_redcap` 是**判据侧的测量工具**，线上不调它们
（`margin` 只从这儿借了 `TOP_STRIP_HEIGHT` 那个常量）。

## 罩子是什么（实测）

**它盖住整张卡**，连职业图标与费用数字都盖在底下（那两样被压成灰的，不是白的）。
所以「找一块没被盖住的画面」这条路不存在 —— 卡上**没有**未覆盖区。

模型取**逐通道仿射**：

    capped = p * clean + q          （p 是标量，q 是三通道的常数）

**四个参数**：`p` 与 `q` 的三个分量。为什么是仿射而不是「alpha 混合」：拟合出来的
`q` 有三个分量为负（实测 `q ≈ (53…63, -4…7, -6…6)`），一个非负颜色的 alpha 混合
给不出负分量。所以它是**一次颜色变换**，不是一次透明叠加。

**它不是精确的**：实测四对配对上的残差 rms 是 **8.9–10.1**（256 级上），也就是这条
仿射只解释得了罩住那张卡约 **七成** 的方差。这条残差是这一单最重要的读数之一 ——
`1/p ≈ 4` 会把它放大四倍，而身份链要的正是细节。

## 圆环与中央数字：掩掉，不还原

环与数字是**逐帧在变**的，红罩是**静态**的 —— 所以前者只能掩掉。掩的那把刀就是
`bench.RING_OUTER`（半径 60，环带的外半径）：圆心在卡内 `(卡宽/2, RING_CENTER_DY)`，
半径 60 以内的像素**不进拟合、也不进匹配**。

**掩掉之后剩多少**：整张卡（204×214）剩 **52.12%**（**22752 / 43656**），立绘区
（顶上那一条之外）剩 **66.78%**，而**身份区那个 100×100 的框**（卡内
`(52, 76, 100, 100)`）只剩 **7.48%**（**748** px）—— 最后那一条是《视觉层/备战席点名》
§红罩卡 早就量到、并且写明了「那件事要连框挪到环外一起考虑」的那条。

（**这一版把「整张卡剩多少」从 74.1% 改成 52.12%**：74.1% 是"整张卡里在半径 60 之外的
比例"，**没有**扣掉顶上那一条，而 `_countdown_mask` 扣了 —— 同一个模块里两个数不该
说两件事。）

## 结论（这一版量出来的）

**读数按哪个取区量，差一个量级。** 把反解后的卡喂给**同一条链**（`roster_scores`）：

| 取区 | 四对的分（cd14#1 / cd22#9 / cd27#6 / cd37#4） |
| --- | --- |
| **整块身份区** —— 环与中央数字**还在框里** | 0.3512 / 0.3824 / 0.4387 / 0.3609 |
| **身份区里排掉倒计时图形之后**那 748 px（7.48%） | **0.9476 / 0.8677 / 0.9598 / 0.9513** |

**第二行才是「罩子底下还剩多少信息」**：环与中央数字**逐帧在变**、不归罩子管，把它们
留在框里，量到的是「环 + 数字 + 反解出来的卡面混在一起」。第一行也留着 ——
它是"量错口径会得到什么"的取证。

（**这一版把「抬到 0.49」删掉了**：那是一个上一版在**别的取区**上读到的数，却写得
像"四对都这样"，与上面任何一行都对不上。现在两个口径并排，各自注明量的是哪一块。）

**「够 / 不够」是《资产生成器/特征库》§1 那张表的事，不是这个模块的事** ——
这里只给数与口径，裁在规划者。

## 两个入口

- `fit_redcap(pairs)`：一批 `(盖着的那块, 没盖的那块)` → 模型（四个参数 + 残差）；
- `undo_redcap(region, model)`：一块盖着的画面 → 解出来的画面。

**模型能存下来**：`model.to_json()` / `Redcap.from_json(text)`；**能报出自己用了几个
参数**：`model.parameters`。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ark.contracts import UI_LAYOUT
from ark.engine.vision.bench import RING_CENTER_DY, RING_OUTER

__all__ = ["fit_redcap", "undo_redcap"]


CAP_REDCAP_FLOOR: float = 0.02
CAP_REDCAP_CEIL: float = 0.60
"""拟合出来的 `p` 必须落在 `FLOOR…CEIL` 里 —— 它就是「罩子透过来多少」。

实测四对是 **0.13…0.25**，所以这两条线各有两倍以上的余量。它们挡的是**退化输入**：
把同一块画面自己配给自己，`p` 会跑到 1（那不是罩子，那是没配）；反过来 `p` 塌到 0
说明「没盖的那块」是一块死画面。
"""


TOP_STRIP_HEIGHT: int = int(
    UI_LAYOUT.rects["squad_card_rightmost_face"].y
) - int(UI_LAYOUT.rects["squad_card_rightmost"].y)
"""卡片顶上那一条（**职业图标 + 费用**）有多高 —— 从契约层那两个矩形量出来，不另抄。

**它不在模型里**：那一条上的图标与数字**不在**罩子那条仿射上 —— 实测同一对配对上，
立绘区的斜率是 **0.19–0.21**，而这一条的斜率按通道是 **0.45–0.51**（干净卡上纯白的数字
255 在罩住那张卡上是 **113–116**）。两段**不是同一层覆盖物**，所以那一片不能与立绘区
共用一个模型。

（**这一版把「画在罩子上面」这句话去掉了**：它既不是"在上面"（在上面就该是 255），
也不是"完全不服从"（它自己是一段斜率约 0.48 的仿射）。**它仍然看得清** —— 职业图标与
费用数字盖着也读得出来，那是另一件事，见 `margin.py`。）
按实测值 47。
"""


def _countdown_mask(shape: tuple[int, ...]) -> np.ndarray:
    """哪些像素**既不在倒计时图形里、又在立绘区里** —— `True` 就是可用。

    两把刀：

    - **倒计时图形**：圆心按卡片自己的几何取（横在卡片正中 `shape[1] / 2`、纵在离卡片
      上缘 `RING_CENTER_DY`），半径用 `bench.RING_OUTER`（环带外半径 60）—— 与读
      倒计时那条链**同一个数**，不另开一份；
    - **顶上那一条**（`TOP_STRIP_HEIGHT`）：职业图标与费用**画在罩子上面**，不是
      `p * clean + q`，所以它不进拟合。

    **它假定喂进来的是整张备战席卡**：不是整卡，圆心就不在那儿。所以这里只要求
    「装得下那个圆盘」，其余交给调用方 —— 这一层没有卡宽那个常量，也不该有。
    """
    if len(shape) < 2:
        raise ValueError(f"要一块 (高, 宽, 3) 的画面，收到形状 {shape}")
    height, width = int(shape[0]), int(shape[1])
    centre_x = width / 2.0
    centre_y = RING_CENTER_DY
    if centre_y - RING_OUTER < 0 or centre_x - RING_OUTER < 0:
        raise ValueError(
            f"{height}×{width} 装不下圆心在 ({centre_x}, {centre_y})、"
            f"半径 {RING_OUTER} 的那个圆盘：这一层要的是整张备战席卡"
        )
    rows = np.arange(height, dtype=np.float64)[:, None]
    cols = np.arange(width, dtype=np.float64)[None, :]
    distance = np.sqrt((cols - centre_x) ** 2 + (rows - centre_y) ** 2)
    inside_art = rows >= float(TOP_STRIP_HEIGHT)
    return (distance > float(RING_OUTER)) & inside_art


def _guard_block(block: object, what: str) -> np.ndarray:
    """一块卡面：`(高, 宽, 3)` 的 uint8 RGB。"""
    if not isinstance(block, np.ndarray):
        raise TypeError(f"{what} 要是一块 (高, 宽, 3) 的 uint8 RGB，收到 {type(block).__name__}")
    if block.dtype != np.uint8 or block.ndim != 3 or block.shape[2] != 3:
        raise TypeError(
            f"{what} 要是 (高, 宽, 3) 的 uint8 RGB，收到 {block.dtype} 的 "
            f"{getattr(block, 'shape', block)}"
        )
    return block


@dataclass(frozen=True)
class Redcap:
    """那层红罩的**四个常数**：`capped = slope * clean + offset`（逐通道）。

    `slope` 是标量、`offset` 是三个分量 —— 一共 **4 个参数**。`residual` 是拟合时
    在这批配对上留下的 rms（256 级），`pixels` 是真正参与拟合的像素数。

    **它不是 `__all__` 里的名字**：公开面就是 `fit_redcap` / `undo_redcap` 两个入口，
    调用方拿到的是 `fit_redcap` 的返回值，不需要自己拼一个 —— 但测试与落盘要用它。
    """

    slope: float
    offset: tuple[float, float, float]
    residual: float
    pixels: int
    pairs: int

    @property
    def parameters(self) -> int:
        """这个模型用了几个参数 —— **4**（一个斜率 + 三个偏置）。"""
        return 1 + len(self.offset)

    def to_json(self) -> str:
        """存下来：一段 JSON，喂回 `from_json` 就是同一个模型。"""
        return json.dumps(
            {
                "slope": self.slope,
                "offset": list(self.offset),
                "residual": self.residual,
                "pixels": self.pixels,
                "pairs": self.pairs,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "Redcap":
        """把 `to_json` 写下的那段读回来。字段缺一个就报错 —— 不补默认值。"""
        record = json.loads(text)
        if not isinstance(record, dict):
            raise TypeError(f"模型是一段 JSON 对象，收到 {type(record).__name__}")
        for key in ("slope", "offset", "residual", "pixels", "pairs"):
            if key not in record:
                raise KeyError(f"模型里没有 {key!r}：缺字段就是坏模型，不补默认值")
        offset = record["offset"]
        if not isinstance(offset, list) or len(offset) != 3:
            raise ValueError(f"offset 要三个数，收到 {offset!r}")
        return cls(
            slope=float(record["slope"]),
            offset=(float(offset[0]), float(offset[1]), float(offset[2])),
            residual=float(record["residual"]),
            pixels=int(record["pixels"]),
            pairs=int(record["pairs"]),
        )


def fit_redcap(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> Redcap:
    """一批 `(盖着的那块, 没盖的那块)` → 那层红罩的四个常数。

    **只拿「不在倒计时图形里」的像素拟合**（`_countdown_mask`）：环与数字逐帧在变，
    把它们算进去，拟合出来的就不是罩子。

    **一批一起拟合，不是一对一个模型**：罩子是静态的，所以四对必须落在同一组数上 ——
    逐对拟合会把每一对自己的噪声也吸收掉，那样「解出来了」是假的。

    **最小二乘只解一次**：把所有配对的可用像素摞起来，解 `capped = slope * clean +
    offset`（斜率三通道共用一个，偏置各通道一个）。方向是**从 clean 回归 capped**，
    不是反过来 —— 罩子把对比度压掉四分之三，反着回归会被量化噪声拉偏（实测反着做
    斜率会从 0.20 缩到 0.14）。
    """
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise TypeError(f"配对要是一串 (盖着的那块, 没盖的那块)，收到 {type(pairs).__name__}")
    if not pairs:
        raise ValueError("一对都没有：没有配对就拟合不出罩子，别硬做")

    shapes: set[tuple[int, ...]] = set()
    capped_parts: list[np.ndarray] = []
    clean_parts: list[np.ndarray] = []
    for position, pair in enumerate(pairs):
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise TypeError(f"第 {position} 对要是 (盖着的那块, 没盖的那块)，收到 {pair!r}")
        capped, clean = pair
        _guard_block(capped, f"第 {position} 对里盖着的那块")
        _guard_block(clean, f"第 {position} 对里没盖的那块")
        if capped.shape != clean.shape:
            raise ValueError(
                f"第 {position} 对两块形状不一样：{capped.shape} 对 {clean.shape} —— "
                "要配的是同一块画面，形状不同就不是同一块"
            )
        shapes.add(capped.shape)
    if len(shapes) != 1:
        raise ValueError(
            f"配对之间形状不一样：{sorted(shapes)} —— 罩子是同一层，块也必须同形"
        )

    shape = shapes.pop()
    mask = _countdown_mask(shape)
    for position, (capped, clean) in enumerate(pairs):
        capped_parts.append(capped[mask].astype(np.float64))
        clean_parts.append(clean[mask].astype(np.float64))
    clean_all = np.concatenate(clean_parts, axis=0)
    capped_all = np.concatenate(capped_parts, axis=0)

    clean_centred = clean_all - clean_all.mean(axis=0)
    capped_centred = capped_all - capped_all.mean(axis=0)
    denominator = float((clean_centred * clean_centred).sum())
    if denominator <= 0.0:
        raise ValueError("没盖的那些块是一块死画面（方差为零）：拟合不出斜率")
    slope = float((clean_centred * capped_centred).sum() / denominator)
    if not CAP_REDCAP_FLOOR < slope < CAP_REDCAP_CEIL:
        raise ValueError(
            f"拟合出来的斜率 {slope:.4f} 落在 {CAP_REDCAP_FLOOR}…{CAP_REDCAP_CEIL} 之外："
            "这不是「罩子透过来一点」，是这两批画面本来就不是同一块（或者根本没配）"
        )
    offset = capped_all.mean(axis=0) - slope * clean_all.mean(axis=0)
    predicted = slope * clean_all + offset
    residual = float(np.sqrt(((predicted - capped_all) ** 2).mean()))
    return Redcap(
        slope=slope,
        offset=(float(offset[0]), float(offset[1]), float(offset[2])),
        residual=residual,
        pixels=int(clean_all.shape[0]),
        pairs=len(pairs),
    )


def undo_redcap(region: np.ndarray, model: Redcap) -> np.ndarray:
    """一块盖着的画面 → **解出来的画面**（同一个形状、uint8）。

    就是那条仿射的逆：`clean = (capped - offset) / slope`，再夹回 0…255。

    **它不动倒计时图形**：环与数字不是罩子，解不出来，也没打算解 —— 那一片要由
    调用方按 `_countdown_mask` 掩掉（这就是为什么身份区那个框只剩 7.48% 可用）。

    **放大是代价**：`1 / slope ≈ 4`，所以拟合残差被放大四倍，量化噪声也一样。
    解出来的画面「看得出是谁」，但不够身份链认 —— 见模块文档。
    """
    _guard_block(region, "要解的那块画面")
    if not isinstance(model, Redcap):
        raise TypeError(f"模型要是 fit_redcap 给的那个，收到 {type(model).__name__}")
    offset = np.array(model.offset, dtype=np.float64)
    solved = (region.astype(np.float64) - offset) / model.slope
    return np.clip(solved, 0.0, 255.0).astype(np.uint8)
