"""字形表：把「数字」变成「10 张可以比对的图」。

《视觉层/识别方法》§已定：数字一律走**固定字形模板**，不上通用 OCR。理由是
失效模式 —— 通用 OCR 认错字是**假事实**，固定模板对不上就是**干净降级**。

这一层只做三件事，都不含语义：

1. **切**：ROI 里哪几列有笔画（`glyph_boxes»）—— 数字是横向排的，笔画之间必然
   有空白列，所以按「有笔画的列连成一段」切就够了；
2. **整**：把每块切成固定尺寸（`normalize»）—— **按高度归一，宽度按比例**：
   高度是字号（固定），宽度随字形变（`1» 只有 `0» 的一半），所以宽度里带着
   「这是哪个字形」的信息，不能一起拉齐；
3. **比**：`similarity» —— 零均值归一化相关（Pearson），配一个 ±2 px 的
   水平平移搜索，吃掉两次读数之间 1 px 的定位抖动。

**这一层不判断「够不够像」**：阈值是读数链（`cost.py»）的事 —— 同一次匹配，
「整块没笔画」和「有一位对不上」要落成不同的槽，那是字段级的降级策略
（契约层 §1「三态是字段级的」）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np

__all__ = [
    "GlyphTable",
    "ink_of",
    "glyph_boxes",
    "normalize",
    "similarity",
    "TEMPLATE_HEIGHT",
    "TEMPLATE_WIDTH",
    "INK_THRESHOLD",
    "SHIFT_SEARCH",
    "GLYPH_TABLE_FORMAT",
    "TABLE_FILENAME",
    "DEFAULT_GLYPHS_PATH",
]


TEMPLATE_HEIGHT: Final[int] = 64
TEMPLATE_WIDTH: Final[int] = 64
"""模板尺寸。笔画高 57–59 px（文档），归一到 64 —— 略微放大，不丢笔画。"""

INK_THRESHOLD: Final[int] = 128
"""「这一列有笔画」的判据：`min(R,G,B)» 到这个值。

面板底色是 47–67 的渐变（文档：暗态每列最亮 ≤67），笔画是 255，
所以 128 稳稳落在两者之间 —— 取 67 与 255 的中点（161）也一样，中间没有东西。
**建表与读表用的是同一个值**：两边不一致，切出来的框就会差一列。
"""

SHIFT_SEARCH: Final[int] = 2
"""比对时允许的水平平移（±px）。字形是按紧外框居中的，两次读数最多差 1–2 px。"""

GLYPH_TABLE_FORMAT: Final[str] = "ark.glyph_table/1"
TABLE_FILENAME: Final[str] = "digits.npz"
"""字形表的文件名。格式是这一层自己定的（T3 任务：文件格式你定）。"""

DEFAULT_GLYPHS_PATH: Final[Path] = (
    Path(__file__).resolve().parents[4] / "assets" / "features" / "glyphs" / TABLE_FILENAME
)
"""出厂字形表（进版本库，clone 下来即用）。**读数链默认读它，不现场重建。**"""


# ---------------------------------------------------------------------------
# 切
# ---------------------------------------------------------------------------


def ink_of(rgb: np.ndarray) -> np.ndarray:
    """墨迹：`min(R,G,B)»，逐像素。

    白字的三通道都亮，所以取**最小值**就是「有多白」；取最大值或均值会让
    带色偏的像素（例如半透明的粉）显得比实际亮。返回 `(高, 宽)» 的 uint8。
    """
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] < 3:
        raise TypeError(
            f"墨迹要从 (高, 宽, ≥3) 的像素里取，收到 {getattr(rgb, 'shape', rgb)}"
        )
    return rgb[:, :, :3].min(axis=2)


def glyph_boxes(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """把 ROI 的笔画切成一块一块，返回 `(x0, y0, x1, y1)»，**右/下开**。

    判据是「这一列有没有笔画」：有笔画的列连成一段就是一块。数字是横向排的，
    块与块之间必然有空白列（实测 9–12 px），所以不需要更聪明的分割 ——
    而「按连通域分割」会把 `4» 的横竖、`7» 的撇捺这类**一个字形内部的断开**
    切成好几块，反而更糟。

    返回的框是**紧外框**（那一块里笔画真正占到的行与列），不是整块 ROI。
    """
    ink = ink_of(rgb)
    mask = ink >= INK_THRESHOLD
    columns = mask.any(axis=0)
    boxes: list[tuple[int, int, int, int]] = []
    width = int(columns.shape[0])
    i = 0
    while i < width:
        if not columns[i]:
            i += 1
            continue
        j = i
        while j + 1 < width and columns[j + 1]:
            j += 1
        rows = np.flatnonzero(mask[:, i:j + 1].any(axis=1))
        boxes.append((i, int(rows[0]), j + 1, int(rows[-1]) + 1))
        i = j + 1
    return boxes


# ---------------------------------------------------------------------------
# 整
# ---------------------------------------------------------------------------


def _resize_bilinear(image: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """双线性缩放（自己写，免得视觉层为了一个缩放去依赖图像库）。

    像素中心对齐：输出第 k 个像素对应输入的
    `(k + 0.5) × 输入尺寸 / 输出尺寸 - 0.5»。越界的取样点夹到边缘。
    **确定性**：同样的输入永远给同样的输出。
    """
    src_h, src_w = image.shape
    ys = (np.arange(out_height, dtype=np.float64) + 0.5) * (src_h / out_height) - 0.5
    xs = (np.arange(out_width, dtype=np.float64) + 0.5) * (src_w / out_width) - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, src_h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = np.clip(ys - y0, 0.0, 1.0)[:, None]
    wx = np.clip(xs - x0, 0.0, 1.0)[None, :]
    top = image[np.ix_(y0, x0)] * (1.0 - wx) + image[np.ix_(y0, x1)] * wx
    bottom = image[np.ix_(y1, x0)] * (1.0 - wx) + image[np.ix_(y1, x1)] * wx
    return top * (1.0 - wy) + bottom * wy


def normalize(ink: np.ndarray) -> np.ndarray:
    """把一块墨迹整成 `TEMPLATE_HEIGHT × TEMPLATE_WIDTH» 的浮点图（0..1）。

    **按高度归一，宽度按比例，再水平居中。** 高度是字号、是固定的；宽度是
    字形自己的属性（`1» 窄、`0» 宽），把它也拉齐等于把一条判据扔掉。

    宽度超过模板宽度时**只缩到模板宽**（不再按高度算），宁可压扁也不裁掉笔画 ——
    裁掉是丢证据，压扁只是变形，而变形对模板匹配是一致的（建表时也这么压）。
    """
    if not isinstance(ink, np.ndarray) or ink.ndim != 2:
        raise TypeError(
            f"要归一的是 (高, 宽) 的一块墨迹，收到 {getattr(ink, 'shape', ink)}"
        )
    height, width = ink.shape
    if height <= 0 or width <= 0:
        raise ValueError(f"这块墨迹是空的：{ink.shape}")
    scaled_width = int(round(width * TEMPLATE_HEIGHT / float(height)))
    scaled_width = max(1, min(scaled_width, TEMPLATE_WIDTH))
    resized = _resize_bilinear(ink.astype(np.float64), TEMPLATE_HEIGHT, scaled_width)
    canvas = np.zeros((TEMPLATE_HEIGHT, TEMPLATE_WIDTH), dtype=np.float64)
    left = (TEMPLATE_WIDTH - scaled_width) // 2
    canvas[:, left:left + scaled_width] = resized
    return canvas / 255.0


# ---------------------------------------------------------------------------
# 比
# ---------------------------------------------------------------------------


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """零均值归一化相关：亮度整体偏一点不影响，形状像不像才算数。"""
    av = a - a.mean()
    bv = b - b.mean()
    na = float(np.sqrt(float((av * av).sum())))
    nb = float(np.sqrt(float((bv * bv).sum())))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float((av * bv).sum() / (na * nb))


def _shift_columns(image: np.ndarray, dx: int) -> np.ndarray:
    """水平平移，越出的那几列补 0（不是绕回来 —— 绕回来会伪造笔画）。"""
    if dx == 0:
        return image
    out = np.zeros_like(image)
    if dx > 0:
        out[:, dx:] = image[:, :-dx]
    else:
        out[:, :dx] = image[:, -dx:]
    return out


def similarity(query: np.ndarray, template: np.ndarray) -> float:
    """两块归一化墨迹有多像：`[-1, 1]» 的相关系数，取 ±SHIFT_SEARCH 里最好的那个。

    上界 1.0 = 逐像素一致。**不是概率**，也别当概率用 —— 契约层只说置信度是
    「无量纲的连续量」，没给它一个 [0,1] 的语义。
    """
    if query.shape != template.shape:
        raise ValueError(f"两块墨迹尺寸不一样：{query.shape} vs {template.shape}")
    best = -1.0
    for dx in range(-SHIFT_SEARCH, SHIFT_SEARCH + 1):
        score = _pearson(query, _shift_columns(template, dx))
        if score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# 表
# ---------------------------------------------------------------------------


def _resolve_table_path(path: Path, *, for_write: bool) -> Path:
    """`load» / `save» 收的既可以是那个文件，也可以是它所在的目录。

    **判据是后缀**：写盘时给一个没有后缀的路径（`assets/features/glyphs»）就是
    「写到这个目录里去」，给一个带后缀的路径（`…/digits.npz»）就是那个文件本身。
    读盘时目录必须已经存在。
    """
    path = Path(path)
    if not path.is_dir() and for_write and path.suffix == "":
        return path / TABLE_FILENAME
    if path.is_dir():
        named = path / TABLE_FILENAME
        if named.exists() or for_write:
            return named
        candidates = sorted(p for p in path.iterdir() if p.is_file())
        if len(candidates) == 1:
            return candidates[0]
        raise FileNotFoundError(
            f"{path} 里没有 {TABLE_FILENAME}，也不是「只有一个文件」的目录："
            f"找到 {[p.name for p in candidates]}"
        )
    return path


@dataclass(frozen=True)
class GlyphTable:
    """字形表：`digits» 是表里有哪些数字，`templates» 是每个数字那一张图。

    - `digits» 与 `templates» 的键必须**一模一样**：少一个键就是一张读不出
      那个数字的表，而它看起来仍然「有表」，那是最坏的一种不知道；
    - 模板是 `(TEMPLATE_HEIGHT, TEMPLATE_WIDTH)» 的 uint8：**存 uint8 不存浮点**，
      体积小一个数量级，而「出厂表能不能从取材帧重现」要的是**完全相等**，
      不是「近似相等」；
    - 表是只读的（交出去之后谁也不能改一张模板）。
    """

    digits: tuple[int, ...]
    templates: Mapping[int, np.ndarray]

    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        digits = tuple(self.digits)
        if not digits:
            raise ValueError("字形表至少要有一种字形")
        for digit in digits:
            if isinstance(digit, bool) or not isinstance(digit, int):
                raise TypeError(f"字形表的键是数字，收到 {type(digit).__name__}")
            if not 0 <= digit <= 9:
                raise ValueError(f"字形表的键是 0–9，收到 {digit}")
        if len(set(digits)) != len(digits):
            raise ValueError(f"字形表里的数字不许重复：{digits}")
        if not isinstance(self.templates, Mapping):
            raise TypeError(f"字形表的模板要是映射，收到 {type(self.templates).__name__}")
        if set(self.templates) != set(digits):
            raise ValueError(
                f"digits 与 templates 的键必须一致："
                f"digits={sorted(digits)}，templates={sorted(self.templates)}"
            )
        for digit in digits:
            block = self.templates[digit]
            if not isinstance(block, np.ndarray):
                raise TypeError(
                    f"字形 {digit} 的模板要是 ndarray，收到 {type(block).__name__}"
                )
            if block.shape != (TEMPLATE_HEIGHT, TEMPLATE_WIDTH):
                raise ValueError(
                    f"字形 {digit} 的模板要是 "
                    f"{(TEMPLATE_HEIGHT, TEMPLATE_WIDTH)}，收到 {block.shape}"
                )
            if block.dtype != np.uint8:
                raise TypeError(f"字形 {digit} 的模板要是 uint8，收到 {block.dtype}")
        object.__setattr__(self, "digits", digits)
        object.__setattr__(
            self,
            "templates",
            MappingProxyType({digit: self.templates[digit] for digit in digits}),
        )

    @classmethod
    def load(cls, path: Path) -> "GlyphTable":
        """从磁盘读回一张表。**只认自己写出来的那个格式**（版本对不上就报错）。"""
        resolved = _resolve_table_path(path, for_write=False)
        with np.load(resolved, allow_pickle=False) as data:
            keys = set(data.files)
            required = {"format", "digits", "shape", "templates"}
            if keys != required:
                raise ValueError(
                    f"{resolved} 不是字形表：字段是 {sorted(keys)}，"
                    f"要的是 {sorted(required)}"
                )
            fmt = str(data["format"])
            if fmt != GLYPH_TABLE_FORMAT:
                raise ValueError(
                    f"{resolved} 的格式是 {fmt!r}，这一版只认 {GLYPH_TABLE_FORMAT!r}"
                )
            shape = tuple(int(v) for v in data["shape"])
            if shape != (TEMPLATE_HEIGHT, TEMPLATE_WIDTH):
                raise ValueError(
                    f"{resolved} 的模板尺寸是 {shape}，"
                    f"要的是 {(TEMPLATE_HEIGHT, TEMPLATE_WIDTH)}"
                )
            digits = tuple(int(v) for v in data["digits"])
            stacked = data["templates"]
        if stacked.shape != (len(digits), TEMPLATE_HEIGHT, TEMPLATE_WIDTH):
            raise ValueError(
                f"{resolved} 的 templates 形状 {stacked.shape} 与 "
                f"{len(digits)} 个字形对不上"
            )
        return cls(
            digits=digits,
            templates={digit: np.array(stacked[i]) for i, digit in enumerate(digits)},
        )

    def save(self, path: Path) -> Path:
        """写出一张表，返回**真正落盘的那个文件**的路径。"""
        target = _resolve_table_path(path, for_write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        stacked = np.stack([self.templates[digit] for digit in self.digits]).astype(np.uint8)
        np.savez_compressed(
            target,
            format=np.asarray(GLYPH_TABLE_FORMAT),
            digits=np.asarray(self.digits, dtype=np.int64),
            shape=np.asarray([TEMPLATE_HEIGHT, TEMPLATE_WIDTH], dtype=np.int64),
            templates=stacked,
        )
        return target

    def scores(self, query: np.ndarray) -> dict[int, float]:
        """这块墨迹对每个字形各得多少分（**不做判断**，判断留给读数链）。"""
        return {
            digit: similarity(query, template)
            for digit, template in self.templates.items()
        }
