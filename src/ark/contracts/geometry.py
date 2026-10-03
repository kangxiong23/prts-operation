"""§2 几何：格坐标 · 像素 · 画布 · 标量区间 · 二维区域 · 投影 · UI 布局常量。

这一节有两条硬要求：

- **格坐标是连续的**：整数部分是格，小数部分是**格内偏移**。多出来的那一位
  表达的是"格内偏移"，不是精度炫耀 —— 精度来自渠道（识别 / 推算），不来自刻度。
- **像素与格坐标在类型上分得开**：别让一个 (float, float) 既能当格又能当像素。
  两个点是两个类型，跨单位的每一处入口都带守卫（投影的两个方向、二维区域的心）。

两件事**不在**这里：

- 投影参数与 UI 布局常量的**值**（投影值在资产里；UI 常量等 T3 量出来再填，
  这一批只给类型）；
- 标量区间那套操作（包含 / 相交）**住在 §1 的三个载荷上**（Interval /
  LowerBound / UpperBound）——§2 的标量区间不是第四个类型，它就是那三个载荷
  共用的那套操作，别在这里再开一个。

**二维区域是个例外**：它的形状在 §2（本文件 Region），词表在 §1（PAYLOAD_TYPES）：
判据是"载荷的区分是知道得多准，值的种类是知道的是什么"——区域表达精度，
所以它是形态。于是本文件被 §1 的 slot 引用，方向是 slot → geometry。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias, final

__all__ = [
    "GridCoord",
    "Pixel",
    "Canvas",
    "PixelRect",
    "Region",
    "View",
    "Projection",
    "UiLayout",
]


def _is_number(value: object) -> bool:
    """实数（int 或 float）。bool 不算 —— 把 True 当 1 用只会掩盖错误。

    与 slot / containers 里同一条判据：契约层不 import 任何内部模块，几个叶子
    各自守着这两行，不为它多开一个文件。
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_number(value: object, what: str) -> None:
    if not _is_number(value):
        raise TypeError(f"{what}必须是数，收到 {type(value).__name__}")


# ---------------------------------------------------------------------------
# 格坐标与像素：两个类型，不是一个 (float, float)
# ---------------------------------------------------------------------------

_POINT_TYPES: tuple[type, ...] = ()


def _check_point_component(value: object, owner: str) -> None:
    """点的分量是数。**是"点"就拒**：那是拿一个点当另一个点的坐标用。"""
    if isinstance(value, _POINT_TYPES):
        raise TypeError(
            f"{owner}的分量必须是数，收到 {type(value).__name__}："
            "两个点是两个单位，要换算请用投影（Projection）"
        )
    if not _is_number(value):
        raise TypeError(f"{owner}的分量必须是数，收到 {type(value).__name__}")


@final
@dataclass(frozen=True)
class GridCoord:
    """格坐标：连续的二维点，单位是**格**。

    整数部分是格，小数部分是**格内偏移**：(5.4, 3.0) = 第 5 格内偏右 0.4 格、
    第 3 行。**它不产生精度**——精度来自渠道，写成小数不会让它变准；一格多物
    因此天然可区分，"跨在两格之间算哪格"不需要回答。

    **与像素分得开**：这是两个类型。**裸数不拦**——Pixel(gp.x, gp.y) 仍然构造
    得出来，那是调用方显式把两个数搬过去，不是类型在冒充；要连它也拦，就得把
    每个分量也做成单位类型，代价是裸数比较会静默变 False（刻度那边同样取舍过）。
    """

    x: int | float
    y: int | float

    def __post_init__(self) -> None:
        _check_point_component(self.x, "格坐标")
        _check_point_component(self.y, "格坐标")


@final
@dataclass(frozen=True)
class Pixel:
    """像素坐标：屏幕上的一个点，单位是**像素**。

    像素只在调试与 ROI 追溯里出现（格才是 AI 全部决策的单位）。它与格坐标是
    两个类型：**一个像素值冒充不了格坐标**，投影的两个方向各有一道守卫挡着。
    """

    x: int | float
    y: int | float

    def __post_init__(self) -> None:
        _check_point_component(self.x, "像素坐标")
        _check_point_component(self.y, "像素坐标")


_POINT_TYPES = (GridCoord, Pixel)


@final
@dataclass(frozen=True)
class Canvas:
    """画布：像素尺寸（宽 × 高）。

    归一化画布 = **捕获分辨率本身**，不缩放（用户裁定）：固定尺寸会引入插值
    误差，还会让"不吻合即拒绝"变成"先缩放再拒绝，等于已经适配过了"。画布尺寸
    进**前提头**（时序层 §3），不吻合即拒绝执行。M1 即 1920×864 —— 值是前提头
    的事，这一层只给类型。
    """

    width: int
    height: int

    def __post_init__(self) -> None:
        for what, value in (("宽", self.width), ("高", self.height)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"画布的{what}必须是整数像素，收到 {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"画布的{what}必须为正，收到 {value}")


@final
@dataclass(frozen=True)
class PixelRect:
    """像素矩形：像素坐标系里的一个矩形（ROI 与 UI 常量的形状）。

    x / y 是矩形的一角，宽高必须为正。**原点与朝向不在这里定**（左上还是左下、
    y 向上还是向下）——那是视觉层与资产的事。用 x / y / 宽 / 高 而不是
    left / top / right / bottom：后者把四个数里塞进两个冗余约束，更容易写出
    自相矛盾的矩形。
    """

    x: int | float
    y: int | float
    width: int | float
    height: int | float

    def __post_init__(self) -> None:
        _check_number(self.x, "矩形的 x")
        _check_number(self.y, "矩形的 y")
        for what, value in (("宽", self.width), ("高", self.height)):
            _check_number(value, f"矩形的{what}")
            if value <= 0:
                raise ValueError(f"矩形的{what}必须为正，收到 {value}")


# ---------------------------------------------------------------------------
# 二维区域：§2 的形状，§1 的第六种载荷
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class Region:
    """二维区域：点 + 不确定半径（格）。它是 **locate 的答案，也是第六种载荷**。

    为什么位置是二维的：路线资产是**格级的名义路径**，怪不走正中、拐弯不直角，
    被位移之后还会重走一条新的最短路径 —— **路径本身就不确定**，横向也有一份
    不确定。所以"沿路还有多远"是把这个区域投影到名义路径上的**一次算**，
    不是字段的形态。

    为什么它是**载荷**而不是"值的种类"：判据是「载荷的区分是知道得多准，
    值的种类是知道的是什么」——区域越小越准，所以它是**形态**。形状的出处是
    §2（本文件），词表的出处是 §1（slot 的 PAYLOAD_TYPES 收着它）。
    """

    center: GridCoord
    radius: int | float

    def __post_init__(self) -> None:
        if not isinstance(self.center, GridCoord):
            raise TypeError(
                f"二维区域的心是格坐标，收到 {type(self.center).__name__}："
                "像素请先用投影换成格"
            )
        _check_number(self.radius, "不确定半径")
        if self.radius < 0:
            raise ValueError(f"不确定半径不许为负，收到 {self.radius}")


# ---------------------------------------------------------------------------
# 投影：格坐标 ↔ 像素，自带「视图」标识
# ---------------------------------------------------------------------------


class View(Enum):
    """**视图**：投影有两套，常速一套、慢放一套（§2）。

    注意它与「**时间态**」不是一回事：时间态是三值的（常速 / 慢放 / 暂停），
    而投影只有两套 —— 暂停不产生第三套投影。两者混掉，投影就会去带一个它
    没有的值，所以这里**只有两支**。
    """

    NORMAL = "常速"
    SLOW = "慢放"


@final
@dataclass(frozen=True)
class Projection:
    """格坐标 ↔ 像素 的映射，**自带视图标识**（§2）。值在资产里。

    - view 是**类型上的标识**：拿常速那套去算慢放下的部署点击或识别 ROI，
      错得静默且系统性；require() 是它在运行期的落点（要哪一套就点哪一套）；
    - grid_to_pixel / pixel_to_grid 是资产给的函数（等比缩放、偏移、慢放下的
      倾斜都在里面）：这一层不规定算法，只规定两个方向都要有；
    - 两个方向各带一道单位守卫：to_pixels 只收格坐标、to_grid 只收像素。
    """

    view: View
    grid_to_pixel: Callable[[GridCoord], Pixel]
    pixel_to_grid: Callable[[Pixel], GridCoord]

    def __post_init__(self) -> None:
        if not isinstance(self.view, View):
            raise TypeError(f"投影的视图标识要是 View，收到 {type(self.view).__name__}")
        for what, func in (
            ("格坐标 → 像素", self.grid_to_pixel),
            ("像素 → 格坐标", self.pixel_to_grid),
        ):
            if not callable(func):
                raise TypeError(f"投影的{what}必须可调用，收到 {type(func).__name__}")

    def require(self, view: View) -> Projection:
        """要一套指定视图的投影：给错了就抛 TypeError。

        两套混用是只有类型能拦住的错；调用方手里那套是哪一个，快照页眉知道。
        """
        if self.view is not view:
            raise TypeError(
                f"这里要「{view.value}」那一套投影，拿到的是「{self.view.value}」："
                "两套混用是只有类型能拦住的错"
            )
        return self

    def to_pixels(self, point: GridCoord) -> Pixel:
        """格坐标 → 像素。**只收格坐标**：喂像素进来就抛 TypeError。"""
        if not isinstance(point, GridCoord):
            raise TypeError(
                f"要的是格坐标，收到 {type(point).__name__}："
                "像素请走 to_grid（或先用 to_grid 换回格）"
            )
        return self.grid_to_pixel(point)

    def to_grid(self, point: Pixel) -> GridCoord:
        """像素 → 格坐标。**只收像素**：喂格坐标进来就抛 TypeError。"""
        if not isinstance(point, Pixel):
            raise TypeError(
                f"要的是像素坐标，收到 {type(point).__name__}：格坐标请走 to_pixels"
            )
        return self.pixel_to_grid(point)


# ---------------------------------------------------------------------------
# UI 布局常量：具名像素矩形的表
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class UiLayout:
    """UI 布局常量：**具名像素矩形的表**（键是名字，值是矩形）。

    它不住在资产端口里（用户裁定）：它不是生成物、不进资产库，而设备层与视觉
    层都要读它、两者又互不 import —— 所以它住这一层。**值是 T3 量出来再填的**，
    这一批只给类型。

    表本身是只读的（交出去之后谁也不能改一个格子）；名字有哪些不在这里定
    （分界线表：那是状态层与视觉层的事）。
    """

    rects: Mapping[str, PixelRect]

    # 映射本来就不可哈希：显式声明，比留一个到运行期才炸的生成哈希好。
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.rects, Mapping):
            raise TypeError(f"UI 布局常量要收映射，收到 {type(self.rects).__name__}")
        for name, rect in self.rects.items():
            if not isinstance(name, str) or not name:
                raise TypeError(f"UI 常量的键是非空字符串，收到 {name!r}")
            if not isinstance(rect, PixelRect):
                raise TypeError(f"UI 常量 {name!r} 的值要是像素矩形，收到 {type(rect).__name__}")
        object.__setattr__(self, "rects", MappingProxyType(dict(self.rects)))
