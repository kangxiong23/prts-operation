"""§2 几何：格坐标 / 像素 / 画布 / 像素矩形 / 二维区域 / 视图与投影 / UI 布局常量。

以 test_a_ 开头的那组是 T1.2 验收 a：**一个像素值冒充不了格坐标（反之亦然）**，
并且每条断言都指名道姓地说出**是哪一处**把它拦住了。

标量区间那套操作（包含 / 相交，验收 b）**住在 §1 的三个载荷上**（Interval /
LowerBound / UpperBound）——它们共用一个载体，所以那几条在
test_contracts_slot.py 里，不在这里。
"""

from __future__ import annotations

import pytest

from ark.contracts import (
    PAYLOAD_TYPES,
    Canvas,
    Exact,
    GridCoord,
    Interval,
    LowerBound,
    Pixel,
    PixelRect,
    Projection,
    Region,
    UiLayout,
    Unknown,
    UpperBound,
    Valued,
    View,
)


def _projection(view: View = View.NORMAL) -> Projection:
    """一套最小可用的投影：192 像素一格（值在资产里，测试里自己搭）。"""

    def to_pixels(point: GridCoord) -> Pixel:
        return Pixel(point.x * 192, point.y * 192)

    def to_grid(point: Pixel) -> GridCoord:
        return GridCoord(point.x / 192, point.y / 192)

    return Projection(view=view, grid_to_pixel=to_pixels, pixel_to_grid=to_grid)


# ---------------------------------------------------------------------------
# a. 像素值冒充不了格坐标（反之亦然）
# ---------------------------------------------------------------------------


def test_a_pixel_cannot_pass_as_a_grid_coord() -> None:
    """a. 四处守卫各拦一种冒充，每一处都能指名。"""
    grid = GridCoord(5.4, 3.0)
    pixel = Pixel(1036.8, 300.0)
    projection = _projection()

    # ① 投影"格坐标 → 像素"只收格坐标
    with pytest.raises(TypeError) as excinfo:
        projection.to_pixels(pixel)
    assert "要的是格坐标" in str(excinfo.value)

    # ② 投影"像素 → 格坐标"只收像素
    with pytest.raises(TypeError) as excinfo:
        projection.to_grid(grid)
    assert "要的是像素坐标" in str(excinfo.value)

    # ③ 二维区域的心是格坐标
    with pytest.raises(TypeError) as excinfo:
        Region(pixel, 0.5)
    assert "二维区域的心是格坐标" in str(excinfo.value)

    # ④ 点的分量是数，不是另一个点
    with pytest.raises(TypeError) as excinfo:
        GridCoord(pixel, 0.0)
    assert "分量必须是数" in str(excinfo.value)
    with pytest.raises(TypeError):
        Pixel(grid, 0.0)

    # 而且两个点类型互相 isinstance 不成立、数值相同也不相等
    assert not isinstance(grid, Pixel)
    assert not isinstance(pixel, GridCoord)
    assert type(grid) is not type(pixel)
    assert GridCoord(1.0, 1.0) != Pixel(1.0, 1.0)


def test_a_bare_numbers_are_not_blocked() -> None:
    """a 的边界，显式记下来：**裸数不拦**（与刻度那边"裸 int 不拦"同一条）。

    Pixel(gp.x, gp.y) 仍然构造得出来 —— 那是调用方显式把两个数搬过去，不是
    类型在冒充。要连它也拦，就得把每个分量也做成单位类型，代价是裸数比较会
    静默变 False（刻度那边同样取舍过一次）。
    """
    grid = GridCoord(5.4, 3.0)
    assert Pixel(grid.x, grid.y) == Pixel(5.4, 3.0)


def test_point_components_must_be_numbers() -> None:
    for bad in (True, "5.4", None, [5.4], (5.4, 3.0)):
        with pytest.raises(TypeError):
            GridCoord(bad, 0.0)
        with pytest.raises(TypeError):
            Pixel(0.0, bad)
    # 连续格坐标：小数位是格内偏移，写得出来
    assert GridCoord(5.4, 3.0).x == 5.4
    assert GridCoord(5, 3).x == 5


# ---------------------------------------------------------------------------
# 画布 / 像素矩形
# ---------------------------------------------------------------------------


def test_canvas_is_a_positive_pixel_size() -> None:
    """画布是像素尺寸：正整数宽高。M1 的 1920×864 是前提头的值，不是这里的常数。"""
    m1 = Canvas(1920, 864)
    assert (m1.width, m1.height) == (1920, 864)
    with pytest.raises(ValueError):
        Canvas(0, 864)
    with pytest.raises(ValueError):
        Canvas(1920, -1)
    with pytest.raises(TypeError):
        Canvas(1920.0, 864)
    with pytest.raises(TypeError):
        Canvas(True, 864)


def test_pixel_rect_needs_a_positive_extent() -> None:
    rect = PixelRect(0, 0, 1920, 96)
    assert (rect.x, rect.y, rect.width, rect.height) == (0, 0, 1920, 96)
    with pytest.raises(ValueError):
        PixelRect(0, 0, 0, 96)
    with pytest.raises(ValueError):
        PixelRect(0, 0, 1920, -96)
    with pytest.raises(TypeError):
        PixelRect(0, 0, "1920", 96)


# ---------------------------------------------------------------------------
# 二维区域：§2 的形状，§1 的第六种载荷
# ---------------------------------------------------------------------------


def test_region_is_the_sixth_payload() -> None:
    """区域是一条载荷（不是"值的种类"）：它表达的是精度。"""
    region = Region(GridCoord(5.4, 3.0), 0.5)
    assert Region in PAYLOAD_TYPES
    assert len(PAYLOAD_TYPES) == 6
    assert Valued(region, 0.61).payload is region
    assert not hasattr(region, "form")


def test_region_is_a_point_plus_an_uncertain_radius() -> None:
    region = Region(GridCoord(5.4, 3.0), 0.5)
    assert region.center == GridCoord(5.4, 3.0)
    assert region.radius == 0.5
    assert Region(GridCoord(5.0, 3.0), 0).radius == 0  # 0 允许：就是一个点
    with pytest.raises(ValueError):
        Region(GridCoord(5.0, 3.0), -0.1)
    with pytest.raises(TypeError):
        Region(GridCoord(5.0, 3.0), "0.5")


# ---------------------------------------------------------------------------
# 视图与投影
# ---------------------------------------------------------------------------


def test_view_is_two_valued_and_excludes_pause() -> None:
    """视图只有两套（常速 / 慢放）——暂停不产生第三套投影，它不是视图。"""
    members = list(View)
    assert len(members) == 2
    assert [member.value for member in members] == ["常速", "慢放"]
    assert all("暂停" not in str(member) for member in members)


def test_projection_carries_its_view_in_the_type() -> None:
    """投影自带视图标识：标识是**类型上的一栏**，不是随手可改的一个标记。"""
    projection = _projection(View.NORMAL)
    assert projection.view is View.NORMAL
    fields = {name: str(field.type) for name, field in Projection.__dataclass_fields__.items()}
    assert fields["view"] == "View"
    with pytest.raises(TypeError):
        Projection(view="常速", grid_to_pixel=lambda p: p, pixel_to_grid=lambda p: p)


def test_projection_require_refuses_the_other_view() -> None:
    """两套混用是只有类型能拦住的错：要哪一套就点哪一套。"""
    normal = _projection(View.NORMAL)
    assert normal.require(View.NORMAL) is normal
    with pytest.raises(TypeError) as excinfo:
        normal.require(View.SLOW)
    assert "两套混用" in str(excinfo.value)


def test_projection_converts_in_both_directions() -> None:
    projection = _projection()
    pixels = projection.to_pixels(GridCoord(5.4, 3.0))
    assert (pixels.x, pixels.y) == pytest.approx((5.4 * 192, 3.0 * 192))
    back = projection.to_grid(Pixel(1036.8, 576.0))
    assert (back.x, back.y) == pytest.approx((5.4, 3.0))


def test_projection_needs_both_directions() -> None:
    with pytest.raises(TypeError):
        Projection(view=View.NORMAL, grid_to_pixel="不是函数", pixel_to_grid=lambda p: p)


# ---------------------------------------------------------------------------
# UI 布局常量
# ---------------------------------------------------------------------------


def test_ui_layout_is_a_read_only_table_of_named_pixel_rects() -> None:
    layout = UiLayout(rects={"hud": PixelRect(0, 0, 1920, 96), "cost": PixelRect(20, 8, 120, 40)})
    assert layout.rects["hud"] == PixelRect(0, 0, 1920, 96)
    assert list(layout.rects) == ["hud", "cost"]  # 有序
    with pytest.raises(TypeError):
        layout.rects["hud"] = PixelRect(0, 0, 1, 1)  # 表交出去就不能再改
    with pytest.raises(TypeError):
        UiLayout(rects={1: PixelRect(0, 0, 1, 1)})
    with pytest.raises(TypeError):
        UiLayout(rects={"hud": (0, 0, 1920, 96)})
    with pytest.raises(TypeError):
        UiLayout(rects=[("hud", PixelRect(0, 0, 1, 1))])


def test_exact_still_rejects_a_region() -> None:
    """放宽值域也不收载荷：二维区域是另一种形态，进不了精确。"""
    with pytest.raises(TypeError):
        Exact(Region(GridCoord(5.0, 3.0), 0.5))
    assert isinstance(Valued(Unknown(), None).payload, Unknown)
    assert isinstance(Valued(Interval(1, 2), 0.5).payload, Interval)
    assert isinstance(Valued(LowerBound(1), 0.5).payload, LowerBound)
    assert isinstance(Valued(UpperBound(2), 0.5).payload, UpperBound)
