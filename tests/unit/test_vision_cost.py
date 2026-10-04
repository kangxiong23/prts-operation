"""视觉层读数链（T3）的单元测试。

这一批**不 import tests/spec**（D-全局-17：那是规划者的验收套件），也**不抄**
那 31 帧的真值 —— 这里测的是「读数的形状与边界」：常量抄得对不对、原始帧解得
对不对、字形表的往返、以及三种降级各自落在哪一支上。

用到夹具的地方只有一处（出厂字形表能不能从取材帧重建），用的是
`tests/帧样本/费用/» 这份数据夹具，不是验收套件。
"""

from __future__ import annotations

import shutil
import struct
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from ark.contracts import Canvas, Exact, Interval, PixelRect, UiLayout, Unknown, Unsampled, Valued
from ark.engine.vision import cost as cost_mod
from ark.engine.vision.frame import RAW_HEADER_BYTES, Frame, frame_from_raw
from ark.engine.vision.glyphs import (
    DEFAULT_GLYPHS_PATH,
    GlyphTable,
    glyph_boxes,
    ink_of,
    normalize,
    similarity,
)
from ark.engine.vision.layout import DOCUMENTED_WINDOWS, RECT_NAMES, UI

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "帧样本" / "费用"


@pytest.fixture
def scratch() -> Iterator[Path]:
    """一个**自己能进去**的临时目录。

    **不用 pytest 的 tmp_path**：它建目录时用 0o700，而 0o700 的 DACL 不继承
    父目录 —— 在受限令牌下（本机 DSH 沙箱就是这样），那个目录连创建它的进程
    自己都进不去（icacls 都读不到）。这里用**默认权限**建，父目录的 ACL 会继承
    下来，所以在沙箱里与不在沙箱里都能跑。
    """
    root = Path(tempfile.gettempdir()) / "ark-vision-cost-tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex[:12]
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# §区域与常量：8 个具名矩形
# ---------------------------------------------------------------------------


def test_ui_layout_has_exactly_the_eight_documented_rects() -> None:
    assert len(RECT_NAMES) == 8
    assert set(UI.rects) == set(DOCUMENTED_WINDOWS)
    # squad_row 是「边缘 + 卡距」，卡距是标量，UiLayout 装不下 —— 不许硬造矩形。
    assert "squad_row" not in UI.rects


@pytest.mark.parametrize("name", sorted(DOCUMENTED_WINDOWS))
def test_every_rect_is_the_documented_window_plus_its_inclusive_end(name: str) -> None:
    x0, y0, x1, y1 = DOCUMENTED_WINDOWS[name]
    rect = UI.rects[name]
    assert (rect.x, rect.y) == (x0, y0)
    assert rect.x + rect.width - 1 == x1
    assert rect.y + rect.height - 1 == y1


def test_cost_bar_is_the_track_itself_153_px() -> None:
    rect = UI.rects["cost_bar"]
    assert rect.width == 153  # 文档自己写的：2741 - 2589 + 1 = 153
    assert rect.x == 2589 and rect.y == 945


# ---------------------------------------------------------------------------
# 原始帧 → 画布
# ---------------------------------------------------------------------------


def _raw(width: int, height: int, rgba: np.ndarray, fmt: int = 1, space: int = 1) -> bytes:
    head = struct.pack("<4I", width, height, fmt, space)
    return head + rgba.astype(np.uint8).tobytes()


def test_frame_from_raw_reads_rgba_in_channel_order() -> None:
    rgba = np.array([[[10, 20, 30, 255], [40, 50, 60, 128]]], dtype=np.uint8)
    frame = frame_from_raw(_raw(2, 1, rgba))
    assert frame.canvas == Canvas(width=2, height=1)
    assert frame.pixels.shape == (1, 2, 3)
    assert frame.pixels.dtype == np.uint8
    # R 最大、G 次之、B 最小 —— 解成 BGRA 的话顺序会反过来。
    assert frame.pixels[0, 0].tolist() == [10, 20, 30]
    assert frame.pixels[0, 1].tolist() == [40, 50, 60]


def test_frame_from_raw_rejects_a_length_that_does_not_match_the_header() -> None:
    rgba = np.zeros((1, 2, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="长度对不上"):
        frame_from_raw(_raw(2, 1, rgba)[:-1])


def test_frame_from_raw_rejects_an_unknown_pixel_format() -> None:
    rgba = np.zeros((1, 2, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="像素格式"):
        frame_from_raw(_raw(2, 1, rgba, fmt=2))


def test_frame_from_raw_rejects_a_truncated_header() -> None:
    with pytest.raises(ValueError, match="头"):
        frame_from_raw(b"\x00" * (RAW_HEADER_BYTES - 1))


def test_frame_crop_takes_the_rect_shape_and_is_a_view() -> None:
    pixels = np.zeros((10, 20, 3), dtype=np.uint8)
    frame = Frame(canvas=Canvas(width=20, height=10), pixels=pixels)
    roi = frame.crop(PixelRect(x=2, y=3, width=5, height=4))
    assert roi.shape == (4, 5, 3)
    roi[0, 0, 0] = 9
    assert pixels[3, 2, 0] == 9  # 视图，不是拷贝


def test_frame_crop_clips_at_the_canvas_edge_instead_of_inventing_pixels() -> None:
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    frame = Frame(canvas=Canvas(width=6, height=4), pixels=pixels)
    # 文档的 cost_panel 右缘写到了 2800（含两端口径），比画布宽 1 px。
    assert frame.crop(PixelRect(x=4, y=1, width=3, height=2)).shape == (2, 2, 3)


def test_frame_rejects_pixels_that_do_not_match_the_canvas() -> None:
    with pytest.raises(ValueError, match="对不上"):
        Frame(canvas=Canvas(width=6, height=4), pixels=np.zeros((4, 5, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 切 / 整 / 比
# ---------------------------------------------------------------------------


def test_glyph_boxes_splits_on_blank_columns() -> None:
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    rgb[:, 1:3] = 255
    rgb[:, 7:11] = 255
    assert glyph_boxes(rgb) == [(1, 0, 3, 8), (7, 0, 11, 8)]


def test_glyph_boxes_finds_nothing_in_a_blank_roi() -> None:
    assert glyph_boxes(np.zeros((8, 12, 3), dtype=np.uint8)) == []


def test_ink_of_is_the_minimum_channel() -> None:
    rgb = np.array([[[255, 10, 200]]], dtype=np.uint8)
    assert ink_of(rgb).tolist() == [[10]]


def test_normalize_is_deterministic_and_keeps_the_aspect_ratio() -> None:
    wide = np.full((40, 40), 255, dtype=np.uint8)
    narrow = np.full((40, 10), 255, dtype=np.uint8)
    assert np.array_equal(normalize(wide), normalize(wide))
    # 窄的字形归一之后仍然是窄的 —— 宽度里带着「这是哪个字形」的信息。
    assert normalize(wide).shape == normalize(narrow).shape == (64, 64)
    assert int((normalize(narrow).max(axis=0) > 0).sum()) < int((normalize(wide).max(axis=0) > 0).sum())


def test_similarity_is_one_for_itself_and_rejects_mismatched_shapes() -> None:
    block = normalize(np.full((40, 30), 255, dtype=np.uint8))
    assert similarity(block, block) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="尺寸不一样"):
        similarity(block, block[:, :10])


# ---------------------------------------------------------------------------
# 字形表：往返与格式守卫
# ---------------------------------------------------------------------------


def _table(digits=(0, 1)) -> GlyphTable:
    templates = {
        digit: np.full((64, 64), 0 if digit == 0 else 255, dtype=np.uint8)
        for digit in digits
    }
    return GlyphTable(digits=tuple(digits), templates=templates)


def test_glyph_table_round_trips_through_disk(scratch: Path) -> None:
    table = _table()
    path = table.save(scratch / "glyphs")
    assert path.name == "digits.npz"
    back = GlyphTable.load(path)
    assert back.digits == table.digits
    for digit in table.digits:
        assert np.array_equal(back.templates[digit], table.templates[digit])
    # 收目录也收文件
    assert GlyphTable.load(scratch / "glyphs").digits == table.digits


def test_glyph_table_rejects_a_key_set_that_does_not_match_digits() -> None:
    with pytest.raises(ValueError, match="键必须一致"):
        GlyphTable(digits=(1,), templates={1: np.zeros((64, 64), np.uint8)} | {2: np.zeros((64, 64), np.uint8)})


def test_glyph_table_rejects_another_format_version(scratch: Path) -> None:
    path = scratch / "digits.npz"
    np.savez_compressed(
        path,
        format=np.asarray("ark.glyph_table/999"),
        digits=np.asarray([1], dtype=np.int64),
        shape=np.asarray([64, 64], dtype=np.int64),
        templates=np.zeros((1, 64, 64), np.uint8),
    )
    with pytest.raises(ValueError, match="只认"):
        GlyphTable.load(path)


def test_shipped_glyph_table_covers_all_ten_digits() -> None:
    table = GlyphTable.load(DEFAULT_GLYPHS_PATH)
    assert table.digits == tuple(range(10))


# ---------------------------------------------------------------------------
# 读数链
# ---------------------------------------------------------------------------


def _digits_frame(templates: list[tuple[int, int, np.ndarray]], width: int = 220, height: int = 80):
    """把几张模板贴到一张画布上，ROI 就是整张画布。"""
    rect = PixelRect(x=0, y=0, width=width, height=height)
    ui = UiLayout(rects={"cost_digits": rect})
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for x, y, template in templates:
        canvas[y:y + template.shape[0], x:x + template.shape[1], :] = template[:, :, None]
    frame = Frame(canvas=Canvas(width=width, height=height), pixels=canvas)
    return frame, ui


def test_read_cost_digits_reads_the_templates_back_as_an_exact_integer() -> None:
    table = GlyphTable.load(DEFAULT_GLYPHS_PATH)
    frame, ui = _digits_frame([(5, 8, table.templates[4]), (110, 8, table.templates[2])])
    got = cost_mod.read_cost_digits(frame, table, ui=ui)
    assert isinstance(got, Valued) and isinstance(got.payload, Exact)
    assert got.payload.value == 42
    assert got.confidence is not None


def test_read_cost_digits_without_any_stroke_is_unsampled() -> None:
    table = GlyphTable.load(DEFAULT_GLYPHS_PATH)
    frame, ui = _digits_frame([])
    assert cost_mod.read_cost_digits(frame, table, ui=ui) == Unsampled()


def test_read_cost_digits_that_match_nothing_widens_to_the_whole_range() -> None:
    table = GlyphTable.load(DEFAULT_GLYPHS_PATH)
    rng = np.random.default_rng(20240617)
    noise = rng.integers(0, 256, size=(64, 40), dtype=np.uint8)
    # **必须摆在中间**：真读数的数字水平居中于窗口中心，摆在边上会被
    # 「缺位判据」当成缺了一位的两位数（见下一条）。
    frame, ui = _digits_frame([(90, 8, noise)])
    got = cost_mod.read_cost_digits(frame, table, ui=ui)
    assert isinstance(got, Valued) and isinstance(got.payload, Interval)
    assert (got.payload.lo, got.payload.hi) == (0, 9)  # 一位：全部十个数字


def test_read_cost_digits_lone_glyph_off_center_is_a_missing_digit_not_a_shorter_number() -> None:
    """缺了一位的两位数，长得就像一个更短的一位数 —— 这是最危险的那种假事实。"""
    table = GlyphTable.load(DEFAULT_GLYPHS_PATH)
    # `10» 的末位被整块涂掉之后，剩下的那个 `1» 偏在窗口中心**左边** 31 px。
    frame, ui = _digits_frame([(47, 8, table.templates[1])])
    got = cost_mod.read_cost_digits(frame, table, ui=ui)
    assert isinstance(got, Valued) and isinstance(got.payload, Interval)
    assert (got.payload.lo, got.payload.hi) == (10, 19)
    # 反过来：偏在右边 → 缺的那一位在它**前**面
    frame, ui = _digits_frame([(109, 8, table.templates[0])])
    got = cost_mod.read_cost_digits(frame, table, ui=ui)
    assert isinstance(got, Valued) and isinstance(got.payload, Interval)
    assert got.payload.lo == 0 and got.payload.hi == 90


def test_read_cost_digits_lone_glyph_at_the_center_is_still_an_exact_integer() -> None:
    table = GlyphTable.load(DEFAULT_GLYPHS_PATH)
    frame, ui = _digits_frame([(78, 8, table.templates[7])])
    got = cost_mod.read_cost_digits(frame, table, ui=ui)
    assert isinstance(got, Valued) and isinstance(got.payload, Exact)
    assert got.payload.value == 7


def test_read_cost_digits_more_than_two_blocks_is_unknown() -> None:
    table = GlyphTable.load(DEFAULT_GLYPHS_PATH)
    frame, ui = _digits_frame(
        [(5, 8, table.templates[1]), (70, 8, table.templates[2]), (135, 8, table.templates[3])]
    )
    got = cost_mod.read_cost_digits(frame, table, ui=ui)
    assert isinstance(got, Valued) and isinstance(got.payload, Unknown)


def _bar_frame(columns: list[int], height: int = 6):
    width = len(columns)
    ui = UiLayout(rects={"cost_bar": PixelRect(x=0, y=0, width=width, height=height)})
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for x, value in enumerate(columns):
        canvas[:, x, :] = value
    return Frame(canvas=Canvas(width=width, height=height), pixels=canvas), ui


def test_read_cost_bar_counts_the_leading_bright_columns() -> None:
    frame, ui = _bar_frame([255] * 7 + [50] * 13)
    got = cost_mod.read_cost_bar(frame, ui=ui)
    assert isinstance(got, Valued) and isinstance(got.payload, Exact)
    assert got.payload.value == 7


def test_read_cost_bar_all_dark_is_exact_zero() -> None:
    frame, ui = _bar_frame([50] * 20)
    got = cost_mod.read_cost_bar(frame, ui=ui)
    assert isinstance(got, Valued) and isinstance(got.payload, Exact)
    assert got.payload.value == 0


def test_read_cost_bar_entirely_bright_is_the_whole_track_not_the_full_state() -> None:
    frame, ui = _bar_frame([255] * 20)
    got = cost_mod.read_cost_bar(frame, ui=ui)
    assert isinstance(got, Valued) and isinstance(got.payload, Exact)
    assert got.payload.value == 20


def test_read_cost_bar_middle_gray_with_max_is_unknown_not_zero() -> None:
    frame, ui = _bar_frame([146] * 8 + [255] * 4 + [146] * 8)
    got = cost_mod.read_cost_bar(frame, ui=ui)
    assert isinstance(got, Valued) and isinstance(got.payload, Unknown)
    assert got.confidence is None


# ---------------------------------------------------------------------------
# 资产生成器
# ---------------------------------------------------------------------------


def _harvest() -> dict[int, int]:
    import json

    payload = json.loads((FIXTURES / "取材.json").read_text(encoding="utf-8"))
    return {int(item["shot"]): int(item["cost"]) for item in payload["harvest"]}


def test_build_digit_glyphs_is_reproducible_pixel_for_pixel(scratch: Path) -> None:
    from ark.offline.assets_gen.glyphs import build_digit_glyphs

    first = build_digit_glyphs(
        _harvest(), scratch / "a", crops_dir=FIXTURES, rect=UI.rects["cost_digits"]
    )
    second = build_digit_glyphs(
        _harvest(), scratch / "b", crops_dir=FIXTURES, rect=UI.rects["cost_digits"]
    )
    assert first.read_bytes() == second.read_bytes()
    assert GlyphTable.load(first).digits == tuple(range(10))


def test_build_digit_glyphs_refuses_a_harvest_that_misses_a_digit(scratch: Path) -> None:
    from ark.offline.assets_gen.glyphs import build_digit_glyphs

    harvest = _harvest()
    harvest.pop(1)  # shot01 是「2」的唯一来源
    with pytest.raises(ValueError, match="覆盖不到"):
        build_digit_glyphs(
            harvest, scratch / "c", crops_dir=FIXTURES, rect=UI.rects["cost_digits"]
        )