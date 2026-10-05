"""视觉层「顶部两个数」读数链（T7）的单元测试。

这一批**不 import tests/spec**（D-全局-17），也不带规划者那套验收的检查编号 ——
这里测的是**这条链自己的形状与边界**：取墨的两条口径、分隔符的形状判据、
三条降级各自落在哪一支上、以及「一边降级不牵连另一边」。

用到的夹具是 `tests/帧样本/顶部两个数/`（第二批基准集，35 帧）。它**没有真值
文件**（基准集 §1：真值住 `tests/spec/`），所以下面那张 `TRUTH` 是**按像素读
出来的**：35 帧里每一个真数字的分数都在 0.933 以上、分隔符在 0.708–0.778，
中间隔着 0.90 那条线，没有一处要猜。

**降级用例是在真帧上涂改出来的**，不是自己画的图：涂黑、涂成实心白块、抹掉某
一位 —— 这样量到的仍然是**真帧上的几何**。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ark.contracts import Exact, Interval, PixelRect, UiLayout, Unknown, Unsampled, Valued
from ark.engine.vision.frame import Frame, frame_from_raw
from ark.engine.vision.glyphs import (
    DEFAULT_GLYPHS_PATH,
    INK_MAX,
    INK_MIN,
    INK_THRESHOLD,
    GlyphTable,
    glyph_boxes,
    ink_of,
    normalize,
)
from ark.engine.vision.layout import UI
from ark.engine.vision.topbar import (
    ENEMY_INK_TAKE,
    ENEMY_INK_THRESHOLD,
    read_enemy_count,
    read_life_digits,
)
from ark.offline.bench.batch import BenchBatch, load_batch

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "帧样本" / "顶部两个数"

TRUTH: dict[str, tuple[int, int, int]] = {
    # 帧名: (已击杀, 敌人总数, 生命)
    "shot01": (0, 8, 1),
    "shot02": (1, 8, 1),
    "shot03": (1, 8, 1),
    "shot04": (1, 8, 1),
    "shot05": (3, 8, 1),
    "shot06": (3, 8, 1),
    "shot07": (3, 8, 1),
    "shot08": (3, 8, 1),
    "shot09": (3, 8, 1),
    "shot16": (0, 22, 1),
    "shot17": (0, 22, 1),
    "shot18": (0, 22, 1),
    "shot19": (0, 22, 1),
    "shot20": (0, 22, 1),
    "shot21": (0, 22, 1),
    "shot22": (0, 22, 1),
    "shot23": (0, 22, 1),
    "shot24": (2, 22, 1),
    "shot25": (3, 22, 1),
    "shot26": (14, 22, 1),
    "shot27": (14, 22, 1),
    "shot28": (14, 22, 1),
    "shot29": (14, 22, 1),
    "shot30": (15, 22, 1),
    "shot31": (17, 22, 1),
    "shot32": (17, 22, 1),
    "shot33": (19, 22, 1),
    "shot34": (20, 22, 1),
    "shot35": (20, 22, 1),
    "shot36": (21, 22, 1),
    "shot37": (22, 22, 1),
    "m2_view": (40, 41, 3),
    "m2x_pview": (40, 41, 3),
    "raw_a": (21, 41, 3),
    "raw_b": (25, 41, 3),
}


@pytest.fixture(scope="module")
def batch() -> BenchBatch:
    return load_batch(FIXTURES)


@pytest.fixture(scope="module")
def table() -> GlyphTable:
    return GlyphTable.load(DEFAULT_GLYPHS_PATH)


def _frame(batch: BenchBatch, name: str) -> Frame:
    """一条裁剪 → 一整帧（贴回 2800×1260 的画布，ROI 的取法仍然是真的在被测）。"""
    case = next(case for case in batch.cases if case.name == name)
    return frame_from_raw(case.raw())


def _paint(frame: Frame, rect: PixelRect, spans, color) -> Frame:
    """在帧的副本上，把 ROI 里的几个矩形涂成一个颜色。`spans` 是 **ROI 坐标**。"""
    pixels = frame.pixels.copy()
    for x0, y0, x1, y1 in spans:
        pixels[rect.y + y0:rect.y + y1, rect.x + x0:rect.x + x1, :] = color
    return Frame(canvas=frame.canvas, pixels=pixels)


def _enemy_boxes(frame: Frame, table: GlyphTable):
    """敌数窗口里每一个框的「框 + 最像的那个字形的分数」，ROI 坐标。"""
    rect = UI.rects["enemy_count"]
    roi = frame.crop(rect)
    ink = ink_of(roi, take=ENEMY_INK_TAKE)
    scored = []
    for box in glyph_boxes(roi, take=ENEMY_INK_TAKE, threshold=ENEMY_INK_THRESHOLD):
        block = ink[box[1]:box[3], box[0]:box[2]]
        scored.append((box, max(table.scores(normalize(block)).values())))
    return scored


def _enemy_parts(frame: Frame, table: GlyphTable):
    """(甲, 分隔符, 乙) 三组框 —— 分隔符按**分数**认（它是唯一对不上任何数字的框）。

    只有测试里这么认：链本身用的是**形状**判据，那正是这一单的要害（挡住一位时
    分数会同时废掉两个框，形状不会）。
    """
    scored = _enemy_boxes(frame, table)
    cut = min(range(len(scored)), key=lambda index: scored[index][1])
    return (
        [box for box, _ in scored[:cut]],
        scored[cut][0],
        [box for box, _ in scored[cut + 1:]],
    )


def _life_with_ones(frame: Frame, at: list[int]) -> Frame:
    """把生命窗口里那个 `1` 复制到给定的几个位置（基准集里一张两位数的生命帧都没有）。"""
    rect = UI.rects["life_digits"]
    roi = frame.crop(rect)
    glyph = roi[:, 30:41, :].copy()
    blank = np.zeros_like(roi)
    for x in at:
        blank[:, x:x + 11, :] = glyph
    pixels = frame.pixels.copy()
    pixels[rect.y:rect.y + rect.height, rect.x:rect.x + rect.width, :] = blank
    return Frame(canvas=frame.canvas, pixels=pixels)


# ---------------------------------------------------------------------------
# 两个窗口：复量后的值
# ---------------------------------------------------------------------------


def test_the_two_topbar_windows_are_the_remeasured_ones() -> None:
    """两个窗口都换过值（T7 复量），而且判据锚在窗口中心上，所以值不许自己改。"""
    assert UI.rects["enemy_count"] == PixelRect(x=1200, y=33, width=186, height=40)
    assert UI.rects["life_digits"] == PixelRect(x=1574, y=33, width=70, height=40)
    # 锚点 x = 1292.5 ≈ 窗口中心 1293（敌数）；x = 1608.5 ≈ 窗口中心 1609（生命）。
    assert UI.rects["enemy_count"].x + UI.rects["enemy_count"].width / 2 == 1293.0
    assert UI.rects["life_digits"].x + UI.rects["life_digits"].width / 2 == 1609.0


def test_the_four_cost_windows_are_untouched_by_this_chain() -> None:
    """生命与敌数换窗口**不牵连**费用那四个。"""
    assert UI.rects["cost_panel"] == PixelRect(x=2585, y=845, width=215, height=128)
    assert UI.rects["cost_icon"] == PixelRect(x=2595, y=873, width=64, height=64)
    assert UI.rects["cost_digits"] == PixelRect(x=2666, y=868, width=111, height=69)
    assert UI.rects["cost_bar"] == PixelRect(x=2589, y=945, width=153, height=10)


# ---------------------------------------------------------------------------
# 取墨：按通道
# ---------------------------------------------------------------------------


def test_ink_of_takes_the_minimum_channel_by_default() -> None:
    rgb = np.array([[[255, 10, 200], [255, 120, 120]]], dtype=np.uint8)
    assert ink_of(rgb).tolist() == [[10, 120]]
    assert ink_of(rgb, take=INK_MIN).tolist() == ink_of(rgb).tolist()


def test_ink_of_can_take_the_maximum_channel_instead() -> None:
    """粉字 `(255,120,120)` 的 `min` 只有 120 —— 只看 `max` 才看得见它。"""
    rgb = np.array([[[255, 120, 120]]], dtype=np.uint8)
    assert ink_of(rgb, take=INK_MAX).tolist() == [[255]]


def test_ink_of_refuses_a_take_it_does_not_know() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="取墨口径"):
        ink_of(rgb, take="mean")


def test_glyph_boxes_defaults_stay_on_the_minimum_channel_route() -> None:
    """默认路径必须与今天逐位一致 —— 费用那条链走的就是它。"""
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    rgb[:, 1:3] = 255
    rgb[:, 7:11] = 255
    assert glyph_boxes(rgb) == [(1, 0, 3, 8), (7, 0, 11, 8)]
    assert glyph_boxes(rgb) == glyph_boxes(rgb, take=INK_MIN, threshold=INK_THRESHOLD)


def test_glyph_boxes_can_split_a_low_contrast_pink_stroke() -> None:
    """共用那套 `min(R,G,B) ≥ 128` 对生命整个是瞎的；换 `max` 才切得出来。"""
    rgb = np.zeros((8, 12, 3), dtype=np.uint8)
    rgb[:, 3:6] = (255, 120, 120)
    assert glyph_boxes(rgb) == []
    assert glyph_boxes(rgb, take=INK_MAX, threshold=150) == [(3, 0, 6, 8)]


def test_glyph_boxes_refuses_a_threshold_that_is_not_an_integer() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="墨线"):
        glyph_boxes(rgb, threshold=128.0)


# ---------------------------------------------------------------------------
# 正常读：35 帧基准集
# ---------------------------------------------------------------------------


def test_the_truth_table_covers_every_frame_in_the_batch(batch: BenchBatch) -> None:
    assert {case.name for case in batch.cases} == set(TRUTH)


@pytest.mark.parametrize("name", sorted(TRUTH))
def test_every_baseline_frame_reads_both_numbers_and_life_exactly(
    batch: BenchBatch, table: GlyphTable, name: str
) -> None:
    kills, total, life = TRUTH[name]
    frame = _frame(batch, name)
    got_kills, got_total = read_enemy_count(frame, table)
    assert isinstance(got_kills, Valued) and isinstance(got_kills.payload, Exact)
    assert got_kills.payload.value == kills
    assert isinstance(got_total, Valued) and isinstance(got_total.payload, Exact)
    assert got_total.payload.value == total
    got_life = read_life_digits(frame, table)
    assert isinstance(got_life, Valued) and isinstance(got_life.payload, Exact)
    assert got_life.payload.value == life


# ---------------------------------------------------------------------------
# 降级一：整块没笔画 → 未采样
# ---------------------------------------------------------------------------


def test_a_whole_roi_without_a_stroke_is_unsampled(batch: BenchBatch, table: GlyphTable) -> None:
    """涂黑：连分隔符也没有 —— 那时报「未知」等于说「那里有东西」。"""
    frame = _frame(batch, "shot26")
    rect = UI.rects["enemy_count"]
    blacked = _paint(frame, rect, [(0, 0, rect.width, rect.height)], (0, 0, 0))
    assert read_enemy_count(blacked, table) == (Unsampled(), Unsampled())
    life_rect = UI.rects["life_digits"]
    blacked = _paint(frame, life_rect, [(0, 0, life_rect.width, life_rect.height)], (0, 0, 0))
    assert read_life_digits(blacked, table) == Unsampled()


# ---------------------------------------------------------------------------
# 降级二：某一位对不上任何字形 → 区间（只那一边）
# ---------------------------------------------------------------------------


def test_a_digit_that_matches_nothing_widens_only_that_side(
    batch: BenchBatch, table: GlyphTable
) -> None:
    """实心白块：有笔画、每一列都是整个字高，但哪个数字都不是。"""
    frame = _frame(batch, "shot26")  # 甲 14、乙 22
    rect = UI.rects["enemy_count"]
    kills, _, total = _enemy_parts(frame, table)
    assert len(kills) == 2 and len(total) == 2
    painted = _paint(frame, rect, [kills[1]], (255, 255, 255))
    got_kills, got_total = read_enemy_count(painted, table)
    assert isinstance(got_kills, Valued) and isinstance(got_kills.payload, Interval)
    assert (got_kills.payload.lo, got_kills.payload.hi) == (10, 19)
    # 另一边一位都没少
    assert isinstance(got_total, Valued) and isinstance(got_total.payload, Exact)
    assert got_total.payload.value == 22


# ---------------------------------------------------------------------------
# 降级三：缺一位 → 区间（不许报成更短的确切数）
# ---------------------------------------------------------------------------


def test_a_missing_inner_digit_widens_only_that_side(
    batch: BenchBatch, table: GlyphTable
) -> None:
    """抹掉甲的内端那位（末位）：它就不再贴着分隔符了。"""
    frame = _frame(batch, "shot26")
    rect = UI.rects["enemy_count"]
    kills, _, _ = _enemy_parts(frame, table)
    erased = _paint(frame, rect, [kills[1]], (0, 0, 0))
    got_kills, got_total = read_enemy_count(erased, table)
    assert isinstance(got_kills, Valued) and isinstance(got_kills.payload, Interval)
    assert (got_kills.payload.lo, got_kills.payload.hi) == (10, 19)
    assert isinstance(got_total, Valued) and isinstance(got_total.payload, Exact)
    assert got_total.payload.value == 22


def test_a_missing_outer_digit_is_not_a_shorter_number(
    batch: BenchBatch, table: GlyphTable
) -> None:
    """抹掉甲的外端那位（首位）：整个复合体因此不再居中，剩下的那一位不许报成确切值。"""
    frame = _frame(batch, "shot26")
    rect = UI.rects["enemy_count"]
    kills, _, _ = _enemy_parts(frame, table)
    erased = _paint(frame, rect, [kills[0]], (0, 0, 0))
    got_kills, got_total = read_enemy_count(erased, table)
    assert isinstance(got_kills, Valued) and isinstance(got_kills.payload, Interval)
    assert (got_kills.payload.lo, got_kills.payload.hi) == (4, 94)
    assert isinstance(got_total, Valued) and isinstance(got_total.payload, Exact)
    assert got_total.payload.value == 22


def test_a_missing_outer_digit_of_the_total_is_not_a_shorter_number(
    batch: BenchBatch, table: GlyphTable
) -> None:
    """反过来：抹掉乙的外端那位（末位），偏的是另一个方向。"""
    frame = _frame(batch, "raw_b")  # 甲 25、乙 41
    rect = UI.rects["enemy_count"]
    _, _, total = _enemy_parts(frame, table)
    erased = _paint(frame, rect, [total[1]], (0, 0, 0))
    got_kills, got_total = read_enemy_count(erased, table)
    assert isinstance(got_total, Valued) and isinstance(got_total.payload, Interval)
    assert (got_total.payload.lo, got_total.payload.hi) == (40, 49)
    assert isinstance(got_kills, Valued) and isinstance(got_kills.payload, Exact)
    assert got_kills.payload.value == 25


# ---------------------------------------------------------------------------
# 分隔符：有它自己的判据
# ---------------------------------------------------------------------------


def test_the_separator_disappearing_makes_both_sides_unknown(
    batch: BenchBatch, table: GlyphTable
) -> None:
    frame = _frame(batch, "shot26")
    rect = UI.rects["enemy_count"]
    _, separator, _ = _enemy_parts(frame, table)
    erased = _paint(frame, rect, [separator], (0, 0, 0))
    assert read_enemy_count(erased, table) == (
        Valued(Unknown(), None),
        Valued(Unknown(), None),
    )


def test_a_blob_over_the_separator_is_not_a_separator(
    batch: BenchBatch, table: GlyphTable
) -> None:
    """一团实心白块盖在分隔符上：它有笔画，但**不是一条细斜笔画**。"""
    frame = _frame(batch, "shot26")
    rect = UI.rects["enemy_count"]
    _, separator, _ = _enemy_parts(frame, table)
    painted = _paint(frame, rect, [separator], (255, 255, 255))
    assert read_enemy_count(painted, table) == (
        Valued(Unknown(), None),
        Valued(Unknown(), None),
    )


def test_a_wide_blob_is_not_mistaken_for_the_separator(
    batch: BenchBatch, table: GlyphTable
) -> None:
    """把甲的两位涂成一整块：它自己过不了形状判据，**真分隔符照样是那一个**。

    于是这一块被当成「一个认不出的字形」→ 那一位的候选是全部十个数字。
    （它其实盖住了两位 —— 规格里没有「一块太宽，当不了一位」这条判据，
    见报告的「撞到的规格洞」。）
    """
    frame = _frame(batch, "shot26")
    rect = UI.rects["enemy_count"]
    kills, separator, _ = _enemy_parts(frame, table)
    span = (kills[0][0], kills[0][1], kills[-1][2], kills[-1][3])
    assert separator[0] > span[2]  # 那一块确实没盖到分隔符
    painted = _paint(frame, rect, [span], (255, 255, 255))
    got_kills, got_total = read_enemy_count(painted, table)
    assert isinstance(got_kills, Valued) and isinstance(got_kills.payload, Interval)
    assert (got_kills.payload.lo, got_kills.payload.hi) == (0, 9)
    assert isinstance(got_total, Valued) and isinstance(got_total.payload, Exact)
    assert got_total.payload.value == 22


# ---------------------------------------------------------------------------
# 生命：三条降级 + 缺一位（基准集里没有两位数，所以自己拼）
# ---------------------------------------------------------------------------


def test_life_without_a_stroke_is_unsampled(batch: BenchBatch, table: GlyphTable) -> None:
    frame = _frame(batch, "shot01")
    rect = UI.rects["life_digits"]
    erased = _paint(frame, rect, [(30, 0, 41, rect.height)], (0, 0, 0))
    assert read_life_digits(erased, table) == Unsampled()


def test_life_that_matches_nothing_widens_to_all_ten_digits(
    batch: BenchBatch, table: GlyphTable
) -> None:
    frame = _frame(batch, "shot01")
    rect = UI.rects["life_digits"]
    painted = _paint(frame, rect, [(30, 0, 41, rect.height)], (255, 255, 255))
    got = read_life_digits(painted, table)
    assert isinstance(got, Valued) and isinstance(got.payload, Interval)
    assert (got.payload.lo, got.payload.hi) == (0, 9)


def test_life_two_digits_centred_is_exact(batch: BenchBatch, table: GlyphTable) -> None:
    frame = _life_with_ones(_frame(batch, "shot01"), [14, 44])
    got = read_life_digits(frame, table)
    assert isinstance(got, Valued) and isinstance(got.payload, Exact)
    assert got.payload.value == 11


def test_life_lone_digit_off_centre_is_a_missing_digit(batch: BenchBatch, table: GlyphTable) -> None:
    """偏左说明缺的在它右边，偏右说明缺的在它左边 —— 都不许报成更短的确切数。"""
    frame = _life_with_ones(_frame(batch, "shot01"), [14])
    got = read_life_digits(frame, table)
    assert isinstance(got, Valued) and isinstance(got.payload, Interval)
    assert (got.payload.lo, got.payload.hi) == (10, 19)
    frame = _life_with_ones(_frame(batch, "shot01"), [44])
    got = read_life_digits(frame, table)
    assert isinstance(got, Valued) and isinstance(got.payload, Interval)
    assert (got.payload.lo, got.payload.hi) == (1, 91)


def test_life_more_than_two_blocks_is_unknown(batch: BenchBatch, table: GlyphTable) -> None:
    frame = _life_with_ones(_frame(batch, "shot01"), [4, 27, 50])
    got = read_life_digits(frame, table)
    assert isinstance(got, Valued) and isinstance(got.payload, Unknown)
    assert got.confidence is None


# ---------------------------------------------------------------------------
# 闸门
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reader", [read_life_digits, read_enemy_count])
def test_readers_refuse_an_object_that_is_not_a_frame(reader, table: GlyphTable) -> None:
    with pytest.raises(TypeError, match="要读的是一帧"):
        reader(np.zeros((4, 4, 3), dtype=np.uint8), table)


@pytest.mark.parametrize("reader", [read_life_digits, read_enemy_count])
def test_readers_refuse_a_table_that_is_not_a_glyph_table(reader, batch: BenchBatch) -> None:
    frame = _frame(batch, "shot01")
    with pytest.raises(TypeError, match="要一张字形表"):
        reader(frame, {0: np.zeros((64, 64), np.uint8)})


@pytest.mark.parametrize("reader", [read_life_digits, read_enemy_count])
def test_readers_refuse_a_ui_that_does_not_know_the_window(
    reader, batch: BenchBatch, table: GlyphTable
) -> None:
    frame = _frame(batch, "shot01")
    with pytest.raises(KeyError, match="UI 常量里没有"):
        reader(frame, table, ui=UiLayout(rects={}))
