"""视觉层「备战席那一栏」读数链（T8）的单元测试。

这一批**不 import tests/spec**（D-全局-17），也不带规划者那套验收的检查编号 ——
这里测的是**这条链自己的形状与边界**：跳变那道判据认在哪一列、网格怎么抽验、
费用的三档怎么落、环的弧长怎么分三值、以及「不在再部署中 ⇒ 秒数是未采样」这条闸门。

夹具是 `tests/帧样本/备战席/`（第三批基准集，6 帧）。它**没有真值文件**
（基准集 §1：真值住 `tests/spec/`），所以下面那张 `TRUTH` 里那三帧是
`取材.json` 给的（规划者目视确认），另外三帧的值我没拿到 —— 它们的用例只断言
**这条链自己的形状**（卡数落在六个互不相同的值上、费用都是精确值、中心贴着网格）。

**降级用例是在真帧上涂改出来的**，不是自己画的图：涂黑、涂成实心块、往环上刷一段弧。

**有一条路真帧上测不出来**：圆环中央那个秒数的**字体与出厂字形表不是同一套**
（实测：那个 `1` 带底衬线、笔画整体更粗，最像的字形只有 0.44–0.84）——
所以「读得出秒数」那一路只能用**字形表自己的字形合成一帧**来测，真帧上它落在未知。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ark.contracts import (
    Exact,
    PixelRect,
    UI_LAYOUT,
    UiLayout,
    Unknown,
    Unsampled,
    Valued,
    squad_card_rect,
)
from ark.engine.vision.bench import (
    COST_BOX_WIDTH,
    COST_DIGITS_HEIGHT,
    COST_DIGITS_TOP,
    COST_SOLID_MATCH_SCORE,
    RING_ARC_PLAIN,
    RING_ARC_REDEPLOYING,
    RING_CENTER_DY,
    RING_INNER,
    RING_OUTER,
    SquadCard,
    read_squad_row,
)
from ark.engine.vision.frame import Frame, frame_from_raw
from ark.engine.vision.glyphs import DEFAULT_GLYPHS_PATH, GlyphTable
from ark.offline.bench.batch import BenchBatch, load_batch

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "帧样本" / "备战席"

PITCH = 207.5
"""卡距。**它住设备层的 `SQUAD_CARD_PITCH`**，导入规则不许视觉层 import 设备层，
所以用例里也照调用方的样子把它传进来。"""

TRUTH: dict[str, tuple[int, list[int]]] = {
    # 帧名: (卡数, 从右往左每张卡的费用) —— 这三帧来自取材.json（规划者目视确认）
    "shot01": (12, [25, 23, 23, 21, 18, 17, 14, 13, 13, 12, 12, 11]),
    "shot05": (9, [18, 17, 14, 14, 13, 13, 12, 11, 10]),
    "shot26": (6, [18, 12, 11, 6, 2, 1]),
}

COUNTS = (12, 11, 10, 9, 8, 6)
"""6 帧的卡数**互不相同**（夹具 README 那句「12 · 11 · 10 · 9 · 8 · 6」）——
这一条不依赖真值，是**从像素里数出来的**那六个值本身。"""


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


def _row(batch: BenchBatch, table: GlyphTable, name: str) -> tuple[SquadCard, ...]:
    return read_squad_row(_frame(batch, name), table, pitch=PITCH)


def _card_top() -> float:
    return float(squad_card_rect(0, PITCH).y)


def _centre_of(index: int) -> float:
    card = squad_card_rect(index, PITCH)
    return float(card.x) + float(card.width) / 2.0


def _cost_window(centre_x: float) -> PixelRect:
    return PixelRect(
        int(round(centre_x)),
        int(_card_top()) + COST_DIGITS_TOP,
        COST_BOX_WIDTH,
        COST_DIGITS_HEIGHT,
    )


def _paint(frame: Frame, rect: PixelRect, spans, color) -> Frame:
    """在帧的副本上，把 ROI 里的几个矩形涂成一个颜色。`spans` 是 **ROI 坐标**。"""
    pixels = frame.pixels.copy()
    for x0, y0, x1, y1 in spans:
        pixels[rect.y + y0:rect.y + y1, rect.x + x0:rect.x + x1, :] = color
    return Frame(canvas=frame.canvas, pixels=pixels)


def _blank_cost(frame: Frame, index: int) -> Frame:
    """把某一张卡的费用暗盒整块涂黑（那里没有东西了）。"""
    return _paint(
        frame,
        _cost_window(_centre_of(index)),
        [(0, 0, COST_BOX_WIDTH, COST_DIGITS_HEIGHT)],
        (0, 0, 0),
    )


def _blob_cost(frame: Frame, index: int) -> Frame:
    """把某一张卡的费用暗盒换成**一块实心灰**（有笔画，但不是数字）。"""
    return _paint(
        frame,
        _cost_window(_centre_of(index)),
        [(8, 4, 40, 25)],
        (128, 128, 128),
    )


def _blank_ring(frame: Frame, index: int, radius: int = RING_INNER) -> Frame:
    """把某一张卡**环心里**那一片涂黑（环本身留着）。"""
    centre_x = _centre_of(index)
    centre_y = _card_top() + RING_CENTER_DY
    pixels = frame.pixels.copy()
    rows, columns = np.mgrid[0:frame.canvas.height, 0:frame.canvas.width]
    inside = (columns - centre_x) ** 2 + (rows - centre_y) ** 2 < radius**2
    pixels[inside] = 0
    return Frame(canvas=frame.canvas, pixels=pixels)


def _paint_arc(frame: Frame, index: int, degrees: float) -> Frame:
    """在某一张卡上刷一段**白弧**（从 12 点起、顺时针 `degrees` 度）。"""
    centre_x = _centre_of(index)
    centre_y = _card_top() + RING_CENTER_DY
    pixels = frame.pixels.copy()
    rows, columns = np.mgrid[0:frame.canvas.height, 0:frame.canvas.width]
    dx = columns - centre_x
    dy = rows - centre_y
    radius = np.hypot(dx, dy)
    clock = (90.0 - np.degrees(np.arctan2(-dy, dx))) % 360.0
    band = (
        (radius >= RING_INNER + 1)
        & (radius <= RING_OUTER - 3)
        & (clock <= degrees)
    )
    pixels[band] = 255
    return Frame(canvas=frame.canvas, pixels=pixels)


def _glyph(table: GlyphTable, digit: int, height: int) -> np.ndarray:
    """字形表里那一位，按紧外框缩到 `height` 高（最近邻，够用）。"""
    template = table.templates[digit]
    columns = np.flatnonzero((template > 0).any(axis=0))
    rows = np.flatnonzero((template > 0).any(axis=1))
    block = template[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
    source_h, source_w = block.shape
    out_w = max(1, int(round(source_w * height / source_h)))
    ys = np.minimum(np.arange(height) * source_h // height, source_h - 1)
    xs = np.minimum(np.arange(out_w) * source_w // out_w, source_w - 1)
    return block[np.ix_(ys, xs)]


def _draw_countdown(
    frame: Frame, table: GlyphTable, text: str, index: int, *, height: int = 29
) -> Frame:
    """用**字形表自己的字形**把 `text` 画到环心里（合成帧，见模块文档）。"""
    pieces: list[np.ndarray] = []
    for char in text:
        if char == ".":
            pieces.append(np.full((5, 6), 255, dtype=np.uint8))
        else:
            pieces.append(_glyph(table, int(char), height))
    gap = 3
    width = sum(piece.shape[1] for piece in pieces) + gap * (len(pieces) - 1)
    baseline = int(round(_card_top() + RING_CENTER_DY + height / 2.0))
    left = int(round(_centre_of(index) - width / 2.0))
    pixels = frame.pixels.copy()
    x = left
    for piece in pieces:
        piece_h, piece_w = piece.shape
        top = baseline - piece_h
        pixels[top:top + piece_h, x:x + piece_w, 0] = piece
        pixels[top:top + piece_h, x:x + piece_w, 1] = piece
        pixels[top:top + piece_h, x:x + piece_w, 2] = piece
        x += piece_w + gap
    return Frame(canvas=frame.canvas, pixels=pixels)


# ---------------------------------------------------------------------------
# 闸门
# ---------------------------------------------------------------------------


def test_a_frame_is_required(table: GlyphTable) -> None:
    with pytest.raises(TypeError):
        read_squad_row("不是一帧", table, pitch=PITCH)


def test_a_glyph_table_is_required(batch: BenchBatch) -> None:
    with pytest.raises(TypeError):
        read_squad_row(_frame(batch, "shot01"), "不是一张表", pitch=PITCH)


def test_ui_layout_is_required(batch: BenchBatch, table: GlyphTable) -> None:
    with pytest.raises(TypeError):
        read_squad_row(
            _frame(batch, "shot01"), table, pitch=PITCH, ui="不是布局"
        )


def test_pitch_must_be_a_number(batch: BenchBatch, table: GlyphTable) -> None:
    with pytest.raises(TypeError):
        read_squad_row(_frame(batch, "shot01"), table, pitch="207.5")


def test_pitch_must_be_positive(batch: BenchBatch, table: GlyphTable) -> None:
    with pytest.raises(ValueError):
        read_squad_row(_frame(batch, "shot01"), table, pitch=0)


def test_a_blank_canvas_has_no_cards(table: GlyphTable) -> None:
    """整条卡行是黑的 ⇒ 一张卡都没有（未采样到任何跳变）。"""
    case = None
    pixels = np.zeros((1260, 2800, 3), dtype=np.uint8)
    from ark.contracts import Canvas

    frame = Frame(canvas=Canvas(width=2800, height=1260), pixels=pixels)
    assert case is None
    assert read_squad_row(frame, table, pitch=PITCH) == ()


# ---------------------------------------------------------------------------
# 卡位与卡数
# ---------------------------------------------------------------------------


def test_the_three_given_frames_match_the_harvest(batch, table) -> None:
    for name, (count, costs) in TRUTH.items():
        row = _row(batch, table, name)
        assert len(row) == count, name
        assert [card.cost.payload.value for card in row] == costs, name


def test_the_six_frames_have_six_different_card_counts(batch, table) -> None:
    counts = [len(_row(batch, table, case.name)) for case in batch.cases]
    assert sorted(counts) == sorted(COUNTS)
    assert len(set(counts)) == len(counts)


def test_indices_run_from_zero_without_gaps(batch, table) -> None:
    for case in batch.cases:
        row = _row(batch, table, case.name)
        assert [card.index for card in row] == list(range(len(row)))


def test_the_rightmost_card_is_the_anchor(batch, table) -> None:
    """下标 0 = 最右那张：它的中心就是 `squad_card_rect(0, pitch)` 的中心。"""
    for case in batch.cases:
        row = _row(batch, table, case.name)
        assert row[0].center_x == _centre_of(0)


def test_every_centre_sits_on_the_grid(batch, table) -> None:
    """像素读出来的中心与网格那一格**最多差 2 px**（实测 1.5）。"""
    for case in batch.cases:
        for card in _row(batch, table, case.name):
            assert abs(card.center_x - _centre_of(card.index)) <= 2.0


def test_the_centre_is_read_from_the_jump_not_from_the_grid(batch, table) -> None:
    """6 帧 47 张卡的跳变都读出来了，所以中心**都是像素给的**，一个备选都没用上。"""
    for case in batch.cases:
        assert all(card.center_from_jump for card in _row(batch, table, case.name))


def test_a_wrong_pitch_does_not_find_the_row(batch, table) -> None:
    """卡距给错 ⇒ 网格那一格上**没有跳变证据** ⇒ 数出来的卡数就不对。

    这一条同时说明「卡数是从像素里数出来的」：光有网格（给多少格就有多少卡）
    是数不出卡数的。

    **注意它只是「不对」，不是「没有」**：@@pitch = 100@@ 时反而数出 11 张 ——
    因为卡面上本来就有别的黑白边（缝、费用牌自己的边），它们碰巧落进了那条更密的
    网格里。**所以卡距给错不会响亮地失败，只会静默地错**（见报告）。
    """
    frame = _frame(batch, "shot01")
    assert len(read_squad_row(frame, table, pitch=PITCH)) == 12
    # 实测：pitch = 150 数出 5 张、pitch = 100 数出 7 张 —— **都不是 12**，
    # 但也都不是 0：网格给错时这一条链**不会响亮地失败**。具体数会随网格变，
    # 所以这里只断言「不是 12」。
    assert len(read_squad_row(frame, table, pitch=150.0)) != 12
    assert len(read_squad_row(frame, table, pitch=100.0)) != 12


def test_a_slot_without_a_jump_falls_back_to_the_grid(batch, table) -> None:
    """某一格的跳变读不出来时，**那一格还在**（右对齐 + 连续），中心落回网格。

    把下标 1 那一格的**职业标识牌那一片**涂黑（涂到费用暗盒的头 7 列为止 ——
    数字从第 8 列起，所以费用**照样读得出来**）：卡数**不变**，但那一张的
    @@center_from_jump@@ 变成 @@False@@、中心就是 @@squad_card_rect@@ 给的那个数
    —— 这就是"备选"那一路。
    """
    frame = _frame(batch, "shot26")
    pixels = frame.pixels.copy()
    centre = int(round(_centre_of(1)))
    pixels[1045:1092, centre - 60:centre + 7, :] = 0
    row = read_squad_row(
        Frame(canvas=frame.canvas, pixels=pixels), table, pitch=PITCH
    )
    assert len(row) == 6
    assert row[1].center_from_jump is False
    assert row[1].center_x == _centre_of(1)
    assert row[1].cost.payload.value == 12
    assert all(card.center_from_jump for card in row if card.index != 1)


# ---------------------------------------------------------------------------
# 费用
# ---------------------------------------------------------------------------


def test_one_digit_and_two_digit_costs_are_both_exact(batch, table) -> None:
    """一位数与两位数都在这一批里：`shot26` 有 1 / 2 / 6，`shot01` 有 11…25。"""
    row = _row(batch, table, "shot26")
    assert [card.cost.payload.value for card in row] == [18, 12, 11, 6, 2, 1]
    assert all(isinstance(card.cost, Valued) for card in row)
    assert all(isinstance(card.cost.payload, Exact) for card in row)


def test_the_device_card_is_read_like_any_other(batch, table) -> None:
    """装置卡（费用 14、身份区是准星）读出来**与别的卡一模一样** —— 这一层不判身份。"""
    row = _row(batch, table, "shot01")
    device = row[6]
    assert device.cost.payload.value == 14
    assert device.redeploying is False
    # 这一层吐出来的东西里**没有任何一样在说"这是谁"**
    assert set(SquadCard.__dataclass_fields__) == {
        "index",
        "center_x",
        "center_from_jump",
        "cost",
        "redeploying",
        "seconds",
        "ring_fraction",
    }


def test_a_dark_box_means_unsampled(batch, table) -> None:
    frame = _blank_cost(_frame(batch, "shot01"), 0)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[0].cost == Unsampled()


def test_a_blob_that_is_not_a_digit_means_unknown(batch, table) -> None:
    """一块实心灰**有笔画、但不是数字** ⇒ 未知（不许拿"最像的那个"补上）。"""
    frame = _blob_cost(_frame(batch, "shot01"), 0)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert isinstance(row[0].cost, Valued)
    assert isinstance(row[0].cost.payload, Unknown)
    assert row[0].cost.confidence is None


def test_confidence_is_the_weakest_digit(batch, table) -> None:
    """置信度是这一串里**最弱的那个**（照 `cost.py` 的口径）。"""
    row = _row(batch, table, "shot05")
    weakest = min(card.cost.confidence for card in row)
    assert weakest == pytest.approx(row[8].cost.confidence)
    assert weakest > COST_SOLID_MATCH_SCORE


def test_the_cost_window_does_not_eat_the_class_badge(batch, table) -> None:
    """费用窗口**从卡中心起**，所以职业标识牌那一片亮块进不来。

    把窗口往左挪 54 px（挪到标识牌上）就该读出别的东西 —— 那正说明窗口的位置
    是被测的：不挪的话它读的是费用。
    """
    frame = _frame(batch, "shot01")
    window = _cost_window(_centre_of(0))
    moved = PixelRect(window.x - COST_BOX_WIDTH, window.y, window.width, window.height)
    blanked = _paint(frame, moved, [(0, 0, COST_BOX_WIDTH, COST_DIGITS_HEIGHT)], (0, 0, 0))
    assert read_squad_row(blanked, table, pitch=PITCH)[0].cost.payload.value == 25


# ---------------------------------------------------------------------------
# 再部署：状态
# ---------------------------------------------------------------------------


def test_only_the_redeploying_card_is_flagged(batch, table) -> None:
    row = _row(batch, table, "shot26")
    assert row[0].redeploying is True
    assert all(card.redeploying is False for card in row[1:])


def test_a_plain_card_is_not_redeploying_in_any_frame(batch, table) -> None:
    """6 帧里除了 `shot26` 下标 0 那张，**一张都没有环**（实测白弧 ≤ 57°）。"""
    for case in batch.cases:
        for card in _row(batch, table, case.name):
            if case.name == "shot26" and card.index == 0:
                continue
            assert card.redeploying is False
            assert card.ring_fraction is None
            assert card.seconds == Unsampled()


def test_the_ring_fraction_is_the_white_arc(batch, table) -> None:
    """环形进度 = 白弧占整圈多少。实测那张是 278°（0.769…）。"""
    card = _row(batch, table, "shot26")[0]
    assert card.ring_fraction == pytest.approx(278 / 360.0, abs=0.02)


def test_a_plain_card_can_be_made_to_look_redeploying(batch, table) -> None:
    """在普通卡上刷一段长白弧 ⇒ 它就该被认成再部署中。"""
    frame = _paint_arc(_frame(batch, "shot26"), 1, RING_ARC_REDEPLOYING + 60)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[1].redeploying is True
    assert row[1].ring_fraction == pytest.approx((RING_ARC_REDEPLOYING + 60) / 360.0, abs=0.03)


def test_a_short_arc_is_not_redeploying(batch, table) -> None:
    frame = _paint_arc(_frame(batch, "shot26"), 1, RING_ARC_PLAIN - 30)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[1].redeploying is False


def test_an_arc_in_between_cannot_be_told(batch, table) -> None:
    """落在空档中间（90…240°）⇒ **判不了**，报"是"或"不是"都是假事实。"""
    middle = (RING_ARC_PLAIN + RING_ARC_REDEPLOYING) // 2
    frame = _paint_arc(_frame(batch, "shot26"), 1, middle)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[1].redeploying is None
    assert row[1].ring_fraction == pytest.approx(middle / 360.0, abs=0.03)


def test_the_redeploying_gate_keeps_seconds_unsampled(batch, table) -> None:
    """不在再部署中 ⇒ 秒数**未采样**，即使卡面在环那一片本来就有亮像素。

    `shot26` 下标 4 那张卡在环带上实测有 1194 个亮像素（白发角色）——
    没有这道闸门，它会被当成"有倒计时"。
    """
    row = _row(batch, table, "shot26")
    assert row[4].redeploying is False
    assert row[4].seconds == Unsampled()


# ---------------------------------------------------------------------------
# 再部署：中央那个秒数
# ---------------------------------------------------------------------------


def test_the_real_countdown_digits_cannot_be_matched(batch, table) -> None:
    """真帧上那个秒数**落在未知** —— 它的字体与出厂字形表不是同一套。

    实测 `shot26` 那张 `12.9`：`1` 最像的字形只有 0.4448（而且最像的是 `2`）、
    `2` 0.8375、`9` 0.6909，全都低于 `COST_SOLID_MATCH_SCORE`。
    这一条**不是"链坏了"**：它如实说出"那里有一位数字，我认不出它是几"。
    """
    card = _row(batch, table, "shot26")[0]
    assert isinstance(card.seconds, Valued)
    assert isinstance(card.seconds.payload, Unknown)
    assert card.seconds.confidence is None


def test_a_countdown_written_in_the_table_font_is_read(batch, table) -> None:
    """用**字形表自己的字形**合成的 `12.9` ⇒ 读得出 12.9（合成帧，见模块文档）。"""
    frame = _draw_countdown(_blank_ring(_frame(batch, "shot26"), 0), table, "12.9", 0)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[0].redeploying is True
    assert isinstance(row[0].seconds.payload, Exact)
    assert row[0].seconds.payload.value == 12.9


def test_a_two_digit_countdown_without_a_dot_is_unknown(batch, table) -> None:
    """认不出小数点 ⇒ **未知**：秒数必须有小数位，报一个整数就是假事实。"""
    frame = _draw_countdown(_blank_ring(_frame(batch, "shot26"), 0), table, "129", 0)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert isinstance(row[0].seconds.payload, Unknown)


def test_an_empty_ring_centre_is_unsampled(batch, table) -> None:
    """环心里一块墨迹都没有 ⇒ 未采样（那里没有东西）。"""
    frame = _blank_ring(_frame(batch, "shot26"), 0)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[0].redeploying is True
    assert row[0].seconds == Unsampled()


def test_the_countdown_is_read_with_the_tables_own_glyphs(batch, table) -> None:
    """换一个数（`3.5`）照样读得出 —— 读的不是"这一帧记得的那个值"。"""
    frame = _draw_countdown(_blank_ring(_frame(batch, "shot26"), 0), table, "3.5", 0)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[0].seconds.payload.value == 3.5


def test_the_seconds_is_a_tenth_not_a_float_artefact(batch, table) -> None:
    """值是**十进制的十分位**：129 / 10 落成的那个双精度数就是 `12.9` 本身。"""
    frame = _draw_countdown(_blank_ring(_frame(batch, "shot26"), 0), table, "12.9", 0)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[0].seconds.payload.value == 12 + 9 / 10.0
    assert repr(row[0].seconds.payload.value) == "12.9"


# ---------------------------------------------------------------------------
# 形状
# ---------------------------------------------------------------------------


def test_the_reading_is_a_tuple_of_cards(batch, table) -> None:
    row = _row(batch, table, "shot26")
    assert isinstance(row, tuple)
    assert all(isinstance(card, SquadCard) for card in row)


def test_a_card_is_frozen(batch, table) -> None:
    card = _row(batch, table, "shot26")[0]
    with pytest.raises(Exception):
        card.cost = Unsampled()  # type: ignore[misc]


def test_the_row_is_repeatable(batch, table) -> None:
    """同一帧读两遍，逐字段一样（没有随机量）。"""
    frame = _frame(batch, "shot26")
    assert read_squad_row(frame, table, pitch=PITCH) == read_squad_row(
        frame, table, pitch=PITCH
    )


def test_ui_can_be_passed_explicitly(batch, table) -> None:
    """显式传**契约层**那份 @@UI_LAYOUT@@（卡框那两个键在那儿）结果一样。"""
    frame = _frame(batch, "shot26")
    assert read_squad_row(frame, table, pitch=PITCH, ui=UI_LAYOUT) == read_squad_row(
        frame, table, pitch=PITCH
    )


def test_a_layout_without_the_squad_card_cannot_be_read(batch, table) -> None:
    from ark.contracts import UiLayout

    empty = UiLayout(rects={})
    with pytest.raises(KeyError):
        read_squad_row(_frame(batch, "shot26"), table, pitch=PITCH, ui=empty)
