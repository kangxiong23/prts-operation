"""视觉层「备战席那一栏」读数链（T8）的单元测试。

这一批**不 import tests/spec**（D-全局-17），也不带规划者那套验收的检查编号 ——
这里测的是**这条链自己的形状与边界**：跳变那道判据认在哪一列、网格怎么抽验、
费用的三档怎么落、**再部署中的判据（环比周围亮多少）怎么分三值**、以及
「不在再部署中 ⇒ 秒数是未采样」这条闸门。

**判据是相对的，不是绝对的**（v2 改）：白弧量的是**已过比例**（所以把它压短不改判），
而"环带本身够不够亮"会被**亮卡面**顶穿 —— 实测**同一张卡面**在批里出现两次，
真环那次环带最暗 **110**、假环那次 **102**，绝对亮度只差 8 而真值反着。所以下面
既测那一对，也测**这条判据自己的边界**：往环带里刷一整圈亮环，它照样认成"有环"
（合成帧；真机素材上那个假事实是"亮卡面"，这一版就是为它改的）。

夹具是 `tests/帧样本/备战席/`（**第 2 版**基准集，**10 帧**）。`取材.json` 给了
**6 帧**的真值（`shot01` · `shot05` · `shot26` · `cd01` · `cd08` ·
`cd60`），另外 4 帧是**验收用的**、值不在这一层 —— 所以下面**值型的断言只落在
取材那 6 帧上**（`HARVEST`）。

**降级用例是在真帧上涂改出来的**，不是自己画的图：涂黑、涂成实心块、把白弧压短、
把环带调暗、往环带里刷一圈亮。

**圆环中央那个秒数用的是另一套字形**（那个 `1` 带底衬线、笔画整体更粗）：出厂那套表
读它，`1` 最像的字形反而是 `2`（0.4448）。这一单把**粗体那套资产**补上了
（`assets/features/glyphs_countdown/`），所以真帧上那 14 个秒数**逐位读出** ——
那是这一节的主判据。**「表没建好」与「这张卡没有倒计时」是两支**：前者抛错，后者
`Unsampled()`，下面两条各测一头。

合成帧那一路仍然留着（换一个数照样读得出 —— 读的不是"这一帧记得的那个值"），
但它现在用的是**粗体那套表**。
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from ark.contracts import (
    Exact,
    PixelRect,
    UI_LAYOUT,
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
    COUNTDOWN_SOLID_MATCH_SCORE,
    RING_BRIGHT_MIN,
    RING_CENTER_DY,
    RING_GAIN_ABSENT_MAX,
    RING_GAIN_PRESENT_MIN,
    RING_INNER,
    RING_OUTER,
    RING_PRESENT_OUTER,
    SquadCard,
    _ring_gain,
    _ring_profile,
    countdown_boxes,
    read_squad_row,
)
from ark.engine.vision.frame import Frame, frame_from_raw
from ark.engine.vision.glyphs import (
    COUNTDOWN_GLYPHS_PATH,
    DEFAULT_GLYPHS_PATH,
    GlyphTable,
    load_countdown_glyphs,
    normalize,
)
from ark.offline.bench.batch import BenchBatch, load_batch

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "帧样本" / "备战席"

PITCH = 207.5
"""卡距。**它住设备层的 `SQUAD_CARD_PITCH`**，导入规则不许视觉层 import 设备层，
所以用例里也照调用方的样子把它传进来。"""

GREY_DEGREE = 320.0
"""**灰环**那一段上的一个角度（0° = 12 点、顺时针）。

实测 `shot26` 下标 0 那张：白弧是 0…278°，灰环是 278…360°，所以 320° 稳稳落在灰环上。
"把白弧压成灰环"要取的就是**同一个半径上、这个角度上的真实像素** —— 不猜一个灰色。"""

TRUTH: dict[str, tuple[int, list[int]]] = {
    # 帧名: (卡数, 从右往左每张卡的费用) —— 这三帧来自取材.json（规划者目视确认）
    "shot01": (12, [25, 23, 23, 21, 18, 17, 14, 13, 13, 12, 12, 11]),
    "shot05": (9, [18, 17, 14, 14, 13, 13, 12, 11, 10]),
    "shot26": (6, [18, 12, 11, 6, 2, 1]),
}

HARVEST = ("shot01", "shot05", "shot26", "cd01", "cd08", "cd60")
"""`取材.json` 里**给了真值**的那 6 帧。**值型的断言只准落在它们身上** ——
另外 4 帧是验收用的（基准集 §取材与验收是分开的两批）。"""

CARDS: dict[str, int] = {
    "shot01": 12,
    "shot05": 9,
    "shot26": 6,
    "cd01": 9,
    "cd08": 9,
    "cd60": 11,
}
"""这 6 帧的**卡数**（取材.json）。

**它们不再是互不相同的六个数**（9 出现三次）—— 所以下面那条测的是"逐帧都对上取材"，
而不是"六个数互不相同"。卡数仍然是从像素里数出来的，只是这一批的六个值恰好重了。"""

RINGS: dict[str, tuple[int, ...]] = {
    "shot01": (),
    "shot05": (),
    "shot26": (0,),
    "cd01": (0, 1, 2, 5, 7, 8),
    "cd08": (0, 1, 2, 3, 4, 5, 7),
    "cd60": (),
}
"""这 6 帧上**在再部署中**的下标（取材.json）—— 一共 **14 张**真环。

两个关键样本：`cd08` 的 **5**（剩 16.3 s、白弧 277°）**被旧判据吃进了「判不了」**；
`cd60` 那一帧**一张都不在再部署中**，而它的下标 **2** 在旧判据下被判成「是」
（亮卡面顶穿了绝对亮度线）。"""


SECONDS: dict[str, dict[int, float]] = {
    "shot26": {0: 12.9},
    "cd01": {0: 32.2, 1: 39.5, 2: 54.7, 5: 36.0, 7: 64.2, 8: 16.0},
    "cd08": {0: 12.6, 1: 19.9, 2: 35.0, 3: 69.4, 4: 57.2, 5: 16.3, 7: 44.6},
    "cd60": {},
}
"""这 6 帧上**每张卡的秒数真值**（取材.json）—— 一共 **14 个数**，正好对上 14 张真环。

`cd60` 是空的：那一帧一张都不在再部署中，所以它那一栏**一张倒计时都没有** ——
这与「读不出来」是两支（见 `test_the_harvest_frames_countdowns_read_exactly`）。
"""


@pytest.fixture
def scratch() -> Iterator[Path]:
    """一个**自己能进去**的临时目录（照 `test_vision_cost.py` 那份）。

    **不用 pytest 的 `tmp_path`**：它建目录时用 0o700，而 0o700 的 DACL 不继承
    父目录 —— 在受限令牌下（本机 DSH 沙箱就是这样），那个目录连创建它的进程
    自己都进不去。这里用**默认权限**建，父目录的 ACL 会继承下来，所以在沙箱里
    与不在沙箱里都能跑。
    """
    root = Path(tempfile.gettempdir()) / "ark-vision-bench-tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex[:12]
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def batch() -> BenchBatch:
    return load_batch(FIXTURES)


@pytest.fixture(scope="module")
def table() -> GlyphTable:
    """**费用**那套出厂字形表（卡上那个暗盒里的数字）。"""
    return GlyphTable.load(DEFAULT_GLYPHS_PATH)


@pytest.fixture(scope="module")
def countdown() -> GlyphTable:
    """**倒计时**那套粗体字形表（环心那个秒数）—— 与费用那套并列的另一份资产。"""
    return load_countdown_glyphs()


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


def _press_arc(frame: Frame, index: int, *, keep: float) -> Frame:
    """把某一张卡的白弧**压短**到只剩 `keep` 度：超出的那一段换成**灰环**的颜色。

    环还在、红罩还在、中央那个数字也还在 —— 换掉的只是"已过"那一段的亮度。
    替换值取的是**同一个半径、`GREY_DEGREE` 那个角度上的真实像素**（实测是
    153/117/115 那一档），不是涂一个猜出来的灰。
    """
    centre_x = _centre_of(index)
    centre_y = _card_top() + RING_CENTER_DY
    pixels = frame.pixels.copy()
    rows, columns = np.mgrid[0:frame.canvas.height, 0:frame.canvas.width]
    dx = columns - centre_x
    dy = rows - centre_y
    radius = np.hypot(dx, dy)
    clock = (90.0 - np.degrees(np.arctan2(-dy, dx))) % 360.0
    theta = np.radians(90.0 - GREY_DEGREE)
    radii = np.arange(RING_INNER, RING_PRESENT_OUTER + 1)
    source = frame.pixels[
        np.round(centre_y - radii * np.sin(theta)).astype(int),
        np.round(centre_x + radii * np.cos(theta)).astype(int),
    ]
    step = (
        np.clip(np.round(radius).astype(int), RING_INNER, RING_PRESENT_OUTER)
        - RING_INNER
    )
    band = (
        (radius >= RING_INNER - 1)
        & (radius <= RING_PRESENT_OUTER + 1)
        & (clock > keep)
    )
    press = band & (pixels.min(axis=2) >= RING_BRIGHT_MIN)
    pixels[press] = source[step[press]]
    return Frame(canvas=frame.canvas, pixels=pixels)


def _dim_ring(frame: Frame, index: int, factor: float) -> Frame:
    """把某一张卡的**环带**（半径 `RING_INNER`…`RING_PRESENT_OUTER`）整圈调暗。

    用来造"有一圈东西、但不够亮"那一档 —— 两条线中间那一段真帧里没有样本。
    """
    centre_x = _centre_of(index)
    centre_y = _card_top() + RING_CENTER_DY
    pixels = frame.pixels.copy()
    rows, columns = np.mgrid[0:frame.canvas.height, 0:frame.canvas.width]
    radius = np.hypot(columns - centre_x, rows - centre_y)
    band = (radius >= RING_INNER - 1) & (radius <= RING_PRESENT_OUTER + 1)
    pixels[band] = (pixels[band].astype(np.float64) * factor).astype(np.uint8)
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


def test_a_ui_argument_is_not_accepted(batch: BenchBatch, table: GlyphTable) -> None:
    """这一层**不收 `ui`**：卡框只有一份表（契约层 `UI_LAYOUT`），
    `squad_card_rect` 也写死读它 —— 只能换掉锚点、换不掉位移的参数是个陷阱。"""
    with pytest.raises(TypeError):
        read_squad_row(
            _frame(batch, "shot01"), table, pitch=PITCH, ui=UI_LAYOUT
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


def test_the_harvest_frames_have_their_harvest_card_counts(batch, table) -> None:
    """取材那 6 帧的卡数逐帧对上（12 · 9 · 6 · 9 · 9 · 11）。

    **卡数是数出来的，不是参数** —— 这一条只说明"逐帧都对上取材"，不说明"六个值互不
    相同"：这一批里 9 出现了三次。
    """
    for name, want in CARDS.items():
        assert len(_row(batch, table, name)) == want, name


def test_the_batch_is_version_two_with_ten_frames(batch: BenchBatch) -> None:
    """这一批是 **v2 / 10 帧**（6 帧扫描 + 4 帧连拍）。

    验收那 4 帧的**名字与值都不在这一层** —— 下面值型的断言只落在 `HARVEST` 上。
    """
    assert batch.version == 2
    assert len(batch.cases) == 10
    assert set(HARVEST) <= {case.name for case in batch.cases}


def test_indices_run_from_zero_without_gaps(batch, table) -> None:
    for name in HARVEST:
        row = _row(batch, table, name)
        assert [card.index for card in row] == list(range(len(row)))


def test_the_rightmost_card_is_the_anchor(batch, table) -> None:
    """下标 0 = 最右那张：它的中心就是 `squad_card_rect(0, pitch)` 的中心。"""
    for name in HARVEST:
        row = _row(batch, table, name)
        assert row[0].center_x == _centre_of(0)


def test_every_centre_sits_on_the_grid(batch, table) -> None:
    """读出来的中心与网格那一格**最多差 2 px**（实测就是 2.0）。"""
    for name in HARVEST:
        for card in _row(batch, table, name):
            assert abs(card.center_x - _centre_of(card.index)) <= 2.0


def test_a_card_whose_badge_is_washed_out_falls_back_to_the_grid(batch, table) -> None:
    """**顶那一条本身亮的卡靠网格兜**：它的跳变读不出来，中心落回网格。

    实测 `cd01` 与 `cd08` 的**下标 6** 都是这种卡（职业标识牌那一片被亮立绘顶穿，
    跳变那条判据找不到它）—— 卡还在那一格里，只是"谁给的"变成了网格。
    这一条**不假设"全都由像素给出"**：那种卡就是有，而且两路给的中心差 ≤ 2 px。
    """
    for name in ("cd01", "cd08"):
        row = _row(batch, table, name)
        assert row[6].center_from_jump is False
        assert row[6].center_x == _centre_of(6)
        assert all(card.center_from_jump for card in row if card.index != 6)


def test_a_wrong_pitch_does_not_find_the_row(batch, table) -> None:
    """卡距给错 ⇒ 网格那一格上**没有跳变证据** ⇒ 数出来的卡数就不对。

    这一条同时说明「卡数是从像素里数出来的」：光有网格（给多少格就有多少卡）
    是数不出卡数的。

    **注意它只是「不对」，不是「没有」**：`pitch = 100` 时反而数出 7 张 ——
    因为卡面上本来就有别的黑白边（缝、费用牌自己的边），它们碰巧落进了那条更密的
    网格里。**所以卡距给错不会响亮地失败，只会静默地错**（见报告）。
    """
    frame = _frame(batch, "shot01")
    assert len(read_squad_row(frame, table, pitch=PITCH)) == 12
    # 实测（shot01，真值 12）：pitch = 180 → 1 张、150 → 5 张、130 → **17 张**、
    # 100 → 7 张。**都不是 12**，但也都不是 0 —— 网格给错时这一条链既会少数也会多数，
    # **就是不会响亮地失败**。
    assert len(read_squad_row(frame, table, pitch=180.0)) == 1
    assert len(read_squad_row(frame, table, pitch=150.0)) == 5
    assert len(read_squad_row(frame, table, pitch=130.0)) == 17
    assert len(read_squad_row(frame, table, pitch=100.0)) == 7


def test_a_slot_without_a_jump_falls_back_to_the_grid(batch, table) -> None:
    """某一格的跳变读不出来时，**那一格还在**（右对齐 + 连续），中心落回网格。

    把下标 1 那一格的**职业标识牌那一片**涂黑（涂到费用暗盒的头 7 列为止 ——
    数字从第 8 列起，所以费用**照样读得出来**）：卡数**不变**，但那一张的
    `center_from_jump` 变成 `False`、中心就是 `squad_card_rect` 给的那个数
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


def test_the_harvest_frames_match_the_redeploy_truth(batch, table) -> None:
    """**这一列的主判据**：取材那 6 帧的再部署状态**逐张对上**（14 张真环 · 42 张普通卡）。

    它同时说明两件旧判据做不到的事：`cd08` 的 **5**（白弧 277°）不再掉进"判不了"，
    `cd60` 的 **2**（亮卡面）不再被判成"是"。
    """
    for name in HARVEST:
        row = _row(batch, table, name)
        got = tuple(card.index for card in row if card.redeploying is True)
        assert got == RINGS[name], name
        assert all(card.redeploying is not None for card in row), name
        for card in row:
            if card.redeploying is False:
                assert card.ring_fraction is None
                assert card.seconds == Unsampled()


def test_a_bright_card_face_is_not_a_ring(batch, table) -> None:
    """**这一版要修的假事实**：亮卡面（白发 / 亮底）不是环。

    `cd01` 下标 2 与 `cd60` 下标 2 是**同一个干员、同一个费用**，一次在再部署中、
    一次不在 —— 最紧的一对证据。实测两者的**环带最暗那一度**只差 8（110 对 102），
    而**环比环外亮多少**是 80 对 −58：任何画在绝对亮度上的线都会把其中一个说错。
    """
    assert _row(batch, table, "cd60")[2].redeploying is False
    assert _row(batch, table, "cd01")[2].redeploying is True


def test_the_gain_separates_true_rings_from_false_ones(batch, table) -> None:
    """**判据自己那个数**：真环那 14 张的增益 **70…86**，其余 42 张 **−154…−12**。

    两条线（`RING_GAIN_PRESENT_MIN` / `RING_GAIN_ABSENT_MAX`）就落在这条空档里 ——
    这一条把空档钉进测试，免得以后有人把线挪进样本里还不知道。
    """
    real: list[int] = []
    fake: list[int] = []
    for name in HARVEST:
        frame = _frame(batch, name)
        for card in _row(batch, table, name):
            gain = _ring_gain(_ring_profile(frame, card.center_x, _card_top()))
            (real if card.index in RINGS[name] else fake).append(gain)
    assert len(real) == 14 and len(fake) == 42
    assert min(real) == 70 and max(real) == 86
    assert min(fake) == -154 and max(fake) == -12
    assert max(fake) <= RING_GAIN_ABSENT_MAX < RING_GAIN_PRESENT_MIN <= min(real)


def test_the_ring_fraction_is_the_elapsed_fraction(batch, table) -> None:
    """`ring_fraction` = 白弧占整圈多少 = **已过比例**。实测那张是 278°（0.769…）。"""
    card = _row(batch, table, "shot26")[0]
    assert card.ring_fraction == pytest.approx(278 / 360.0, abs=0.02)


def test_a_pressed_arc_still_counts_as_redeploying(batch, table) -> None:
    """把再部署那张的白弧从 278° 压到约 60°（**环还在、红罩还在、中央还是 12.9**）
    ⇒ 它**必须**判成再部署中。

    判据问的是环这道笔画在不在，不是白弧多长：压短白弧只把"已过比例"改小，环带本身
    照样比环外亮。压过之后费用与中央那个数字都没动，正好一并说明**改的只是那段白弧**。
    """
    frame = _press_arc(_frame(batch, "shot26"), 0, keep=60.0)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[0].redeploying is True
    assert row[0].cost.payload.value == 18
    assert isinstance(row[0].seconds.payload, Exact)
    assert row[0].seconds.payload.value == 12.9
    assert row[0].ring_fraction is not None and row[0].ring_fraction < 0.25
    assert all(card.redeploying is False for card in row[1:])


def test_a_painted_bright_ring_still_reads_as_redeploying(batch, table) -> None:
    """**这条判据的边界**：往环带里刷一整圈亮环 ⇒ 照样认成"有环"。

    判据问的是"环比它周围亮多少" —— 往环带里画一道亮笔画，那就是一道笔画。
    这一条是**合成帧**上的边界；真机素材上那个假事实是"**亮卡面**"（卡面整片亮、
    并没有环），它由 `test_a_bright_card_face_is_not_a_ring` 管住 —— 这一版就是为它改的。
    """
    frame = _paint_arc(_frame(batch, "shot26"), 1, 360)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[1].redeploying is True


def test_a_ring_that_is_too_dim_cannot_be_told(batch, table) -> None:
    """环比周围只亮一点点 ⇒ **判不了**，报"是"或"不是"都是假事实。

    把环带整圈调到 0.5 倍（环还在、红罩还在）：增益落到 `RING_GAIN_ABSENT_MAX`…
    `RING_GAIN_PRESENT_MIN` 那一段。
    """
    frame = _dim_ring(_frame(batch, "shot26"), 0, 0.5)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[0].redeploying is None
    assert row[0].ring_fraction is not None


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


def test_the_harvest_frames_countdowns_read_exactly(batch, table) -> None:
    """**这一列的主判据**：取材帧上那 **14 个秒数逐位读出、精确等于真值**。

    落的是 `Exact(12.9)` 这种载荷 —— **不是"读得出大概"**，小数位也在里面。
    不在再部署中的那些卡一律 `Unsampled()`（那里没有倒计时可读），`cd60` 整帧如此 ——
    **「没有」与「读不出来」是两支**，那 14 个数一个都不许落进后者。
    """
    read = 0
    for name in HARVEST:
        for card in _row(batch, table, name):
            want = SECONDS.get(name, {}).get(card.index)
            if want is None:
                assert card.seconds == Unsampled(), (name, card.index)
                continue
            assert isinstance(card.seconds, Valued), (name, card.index)
            assert isinstance(card.seconds.payload, Exact), (name, card.index)
            assert card.seconds.payload.value == want, (name, card.index)
            read += 1
    assert read == 14


def test_the_cost_table_cannot_read_the_bold_digits(batch, table, countdown) -> None:
    """**为什么要有第二张表**：出厂那套字形读不出这串粗体数字 —— 同一块墨迹，两张表反着答。

    实测 `shot26` 那张 `12.9` 的 `1`：出厂表最像的**反而是 `2`**（0.4448，低于它
    自己的 0.86 线），粗体表最像的正是 `1`（0.9964，过它自己的 0.90 线）。所以这不是
    "链坏了"，是**欠了一套资产** —— 这一单把它补上了。
    """
    frame = _frame(batch, "shot26")
    card = read_squad_row(frame, table, pitch=PITCH)[0]
    ink, _, digits, dots = countdown_boxes(frame, card.center_x, _card_top())
    whole = [box for box in digits if box[2] <= dots[0][0]]
    x0, y0, x1, y1 = whole[0]
    block = normalize(ink[y0:y1, x0:x1])

    cost_scores = table.scores(block)
    bold_scores = countdown.scores(block)
    assert max(cost_scores.values()) == pytest.approx(0.4448, abs=0.0002)
    assert max(cost_scores, key=lambda digit: cost_scores[digit]) == 2
    assert max(cost_scores.values()) < COST_SOLID_MATCH_SCORE
    assert bold_scores[1] == pytest.approx(0.9964, abs=0.0002)
    assert max(bold_scores, key=lambda digit: bold_scores[digit]) == 1
    assert bold_scores[1] >= COUNTDOWN_SOLID_MATCH_SCORE


def test_a_countdown_drawn_with_the_bold_glyphs_is_read(batch, table, countdown) -> None:
    """用**粗体那套表自己的字形**合成的 `12.9` ⇒ 读得出 12.9（合成帧，见模块文档）。"""
    frame = _draw_countdown(
        _blank_ring(_frame(batch, "shot26"), 0), countdown, "12.9", 0
    )
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[0].redeploying is True
    assert isinstance(row[0].seconds.payload, Exact)
    assert row[0].seconds.payload.value == 12.9


def test_a_two_digit_countdown_without_a_dot_is_unknown(batch, table, countdown) -> None:
    """认不出小数点 ⇒ **未知**：秒数必须有小数位，报一个整数就是假事实。

    画的是**粗体那套表自己的字形**（所以字形认得出来），坏掉的只有小数点这一件事。
    """
    frame = _draw_countdown(
        _blank_ring(_frame(batch, "shot26"), 0), countdown, "129", 0
    )
    row = read_squad_row(frame, table, pitch=PITCH)
    assert isinstance(row[0].seconds.payload, Unknown)


def test_an_empty_ring_centre_is_unsampled(batch, table) -> None:
    """环心里一块墨迹都没有 ⇒ 未采样（那里没有东西）。"""
    frame = _blank_ring(_frame(batch, "shot26"), 0)
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[0].redeploying is True
    assert row[0].seconds == Unsampled()


def test_the_countdown_is_read_with_the_tables_own_glyphs(batch, table, countdown) -> None:
    """换一个数（`3.5`）照样读得出 —— 读的不是"这一帧记得的那个值"。"""
    frame = _draw_countdown(
        _blank_ring(_frame(batch, "shot26"), 0), countdown, "3.5", 0
    )
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[0].seconds.payload.value == 3.5


def test_the_seconds_is_a_tenth_not_a_float_artefact(batch, table, countdown) -> None:
    """值是**十进制的十分位**：129 / 10 落成的那个双精度数就是 `12.9` 本身。"""
    frame = _draw_countdown(
        _blank_ring(_frame(batch, "shot26"), 0), countdown, "12.9", 0
    )
    row = read_squad_row(frame, table, pitch=PITCH)
    assert row[0].seconds.payload.value == 12 + 9 / 10.0
    assert repr(row[0].seconds.payload.value) == "12.9"


# ---------------------------------------------------------------------------
# 资产：粗体那套表
# ---------------------------------------------------------------------------


def test_the_shipped_countdown_table_covers_all_ten_digits() -> None:
    """出厂那份粗体表**十个数字一个不缺**，落点与文件名照规格钉的来。

    「表里少一个键，那个数字就永远读不出来，而表看起来仍然是好的」—— 所以这一条与
    `build_countdown_glyphs` 里那句报错是一对。
    """
    table = load_countdown_glyphs()
    assert table.digits == tuple(range(10))
    assert COUNTDOWN_GLYPHS_PATH.name == "digits.npz"
    assert COUNTDOWN_GLYPHS_PATH.parent.name == "glyphs_countdown"
    assert COUNTDOWN_GLYPHS_PATH.parent.parent.name == "features"


def test_the_countdown_table_is_reproducible_from_the_harvest(scratch: Path) -> None:
    """同一批取材卡跑两遍，**逐字节一样**；而且出厂那份就是这条路造出来的。

    字节比的是两次构建（与压缩实现无关）；与出厂那份比的是**模板逐像素**（不让
    `savez_compressed` 换个实现就把这一条变成假的）。
    """
    from ark.offline.assets_gen.countdown import (
        build_countdown_glyphs,
        load_countdown_harvest,
    )

    harvest = load_countdown_harvest()
    first = build_countdown_glyphs(harvest, scratch / "a")
    second = build_countdown_glyphs(harvest, scratch / "b")
    assert first.read_bytes() == second.read_bytes()

    fresh = GlyphTable.load(first)
    shipped = load_countdown_glyphs()
    assert fresh.digits == shipped.digits
    for digit in shipped.digits:
        assert np.array_equal(fresh.templates[digit], shipped.templates[digit])


def test_a_missing_countdown_table_is_not_a_missing_countdown(
    batch, table, scratch: Path
) -> None:
    """**能力缺失 ≠ 这张卡没有倒计时**：路径指到不存在的地方 ⇒ **抛错**。

    静默返回一行「未采样」的读数，吐出来的是**假缺席** —— 与假事实一样坏
    （《识别方法》§已定）。报什么错不管，**报就行**。
    """
    with pytest.raises(FileNotFoundError) as bad:
        read_squad_row(
            _frame(batch, "shot26"),
            table,
            pitch=PITCH,
            countdown_path=scratch / "还没有这张表",
        )
    assert "倒计时" in str(bad.value)


def test_an_empty_countdown_directory_is_also_loud(batch, table, scratch: Path) -> None:
    """目录在、表不在（空目录）⇒ 同样抛错 —— 那也是「表没建好」。"""
    empty = scratch / "空的"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        read_squad_row(
            _frame(batch, "shot26"), table, pitch=PITCH, countdown_path=empty
        )


def test_the_caller_may_point_at_the_shipped_table_explicitly(batch, table) -> None:
    """`countdown_path` 默认就是出厂那份 —— 显式传同一个路径，读数**逐字段一样**。"""
    frame = _frame(batch, "shot26")
    assert read_squad_row(
        frame, table, pitch=PITCH, countdown_path=COUNTDOWN_GLYPHS_PATH
    ) == read_squad_row(frame, table, pitch=PITCH)


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
