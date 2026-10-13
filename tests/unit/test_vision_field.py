"""场上我方干员那三条读数链的单元测试。

**不 import tests/spec**（D-全局-17），也不带验收的检查编号 —— 这里测的是**这三条链
自己的形状与边界**：取区怎么折、判据用哪个统计量、阈值落在哪段空档里、以及**每一档
降级各落什么形态**。

## 这一份盯得最紧的是两件事

1. **「看不到」与「认不出」分得开。** 每一条链都把两支各钉一次：
   - 血条 -> 找不到条落 `Unsampled`（**绝不是"满血"**）；边界渐变 / 被切断落区间 / 下界；
   - 技力条 -> 找不到条落未采样；含义那两条线之间落 `Unknown`；
   - 菱形 -> 中间那一段落 `Unsampled`（**绝不是 `Exact(False)`** —— 那是漏检方向）。
2. **每一条链只产它那个字段「允许的形态」里的东西。** 血量那个字段没有未知，所以
   血条那条链任何一支都不许给 `Unknown`；可手动触发只有布尔 / 未采样，所以
   菱形那条链任何一支都不许给区间。

## 合成用例是在**真帧之外**造的

真帧里**没有**「极低填充」「血条被切断」「菱形分数落在两条线之间」这三档
（它们如实写在 `tests/帧样本/场上干员/README.md` 的「没覆盖到什么」里）。所以
那三支用合成的东西钉：

- 「极低填充」与「被切断」：拿真帧里那根条**抹掉几列**——判据的形状与"哪一列被抹"
  无关，所以抹出来的那一支是真的走了一遍那条分支；
- 「菱形落在两条线之间」：把 `diamond_score` 换成一个固定的中间分数（它**就是**
  判据本身，换它等于把"读数恰好落在空档里"这件事直接摆进去）。

## 重活只跑一次

一帧解出来是 14 MB，16 条。所以**整批只跑一遍、装在模块级夹具里**，用例只读结果。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ark.contracts import (
    Exact,
    Interval,
    LowerBound,
    PixelRect,
    SlotValue,
    Unknown,
    Unsampled,
    Valued,
)
from ark.engine.vision import field
from ark.engine.vision.frame import CANVAS_2800x1260, Frame, frame_from_raw
from ark.offline.bench.batch import load_batch
from ark.offline.bench.feed import replay_frames

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "tests" / "帧样本" / "场上干员"
HARVEST = BATCH / "取材.json"
REPORT = BATCH / "判据读数.json"


@pytest.fixture(scope="module")
def batch():
    return load_batch(BATCH)


@pytest.fixture(scope="module")
def harvest() -> dict:
    return json.loads(HARVEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readings(batch) -> dict[str, dict]:
    """整批跑一遍：帧名 -> 三条链的槽 + 菱形的分数。**只跑一次。**"""
    windows = {
        item["file"]: item["window"]
        for item in json.loads(HARVEST.read_text(encoding="utf-8"))["crops"]
    }
    template = field.load_ready_diamond()
    out: dict[str, dict] = {}
    for case, raw in replay_frames(batch):
        frame = frame_from_raw(raw)
        window = PixelRect(*windows[case.name + ".png"])
        fill, meaning = field.read_skill(frame, window)
        score, _, _ = field.diamond_score(frame, window, template)
        out[case.name] = {
            "frame": frame,
            "window": window,
            "health": field.read_health(frame, window),
            "fill": fill,
            "meaning": meaning,
            "sp": field.sp_of(fill, meaning),
            "diamond": field.find_ready_diamond(frame, window, template),
            "score": score,
        }
    return out


# ---------------------------------------------------------------------------
# 取区
# ---------------------------------------------------------------------------


def test_window_folds_from_the_bar_corner() -> None:
    """取区是从**血条左上角**折出来的：原点 = 血条左上角 − BARS_IN_WINDOW。"""
    window = field.operator_ui_window(PixelRect(1042, 629, 146, 16))
    assert (window.x, window.y) == (1042 - field.BARS_IN_WINDOW[0], 629 - field.BARS_IN_WINDOW[1])
    assert (window.width, window.height) == (field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT)


def test_window_diamond_centre_sits_above_the_bars() -> None:
    """菱形中心在血条左上角 **+(72, −233)** 处 —— 换算到取区就是 DIAMOND_IN_WINDOW。"""
    col, row = field.BARS_IN_WINDOW
    diamond_col, diamond_row = field.DIAMOND_IN_WINDOW
    assert diamond_col - col == 72
    assert row - diamond_row == 233


def test_window_outside_canvas_is_refused(batch) -> None:
    """取区越出画布 -> **抛异常**（能力缺失），不是静默裁到边上。"""
    case = batch.cases[0]
    frame = frame_from_raw(case.raw())
    outside = PixelRect(CANVAS_2800x1260.width - 10, 100, 240, 360)
    with pytest.raises(ValueError):
        field.read_health(frame, outside)


def test_window_smaller_than_the_ui_is_refused(batch) -> None:
    """比 240×360 还小的取区 -> 抛异常（它装不下菱形，而少了哪一样都是静默的）。"""
    case = batch.cases[0]
    frame = frame_from_raw(case.raw())
    with pytest.raises(ValueError):
        field.read_health(frame, PixelRect(100, 100, 100, 100))


# ---------------------------------------------------------------------------
# 血条
# ---------------------------------------------------------------------------


def test_health_reads_the_measured_fills(readings) -> None:
    """血条：满的读满，低的读低 —— 真帧上量出来的那几个数。"""
    full = readings["cd01_a"]["health"]
    assert isinstance(full, Valued) and full.payload == Exact(1.0) and full.confidence is not None
    assert readings["cd01_c"]["health"].payload.value == pytest.approx(129 / 146)
    assert readings["cd12_a"]["health"].payload.value == pytest.approx(108 / 146)
    assert readings["shot02_a"]["health"].payload.value == pytest.approx(42 / 146, abs=2 / 146)
    assert readings["field_b"]["health"].payload.value == pytest.approx(114 / 146, abs=2 / 146)


def test_health_never_reads_full_when_the_bar_is_gone(batch, readings) -> None:
    """**看不到 ≠ 满血**：把取区挪到一块没有条的地面上，血条必须落**未采样**。

    这是《状态层/我方干员》§血量 那条硬规则在这一层的样子 —— 我方血条常驻，
    所以"看不到"只能是识别失败；读成满血就是假事实。
    """
    case = batch.cases[0]
    frame = frame_from_raw(case.raw())
    empty = PixelRect(100, 100, field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT)
    assert field.read_health(frame, empty) == Unsampled()


def test_health_forms_are_only_the_fields_forms(readings) -> None:
    """血量那个字段的形态是 精确 / 区间 / 下界 / 未采样 —— **一条链都不许给未知**。"""
    for row in readings.values():
        slot = row["health"]
        if isinstance(slot, Valued):
            assert not isinstance(slot.payload, Unknown)


def test_health_lower_bound_when_an_effect_covers_the_bar(readings) -> None:
    """被**别的条**压住 -> **下界**。真帧上那一处是 shot04_a（那条血条上压着另一根条，
    填着的那一截后面还有一小段蓝）。

    **它是单边的那一侧**：被压住只会让比例读低，所以"至少这么多"是诚实的读数
    （《识别方法》「亮区比例」那行的 ⚠）。
    """
    slot = readings["shot04_a"]["health"]
    assert isinstance(slot, Valued) and isinstance(slot.payload, LowerBound)
    assert 0 < slot.payload.lo < 0.35


def test_health_interval_when_the_boundary_is_soft(batch, harvest) -> None:
    """边界渐变（右边那一列半亮）-> **区间**，不是精确值、也不是下界。

    真帧里没有这一档（见 README 的「没覆盖到什么」），所以拿真帧那根条把边界那几列
    **压成半亮**造一个：它走的是**同一条分支**。
    """
    entry = next(item for item in harvest["crops"] if item["file"] == "cd01_c.png")
    case = next(c for c in load_batch(BATCH).cases if c.name == "cd01_c")
    frame = frame_from_raw(case.raw())
    window = PixelRect(*entry["window"])
    pixels = frame.pixels.copy()
    bar_x = int(window.x) + field.BARS_IN_WINDOW[0]
    bar_y = int(window.y) + field.BARS_IN_WINDOW[1]
    half = tuple(min(255, channel + 40) for channel in field.HP_BLUE)
    pixels[bar_y : bar_y + field.BAR_HEIGHT, bar_x + 129 : bar_x + 132] = half
    soft = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    slot = field.read_health(soft, window)
    assert isinstance(slot, Valued) and isinstance(slot.payload, Interval)
    assert slot.payload.lo == pytest.approx(129 / 146)
    assert slot.payload.hi > slot.payload.lo


def test_health_lower_bound_when_the_fill_is_cut(batch, harvest) -> None:
    """**填着的那一截被切断**（中间被压住、后面又亮回来）-> **下界**。

    真帧里没有这一档（见 README 的「没覆盖到什么」），所以拿真帧那根条**抹掉中间几列**
    造一个：抹出来的这一支走的是**同一条分支**。
    """
    entry = next(item for item in harvest["crops"] if item["file"] == "cd01_c.png")
    case = next(c for c in load_batch(BATCH).cases if c.name == "cd01_c")
    frame = frame_from_raw(case.raw())
    window = PixelRect(*entry["window"])
    pixels = frame.pixels.copy()
    bar_x = int(window.x) + field.BARS_IN_WINDOW[0]
    bar_y = int(window.y) + field.BARS_IN_WINDOW[1]
    pixels[bar_y : bar_y + field.BAR_HEIGHT, bar_x + 40 : bar_x + 46] = 0
    cut = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    slot = field.read_health(cut, window)
    assert isinstance(slot, Valued) and isinstance(slot.payload, LowerBound)
    assert slot.payload.lo == pytest.approx(40 / 146, abs=2 / 146)


def test_health_unsampled_when_the_fill_is_too_short(batch, harvest) -> None:
    """极低填充（少于 MIN_FILL_COLUMNS 列）-> **未采样**（保守方向：绝不给一个编的数）。"""
    entry = next(item for item in harvest["crops"] if item["file"] == "cd01_c.png")
    case = next(c for c in load_batch(BATCH).cases if c.name == "cd01_c")
    frame = frame_from_raw(case.raw())
    window = PixelRect(*entry["window"])
    pixels = frame.pixels.copy()
    bar_x = int(window.x) + field.BARS_IN_WINDOW[0]
    bar_y = int(window.y) + field.BARS_IN_WINDOW[1]
    pixels[bar_y : bar_y + field.BAR_HEIGHT, bar_x : bar_x + field.BAR_WIDTH] = 0
    pixels[bar_y : bar_y + field.BAR_HEIGHT, bar_x : bar_x + 2] = (110, 190, 228)
    starved = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    assert field.read_health(starved, window) == Unsampled()


# ---------------------------------------------------------------------------
# 技力条
# ---------------------------------------------------------------------------


def test_skill_meanings_are_all_three_kinds(readings) -> None:
    """含义三档在真帧上都出现过：充能中 / 生效中 / 弹药分格。"""
    meanings = {
        name: row["meaning"].payload.value
        for name, row in readings.items()
        if isinstance(row["meaning"], Valued) and isinstance(row["meaning"].payload, Exact)
    }
    assert meanings["cd01_a"] == "充能中"
    assert meanings["cd01_b"] == "生效中"
    assert meanings["cd01_c"] == "弹药分格"
    assert meanings["shot26_a"] == "弹药分格"


def test_skill_fill_is_a_lower_bound_for_ammo_that_is_not_full(readings) -> None:
    """**弹药分界模糊 -> 下界**：分格条没填满时只给下界，不给精确值。"""
    slot = readings["shot26_a"]["fill"]
    assert isinstance(slot, Valued) and isinstance(slot.payload, LowerBound)
    assert 0 < slot.payload.lo < 1


def test_skill_fill_crosses_the_printed_numbers(readings) -> None:
    """**条上压着别的条 -> 跨过去接着算**，不是下界。

    shot04_a：那条缝是 **5 / 7 / 6 / 6 / 8 / 6 / 8** 列，最宽 8 ≤
    `COVER_GAP_TOLERANCE` —— 跨过去之后最后一列还是填充色 → 满条。

    （上一版这一条落的是下界 `0.192`，比实际的弱得多；再上一版是精确 1.0，但那是
    把数字按"亮"算**成了填充**。三个版本同一个数、三条完全不同的路。）
    """
    slot = readings["shot04_a"]["fill"]
    assert slot == Valued(Exact(1.0), None)
    assert readings["shot04_a"]["meaning"].payload == Exact("充能中")


def test_skill_fill_reads_a_partial_bar(readings) -> None:
    """技力条没满时读得出那个比例（真帧上 0.863 / 0.808 / 0.527 那三条）。

    shot31_a 那条**边界是干净的一步**（右边一列就直接跳到轨道那一档，跳过去之后
    再没有一列是填充色）→ **精确值**。它的旧读数是**下界** —— 同一条支路，旧的那把
    尺子量的是亮度，把边界那一列算成了"半亮"。
    """
    exact = readings["shot21_a"]["fill"]
    assert isinstance(exact, Valued) and isinstance(exact.payload, Exact)
    assert exact.payload.value == pytest.approx(126 / 146)
    cut = readings["shot31_a"]["fill"]
    assert isinstance(cut, Valued) and isinstance(cut.payload, Exact)
    assert cut.payload.value == pytest.approx(77 / 146)


def test_skill_unsampled_when_there_is_no_bar(batch) -> None:
    """看不到技力条 -> 两个量都**未采样**。"""
    case = next(c for c in load_batch(BATCH).cases if c.name == "cd01_a")
    frame = frame_from_raw(case.raw())
    empty = PixelRect(100, 100, field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT)
    fill, meaning = field.read_skill(frame, empty)
    assert fill == Unsampled() and meaning == Unsampled()


def test_skill_forms_are_only_the_fields_forms(readings) -> None:
    """技力那个字段的形态只有 精确 / 区间 / 未采样 —— 读数那一支不许给未知。"""
    for row in readings.values():
        if isinstance(row["fill"], Valued):
            assert not isinstance(row["fill"].payload, Unknown)
        if isinstance(row["sp"], Valued):
            assert isinstance(row["sp"].payload, (Exact, Interval)), row["sp"]


# ---------------------------------------------------------------------------
# 技力填充那一把尺子：**颜色**，不是亮度
# ---------------------------------------------------------------------------


def _synthetic_bar(
    filled: int,
    fill: tuple[int, int, int],
    track: tuple[int, int, int],
    background: tuple[int, int, int],
) -> tuple[Frame, PixelRect]:
    """造一帧：一张**亮底** + 一根血条 + 一根技力条（自左起 `filled` 列是填充色）。

    技力那一行在填充右边**不是暗的**，而是亮底透出来的那一截 —— 这正是「亮底上整条
    判成填满」那个现象的形状（真帧在 `data/probe/sp_series`：橙条纹地板 / 青地板 /
    花丛，取区 = 血条左上角 − BARS_IN_WINDOW）。
    """
    pixels = np.full(
        (CANVAS_2800x1260.height, CANVAS_2800x1260.width, 3), background, dtype=np.uint8
    )
    window = PixelRect(0, 0, field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT)
    bar_x, bar_y = field.BARS_IN_WINDOW
    pixels[bar_y : bar_y + field.BAR_HEIGHT, bar_x : bar_x + field.BAR_WIDTH] = field.HP_BLUE
    skill = bar_y + field.SKILL_ROW_OFFSET
    pixels[skill : skill + field.BAR_HEIGHT, bar_x : bar_x + field.BAR_WIDTH] = track
    pixels[skill : skill + field.BAR_HEIGHT, bar_x : bar_x + filled] = fill
    return Frame(canvas=CANVAS_2800x1260, pixels=pixels), window


def _skill_span(frame: Frame, window: PixelRect) -> np.ndarray:
    """取这一帧技力条那 146 列 —— 走的**就是判据自己那几步**（不是另写一份定位）。"""
    region = frame.crop(window)
    found = field._locate_health(region)
    assert found is not None
    row, start, profile = found
    flags = field._blue_distance(profile) <= field.HP_COLOUR_TOL
    first = field._left_edge(flags, start)
    skill = field._column_profile(
        region, row + field.SKILL_ROW_OFFSET, start, profile.shape[0]
    )
    return np.ascontiguousarray(skill[first : first + field.BAR_WIDTH])


def test_a_bright_track_no_longer_reads_as_a_full_bar() -> None:
    """**这一条改动的那个现象**：亮底上，旧判据（"比这一条自己亮"）把整条判成填满。

    合成帧按 `sp_series/s00` 那条的形状摆：填充是亮的橙 (227,148,2)，右边
    146−11 列是**橙条纹地板透出来的那一截** (138,80,25)。它的亮度是 80.7，而
    "这一条自己逐列中位亮度 × 0.55" = 44.4 —— 够得上。于是 146 列全算填着 → 读成 1.0。

    （**中位数本身也是被轨道撑起来的**：填 11 列时轨道占 135 列，那个基准就是轨道。）
    """
    frame, window = _synthetic_bar(11, (227, 148, 2), (138, 80, 25), (205, 115, 35))
    span = _skill_span(frame, window)
    luminance = span.mean(axis=1)
    old = luminance >= float(np.median(luminance)) * 0.55
    assert int(old.sum()) == field.BAR_WIDTH, "旧尺子在这张帧上应当把整条算成填着"
    assert field.read_skill(frame, window)[0] == Valued(
        Exact(11 / field.BAR_WIDTH), None
    )


def test_a_green_bar_over_a_bright_scene_reads_its_own_extent() -> None:
    """绿条那一条同理（`sp_series/s01` 的形状）。

    轨道是亮的蓝灰 (121,126,139)，亮度 128.7，与填充 (157,178,71) 的 139.7 只差一点 ——
    **按亮度分不开**（129 ≥ 0.55 × 140 成立），**按颜色分得开**（逐通道最大差 93）。
    """
    frame, window = _synthetic_bar(76, (157, 178, 71), (121, 126, 139), (150, 160, 175))
    fill, meaning = field.read_skill(frame, window)
    assert fill == Valued(Exact(76 / field.BAR_WIDTH), None)
    assert isinstance(meaning, Valued) and meaning.payload == Exact("充能中")


def test_the_fill_colour_is_taken_from_the_left_edge() -> None:
    """填充色是**从这一条自己的左缘**量的，不是写死的一个颜色：两个不同色号的绿条，
    各按各的量，都读得出自己的 30 列。"""
    left = _skill_span(*_synthetic_bar(30, (131, 183, 85), (60, 70, 80), (90, 100, 110)))
    right = _skill_span(*_synthetic_bar(30, (171, 176, 53), (60, 70, 80), (90, 100, 110)))
    assert int(field._skill_fill_flags(left).sum()) == 30
    assert int(field._skill_fill_flags(right).sum()) == 30


def test_a_green_bar_never_reads_as_unknown_while_it_is_still_green() -> None:
    """**绿条不许读成认不出** —— 这是 `GREEN_MAX_RG` 从 −8 挪到 0 的理由。

    实测绿条那一簇到 **−5** 为止（`sp_series/s04` 那条 32 列的浅绿 (171,176,53)），
    而 −8 落在**绿条自己那一簇的里面** → 货真价实的绿条被判成"认不出"。
    那一段只该罩住 R > G 的列。
    """
    frame, window = _synthetic_bar(32, (171, 176, 53), (140, 82, 27), (205, 115, 35))
    fill, meaning = field.read_skill(frame, window)
    assert fill == Valued(Exact(32 / field.BAR_WIDTH), None)
    assert isinstance(meaning, Valued) and meaning.payload == Exact("充能中")
    assert 171 - 176 <= field.GREEN_MAX_RG


def test_the_hue_lines_sit_in_the_measured_gap() -> None:
    """**两条线的每一侧余量都指得出出处**（三批 34 帧上逐帧量的，量的是填着那一截的中位 `R − G`）。

    - 绿那一侧：绿条那一簇到 **−5** 为止 —— 出处是回归批 `s04`（整条 32 列）。所以 0 到那一簇 5 格。
    - 橙那一侧：橙条那一簇从 **+65** 起 —— 出处是第一批 `cd01_c` / `cd12_a` / `shot26_a`（都是 +65；
      同批 `cd01_b` 是 +71、回归批 `s00` 是 +77）。所以 0 到那一簇 65 格。
    - **这 65 格里一列样本都没有**：那个"空"不是量出来的，是没有样本。
    """
    assert -5 < field.GREEN_MAX_RG <= 0, "绿那一簇到 −5 为止：线要在它上面，且不越到正的那一侧"
    assert 0 < field.ORANGE_MIN_RG <= 65, "橙那一簇从 +65 起"
    # 认不出那一档（两条线之间）就是 GREEN_MAX_RG … ORANGE_MIN_RG 那一段 —— **没有样本**。
    assert field.GREEN_MAX_RG < field.ORANGE_MIN_RG


def test_a_middle_band_track_is_not_counted_as_fill() -> None:
    """**合成一例**：轨道被背景染成"两族都不是"的颜色时，**不算填充**。

    这是 `r01_06_04` 那一档的形状：一条**空的**技力条，轨道是半透明的、底下的东西透出来 ——
    那个颜色两族都不是（`R − G = +5`）。而 `_skill_fill_reference` 取的是"条最左那 8 列"，
    于是**参考色就是轨道自己**：按距离尺子**整条都"像填充"**，旧读数会是 `Exact(1.0)`
    （**假满** —— 拿一个空的条报成满条）。

    新读数：含义认不出 → 再问"这一行是不是一根空的轨道"（`_looks_like_an_empty_track`：
    两端各量一次"条内比条外暗了多少"；这一帧的轨道比背景暗，两端都落在
    `_TRACK_DROP_MIN`…`_TRACK_DROP_MAX` 里）→ 那是**这一条自己的空轨道** → 填充 **0**、
    含义**照实交 `Unknown`**（认得出有东西、只是读不准）—— **不是**未采样。
    """
    frame, window = _synthetic_bar(0, (150, 145, 60), (150, 145, 60), (200, 205, 215))
    assert field.GREEN_MAX_RG < 150 - 145 < field.ORANGE_MIN_RG
    span = _skill_span(frame, window)
    assert int(field._skill_fill_flags(span).sum()) == field.BAR_WIDTH, (
        "按距离尺子整条都像填充 —— 这正是旧读数会读成满的原因"
    )
    fill, meaning = field.read_skill(frame, window)
    assert fill == Valued(Exact(0.0), None)
    assert meaning == Valued(Unknown(), None)
    assert field.sp_of(fill, meaning) == Valued(Exact(0.0), None)


def _empty_track_frame(background, track) -> tuple[Frame, PixelRect]:
    """一根**空**的技力条（轨道比背景暗）+ 一根满的血条 —— 给下面两条合成用例用。

    与 `_synthetic_bar` 同一个摆法，只是**不带填充**；边缘那几列怎么动由用例自己改。
    """
    return _synthetic_bar(0, track, track, background)


def _edge_drops_of(frame: Frame, window: PixelRect) -> tuple[
    np.ndarray, int, float | None, float | None
]:
    """走判据自己那几步，把 `(技力那一行, first, 左落差, 右落差)` 取出来。"""
    region, _, _ = field._region(frame, window)
    found = field._locate_health(region)
    assert found is not None
    row, start, profile = found
    flags = field._blue_distance(profile) <= field.HP_COLOUR_TOL
    first = field._left_edge(flags, start)
    assert first >= 0
    skill = field._column_profile(
        region, row + field.SKILL_ROW_OFFSET, start, profile.shape[0]
    )
    left, right = field._edge_drops(skill, first)
    return skill, first, left, right


def test_an_empty_track_with_antialiased_edges_still_reads_zero() -> None:
    """**合成一例**：空轨道两端**带抗锯齿混色**时，那几列不许把结论弄反。

    用户提醒过：抗锯齿下边缘那一两列是**混色**（轨道色与背景各半），不是条自己的颜色。
    这里造两根一模一样的空轨道，只有**边缘那 2 列**不同：一版硬边、一版把两端各 2 列
    与背景对半混色。钉三件事：

    1. 混色那几列**真的是混色**（既不是轨道色、也不是背景色）—— 否则这条用例没压到东西；
    2. 两版的 `_edge_drops` **一模一样**（跳 `_SKILL_FILL_REF_SKIP` 列那条口径把那几列
       挡在窗口外），而且两端都落在带子里 → `_looks_like_an_empty_track` 为 True；
    3. 混色那一版的读数是**精确 0**、含义 `Unknown`（不是未采样）。
    """
    background = (200, 205, 215)
    track = (150, 145, 60)
    blended = tuple((int(t) + int(b)) // 2 for t, b in zip(track, background))
    assert blended != track and blended != background

    hard, window = _empty_track_frame(background, track)
    pixels = hard.pixels.copy()
    bar_x, bar_y = field.BARS_IN_WINDOW
    row = bar_y + field.SKILL_ROW_OFFSET
    edges = (bar_x, bar_x + 1, bar_x + field.BAR_WIDTH - 2, bar_x + field.BAR_WIDTH - 1)
    for column in edges:
        pixels[row : row + field.BAR_HEIGHT, column] = blended
    for column in edges:
        assert tuple(int(v) for v in pixels[row, column]) == blended, "混色那几列没混上"
    soft = Frame(canvas=CANVAS_2800x1260, pixels=pixels)

    hard_skill, _hf, hard_left, hard_right = _edge_drops_of(hard, window)
    soft_skill, _sf, soft_left, soft_right = _edge_drops_of(soft, window)
    assert (hard_left, hard_right) == (soft_left, soft_right), (
        "两端那 2 列混色进了窗口 —— 跳列那条口径没挡住"
    )
    for drop in (soft_left, soft_right):
        assert drop is not None and field._TRACK_DROP_MIN <= drop <= field._TRACK_DROP_MAX
    assert field._looks_like_an_empty_track(soft_skill, _sf)
    fill, meaning = field.read_skill(soft, window)
    assert fill == Valued(Exact(0.0), None)
    assert meaning == Valued(Unknown(), None)
    assert field.sp_of(fill, meaning) == Valued(Exact(0.0), None)
    assert field._looks_like_an_empty_track(hard_skill, _hf)


def test_a_near_black_cover_over_the_bar_is_not_an_empty_track() -> None:
    """**合成一例**：条头被一个**近黑的盒子**压住时，左缘的落差**过头**了 → 不许认成空轨道。

    真帧 `r01_01_05` 就是这一档（左缘落差 **0.868**）。合成帧里把条头那 10 列刷成近黑
    （`(8, 8, 8)`），背景是亮的 —— 左缘"外 − 内"接近 1.0，上沿 `_TRACK_DROP_MAX` 挡住；
    右缘那一侧仍然落在"半透明该有的那一段"（0.43）—— **一头过头就够否掉它**。
    """
    background = (200, 205, 215)
    track = (150, 145, 60)
    frame, window = _empty_track_frame(background, track)
    pixels = frame.pixels.copy()
    bar_x, bar_y = field.BARS_IN_WINDOW
    row = bar_y + field.SKILL_ROW_OFFSET
    pixels[row : row + field.BAR_HEIGHT, bar_x : bar_x + 10] = (8, 8, 8)
    covered = Frame(canvas=CANVAS_2800x1260, pixels=pixels)

    skill, first, left, right = _edge_drops_of(covered, window)
    assert left is not None and left > field._TRACK_DROP_MAX, (left, right)
    assert right is not None and field._TRACK_DROP_MIN <= right <= field._TRACK_DROP_MAX
    assert not field._looks_like_an_empty_track(skill, first)


def _marked_bar(cover: int) -> tuple[Frame, PixelRect]:
    """一根**满**的绿条，从第 40 列起被一块 **cover 列宽**的"别的东西"压住。"""
    frame, window = _synthetic_bar(
        field.BAR_WIDTH, (157, 178, 71), (80, 90, 100), (60, 70, 80)
    )
    bar_x, bar_y = field.BARS_IN_WINDOW
    skill = bar_y + field.SKILL_ROW_OFFSET
    pixels = frame.pixels.copy()
    pixels[skill : skill + field.BAR_HEIGHT, bar_x + 40 : bar_x + 40 + cover] = (255, 58, 0)
    return Frame(canvas=CANVAS_2800x1260, pixels=pixels), window


def test_a_narrow_cover_over_the_bar_is_crossed() -> None:
    """压在上面的是**另一种颜色**（别的条）：缝 ≤ `COVER_GAP_TOLERANCE` 列时**跨过去接着算**。

    真帧见 shot04_a：**另一根条**压在绿条上，缝最宽 8 列。跨过去之后**最后一列**
    还是填充色，而填充是**自左端起**的 —— 那一列只能是同一条填充的一部分，所以这一条是
    满的。**这不是猜**。

    （旧读数也是 1.0，但走的路完全不同：旧判据把数字按"亮"算**成了填充**。）
    """
    assert field.COVER_GAP_TOLERANCE >= 8, "shot04_a 那排数字压出来的缝最宽 8 列"
    marked, window = _marked_bar(8)
    slot, meaning = field.read_skill(marked, window)
    assert slot == Valued(Exact(1.0), None)
    assert isinstance(meaning, Valued) and meaning.payload == Exact("充能中")


def test_a_wide_cover_leaves_only_a_lower_bound() -> None:
    """缝比 `COVER_GAP_TOLERANCE` 还宽 → 跨不过去，只能说"缝之前那一截"——**下界**。

    而且**下界不许高过实际的填充**：缝底下藏着多少看不见，所以那两个数都取在缝之前。
    """
    wide = field.COVER_GAP_TOLERANCE + 10
    marked, window = _marked_bar(wide)
    slot, meaning = field.read_skill(marked, window)
    assert isinstance(slot, Valued) and isinstance(slot.payload, LowerBound)
    assert slot.payload.lo == pytest.approx(40 / field.BAR_WIDTH)
    assert isinstance(meaning, Valued) and meaning.payload == Exact("充能中")


def test_the_cover_tolerance_does_not_touch_the_ammo_separators() -> None:
    """**跨缝那一条管不到分格橙条**：分格线是 3–4 列，靠 `GAP_TOLERANCE = 4` 跨。

    这一条是"两条缝分开"那个决定的护栏：`COVER_GAP_TOLERANCE` 只管被压住那一档，
    弹药那一档的比例**一个字都不许跟着动**。
    """
    assert field.COVER_GAP_TOLERANCE > field.GAP_TOLERANCE


# ---------------------------------------------------------------------------
# 「满」要自证：最右那一截里得看得到**本条填充色族**的颜色
# ---------------------------------------------------------------------------


HARD_BATCH = REPO / "tests" / "帧样本" / "场上干员难例"
HARD_CROPS = (
    "r01_04_03",
    "r01_01_05",
    "r01_01_06",
    "r01_06_04",
    "r01_05_04",
    "r01_07_02",
)
"""要钉的六块（源帧见各自的 `取材.json`）：

- `r01_04_03`（`shot_20261007-202959-919`）：尾巴 8 列是外来颜色 —— 技能按钮 28/30，**不是满**；
- `r01_01_05` / `r01_01_06`（`hard_多个技能条紧挨着`）：技力条被**详情界面的图标**盖住，
  血条的高度也被遮了一截 —— 技力落**未采样**；
- `r01_06_04`（`shot_20261007-203049-920`）：**空轨道被背景染了色** —— 技力 **0**（含义**不是**
  未采样）、血条**满**；
- `r01_05_04`（`shot_20261007-203019-925`）与 `r01_07_02`（`shot_20261007-203958-894`）：
  右端被压暗 —— 血条**满**（差一两列也算过）。
"""


@pytest.fixture(scope="module")
def hard_frames() -> dict[str, tuple[Frame, PixelRect]]:
    """只解**要用的那几条**（`case.raw()`），不重放整批 —— 52 帧一起解是 700 MB。

    窗口还是从 `取材.json` 里读（与建批侧**同一份坐标**，不另算一份）。
    """
    batch = load_batch(HARD_BATCH)
    by_name = {case.name: case for case in batch.cases}
    windows = {
        item["file"]: item["window"]
        for item in json.loads((HARD_BATCH / "取材.json").read_text(encoding="utf-8"))["crops"]
    }
    return {
        name: (
            frame_from_raw(by_name[name].raw()),
            PixelRect(*windows[name + ".png"]),
        )
        for name in HARD_CROPS
    }


def test_a_neighbouring_fill_lands_inside_the_scan_span(hard_frames) -> None:
    """**「左缘被旁边那根条带偏」的根**：这一批的条只差 138 px，邻条的填充落进了本条的扫区。

    `hard_多个技能条紧挨着.png` 上那四根（`r01_01_03` / `04` / `05` / `06`）左缘两两相差
    **138 / 139 px**，而每根有 146 宽 —— 于是每根的填充都铺到**下一根左缘的左面 1 列**为止
    （左邻 `r01_01_04` 铺到第 1463 列，本条的左缘是 1465；中间那一列 1464 是那道抗锯齿的
    接缝，实测离血条蓝 29）。

    `_left_edge` 的扫区起点 = 应然左缘 − `COL_TOL`（= 12），所以邻条那 **11** 列填充
    **本来就在扫区里**，而且**本来就是血条蓝**。判据问的是「自起点起 146 列里亮列最多的
    那个起点」—— 在 `r01_01_06` 上：起点 0 得 **145** 列（邻条 11 列 + 本条 134 列），
    应然那一列得 **144** 列（本条 137 列 + 右邻 7 列）。**邻条那 11 列比本条那 3 列尾巴
    值钱**，所以判据选了 0（差一列，不是打平；`r01_01_03` 那根才是打平 144 对 144）。

    **后果**（这一条不钉落点，只钉它为什么会伤人）：那 11 列在**技力条那一行**是暗的
    （逐列中位 ~(52,47,42)），本条自己的填充色在那条边右边是绿的 —— 所以起点一偏左，
    `_skill_fill_reference` 取到的参考色就变成暗色，读数列会跟着翻。

    这一条钉的是**画面事实**（邻条蓝落在本条扫区最左那一列、两者只隔一列、邻条比本条早
    138）：判据怎么修都不该改变这些数。
    """
    frame, window = hard_frames["r01_01_06"]
    crops = json.loads((HARD_BATCH / "取材.json").read_text(encoding="utf-8"))["crops"]
    bars = {item["file"][:-4]: item["bar"] for item in crops}
    assert bars["r01_01_04"][0] - bars["r01_01_06"][0] == -138, "左邻就在 138 那一格上"
    assert bars["r01_01_05"][0] - bars["r01_01_06"][0] == 139, "右邻在 139 那一格上"

    region = frame.pixels[
        window.y : window.y + field.OPERATOR_UI_HEIGHT,
        window.x : window.x + field.OPERATOR_UI_WIDTH,
    ]
    row, start, profile = field._locate_health(region)
    flags = field._blue_distance(profile) <= field.HP_COLOUR_TOL
    expected = field.BARS_IN_WINDOW[0] - start
    assert expected == 12, "应然左缘在扫区里的第 12 列"
    assert bool(flags[0]), "扫区最左那一列（应然左缘 −12）本来就是蓝的 —— 那是邻条"
    assert [int(v) for v in flags[:expected]] == [1] * 11 + [0], "邻条铺到本条左缘的左面 1 列"
    assert int(flags[0 : field.BAR_WIDTH].sum()) == 145
    assert int(flags[expected : expected + field.BAR_WIDTH].sum()) == 144

    skill = field._column_profile(
        region, row + field.SKILL_ROW_OFFSET, start, profile.shape[0]
    )
    near = field._skill_fill_reference(
        np.ascontiguousarray(skill[: field.BAR_WIDTH])
    )
    own = field._skill_fill_reference(
        np.ascontiguousarray(skill[expected : expected + field.BAR_WIDTH])
    )
    assert near is not None and int(near.max()) < 70, "起点偏左时取到的参考色是暗的"
    assert own is not None and int(own[0]) - int(own[1]) <= field.GREEN_MAX_RG, (
        "本条自己那一侧的填充色是绿族（R − G ≤ 0）"
    )


def test_a_dead_operator_beside_a_healthy_neighbour_reads_as_alive(hard_frames) -> None:
    """**核实过的一条假事实**：本条阵亡了，只要**左邻**还活着，这一格照样读得出血条 ——
    而且是**精确**值。

    线上那件事：部署位置程序自己记着（干员是程序通过 adb 部署的），所以「这个干员死没死」
    是拿**记下来的那个位置**开取区、看还认不认得出条。左缘被邻条带偏（见上一条）之后，
    那个取区里剩下的蓝**不是本条的，是左邻的** —— 于是本条的条消失之后，这一格
    仍然有东西可数：干员没了，读数却是「还有一点血」。

    实测（`r01_01_06`，左邻 `r01_01_04` 在 138 px 那一格上；把本条的条区抹掉、把左邻
    按它还剩多少列画回它自己的位置上）：

    - 左邻剩 **129 列（88.4%）** 以下 ⇒ `Unsampled()`（认得出「这里没有条」）；
    - 左邻剩 **130 列（89.0%）** 起 ⇒ 开始读成**活的**；
    - 左邻满血 ⇒ `Exact(20/146)` = **0.1370**。

    账是这么算的：扫区最左那一列 = 本条的左缘 − `COL_TOL`，所以左邻的填充只要伸进这
    12 列里就能被数到；再要过 `MIN_FILL_COLUMNS`，阈值就是

        间距 − COL_TOL + MIN_FILL_COLUMNS = 138 − 12 + 4 = 130 列

    而那个假比例 = (左邻填充 − (间距 − COL_TOL)) / 146。间距 139 的邻居同理是 131 列 /
    0.1301（`r01_01_05` / `r01_01_03` 都是这个数）。

    **修好之后这三行都翻成了 `Unsampled()`**（改动落在 `_left_edge` 的轴 I 里）。
    上面那几条画面事实（间距 138 / 两根只差一行 / 130 列那个算式）不用动 ——
    它们说的是「旧判据为什么会错」，不是落点。
    """
    frame, window = hard_frames["r01_01_06"]
    crops = json.loads((HARD_BATCH / "取材.json").read_text(encoding="utf-8"))["crops"]
    bars = {item["file"][:-4]: item["bar"] for item in crops}
    bar = bars["r01_01_06"]
    left = bars["r01_01_04"]
    assert left[0] - bar[0] == -138, "同一行上左边那根在 138 那一格"
    assert left[1] == bar[1] - 1, "两根只差一行（实测 630 对 631）"

    def dead(neighbour_columns: int) -> Frame:
        """把**本条**的条区抹掉，再把左邻按它还剩多少列画回它自己的位置上。"""
        pixels = frame.pixels.copy()
        pixels[bar[1] : bar[1] + 16, bar[0] : bar[0] + field.BAR_WIDTH, :] = 128
        rows = slice(bar[1], bar[1] + field.BAR_HEIGHT)
        pixels[rows, left[0] : left[0] + field.BAR_WIDTH, :] = 60
        pixels[rows, left[0] : left[0] + neighbour_columns, :] = field.HP_BLUE
        return Frame(canvas=frame.canvas, pixels=pixels)

    assert field.read_health(frame, window) == Valued(Exact(1.0), 0.0), "活着时是满的"
    # **修好之后：三行都落未采样。** 修的是 _left_edge（轴 I）：候选只认「看得见的亮段
    # 起点」，而左邻那一截的**起点在扫区外面**（相邻两条左缘只差 138 列、条宽 146），
    # 所以它不算候选 —— 就算 _locate_health 被那 4 列凑够了最小列数，左缘也找不到，
    # 这一格就是「没有条」。dead(130) / dead(BAR_WIDTH) 都不再假活。
    assert field.read_health(dead(129), window) == Unsampled(), "左邻 88.4% —— 还认得出没有条"
    assert field.read_health(dead(130), window) == Unsampled(), "左邻 89.0% —— 修好之后不再假活"
    assert field.read_health(dead(field.BAR_WIDTH), window) == Unsampled(), "左邻满血 —— 同上"


def test_a_low_health_bar_with_an_empty_track_reads_as_zero() -> None:
    """**合成用例**：一个蓝像素都没有、但那一行**量得出空轨道** → 血条读 **0**（生命极低）。

    真帧里没有这一档（素材里没有"生命极低"的样本，见难例批 README 的「没覆盖到什么」），
    所以按画面事实造一帧：平底 **128**；左边 **8 列**画成血条蓝 —— 那是**左邻的尾巴**
    （`_locate_health` 靠它凑够最小列数、定位到这一行；`_left_edge` 不认它，因为那一段
    的起点在扫区外面，而"从它起连得出 134 列"那道门也没过）；血条自己那 146 列画成
    **83 ≈ 128 × 0.6485** —— 也就是「半透明轨道把底压暗到六成半」那件事（实测 α = 0.3515）。

    **这一档与"人没了"必须分开**（用户裁定）：前者读 **0**、后者读**未采样**，分开它们的
    正是这一问 ——「这一行是不是一根空轨道」（**上沿**那一侧量，见 `_empty_health_track`）。
    这里量到 1 − 83/128 = **0.352**，落在 `_TRACK_DROP_MIN`…`_TRACK_DROP_MAX` 里。
    """
    pixels = np.full(
        (CANVAS_2800x1260.height, CANVAS_2800x1260.width, 3), 128, dtype=np.uint8
    )
    window = PixelRect(0, 0, field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT)
    bar_x, bar_y = field.BARS_IN_WINDOW
    start = bar_x - field.COL_TOL
    pixels[bar_y : bar_y + field.BAR_HEIGHT, start : start + 8] = field.HP_BLUE
    pixels[bar_y : bar_y + field.BAR_HEIGHT, bar_x : bar_x + field.BAR_WIDTH] = int(
        round(128 * 0.6485)
    )
    frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    assert field.read_health(frame, window) == Valued(Exact(0.0), None)


def test_the_far_tail_of_a_covered_green_bar_is_not_full(hard_frames) -> None:
    """**真帧一例**：最右那一截是外来颜色时，**不许默认读满**。

    这一条的最右 8 列（第 138…145 列）离参考色 **30–44**，而它**自己的填充体**
    （第 0…137 列）**≤ 17** —— 所以**按旧的距离尺子它们算"填着"**：`_leading_run`
    跨过 3 列的断口一路到 146，旧读数 `Exact(1.0)`。而那 8 列的逐列中位
    `R − G ≈ +28…+31`（**两族都不是**）—— 于是"填充画到右端"这句话**没有证据**了。

    技能按钮上的 28/30 也证实它**不是满**（真值不下发，这里只引用旁证）。
    """
    frame, window = hard_frames["r01_04_03"]
    span = _skill_span(frame, window)
    lit = field._skill_fill_flags(span)
    strict, tolerant = field._leading_run(lit, field.GAP_TOLERANCE)
    assert min(tolerant, field.BAR_WIDTH) == field.BAR_WIDTH, "旧尺子在这条上一直亮到最后一列"
    assert strict == 138, "填充体（缝之前那一截）是 138 列"
    fill, meaning = field.read_skill(frame, window)
    assert isinstance(meaning, Valued) and meaning.payload == Exact("充能中")
    assert not field._right_end_is_fill(span, lit, meaning), "最右那一截里没有一列是绿族"
    assert fill == Valued(LowerBound(138 / field.BAR_WIDTH), None)
    assert field.sp_of(fill, meaning) == Valued(
        Interval(138 / field.BAR_WIDTH, 1.0), None
    )


def test_a_skill_bar_under_the_detail_panel_falls_unsampled(hard_frames) -> None:
    """**真帧一例**：技力条被**详情界面的图标**盖住时，技力落**未采样**（血条照旧认）。

    `hard_多个技能条紧挨着.png` 的 `r01_01_05`（条 1604,630）与 `r01_01_06`（条 1465,631）：
    那一行是**详情界面那个黑盒**（逐列中位 ~(16,15,15) / (52,47,42)，彩度 1 / 10），
    所以 `_skill_fill_flags` 拿到的"填充参考色"是黑的，按距离尺子还能凑出 74 / 12 列。

    用户给的画面事实：**蓝条（血条）能认、但高度被遮了一半** —— 压住技力条的是**那一个
    近黑的盒子**，不是半透明的轨道。两端量出来的就不是"半透明该有的那一段"：

    - `r01_01_05`：左缘落差 **0.86** —— **过头**了（上沿 `_TRACK_DROP_MAX` 挡住）；
    - `r01_01_06`：左缘窗口越出取区（量不出来），右缘 **−0.15** —— 条内比条外还**亮**。

    所以 `_looks_like_an_empty_track` 为 False → 这一行不是这根条 → 三个槽全落未采样；
    **血条那边的读数一个字不动**（这一条只钉技力那三个槽）。

    **轴 I 之后 `r01_01_06` 换了条路**：左缘被修成 12（原来是 0，读在左邻的尾巴上），
    于是颜色**认得出**（绿、充能中），上面那条「认不出族」的路到不了它了。现在它由
    **纵向判据**（行带 +8…+15 各自对往下第十行、按 `_TRACK_SHADOW` 除阴影、否决线 0.17）
    自证「断口之后那一截不是半透明轨道」（它的行带中位 0.111，自控 0.12 是过的）
    → 比例报到断口、含义未采样。
    """
    for name in ("r01_01_05", "r01_01_06"):
        frame, window = hard_frames[name]
        region, _, _ = field._region(frame, window)
        row, start, profile = field._locate_health(region)
        flags = field._blue_distance(profile) <= field.HP_COLOUR_TOL
        first = field._left_edge(flags, start)
        skill = field._column_profile(
            region, row + field.SKILL_ROW_OFFSET, start, profile.shape[0]
        )
        assert not field._looks_like_an_empty_track(skill, first), name
        left, right = field._edge_drops(skill, first)
        if name == "r01_01_05":
            assert left is not None and left > field._TRACK_DROP_MAX, (name, left)
        else:
            assert right is not None and right < field._TRACK_DROP_MIN, (name, right)
        fill, meaning = field.read_skill(frame, window)
        assert meaning == Unsampled(), name
        if name == "r01_01_05":
            # 含义**认不出族**（左缘被黑盒带偏那一路）→「这一行不是这根条」→ 两个量都未采样。
            assert fill == Unsampled(), name
        else:
            # r01_01_06：轴 I 把左缘修成 12，于是颜色**认得出**（绿、充能中），上面那一支
            # 到不了了；改由**纵向判据**自证断口之后那一截不是轨道 → 比例只报到**断口**
            # （46 列）、含义未采样（用户裁定：重点是未采样，下界有没有都行，取有下界那版）。
            assert fill == Valued(LowerBound(46 / field.BAR_WIDTH), None), name
        assert field.sp_of(fill, meaning) == Unsampled(), name


def test_an_empty_track_tinted_by_the_background_is_fill_zero(hard_frames) -> None:
    """**真帧一例**：空轨道被背景染了色 → 技力 **0**；同一块的血条是**满**。

    `shot_20261007-203049-920.png` 的 `r01_06_04`（条 1054,666）：那一行是**条自己的空轨道**
    （半透明，底下的画透出来，逐列中位 `R − G` 两族都不是）。判据是**两端各量一次
    "条内比条外暗了多少"**：左 **0.159** / 右 **0.300** —— 都落在
    `_TRACK_DROP_MIN`…`_TRACK_DROP_MAX` 里 → 那是**半透明**该有的那一段 → 填充 0。

    含义**照实交 `Valued(Unknown())`**（「认得出有东西、但读不准」那一族，也是状态层
    「技能是否生效中 / 剩余时长」形态表里的**未知**）—— 用户裁的：**技力未知与技力为 0
    是两个完全不同的行为**，「看不到」才落未采样；这一块是**看得到、只是空的**。

    血条那边：最后 6 列离血条蓝 **25–29**（半亮）**一直顶到条的右端**（条外那一列 186）
    → 那不是"边界落在条里面"，是**整条都在、右端被压暗** → 满。
    """
    frame, window = hard_frames["r01_06_04"]
    region, _, _ = field._region(frame, window)
    row, start, profile = field._locate_health(region)
    flags = field._blue_distance(profile) <= field.HP_COLOUR_TOL
    first = field._left_edge(flags, start)
    skill = field._column_profile(
        region, row + field.SKILL_ROW_OFFSET, start, profile.shape[0]
    )
    assert field._looks_like_an_empty_track(skill, first)
    left, right = field._edge_drops(skill, first)
    for drop in (left, right):
        assert drop is not None and field._TRACK_DROP_MIN <= drop <= field._TRACK_DROP_MAX
    assert field.read_health(frame, window).payload == Exact(1.0)
    fill, meaning = field.read_skill(frame, window)
    assert fill == Valued(Exact(0.0), None)
    # **含义照实交 Unknown**（「认得出有东西、但读不准」那一族），**不是未采样**：
    # 「看不到」与「看得到、只是空的」是两件事 —— 技力未知与技力为 0 是两个完全不同的行为。
    assert meaning == Valued(Unknown(), None)
    assert field.sp_of(fill, meaning) == Valued(Exact(0.0), None)


def test_a_dim_right_end_still_reads_full_with_a_ragged_band(hard_frames) -> None:
    """**真帧两例**：右端被压暗时血条是**满**。

    `shot_20261007-203019-925.png` 的 `r01_05_04`（条 1036,636）与
    `shot_20261007-203958-894.png` 的 `r01_07_02`（条 1580,467）：

    - **半亮那一截一直顶到条的右端**：`r01_05_04` 第 141…145 列离血条蓝
      **24 / 25 / 26 / 27 / 27**、条外那一列 **191**；`r01_07_02` 是
      **32 / 36 / 40 / 41 / 41**、条外 **203** —— 条外那一列早就不蓝了，
      也就是**没有边界落在条里面**（用户裁定：这就是满，**差一两列也算过**）。

    **这一支不看任何"血条那一带对不对得上条"** —— 那条判据（按血条蓝的列数看带子齐不齐）
    已经撤掉了：这两块的 8 行蓝列数是 `[139, 138, 140, 140, 142, 143, 144, 145]` /
    `[140, 140, 140, 141, 143, 146, 146, 146]`，各有一两行差 1–3 列（抗锯齿噪声），
    旧判据照样挡回去；而用户裁定这两块都是**满**。
    """
    for name, tail, outside in (
        ("r01_05_04", (24, 25, 26, 27, 27), 191),
        ("r01_07_02", (32, 36, 40, 41, 41), 203),
    ):
        frame, window = hard_frames[name]
        region, _, _ = field._region(frame, window)
        row, start, profile = field._locate_health(region)
        flags = field._blue_distance(profile) <= field.HP_COLOUR_TOL
        first = field._left_edge(flags, start)
        sliced = flags[first : first + field.BAR_WIDTH + 1]
        distances = field._blue_distance(profile)[first : first + field.BAR_WIDTH + 1]
        _strict, tolerant = field._leading_run(sliced, field.GAP_TOLERANCE)
        tolerant = min(tolerant, field.BAR_WIDTH)
        assert tolerant < field.BAR_WIDTH, name
        got = tuple(int(v) for v in distances[tolerant : field.BAR_WIDTH])
        assert got == tail, (name, got)
        assert distances[field.BAR_WIDTH] > 2 * field.HP_COLOUR_TOL, (name, outside)
        assert field.read_health(frame, window).payload == Exact(1.0), name


def _bar_with_a_foreign_tail(
    seams: int, tail: int,
) -> tuple[Frame, PixelRect]:
    """一根**满**的绿条，但尾巴那 `seams + tail` 列换成外来的颜色（合成）。

    形状照着实测那条摆：**先一截距离够远的边界**（算断口），
    **再一截距离够近、但两族都不是的尾巴**（旧尺子会把它算成填着）。
    """
    frame, window = _synthetic_bar(
        field.BAR_WIDTH, _FOREIGN_FILL, (80, 90, 100), (60, 70, 80)
    )
    bar_x, bar_y = field.BARS_IN_WINDOW
    skill = bar_y + field.SKILL_ROW_OFFSET
    pixels = frame.pixels.copy()
    start = field.BAR_WIDTH - seams - tail
    pixels[skill : skill + field.BAR_HEIGHT, bar_x + start : bar_x + start + seams] = _FOREIGN_SEAM
    pixels[skill : skill + field.BAR_HEIGHT, bar_x + start + seams : bar_x + field.BAR_WIDTH] = _FOREIGN_TAIL
    return Frame(canvas=CANVAS_2800x1260, pixels=pixels), window


_FOREIGN_FILL: tuple[int, int, int] = (153, 164, 54)
"""填充色：绿族（`R − G = −11`）。"""

_FOREIGN_SEAM: tuple[int, int, int] = (129, 101, 68)
"""边界那 3 列：离填充色 **63**（> `SKILL_FILL_TOL`）→ 算**断口**。"""

_FOREIGN_TAIL: tuple[int, int, int] = (165, 135, 85)
"""尾巴那 5 列：离填充色 **31**（≤ `SKILL_FILL_TOL`），但 `R − G = +30`
落在 `GREEN_MAX_RG … ORANGE_MIN_RG` 那一档里 —— **两族都不是**。"""


def test_a_foreign_tail_on_a_green_bar_is_not_read_as_full() -> None:
    """**合成一例**：同一套颜色下，**满** 与**差最后一小截**不再落同一个读数。

    两条条**只差尾巴那 8 列**：

    - 一条真的满（全 146 列都是填充色）→ `Exact(1.0)`；
    - 一条差最后一小截（尾巴 8 列是外来颜色）→ `LowerBound(138/146)`，
      经 `sp_of` 加宽成 `Interval(138/146, 1.0)`。

    尾巴那个颜色**故意取在距离尺子里面**（离填充色 31 ≤ 48）——
    这就是旧读数为什么会读成满，也是这一档名字里那个"读不准"的样子。
    """
    assert max(abs(a - b) for a, b in zip(_FOREIGN_TAIL, _FOREIGN_FILL)) <= field.SKILL_FILL_TOL
    assert field.GREEN_MAX_RG < _FOREIGN_TAIL[0] - _FOREIGN_TAIL[1] < field.ORANGE_MIN_RG
    assert max(abs(a - b) for a, b in zip(_FOREIGN_SEAM, _FOREIGN_FILL)) > field.SKILL_FILL_TOL

    full, window = _synthetic_bar(
        field.BAR_WIDTH, _FOREIGN_FILL, (80, 90, 100), (60, 70, 80)
    )
    assert field.read_skill(full, window)[0] == Valued(Exact(1.0), None)

    marked, window = _bar_with_a_foreign_tail(3, 5)
    fill, meaning = field.read_skill(marked, window)
    assert fill == Valued(LowerBound(138 / field.BAR_WIDTH), None)
    assert isinstance(meaning, Valued) and meaning.payload == Exact("充能中")
    assert fill != Valued(Exact(1.0), None)
    assert field.sp_of(fill, meaning) == Valued(
        Interval(138 / field.BAR_WIDTH, 1.0), None
    )


def test_a_foreign_tail_does_not_change_the_ammo_reading() -> None:
    """新加那一步**只管绿条那边**：橙条的尾巴本来就是橙族，分格橙条一字不动。

    分格线是**暗线**（离填充色 106–123），它们不在 `lit` 里，但**在橙族里**；
    而一条满的分格橙条最右那一截里总有橙色的填充列 —— 所以它还是满。
    """
    orange = (227, 148, 2)
    track = (138, 80, 25)
    frame, window = _synthetic_bar(field.BAR_WIDTH, orange, track, (205, 115, 35))
    bar_x, bar_y = field.BARS_IN_WINDOW
    skill = bar_y + field.SKILL_ROW_OFFSET
    pixels = frame.pixels.copy()
    for column in range(20, field.BAR_WIDTH, 20):
        pixels[skill : skill + field.BAR_HEIGHT, bar_x + column : bar_x + column + 3] = (60, 40, 30)
    frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    span = _skill_span(frame, window)
    lit = field._skill_fill_flags(span)
    meaning = field._meaning_of(span, min(field._leading_run(lit, field.GAP_TOLERANCE)[1], field.BAR_WIDTH))
    assert isinstance(meaning, Valued) and meaning.payload == Exact("弹药分格")
    assert field._right_end_is_fill(span, lit, meaning)
    assert field.read_skill(frame, window)[0] == Valued(Exact(1.0), None)


# ---------------------------------------------------------------------------
# 技力那个字段：生效期间读不出来
# ---------------------------------------------------------------------------


def test_sp_is_unsampled_while_the_skill_is_active(readings) -> None:
    """**技力生效期间读不出来 -> 未采样**：含义 = 生效中 / 弹药分格 时，技力那个字段
    绝不许拿那条橙条的比例顶上去（那是编一个比例）。"""
    for name in ("cd01_b", "cd01_c", "shot26_a"):
        assert readings[name]["sp"] == Unsampled(), name
        assert isinstance(readings[name]["fill"], Valued)


def test_sp_passes_the_fill_through_while_charging(readings) -> None:
    """充能中那一条原样交出去 —— 技力读的就是绿条那个比例。"""
    assert readings["cd01_a"]["sp"] == readings["cd01_a"]["fill"]
    assert readings["shot21_a"]["sp"] == readings["shot21_a"]["fill"]


def test_sp_does_not_decide_when_the_meaning_is_unknown() -> None:
    """含义认不出时**不替它判**：那时比例自己已经是区间，落未采样反而是把"读不准"
    伪装成"没看到"。"""
    fill = Valued(Interval(0.5, 0.7), None)
    same = field.sp_of(fill, Valued(Unknown(), None))
    assert same == fill


def test_sp_widens_a_lower_bound_reading_into_an_interval() -> None:
    """**下界不是技力那个字段的形态。**

    读数可以是下界（那是**弹药剩余**那一行的形态），但路由进技力之后要**加宽成区间**
    —— 下界说的是"至少这么多"，写成 [下沿, 1] 是**同一个事实的另一种写法**，一个字都
    没有多、也没有少。原样透出去就等于替那个字段加了一个它没收的形态。

    **也不许落未采样**：那时那里确实有东西、下沿也读到了，落未采样是把"读不准"伪装成
    "没看到"。
    """
    fill = Valued(LowerBound(0.5273972602739726), None)
    widened = field.sp_of(fill, Valued(Exact("充能中"), 0.1))
    assert widened == Valued(Interval(0.5273972602739726, 1.0), None)
    assert not isinstance(widened.payload, LowerBound)


def test_sp_widening_keeps_the_confidence() -> None:
    """加宽只换写法，**余量照原样带过去** —— 不许顺手把它抹成 None。"""
    fill = Valued(LowerBound(0.25), 0.375)
    widened = field.sp_of(fill, Valued(Exact("充能中"), 0.1))
    assert widened.confidence == 0.375


def test_sp_never_emits_a_lower_bound_on_the_whole_batch(readings) -> None:
    """整批 16 条：技力那个字段只出现 精确 / 区间 / 未采样，**一条下界都没有**。

    这一批里 `shot26_a` 的**读数**正是下界 —— 所以这条不是空跑。
    """
    lower_bound_readings = [
        name
        for name, row in readings.items()
        if isinstance(row["fill"], Valued) and isinstance(row["fill"].payload, LowerBound)
    ]
    assert lower_bound_readings, "这一批里该有下界读数，否则这条测试没盖到东西"
    forms = {
        "unsampled" if isinstance(row["sp"], Unsampled) else type(row["sp"].payload).__name__
        for row in readings.values()
    }
    assert forms <= {"Exact", "Interval", "unsampled"}, forms
    assert "LowerBound" not in forms
    # **加宽那一支真帧上不触发**：这一批里唯一的读数下界（shot26_a）含义是**弹药分格**，
    # 按字段规则直接落**未采样**，轮不到加宽 —— 所以 **Interval 也不出现**。
    # 那一条支路只剩合成用例（test_sp_widens_a_lower_bound_reading_into_an_interval）。
    assert "Interval" not in forms, forms


# ---------------------------------------------------------------------------
# 可释放菱形
# ---------------------------------------------------------------------------


def test_diamond_separates_the_two_clusters(readings, report) -> None:
    """在 / 不在两簇：分数与批内空档都照实对一遍。"""
    present = [
        row["score"]
        for row in readings.values()
        if isinstance(row["diamond"], Valued) and row["diamond"].payload == Exact(True)
    ]
    absent = [
        row["score"]
        for row in readings.values()
        if isinstance(row["diamond"], Valued) and row["diamond"].payload == Exact(False)
    ]
    assert present and absent
    assert min(present) >= field.DIAMOND_SOLID_SCORE
    assert max(absent) <= field.DIAMOND_ABSENT_SCORE
    gap = report["chains"]["diamond"]["gap"]
    assert gap["absent_max"] == pytest.approx(max(absent), abs=1e-5)
    assert gap["present_min"] == pytest.approx(min(present), abs=1e-5)
    assert gap["absent_max"] < gap["present_min"]


def test_diamond_slot_agrees_with_its_own_score_line(readings) -> None:
    """**自洽**：`Valued` 的取值与它自己那个分数落在线的哪一边，必须一致。

    这一条**不是真值核对** —— 它拿链自己的分数当参照，只能证「分数线那一支没接错」。
    真值核对是下面那条（真值来自夹具的 `covers`，与分数无关）。
    """
    for name, row in readings.items():
        slot = row["diamond"]
        assert isinstance(slot, Valued)
        if isinstance(slot.payload, Exact):
            assert slot.payload.value is (row["score"] >= field.DIAMOND_SOLID_SCORE)


def test_diamond_verdicts_match_the_batch_truth(readings, harvest) -> None:
    """**这一批 16 条：判决 == 真值** —— 真值来自夹具自己的 `covers`，不是链的分数。

    真值怎么读（`取材.json` 的 `covers`）：写了「菱形在」的那 8 条是"在"，写了
    「菱形不在」的那 1 条（`shot01_a`）是"不在"，**其余 7 条不写**（README 记着
    「在 8 条 · 不在 8 条」）。所以先按 `covers` 折出真值、再核它**正好是 8 在 /
    8 不在**：真值自己少写一条时，这条会先红在真值上，而不是把一次漏检吞掉。

    **真值不许拿链的分数倒推**（本单裁定 2）：拿分数当面筛，量到的只有「判决与自己的
    分数线一致」，量不到「判决与画面对得上」。画面对不对由取证图管 —— 16 格逐格裁出
    菱形那个邻域，8 条"在"上都能看见那个黄菱形、8 条"不在"上一个都没有。

    **两个方向分开点数**：漏检（该在却报不在 / 认不出）与**误报（不在却报在 —— 假事实，
    最重）**。这一批实测：命中 **16** · 漏检 **0** · 误报 **0** · 认不出 **0**。

    **两条余量**（这一批实测，判据一改要重跑）："在"那 8 条最低 **0.543291**，离
    `DIAMOND_SOLID_SCORE = 0.45` 还有 **0.093291**；"不在"那 8 条最高 **0.279461**，
    离 `DIAMOND_ABSENT_SCORE = 0.34` 还有 **0.060539**；两簇之间的空档 **0.263830** 宽。
    最贴线的两条是 `shot22_a`（0.543291）与 `cd01_c`（0.279461）。
    """
    truth = {item["file"][:-4]: "菱形在" in item["covers"] for item in harvest["crops"]}
    assert len(truth) == 16, sorted(truth)
    assert sum(truth.values()) == 8, "真值自己不对：`covers` 里数不出 8 条「菱形在」"

    counted = {"在": 0, "不在": 0, "认不出": 0}
    missed: list[str] = []
    false_positives: list[str] = []
    for name, want in truth.items():
        slot = readings[name]["diamond"]
        if isinstance(slot, Valued) and isinstance(slot.payload, Exact):
            got = "在" if slot.payload.value else "不在"
        else:
            got = "认不出"
        counted[got] += 1
        if got == "认不出":
            continue
        if got == ("在" if want else "不在"):
            continue
        if want:
            missed.append(name)
        else:
            false_positives.append(name)

    assert counted["认不出"] == 0, f"这一批里出现了认不出（单独点数，不算错）：{counted}"
    assert counted["在"] == 8 and counted["不在"] == 8, counted
    assert missed == [], f"漏检（该在却没报在）：{missed}"
    assert false_positives == [], f"误报（不在却报在 —— 假事实，最重）：{false_positives}"

    present = [readings[name]["score"] for name, want in truth.items() if want]
    absent = [readings[name]["score"] for name, want in truth.items() if not want]
    assert max(absent) < field.DIAMOND_ABSENT_SCORE, "不在那一簇顶到 0.34 线上了：这条线没有余量"
    assert min(present) > field.DIAMOND_SOLID_SCORE, "在那一簇掉到 0.45 线上了：这条线没有余量"


def test_diamond_is_searched_not_pinned(readings, batch) -> None:
    """**它是搜出来的**：取区挪几像素之后，找到的那个点在**画布上还是同一个点**。

    钉死在小框里的做法做不到这一条 —— 窗口一挪，框里的东西就变了。
    """
    template = field.load_ready_diamond()
    row = readings["board_a"]
    frame = row["frame"]
    window = row["window"]
    _, x0, y0 = field.diamond_score(frame, window, template)
    shifted = PixelRect(window.x + 8, window.y + 8, window.width, window.height)
    _, x1, y1 = field.diamond_score(frame, shifted, template)
    assert abs(x1 - x0) <= 1 and abs(y1 - y0) <= 1


def test_diamond_ambiguous_band_falls_to_unsampled(batch, monkeypatch) -> None:
    """**两条线之间那一段落未采样**，而且**绝不落 Exact(False)** —— 那是漏检方向。

    真帧的分数不落在那一段里（空档是 0.279…0.543），所以把判据本身换成一个固定分数：
    它**就是** `diamond_score`，换它等于把"读数恰好落在空档里"直接摆进去。
    """
    case = next(c for c in load_batch(BATCH).cases if c.name == "cd01_a")
    frame = frame_from_raw(case.raw())
    window = PixelRect(*next(item for item in json.loads(HARVEST.read_text(encoding="utf-8"))["crops"] if item["file"] == "cd01_a.png")["window"])
    middle = (field.DIAMOND_ABSENT_SCORE + field.DIAMOND_SOLID_SCORE) / 2
    monkeypatch.setattr(field, "diamond_score", lambda *a, **k: (middle, 0, 0))
    slot = field.find_ready_diamond(frame, window, field.load_ready_diamond())
    assert slot == Unsampled()
    assert slot != Valued(Exact(False), middle)


def test_diamond_forms_are_only_the_fields_forms(readings) -> None:
    """可手动触发那个字段的形态是 布尔 / 未采样 —— 这条链不许给区间、也不许给未知。"""
    for row in readings.values():
        slot = row["diamond"]
        if isinstance(slot, Valued):
            assert isinstance(slot.payload, Exact) and isinstance(slot.payload.value, bool)
        else:
            assert isinstance(slot, Unsampled)


def test_diamond_does_not_use_a_frame_from_the_batch(batch) -> None:
    """模板**不许取自验收批的帧**（《视觉层/识别方法》§已定 那条纪律）。

    资产里那个 source 得是**这一批之外**的一张原生帧。
    """
    with np.load(field.READY_DIAMOND_PATH) as data:
        source = str(data["source"])
    assert source not in {c.source for c in batch.cases}
    assert not any(source == c.source for c in batch.cases)


# ---------------------------------------------------------------------------
# 能力缺失要响亮地报出来
# ---------------------------------------------------------------------------


def test_missing_template_asset_is_loud() -> None:
    """模板不在 -> **抛**（能力缺失），不是静默给一个空模板。

    **不用 pytest 的 tmp_path**：这台机器的临时目录被沙箱挡着（建不起来）。
    判据与"目录存不存在"无关，所以拿批目录里一个不存在的文件名就够。
    """
    with pytest.raises(FileNotFoundError):
        field.load_ready_diamond(BATCH / "没有这个模板.npz")


def test_template_that_is_too_small_is_refused(batch, readings) -> None:
    """比邻域还大 / 小得像噪声的模板 -> 抛。"""
    frame = readings["cd01_a"]["frame"]
    window = readings["cd01_a"]["window"]
    with pytest.raises(ValueError):
        field.diamond_score(frame, window, np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises((TypeError, ValueError)):
        field.diamond_score(frame, window, np.zeros((8, 8, 3), dtype=np.int16))


def test_template_asset_carries_its_origin() -> None:
    """模板资产里带着**它从哪来** —— 否则"是不是取自验收帧"就无从回答。"""
    with np.load(field.READY_DIAMOND_PATH) as data:
        assert set(data.files) == {"template", "source", "box"}
        assert str(data["source"]).endswith(("_1.bin", ".raw", ".png"))
        assert data["template"].dtype == np.uint8 and data["template"].shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# 判据读数这份报告
# ---------------------------------------------------------------------------


def test_report_pins_the_four_things_the_diagnostics_doc_asks(report) -> None:
    """[像素匹配诊断] §5 要的四样：哪把尺子 · 哪个取区 · 取区原点从哪来 · 去重按什么键。"""
    assert set(report["ruler"]) == {"health", "skill", "diamond"}
    assert report["window"]["size"] == [field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT]
    assert report["window"]["pixels"] == field.OPERATOR_UI_WIDTH * field.OPERATOR_UI_HEIGHT
    assert report["window_from"].startswith("取区左上角 = 血条左上角")
    assert report["dedup"]["key"]


def test_report_has_a_gap_and_a_coverage_note_for_every_chain(report) -> None:
    """每条链都要有**空档**与**覆盖说明** —— 不许只报一堆读数。"""
    for chain in ("health", "skill", "diamond"):
        assert report["chains"][chain]["reads"]
    assert report["chains"]["diamond"]["gap"]["absent_max"] < report["chains"]["diamond"]["gap"]["present_min"]
    assert report["coverage"]["covered"] and report["coverage"]["not_covered"]
    assert report["coverage"]["stuck"]


def test_report_is_byte_stable() -> None:
    """同一批跑两次，报告逐字节一样 —— 没有时间戳、没有绝对路径、没有随机量。"""
    first = REPORT.read_bytes()
    payload = json.loads(first.decode("utf-8"))
    again = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    assert first == again


def test_sensitivity_covers_every_crop_not_just_one(report) -> None:
    """**锚点摆动铺到整批**：16 条**每一条各自**挪 ±δ，不许只报一条帧。

    （上一轮只报了 `cd01_a` 一条，并据此写下"±12 px 结论一字不变" —— 那是拿一条帧的
    读数代替整批；铺开之后不成立，见下一条。）
    """
    sensitivity = report["sensitivity"]
    assert len(sensitivity["crops"]) == len(report["chains"]["health"]["reads"])
    assert set(sensitivity["shifts"]) == {-24, -16, -12, -4, -1, 0, 1, 4, 12, 16, 24}
    for crop in sensitivity["crops"]:
        assert crop["rows"], crop["name"]
        assert {row["axis"] for row in crop["rows"]} == {"x", "y"}
        for row in crop["rows"]:
            assert {"shift", "same", "health_form", "skill_form", "meaning", "diamond"} <= set(row)
    summary = sensitivity["summary"]["within_tolerance"]
    assert summary["x"]["tolerance"] == field.COL_TOL
    assert summary["y"]["tolerance"] == field.ROW_TOL
    for axis in ("x", "y"):
        assert summary[axis]["ok"] <= summary[axis]["total"] == len(sensitivity["crops"])


def test_sensitivity_reports_the_frames_that_flip_inside_the_tolerance(report) -> None:
    """±12 以内**不是一条都不翻** —— 翻的那几条要**逐条**出现在读数里，不许只留一句总结。

    这一批实测翻面 **7 帧 12 行**，两种形状：

    - **条挪了几行**（y）：`shot02_a`（y −4 / −12：技力填充 精确 → 下界）、
      `shot04_a`（y +4 / +12：同上 —— 条挪了几行之后，那条缝就跨不过去了）、
      `shot26_a`（y −4 / −12：含义 弹药分格 → 充能中）。这一档与技力那条判据无关；
    - **取区往右挪 12 px**（x +12）：`board_b` / `cd01_c` / `cd12_a` / `field_b` /
      `shot02_a` / `shot04_a` —— 条的真左缘这时正好落在**扫区最左那一列**，而那一列
      只有在「从它起能连出至少 BAR_WIDTH − COL_TOL 列」时才认（否则它长得跟
      「左邻的尾巴」一模一样，见 `_left_edge`）。这六块的填充都不到 134 列，认不出来
      ⇒ 落未采样（**丢读数，不给错值**）。同一批里**满条/接近满条**的那 10 块被救了回来
      —— 摆动表里 x 那一栏 `ok 10 / 16`，加这道门之前是 `0 / 16`。

    **翻，不等于判据坏了**：窄窗里每一条撞到的东西不一样（旁边那个干员的条、条右边
    那截亮背景）。窗口该多宽是规格问题，不是拿一条帧上的一句总结能代替的。
    """
    sensitivity = report["sensitivity"]
    swing = report["coverage"]["swing_within_tolerance"]
    inside = [
        row
        for crop in sensitivity["crops"]
        for row in crop["rows"]
        if not row["same"]
        and abs(row["shift"]) <= (field.COL_TOL if row["axis"] == "x" else field.ROW_TOL)
    ]
    assert swing["count"] == len(inside)
    assert swing["count"] == len(swing["rows"])
    assert {row["name"] for row in swing["rows"]} == {
        "board_b.png",
        "cd01_c.png",
        "cd12_a.png",
        "field_b.png",
        "shot02_a.png",
        "shot04_a.png",
        "shot26_a.png",
    }
    for axis in ("x", "y"):
        summary = sensitivity["summary"]["within_tolerance"][axis]
        broken = {
            crop["name"]
            for crop in sensitivity["crops"]
            for row in crop["rows"]
            if row["axis"] == axis and not row["same"] and abs(row["shift"]) <= summary["tolerance"]
        }
        assert summary["ok"] == summary["total"] - len(broken), axis

# ---------------------------------------------------------------------------
# 面板真值批：链读数 vs 独立真值
# ---------------------------------------------------------------------------
#
# 这一批与上面那条**不是一个东西**：上面那批的常量是从第一批量出来的，这一批是
# **没被动过的干净基线**。这里测的是**读数与真值对得上对不上**，不是链的形状。

TRUTH_BATCH = REPO / "tests" / "帧样本" / "场上干员真值"


def _slot_as_json(slot: SlotValue) -> dict:
    """把槽写成报告里那一套 JSON —— **在测试里再写一遍**。

    这样"报告里的读数是跑出来的"才有意义：重跑一遍、按**另一份**序列化写出同样的形状，
    对不上就说明报告里那个数是抄来的。
    """
    if isinstance(slot, Unsampled):
        return {"state": "unsampled"}
    if isinstance(slot, Valued):
        payload = slot.payload
        if isinstance(payload, Exact):
            body: dict = {"form": "exact", "value": payload.value}
        elif isinstance(payload, Interval):
            body = {"form": "interval", "lo": payload.lo, "hi": payload.hi}
        elif isinstance(payload, LowerBound):
            body = {"form": "lower_bound", "lo": payload.lo}
        elif isinstance(payload, Unknown):
            body = {"form": "unknown"}
        else:
            raise AssertionError(payload)
        return {"state": "valued", "payload": body, "confidence": slot.confidence}
    raise AssertionError(slot)


@pytest.fixture(scope="module")
def truth_harvest() -> dict:
    return json.loads((TRUTH_BATCH / "取材.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def truth_report() -> dict:
    return json.loads((TRUTH_BATCH / "判据读数.json").read_text(encoding="utf-8"))


def test_truth_batch_is_a_clean_version_one() -> None:
    """真值那一批是**没被动过的干净基线**：version 1、画布对得上、13 块。

    （14 块里那一块 `look_now` 按裁定**整个拿掉**了 —— 它是备战席详情页，
    不属于这一批的取帧场景，见下一条。）
    """
    manifest = json.loads((TRUTH_BATCH / "index.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["canvas"] == [CANVAS_2800x1260.width, CANVAS_2800x1260.height]
    assert len(manifest["crops"]) == 13
    for crop in manifest["crops"]:
        x, y = crop["origin"]
        width, height = crop["size"]
        assert x >= 0 and y >= 0
        assert x + width <= CANVAS_2800x1260.width
        assert y + height <= CANVAS_2800x1260.height
        assert width > field.OPERATOR_UI_WIDTH and height > field.OPERATOR_UI_HEIGHT


def test_truth_crops_hold_their_window(truth_harvest) -> None:
    """裁剪**比取区大一圈**：取区整个落在块里（否则测的就不是开窗那一段）。"""
    for item in truth_harvest["crops"]:
        assert item["window"] == [
            item["bar"][0] - field.BARS_IN_WINDOW[0],
            item["bar"][1] - field.BARS_IN_WINDOW[1],
            field.OPERATOR_UI_WIDTH,
            field.OPERATOR_UI_HEIGHT,
        ]


def test_truth_batch_pairs_the_bar_nearest_the_canvas_centre(truth_harvest) -> None:
    """配对规则是**几何**的：被选中的那根条离画布中心最近 —— 第二名必须比它远。"""
    assert truth_harvest["pairing"]["centre"] == [
        CANVAS_2800x1260.width / 2,
        CANVAS_2800x1260.height / 2,
    ]
    for item in truth_harvest["crops"]:
        assert item["distance_to_centre"] > 0
        assert item["runner_up"] is None or (
            item["distance_to_centre"] < item["runner_up"]["distance_to_centre"]
        )
        assert item["candidates"][0]["bar"] == item["bar"]
        assert item["candidates"][0]["distance_to_centre"] == item["distance_to_centre"]


def test_truth_records_where_every_number_was_read(truth_harvest) -> None:
    """真值**从画布上哪一块读的**要记下来 —— 否则"读的是面板那一行"是一句空话。"""
    for item in truth_harvest["crops"]:
        x0, y0, x1, y1 = item["truth_row"]
        assert 0 <= x0 < x1 <= CANVAS_2800x1260.width
        assert 0 <= y0 < y1 <= CANVAS_2800x1260.height
        life = item["truth"]["life"]
        assert 0 < life["current"] <= life["max"]
        assert life["ratio"] == round(life["current"] / life["max"], 6)


def test_truth_readings_are_re_derivable_from_the_batch(truth_harvest, truth_report) -> None:
    """报告里的读数是**跑出来的**：拿这一批重跑一遍，逐帧对得上。"""
    windows = {item["file"]: item["window"] for item in truth_harvest["crops"]}
    template = field.load_ready_diamond()
    recorded = {frame["name"]: frame for frame in truth_report["frames"]}
    seen = set()
    for case, raw in replay_frames(load_batch(TRUTH_BATCH)):
        frame = frame_from_raw(raw)
        window = PixelRect(*windows[case.name + ".png"])
        row = recorded[case.name]
        assert row["health"] == _slot_as_json(field.read_health(frame, window))
        fill, meaning = field.read_skill(frame, window)
        assert row["skill_fill"] == _slot_as_json(fill)
        assert row["skill_meaning"] == _slot_as_json(meaning)
        assert row["sp"] == _slot_as_json(field.sp_of(fill, meaning))
        assert row["diamond"] == _slot_as_json(field.find_ready_diamond(frame, window, template))
        seen.add(case.name)
    assert seen == set(recorded)


def test_truth_errors_are_computed_from_the_readings(truth_report) -> None:
    """每一帧的误差 = 链读数 − 真值；分布是从逐帧误差算出来的。"""
    errors = []
    for frame in truth_report["frames"]:
        ratio = frame["truth"]["life"]["ratio"]
        value = frame["health"]["payload"]["value"]
        assert frame["health_error"] == round(value - ratio, 6)
        assert frame["health_error_px"] == round(frame["health_error"] * field.BAR_WIDTH, 3)
        errors.append(frame["health_error"])
    stats = truth_report["health_errors"]
    assert stats["count"] == len(errors)
    assert stats["min"] == round(min(errors), 6)
    assert stats["max"] == round(max(errors), 6)
    assert stats["median"] == round(sorted(errors)[len(errors) // 2], 6)
    assert ("双边" in stats["sided"]) or ("单边" in stats["sided"])


def test_truth_batch_dropped_the_bench_detail_frame(truth_report, truth_harvest) -> None:
    """**备战席详情页那一帧整个不在这一批里**（裁定）：配对规则**管不到它**。

    点场上干员开面板时，被点中的干员会被挪到屏幕较中心的位置 —— 「离画布中心最近的
    那根 = 被选中的那根」靠的是这条。**点备战席是另一回事**：整个场景朝固定的右上方
    挪，与被点的卡在哪无关，所以那一帧上按这条规则挑出来的东西根本不是被选中的那根。

    它**不是"配对可疑"**，是**规则不适用于这一档场景** —— 所以没有"另列一列剔掉"，
    而是**整个拿掉**：留着会让人以为配对只是可疑。
    """
    names = [frame["name"] for frame in truth_report["frames"]]
    assert names == [f"p{index:02d}" for index in range(12)] + ["sp_now"]
    assert [item["file"] for item in truth_harvest["crops"]] == [name + ".png" for name in names]
    assert not (TRUTH_BATCH / "look_now.png").exists()
    assert "look_now" not in json.dumps(truth_report, ensure_ascii=False)
    assert "备战席" in truth_report["coverage"]["pairing_caveat"]
    assert truth_report["health_errors"]["count"] == 13


def test_truth_batch_sp_side_is_the_button_reading(truth_report) -> None:
    """技力那一侧的真值是**场上那个技能按钮上的 N/M**（这一批只有一帧有）。"""
    with_sp = [frame for frame in truth_report["frames"] if frame["truth"]["sp"] is not None]
    assert [frame["name"] for frame in with_sp] == ["sp_now"]
    assert with_sp[0]["truth"]["sp"] == {"current": 16, "max": 26, "ratio": round(16 / 26, 6)}
    assert truth_report["sp_errors"]["count"] == 1
    assert truth_report["observed"]["sp_frame"]["name"] == "sp_now"


def test_truth_batch_says_what_it_did_not_touch(truth_report) -> None:
    """这一批**一帧都没参与定参** —— 这句话要写在报告里，否则这一批就不独立了。

    而且要说清楚：**这一版链在别处改过了**（技力那一侧的填充判据与 `GREEN_MAX_RG`
    是拿 `data/probe/sp_series` 那五帧定的），这一批的读数**在那之后重跑**了一遍 ——
    所以它对"**下一次**"改动才重新算"没见过"。这句不许省，省了就成了假话。
    """
    text = truth_report["observed"]["what_i_did_not_do"]
    assert "一帧都没有参与定参" in text
    assert "重跑" in text
    assert "只进对照" in truth_report["hard_rule"]


def test_truth_report_is_byte_stable(truth_report) -> None:
    """同一批跑两次，报告逐字节一样。"""
    first = (TRUTH_BATCH / "判据读数.json").read_bytes()
    again = (json.dumps(truth_report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    assert first == again


def test_the_chains_do_not_know_about_this_batch() -> None:
    """链那一层**不知道真有这一批**：field.py 里不许出现这一批的名字与帧名。

    **回归批是反过来的**：技力那一侧换判据的**理由**就写在 field.py 里，它引的是**源帧**
    （`data/probe/sp_series`）—— 与第一批里引 `shot30` / `cd01` 同一个做法。所以这一条只管
    真值那一批；另外多钉一条：链**也不许知道回归批那个目录**（不许把夹具当依赖）。
    """
    source = (REPO / "src" / "ark" / "engine" / "vision" / "field.py").read_text(encoding="utf-8")
    for token in (
        "场上干员真值",
        "panel_run",
        "sp_now",
        "look_now",
        "TRUTH_PICKS",
        "场上干员回归",
        "REGRESSION_PICKS",
        "REGRESSION_READINGS",
    ):
        assert token not in source, token


# ---------------------------------------------------------------------------
# 回归批：亮底那五帧的读数**一字不许变**
# ---------------------------------------------------------------------------
#
# 它是**回归批**，不是干净基线：技力那一侧的判据就是拿这五帧的现象改的，所以它见过。
# 这里钉的是**链的读数**（不是真值）—— 真值住在规划者那边，这一批一个真值字段都没有。

REGRESSION_BATCH = REPO / "tests" / "帧样本" / "场上干员回归"

REGRESSION_READINGS: dict[str, tuple[float, str]] = {
    "s00": (11 / field.BAR_WIDTH, "生效中"),
    "s01": (76 / field.BAR_WIDTH, "充能中"),
    "s02": (127 / field.BAR_WIDTH, "充能中"),
    "s03": (127 / field.BAR_WIDTH, "充能中"),
    "s04": (32 / field.BAR_WIDTH, "充能中"),
}
"""这五帧的**冻结读数**（帧名 → (比例, 含义)）。

**它们是链的输出，不是真值。** `s02` / `s03` 那两个 127 规划者独立核过。
以后谁再动技力那一侧，这五个数**一字不许变**；要变就得先说清楚为什么 ——
这一批**不是**"没见过"的独立判据，**改它是可以的，但不许悄悄改**。
"""


@pytest.fixture(scope="module")
def regression_harvest() -> dict:
    return json.loads((REGRESSION_BATCH / "取材.json").read_text(encoding="utf-8"))


def test_regression_batch_carries_no_truth(regression_harvest) -> None:
    """回归批**一个真值字段都没有**：只有取区、来源与「这一帧的底是什么」。"""
    assert set(regression_harvest) == {
        "note",
        "window_size",
        "bars_in_window",
        "crop_size",
        "crop_left_margin",
        "crops",
    }
    for item in regression_harvest["crops"]:
        assert set(item) == {"file", "source", "bar", "window", "window_from", "note"}, item
    assert not (REGRESSION_BATCH / "判据读数.json").exists()


def test_regression_crops_hold_their_window(regression_harvest) -> None:
    """取区与另外两批同一把折法：原点 = 血条左上角 − `BARS_IN_WINDOW`，块比取区大一圈。"""
    for item in regression_harvest["crops"]:
        assert item["window"] == [
            item["bar"][0] - field.BARS_IN_WINDOW[0],
            item["bar"][1] - field.BARS_IN_WINDOW[1],
            field.OPERATOR_UI_WIDTH,
            field.OPERATOR_UI_HEIGHT,
        ]


def test_regression_readings_are_frozen(regression_harvest) -> None:
    """这五帧的读数与 `REGRESSION_READINGS` **逐帧对得上** —— 这就是那句话的载体。"""
    windows = {item["file"]: item["window"] for item in regression_harvest["crops"]}
    seen = set()
    for case, raw in replay_frames(load_batch(REGRESSION_BATCH)):
        frame = frame_from_raw(raw)
        window = PixelRect(*windows[case.name + ".png"])
        fill, meaning = field.read_skill(frame, window)
        value, label = REGRESSION_READINGS[case.name]
        assert fill == Valued(Exact(value), None), case.name
        assert isinstance(meaning, Valued) and meaning.payload == Exact(label), case.name
        seen.add(case.name)
    assert seen == set(REGRESSION_READINGS)
