"""红罩反解那一单（判据）的单元测试。

**不 import tests/spec**（D-全局-17）。这里钉四件事：

1. **配对判据**（`pair_ok`）—— 凭什么说两块是同一块画面，以及它拒什么；
2. **模型**（`fit_redcap` / `undo_redcap`）—— 几个参数、残差多大、能不能存下来；
3. **够不够** —— 反解后的卡喂**同一条链**（`roster_scores`），真匹配那一簇抬到哪儿；
4. **新取区**（`margin.identity_disc_*`）—— 「环带以内的圆盘 − 中央秒数那一条」那个
   形状、它取的那个值、以及那个值离环带的余量（读数的铺开在
   `tests/帧样本/预处理/圆盘取区.json`，那一项的网在规划者那一侧）。

## 标定与验收不相交

`fit_redcap` 只吃**标定**那两对；验收那两对只在最后读数一次，**没有回头调过参数**。

## 判据与模型的关系要记着

`pair_ok` 的第 4 条（那个卡位自己的仿射残差 ≤ 12）与模型**同形** —— 它挡得住
「明显不是同一块画面」，但它**不是**「罩子可解」的独立证据。那一条只有「一批配对
落在一组四参数上」才谈得上，所以下面分开断言：判据那条只钉**拒得掉什么**。
"""

from __future__ import annotations

import json
import tempfile
import uuid
import shutil
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from ark.contracts import squad_card_rect
from ark.engine.device.adb import SQUAD_CARD_PITCH
from ark.engine.vision.bench import (
    COUNTDOWN_MASK_RADIUS,
    RING_CENTER_DY,
    squad_card_centres,
)
from ark.engine.vision.frame import Frame, frame_from_raw
from ark.engine.vision.margin import (
    CARD_HEIGHT,
    CARD_WIDTH,
    IDENTITY_DISC_BOX,
    IDENTITY_DISC_RADIUS,
    IDENTITY_DISC_STRIP_HALF,
    MaskedRuler,
    free_mask,
    identity_disc_region,
)
from ark.engine.vision.redcap import (
    TOP_STRIP_HEIGHT,
    Redcap,
    _countdown_mask,
    fit_redcap,
    undo_redcap,
)
from ark.engine.vision.roster import (
    IDENTITY_BOX_HEIGHT,
    IDENTITY_BOX_WIDTH,
    IDENTITY_BOX_X,
    IDENTITY_BOX_Y,
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
    identity_region,
    roster_scores,
)
from ark.offline.assets_gen.cardface import load_cardfaces
from ark.offline.bench.batch import load_batch

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "tests" / "帧样本" / "红罩"
HARVEST = BATCH / "取材.json"
PRE_BATCH = REPO / "tests" / "帧样本" / "预处理"
PRE_HARVEST = PRE_BATCH / "取材.json"
"""**新取区**那三条用例用的那一批：44 对（39 张裁剪、19 张互不相同的罩着卡）。
取区那件事与红罩这 8 帧无关，所以在那边另开一批。
"""

PITCH = SQUAD_CARD_PITCH
"""卡距。它住设备层，视觉层不许 import 设备层，所以照调用方的样子传进来。"""

CARD = (214, 204, 3)
"""整张备战席卡的形状 —— 配对与模型都吃这个形状的块。"""


def _affine(capped: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """`capped = slope * clean + offset` 的逐通道斜率与**最大**通道残差 rms。"""
    a = clean[mask].reshape(-1, 3).astype(np.float64)
    b = capped[mask].reshape(-1, 3).astype(np.float64)
    slopes = []
    worst = 0.0
    for channel in range(3):
        x = a[:, channel]
        y = b[:, channel]
        x_mean = x.mean()
        y_mean = y.mean()
        variance = float(((x - x_mean) ** 2).mean())
        if variance <= 1e-9:
            slopes.append(1.0)
            continue
        slope = float(((x - x_mean) * (y - y_mean)).mean() / variance)
        offset = float(y_mean - slope * x_mean)
        slopes.append(slope)
        worst = max(worst, float(np.sqrt(((slope * x + offset - y) ** 2).mean())))
    return np.array(slopes), worst


def pair_ok(
    capped_frame: Frame,
    clean_frame: Frame,
    index: int,
    *,
    pitch: float = PITCH,
    row_rms: float = 3.0,
    card_rms: float = 12.0,
) -> bool:
    """这两帧的**第 index 张卡**，是不是「同一块画面，一次没罩、一次罩着」。

    四条结构条件（都不是「看起来像」）：

    1. **卡数相同**；
    2. **恰有一个卡位**的罩状态不同（就是 `index`）；
    3. **其余每个卡位**只差一次亮度：逐通道仿射，斜率落在 `0.9…1.1`、残差 rms ≤ `row_rms`；
    4. **那个卡位自己**差一次仿射，残差 rms ≤ `card_rms`。

    **罩状态用整块画面的亮度判**：罩子把立绘压掉四分之三，所以「平均亮度 < 90」
    这一条在 61 帧上把两档分得干干净净（实测罩着 55…75、干净 96…176）。
    """
    if len(squad_card_centres(capped_frame, pitch=pitch)) != len(
        squad_card_centres(clean_frame, pitch=pitch)
    ):
        return False
    count = len(squad_card_centres(capped_frame, pitch=pitch))
    if not 0 <= index < count:
        return False
    rect = squad_card_rect(index, pitch)
    x0, y0 = int(rect.x), int(rect.y)

    def block(frame: Frame, at: int) -> np.ndarray:
        card = squad_card_rect(at, pitch)
        return frame.pixels[
            int(card.y) : int(card.y) + CARD[0], int(card.x) : int(card.x) + CARD[1]
        ]

    mask = _countdown_mask(CARD)
    rows = [i for i in range(count) if i != index]
    toggled = [
        i
        for i in range(count)
        if (block(capped_frame, i)[47:].mean() < 90) != (block(clean_frame, i)[47:].mean() < 90)
    ]
    if toggled != [index]:
        return False
    for other in rows:
        slopes, rms = _affine(block(capped_frame, other), block(clean_frame, other), mask)
        if rms > row_rms:
            return False
        if not np.all((slopes >= 0.9) & (slopes <= 1.1)):
            return False
    _, own = _affine(block(capped_frame, index), block(clean_frame, index), mask)
    return own <= card_rms


@pytest.fixture(scope="module")
def frames() -> dict[str, Frame]:
    """红罩那 8 帧（贴回画布之后）。"""
    return {case.name: frame_from_raw(case.raw()) for case in load_batch(BATCH).cases}


@pytest.fixture(scope="module")
def faces() -> tuple[tuple[str, np.ndarray], ...]:
    """**库**：编队页那一帧切出来、配好名字的 11 块立绘（借 `点名/编队页.json`）。"""
    return load_cardfaces()


@pytest.fixture(scope="module")
def pairs() -> list[dict]:
    """`取材.json` 里那 4 对。"""
    return json.loads(HARVEST.read_text(encoding="utf-8"))["pairs"]


def _card(frame: Frame, index: int, pitch: float = PITCH) -> np.ndarray:
    rect = squad_card_rect(index, pitch)
    return np.array(
        frame.pixels[
            int(rect.y) : int(rect.y) + CARD[0], int(rect.x) : int(rect.x) + CARD[1]
        ]
    )


def _reading(frame: Frame, index: int, faces) -> tuple[float, str, float]:
    """那张卡的身份区喂**同一条链**（`roster_scores`）：最高分 · 名字 · 次高分。"""
    centre = squad_card_centres(frame, pitch=PITCH)[index][0]
    region = np.ascontiguousarray(frame.crop(identity_region(centre, pitch=PITCH)))
    scored = roster_scores(region, faces)
    runner = scored[1][0] if len(scored) > 1 else -2.0
    return scored[0][0], scored[0][1], runner


def test_the_batch_loads_as_a_bench_batch() -> None:
    """新样本进版本库、能被 `load_batch` 那套路子加载。"""
    batch = load_batch(BATCH)
    assert batch.version == 1
    assert (batch.canvas.width, batch.canvas.height) == (2800, 1260)
    assert batch.facts is None
    assert len(batch.cases) == 8
    assert all(case.rect.width == 2500 and case.rect.height == 220 for case in batch.cases)


def test_the_pair_criterion_accepts_the_documented_pairs(pairs, frames) -> None:
    """判据认下 `取材.json` 里那 4 对 —— 逐对跑一遍，不是只看其中一对。"""
    for pair in pairs:
        capped = frames[Path(pair["capped"]["file"]).stem]
        clean = frames[Path(pair["clean"]["file"]).stem]
        assert pair_ok(capped, clean, pair["capped"]["index"]), pair


def test_the_pair_criterion_rejects_frames_with_different_card_counts(frames) -> None:
    """**反例一**：两张卡数不同的帧去配 —— 必须报「不是一对」，不许配出一对来。"""
    assert len(squad_card_centres(frames["cd14"], pitch=PITCH)) == 10
    assert len(squad_card_centres(frames["cd22"], pitch=PITCH)) == 11
    assert not pair_ok(frames["cd14"], frames["cd22"], 1)
    assert not pair_ok(frames["cd22"], frames["cd14"], 1)


def test_the_pair_criterion_rejects_a_card_of_a_different_operator(frames) -> None:
    """**反例二**：拿两张不同干员的卡去配 —— 那个卡位自己的仿射残差超线。"""
    # 条件 4 问的正是这一条：同一块画面的两端只差一层罩子，两位不同干员的两块画面
    # 差的是内容。拿同一帧里两个卡位（同一个人不可能占两格）当那个反例。
    assert len(squad_card_centres(frames["cd59"], pitch=PITCH)) == 11
    mask = _countdown_mask(CARD)
    _, real = _affine(_card(frames["cd37"], 4), _card(frames["cd59"], 4), mask)
    _, wrong = _affine(_card(frames["cd59"], 6), _card(frames["cd59"], 4), mask)
    assert real <= 10.5
    assert wrong > 12.0
    # 帧一级也拒：位 9 在两帧里都是干净的，所以它压根不是「那个卡位」。
    assert not pair_ok(frames["cd37"], frames["cd59"], 9)


def test_the_countdown_mask_leaves_about_half_of_the_card() -> None:
    """掩掉倒计时图形与顶上那一条之后剩多少 —— 整卡与身份区两个数都要报。"""
    mask = _countdown_mask(CARD)
    assert abs(float(mask.mean()) - 0.5212) < 0.005
    box = np.zeros(CARD[:2], bool)
    box[76 : 76 + 100, 52 : 52 + 100] = True
    available = float((mask & box).sum()) / float(box.sum())
    assert abs(available - 0.0748) < 0.005


def test_the_model_has_four_parameters_and_a_stated_residual(pairs, frames) -> None:
    """四个参数、残差多大 —— 两个数都要在模型自己身上读得出来。"""
    calibration = [pair for pair in pairs if pair["set"] == "calibration"]
    blocks = [
        (
            _card(frames[Path(pair["capped"]["file"]).stem], pair["capped"]["index"]),
            _card(frames[Path(pair["clean"]["file"]).stem], pair["clean"]["index"]),
        )
        for pair in calibration
    ]
    model = fit_redcap(blocks)
    assert isinstance(model, Redcap)
    assert model.parameters == 4
    assert model.pairs == 2
    assert 0.1 < model.slope < 0.3
    assert 5.0 < model.residual < 15.0
    assert model.pixels == 2 * int(_countdown_mask(CARD).sum())


def test_the_model_round_trips_through_json(pairs, frames) -> None:
    """模型能存下来、能再喂给下一个入口。"""
    pair = [p for p in pairs if p["set"] == "calibration"][0]
    blocks = [
        (
            _card(frames[Path(pair["capped"]["file"]).stem], pair["capped"]["index"]),
            _card(frames[Path(pair["clean"]["file"]).stem], pair["clean"]["index"]),
        )
    ]
    model = fit_redcap(blocks)
    again = Redcap.from_json(model.to_json())
    assert again == model
    assert again.parameters == 4
    assert np.array_equal(
        undo_redcap(blocks[0][0], model), undo_redcap(blocks[0][0], again)
    )


def test_the_model_refuses_degenerate_input(frames) -> None:
    """没有配对 / 形状不同 / 斜率塌掉 —— 三种都要响亮，不静默给一组数。"""
    with pytest.raises(ValueError, match="一对都没有"):
        fit_redcap([])
    card = _card(frames["cd59"], 4)
    with pytest.raises(ValueError, match="形状不一样"):
        fit_redcap([(card, card[:100])])
    with pytest.raises(TypeError, match="uint8"):
        fit_redcap([(card.astype(np.float32), card)])
    with pytest.raises(ValueError, match="方差为零"):
        fit_redcap([(card, np.zeros_like(card))])
    with pytest.raises(ValueError, match="斜率"):
        fit_redcap([(card, card)])


def test_undo_redcap_refuses_a_foreign_model(frames) -> None:
    """模型闸门：不是 `fit_redcap` 给的那个就报错。"""
    with pytest.raises(TypeError, match="模型"):
        undo_redcap(_card(frames["cd59"], 4), {"slope": 0.2})


def test_undoing_the_cap_with_the_ring_still_in_the_box_reads_low(
    pairs, frames, faces
) -> None:
    """**旧口径**（整块身份区，环与中央数字**还在框里**）：0.35–0.44，离 0.85 远得很。

    **这一条留着的用处是当反例**：它量的**不是**「罩子底下还剩多少信息」，量的是
    「环、数字与反解出来的卡面混在一个框里还剩多少」。拿这条去答《特征库》§1 那张表
    就答错了 —— 正口径在下一条用例（只留排掉倒计时图形之后的那 748 px）。

    **标定只吃标定那两对**；验收那两对在这条用例里只**读数**，不回头调参数。

    **这三个数是第五单重录的**（判据满足即停 + 十八档 + 粗层直接接受）：`取材.json` 里
    记的 `clean_score` / `clean_runner_up` / `capped_score` 是**十一档时代**的读数
    （`0.9921/0.3989/0.3047` 与 `0.9774/0.3656/0.2928`，那一份没有重跑，见
    `../帧样本/余量/README.md` 的备注）。名字与"反解之后排到第一、但过不了线"这几条
    **一个都没变**。
    """
    calibration = [pair for pair in pairs if pair["set"] == "calibration"]
    validation = [pair for pair in pairs if pair["set"] == "validation"]
    model = fit_redcap([
        (
            _card(frames[Path(pair["capped"]["file"]).stem], pair["capped"]["index"]),
            _card(frames[Path(pair["clean"]["file"]).stem], pair["clean"]["index"]),
        )
        for pair in calibration
    ])
    raised = 0
    for pair in validation:
        capped_frame = frames[Path(pair["capped"]["file"]).stem]
        clean_frame = frames[Path(pair["clean"]["file"]).stem]
        index = pair["capped"]["index"]
        capped_block = _card(capped_frame, index)
        solved = undo_redcap(capped_block, model)
        # 反解出来的那块，要按**同一个身份区**取；身份区在卡内的偏移是固定的。
        rect = squad_card_rect(index, PITCH)
        centre = squad_card_centres(capped_frame, pitch=PITCH)[index][0]
        region = identity_region(centre, pitch=PITCH)
        ox, oy = int(region.x) - int(rect.x), int(region.y) - int(rect.y)
        solved_box = np.ascontiguousarray(
            solved[oy : oy + 100, ox : ox + 100]
        )
        capped_score, _, _ = _reading(capped_frame, index, faces)
        clean_score, clean_name, clean_runner = _reading(clean_frame, index, faces)
        solved_ranked = roster_scores(solved_box, faces)
        solved_score, solved_name, solved_runner = (
            solved_ranked[0][0],
            solved_ranked[0][1],
            solved_ranked[1][0],
        )
        key = "%s#%d" % (Path(pair["capped"]["file"]).stem, index)
        recorded = {
            "cd27#6": (0.942234, 0.475289, 0.318362),
            "cd37#4": (0.949650, 0.450198, 0.303134),
        }[key]
        assert clean_score >= ROSTER_SOLID_MATCH_SCORE
        assert clean_name == pair["under_the_cap"]
        assert abs(clean_score - recorded[0]) < 5e-4
        assert abs(clean_runner - recorded[1]) < 5e-4
        assert abs(capped_score - recorded[2]) < 5e-4
        assert capped_score < ROSTER_SOLID_MATCH_SCORE
        # 反解：名字排到了第一，而且比罩着的时候高。
        assert solved_name == pair["under_the_cap"]
        assert solved_score > capped_score
        # 但**这个口径下**过不了线 —— 因为环与数字还在框里，与「罩子底下剩多少」无关。
        assert solved_score < ROSTER_SOLID_MATCH_SCORE
        assert solved_score < 0.60
        raised += 1
    assert raised == 2


def test_undoing_the_cap_on_the_masked_part_of_the_box(pairs, frames, faces) -> None:
    """**正口径**：只拿身份区里**排掉倒计时图形之后**的那一块（748 px，7.48%）去比。

    环与中央数字**逐帧在变**、不归罩子管（罩子是静态的），所以它们不该进这个读数。
    排掉之后四对是 **0.8673 / 0.9475 / 0.9513 / 0.9598**（旧口径是 0.3512 / 0.3824 /
    0.4387 / 0.3609），名字四对全对。

    **这四个数是第五单重录的**：尺度区间从十一档（`1.05…1.15`）扩到十八档
    （`1.00…1.17`）之后，尺子滑的范围大了，**分基本没动**（`0.9476 → 0.9475`、
    `0.8677 → 0.8673`、`0.9598 → 0.9598`、`0.9513 → 0.9513`），**余量动了**
    （`0.3456 → 0.3096`、`0.2887 → 0.2843`、`0.3496 → 0.3499`、`0.4078 → 0.4074`）——
    次高在低档上找回了一点分。

    **「够 / 不够」那一格不是这条用例的事**（那是《特征库》§1 那张表的），这里只钉数。
    """
    calibration = [pair for pair in pairs if pair["set"] == "calibration"]
    model = fit_redcap(
        [
            (
                _card(frames[Path(pair["capped"]["file"]).stem], pair["capped"]["index"]),
                _card(frames[Path(pair["clean"]["file"]).stem], pair["clean"]["index"]),
            )
            for pair in calibration
        ]
    )
    box = (IDENTITY_BOX_X, IDENTITY_BOX_Y, IDENTITY_BOX_WIDTH, IDENTITY_BOX_HEIGHT)
    mask = free_mask(box)
    assert int(mask.sum()) == 748
    ruler = MaskedRuler(mask, faces)
    measured = {}
    for pair in pairs:
        index = pair["capped"]["index"]
        capped_frame = frames[Path(pair["capped"]["file"]).stem]
        solved = undo_redcap(_card(capped_frame, index), model)
        rect = squad_card_rect(index, PITCH)
        centre = squad_card_centres(capped_frame, pitch=PITCH)[index][0]
        region = identity_region(centre, pitch=PITCH)
        ox, oy = int(region.x) - int(rect.x), int(region.y) - int(rect.y)
        solved_box = np.ascontiguousarray(solved[oy : oy + box[3], ox : ox + box[2]])
        ranked = ruler.scores(solved_box)
        runner = ranked[1][0] if len(ranked) > 1 else -2.0
        measured["%s#%d" % (Path(pair["capped"]["file"]).stem, index)] = (
            ranked[0][0],
            ranked[0][1],
            ranked[0][0] - runner,
        )
    expected = {
        "cd14#1": (0.947492, 0.309561),
        "cd22#9": (0.867343, 0.349946),
        "cd27#6": (0.959767, 0.284315),
        "cd37#4": (0.951324, 0.407378),
    }
    assert set(measured) == set(expected)
    for key, (score, margin) in expected.items():
        got, name, got_margin = measured[key]
        assert got == pytest.approx(score, abs=0.02), key
        assert got_margin == pytest.approx(margin, abs=0.02), key
        assert name == next(p["under_the_cap"] for p in pairs
                            if Path(p["capped"]["file"]).stem + "#" + str(p["capped"]["index"]) == key)
    # 四对**两半都过**。余量线还在 0.30 时第四对（cd27#6，0.2887）只差 0.0113；
    # 线挪到 0.25 之后它也进来了 —— 这一条钉的就是「挪线放回了谁」。
    over = [key for key, (s, _n, m) in measured.items()
            if s >= ROSTER_SOLID_MATCH_SCORE and m >= ROSTER_AMBIGUITY_MARGIN]
    assert len(over) == 4


def test_every_pair_stays_inside_the_criterion_band(pairs, frames) -> None:
    """四对**全部**落在判据那条带里 —— 真配对 8.9–10.1，跨干员 12.2 起。

    这一条把标定那两对也走一遍：结论不能只在验收那一半成立。
    """
    mask = _countdown_mask(CARD)
    for pair in pairs:
        capped_frame = frames[Path(pair["capped"]["file"]).stem]
        clean_frame = frames[Path(pair["clean"]["file"]).stem]
        index = pair["capped"]["index"]
        _, own = _affine(_card(capped_frame, index), _card(clean_frame, index), mask)
        assert own <= 10.5, (pair, own)

@pytest.fixture(scope="module")
def pre_capped() -> dict[str, np.ndarray]:
    """预处理那批里**互不相同的罩着卡**（卡内 204×214，原生像素）。

    取区那件事只用到「罩着的卡」这一侧：中央那一条有多高、环带离环心多远，两个都是
    **画面事实**，与配对无关。这一批 44 对里只有 **19 张**互不相同。
    """
    harvest = json.loads(PRE_HARVEST.read_text(encoding="utf-8"))
    batch = load_batch(PRE_BATCH)
    cards: dict[str, np.ndarray] = {}
    for case in batch.cases:
        frame = frame_from_raw(case.raw())
        x0, y0 = int(case.rect.x), int(case.rect.y)
        cards[case.name] = np.ascontiguousarray(
            frame.pixels[y0 : y0 + CARD[0], x0 : x0 + CARD[1]]
        )
    out: dict[str, np.ndarray] = {}
    for pair in harvest["pairs"]:
        stem = Path(pair["capped"]["file"]).stem
        if stem not in out:
            out[stem] = cards[stem]
    assert len(out) == 19
    return out


def test_the_new_region_is_the_disc_minus_the_seconds_band() -> None:
    """新取区就是**环带以内的圆盘 − 中央秒数那一条**，而且整块落在卡片以内。

    **不拿常量当答案**：期望值在这里**按定义重新算一遍**（圆心在卡内
    `(卡宽 / 2, RING_CENTER_DY)`、半径 `IDENTITY_DISC_RADIUS`、挖掉
    `|行 − 环心| < IDENTITY_DISC_STRIP_HALF` 那一条），再与取区（框 + 掩码）逐像素对上。
    """
    rows = np.arange(CARD_HEIGHT, dtype=np.float64)[:, None]
    columns = np.arange(CARD_WIDTH, dtype=np.float64)[None, :]
    distance = np.sqrt((columns - CARD_WIDTH / 2) ** 2 + (rows - RING_CENTER_DY) ** 2)
    by_definition = (distance <= float(IDENTITY_DISC_RADIUS)) & (
        np.abs(rows - RING_CENTER_DY) >= float(IDENTITY_DISC_STRIP_HALF)
    )
    box, mask = identity_disc_region()
    assert box == IDENTITY_DISC_BOX
    x0, y0, width, height = box
    assert (height, width) == mask.shape
    assert np.array_equal(mask, by_definition[y0 : y0 + height, x0 : x0 + width])
    # 框是**紧**的：四条边上都有取区的像素 —— 不紧就不是包围盒，是块留白
    assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
    # **整块落在卡片以内**，而且不碰顶上那一条（职业图标 + 费用画在罩子上面）
    assert 0 <= x0 and x0 + width <= CARD_WIDTH
    assert 0 <= y0 and y0 + height <= CARD_HEIGHT
    assert y0 >= TOP_STRIP_HEIGHT


def test_the_region_values_are_pinned() -> None:
    """单测**钉住你取的那个值**：半径 47 · 中央条半高 16 · 框 (58, 70, 89, 94) · 3986 px。

    这一条只钉数；那两个值**为什么**是它在 `margin.IDENTITY_DISC_RADIUS` /
    `IDENTITY_DISC_STRIP_HALF` 自己的文档里（逐环的亮度表 · 秒数可见范围那张表），
    **余量**在下面两条用例里各证一半 —— 两条都是数据，不是把常量抄一遍。
    """
    assert (IDENTITY_DISC_RADIUS, IDENTITY_DISC_STRIP_HALF) == (47, 16)
    assert IDENTITY_DISC_BOX == (58, 70, 89, 94)
    assert int(identity_disc_region()[1].sum()) == 3986


def test_the_seconds_band_covers_every_glyph_row_the_batch_shows(pre_capped) -> None:
    """中央那一条挖掉的正是秒数**整段可见范围**（101…132）—— 核心两边各多留一行。

    **拿数据钉，不拿常量钉**：按几个阈值把每张罩着卡上圆盘里的秒数扫一遍。核心
    （`min > 200`）是 **102…130**；阈值放宽到 40…150（卡面自己的暗底 p99 只有 38，
    所以再低就没有边了）时，最外两行伸到 **101** 与 **132**。半高 16 挖 **101…132**；
    半高 15 只挖 102…131 —— 第 101 与第 132 行那两圈抗锯齿的边会留在取区里。
    """
    rows = np.arange(CARD_HEIGHT, dtype=np.float64)[:, None]
    columns = np.arange(CARD_WIDTH, dtype=np.float64)[None, :]
    inside = (
        np.sqrt((columns - CARD_WIDTH / 2) ** 2 + (rows - RING_CENTER_DY) ** 2)
        < float(COUNTDOWN_MASK_RADIUS)
    )
    removed = np.abs(rows - RING_CENTER_DY) < float(IDENTITY_DISC_STRIP_HALF)
    removed_rows = [int(r) for r in np.flatnonzero(removed[:, 0])]
    assert removed_rows == list(range(101, 133))

    def band(threshold: int) -> tuple[int, int]:
        top, bottom = CARD_HEIGHT, -1
        for card in pre_capped.values():
            ink = (card.min(axis=2) > threshold) & inside
            hit = np.flatnonzero(ink.any(axis=1))
            assert hit.size, "这一批每张罩着卡上都该读得到秒数的笔画"
            assert removed[hit].all(), "每一条笔画都该在挖掉的那一段里"
            top, bottom = min(top, int(hit[0])), max(bottom, int(hit[-1]))
        return top, bottom

    core = band(200)
    assert core == (102, 130)
    for threshold in (40, 60, 80, 100, 150):
        assert band(threshold) in ((101, 131), (101, 132))
    # **核心每边各多留一行** —— 这就是半高 16 与 15 的差别
    assert removed_rows[0] == core[0] - 1
    assert removed_rows[-1] == core[1] + 2
    narrower = np.abs(rows - RING_CENTER_DY) < float(IDENTITY_DISC_STRIP_HALF) - 1.0
    assert not bool(narrower[101, 0])
    assert not bool(narrower[132, 0])


def test_the_disc_stops_short_of_the_ring(pre_capped) -> None:
    """半径那一档**一个非暗底像素都没碰**，往上两档才开始碰 —— 余量就是这一档。

    实测（19 张互不相同的罩着卡，中央那一条之外）：环形亮像素（`min(R,G,B) > 200`）
    离环心最近 **49.85**；而**掩码里**最亮的那个像素（中央条半高 16 那一档）在半径
    **47 与 48** 上都是 **38**（就是卡面自己的暗底），**49** 上是 **87**（环的半亮内边），
    **50** 上才见 226（环本体）。所以 47 是「往上一档（48）仍然全干净」的最大一档 ——
    取它靠的是**余量**，不是分数。
    """
    rows = np.arange(CARD_HEIGHT, dtype=np.float64)[:, None]
    columns = np.arange(CARD_WIDTH, dtype=np.float64)[None, :]
    distance = np.sqrt((columns - CARD_WIDTH / 2) ** 2 + (rows - RING_CENTER_DY) ** 2)
    outside = np.abs(rows - RING_CENTER_DY) >= float(IDENTITY_DISC_STRIP_HALF)
    innermost = []
    for card in pre_capped.values():
        bright = (card.min(axis=2) > 200) & outside
        assert bright.any(), "每张罩着卡上都有那圈白环"
        innermost.append(float(distance[bright].min()))
    assert min(innermost) == pytest.approx(49.85, abs=0.02)
    assert max(innermost) == pytest.approx(50.50, abs=0.02)
    # 定下的半径离最近的环形亮像素还有**两档以上**：半径 +2 才把 87 那一圈吃进来
    assert min(innermost) > float(IDENTITY_DISC_RADIUS) + 2.0

    def brightest(radius: int) -> float:
        box, mask = identity_disc_region(radius=radius)
        x0, y0, width, height = box
        return max(
            float(
                card[y0 : y0 + height, x0 : x0 + width]
                .min(axis=2)
                .astype(np.float64)[mask]
                .max()
            )
            for card in pre_capped.values()
        )

    assert brightest(IDENTITY_DISC_RADIUS) == 38.0
    assert brightest(IDENTITY_DISC_RADIUS + 1) == 38.0
    assert brightest(IDENTITY_DISC_RADIUS + 2) == 87.0
    assert brightest(IDENTITY_DISC_RADIUS + 3) >= 200.0
