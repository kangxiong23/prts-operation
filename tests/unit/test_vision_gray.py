"""备战席第三态（缺费 / 灰卡）的单元测试（第八单）。

**它钉六件事**：灰罩是**整片乘 0.5**（不是抹平） · **Algebra 是判据、像素只做兜底** ·
**自比**两簇（没变 1.0000 / 一净一灰 ≈ 0.51）与两条线 · **还原**把身份区从平均差 89.6
拉回到 2.0 · **缺费卡走的是原来那条线**（判据一个数没动） · 夹具的代数标签与判据逐格一致。

重活（整帧、10 格 × 10 条库）只跑一次，装在模块级的夹具里。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ark.contracts import Exact, Unknown, Unsampled, Valued, squad_card_face_rect
from ark.engine.device.adb import SQUAD_CARD_PITCH as PITCH
from ark.engine.vision import gray
from ark.engine.vision.bench import squad_card_rect
from ark.engine.vision.frame import Frame, frame_from_raw
from ark.engine.vision.roster import (
    ROSTER_SOLID_MATCH_SCORE,
    identity_is_exact,
    read_squad_identities,
    roster_scores,
)
from ark.offline.bench.batch import load_batch

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "tests" / "帧样本" / "缺费"
INDEX = json.loads((BATCH / "index.json").read_text(encoding="utf-8"))
LABELS = INDEX["labels"]
GRAY_NAME = "灰罩卡_备战席"
CLEAR_NAME = "干净卡对照_备战席"
CURRENT = LABELS["current_cost"]["灰罩卡.png"]
COSTS = LABELS["card_cost_right_to_left"]
VERDICTS = LABELS["insufficient_right_to_left"]
IDENT = (52, 76, 100, 100)


def _frames() -> dict[str, Frame]:
    global _FRAMES
    if _FRAMES is None:
        _FRAMES = {c.name: frame_from_raw(c.raw()) for c in load_batch(BATCH).cases}
    return _FRAMES


_FRAMES: dict[str, Frame] | None = None


def _faces() -> list[tuple[str, np.ndarray]]:
    """库 = **对照帧那一帧自己的 10 张卡面**，名字是卡位 —— 这样比的是"灰罩掉了多少"。"""
    global _FACES
    if _FACES is None:
        pixels = _frames()[CLEAR_NAME].pixels
        out = []
        for i in range(len(COSTS)):
            rect = squad_card_face_rect(i, PITCH)
            out.append(("卡位%d" % i,
                        pixels[int(rect.y):int(rect.y) + int(rect.height),
                               int(rect.x):int(rect.x) + int(rect.width)].copy()))
        _FACES = out
    return _FACES


_FACES: list[tuple[str, np.ndarray]] | None = None


def _ident_pixels(frame: Frame, index: int) -> np.ndarray:
    card = squad_card_rect(index, PITCH)
    x = int(card.x) + IDENT[0]
    y = int(card.y) + IDENT[1]
    return frame.pixels[y:y + IDENT[3], x:x + IDENT[2]]


def _exact_int(value) -> int:
    return value.payload.value


def _ratio(now: np.ndarray, before: np.ndarray) -> float:
    """逐像素 G/C 的中位（只看对照帧那边够亮的像素），与秤上那一段同一个口径。"""
    c = before.astype(np.float64)
    g = now.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(c.mean(axis=2, keepdims=True) >= 60, g / np.maximum(c, 1.0), np.nan)
    return float(np.nanmedian(r))


def _top(region: np.ndarray, faces):
    ranked = roster_scores(np.ascontiguousarray(region), faces)
    return ranked[0][0], ranked[0][1], (ranked[1][0] if len(ranked) > 1 else None)


def test_restore_multiplies_by_two_and_clips_at_255() -> None:
    """还原 = 乘回去（`GRAY_LIFT`），并夹到 0…255；**不改调用方那块数组**。"""
    pixels = np.array([[[0, 100, 128], [200, 255, 64]]], dtype=np.uint8)
    out = gray.restore(pixels)
    assert out.dtype == np.uint8
    assert out.tolist() == [[[0, 200, 255], [255, 255, 128]]]
    assert pixels.tolist() == [[[0, 100, 128], [200, 255, 64]]], "入参不许被就地改"


def test_restore_refuses_anything_that_is_not_an_rgb_region() -> None:
    """形状闸门照 `roster._guard_region`：要 `(高, 宽, 3)` 的 uint8 RGB。"""
    with pytest.raises(TypeError):
        gray.restore(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(TypeError):
        gray.restore(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(TypeError):
        gray.restore([0, 1, 2])


def test_insufficient_cost_reads_the_algebra_when_both_numbers_are_exact() -> None:
    """**代数才是口径**：`卡费 > 当前费`。相等不算缺费，少一费也不算。"""
    exact = lambda n: Valued(Exact(n), 0.99)  # noqa: E731
    assert gray.insufficient_cost(exact(21), exact(20)) is True
    assert gray.insufficient_cost(exact(20), exact(20)) is False
    assert gray.insufficient_cost(exact(19), exact(20)) is False
    assert gray.insufficient_cost(exact(1), exact(99)) is False


def test_insufficient_cost_ignores_the_share_while_the_algebra_can_answer() -> None:
    """两个读数都在时**不看 share**：像素那一边没有资格推翻一个读出来的数。"""
    faces = Valued(Exact(19), 0.99)
    current = Valued(Exact(20), 0.99)
    assert gray.insufficient_cost(faces, current, share=0.50) is False
    assert gray.insufficient_cost(Valued(Exact(21), 0.9), current, share=1.00) is True


def test_insufficient_cost_falls_back_to_the_self_share() -> None:
    """有任一读数不是精确时才问 share —— 它同时是"兜底"与"验证"。

    区间 / 未知 / 未采样都**不是**一个可以被比出来的数：那时只能看自比。
    """
    missing = Valued(Unknown(), None)
    current = Valued(Exact(20), 0.99)
    assert gray.insufficient_cost(missing, current, share=0.5055) is True
    assert gray.insufficient_cost(missing, current, share=1.0000) is False
    assert gray.insufficient_cost(missing, current, share=0.80) is None, "空档里判不了"
    assert gray.insufficient_cost(Unsampled(), current, share=0.55) is True
    assert gray.insufficient_cost(missing, current) is None, "没有 share 就只能判不了"
    assert gray.insufficient_cost(missing, Unsampled(), share=0.40) is True


def test_own_share_is_exactly_one_where_the_two_frames_agree() -> None:
    """1–5 号两帧状态没变 ⇒ 自比 **1.0000**（两帧逐位相同，不是"接近 1"）。"""
    frames = _frames()
    for index in range(len(COSTS) - 1, len(COSTS) - 6, -1):
        assert not VERDICTS[index], "这一组是零假设：裁决不该说缺费"
        share = gray.own_share(_ident_pixels(frames[GRAY_NAME], index),
                               _ident_pixels(frames[CLEAR_NAME], index))
        assert share == 1.0, (index, share)


def test_own_share_separates_the_two_clusters_by_a_wide_gap() -> None:
    """6–10 号：自比 **0.50 上下**；两簇之间是 `GRAY_SHARE_MAX`…`CLEAR_SHARE_MIN` 那条空档。"""
    frames = _frames()
    for index in range(5):
        assert VERDICTS[index], "这一组是效应组"
        share = gray.own_share(_ident_pixels(frames[GRAY_NAME], index),
                               _ident_pixels(frames[CLEAR_NAME], index))
        assert 0.45 <= share <= 0.60, (index, share)
        assert share <= gray.GRAY_SHARE_MAX < gray.CLEAR_SHARE_MIN
        assert gray.insufficient_cost(
            Valued(Unknown(), None), Valued(Exact(CURRENT), None), share=share
        ) is True


def test_the_fixture_labels_are_exactly_what_the_criterion_says() -> None:
    """夹具里那串代数标签与判据**逐格一致**：标签不是抄来的，是同一个算式给的。"""
    current = Valued(Exact(CURRENT), 0.99)
    for index, cost in enumerate(COSTS):
        verdict = gray.insufficient_cost(Valued(Exact(cost), 0.99), current)
        assert verdict is VERDICTS[index], (index, cost, verdict)
    assert VERDICTS == [True] * 5 + [False] * 5
    assert COSTS == [25, 23, 22, 21, 21, 19, 16, 13, 12, 11]


def test_the_two_frames_differ_only_in_the_insufficient_slots() -> None:
    """**零假设组逐位相同、效应组整片不同** —— 这一对图就只有那一个变量。"""
    frames = _frames()
    for index in range(len(COSTS)):
        g = _ident_pixels(frames[GRAY_NAME], index).astype(np.int32)
        c = _ident_pixels(frames[CLEAR_NAME], index).astype(np.int32)
        differing = int((np.abs(g - c).max(axis=2) > 0).sum())
        if VERDICTS[index]:
            assert differing > 9000, (index, differing)
        else:
            assert differing == 0, (index, differing)


def test_the_grey_multiplies_the_card_instead_of_flattening_its_colour() -> None:
    """**三个模型排到只剩一个**：色度被乘小（≈0.50），不是被抹平，亮度同乘同一个标量。

    抹平的话灰罩卡的身份区该没有色度 —— 实测它还留着成千上万个 `chr > 20` 的像素。
    """
    frames = _frames()
    for index in range(5):
        g = _ident_pixels(frames[GRAY_NAME], index).astype(np.float64)
        c = _ident_pixels(frames[CLEAR_NAME], index).astype(np.float64)
        chroma_g = (g.max(axis=2) - g.min(axis=2))
        chroma_c = (c.max(axis=2) - c.min(axis=2))
        assert np.median(chroma_g) / max(np.median(chroma_c), 1e-9) <= 0.60
        assert int((chroma_g > 20).sum()) >= 20, (index, int((chroma_g > 20).sum()))
        luma_g = float((0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]).mean())
        luma_c = float((0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]).mean())
        assert abs(luma_g / luma_c - gray.GRAY_SHARE) <= 0.02, (index, luma_g / luma_c)
        assert abs(_ratio(g, c) - gray.GRAY_SHARE) <= 0.02, (index, _ratio(g, c))


def test_restoring_brings_the_grey_card_back_to_the_clean_one() -> None:
    """**还原的落点是"回到和干净卡一样"**：平均差 89.6 → 2.0（逐位那点残差是加法项）。"""
    frames = _frames()
    before, after = [], []
    for index in range(5):
        g = _ident_pixels(frames[GRAY_NAME], index)
        c = _ident_pixels(frames[CLEAR_NAME], index).astype(np.int32)
        before.append(float(np.abs(g.astype(np.int32) - c).mean()))
        after.append(float(np.abs(gray.restore(g).astype(np.int32) - c).mean()))
    assert min(before) > 70.0, before
    assert max(after) < 3.0, after
    assert float(np.mean(after)) < float(np.mean(before)) / 30.0


def test_the_insufficient_slots_go_through_the_same_line() -> None:
    """**同一条链、同一条线**（用户裁定 4）：还原只改像素，判定与名字一个都不许变。

    右边是"不还原"、左边是"还原"，两者的（判定, 名字）必须逐格相同；而且那 10 格
    的名字都必须是**它自己那个卡位** —— 灰罩没有把任何一格挤到别人身上去。
    """
    frames = _frames()
    faces = _faces()
    restored = read_squad_identities(
        frames[GRAY_NAME], faces, pitch=PITCH,
        insufficient=list(VERDICTS), trace=(trace := []),
    )
    plain = read_squad_identities(frames[GRAY_NAME], faces, pitch=PITCH)
    assert [t.get("gray") for t in trace] == list(VERDICTS), "足迹要记下这一格还原没还原"
    for index in range(len(COSTS)):
        assert _exact_int_or_none(restored[index]) == _exact_int_or_none(plain[index])
        assert _exact_int_or_none(restored[index]) == "卡位%d" % index


def test_the_line_is_not_relaxed_for_the_insufficient_cards() -> None:
    """**不许给缺费开一条更宽的线**：还原之后那几格走的就是 `ROSTER_SOLID_MATCH_SCORE`
    加余量公式，而且分数离干净那一侧很近（实测掉 0.0013–0.0045）。"""
    frames = _frames()
    faces = _faces()
    for index in range(5):
        region = _ident_pixels(frames[GRAY_NAME], index)
        clean = _ident_pixels(frames[CLEAR_NAME], index)
        top_g, name_g, runner_g = _top(region, faces)
        top_c, name_c, runner_c = _top(clean, faces)
        assert name_g == name_c == "卡位%d" % index
        assert identity_is_exact(top_g, runner_g) is True
        assert top_g >= ROSTER_SOLID_MATCH_SCORE
        assert 0.0 <= top_c - top_g <= 0.01, (index, top_c, top_g)
        assert identity_is_exact(top_g, runner_g) == identity_is_exact(top_c, runner_c)


def test_insufficient_wants_exactly_one_verdict_per_slot() -> None:
    """裁决是按卡位给的：少一个多一个都说明调用方把别的帧的裁决递进来了。"""
    frames = _frames()
    faces = _faces()
    with pytest.raises(ValueError):
        read_squad_identities(frames[GRAY_NAME], faces, pitch=PITCH, insufficient=[True])
    with pytest.raises(TypeError):
        read_squad_identities(frames[GRAY_NAME], faces, pitch=PITCH, insufficient="缺费")
    with pytest.raises(TypeError):
        read_squad_identities(frames[GRAY_NAME], faces, pitch=PITCH,
                              insufficient=[1] * len(COSTS))
    assert len(read_squad_identities(frames[GRAY_NAME], faces, pitch=PITCH,
                                     insufficient=[None] * len(COSTS))) == len(COSTS)


def _exact_int_or_none(value) -> str | None:
    payload = getattr(value, "payload", None)
    if payload is None or type(payload).__name__ != "Exact":
        return None
    return payload.value
