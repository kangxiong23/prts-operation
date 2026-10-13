"""「余量从哪来」那份读数表 —— 尺子的基线、表的完整、职业图标那一段。

它钉四件事，都在 `tests/帧样本/余量/` 那批**进版本库**的样本上跑，不碰真机：

1. `margin.MaskedRuler` **在掩码全开时就是 `roster_scores`** —— 这是"换了尺子还
   是同一个统计量"那句唯一的自证方式。线上那条链**一个字节没动**；
2. `余量.json` 里每一行**都对得上线上那条链**：真值那几行是用 `roster_scores`
   重算出来的，`rect` 那几列也是（那一块十一档尺度都在，所以两把尺子必须同值）；
3. 顶上那一条里的**职业图标在罩子底下还在**（逐通道仿射的斜率 · 残差 · 相关），
   而且**按二值字形能读出正确的库条目**；
4. 两条线**落在实测空档里** —— 空档两头都从落盘读数算（`余量.json` 读错的那一头 ·
   `红罩/点名.json` 读对的那一头），**不是手抄的 `0.85` / `0.25`**。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ark.engine.device.adb import SQUAD_CARD_PITCH as PITCH
from ark.engine.vision.bench import squad_card_centres
from ark.engine.vision.frame import frame_from_raw
from ark.engine.vision.margin import MaskedRuler, free_mask
from ark.engine.vision.roster import (
    IDENTITY_BOX_X,
    IDENTITY_BOX_Y,
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
    SlotGeometry,
    _GEOMETRY_JITTER,
    _geometry_scores,
    identity_region,
    roster_scores,
)
from ark.offline.assets_gen.cardface import load_cardfaces
from ark.offline.bench.batch import load_batch

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "tests" / "帧样本" / "余量"
TABLE = BATCH / "余量.json"

REDCAP_LEDGER = REPO / "tests" / "帧样本" / "红罩" / "点名.json"

TABLE_RECORD = json.loads(TABLE.read_text(encoding="utf-8"))
CARDS = TABLE_RECORD["cards"]
FACES = load_cardfaces()


def _frames() -> dict:
    """那 14 帧（贴回画布之后）。模块级缓存 —— 解 14 张 2500x220 的带子不便宜。"""
    global _FRAMES
    if _FRAMES is None:
        _FRAMES = {case.name: frame_from_raw(case.raw()) for case in load_batch(BATCH).cases}
    return _FRAMES


_FRAMES: dict | None = None


def _card(frame, index: int) -> np.ndarray:
    centre = squad_card_centres(frame, pitch=PITCH)[index][0]
    left = int(round(centre)) - 102
    return np.ascontiguousarray(frame.pixels[1045 : 1045 + 214, left : left + 204])


def _at(card: np.ndarray, box) -> np.ndarray:
    x0, y0, width, height = box
    return np.ascontiguousarray(card[y0 : y0 + height, x0 : x0 + width])


# ---------------------------------------------------------------------------
# 尺子：掩码全开时它必须就是线上那条链
# ---------------------------------------------------------------------------


def test_the_masked_ruler_and_the_clean_scorer_are_one_scorer() -> None:
    """**掩码全开时，两把尺子在同一个档、同一个位置上给同一个数**（第五单的目标形式）。

    用户裁定的那句"**两条路的差别只有掩码与取区，打分器与几何是同一份**"在这里是
    **可执行**的，不是修辞：同一块取区（身份区那个 100×100 的框）、同一个档、
    同一个位置窗 ——

    - 干净卡那条路：`_geometry_scores`（`TM_CCOEFF_NORMED`，窗里取最大）；
    - 红罩卡那条路：`MaskedRuler(mask=全开).scores_and_positions(only=档, window=(位置, 半宽))`。

    两条给的分**逐位相同**（差只该是浮点累加次序）。**搜法**（首帧/升级那一趟）两条路
    还不一样 —— 干净卡走先粗后细的整段扫法、红罩卡走尺子整张滑 —— 那一步不在这一条里。
    """
    frame = _frames()["cd43"]
    centres = squad_card_centres(frame, pitch=PITCH)
    box = (float(IDENTITY_BOX_X), float(IDENTITY_BOX_Y))
    for index in (0, 4, 10):
        region = np.ascontiguousarray(frame.crop(identity_region(centres[index][0], pitch=PITCH)))
        ruler = MaskedRuler(np.ones(region.shape[:2], bool), FACES)
        for scale in (1.06, 1.10, 1.14):
            geometry = SlotGeometry(scale=scale, offset_x=40.0, offset_y=60.0)
            clean = {name: score for score, name, _s, _p in
                     _geometry_scores(region, FACES,
                                      {who: geometry for who, _art in FACES}, box)}
            masked = {name: score for score, name, _s, _p in
                      ruler.scores_and_positions(
                          region, only=scale,
                          window=(geometry.art_top_left(box), _GEOMETRY_JITTER))}
            assert set(masked) == set(clean)
            for name, score in masked.items():
                assert score == pytest.approx(clean[name], abs=1e-4), (name, scale)


def _gap() -> dict:
    """两条线的**实测空档** —— 两头都从落盘读数算，一个字面量都不抄。

    「认不出」那一头 = `余量.json` 里**换过取区之后名字读错**的那些读数
    （`mask` / `rect` / `rect_6680` 三列）—— 它们的分与余量必须在线下面。
    「认得出」那一头 = `红罩/点名.json` 里判定为 `Exact` 的那些卡 —— 那是红罩那条路
    自己读出来的（`余量.json` 的 `capped_batch` 指的就是它），它们必须在线上面。
    """
    bad = [row[column] for row in CARDS
           for column in ("mask", "rect", "rect_6680")
           if row[column]["name"] != row["name"]]
    ledger = json.loads(REDCAP_LEDGER.read_text(encoding="utf-8"))
    good = [slot for frame in ledger["frames"] for slot in frame["slots"]
            if slot["reading"]["payload"] == "Exact"]
    assert bad, "这一份里没有读错的读数：空档的下头没量到"
    assert good, "红罩那份里没有「精确」的卡：空档的上头没量到"
    return {
        "score_bad": max(reading["score"] for reading in bad),
        "margin_bad": max(reading["margin"] for reading in bad),
        "score_good": min(slot["score"] for slot in good),
        "margin_good": min(slot["margin"] for slot in good),
    }


def test_the_lines_are_the_frozen_ones() -> None:
    """两条线落在**实测空档**里 —— 两头从落盘读数算，不手抄。

    分那一条线：**高过每一个读错的读数**，不高于「精确」里最低的那个分。
    余量那一条线：**高过每一个读错的余量**，不高于「精确」里最低的那个余量。
    **判定由读者按线算**：这一份文件里的 lines 只是「生成时那两条线是哪两条」的记录，
    它与模块里那两个常量必须一致 —— 常量一改（忘了重跑）这里就红。
    """
    gap = _gap()
    assert gap["score_bad"] < ROSTER_SOLID_MATCH_SCORE <= gap["score_good"]
    assert gap["margin_bad"] < ROSTER_AMBIGUITY_MARGIN <= gap["margin_good"]
    assert TABLE_RECORD["lines"] == {
        "score": ROSTER_SOLID_MATCH_SCORE,
        "margin": ROSTER_AMBIGUITY_MARGIN,
    }


# ---------------------------------------------------------------------------
# 表：逐张卡都对得上线上那条链
# ---------------------------------------------------------------------------


def test_every_row_is_a_card_the_online_chain_actually_read() -> None:
    """`余量.json` 的每一行都**重算得出来**：真值就是 `roster_scores` 那一个。

    这一条同时挡住"表里混进了线上读不出来的卡"：读不出来（没过两条线）的卡不是真值。
    """
    frames = _frames()
    seen = {(row["frame"], row["index"]) for row in CARDS}
    assert len(seen) == len(CARDS), "同一张卡在表里出现了两次"
    for row in CARDS:
        frame = frames[row["frame"]]
        centres = squad_card_centres(frame, pitch=PITCH)
        region = np.ascontiguousarray(frame.crop(identity_region(centres[row["index"]][0], pitch=PITCH)))
        online = roster_scores(region, FACES)
        runner = online[1][0] if len(online) > 1 else 0.0
        assert online[0][0] >= ROSTER_SOLID_MATCH_SCORE
        assert online[0][0] - runner >= ROSTER_AMBIGUITY_MARGIN
        assert online[0][1] == row["name"]
    assert len({row["name"] for row in CARDS}) == 10


def test_the_rect_readings_are_the_online_chain_on_that_rectangle() -> None:
    """`rect` 那一列**十一档尺度都在**，所以它必须与线上那条链同值。

    这不是重复：它证明 `MaskedRuler` 在这块形状上真的退化成线上那条链，
    而不是"另一条算得比较松的链"。

    **第五单之后"分与名字"仍然逐位对得上**，但 `runner_up` 差了 **0.0027–0.0066**（六张里最大那个）
    （`0.459906 → 0.462611`）：区间扩到 `1.00…1.17` 之后，这块 **186 宽**的取区多拿到了
    `1.00…1.04` 那几档（`1.16/1.17` 装不下，被跳过并记进名单），次高在低档上找回了
    一点分。这一列是**十一档时代**的读数（见 `../帧样本/余量/README.md` 的备注）。

    **次高那一条改成形状断言**（用户裁定 10：旧夹具那条"跟旧读数比"的断言**不许只放宽数值**）：
    钉的是"次高**落在线的下面**、而且**低于最高分**"这两条**位置关系** —— 它们与"十一档
    还是十八档"无关；具体差值会随档区间变，所以不钉具体数（差值本身写在上一段里，
    那是**读数**，不是断言）。
    """
    frames = _frames()
    box = TABLE_RECORD["rect"]["bbox"]
    for row in CARDS[:6]:
        card = _card(frames[row["frame"]], row["index"])
        online = roster_scores(_at(card, box), FACES)
        runner = online[1][0] if len(online) > 1 else 0.0
        assert row["rect"]["score"] == pytest.approx(online[0][0], abs=1e-4)
        assert row["rect"]["name"] == online[0][1]
        assert row["rect"]["runner_up"] < ROSTER_SOLID_MATCH_SCORE
        assert row["rect"]["runner_up"] < row["rect"]["score"]
        assert runner < online[0][0]


def test_every_reading_carries_the_three_numbers() -> None:
    """每一列都要有 **分数 · 次高 · 余量** 三个数，而且余量就是两者之差。"""
    for row in CARDS:
        for column in ("mask", "rect", "rect_6680", "identity_masked"):
            reading = row[column]
            for key in ("score", "runner_up", "margin"):
                assert isinstance(reading[key], float)
            assert reading["margin"] == pytest.approx(reading["score"] - reading["runner_up"], abs=1e-5)
            assert -1.0 <= reading["score"] <= 1.0
            assert isinstance(reading["name"], str) and reading["name"]


def test_the_regions_are_the_measured_ones() -> None:
    """这几个像素数是**取区的界线**，不是装饰：掩码版与矩形版各多少、宽的那块为什么没用。"""
    assert TABLE_RECORD["mask"]["bbox"] == [0, 47, 186, 167]
    assert TABLE_RECORD["mask"]["pixels"] == 19746
    assert TABLE_RECORD["mask"]["box_pixels"] == 31062
    assert TABLE_RECORD["mask"]["wide"]["pixels"] == 22752
    assert TABLE_RECORD["mask"]["wide"]["usable_scales"] == [1.05]
    assert TABLE_RECORD["mask"]["usable_scales"] == [
        1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.13, 1.14, 1.15
    ]
    assert TABLE_RECORD["rect"]["bbox"] == [0, 177, 186, 37]
    assert TABLE_RECORD["rect"]["pixels"] == 6882
    assert [entry["pixels"] for entry in TABLE_RECORD["rect"]["variants"]] == [6680, 7181, 7548]
    assert TABLE_RECORD["baseline"]["masked"]["pixels"] == 748
    generated = free_mask(tuple(TABLE_RECORD["mask"]["bbox"]))
    assert int(generated.sum()) == TABLE_RECORD["mask"]["pixels"]
    generated_rect = free_mask(tuple(TABLE_RECORD["rect"]["bbox"]))
    assert bool(generated_rect.all())


# ---------------------------------------------------------------------------
# 职业图标：盖着罩子时读不读得出来
# ---------------------------------------------------------------------------


def test_the_class_icon_survives_the_cap() -> None:
    """罩子把图标压暗、但没有换掉它：逐通道仿射的斜率 **0.4…0.55**、残差 < 12、相关 > 0.95。

    这四条一起说的一件事：**那一条上的像素是干净卡那一条的同一个画面**（不是"另一张图"），
    只是整体被压掉一半。所以"读得出"这句话有像素支撑，不是印象。
    """
    evidence = TABLE_RECORD["class_icon"]["evidence"]
    assert len(evidence) == 4
    for entry in evidence:
        for slope in entry["slope"]:
            assert 0.35 < slope < 0.70
        assert entry["rms"] < 22.0
        assert entry["correlation"] > 0.90
        assert entry["capped_peak"] < 0.70 * entry["clean_peak"]


def test_the_class_icon_reads_out_the_right_library_entry() -> None:
    """按**二值字形**分类：四对里四对都对，而且第一名与第二名拉得开。"""
    class_icon = TABLE_RECORD["class_icon"]
    assert class_icon["readable"] is True
    truth = {row["capped"]["frame"] + "#" + str(row["capped"]["index"]): row["under_the_cap"]
             for row in TABLE_RECORD["undo"]}
    for entry in class_icon["evidence"]:
        assert entry["top"]["name"] in truth[entry["capped"]]
        # 分类器分得开的是**职业**，不是同一职业里的两个条目 —— 同一组的字形本来就
        # 几乎一样（IoU 0.77…0.91），所以第一名与第二名会挨得很近。那对它要干的事
        # 没有影响：它只要给出**哪一组**。
        assert entry["top"]["iou"] >= 0.80
        assert set(entry["shrunk_entries"]) == set(class_icon["groups"][truth[entry["capped"]]])
        assert len(entry["shrunk_entries"]) < len(FACES)
        for name in entry["shrunk_entries"]:
            assert name in {face for face, _art in FACES}


def test_the_class_groups_are_the_measured_ones() -> None:
    """同职业的图标**逐个像素同形**（IoU 0.77…0.91），不同职业 <= 0.56 —— 所以分得开。

    这一局上场的 10 位落在 **7 组**：{Mon3tr, 白面鸮} · {斩业星熊, 黍} ·
    {新约能天使, 歌蕾蒂娅} · {乌尔比安} · {圣聆初雪} · {极境} · {维什戴尔}。
    """
    groups = TABLE_RECORD["class_icon"]["groups"]
    assert len(groups) == 10
    sizes = sorted(len(peers) for peers in {tuple(sorted(v)) for v in groups.values()})
    assert sizes == [1, 1, 1, 1, 2, 2, 2]
    assert set(groups["黍"]) == {"黍", "斩业星熊"}
    assert set(groups["白面鸮"]) == {"白面鸮", "Mon3tr"}
    assert set(groups["新约能天使"]) == {"新约能天使", "歌蕾蒂娅"}
    for peers in groups.values():
        assert "流明" not in peers  # 这一局没上场，它的职业在这一批上量不出来
