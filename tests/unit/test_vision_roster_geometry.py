"""几何资产那条路的单元测试（第五单 · 第六单）。

**它钉九件事**：键盖住了什么 · "第一次"是按卡位算的 · 对应关系换块取区照样成立 ·
红罩卡用干净那帧留下的几何读出来与整张滑**同一个答案** · 粗层直接接受**报得出自己在哪一层** ·
**资产是每卡位 × 每条**（每条用自己的对位） · **红罩那一刀**判定与名字不变 ·
**复用那条路的线在非匹配天花板之上（保守侧）** · **画面没变就整格跳过** · **升级留账** ·
**锚点层与掩码尺子都按模板尺寸去重**（第七单） · **掩码尺子懒建** · **每层成本报得出来**。

重活（整帧）只跑一次，装在模块级的夹具里。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ark.engine.device.adb import SQUAD_CARD_PITCH as PITCH
from ark.engine.vision import margin
from ark.engine.vision.bench import redeploying, squad_card_centres, squad_card_rect
from ark.engine.vision.frame import Canvas, Frame, frame_from_raw
from ark.engine.vision.roster import (
    IDENTITY_BOX_X,
    IDENTITY_BOX_Y,
    IDENTITY_SCALES,
    ROSTER_COARSE_ACCEPT_SCORE,
    ROSTER_REUSE_MATCH_SCORE,
    ROSTER_SOLID_MATCH_SCORE,
    SlotGeometry,
    SquadGeometry,
    _geometry_scores,
    _layer_rows,
    _layered_scores,
    _masked_layered_scores,
    _redcap_ruler,
    _template_shape,
    _window_block,
    _reuse_scores,
    identity_is_exact,
    identity_region,
    read_squad_identities,
    redcap_region,
)
from ark.offline.assets_gen.cardface import load_cardfaces
from ark.offline.bench.batch import load_batch

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "tests" / "帧样本" / "红罩"
HARVEST = json.loads((BATCH / "取材.json").read_text(encoding="utf-8"))
FACES = load_cardfaces(REPO / "tests" / "帧样本" / "点名" / "编队页.json")


def _frames() -> dict[str, Frame]:
    global _FRAMES
    if _FRAMES is None:
        _FRAMES = {c.name: frame_from_raw(c.raw()) for c in load_batch(BATCH).cases}
    return _FRAMES


_FRAMES: dict[str, Frame] | None = None


def _blank() -> Frame:
    return Frame(canvas=Canvas(width=2800, height=1260),
                 pixels=np.zeros((1260, 2800, 3), dtype=np.uint8))


def _name_of(value):
    payload = getattr(value, "payload", None)
    if payload is None or type(payload).__name__ != "Exact":
        return None
    return payload.value


def test_the_key_covers_the_library_the_pitch_and_the_canvas() -> None:
    """键 = (库指纹, 卡距, 画布尺寸)：**任一样变了，旧几何不许再用**。

    同一个键连着问两次**不重建**；键一变就**清空整本账**并记一次重建 ——
    这就是"不许拿旧几何去量新局面"那条裁定的可执行形式。
    """
    faces = [("甲", np.zeros((480, 214, 3), dtype=np.uint8)),
             ("乙", np.full((480, 214, 3), 7, dtype=np.uint8))]
    frame = _blank()
    assets = SquadGeometry()
    assets.ensure(faces, pitch=PITCH, frame=frame)
    assets.record(0, {"甲": SlotGeometry(1.1, 1.0, 2.0)})
    assert assets.slot_of(0) is not None
    assets.ensure(faces, pitch=PITCH, frame=frame)
    assert assets.rebuilds == 0 and assets.slot_of(0) is not None
    assets.ensure(faces, pitch=PITCH + 1.0, frame=frame)
    assert assets.rebuilds == 1 and assets.slot_of(0) is None
    assets.record(0, {"甲": SlotGeometry(1.1, 1.0, 2.0)})
    assets.ensure(faces, pitch=PITCH + 1.0,
                  frame=Frame(canvas=Canvas(width=2500, height=220),
                              pixels=np.zeros((220, 2500, 3), dtype=np.uint8)))
    assert assets.rebuilds == 2 and assets.slot_of(0) is None
    assets.record(0, {"甲": SlotGeometry(1.1, 1.0, 2.0)})
    changed = [("甲", np.zeros((480, 214, 3), dtype=np.uint8)),
               ("乙", np.full((480, 214, 3), 9, dtype=np.uint8))]
    assets.ensure(changed, pitch=PITCH + 1.0,
                  frame=Frame(canvas=Canvas(width=2500, height=220),
                              pixels=np.zeros((220, 2500, 3), dtype=np.uint8)))
    assert assets.rebuilds == 3 and assets.slot_of(0) is None


def test_the_correspondence_maps_any_box_on_the_card() -> None:
    """**换哪块取区都算得出来**：对应关系是"卡内坐标 → 立绘坐标"，不是那两个数。

    方框 `(52, 76)` 上反推出来的对应关系，搬到红罩那个圆盘 `(58, 70)` 上照样成立 ——
    两个落点差**正好是那一档下的框差**，与取区形状无关。
    """
    scale = 1.13
    geometry = SlotGeometry.from_position(scale, (100.0, 200.0),
                                          (IDENTITY_BOX_X, IDENTITY_BOX_Y))
    assert geometry.art_top_left((IDENTITY_BOX_X, IDENTITY_BOX_Y)) == pytest.approx(
        (100.0, 200.0))
    here = geometry.art_top_left((58.0, 70.0))
    assert here == pytest.approx(
        (100.0 + scale * (58 - IDENTITY_BOX_X), 200.0 + scale * (70 - IDENTITY_BOX_Y)))
    # 同一份对应关系上，两块取区各自的模板左上角只差"框差 × 档"。
    assert here[0] - 100.0 == pytest.approx(scale * (58 - IDENTITY_BOX_X))


def test_the_first_time_is_per_slot() -> None:
    """**第一次是每个卡位各自的第一次**：没有卡的格子没有几何，它回来时才跑满。

    跑一帧 `cd15`（10 格）之后，**这 10 格各自的**几何都在、第 11 格**没有**；
    再跑 `cd14`（还是 10 格、同一局）时**一格都不用跑满** —— 全都在复用或升级，
    键一次都没变（`rebuilds == 0`）。
    """
    frames = _frames()
    assets = SquadGeometry()
    trace: list = []
    values = read_squad_identities(frames["cd15"], FACES, pitch=PITCH, assets=assets,
                                   trace=trace)
    count = len(squad_card_centres(frames["cd15"], pitch=PITCH))
    exact = [row["index"] for row, value in zip(trace, values)
             if _name_of(value) is not None]
    assert exact and len(exact) < count, (exact, count)   # 有一格认不出
    assert [i for i in range(count) if assets.slot_of(i) is not None] == exact
    assert assets.slot_of(count) is None                   # 没卡的格子没有几何
    assert assets.counts["首帧"] + assets.counts["红罩首帧"] == count
    second: list = []
    read_squad_identities(frames["cd14"], FACES, pitch=PITCH, assets=assets, trace=second)
    assert assets.counts["复用"] > 0
    # **有几何的格子不会再"第一次"** —— 它只会复用或者升级。
    assert all(second[i]["how"] != "首帧" for i in exact)
    assert assets.rebuilds == 0


def test_a_capped_card_reads_the_same_as_the_full_slide_from_a_clean_frame() -> None:
    """**裁定 3 的直接取证**：红罩那一格用**同一份几何**读出来，与整张滑同一个答案。

    真值来自 `取材.json` 的 `under_the_cap`（拿同一张卡后来干净了那一帧读出来的）。
    这一条同时钉住"**变红罩之后位置与档位不变**"那句话：`cd15`（干净）留下几何，
    `cd14`（同一格罩着）复用它 —— **判定与名字**与整张滑那条路一致。

    **分不逐位相等，而且不该相等**：这里那份几何是从**干净卡那个方框**上反推的，
    整张滑那条路滑的是**圆盘**、而且每个档都试 —— 复用只在一个档、±3 像素的窗里比。
    裁定 3 要的那句话是"**判定与名字**不变"，不是"分不变"。
    """
    frames = _frames()
    assets = SquadGeometry()
    read_squad_identities(frames["cd15"], FACES, pitch=PITCH, assets=assets)
    pair = next(p for p in HARVEST["pairs"] if p["capped"]["file"] == "cd14.png")
    index = int(pair["capped"]["index"])
    assert pair["under_the_cap"] == "维什戴尔"
    assert assets.slot_of(index) is not None
    trace: list = []
    values = read_squad_identities(frames["cd14"], FACES, pitch=PITCH, assets=assets,
                                   trace=trace)
    assert _name_of(values[index]) == pair["under_the_cap"]
    assert trace[index]["how"] == "复用" and trace[index]["red"] is True
    centre = squad_card_centres(frames["cd14"], pitch=PITCH)[index][0]
    region = np.ascontiguousarray(frames["cd14"].crop(redcap_region(centre, pitch=PITCH)))
    ranked = tuple((s, n, c) for s, n, c, _p in _redcap_ruler(FACES).scores_and_positions(region))
    assert ranked[0][1] == pair["under_the_cap"]
    assert values[index].confidence >= 0.85
    assert values[index].confidence <= ranked[0][0] + 1e-9
    top, runner = ranked[0][0], ranked[1][0]
    assert identity_is_exact(top, runner)


def test_the_coarse_layer_accepts_and_says_which_layer() -> None:
    """**粗层直接接受报得出自己在哪一层**（用户裁定 4）—— 而且它不换答案。

    锚点层一过 0.9 就停：`trace["coarse"]` 说"锚点层"、`fine["tried"]` 是空的
    （位置层与全分辨率那一步**一次都没跑**）。`coarse_accept=False` 那一趟
    一定走到全分辨率。
    """
    frames = _frames()
    centre = squad_card_centres(frames["cd14"], pitch=PITCH)[0][0]
    region = np.ascontiguousarray(frames["cd14"].crop(identity_region(centre, pitch=PITCH)))
    ranked, trace = _layered_scores(region, FACES)
    assert trace["coarse"] is not None
    assert trace["coarse"]["layer"] == "锚点层"
    assert trace["coarse"]["top"] >= ROSTER_COARSE_ACCEPT_SCORE
    assert trace["fine"]["tried"] == []
    full, other = _layered_scores(region, FACES, coarse_accept=False)
    assert other["coarse"] is None
    assert other["fine"]["tried"]
    assert (ranked[0][1], ranked[0][0] >= 0.85) == (full[0][1], True)
    assert len(ranked) == len(full) == len(FACES)


def test_the_reused_geometry_scores_the_same_number_at_the_same_place() -> None:
    """复用那一步**不是另算一条统计量**：同一个档、同一个位置上给同一个数。"""
    frames = _frames()
    centre = squad_card_centres(frames["cd14"], pitch=PITCH)[0][0]
    region = np.ascontiguousarray(frames["cd14"].crop(identity_region(centre, pitch=PITCH)))
    box = (float(IDENTITY_BOX_X), float(IDENTITY_BOX_Y))
    geometry = SlotGeometry.from_position(1.10, (60.0, 90.0), box)
    ranked = _geometry_scores(region, FACES, {who: geometry for who, _art in FACES}, box)
    assert len(ranked) == len(FACES)
    assert all(scale == 1.10 for _s, _n, scale, _p in ranked)
    assert ranked == tuple(sorted(ranked, key=lambda item: (-item[0], item[1])))


def _red_ruler():
    global _RULER
    if _RULER is None:
        _RULER = _redcap_ruler(FACES)
    return _RULER


_RULER = None


def test_each_entry_gets_its_own_alignment() -> None:
    """**资产是"每卡位 × 每条"**（用户裁定 3）：每条库拿**它自己**的对位去比。

    跑满一趟之后，那一格的账里**每一条库**都有一份对位（`walk["geometry"]`）；把它们
    喂回 `_geometry_scores`，每一条都在**自己的档**、**自己的落点**上比。这一条量的是
    那件事的差别：拿**赢家那一份**几何喂给所有条目（从前那个局面），至少有一条的分数会
    明显更低 —— "错条目全塌到赢家那个位置上"就是这里被消掉的。
    """
    frames = _frames()
    centre = squad_card_centres(frames["cd15"], pitch=PITCH)[0][0]
    region = np.ascontiguousarray(frames["cd15"].crop(identity_region(centre, pitch=PITCH)))
    box = (float(IDENTITY_BOX_X), float(IDENTITY_BOX_Y))
    _ranked, walk = _layered_scores(region, FACES)
    mine = {name: SlotGeometry.from_position(scale, (x, y), box)
            for name, (scale, x, y) in walk["geometry"].items()}
    assert set(mine) == {name for name, _art in FACES}, "每一条库都要有自己的对位"
    own = {name: score for score, name, _s, _p in _geometry_scores(region, FACES, mine, box)}
    winner = max(own, key=lambda name: own[name])
    collapsed = {name: score for score, name, _s, _p in _geometry_scores(
        region, FACES, {who: mine[winner] for who, _art in FACES}, box)}
    assert own[winner] == pytest.approx(collapsed[winner], abs=1e-12)
    assert max(collapsed.values()) <= max(own.values()) + 1e-12
    gap = max(own[name] - collapsed[name] for name in own)
    assert gap > 1e-3, ("每条自己的对位与共用赢家那一份没有差别", gap)


def test_the_masked_walk_keeps_the_full_slide_answer() -> None:
    """**红罩那一刀**（用户裁定 1）：分层走完的**判定与名字**与整张滑**逐对相同**。

    掩码那条路从前整张滑 18 档；现在它走干净那条路同一套三层（锚点 25 像素 → 位置 50 像素 →
    全分辨率只在窗里）。**变的只有"搜哪儿"**：这一条钉住"变的不是答案" —— 那一帧上
    每一个红罩格，两条路的（判定, 名字）逐对比。

    （墙钟与 633 张的逐张对比在 `scripts/bench_roster_geometry.py` 的读数里；这里只钉
    一个可复现的小样本。）
    """
    frames = _frames()
    frame = frames["cd14"]
    card_top = float(squad_card_rect(0, PITCH).y)
    ruler = _red_ruler()
    compared = 0
    for index, (centre, _) in enumerate(squad_card_centres(frame, pitch=PITCH)):
        if redeploying(frame, centre, card_top=card_top) is not True:
            continue
        region = np.ascontiguousarray(frame.crop(redcap_region(centre, pitch=PITCH)))
        full = tuple((s, n, c) for s, n, c, _p in ruler.scores_and_positions(region))
        layered, walk = _masked_layered_scores(region, ruler)
        assert (full[0][1], identity_is_exact(*_pair_of(full))) == (
            layered[0][1], identity_is_exact(*_pair_of(layered))), f"cd14#{index}"
        assert walk["fine"]["layer"] in ("锚点层(掩码)", "全分辨率(掩码尺子)")
        compared += 1
    assert compared == 6, compared


def _pair_of(ranked):
    return ranked[0][0], (ranked[1][0] if len(ranked) > 1 else -2.0)


def test_the_reuse_line_and_its_own_two_clusters() -> None:
    """**复用那条路自己的两簇**（用户裁定 2/4）：真匹配是第一名，其余全在线下面。

    读数逐条来自 `取材.json` 的 `under_the_cap`（真值不是目视）：先读干净那一帧
    （留下几何），再读罩着那一帧 —— 对那个格子，每条库在**它自己的对位上**拿一个分。

    两条事实钉在这里：

    1. **真匹配就是那一格的第一名**，其余每一条都在**线（`ROSTER_REUSE_MATCH_SCORE`）下面**；
    2. **线在非匹配天花板之上** —— 全语料 633 张上量到的三个数（`几何复用.json`
       的 `reuse_clusters`，两种口径给的是同一组数）：

       | 量 | 值 |
       | --- | --- |
       | 非匹配天花板（480 对最高） | **0.662958** |
       | 真匹配地板（48 对最低） | **0.802482** |
       | 空档 | **0.139524** |
       | 线 | **0.85**（离天花板 **0.187**） |

       **线落在空档上面是保守侧，不是缺陷**（用户裁定 4）：这条路是**验证**那条路 ——
       它先问"上次留下的那份对应关系，还认得出这一格吗"。真匹配里 `[0.802482, 0.85)`
       那一段因此被**拒**，那一格就**升级**、退回**标定过的**全分辨率扫法去定答案。
       于是"线太高"的代价是**多花一次跑满**，而"线太低"的代价是**拿坏几何去认人**
       （假精确）。633 张上丢 0 · 改名 0 · 新错 0 就是这么来的。
       **不许为了少升级把线往下挪**（那就走进第二个代价）。

    （低的那几格几何来自**锚点层直接接受**：那一层的档只有 ~5 个取值、档位粗，
    复用那条路在那一档上给的分因此低于 0.85。裁定 5 **不**补跑位置层 —— 升级会把更细的
    几何写回资产，所以那几格是**一次性**代价；实测见
    `test_an_upgrade_writes_a_sharper_geometry_back_and_it_converges`。）
    """
    frames = _frames()
    pairs = 0
    floors: list[float] = []
    for pair in HARVEST["pairs"]:
        clean = frames[pair["clean"]["file"].replace(".png", "")]
        capped = frames[pair["capped"]["file"].replace(".png", "")]
        index = int(pair["capped"]["index"])
        truth = pair["under_the_cap"]
        assets = SquadGeometry()
        read_squad_identities(clean, FACES, pitch=PITCH, assets=assets)
        stored = assets.slot_of(index)
        assert stored is not None, f"{pair['clean']['file']}#{index} 干净那帧没留下几何"
        centre = squad_card_centres(capped, pitch=PITCH)[index][0]
        region = np.ascontiguousarray(capped.crop(redcap_region(centre, pitch=PITCH)))
        disc = (float(margin.IDENTITY_DISC_BOX[0]), float(margin.IDENTITY_DISC_BOX[1]))
        reading = _reuse_scores(region, FACES, stored, disc, red=True, ruler=_red_ruler())
        scores = {who: score for score, who, _s, _p in reading}
        assert set(scores) == {name for name, _art in FACES}
        assert max(scores, key=lambda who: scores[who]) == truth, (
            pair["capped"]["file"], scores)
        assert max(score for who, score in scores.items() if who != truth) < (
            ROSTER_REUSE_MATCH_SCORE), (pair["capped"]["file"], scores)
        floors.append(scores[truth])
        pairs += 1
    assert pairs == 4, pairs
    assert min(floors) < ROSTER_REUSE_MATCH_SCORE, (min(floors), floors)


def test_a_slot_whose_pixels_are_identical_is_skipped() -> None:
    """**画面没变就整格跳过**（用户裁定 4）：判据是**逐位相同**，不是分数。

    同一帧读两遍：第二遍每一格都跳过（画面逐位相同），而且沿用的判定与第一遍**逐位相同**
    （`Valued` 是冻结数据类）。**跳过的格照样出判定**：`trace` 里记着沿用的是哪一个。
    """
    frames = _frames()
    frame = frames["cd15"]
    assets = SquadGeometry()
    first = read_squad_identities(frame, FACES, pitch=PITCH, assets=assets)
    trace: list = []
    second = read_squad_identities(frame, FACES, pitch=PITCH, assets=assets, trace=trace)
    assert second == first
    assert [row["how"] for row in trace] == ["跳过"] * len(trace)
    assert assets.counts["跳过"] == len(trace)
    for row, value in zip(trace, second):
        assert row["reused"] == _name_of(value)


def test_only_verified_slots_leave_geometry_and_every_upgrade_leaves_a_row() -> None:
    """**几何只在一格真读出来时留下**；**每一次升级都留账**（用户裁定 5）。

    资产是一份**验过的**对应关系：判成未知的那一格**不留**几何（它的对位是"最像的巧合"，
    拿它当几何等于下一帧拿巧合当证据）。升级则**每一行都要有**前名（或 `None` = 未知）·
    后名 · 当时的最高分 —— 它是"卡位换人"唯一的账。
    """
    frames = _frames()
    assets = SquadGeometry()
    first = read_squad_identities(frames["cd15"], FACES, pitch=PITCH, assets=assets)
    exact = {index for index, value in enumerate(first) if _name_of(value) is not None}
    assert exact and len(exact) < len(first), exact
    assert {index for index in range(len(first)) if assets.slot_of(index) is not None} == exact
    # 升级那一条用一个**确定性的**逼法：把留下的几何**故意挪歪 200 像素**，
    # 下一帧那一格就一定过不了判据 —— 于是它退回跑满、留下一行账。
    for index in exact:
        stored = assets.slot_of(index)
        assets.record(index, {who: SlotGeometry(g.scale, g.offset_x + 200.0, g.offset_y)
                              for who, g in stored.items()})
    second = read_squad_identities(frames["cd14"], FACES, pitch=PITCH, assets=assets)
    assert assets.upgrades, "挪歪之后没有留下升级账"
    assert len(assets.upgrades) == assets.counts["升级"] >= 1
    for row in assets.upgrades:
        assert row["before"] is not None and row["before_top"] is not None
        assert row["after"] is not None, "挪歪几何的格子退回跑满之后应当还是读得出来"
        assert row["after_top"] >= ROSTER_SOLID_MATCH_SCORE
        assert _name_of(second[row["index"]]) == row["after"]

# ---------------------------------------------------------------------------
# 第七单：去重 · 尺子懒建 · 按层拆解 · 升级自愈
# ---------------------------------------------------------------------------


def _first_red_region() -> np.ndarray:
    """`cd14` 上**第一张红罩卡**的那块圆盘取区。"""
    frame = _frames()["cd14"]
    card_top = float(squad_card_rect(0, PITCH).y)
    for centre, _ in squad_card_centres(frame, pitch=PITCH):
        if redeploying(frame, centre, card_top=card_top) is True:
            return np.ascontiguousarray(frame.crop(redcap_region(centre, pitch=PITCH)))
    raise AssertionError("cd14 上没有红罩卡")


def _naive_layer_rows(small_query, small_art, scales, positions, position_half, factor):
    """**不去重**那一份 —— 照 `_layer_rows` 从前的代码重写一遍（同一串调用、同一串算式）。"""
    rows: dict[float, tuple[float, float, float]] = {}
    skipped: list[float] = []
    for scale in scales:
        height, width = _template_shape(small_query, scale)
        if height > small_art.shape[0] or width > small_art.shape[1]:
            skipped.append(scale)
            continue
        template = cv2.resize(small_query, (width, height), interpolation=cv2.INTER_LINEAR)
        block, x0, y0 = _window_block(
            small_art, positions, scale, position_half, factor, width, height
        )
        result = cv2.matchTemplate(block, template, cv2.TM_CCOEFF_NORMED)
        _low, high, _low_loc, high_loc = cv2.minMaxLoc(result)
        rows[scale] = (float(high), (high_loc[0] + x0) / factor, (high_loc[1] + y0) / factor)
    return rows, skipped


def test_the_anchor_layer_does_not_score_one_template_size_twice(monkeypatch) -> None:
    """**锚点层按模板尺寸去重**（用户裁定 1）：同一尺寸的档只算一次，读数一个字节不变。

    `round(25 × 档)` 在 18 个档上只落 **5** 个值（25/26/27/28/29，分布 2/5/3/5/3）。
    尺寸一样时模板逐位相同、`_window_block` 的块逐位相同（锚点层不裁窗，落点一律
    (0, 0)）⇒ `matchTemplate` 与 `minMaxLoc` 逐位相同 —— 所以从前那 **13** 次是白算的。
    这一条钉两件事：**调用次数 = 不同尺寸的个数**，以及**去重那一份与不去重那一份逐档相同**。
    """
    rng = np.random.RandomState(3)
    query = rng.randint(0, 255, (25, 25, 3), np.uint8)
    art = rng.randint(0, 255, (214, 204, 3), np.uint8)
    shapes = {_template_shape(query, scale) for scale in IDENTITY_SCALES}
    assert len(shapes) == 5, sorted(shapes)
    calls = {"n": 0}
    real = cv2.matchTemplate

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(cv2, "matchTemplate", counting)
    got, skipped = _layer_rows(query, art, IDENTITY_SCALES, {}, None, 1.0)
    monkeypatch.undo()
    assert skipped == []
    assert calls["n"] == len(shapes) == 5, calls["n"]
    want, want_skipped = _naive_layer_rows(query, art, IDENTITY_SCALES, {}, None, 1.0)
    assert got == want and skipped == want_skipped
    assert sorted(got) == list(IDENTITY_SCALES)


def test_the_anchor_layer_dedup_also_holds_when_a_window_is_cut() -> None:
    """位置层也吃同一条去重，但**键带上落点**：窗不一样就照算，读数照样逐档相同。"""
    rng = np.random.RandomState(5)
    query = rng.randint(0, 255, (50, 50, 3), np.uint8)
    art = rng.randint(0, 255, (214, 204, 3), np.uint8)
    positions = {scale: (float(30 + 40 * i), float(20 + 30 * i))
                 for i, scale in enumerate(IDENTITY_SCALES)}
    got, skipped = _layer_rows(query, art, IDENTITY_SCALES, positions, 12, 1.0)
    want, want_skipped = _naive_layer_rows(query, art, IDENTITY_SCALES, positions, 12, 1.0)
    assert got == want and skipped == want_skipped


def test_the_masked_ruler_builds_nothing_until_it_is_asked() -> None:
    """**尺子懒建**（用户裁定 2）：构造**一层都不建**，问到哪一档才建哪一档。

    `usable_scales` 是一次**形状检查**（掩码 resize + 数像素），不碰 `matchTemplate`
    —— 所以构造完就能问，不会把懒建变成"一建就建满"。
    """
    _box, mask = margin.identity_disc_region()
    ruler = margin.MaskedRuler(mask, FACES)
    assert ruler.built_cells == {"queried": 0, "built": 0, "art": 0, "full_layers": 0}
    assert ruler.usable_scales == IDENTITY_SCALES
    assert ruler.built_cells["queried"] == 0
    region = _first_red_region()
    scale = IDENTITY_SCALES[3]
    windows = {name: {scale: (10.0, 20.0)} for name, _art in FACES}
    ruler.rows(region, only=scale, windows=windows, half=3)
    assert ruler.built_cells["queried"] == len(FACES)
    assert ruler.built_cells["built"] == len(FACES)


def test_the_masked_ruler_does_not_build_one_template_size_twice() -> None:
    """**掩码尺子也按模板尺寸去重**（用户裁定 1）：粗层那一半里有一半是白算的。

    掩码 94×89，25 像素那一层 18 个档只落在 **9** 个尺寸上（50 像素那一层 12 个、
    全分辨率 18 个）。这一条钉住"建出来的 (层, 条目, 尺寸) 个数 = 条目数 × 尺寸数"。
    """
    _box, mask = margin.identity_disc_region()
    ruler = margin.MaskedRuler(mask, FACES)
    region = _first_red_region()
    height, width = ruler.mask_shape
    factor = 25.0 / float(height)
    shapes = {(int(round(height * scale * factor)), int(round(width * scale * factor)))
              for scale in IDENTITY_SCALES}
    assert len(shapes) == 9, sorted(shapes)
    ruler.rows(region, pixels=25)
    assert ruler.built_cells["queried"] == len(FACES) * len(shapes)
    assert ruler.built_cells["built"] == len(FACES) * len(shapes)


def test_each_walk_reports_its_own_cost_by_layer() -> None:
    """**按层拆解**（用户裁定 3）：每一层的墙钟落进 `trace["cost"]`，两条路同一套键。

    粗层直接接受那一趟**只花锚点层那一笔**（位置层与全分辨率一次都没跑）—— 这正是一格
    省下来的地方。`coarse_accept=False` 那一趟三层都在。红罩那条路同键。
    """
    frames = _frames()
    centre = squad_card_centres(frames["cd14"], pitch=PITCH)[0][0]
    region = np.ascontiguousarray(frames["cd14"].crop(identity_region(centre, pitch=PITCH)))
    _ranked, full = _layered_scores(region, FACES, coarse_accept=False)
    assert set(full["cost"]) == {"锚点层", "位置层", "精算层"}
    assert all(value >= 0.0 for value in full["cost"].values())
    _ranked, coarse = _layered_scores(region, FACES)
    assert coarse["coarse"] is not None and coarse["coarse"]["layer"] == "锚点层"
    assert set(coarse["cost"]) == {"锚点层"}
    _ranked, walk = _masked_layered_scores(_first_red_region(), _red_ruler())
    assert "锚点层" in walk["cost"]
    assert set(walk["cost"]) <= {"锚点层", "位置层", "精算层"}
    _ranked, deep = _masked_layered_scores(_first_red_region(), _red_ruler(),
                                           coarse_accept=False)
    assert set(deep["cost"]) == {"锚点层", "位置层", "精算层"}


def test_an_upgrade_writes_a_sharper_geometry_back_and_it_converges() -> None:
    """**升级自愈**（用户裁定 5）：升级把更细的几何写回资产，所以那几格**会收敛**。

    `红罩` 那 8 帧按批次顺序读一遍（7 次升级、5 个卡位）。**"升级一次就再也不升"不成立**：
    实测 2 个卡位升了**两次**（卡位 6：cd22 白面鸮 0.8536 → cd23 白面鸮 0.9716；
    卡位 5：cd22 Mon3tr 0.9535 → cd37 Mon3tr 0.9642）。但两次都**没有第三次**，
    而且**重复升级不换名字** —— 它是"几何变细"，不是"换人"。
    这一条钉的就是那两条形状：**没有第三次**、**重复升级同名**。
    """
    assets = SquadGeometry()
    per_slot: dict[int, list[dict]] = {}
    for case in load_batch(BATCH).cases:
        before = len(assets.upgrades)
        read_squad_identities(frame_from_raw(case.raw()), FACES, pitch=PITCH, assets=assets)
        for row in assets.upgrades[before:]:
            per_slot.setdefault(row["index"], []).append(row)
    assert len(assets.upgrades) == assets.counts["升级"] >= 1
    assert max(len(rows) for rows in per_slot.values()) == 2, {
        index: len(rows) for index, rows in per_slot.items()}
    repeats = [rows for rows in per_slot.values() if len(rows) > 1]
    assert repeats, "这一批里应当有升了两次的卡位"
    for rows in repeats:
        names = {row["after"] for row in rows}
        assert len(names) == 1 and None not in names, rows
