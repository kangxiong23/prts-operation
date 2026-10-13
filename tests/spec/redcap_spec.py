"""红罩反解那一单的规格符合性验收（`tests/spec/`，**R 族**）。

跑法（不需要 pytest）：

    $env:PYTHONPATH="src"
    .venv\\Scripts\\python.exe tests\\spec\\redcap_spec.py

**为什么另开一族、不并进 `roster_spec.py`**：那一单自己就是「**单独一件实测，不跟
读数链混在一张单里**」（[特征库] §1）。并进去，同一份样本就既当读数链的验收帧、
又当反解的取材帧了。所以这一族叫 **R**，从 R1 起，自己一张 `CHECKS`。

**这一套只用公开面**：`roster.roster_scores` / `identity_region` / `IDENTITY_BOX_*` /
`IDENTITY_SCALE_MAX` / `ROSTER_SOLID_MATCH_SCORE` / `identity_is_exact` ·
`redcap.fit_redcap` / `undo_redcap` · `margin.MaskedRuler` / `identity_disc_region` /
`IDENTITY_DISC_BOX` / `IDENTITY_DISC_RADIUS` / `IDENTITY_DISC_STRIP_HALF` ·
`bench.squad_card_centres` / `RING_*` ·
`contracts.squad_card_rect` / `UI_LAYOUT` · `device.adb.SQUAD_CARD_PITCH` ·
`cardface.load_cardfaces` · `batch.load_batch` · `frame.frame_from_raw`。
`_` 开头的内部名**一个都不碰** —— 包括 `redcap._countdown_mask`：**掩码在这里自己
算一遍**（它是判据的一部分，抄它的实现就等于拿实现验实现）。

**真值是怎么来的**：`tests/帧样本/红罩/取材.json` 是**交付时给执行者的那份**，
所以它只能当线索。「罩下是谁」这条真值**不是目视的**，是同一张卡后来干净了那一帧
喂同一条链读出来的 —— 所以这一套**把那次干净读数自己复量一遍**（R3）。不复量，
就等于把"真值"建立在执行者抄给自己的一个数上。

## 「不够」这个结论要经得起哪一种检验

**它不能靠"反解后分数低"来立。** 身份区那个框（规格指定的取区）**92.5% 落在倒计时
图形里**（R7），而倒计时图形是**画在罩子之上**的（R4：那些像素饱和到 255，仿射对
clean=255 只能给出 112/57/52）—— 所以那 92.5% 是**死像素**。拿一个 92.5% 是死的框
去量"罩子底下还剩多少信息"，量的是"**环与数字还在框里**"，不是信息量。

**真正的检验是**：把**完全落在倒计时图形之外**的取区扫一遍（那些像素一个都没死），
看反解之后有没有**任何一块**过得了线 —— 判据与线上那条链同一个：分数过线**且**余量够。

- **R5** 是执行者交的那条读数，这一套复量一遍（在那个 92.5% 是死的框上）；
- **R6** 是真正的检验：圆外那一族逐块量**反解后**的分数与余量，报出最好的一块。
  这一族必须含**角上的小块** —— 执行者那张取区表只试了 10 块，**漏掉了它们**，
  而它们**分数真的到得了线**（`cd59#4` 的 `(0, 47, 60, 23)` 干净卡 **0.9588**；
  反解之后 `cd27#6` 到 **0.9303**）—— **但余量立不住**（那两个余量实测是 0.2032 /
  0.1847；够不够**判据**一律调 `roster.identity_is_exact`（判据的唯一定义），这里不把线写死）。
  **所以「圆外挪不出去」这个结论是对的，它错的是理由**：不是"分数够不到线"，
  是"**余量够不到**"。**只看分数那一半，就会得出相反的结论** —— 这一条网自己先踩过
  一次（第一版 R6 只断言分数，红了；红的是网，不是那一单）；
- **R7** 说的是那个 **100×100 的框挪不出去**（在立绘区里逐位置枚举，最近像素离环心
  最远只有 3.0），而且整块在圆外的最大矩形（6680 px）**比这个框（10000 px）还小**。
  所以"把框挪出环外"这条路要成立，就得换成一个**更小的框** —— 而更小的框正是
  **余量立不住**的那种（R6 的读数）。

**R6 一旦有哪一块两半都过（分数 ≥ 线 **且** 余量 ≥ 线），这一单的结论就翻了。**

## 线上现在走的是另一条路：这一族跟着改瞄

R5 · R6 · R7 量的都是**反解那条路**（`redcap.undo_redcap`）。线上**不反解**，也不把
取区挪到圆外 —— 红罩卡换的是**圆内那一块**：`margin.identity_disc_region()`（环带以内
的圆盘减去中央秒数那一条，卡内 `(58, 70, 89, 94)` / 掩码 3986 px），**原样**喂
`margin.MaskedRuler`（D-全局-34）。所以：

- **R5 / R6 / R7 的读数仍然成立**，但它们已经不是支点：R6 说的是"**圆外那条路不是
  出路**"（现在仍然真，而且它正是"取区只能待在圆内"的理由），R7 量的是"那个 100×100
  的方框挪不出去"（也仍然真）；
- **支点搬到了 R12**：判「够 / 不够」的那条读数，**现在这块取区**的原点动 1 px 必须
  给出同一个判定 —— 这一版把它按新取区重问了一遍。

**这一族还没有探针**

按 `README.md` 那条：探针优先盖"断言的是形状"的那些，而这里多半断言的是**值**。
但"一条都没有"是欠账 —— **最该先补的是 R7**（它是整个结论的几何支点，写歪了没人
看得见：把界线换成 30，它就永远绿）。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _entry in (str(REPO / "src"), str(_HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from ark.contracts import UI_LAYOUT, squad_card_rect  # noqa: E402
from ark.engine.device.adb import SQUAD_CARD_PITCH  # noqa: E402
from ark.engine.vision import bench  # noqa: E402
from ark.engine.vision.frame import Frame, frame_from_raw  # noqa: E402
from ark.engine.vision.redcap import fit_redcap, undo_redcap  # noqa: E402
from ark.engine.vision.roster import (  # noqa: E402
    IDENTITY_BOX_HEIGHT,
    IDENTITY_BOX_WIDTH,
    IDENTITY_BOX_X,
    IDENTITY_BOX_Y,
    IDENTITY_SCALE_MAX,
    IDENTITY_SCALES,
    ROSTER_SOLID_MATCH_SCORE,
    identity_is_exact,
    identity_region,
    required_runner_up,
    roster_scores,
)
from ark.offline.assets_gen.cardface import load_cardfaces  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402

BATCH = REPO / "tests" / "帧样本" / "红罩"
HARVEST = BATCH / "取材.json"
PITCH = SQUAD_CARD_PITCH
CARD_H, CARD_W = 214, 204
TOP_STRIP = int(UI_LAYOUT.rects["squad_card_rightmost_face"].y) - int(
    UI_LAYOUT.rects["squad_card_rightmost"].y
)
IDENTITY_BOX = (IDENTITY_BOX_X, IDENTITY_BOX_Y, IDENTITY_BOX_WIDTH, IDENTITY_BOX_HEIGHT)

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Check:
    """一条规格检查：**出处 · 一句话结论 · 探针**。"""

    ident: str
    spec: str
    claim: str
    probe: Callable[[], str]


@cache
def _frames() -> dict[str, Frame]:
    return {case.name: frame_from_raw(case.raw()) for case in load_batch(BATCH).cases}


@cache
def _faces() -> tuple[tuple[str, np.ndarray], ...]:
    return load_cardfaces()


@cache
def _pairs() -> tuple[dict, ...]:
    return tuple(json.loads(HARVEST.read_text(encoding="utf-8"))["pairs"])


def _by_set(which: str) -> tuple[dict, ...]:
    return tuple(pair for pair in _pairs() if pair["set"] == which)


def _card(frame: Frame, index: int) -> np.ndarray:
    """一帧里第 `index` 张卡的那**整块**卡面（卡内坐标从 (0,0) 起）。"""
    rect = squad_card_rect(index, PITCH)
    return np.array(
        frame.pixels[
            int(rect.y) : int(rect.y) + CARD_H, int(rect.x) : int(rect.x) + CARD_W
        ]
    )


def _blocks(which: str) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """某一集的配对，逐对给 `(盖着的那块, 没盖的那块)`。"""
    frames = _frames()
    return tuple(
        (
            _card(frames[Path(pair["capped"]["file"]).stem], pair["capped"]["index"]),
            _card(frames[Path(pair["clean"]["file"]).stem], pair["clean"]["index"]),
        )
        for pair in _by_set(which)
    )


@cache
def _model():
    """**只吃标定那两对** —— 验收那两对不参与定任何参数。"""
    return fit_redcap(_blocks("calibration"))


def _usable(shape: tuple[int, int]) -> np.ndarray:
    """哪些像素**既在圆盘外、又在立绘区里** —— 这一份是这里自己算的。

    圆心在卡内 `(卡宽/2, RING_CENTER_DY)`、半径 `RING_OUTER`，与读倒计时那条链
    同一个数；顶上那一条（职业图标与费用）从契约层那两个矩形量出来。
    """
    height, width = shape
    rows = np.arange(height, dtype=np.float64)[:, None]
    cols = np.arange(width, dtype=np.float64)[None, :]
    distance = np.sqrt(
        (cols - width / 2.0) ** 2 + (rows - float(bench.RING_CENTER_DY)) ** 2
    )
    return (distance > float(bench.RING_OUTER)) & (rows >= float(TOP_STRIP))


@cache
def _distance() -> np.ndarray:
    """整块卡面上，每个像素离环心多远 —— 下面几条几何论证都读它。"""
    rows = np.arange(CARD_H, dtype=np.float64)[:, None]
    cols = np.arange(CARD_W, dtype=np.float64)[None, :]
    return np.sqrt(
        (cols - CARD_W / 2.0) ** 2 + (rows - float(bench.RING_CENTER_DY)) ** 2
    )


@cache
def _fit_limits() -> tuple[int, int]:
    """取区在尺度网格里**最大能放到多少** —— 再大就装不进库块（`roster_scores` 会报错）。"""
    art = _faces()[0][1]
    return (
        int(art.shape[1] / IDENTITY_SCALE_MAX),
        int(art.shape[0] / IDENTITY_SCALE_MAX),
    )


def _outside(box: Box) -> bool:
    """这一块是不是**整块**在倒计时图形之外（用那条干净卡面的界线，不是环带外缘）。"""
    x, y, w, h = box
    return (
        float(_distance()[y : y + h, x : x + w].min())
        > float(bench.RING_OUTSIDE_INNER)
    )


@cache
def _outside_strips() -> tuple[tuple[str, Box], ...]:
    """四块**极大的圆外条带**（卡内坐标 `(x, y, 宽, 高)`）—— 四条边各取满一条。"""
    half = CARD_W / 2.0
    left_w = int(half - float(bench.RING_OUTSIDE_INNER))
    right_x = int(half + float(bench.RING_OUTSIDE_INNER)) + 1
    top_y = int(float(bench.RING_CENTER_DY) - float(bench.RING_OUTSIDE_INNER))
    bottom_y = int(float(bench.RING_CENTER_DY) + float(bench.RING_OUTSIDE_INNER)) + 1
    max_w, _ = _fit_limits()
    wide = min(CARD_W, max_w)
    strips: tuple[tuple[str, Box], ...] = (
        ("左边条", (0, TOP_STRIP, left_w, CARD_H - TOP_STRIP)),
        ("右边条", (right_x, TOP_STRIP, CARD_W - right_x, CARD_H - TOP_STRIP)),
        ("上边带", (0, TOP_STRIP, wide, top_y - TOP_STRIP + 1)),
        ("下边带", (0, bottom_y, wide, CARD_H - bottom_y)),
    )
    for name, box in strips:
        if not _outside(box):
            raise AssertionError(f"{name} {box} 不是整块在圆外 —— 这一条自己写歪了")
    return strips


@cache
def _outside_boxes(limit: int = 16) -> tuple[Box, ...]:
    """**扫**出来的圆外取区：粗网格上凡整块在圆外的都收下，再只留极大的那些。

    粗网格**不是穷举** —— 圆外的矩形是连续的一族（角上那一片是斜的边界），穷举不了。
    这里扫的是一圈代表：四条边取满，角上取几档。**所以 R6 说的是一族样本，不是全部**；
    真正把这条路堵死的是 R7 那条几何论证。
    """
    xs = (0, 20, 39, 60, 80, 102, 122, 144, 165, 184, 204)
    ys = (TOP_STRIP, 54, 70, 90, 116, 140, 160, 179, 200, 214)
    max_w, max_h = _fit_limits()
    found: list[Box] = []
    for left in range(len(xs)):
        for right in range(left + 1, len(xs)):
            for top in range(len(ys)):
                for bottom in range(top + 1, len(ys)):
                    x, y = xs[left], ys[top]
                    w, h = xs[right] - x, ys[bottom] - y
                    if w > max_w or h > max_h:
                        continue
                    if _outside((x, y, w, h)):
                        found.append((x, y, w, h))
    kept: list[Box] = []
    for box in sorted(found, key=lambda b: b[2] * b[3], reverse=True):
        x, y, w, h = box
        if any(
            ox <= x and oy <= y and ox + ow >= x + w and oy + oh >= y + h
            for ox, oy, ow, oh in kept
        ):
            continue
        kept.append(box)
        if len(kept) >= limit:
            break
    if not kept:
        raise AssertionError("粗网格上一块圆外取区都没扫出来 —— 这一条自己写歪了")
    return tuple(kept)


def _slice(box: Box) -> tuple[slice, slice]:
    """卡内一块矩形 → 整卡数组上的两把切片。"""
    x, y, w, h = box
    return (slice(y, y + h), slice(x, x + w))


def _cut(block: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    return np.ascontiguousarray(block[y : y + h, x : x + w])


def _score(block: np.ndarray, box: Box):
    return roster_scores(_cut(block, box), _faces())


def _identity_reading(frame: Frame, index: int) -> tuple[float, str, float, tuple[int, int]]:
    """那一张卡的**身份区**喂同一条链 —— 与 `roster_scores` 的调用方同一条路。"""
    centre = bench.squad_card_centres(frame, pitch=PITCH)[index][0]
    rect = identity_region(centre, pitch=PITCH)
    card = squad_card_rect(index, PITCH)
    offset = (int(rect.x) - int(card.x), int(rect.y) - int(card.y))
    ranked = roster_scores(np.ascontiguousarray(frame.crop(rect)), _faces())
    runner = ranked[1][0] if len(ranked) > 1 else -2.0
    return ranked[0][0], ranked[0][1], runner, offset


def check_the_pair_table_is_split_and_disjoint() -> str:
    """**R1** —— 标定集与验收集不相交，而且「罩下是谁」与它的来源逐对写明。"""
    pairs = _pairs()
    calibration = _by_set("calibration")
    validation = _by_set("validation")
    if not calibration or not validation:
        raise AssertionError(
            f"标定 {len(calibration)} 对 · 验收 {len(validation)} 对 —— 两半都要有，"
            "不分就没有验收那一半"
        )
    key = lambda ps: {(p["capped"]["file"], p["capped"]["index"]) for p in ps}  # noqa: E731
    overlap = key(calibration) & key(validation)
    if overlap:
        raise AssertionError(f"标定集与验收集相交于 {sorted(overlap)}：那两对就不算验收")
    for pair in pairs:
        for field in ("under_the_cap", "truth_from"):
            if not pair.get(field):
                raise AssertionError(f"{pair['capped']} 没有写明 {field!r}")
    batch = load_batch(BATCH)
    named = {p["capped"]["file"] for p in pairs} | {p["clean"]["file"] for p in pairs}
    if len(batch.cases) != len(named):
        raise AssertionError(
            f"取材表点到 {len(named)} 帧，而批里搬进来 {len(batch.cases)} 帧 —— "
            "要么多搬了没用的，要么取材表点了没搬的"
        )
    return (
        f"{len(pairs)} 对（标定 {len(calibration)} / 验收 {len(validation)}）· 两集不相交"
        f" ｜ 批 {len(batch.cases)} 帧 · version {batch.version} · "
        f"画布 {batch.canvas.width}×{batch.canvas.height} · facts {batch.facts}"
    )


def check_the_model_is_far_smaller_than_its_sample() -> str:
    """**R2** —— 参数个数远小于样本量，而且换一半标定样本重解落在同一组数附近。"""
    model = _model()
    mask = _usable((CARD_H, CARD_W))
    usable = int(mask.sum())
    pairs = len(_blocks("calibration"))
    if model.parameters >= model.pixels:
        raise AssertionError(
            f"模型报 {model.parameters} 个参数、{model.pixels} 个像素 —— "
            "参数不比样本少，那是把噪声也拟合进去了"
        )
    if model.pixels != pairs * usable:
        raise AssertionError(
            f"模型说它用了 {model.pixels} 个像素，而 {pairs} 对按掩码算出来是 "
            f"{pairs}×{usable} = {pairs * usable} —— 掩码那一侧对不上"
        )
    diff = np.concatenate(
        [
            capped[mask].astype(np.float64) - clean[mask].astype(np.float64)
            for capped, clean in _blocks("calibration")
        ]
    )
    trivial = float(np.sqrt((diff**2).mean()))
    if model.residual * 3.0 >= trivial:
        raise AssertionError(
            f"四个参数把残差压到 {model.residual:.2f}，而「什么都不做」是 {trivial:.2f} —— "
            "压不到三分之一，那这条仿射就没有解释掉多少东西"
        )
    wider = fit_redcap(_blocks("calibration") + _blocks("validation"))
    drift = abs(wider.slope - model.slope) / model.slope
    if drift > 0.5:
        raise AssertionError(
            f"换一半标定样本重解，斜率从 {model.slope:.4f} 跑到 {wider.slope:.4f}"
            f"（差 {drift:.0%}）—— 那就不是「一批配对落在同一组数上」"
        )
    return (
        f"参数 {model.parameters} 个 · 像素 {model.pixels} 个（{pairs} 对 × {usable}）"
        f" ｜ 残差 {model.residual:.2f} 对「什么都不做」{trivial:.2f}"
        f" ｜ 斜率 {model.slope:.4f} → {wider.slope:.4f}（差 {drift:.0%}）· "
        f"偏置 {model.offset[0]:.1f}/{model.offset[1]:.1f}/{model.offset[2]:.1f}"
    )


def check_the_truth_under_the_cap_is_remeasured() -> str:
    """**R3** —— 「罩下是谁」那条真值，这一套自己复量一遍（不是照抄取材表）。

    **分数不比**：取材表里那几个读数会随语义静默过期（报分语义现在是「第一个满足判据的
    那一档」，D-全局-43 —— 它本就小于等于整段尺子上的最大）。真值只认**名字与判决**。
    """
    frames = _frames()
    lines: list[str] = []
    for pair in _pairs():
        frame = frames[Path(pair["clean"]["file"]).stem]
        index = pair["clean"]["index"]
        best, name, runner, _ = _identity_reading(frame, index)
        margin = best - runner
        if name != pair["under_the_cap"]:
            raise AssertionError(
                f"{pair['clean']['file']} 位 {index} 的干净读数给的是「{name}」，"
                f"而取材表写的是「{pair['under_the_cap']}」"
            )
        if not identity_is_exact(best, runner):
            raise AssertionError(
                f"{pair['clean']['file']} 位 {index}：{best:.4f} · 次高 {runner:.4f} —— "
                "那次干净读数自己都没过判据（分数不够，或次高没退到 "
                f"{required_runner_up(best):.4f} 以下），它当不了真值"
            )
        # **不跟取材表里的分数比**：分数随语义变（报分语义是「第一个满足判据的那一档」，
        # D-全局-43），留在真值表里就会静默过期。真值只认名字与判决，分数由这一套现算。
        lines.append(f"{pair['clean']['file']}#{index} {name} {best:.4f}(余量 {margin:.4f})")
    return " ｜ ".join(lines)


def check_the_countdown_graphics_are_painted_over_the_cap() -> str:
    """**R4** —— 环与数字画在**罩子之上**（那些像素饱和），所以只能掩掉、解不回来。"""
    model = _model()
    distance = _distance()
    band = (distance >= float(bench.RING_INNER)) & (distance <= float(bench.RING_OUTER))
    ink_total = 0
    brightest = 0
    for capped, _ in _blocks("calibration") + _blocks("validation"):
        patch = capped[band]
        ink = int((patch.min(axis=1) >= bench.RING_BRIGHT_MIN).sum())
        ink_total += ink
        if ink:
            brightest = max(brightest, int(patch.min(axis=1).max()))
    prediction = [model.slope * 255.0 + offset for offset in model.offset]
    if ink_total == 0:
        raise AssertionError(
            "这几张罩着的卡上，环带里一个饱和像素都没有 —— 这一条**量不出来**，"
            "不是「环画在下面」"
        )
    if brightest <= max(prediction) + 80:
        raise AssertionError(
            f"环带上最亮的那个像素（min 通道 {brightest}）没有明显高过仿射能给的上限 "
            f"{max(prediction):.1f} —— 那它就不是画在罩子之上"
        )
    return (
        f"环带里饱和像素 {ink_total} 个 · 最亮 min 通道 {brightest} ｜ 仿射对 clean=255 "
        f"的预测 {prediction[0]:.1f}/{prediction[1]:.1f}/{prediction[2]:.1f} —— "
        "差着一个量级：那些像素是被毁掉的，不是被压暗的"
    )


def check_undoing_the_cap_still_misses_the_line() -> str:
    """**R5** —— 执行者交的那条读数（整个身份区，环与数字一起喂进链），这一套复量一遍。

    **这个口径对不上【做】4**：那一块 **92.5% 是死的**（R7），所以它量的是"环与数字
    还在框里"。这一条只负责复量，判"够不够"的是 R6。
    """
    model = _model()
    frames = _frames()
    box = np.zeros((CARD_H, CARD_W), bool)
    box[
        IDENTITY_BOX_Y : IDENTITY_BOX_Y + IDENTITY_BOX_HEIGHT,
        IDENTITY_BOX_X : IDENTITY_BOX_X + IDENTITY_BOX_WIDTH,
    ] = True
    live = float((_usable((CARD_H, CARD_W)) & box).sum()) / float(box.sum())
    lines: list[str] = []
    for pair in _by_set("validation"):
        index = pair["capped"]["index"]
        frame = frames[Path(pair["capped"]["file"]).stem]
        capped = _card(frame, index)
        centre = bench.squad_card_centres(frame, pitch=PITCH)[index][0]
        rect = identity_region(centre, pitch=PITCH)
        card_rect = squad_card_rect(index, PITCH)
        offset = (int(rect.x) - int(card_rect.x), int(rect.y) - int(card_rect.y))
        solved = undo_redcap(capped, model)
        ranked = roster_scores(
            _cut(solved, (offset[0], offset[1], IDENTITY_BOX_WIDTH, IDENTITY_BOX_HEIGHT)),
            _faces(),
        )
        runner = ranked[1][0] if len(ranked) > 1 else -2.0
        margin = ranked[0][0] - runner
        if identity_is_exact(ranked[0][0], runner):
            raise AssertionError(
                f"{pair['capped']['file']}#{index} 反解之后到了 {ranked[0][0]:.4f}"
                f"（次高 {runner:.4f}），**过了判据** —— 那《特征库》§1 那张表要重判"
            )
        capped_score = _identity_reading(frame, index)[0]
        lines.append(
            f"{pair['capped']['file']}#{index}: 盖着 {capped_score:.4f}"
            f" → 反解 {ranked[0][0]:.4f}（{ranked[0][1]}，余量 {margin:.4f}）"
        )
    return " ｜ ".join(lines) + f" ｜ 这一块只有 **{live:.4f}** 是活的（{1 - live:.1%} 是死像素）"


def _spec_says_why_the_outside_region_is_not_used() -> str:
    """规格里那句理由在不在：**判据不挡它，挡它的是准确率**。"""
    doc = (REPO / "docs" / "10-规格" / "视觉层" / "备战席点名.md").read_text(encoding="utf-8")
    needle = "认错率太高"
    if needle not in doc:
        raise AssertionError(
            f"规格里没有「{needle}」那句 —— 圆外那块取区为什么不用它，得写着（理由是这个实测，"
            "不是代数）"
        )
    return needle


def check_undoing_misses_the_line_on_every_region_outside_the_graphics() -> str:
    """**R6** —— 圆外那一族取区：**过得了判据，但认人不行**，所以不取它。

    **口径改过一次**（旧说法是「圆外没有一块过得了线」）：那是**旧余量线**下面的代数结论，
    已经翻 —— 实测里就有区域过得了（`cd27#6` 在一块圆外取区上反解到 0.9303、次高 0.7567，
    按现在的判据是**过**的）。**真正不取它的理由是这个实测**：圆外那块**认错率太高**
    （约六成准），不是代数不成立。所以这一条改成两头钉：**把过线的那些如实报出来**，
    而且**规格里那句理由得过在**。

    **它现在量的是哪条路**：**反解**那条路（`undo_redcap`）。线上不反解 —— 红罩卡换的是
    **圆内那一块**（圆盘 − 中央秒数那一条，原样喂 `MaskedRuler`，支点在 R12）。

    **什么会让这一条红**：规格里那句理由不见了；**或者一块圆外取区都没过判据**（那说明
    判据又变了，这句理由该翻回来）。
    """
    model = _model()
    frames = _frames()
    regions: list[tuple[str, Box]] = list(_outside_strips())
    regions += [("扫出来的", box) for box in _outside_boxes()]
    regions += [("角族", box) for box in _corner_boxes()]
    seen: set[Box] = set()
    unique: list[tuple[str, Box]] = []
    for label, box in regions:
        if box not in seen:
            seen.add(box)
            unique.append((label, box))
    lines: list[str] = []
    passed: list[str] = []
    for pair in _by_set("validation"):
        index = pair["capped"]["index"]
        capped_frame = frames[Path(pair["capped"]["file"]).stem]
        clean_frame = frames[Path(pair["clean"]["file"]).stem]
        solved = undo_redcap(_card(capped_frame, index), model)
        clean = _card(clean_frame, pair["clean"]["index"])
        scored: list[tuple[float, float, str, Box]] = []
        for label, box in unique:
            ranked = _score(solved, box)
            margin = ranked[0][0] - (ranked[1][0] if len(ranked) > 1 else -2.0)
            scored.append((ranked[0][0], margin, label, box))
        best = max(scored)
        clean_ranked = _score(clean, best[3])
        if identity_is_exact(best[0], best[0] - best[1]):
            passed.append(
                f"{pair['capped']['file']}#{index} 圆外 {best[3]} 反解 {best[0]:.4f}"
                f"（次高 {best[0] - best[1]:.4f}）**过判据**"
            )
        lines.append(
            f"{pair['capped']['file']}#{index}: 圆外 {len(unique)} 块，反解最好 {best[0]:.4f}"
            f"（{best[3]}，余量 {best[1]:.4f}）· 同一块干净卡 {clean_ranked[0][0]:.4f}"
        )
    if not passed:
        raise AssertionError(
            "一块圆外取区都没过判据 —— 那「判据不挡它、挡它的是准确率」这句该翻回来"
        )
    needle = _spec_says_why_the_outside_region_is_not_used()
    return (
        f"过判据的 {len(passed)} 块：{' ｜ '.join(passed[:3])} ｜ 规格里那句理由在（「{needle}」）"
    )


def check_the_identity_box_cannot_be_moved_off_the_graphics() -> str:
    """**R7** —— 身份区那个框**挪不出**倒计时图形，而整块在圆外的最大矩形比它还小 ——
    所以**取区只能待在圆内**，而线上正是这么办的：圆盘那一块（− 中央秒数那一条）。

    "挪出去"那条路现在是**不通**的（这条量的就是它），而它反过来把**现在用的这一块**
    钉住：`margin.identity_disc_region()` 那一块必须 1）整块落在卡片以内，
    2）离环心最远也够不到**环带的名义内缘**（`IDENTITY_DISC_RADIUS < RING_INNER`）。

    第 2 条不是装饰：半径挪到 **50** 就摸到**环带的名义内缘**（`bench.RING_INNER`），
    而白环本体就在那几环上 —— `margin.IDENTITY_DISC_RADIUS` 那一栏的梯子：**47 及
    以内**掩码里最亮的 `min(R,G,B)` 是 **38**（卡面自己的暗底，一点没漏），**49** 就
    到 **87**（半亮内边），**50** 到 **226**（环本体）。所以"离环心够不到 `RING_INNER`"
    正是"这一块只吃罩子底下的东西"的可量形式，半径一挪到 50 这一条就当场红。
    掩码的中央条同理：半高从 16 降到 15，第 101 与第 132 行那两圈边就留在取区里。

    **什么会让这一条红**：那个 100×100 的框能整个挪到圆外；或圆外最大的矩形不小于它；
    或**现在用的那一块取区**落到了卡片外面、或者离环心够到了环带的名义内缘。
    """
    from ark.engine.vision.margin import (
        IDENTITY_DISC_RADIUS,
        IDENTITY_DISC_STRIP_HALF,
        identity_disc_region,
    )

    distance = _distance()
    best: tuple[float, int, int] | None = None
    for y in range(TOP_STRIP, CARD_H - IDENTITY_BOX_HEIGHT + 1):
        for x in range(0, CARD_W - IDENTITY_BOX_WIDTH + 1):
            nearest = float(
                distance[
                    y : y + IDENTITY_BOX_HEIGHT, x : x + IDENTITY_BOX_WIDTH
                ].min()
            )
            if best is None or nearest > best[0]:
                best = (nearest, x, y)
    assert best is not None
    if best[0] > float(bench.RING_OUTSIDE_INNER):
        raise AssertionError(
            f"身份区那个框可以整个挪到圆外（最好的一处在 ({best[1]},{best[2]})，"
            f"最近像素离环心 {best[0]:.1f}）—— 那【做】6 的结论要重写"
        )
    mask = _usable((CARD_H, CARD_W))
    box = np.zeros((CARD_H, CARD_W), bool)
    box[
        IDENTITY_BOX_Y : IDENTITY_BOX_Y + IDENTITY_BOX_HEIGHT,
        IDENTITY_BOX_X : IDENTITY_BOX_X + IDENTITY_BOX_WIDTH,
    ] = True
    available = float((mask & box).sum()) / float(box.sum())
    if available >= 0.10:
        raise AssertionError(
            f"身份区那个框还剩 {available:.4f} 可用 —— 那它就不是「几乎整块压在"
            "倒计时图形里」，这一条的前提变了"
        )
    largest = max(
        _outside_boxes(limit=64) + tuple(box for _, box in _outside_strips()),
        key=lambda b: b[2] * b[3],
    )
    largest_area = largest[2] * largest[3]
    if largest_area >= IDENTITY_BOX_WIDTH * IDENTITY_BOX_HEIGHT:
        raise AssertionError(
            f"圆外最大的那块矩形有 {largest_area} px，不小于那个框的 "
            f"{IDENTITY_BOX_WIDTH * IDENTITY_BOX_HEIGHT} px —— 「圆外装不下这个框」不成立"
        )
    # ------ 现在用的那一块（圆盘 − 中央条）：它整块在圆内，而且离环带还差着几像素
    disc_box, disc_mask = identity_disc_region()
    dx, dy, dw, dh = disc_box
    if dx < 0 or dy < 0 or dx + dw > CARD_W or dy + dh > CARD_H:
        raise AssertionError(
            f"现在用的那一块取区 {disc_box} 不在卡片以内（卡是 {CARD_W}×{CARD_H}）—— "
            "取区要整块落在卡片以内，裁边不算"
        )
    disc = np.zeros((CARD_H, CARD_W), bool)
    disc[dy : dy + dh, dx : dx + dw] = disc_mask
    furthest = float(distance[disc].max())
    if furthest >= float(bench.RING_INNER):
        raise AssertionError(
            f"现在用的那一块取区离环心最远 {furthest:.1f}，已经到/过了环带的名义内缘 "
            f"{float(bench.RING_INNER)}（半径是 {IDENTITY_DISC_RADIUS}）—— 那它就开始吃"
            "环上那些被画死的像素了，而这一块取区只能吃罩子底下的东西"
        )
    return (
        f"那个框在立绘区里逐位置枚举（{CARD_W - IDENTITY_BOX_WIDTH + 1}×"
        f"{CARD_H - TOP_STRIP - IDENTITY_BOX_HEIGHT + 1} 个位置），最近像素离环心最远"
        f"只有 {best[0]:.1f} ≤ {bench.RING_OUTSIDE_INNER} —— **挪不出去** ｜ "
        f"圆外最大的矩形 {largest}{largest_area} px < 框的 "
        f"{IDENTITY_BOX_WIDTH * IDENTITY_BOX_HEIGHT} px ｜ 整卡可用 {float(mask.mean()):.4f}"
        f" · 那个框可用 **{available:.4f}**（{1 - available:.1%} 是死像素）｜ "
        f"现在用的那一块 {disc_box}（掩码 {int(disc.sum())} px · 离环心最远 "
        f"{furthest:.1f} < 环带内缘 {float(bench.RING_INNER)} · 中央条半高 "
        f"{IDENTITY_DISC_STRIP_HALF}）—— 它整块在圆内，吃的是环与秒数都不占的那些像素"
    )


def _box_mask() -> np.ndarray:
    """身份区那个框里**能用**的像素（框内坐标）—— 环心在框内的位置自己算。

    框内 `y` 都 ≥ `IDENTITY_BOX_Y` = 76 > `TOP_STRIP` = 47，所以顶上那一条那把刀在这里
    是空条件；只剩圆盘那一把。
    """
    rows = np.arange(IDENTITY_BOX_HEIGHT, dtype=np.float64)[:, None]
    cols = np.arange(IDENTITY_BOX_WIDTH, dtype=np.float64)[None, :]
    centre_x = IDENTITY_BOX_WIDTH / 2.0
    centre_y = float(bench.RING_CENTER_DY) - IDENTITY_BOX_Y
    return (
        np.sqrt((cols - centre_x) ** 2 + (rows - centre_y) ** 2)
        > float(bench.RING_OUTER)
    )


def _masked_scores(
    region: np.ndarray,
    mask: np.ndarray,
    faces: Sequence[tuple[str, np.ndarray]],
) -> tuple[tuple[float, str, float], ...]:
    """**自己写一份**掩码归一化相关（与 `margin.MaskedRuler` 无关的一份实现）。

    口径是**逐通道各减自己的均值**、分母把三通道加起来 —— 那就是 OpenCV
    `TM_CCOEFF_NORMED` 的口径。**掩码全开时它必须把 `roster_scores` 复现到浮点误差内**，
    那是它自证的唯一方式（R8）。
    """
    query = region.astype(np.float32) / 255.0
    ranked: list[tuple[float, str, float]] = []
    for name, art in faces:
        image_all = art.astype(np.float32) / 255.0
        best, best_scale = -2.0, IDENTITY_SCALES[0]
        for scale in IDENTITY_SCALES:
            height = int(round(region.shape[0] * scale))
            width = int(round(region.shape[1] * scale))
            if height > image_all.shape[0] or width > image_all.shape[1]:
                continue
            small = cv2.resize(query, (width, height), interpolation=cv2.INTER_LINEAR)
            layer = (
                cv2.resize(
                    mask.astype(np.uint8) * 255,
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 127
            )
            pixels = int(layer.sum())
            if pixels < 16:
                continue
            weight = layer.astype(np.float32)
            numerator: np.ndarray | None = None
            query_energy = 0.0
            image_energy: np.ndarray | None = None
            for channel in range(3):
                image = np.ascontiguousarray(image_all[:, :, channel])
                masked = np.ascontiguousarray(small[:, :, channel] * weight)
                cross = cv2.matchTemplate(image, masked, cv2.TM_CCORR).astype(np.float64)
                image_sum = cv2.matchTemplate(image, weight, cv2.TM_CCORR).astype(
                    np.float64
                )
                image_square = cv2.matchTemplate(
                    np.ascontiguousarray(image * image), weight, cv2.TM_CCORR
                ).astype(np.float64)
                mean = float(masked.sum()) / pixels
                query_energy += float((masked * masked).sum()) - pixels * mean * mean
                part = cross - mean * image_sum
                numerator = part if numerator is None else numerator + part
                variance = image_square - image_sum * image_sum / pixels
                image_energy = variance if image_energy is None else image_energy + variance
            if query_energy <= 0.0 or numerator is None or image_energy is None:
                continue
            denominator = np.sqrt(query_energy) * np.sqrt(np.maximum(image_energy, 0.0))
            value = float((numerator / (denominator + 1e-12)).max())
            if value > best:
                best, best_scale = value, scale
        ranked.append((best, name, best_scale))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return tuple(ranked)


def _corner_boxes(limit: int = 8) -> tuple[Box, ...]:
    """四个角上**整块在圆外**的取区，每角取 Pareto 极大的前几块。

    上一版 R6 的粗网格（`_outside_boxes`）只有 11 块，**漏掉了角上这一族** —— 而那一族
    在干净卡上是分数最好的（这一条正是甲组 2 的结论）。所以这一版把它补上。
    """
    corners = (
        (0, TOP_STRIP, 1, 1),  # 左上：从左边、从立绘上缘往里长
        (CARD_W, TOP_STRIP, -1, 1),  # 右上
        (0, CARD_H, 1, -1),  # 左下
        (CARD_W, CARD_H, -1, -1),  # 右下
    )
    found: list[Box] = []
    for origin_x, origin_y, step_x, step_y in corners:
        for wide in range(38, 76, 2):
            for tall in range(6, 64, 2):
                x = origin_x if step_x > 0 else origin_x - wide
                y = origin_y if step_y > 0 else origin_y - tall
                box = (x, y, wide, tall)
                if _outside(box):
                    found.append(box)
    kept: list[Box] = []
    for box in sorted(found, key=lambda b: b[2] * b[3], reverse=True):
        x, y, w, h = box
        if any(
            ox <= x and oy <= y and ox + ow >= x + w and oy + oh >= y + h
            for ox, oy, ow, oh in kept
        ):
            continue
        kept.append(box)
        if len(kept) >= limit * 4:
            break
    return tuple(kept)


_TRUTH_KEYS = frozenset({"set", "capped", "clean", "under_the_cap", "truth_from", "capped_name"})
"""真值表里**每一对**许有的键 —— 多一个键就是多一份会过期的假话。"""


def check_the_truth_table_carries_no_readings() -> str:
    """**R13** —— 真值表只认**名字 · 位号 · 判决 · 出处**，**不许存读数**（分数/余量那类）。

    **为什么这条要钉住**：读数会随语义**静默过期** —— 报分语义改成「第一个满足判据的
    那一档」（D-全局-43）之后，表里那些旧分数就变成了假话，而 R3 那次红就是这么来的
    （复量 0.9636 对表里 0.9848）。把分数从真值表里赶出去，那类红才不会再长出来：
    分数由秤现算。

    **顺带记一笔旧账**：原来这条位置上是 **R8**（「掩码全开时那把尺子逐位复现
    `roster_scores`」）。**它退役了** —— 那条网比的是**两个不同的数**：`roster_scores`
    报的是「定下判据的那一档」，它**判据一满足就停**、还会在粗层直接接受；而
    `MaskedRuler.scores` 是**每一档整张滑、取全局最大**。差别全在**报哪一个数**，
    不在算式（同档同位置两把逐位同值）。两把尺子于是**不是同一个量**，
    而新尺子靠**它自己那 44 对的实测**站住（见规格）。
    """
    pairs = json.loads(HARVEST.read_text(encoding="utf-8"))["pairs"]
    if not pairs:
        raise AssertionError("真值表一对都没有")
    for pair in pairs:
        extra = sorted(set(pair) - _TRUTH_KEYS)
        if extra:
            raise AssertionError(
                f"{pair.get('set', '?')} 这一对多了键 {extra} —— 真值只认 "
                f"{sorted(_TRUTH_KEYS)}（读数一律由秤现算）"
            )
        for slot in ("capped", "clean"):
            keys = set(pair[slot])
            if keys != {"file", "index"}:
                raise AssertionError(f"{pair['set']} 的 {slot} 只有 file/index，实际 {sorted(keys)}")
        for key in _TRUTH_KEYS - {"capped", "clean"}:
            value = pair[key]
            if not isinstance(value, str) or not value:
                raise AssertionError(
                    f"{pair['set']} 的 {key} 该是非空字符串（判决/出处/名字），收到 {value!r}"
                )
    return f"{len(pairs)} 对：键只有 {sorted(_TRUTH_KEYS)}，一处读数都没有"


def check_the_masked_reading_does_not_hinge_on_one_pixel() -> str:
    """**R12** —— 判「够 / 不够」的那条读数，**现在这块取区**（圆盘 − 中央秒数那一条）
    的原点动 1 px 必须给出**同一个判定**。

    **这一条是规格洞**：A2b 的单子里没写这一条（它只要求"报分数与余量"），是这一套在
    复量时发现的 —— 那条读数在 1 px 上摆得很大，而"够 / 不够"正是它判的。出处那一栏
    写的就是这件事本身。

    **这一版按新取区重问**：线上红罩卡走的是 `margin.identity_disc_region()` 那一块
    （`IDENTITY_DISC_BOX`，掩码 3986 px），而且**不反解**（`undo_redcap` 不在线上那条
    路上）—— 所以这里量的是**原样那张卡**上的那一块。上一版问的是那个 100×100 的方框，
    而那块框线上已经不用了：拿它量出来的摆幅是**另一块取区**的摆幅。

    **纵向不在这一条里**：取区的 y 由卡框给（`contracts.squad_card_rect` 的常数），
    **不是检出来的**；而掩码的中央条上下缘就贴着取区的上下边，纵向挪 1 px 会把秒数
    那一条的边带进来 —— 那不是"1 px 的选择"的问题，是"那一块形状就是这么定的"。
    中心是**检出来的**那一轴是横向（`bench.squad_card_centres` 给的中心逐帧会差零点几
    像素），所以这条问横向。

    **什么会让这一条红**：四对里任何一对，在新取区原点 ±1 px 的三个位置上给出的
    「两半都过 / 不过」不全是同一个答案 —— 那这条读数就撑不起「够 / 不够」。
    """
    from ark.engine.vision.margin import MaskedRuler, identity_disc_region

    frames = _frames()
    box, disc_mask = identity_disc_region()
    ruler = MaskedRuler(disc_mask, _faces())
    x0, y0, width, height = box
    lines: list[str] = []
    flips = 0
    for pair in _pairs():
        capped = _card(
            frames[Path(pair["capped"]["file"]).stem], pair["capped"]["index"]
        )
        scores: list[float] = []
        verdicts: list[bool] = []
        for shift in (-1, 0, 1):
            region = np.ascontiguousarray(
                capped[y0 : y0 + height, x0 + shift : x0 + shift + width]
            )
            ranked = ruler.scores(region)
            margin = ranked[0][0] - ranked[1][0]
            scores.append(ranked[0][0])
            verdicts.append(identity_is_exact(ranked[0][0], ranked[1][0]))
        if len(set(verdicts)) > 1:
            flips += 1
        lines.append(
            f"{pair['capped']['file']}#{pair['capped']['index']}: "
            f"{scores[0]:.4f}/{scores[1]:.4f}/{scores[2]:.4f}"
            f"（摆 {max(scores) - min(scores):.4f}）判定 {['过' if v else '不过' for v in verdicts]}"
        )
    if flips:
        raise AssertionError(
            f"{flips}/{len(_pairs())} 对的判定在新取区原点 ±1 px 上就翻了 —— "
            "那条读数撑不起「够 / 不够」，它撑不起的是一个 1 px 的选择"
        )
    return (
        " ｜ ".join(lines)
        + f" ｜ 取区 {box} · 掩码 {int(disc_mask.sum())} px（环带以内 − 中央秒数那一条）"
    )


def check_the_masked_reading_on_the_red_capped_cards() -> str:
    """**R9** —— 决定数：把倒计时图形排掉之后，**反解出来的**卡过不过线。

    这一条**不给"够 / 不够"的结论**（那一格是规划者的），它只把两件事钉住：
    真值排不排得到第一，以及**反解后的读数与同一张卡干净时的读数是不是同一档**。
    """
    model = _model()
    frames = _frames()
    mask = _box_mask()
    box = (IDENTITY_BOX_X, IDENTITY_BOX_Y, IDENTITY_BOX_WIDTH, IDENTITY_BOX_HEIGHT)
    lines: list[str] = []
    both = 0
    for pair in _pairs():
        capped_frame = frames[Path(pair["capped"]["file"]).stem]
        clean_frame = frames[Path(pair["clean"]["file"]).stem]
        solved = undo_redcap(_card(capped_frame, pair["capped"]["index"]), model)
        ranked = _masked_scores(_cut(solved, box), mask, _faces())
        margin = ranked[0][0] - ranked[1][0]
        clean_ranked = _masked_scores(
            _cut(_card(clean_frame, pair["clean"]["index"]), box), mask, _faces()
        )
        clean_margin = clean_ranked[0][0] - clean_ranked[1][0]
        if ranked[0][1] != pair["under_the_cap"]:
            raise AssertionError(
                f"{pair['capped']['file']}#{pair['capped']['index']}：排掉倒计时图形之后，"
                f"反解出来的第一名是「{ranked[0][1]}」，而真值是「{pair['under_the_cap']}」"
            )
        ok = identity_is_exact(ranked[0][0], ranked[1][0])
        both += 1 if ok else 0
        lines.append(
            f"{pair['capped']['file']}#{pair['capped']['index']} {pair['under_the_cap']}: "
            f"反解 {ranked[0][0]:.4f} 余量 {margin:.4f}{' 两半都过' if ok else ' 余量不够'}"
            f" · 同卡干净 {clean_ranked[0][0]:.4f} 余量 {clean_margin:.4f}"
        )
    return (
        " ｜ ".join(lines)
        + f" ｜ 排掉倒计时之后可用像素 {int(mask.sum())}/{mask.size}"
        f" = {float(mask.mean()):.4f} · 两半都过 {both}/{len(_pairs())}"
    )


def check_the_margin_table_carries_what_the_order_asked() -> str:
    """**R10** —— `余量.json` 的字段齐不齐，而且里面的像素数**这一套自己数一遍**。"""
    record = json.loads(
        (REPO / "tests" / "帧样本" / "余量" / "余量.json").read_text(encoding="utf-8")
    )
    for key in ("mask", "rect", "baseline", "cards", "class_icon", "undo"):
        if key not in record:
            raise AssertionError(f"`余量.json` 里没有 {key!r} 这一段")
    cards = record["cards"]
    if not cards:
        raise AssertionError("`余量.json` 的 cards 是空的")
    for row in cards:
        for field in ("frame", "index", "name", "truth"):
            if field not in row:
                raise AssertionError(f"cards 里有一行没有 {field!r}：{sorted(row)}")
        for column in ("mask", "rect", "identity_masked"):
            if column not in row:
                raise AssertionError(f"{row['frame']}#{row['index']} 少了 {column!r} 那一列")
            for field in ("score", "name", "runner_up", "margin"):
                if field not in row[column]:
                    raise AssertionError(
                        f"{row['frame']}#{row['index']} 的 {column} 少了 {field!r}"
                    )
    mask = _usable((CARD_H, CARD_W))
    wide_box = (0, TOP_STRIP, CARD_W, CARD_H - TOP_STRIP)
    narrow_box = tuple(record["mask"]["bbox"])
    checks = (
        ("mask.bbox", narrow_box, int(mask[_slice(narrow_box)].sum())),
        ("mask.wide", tuple(record["mask"]["wide"]["bbox"]),
         int(mask[_slice(tuple(record["mask"]["wide"]["bbox"]))].sum())),
        ("rect.bbox", tuple(record["rect"]["bbox"]),
         int(mask[_slice(tuple(record["rect"]["bbox"]))].sum())),
    )
    for label, box, mine in checks:
        theirs = int(
            record["mask"]["pixels"]
            if label == "mask.bbox"
            else record["mask"]["wide"]["pixels"]
            if label == "mask.wide"
            else record["rect"]["pixels"]
        )
        if mine != theirs:
            raise AssertionError(
                f"{label} {box}：这一套数出来 {mine} 个可用像素，表里写 {theirs} —— 对不上"
            )
    box_mask = int(_box_mask().sum())
    theirs_box = int(record["baseline"]["masked"]["pixels"])
    if box_mask != theirs_box:
        raise AssertionError(
            f"身份区排掉倒计时之后：这一套数出来 {box_mask} 个，表里写 {theirs_box}"
        )
    if not record["class_icon"].get("readable"):
        raise AssertionError("`class_icon.readable` 不是真 —— 路 B 就没有答案了")
    wide_scales = tuple(record["mask"]["wide"]["usable_scales"])
    if len(wide_scales) != 1:
        raise AssertionError(
            f"整块立绘区那一版报的可用尺度是 {wide_scales} —— 它本该只剩一档"
        )
    return (
        f"{len(cards)} 行 · 三个取区列齐 ｜ 可用像素这一套自己数："
        f"mask {checks[0][2]} · wide {checks[1][2]}（{len(wide_scales)} 档尺度）· "
        f"rect {checks[2][2]} · 身份区排掉倒计时 {box_mask}"
    )


def check_an_outside_block_can_pass_on_a_clean_card() -> str:
    """**R11** —— 圆外那一族里**存在**两半都过的取区（干净卡上）—— 甲组 2 的结论。

    这一条同时是对本族第一版 R6 的更正：那一版的粗网格**漏掉了角族**，所以它给出的
    "圆外没有一块过得了线"是**族不够**，不是事实。**它断言的正是"存在"**：验收那两对
    的干净卡上，凡是整块在圆外的取区（四条边带 + 角族），只要有**一块**分数与余量
    **同时**过线就算成立（实测 `cd59.png#4` 上有 6 块）。

    **"反解之后"那一半不在这里，在 R6**：这条量的是**干净卡**上那一族（甲组 2 问的就是
    它），R6 量的是**罩着的卡反解之后**、而且断言的是"一块都没有"。两条合起来才是那句
    结论：**出路在圆内、不在圆外，反解也救不回来**。所以在这里断言"反解之后存在"会与
    R6 直接打架 —— 那不是这一条要问的事。

    **什么会让这一条红**：两对干净卡上、圆外那一族里**一块两半都过的都没有** —— 那时
    "圆外那一族不行"就不是族不够，而是真的没出路，R6 那条结论得换个理由说。
    """
    frames = _frames()
    boxes = list(_outside_strips()) + [
        ("角族", box) for box in _corner_boxes()
    ]
    seen: set[Box] = set()
    unique: list[tuple[str, Box]] = []
    for label, box in boxes:
        if box not in seen:
            seen.add(box)
            unique.append((label, box))
    lines: list[str] = []
    passing_total = 0
    for pair in _by_set("validation"):
        frame = frames[Path(pair["clean"]["file"]).stem]
        block = _card(frame, pair["clean"]["index"])
        scored: list[tuple[float, float, str, Box]] = []
        for label, box in unique:
            ranked = _score(block, box)
            margin = ranked[0][0] - (ranked[1][0] if len(ranked) > 1 else -2.0)
            scored.append((ranked[0][0], margin, label, box))
        passing = [
            item
            for item in scored
            # 判据的唯一定义：这里的 item[1] 是余量，次高 = 分 − 余量
            if identity_is_exact(item[0], item[0] - item[1])
        ]
        passing_total += len(passing)
        best = max(scored)
        lines.append(
            f"{pair['clean']['file']}#{pair['clean']['index']}: {len(unique)} 块里"
            f"两半都过 {len(passing)} 块 · 最好 {best[0]:.4f} 余量 {best[1]:.4f} {best[3]}"
        )
    if passing_total == 0:
        raise AssertionError(
            f"验收那两对的干净卡上，圆外那 {len(unique)} 块里一块**两半都过**的都没有 —— "
            "那「圆外那一族不行」就不是**族不够**，而是真的没出路："
            "R6 那条结论要换个理由说，而 R7 的几何论证也就成了唯一那条路"
        )
    return (
        f"{len(unique)} 块圆外取区 · 两半都过共 {passing_total} 块（存在性就是这一条断言的）"
        " ｜ " + " ｜ ".join(lines)
    )


CHECKS: tuple[Check, ...] = (
    Check(
        "R1",
        "任务单 A2【做】1/【做】5 + 特征库 §1（这条实测单独一件）",
        "标定集与验收集不相交，「罩下是谁」与它的来源逐对写明，取材点到几帧就搬几帧",
        check_the_pair_table_is_split_and_disjoint,
    ),
    Check(
        "R2",
        "任务单 A2【做】3（几个参数、配了多少像素、必须有对照、换一半要能复算）",
        "四个参数远小于样本量；残差压到「什么都不做」的三分之一以下；换一半重解不跑偏",
        check_the_model_is_far_smaller_than_its_sample,
    ),
    Check(
        "R3",
        "任务单 A2【做】5（真值来源必须写明）+ 帧样本/红罩/取材.json",
        "四次干净读数这一套自己复量：名字与判决都对得上（分数不比 —— 它由这一套现算）",
        check_the_truth_under_the_cap_is_remeasured,
    ),
    Check(
        "R4",
        "任务单 A2【做】4 + 视觉层/备战席 §判死与再部署（环在罩子之上）",
        "倒计时图形是画在罩子之上的：那些像素饱和，仿射给不出那么亮 —— 只能掩掉",
        check_the_countdown_graphics_are_painted_over_the_cap,
    ),
    Check(
        "R5",
        "任务单 A2【做】5 + 特征库 §1 那张表（够 / 不够）",
        "执行者交的那条读数复量一遍：反解之后过不了线，而那一块 92.5% 是死的",
        check_undoing_the_cap_still_misses_the_line,
    ),
    Check(
        "R6",
        "任务单 A2【做】5/【做】6/【做】7 + 特征库 §1 那张表（够 / 不够）",
        "圆外那一族取区（含角上的小块）**反解之后**没有一块过得了线 —— 所以「挪到圆外」不是出路",
        check_undoing_misses_the_line_on_every_region_outside_the_graphics,
    ),
    Check(
        "R7",
        "任务单 A2【做】6 + 视觉层/备战席点名 §红罩卡（身份区 92.5% 压在图形里）· §两块取区",
        "那个 100×100 的框挪不出倒计时图形 —— 所以取区只能待在圆内；而现在用的那一块整块在圆内、够不到环带",
        check_the_identity_box_cannot_be_moved_off_the_graphics,
    ),
    Check(
        "R9",
        "任务单 A2b【做】4 + 特征库 §1 那张表（够 / 不够）",
        "排掉倒计时图形之后，反解出来的卡真值排得到第一，并与同卡干净时同一档",
        check_the_masked_reading_on_the_red_capped_cards,
    ),
    Check(
        "R10",
        "任务单 A2b【做】6（读数表那几个字段名）+ 视觉层/备战席点名 §红罩卡",
        "`余量.json` 的字段齐，而且里面每个像素数这一套自己数一遍都对得上",
        check_the_margin_table_carries_what_the_order_asked,
    ),
    Check(
        "R11",
        "任务单 A2b【做】2（扫法换一族）+ 任务单 A2【做】6",
        "干净卡上圆外那一族里**至少有一块**两半都过 —— 粗网格给的「圆外没有」是族不够，不是事实（反解那一半是 R6）",
        check_an_outside_block_can_pass_on_a_clean_card,
    ),
    Check(
        "R12",
        "**规格洞**：A2b 单子里没写这一条（它只要求报分数与余量）—— 这一套复量时发现 · §两块取区",
        "判「够 / 不够」的那条读数，**现在这块取区**（圆盘 − 中央秒数那一条）的原点动 1 px 必须给出同一个判定",
        check_the_masked_reading_does_not_hinge_on_one_pixel,
    ),
    Check(
        "R13",
        "视觉层/备战席点名 §红罩卡（真值只认名字 · 位号 · 判决 · 出处）",
        "真值表里一对多一个键都不许 —— 读数会随语义静默过期，分数一律由秤现算",
        check_the_truth_table_carries_no_readings,
    ),
)


def main() -> int:
    failures: list[tuple[Check, str]] = []
    print(f"红罩反解的规格符合性验收 —— {len(CHECKS)} 条（R 族，与 N 族分开的一份）\n")
    for check in CHECKS:
        try:
            evidence = check.probe()
        except AssertionError as bad:
            failures.append((check, str(bad)))
            print(f"  ✗ {check.ident}  {check.claim}")
            print(f"      出处 {check.spec}")
            print(f"      证据 {bad}")
        except BaseException as broken:  # noqa: BLE001
            failures.append((check, f"{type(broken).__name__}: {broken}"))
            print(f"  ‼ {check.ident}  {check.claim}")
            print(f"      出处 {check.spec}")
            print(f"      检查本身炸了 {type(broken).__name__}: {broken}")
        else:
            print(f"  ✓ {check.ident}  {check.claim}")
            print(f"      {evidence}")

    print()
    if failures:
        print(f"不成立 {len(failures)} / {len(CHECKS)}：")
        for check, why in failures:
            print(f"  - {check.ident}（{check.spec}）{why}")
        return 1
    print(f"全部成立（{len(CHECKS)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
