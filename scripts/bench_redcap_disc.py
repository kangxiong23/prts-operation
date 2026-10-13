
r"""红罩那条路：把**取区**换成「环带内缘以内的圆盘 − 中央秒数那一条」。

    .venv\Scripts\python.exe scripts\bench_redcap_disc.py [--limit K] [--specs 名字,名字]

## 这一单问什么

「证明场上哪个干员阵亡」要一条不必读技力条的取证路。《视觉层/备战席点名》§红罩卡
把「是谁」那一格挂了（用户裁定），要重开就得先答一件事：**换掉取区之后，余量够不够**。
这一单只做这件事 —— **取区换成示意图那个形状，别的什么都不变**：模型 · 那条链 · 库 ·
尺度区间 · 阈值 · 落法形态，一个都没动，线上那条链**一个字节没动**。

## 形状住哪儿

`margin.identity_disc_mask` 一处定义：**圆心在卡内 `(卡宽/2, RING_CENTER_DY)`、
半径 `IDENTITY_DISC_RADIUS` 的圆盘**（不是 `bench.RING_INNER` —— 那圈白环往内多压了
约 1.5 px，那个常量自己的文档里有逐环的实测），挖掉**中央 `IDENTITY_DISC_STRIP_HALF` 行**
（秒数所占的那一条）。框（`margin.identity_disc_box`）就是那把掩码自己的紧包围盒。
**罩着端与干净端用同一块取区**（罩子只有一层，用户裁定）。

## 并排的东西

- **今天那套取区**两行**原样留着**、一个字不改：线上那条链（整个身份区 100x100 =
  10000 px）与**掩码 552**（身份区排掉倒计时圆盘、半径 62）。它们在
  `tests/帧样本/预处理/绝对线.json` 里已经有一份，这一份是**同一条链、同一批 44 对**
  上的复量 —— 两边的端点对不上就是并排的不是同一条链（见 `self_check`）；
- **新的圆盘取区**：半径梯子 + 中央条梯子，每一档**罩着端与干净端各一组** ——
  真值最低分与出处 · 次高最高分与出处 · 比值 · 过线数 · 互不相同的画面与画面组合。

**判「够 / 不够」不在这份文件里**（那是规划者的）。这里只给数与口径。
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.engine.vision.bench import RING_CENTER_DY  # noqa: E402
from ark.engine.vision.frame import frame_from_raw  # noqa: E402
from ark.engine.vision.margin import (  # noqa: E402
    IDENTITY_DISC_BOX,
    IDENTITY_DISC_RADIUS,
    IDENTITY_DISC_STRIP_HALF,
    MaskedRuler,
    free_mask,
    identity_disc_box,
    identity_disc_mask,
    identity_disc_region,
)
from ark.engine.vision.redcap import (  # noqa: E402
    TOP_STRIP_HEIGHT,
    fit_redcap,
    undo_redcap,
)
from ark.engine.vision.roster import (  # noqa: E402
    IDENTITY_BOX_HEIGHT,
    IDENTITY_BOX_WIDTH,
    IDENTITY_BOX_X,
    IDENTITY_BOX_Y,
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
    roster_scores,
)
from ark.offline.assets_gen.cardface import load_cardfaces  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402

BATCH = REPO / "tests" / "帧样本" / "预处理"
HARVEST = BATCH / "取材.json"
PUBLISHED = BATCH / "绝对线.json"
OUT = BATCH / "圆盘取区.json"

BOX = (IDENTITY_BOX_X, IDENTITY_BOX_Y, IDENTITY_BOX_WIDTH, IDENTITY_BOX_HEIGHT)
CARD = (214, 204, 3)
OLD_MASK_RADIUS = 62
"""今天那套掩码那一行用的半径 —— 与 `绝对线.json` 的 `基线-掩码552` 同一个数。"""

RADIUS_LADDER = (45, 46, 47, 48, 49, 50, 51, 52)
"""半径这把梯子（挪一档 = 1 px）。50 是定下的那一档（环带内缘）。"""

STRIP_LADDER = (13, 14, 15, 16, 17)
"""中央条这把梯子（挪一档 = 1 行）。15 是定下的那一档。"""


def _sha(image: np.ndarray, length: int = 12) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()[:length]


def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, width, height = (int(value) for value in box)
    return np.ascontiguousarray(image[y0 : y0 + height, x0 : x0 + width])


def _specs() -> list[dict]:
    """这一份要铺的那些行 —— 今天那套两行 + 两把梯子。"""
    rows = [
        {
            "name": "基线-不掩码10000",
            "family": "today",
            "mode": "full",
            "how": (
                "今天那套取区之一：线上那条链（`roster_scores`，TM_CCOEFF_NORMED），"
                "整个身份区 100x100 = 10000 px，不掩码"
            ),
        },
        {
            "name": "基线-掩码552",
            "family": "today",
            "mode": "identity-masked",
            "radius": OLD_MASK_RADIUS,
            "how": (
                "今天那套取区之二：身份区 100x100 排掉倒计时圆盘（半径 %d）之后那 552 px，"
                "逐通道减均值的掩码相关" % OLD_MASK_RADIUS
            ),
        },
    ]
    for radius in RADIUS_LADDER:
        rows.append(
            {
                "name": "圆盘取区·半径%d" % radius,
                "family": "disc",
                "mode": "disc",
                "radius": radius,
                "half": IDENTITY_DISC_STRIP_HALF,
                "how": "圆盘半径 %d − 中央 %d 行" % (radius, IDENTITY_DISC_STRIP_HALF),
            }
        )
    for half in STRIP_LADDER:
        if half == IDENTITY_DISC_STRIP_HALF:
            continue
        rows.append(
            {
                "name": "圆盘取区·中央条半高%d" % half,
                "family": "disc",
                "mode": "disc",
                "radius": IDENTITY_DISC_RADIUS,
                "half": half,
                "how": "圆盘半径 %d − 中央 %d 行" % (IDENTITY_DISC_RADIUS, half),
            }
        )
    return rows


def _region_spec(spec: dict) -> dict:
    """一行 → 它的取区：`(框, 掩码)` 这一对。"""
    if spec["mode"] == "full":
        return {"box": list(BOX), "mask": None, "pixels": BOX[2] * BOX[3]}
    if spec["mode"] == "identity-masked":
        mask = free_mask(BOX, radius=int(spec["radius"]))
        return {"box": list(BOX), "mask": mask, "pixels": int(mask.sum())}
    box, mask = identity_disc_region(radius=int(spec["radius"]), half=int(spec["half"]))
    return {"box": list(box), "mask": mask, "pixels": int(mask.sum())}


def _end_summary(
    per: list[dict],
    region_hashes: list[str],
    card_hashes: list[str],
    region_pairs: list[str],
    card_pairs: list[str],
) -> dict:
    min_truth = min(per, key=lambda e: (e["truth_score"], e["key"]))
    max_wrong = min(per, key=lambda e: (-e["wrong_score"], e["key"]))
    margins = [e["margin"] for e in per]
    return {
        "pairs": len(per),
        "min_truth": {
            "score": min_truth["truth_score"],
            "pair": min_truth["key"],
            "frame": min_truth["key"].split("#")[0],
            "slot": min_truth["slot"],
            "entry": min_truth["truth"],
            "set": min_truth["set"],
        },
        "max_wrong": {
            "score": max_wrong["wrong_score"],
            "pair": max_wrong["key"],
            "frame": max_wrong["key"].split("#")[0],
            "slot": max_wrong["slot"],
            "entry": max_wrong["wrong_name"],
        },
        "ratio": round(
            (1.0 - min_truth["truth_score"]) / (1.0 - max_wrong["wrong_score"]), 6
        ),
        "gap": round(min_truth["truth_score"] - max_wrong["wrong_score"], 6),
        "max_truth": round(max(e["truth_score"] for e in per), 6),
        "median_margin": round(statistics.median(margins), 6),
        "min_margin": round(min(margins), 6),
        # **不落判定**：只落分数与余量，判定由读者按 roster 的两条线算（见 criterion）。
        "distinct_readings": len({(e["truth_score"], e["wrong_score"]) for e in per}),
        "distinct_regions": len(set(region_hashes)),
        "distinct_cards": len(set(card_hashes)),
        "distinct_pair_combinations": {
            "region": len(set(region_pairs)),
            "card": len(set(card_pairs)),
        },
        "distinct_note": (
            "「画面」与「画面组合」各去重一次，**两个键都报**：`region` 是这块取区裁出来的"
            "那一块（判据真正看到的东西），`card` 是整张备战席卡。两边数的都是**喂给尺子的"
            "那一态**（罩着端是反解之后的卡、干净端是原样那张卡）—— 与 `预处理.json` 的"
            "19 / 13（那两个数按**原样**的裁剪算）不是同一把键，别混。"
        ),
    }


def main(argv: list[str]) -> int:
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 0
    only = set(argv[argv.index("--specs") + 1].split(",")) if "--specs" in argv else None

    batch = load_batch(BATCH)
    harvest = json.loads(HARVEST.read_text(encoding="utf-8"))
    pairs = list(harvest["pairs"])
    if limit:
        order: list[str] = []
        for pair in pairs:
            if pair["capped"]["key"] not in order:
                order.append(pair["capped"]["key"])
        keep = set(order[:limit]) | {
            pair["capped"]["key"] for pair in pairs if pair["set"] == "calibration"
        }
        pairs = [pair for pair in pairs if pair["capped"]["key"] in keep]

    blocks: dict[str, np.ndarray] = {}
    for case in batch.cases:
        frame = frame_from_raw(case.raw())
        x0, y0 = int(case.rect.x), int(case.rect.y)
        blocks[case.name] = np.ascontiguousarray(
            frame.pixels[y0 : y0 + CARD[0], x0 : x0 + CARD[1]]
        )

    calibration = [pair for pair in pairs if pair["set"] == "calibration"]
    if len(calibration) != 2:
        raise ValueError("标定对要是 2 个，收到 %d" % len(calibration))
    model = fit_redcap(
        [
            (
                blocks[Path(pair["capped"]["file"]).stem],
                blocks[Path(pair["clean"]["file"]).stem],
            )
            for pair in calibration
        ]
    )
    faces = load_cardfaces()

    cards = {}
    for pair in pairs:
        for side in ("capped", "clean"):
            stem = Path(pair[side]["file"]).stem
            if stem not in cards:
                cards[stem] = {
                    "clean": blocks[stem],
                    "capped": undo_redcap(blocks[stem], model),
                }

    rows: list[dict] = []
    timings: list[tuple[str, float]] = []
    cache: dict = {}
    """同一行里，**取区逐字节相同就复用读数** —— 44 对里罩着端只有 19 块、干净端只有 8 块
    互不相同（同一时段的连拍逐字节一样），尺子是确定性的。不复用的话每一行要跑 88 次，
    而那 88 次里有 61 次是同一块画面。
    """
    for spec in _specs():
        if only is not None and spec["name"] not in only:
            continue
        region = _region_spec(spec)
        box = tuple(region["box"])
        started = time.perf_counter()
        ruler = None
        if region["mask"] is not None:
            ruler = MaskedRuler(region["mask"], faces)
        row = {
            "name": spec["name"],
            "family": spec["family"],
            "how": spec["how"],
            "params": {
                "mode": spec["mode"],
                "box": list(box),
                "mask_pixels": region["pixels"],
                "radius": spec.get("radius"),
                "strip_half": spec.get("half"),
            },
            "usable_scales": list(ruler.usable_scales) if ruler is not None else list(
                tuple(round(1.05 + i * 0.01, 3) for i in range(11))
            ),
            "ends": {},
        }
        for side in ("capped", "clean"):
            per: list[dict] = []
            region_hashes: list[str] = []
            card_hashes: list[str] = []
            region_pairs: list[str] = []
            card_pairs: list[str] = []
            for pair in pairs:
                stem = Path(pair[side]["file"]).stem
                query = _crop(cards[stem][side], box)
                region_hashes.append(_sha(query))
                card_hashes.append(_sha(cards[stem][side]))
                capped_picture = cards[Path(pair["capped"]["file"]).stem]["capped"]
                clean_picture = cards[Path(pair["clean"]["file"]).stem]["clean"]
                region_pairs.append(
                    _sha(_crop(capped_picture, box)) + "-" + _sha(_crop(clean_picture, box))
                )
                card_pairs.append(_sha(capped_picture) + "-" + _sha(clean_picture))
                cache_key = (spec["name"], query.tobytes())
                ranked = cache.get(cache_key)
                if ranked is None:
                    if ruler is None:
                        ranked = tuple(
                            (float(score), name, float(scale))
                            for score, name, scale in roster_scores(query, faces)
                        )
                    else:
                        ranked = tuple(
                            (float(score), name, float(scale))
                            for score, name, scale in ruler.scores(query)
                        )
                    cache[cache_key] = ranked
                truth = pair["under_the_cap"]
                truth_scores = [e[0] for e in ranked if e[1] == truth]
                if len(truth_scores) != 1:
                    raise ValueError("库里的真值 %r 在读数里不是恰一条" % truth)
                truth_score = float(truth_scores[0])
                wrong_score, wrong_name = max(
                    [(e[0], e[1]) for e in ranked if e[1] != truth],
                    key=lambda item: item[0],
                )
                per.append(
                    {
                        "key": pair["capped"]["key"],
                        "slot": pair["capped"]["index"],
                        "set": pair["set"],
                        "truth": truth,
                        "truth_score": round(truth_score, 6),
                        "wrong_score": round(wrong_score, 6),
                        "wrong_name": wrong_name,
                        "margin": round(truth_score - wrong_score, 6),
                        "winner_score": float(ranked[0][0]),
                        "winner_name": ranked[0][1],
                        "winner_is_truth": bool(truth_score >= ranked[0][0]),
                    }
                )
            row["ends"][side] = _end_summary(
                per, region_hashes, card_hashes, region_pairs, card_pairs
            )
            row["ends"][side]["per_pair"] = per
        timings.append((spec["name"], time.perf_counter() - started))
        rows.append(row)
        print("  [%6.1f s] %s" % (timings[-1][1], spec["name"]), flush=True)

    # 「库里与备战席各自被不同背景填」那一条的**旁证**（不是证明）：同一干员在两个不同
    # 卡位上的干净卡逐像素相减 —— 卡身要是有一块「被另一种背景填」，这一减就会露出来。
    by_operator: dict[str, dict[int, str]] = {}
    for pair in pairs:
        by_operator.setdefault(pair["under_the_cap"], {}).setdefault(
            int(pair["clean"]["index"]), Path(pair["clean"]["file"]).stem
        )
    background_check: dict = {
        "note": (
            "**旁证，不是证明**：同一干员在两个**不同卡位**上的干净卡逐像素相减 —— 卡身上"
            "要是有「在库里被卡片背景填、在备战席被场地背景填」的那一块，这一减就会露出来。"
            "**它的前提是那两个卡位底下的战场不一样，而这一批裁剪只裁了卡、画布上没有战场"
            "像素**，所以那个前提在这里证不了（汇报 ③ 按「这条我没法自证」报）。它只回答"
            "「这一减看得见看不见那样一块」。"
        ),
    }
    for operator, slots in sorted(by_operator.items(), key=lambda item: item[0]):
        if len(slots) < 2:
            continue
        first, second = sorted(slots)[:2]
        left = cards[slots[first]]["clean"].astype(np.int16)
        right = cards[slots[second]]["clean"].astype(np.int16)
        difference = np.abs(left - right).max(axis=2)
        check_box, check_mask = identity_disc_region()
        x0, y0, width, height = check_box
        inside = difference[y0 : y0 + height, x0 : x0 + width]
        background_check["pair"] = {
            "operator": operator,
            "slots": [int(first), int(second)],
            "files": [slots[first], slots[second]],
            "whole_card": {
                "pixels_over_40": int((difference > 40).sum()),
                "pixels": int(difference.size),
                "max": int(difference.max()),
                "above_the_top_strip_over_40": int(
                    (difference[:TOP_STRIP_HEIGHT] > 40).sum()
                ),
                "below_the_top_strip_over_40": int(
                    (difference[TOP_STRIP_HEIGHT:] > 40).sum()
                ),
            },
            "region": {
                "pixels_over_40": int((inside[check_mask] > 40).sum()),
                "pixels": int(check_mask.sum()),
                "max": int(inside[check_mask].max()),
            },
        }
        break

    if only is not None:
        return 0

    by_name = {row["name"]: row for row in rows}
    chosen = "圆盘取区·半径%d" % IDENTITY_DISC_RADIUS
    chosen_half = "圆盘取区·中央条半高%d" % IDENTITY_DISC_STRIP_HALF

    published = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    published_by = {row["name"]: row for row in published["rows"]}
    self_check = {
        "note": (
            "并排的不是两套口径：今天那套那两行由本脚本用 `" + "margin.MaskedRuler" + "` / "
            "`roster_scores` 复量一遍，与 `绝对线.json` 里同名那两行比端点；"
            "差为 0.0 才算并排的是同一条链。"
        )
    }
    for name in ("基线-掩码552", "基线-不掩码10000"):
        mine = by_name[name]["ends"]
        theirs = published_by[name]["ends"]
        self_check[name] = {
            side: {
                "min_truth_mine": mine[side]["min_truth"]["score"],
                "min_truth_published": theirs[side]["min_truth"]["score"],
                "max_wrong_mine": mine[side]["max_wrong"]["score"],
                "max_wrong_published": theirs[side]["max_wrong"]["score"],
                "ratio_mine": mine[side]["ratio"],
                "ratio_published": theirs[side]["ratio"],
            }
            for side in ("capped", "clean")
        }

    def _swing(rows_subset: list[dict], knob: str) -> dict:
        out = {}
        for row in rows_subset:
            out[str(row["params"][knob])] = {
                side: {
                    "min_truth": row["ends"][side]["min_truth"]["score"],
                    "max_wrong": row["ends"][side]["max_wrong"]["score"],
                    "ratio": row["ends"][side]["ratio"],
                    "median_margin": row["ends"][side]["median_margin"],
                }
                for side in ("capped", "clean")
            }
        return out

    # 余量的**主证据**：环带一环一环有多亮 —— 取区取到哪一环才碰到环。
    ladder_rows = np.arange(CARD[0], dtype=np.float64)[:, None]
    ladder_columns = np.arange(CARD[1], dtype=np.float64)[None, :]
    ladder_distance = np.sqrt(
        (ladder_columns - CARD[1] / 2.0) ** 2 + (ladder_rows - RING_CENTER_DY) ** 2
    )
    ladder_outside = np.abs(ladder_rows - RING_CENTER_DY) >= float(IDENTITY_DISC_STRIP_HALF)
    distinct_capped: list[str] = []
    for pair in pairs:
        stem = Path(pair["capped"]["file"]).stem
        if stem not in distinct_capped:
            distinct_capped.append(stem)
    ring_profile = {}
    for radius in range(44, 61):
        band = (ladder_distance >= radius) & (ladder_distance < radius + 1) & ladder_outside
        if not bool(band.any()):
            continue
        brightest = 0.0
        for stem in distinct_capped:
            values = blocks[stem].min(axis=2).astype(np.float64)[band]
            brightest = max(brightest, float(values.max()))
        ring_profile[str(radius)] = {
            "band": "%d <= distance < %d" % (radius, radius + 1),
            "pixels_in_band": int(band.sum()),
            "brightest_min_channel": brightest,
        }
    innermost_bright = []
    for stem in distinct_capped:
        values = blocks[stem].min(axis=2).astype(np.float64)
        hit = (values > 200) & ladder_outside
        if bool(hit.any()):
            innermost_bright.append(round(float(ladder_distance[hit].min()), 3))
    mask_leak = {}
    for radius in RADIUS_LADDER:
        box, mask = identity_disc_region(radius=radius)
        x0, y0, width, height = box
        brightest = 0.0
        for stem in distinct_capped:
            block = blocks[stem][y0 : y0 + height, x0 : x0 + width]
            values = block.min(axis=2).astype(np.float64)[mask]
            brightest = max(brightest, float(values.max()))
        mask_leak[str(radius)] = {
            "mask_pixels": int(mask.sum()),
            "brightest_min_channel_in_mask": brightest,
        }

    radius_rows = [row for row in rows if row["family"] == "disc"
                   and row["params"]["strip_half"] == IDENTITY_DISC_STRIP_HALF]
    strip_rows = [row for row in rows if row["family"] == "disc"
                  and row["params"]["radius"] == IDENTITY_DISC_RADIUS]
    ladder = {
        "question": (
            "决定取区形状的那个量挪一档，读数摆多少？哪一档算不摆？"
        ),
        "deciding_quantity": "IDENTITY_DISC_RADIUS",
        "deciding_quantity_note": (
            "「实际取多少」由半径定，所以主梯子是半径（挪一档 = 1 px）。中央条那一栏是"
            "第二把梯子（挪一档 = 1 行）—— 它的高低由秒数自己的字形定，不是自由选择，"
            "但同样要报出来证明它也不是悬的。"
        ),
        "radius": _swing(radius_rows, "radius"),
        "strip_half": _swing(strip_rows, "strip_half"),
        "chosen": {
            "radius": IDENTITY_DISC_RADIUS,
            "strip_half": IDENTITY_DISC_STRIP_HALF,
            "box": list(IDENTITY_DISC_BOX),
            "mask_pixels": int(identity_disc_mask().sum()),
        },
        "ring_profile": ring_profile,
        "ring_profile_note": (
            "环带**一环一环**有多亮：只看中央那一条之外，逐环取 min(R,G,B) 在 19 张互不"
            "相同的罩着卡上的最大值。它就是「半径取到哪一环才碰到环」的直接证据。"
        ),
        "innermost_bright_ring_px": {
            "criterion": "min(R,G,B) > 200 的环形像素离环心最近多远，逐张算",
            "per_card": sorted(innermost_bright),
        },
        "mask_leak": mask_leak,
        "mask_leak_note": (
            "每一档半径上**掩码里**最亮的 min(R,G,B)：环的像素进取区了没有，看这一栏。"
        ),
    }
    ladder["radius_swing_vs_chosen"] = {}
    for label, block in ladder["radius"].items():
        base = ladder["radius"][str(IDENTITY_DISC_RADIUS)]
        ladder["radius_swing_vs_chosen"][label] = {
            side: {
                "delta_min_truth": round(
                    block[side]["min_truth"] - base[side]["min_truth"], 6
                ),
                "delta_max_wrong": round(
                    block[side]["max_wrong"] - base[side]["max_wrong"], 6
                ),
                "delta_ratio": round(block[side]["ratio"] - base[side]["ratio"], 6),
                "delta_median_margin": round(
                    block[side]["median_margin"] - base[side]["median_margin"], 6
                ),
            }
            for side in ("capped", "clean")
        }
    ladder["strip_swing_vs_chosen"] = {}
    for label, block in ladder["strip_half"].items():
        base = ladder["strip_half"][str(IDENTITY_DISC_STRIP_HALF)]
        ladder["strip_swing_vs_chosen"][label] = {
            side: {
                "delta_min_truth": round(
                    block[side]["min_truth"] - base[side]["min_truth"], 6
                ),
                "delta_max_wrong": round(
                    block[side]["max_wrong"] - base[side]["max_wrong"], 6
                ),
                "delta_ratio": round(block[side]["ratio"] - base[side]["ratio"], 6),
                "delta_median_margin": round(
                    block[side]["median_margin"] - base[side]["median_margin"], 6
                ),
            }
            for side in ("capped", "clean")
        }

    report = {
        "note": (
            "取区换成「环带内缘以内的圆盘 − 中央秒数那一条」。比值 = "
            "(1 - 真值最低分) / (1 - 次高最高分)，< 1 才叫分得开。判「够 / 不够」不在这份"
            "文件里。线上那条链一个字节没动。"
        ),
        "batch": "tests/帧样本/预处理",
        "harvest": "tests/帧样本/预处理/取材.json",
        "published_side_by_side": "tests/帧样本/预处理/绝对线.json",
        "library": harvest["library"],
        "criterion": published["criterion"],
        "criterion_note": (
            "criterion 整块随 绝对线.json 走（同一批、同一个口径）；那一块里写明了"
            "**这一族文件不落判定** —— 逐对只有分数与余量，判定由读者按 roster 的两条线算。"
        ),
        "chains": {
            "full": "TM_CCOEFF_NORMED —— 线上那条链 roster_scores",
            "masked": (
                "逐通道减均值的掩码相关（TM_CCORR + 预存 sum(t) / sum(t^2)）——"
                "与 ark.engine.vision.margin.MaskedRuler 同一份公式"
            ),
        },
        "region": {
            "shape": (
                "圆心在卡内 `(卡宽/2, RING_CENTER_DY)`、半径 `IDENTITY_DISC_RADIUS`"
                " 的圆盘（**不是** `bench.RING_INNER`：那圈白环往内多压了约 1.5 px），"
                "挖掉中央 `IDENTITY_DISC_STRIP_HALF` 行（秒数所占的那一条）。"
                "一处定义：`ark.engine.vision.margin.identity_disc_mask`。"
            ),
            "radius": IDENTITY_DISC_RADIUS,
            "strip_half": IDENTITY_DISC_STRIP_HALF,
            "box": list(IDENTITY_DISC_BOX),
            "pixels": int(identity_disc_mask().sum()),
            "strip_band_rows": [
                int(np.ceil(RING_CENTER_DY - IDENTITY_DISC_STRIP_HALF)),
                int(np.floor(RING_CENTER_DY + IDENTITY_DISC_STRIP_HALF)),
            ],
            "today": {
                "box": list(BOX),
                "box_pixels": BOX[2] * BOX[3],
                "mask_radius": OLD_MASK_RADIUS,
                "mask_pixels": int(free_mask(BOX, radius=OLD_MASK_RADIUS).sum()),
            },
            "origin": (
                "这一帧像素读出来的卡心（`squad_card_centres`）减 102；纵向是契约层"
                " `squad_card_rect` 的 y=1045。不是常量网格 —— 与 `预处理.json` 同一批原点。"
            ),
        },
        "model": {
            "fit_from": [
                pair["capped"]["key"] + " -> " + pair["clean"]["key"]
                for pair in calibration
            ],
            "slope": model.slope,
            "offset": list(model.offset),
            "parameters": model.parameters,
            "pairs": model.pairs,
        },
        "self_check": self_check,
        "background_fill_check": background_check,
        "margin": ladder,
        "rows": rows,
    }
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print()
    print("=== 并排：今天那套（复量 vs 绝对线.json）===")
    for name in ("基线-掩码552", "基线-不掩码10000"):
        for side in ("capped", "clean"):
            got = by_name[name]["ends"][side]
            want = published_by[name]["ends"][side]
            print(
                "  %-18s %-6s 真值最低 %.4f/%.4f 次高最高 %.4f/%.4f 比值 %.4f/%.4f"
                % (
                    name,
                    side,
                    got["min_truth"]["score"],
                    want["min_truth"]["score"],
                    got["max_wrong"]["score"],
                    want["max_wrong"]["score"],
                    got["ratio"],
                    want["ratio"],
                )
            )
    print()
    print("=== 主梯子：半径（挪一档 = 1 px）===")
    for label in map(str, RADIUS_LADDER):
        block = ladder["radius"][label]
        mark = " <- 定下" if int(label) == IDENTITY_DISC_RADIUS else ""
        print(
            "  r=%s  罩着 %s -> %s 比值 %6.4f | 干净 %s -> %s 比值 %6.4f%s"
            % (
                label,
                "%.4f" % block["capped"]["min_truth"],
                "%.4f" % block["capped"]["max_wrong"],
                block["capped"]["ratio"],
                "%.4f" % block["clean"]["min_truth"],
                "%.4f" % block["clean"]["max_wrong"],
                block["clean"]["ratio"],
                mark,
            )
        )
    print()
    print("=== 第二梯子：中央条半高（挪一档 = 1 行）===")
    for label in map(str, STRIP_LADDER):
        block = ladder["strip_half"][label]
        mark = " <- 定下" if int(label) == IDENTITY_DISC_STRIP_HALF else ""
        print(
            "  half=%s  罩着 %s -> %s 比值 %6.4f | 干净 %s -> %s 比值 %6.4f%s"
            % (
                label,
                "%.4f" % block["capped"]["min_truth"],
                "%.4f" % block["capped"]["max_wrong"],
                block["capped"]["ratio"],
                "%.4f" % block["clean"]["min_truth"],
                "%.4f" % block["clean"]["max_wrong"],
                block["clean"]["ratio"],
                mark,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
