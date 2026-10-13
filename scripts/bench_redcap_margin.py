"""量「红罩那条路要救，余量从哪来」——**掩码版取区 vs 最大矩形取区**，另加职业图标那一线。

    .venv\\Scripts\\python.exe scripts\\bench_redcap_margin.py

它读两批**进版本库**的样本（都在 `tests/帧样本/` 下，全是原生像素的备战席横带）：

    tests/帧样本/余量     14 帧干净卡（这一单新开的，见那里的 README）
    tests/帧样本/红罩     4 对「同一张卡一次罩着、一次干净」（上一单那一批，只读，不改）

**尺子不是线上那条**：`roster_scores` 一个字节没动，这个脚本用的是
`ark.engine.vision.margin.MaskedRuler`（同一统计量、换成「一块形状 + 一个掩码」）。
它**不接线上链条**，只在报告里给数。掩码全开时它与 `roster_scores` 逐位相同 ——
`tests/unit/test_vision_redcap_margin.py` 里那条基线自检钉的就是这件事。

**它落的那一份** `tests/帧样本/余量/余量.json` **逐字节可复现**：没有时间戳、
没有绝对路径、没有随机量。"和上次比一比"这句话才有意义。

**真值从哪来**：干净卡的身份 = **把它自己的身份区喂线上那条链**读出来的那一个
（两边都过 `ROSTER_SOLID_MATCH_SCORE` / `ROSTER_AMBIGUITY_MARGIN` 才算数）。
红罩那 4 对的真值沿用 `tests/帧样本/红罩/取材.json` 的口径：同一张卡后来干净了
那一帧、喂同一条链读出来的。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.engine.device.adb import SQUAD_CARD_PITCH as PITCH  # noqa: E402
from ark.engine.vision.bench import squad_card_centres  # noqa: E402
from ark.engine.vision.frame import frame_from_raw  # noqa: E402
from ark.engine.vision.margin import MaskedRuler, free_mask  # noqa: E402
from ark.engine.vision.redcap import fit_redcap, undo_redcap  # noqa: E402
from ark.engine.vision.roster import (  # noqa: E402
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
    identity_region,
    roster_scores,
)
from ark.offline.assets_gen.cardface import load_cardfaces  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402

BATCH = REPO / "tests" / "帧样本" / "余量"
CAPPED = REPO / "tests" / "帧样本" / "红罩"
OUT = BATCH / "余量.json"

MASK_BOX = (0, 47, 186, 167)
"""掩码版取区的**外框**：立绘区 `(0, 47, 204, 167)` 裁到 **186 宽**。

**为什么是 186 而不是 204**：库立绘只有 **214** 宽，而判据的尺度网格到 **1.15** ——
`204 x 1.15 = 235 > 214`，模板放不下，那一档就没了（`roster_scores` 会直接报错）。
`186 x 1.15 = 213.9 -> 214`，**十一档全在**。所以 186 是「这条尺子还能搜尺度」的上界，
不是随便取的。
"""

WIDE_BOX = (0, 47, 204, 167)
"""整块立绘区 —— 上一单报的 **22752** 个可用像素就是它。**一档尺度都搜不了**（只剩 1.05）。"""

RECT = (0, 177, 186, 37)
"""对照用的**最大那块矩形**：圆外、能过全部十一档尺度、面积最大的那一块（**6882** px）。

（整个圆外**面积**最大的矩形是 `(0, 177, 204, 37)` **7548** px，但它 204 宽，
只剩一档尺度 —— 那是被尺子逼出来的，不是内容差。两个数都落在这份 JSON 里。）
"""

RECT_PLANNER = (164, 47, 40, 167)
"""上一单那个 **6680** = `40 x 167`（右边条）。**在报告里并排放**，因为它就是被点名的那块。"""

RECT_LEFT = (0, 47, 43, 167)
"""左边条 **7181** px —— 圆外能放下的最宽的左边条（十一档全在）。"""

IDENTITY_BOX = (52, 76, 100, 100)
"""线上那条链的身份区（`roster.identity_region` 那个框）。"""

GLYPH_BOX = (44, 0, 62, 47)
"""顶上那一条里的**职业图标**块（卡内坐标）。它**不在**罩子那条仿射里（罩子 p ≈ 0.20，
这一块 p ≈ 0.48），但形状一样在。"""

GLYPH_IOU = 0.75
"""两个图标算**同职业**的 IoU 线。实测同职业对是 0.77 / 0.88 / 0.91，不同职业 <= 0.56。"""


def build_ruler(box, faces, *, masked=True, label=""):
    mask = free_mask(box) if masked else np.ones((box[3], box[2]), bool)
    if not masked and box[3] == 0:
        raise ValueError(label)
    ruler = MaskedRuler(mask, faces)
    return ruler, int(mask.sum())


def at(card, box):
    x0, y0, w, h = box
    return np.ascontiguousarray(card[y0 : y0 + h, x0 : x0 + w])


def reading(scores):
    runner = scores[1][0] if len(scores) > 1 else 0.0
    return {
        "score": round(scores[0][0], 6),
        "name": scores[0][1],
        "runner_up": round(runner, 6),
        "margin": round(scores[0][0] - runner, 6),
    }


def glyph(card):
    x0, y0, w, h = GLYPH_BOX
    block = card[y0 : y0 + h, x0 : x0 + w].astype(np.float32).mean(axis=2)
    block = cv2.resize(block, (64, 48), interpolation=cv2.INTER_AREA)
    binary = block < (block.mean() - 0.55 * block.std())
    binary = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return binary.astype(bool)


def iou(first, second):
    union = int((first | second).sum())
    return float((first & second).sum()) / union if union else 0.0


def card_of(frame, index):
    centre = squad_card_centres(frame, pitch=PITCH)[index][0]
    left = int(round(centre)) - 102
    return np.ascontiguousarray(frame.pixels[1045 : 1045 + 214, left : left + 204])


def main() -> int:
    faces = load_cardfaces()
    clean_ruler, clean_px = build_ruler(MASK_BOX, faces)
    wide_ruler, wide_px = build_ruler(WIDE_BOX, faces)
    rect_ruler, rect_px = build_ruler(RECT, faces, masked=False)
    planner_ruler, planner_px = build_ruler(RECT_PLANNER, faces, masked=False)
    left_ruler, left_px = build_ruler(RECT_LEFT, faces, masked=False)
    identity_ruler, identity_px = build_ruler(IDENTITY_BOX, faces)

    cards = []
    glyphs = {}
    frames = {}
    for case in load_batch(BATCH).cases:
        frame = frame_from_raw(case.raw())
        frames[case.name] = frame
        centres = squad_card_centres(frame, pitch=PITCH)
        for index in range(len(centres)):
            region = np.ascontiguousarray(frame.crop(identity_region(centres[index][0], pitch=PITCH)))
            online = roster_scores(region, faces)
            best = online[0][0]
            runner = online[1][0] if len(online) > 1 else 0.0
            if best < ROSTER_SOLID_MATCH_SCORE or best - runner < ROSTER_AMBIGUITY_MARGIN:
                continue
            name = online[0][1]
            card = card_of(frame, index)
            row = {
                "frame": case.name,
                "index": index,
                "name": name,
                "truth": "线上那条链读自己干净的身份区：" +
                         "%.4f / 次高 %.4f / 余量 %.4f" % (best, runner, best - runner),
                "mask": reading(clean_ruler.scores(at(card, MASK_BOX))),
                "rect": reading(rect_ruler.scores(at(card, RECT))),
                "rect_6680": reading(planner_ruler.scores(at(card, RECT_PLANNER))),
                "identity_masked": reading(identity_ruler.scores(at(card, IDENTITY_BOX))),
            }
            cards.append(row)
            glyphs.setdefault(name, []).append(glyph(card))
            print("%-6s #%-2d %-6s mask %.4f(%.4f) rect %.4f(%.4f) ident %.4f(%.4f)" % (
                case.name, index, name, row["mask"]["score"], row["mask"]["margin"],
                row["rect"]["score"], row["rect"]["margin"],
                row["identity_masked"]["score"], row["identity_masked"]["margin"]))

    references = {}
    for name, blocks in glyphs.items():
        stack = np.mean(np.stack(blocks).astype(np.float32), axis=0)
        references[name] = stack > 0.5
    names = sorted(references)
    groups = {}
    for name in names:
        peers = [other for other in names
                 if other != name and iou(references[name], references[other]) >= GLYPH_IOU]
        groups[name] = sorted(peers + [name])
    print("\n职业分组（二值字形 IoU >= %.2f）：" % GLYPH_IOU)
    for name in names:
        print("   %-8s -> %s" % (name, groups[name]))

    # ---- 红罩那 4 对：反解后的读数（甲组 2 的口径改正是 identity_masked 那一行）
    capped_cases = {case.name: frame_from_raw(case.raw()) for case in load_batch(CAPPED).cases}
    pairs = [
        ("cd14", 1, "cd15", 1, "calibration"),
        ("cd22", 9, "cd23", 9, "calibration"),
        ("cd27", 6, "cd28", 6, "validation"),
        ("cd37", 4, "cd59", 4, "validation"),
    ]
    calibration = [(card_of(capped_cases[a], i), card_of(capped_cases[b], j))
                   for a, i, b, j, s in pairs if s == "calibration"]
    model = fit_redcap(calibration)
    print("\n模型（只用标定那两对）：slope %.4f offset %s residual %.2f pixels %d params %d" % (
        model.slope, tuple(round(v, 2) for v in model.offset), model.residual, model.pixels, model.parameters))

    undo_rows = []
    for capped_name, capped_index, clean_name, clean_index, which in pairs:
        capped_card = card_of(capped_cases[capped_name], capped_index)
        solved = undo_redcap(capped_card, model)
        clean_card = card_of(capped_cases[clean_name], clean_index)
        solved_region = at(solved, IDENTITY_BOX)
        online = roster_scores(solved_region, faces)
        masked = reading(identity_ruler.scores(solved_region))
        mask_region = reading(clean_ruler.scores(at(solved, MASK_BOX)))
        capped_online = roster_scores(at(capped_card, IDENTITY_BOX), faces)
        truth_name = online[0][1]
        row = {
            "set": which,
            "capped": {"frame": capped_name, "index": capped_index},
            "clean": {"frame": clean_name, "index": clean_index},
            "under_the_cap": truth_name,
            "capped_identity": reading(capped_online),
            "solved_identity_full": reading(online),
            "solved_identity_masked": masked,
            "solved_mask_region": mask_region,
        }
        undo_rows.append(row)
        print("%-6s#%-2d -> %-6s#%-2d  %-6s | capped %.4f | solved full %.4f(%.4f) | solved masked %.4f(%.4f) | mask region %.4f(%.4f)" % (
            capped_name, capped_index, clean_name, clean_index, truth_name,
            row["capped_identity"]["score"], online[0][0], online[0][0] - online[1][0],
            masked["score"], masked["margin"], mask_region["score"], mask_region["margin"]))

    # ---- 路 B：盖着罩子时职业图标读不读得出来 + 缩库后的余量
    class_icon = {"readable": True, "evidence": [], "iou_line": GLYPH_IOU,
                  "groups": {name: groups[name] for name in names}}
    for capped_name, capped_index, clean_name, clean_index, which in pairs:
        capped_card = card_of(capped_cases[capped_name], capped_index)
        clean_card = card_of(capped_cases[clean_name], clean_index)
        block_capped = at(capped_card, GLYPH_BOX).astype(np.float64)
        block_clean = at(clean_card, GLYPH_BOX).astype(np.float64)
        slopes = []
        for channel in range(3):
            clean_channel = block_clean[:, :, channel].ravel()
            capped_channel = block_capped[:, :, channel].ravel()
            centred = clean_channel - clean_channel.mean()
            slopes.append(float((centred * (capped_channel - capped_channel.mean())).sum()
                                / (centred * centred).sum()))
        predicted = np.stack(
            [slopes[c] * block_clean[:, :, c] + (block_capped[:, :, c].mean()
             - slopes[c] * block_clean[:, :, c].mean()) for c in range(3)], axis=2)
        rms = float(np.sqrt(((block_capped - predicted) ** 2).mean()))
        first = block_capped.mean(axis=2).ravel()
        second = block_clean.mean(axis=2).ravel()
        correlation = float(((first - first.mean()) * (second - second.mean())).mean()
                            / (first.std() * second.std()))
        capped_glyph = glyph(capped_card)
        ranking = sorted(((iou(capped_glyph, references[n]), n) for n in names), reverse=True)
        truth_name = next(n for n in names if n == ranking[0][1]) if ranking[0][1] else None
        trimmed = groups[ranking[0][1]]
        subset = tuple(entry for entry in faces if entry[0] in trimmed)
        shrunk_ruler = MaskedRuler(free_mask(MASK_BOX), subset)
        solved = undo_redcap(capped_card, model)
        shrunk = reading(shrunk_ruler.scores(at(solved, MASK_BOX)))
        fake = next((r for r in undo_rows if r["capped"]["frame"] == capped_name), None)
        class_icon["evidence"].append({
            "capped": "%s#%d" % (capped_name, capped_index),
            "clean": "%s#%d" % (clean_name, clean_index),
            "slope": [round(v, 4) for v in slopes],
            "rms": round(rms, 3),
            "correlation": round(correlation, 5),
            "clean_peak": int(block_clean.mean(axis=2).max()),
            "capped_peak": int(block_capped.mean(axis=2).max()),
            "top": {"name": ranking[0][1], "iou": round(ranking[0][0], 4)},
            "runner_up": {"name": ranking[1][1], "iou": round(ranking[1][0], 4)},
            "distance": round(ranking[0][0] - ranking[1][0], 4),
            "shrunk_entries": sorted(trimmed),
            "shrunk": shrunk,
            "full": fake["solved_mask_region"] if fake else None,
        })
        print("   %-6s#%-2d icon IoU 最优 %-8s %.3f（次 %-8s %.3f）缩库到 %s -> %.4f(%.4f)" % (
            capped_name, capped_index, ranking[0][1], ranking[0][0],
            ranking[1][1], ranking[1][0], sorted(trimmed), shrunk["score"], shrunk["margin"]))

    payload = {
        "note": "红罩那条路的「余量从哪来」。掩码版取区（不规则）与最大矩形取区（对照）并排；"
                "尺子是 margin.MaskedRuler（线上那条链的统计量、换一块取区），线上链条一个字节没动。",
        "batch": "tests/帧样本/余量",
        "capped_batch": "tests/帧样本/红罩",
        "library": "tests/帧样本/点名/编队页.json",
        "statistic": "逐通道减均值的零均值归一化相关（与 roster_scores 的 TM_CCOEFF_NORMED 同值，"
                     "掩码全开时逐位相同）",
        "lines": {"score": ROSTER_SOLID_MATCH_SCORE, "margin": ROSTER_AMBIGUITY_MARGIN},
        "mask": {
            "boundary": "半径 60 的倒计时圆盘（bench.RING_OUTER）+ 顶上 47 px 那一条",
            "bbox": list(MASK_BOX),
            "pixels": int(clean_ruler.mask_pixels),
            "box_pixels": MASK_BOX[2] * MASK_BOX[3],
            "usable_scales": list(clean_ruler.usable_scales),
            "wide": {"bbox": list(WIDE_BOX), "pixels": int(wide_px),
                     "usable_scales": list(wide_ruler.usable_scales)},
        },
        "rect": {
            "bbox": list(RECT), "pixels": int(rect_px),
            "usable_scales": list(rect_ruler.usable_scales),
            "variants": [
                {"bbox": list(RECT_PLANNER), "pixels": int(planner_px),
                 "usable_scales": list(planner_ruler.usable_scales), "note": "上一单点名的 6680（右边条）"},
                {"bbox": list(RECT_LEFT), "pixels": int(left_px),
                 "usable_scales": list(left_ruler.usable_scales), "note": "圆外最宽的左边条"},
                {"bbox": list(WIDE_BOX)[:2] + [204, 37], "pixels": 7548,
                 "usable_scales": [1.05], "note": "圆外面积最大的矩形，但 204 宽 → 只剩一档尺度"},
            ],
        },
        "baseline": {
            "box": list(IDENTITY_BOX),
            "full": {"pixels": IDENTITY_BOX[2] * IDENTITY_BOX[3],
                     "note": "整块身份区（= 线上那条链的输入），掩码全开时 MaskedRuler 与 roster_scores 逐位相同"},
            "masked": {"pixels": int(identity_px),
                       "note": "只留排掉倒计时图形之后的那一块"},
        },
        "cards": cards,
        "undo": undo_rows,
        "class_icon": class_icon,
    }
    blob = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    OUT.write_bytes(blob.replace("\r\n", "\n").encode("utf-8"))
    print("\nwrote %s (%d bytes)" % (OUT, OUT.stat().st_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
