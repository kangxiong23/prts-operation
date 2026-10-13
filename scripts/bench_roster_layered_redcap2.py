# -*- coding: utf-8 -*-
r"""乙：红罩那条路的**多层细化** —— 掩码内抽样 · top-k 候选 · 覆盖整格 · 收位置窗。

    .venv\Scripts\python.exe scripts\bench_roster_layered_redcap2.py

## 上一单那次为什么不算数

上一单的粗层是**把整张图缩到 25%**（INTER_AREA）—— 环与数字被平均进圆盘里，
掩码只剩 ~249 px，挑档准只有 46%；细层窗只给 ±2 档；位置干脆没收。三条都是设计缺陷。

## 这一版

| | 干什么 |
| --- | --- |
| 粗层 | **点抽样 1/4**（立绘与掩码用**同一张格子** `art[::4, ::4]` / `mask[::4, ::4]`）—— 抽中的点就是掩码自己的像素，**环与数字一个都不进来**；档网格 1.05…1.15 步 0.02（6 档）；整张立绘找位置 |
| 候选 | 粗层分最高的 **k 个条目**，每个带**它自己那一档** |
| 覆盖整格 | 每个候选的**整格**（±0.02 / 2 = ±0.01）都由细层扫到 —— 细层档窗 = 候选档 ±1 档的**并集** |
| 位置 | 粗层给出位置（用它自己的预存表，与 `scores` 自检同值），细层把立绘**裁到 ±12 px** 的窗里再滑 |
| 细层 | 全分辨率 · 只跑并集那几档 —— **报出的分只来自这一层**。两个口径：**全库 11 条**（余量与今天可比）与 **top-k 只 3 条** |

**为什么两个口径都要报**：只在 top-k 里比，**次高分**可能不是真的次高 —— 余量被抬高，
「未知」会假性变成「精确」。所以两个口径的判定账并排报，**差距就是这一刀买来的假事实**。

## 基线那一侧**复用落盘**，不重跑

`tests/帧样本/红罩/点名.json`：红罩 28 张 · 精确 22 · 未知 6，外加它自己的
`cost`（`build_ruler_s = 5.4828` · `one_red_card_s = 5.3029`）。另外**同机同刻**
把今天那条路在 **3 张卡**上重跑一遍当标定（不是基线重跑，是自检）。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import bench_roster_coarse as base  # noqa: E402

from ark.engine.device.adb import SQUAD_CARD_PITCH as PITCH  # noqa: E402
from ark.engine.vision import bench  # noqa: E402
from ark.engine.vision import margin as margin_module  # noqa: E402
from ark.engine.vision.frame import frame_from_raw  # noqa: E402
from ark.engine.vision.roster import (  # noqa: E402
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
    redcap_region,
)
from ark.offline.assets_gen.cardface import load_cardfaces  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402

SAMPLES = base.SAMPLES
OUT_DIR = SAMPLES / "先粗后细"
OUT = OUT_DIR / "红罩细化2.json"
LEDGER = SAMPLES / "红罩" / "点名.json"
NARROW = base.NARROW
STRIDE = 4
COARSE_STEP = 0.02
TOP_K = 3
POS_HALF = 12
ROUNDS = 3
CALIBRATION_CARDS = 3
LIBRARY_MODES = ("全库 11 条", "top-3 只 3 条")


def _coarse_grid(step: float) -> tuple:
    lo, hi = min(NARROW), max(NARROW)
    count = int(round((hi - lo) / step))
    grid = [round(lo + index * step, 2) for index in range(count + 1)]
    if grid[-1] < hi - 1e-9:
        grid.append(round(hi, 2))
    return tuple(sorted(set(grid)))


class Counting:
    def __init__(self) -> None:
        self.calls = 0
        self._real = cv2.matchTemplate

    def __enter__(self):
        counter = self

        def counted(source, template, method):
            counter.calls += 1
            return counter._real(source, template, method)

        cv2.matchTemplate = counted
        return self

    def __exit__(self, *_exc):
        cv2.matchTemplate = self._real
        return False


def _verdict(ranked) -> dict:
    top_score, top_name, top_scale = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else -2.0
    second_name = ranked[1][1] if len(ranked) > 1 else None
    margin = top_score - second
    exact = (top_score >= ROSTER_SOLID_MATCH_SCORE
             and margin >= ROSTER_AMBIGUITY_MARGIN)
    return {"name": top_name, "score": round(top_score, 6), "scale": top_scale,
            "runner_up": second_name, "margin": round(margin, 6),
            "verdict": "精确" if exact else "未知"}


def _cell(scale: float) -> set:
    """一个候选那一**整格**：±(COARSE_STEP / 2)，落在那条链自己的 11 档上。"""
    half = COARSE_STEP / 2.0
    return {s for s in NARROW if abs(s - scale) <= half + 1e-9}


def _coarse_argmax(ruler, entry_name: str, region: np.ndarray):
    """粗尺子对**某一个条目**的相关图峰在哪 —— 用它自己的预存表，与 scores 自检同值。"""
    source = np.ascontiguousarray(region)
    best = (-2.0, 0, 0, None)
    for name, _art, levels in ruler._prepared:
        if name != entry_name:
            continue
        for scale, height, width, art_rgb, layer_f, pixels, channels in levels:
            query = cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)
            query = np.ascontiguousarray(query.astype(np.float32) / 255.0)
            masked = np.ascontiguousarray(query * layer_f[:, :, None])
            cross = cv2.matchTemplate(art_rgb, masked, cv2.TM_CCORR)
            numerator = np.array(cross, dtype=np.float64)
            template_energy = np.zeros_like(numerator)
            query_energy = 0.0
            for index in range(3):
                channel = np.ascontiguousarray(masked[:, :, index])
                mean = float(channel.sum()) / pixels
                image_sum, image_square = channels[index]
                numerator -= mean * image_sum
                template_energy += image_square - image_sum * image_sum / pixels
                query_energy += float((channel * channel).sum()) - pixels * mean * mean
            if query_energy <= 0.0:
                continue
            denominator = np.sqrt(query_energy) * np.sqrt(np.maximum(template_energy, 0.0))
            value_map = numerator / (denominator + 1e-12)
            spot = np.unravel_index(int(np.argmax(value_map)), value_map.shape)
            value = float(value_map[spot])
            if value > best[0]:
                best = (value, int(spot[1]), int(spot[0]), scale)
    return {"score": best[0], "x": best[1], "y": best[2], "scale": best[3]}


def _crop(art: np.ndarray, x: float, y: float, width: int, height: int, half: int) -> tuple:
    x0 = max(0, int(round(x - half)))
    y0 = max(0, int(round(y - half)))
    x1 = min(int(art.shape[1]), int(round(x + width + half)))
    y1 = min(int(art.shape[0]), int(round(y + height + half)))
    return x0, y0, x1, y1


def _one_card(coarse, record, faces, mask, mode: str, boxes: dict):
    small_region = np.ascontiguousarray(record["region"][::STRIDE, ::STRIDE])
    ranked = coarse.scores(small_region)
    chosen = list(ranked[:TOP_K])
    window = set()
    anchor = None
    for _score, name, scale in chosen:
        where = _coarse_argmax(coarse, name, small_region)
        window |= _cell(scale)
        if anchor is None:
            anchor = (where["x"] * STRIDE, where["y"] * STRIDE)
    window = tuple(sorted(window))
    biggest = max(window)
    height = int(round(mask.shape[0] * biggest))
    width = int(round(mask.shape[1] * biggest))
    box = _crop(faces[0][1], anchor[0], anchor[1], width, height, POS_HALF)
    boxes[window] = box
    x0, y0, x1, y1 = box
    if mode.startswith("top"):
        names = {c[1] for c in chosen}
        subset = [(n, a) for n, a in faces if n in names]
    else:
        subset = list(faces)
    cropped = [(n, np.ascontiguousarray(a[y0:y1, x0:x1])) for n, a in subset]
    ruler = margin_module.MaskedRuler(mask, cropped, scales=window)
    return {"chosen": [{"name": n, "score": round(sc, 6), "scale": s} for sc, n, s in chosen],
            "window": list(window), "crop": [x0, y0, x1, y1],
            "library_scored": len(cropped),
            "got": _verdict(ruler.scores(record["region"]))}


def main() -> int:
    started = time.perf_counter()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    slots = []
    for frame in ledger["frames"]:
        for slot in frame["slots"]:
            copy = dict(slot)
            copy["frame"] = frame["name"]
            slots.append(copy)
    by_slot = {(s["frame"], s["index"]): s for s in slots}
    truth_of = {(s["frame"], s["index"]): s.get("truth") for s in slots}
    cost = ledger["cost"]
    today_s = cost["components"]["build_ruler_s"] + ledger["totals"]["red"] * cost["components"]["one_red_card_s"]
    print("=== 基线（复用落盘）===")
    print("  tests/帧样本/红罩/点名.json  red=%d 精确=%d 未知=%d"
          % (ledger["totals"]["red"], ledger["totals"]["exact"], ledger["totals"]["unknown"]))
    print("  它自己的 cost：建尺子 %.4f s + 28 x %.4f s = %.3f s"
          % (cost["components"]["build_ruler_s"], cost["components"]["one_red_card_s"], today_s))

    faces = load_cardfaces(base.SHARED_LIBRARY)
    box, mask = margin_module.identity_disc_region()
    grid = _coarse_grid(COARSE_STEP)
    small_mask = np.ascontiguousarray(mask[::STRIDE, ::STRIDE])
    small_faces = [(name, np.ascontiguousarray(art[::STRIDE, ::STRIDE])) for name, art in faces]
    print("  掩码：全分辨率 %d px（%s）-> 点抽样 1/%d 后 %d px（%s）"
          % (int(mask.sum()), tuple(mask.shape), STRIDE * STRIDE,
             int(small_mask.sum()), tuple(small_mask.shape)))
    print("  粗层档网格（步 %.2f）：%s" % (COARSE_STEP, grid))

    records = []
    for case in load_batch(SAMPLES / "红罩").cases:
        frame = frame_from_raw(case.raw())
        centres = bench.squad_card_centres(frame, pitch=PITCH)
        top_y = float(bench.squad_card_rect(0, PITCH).y)
        for index, (centre, _score) in enumerate(centres):
            if bench.redeploying(frame, centre, card_top=top_y) is not True:
                continue
            records.append({
                "frame": case.name, "index": index,
                "region": np.ascontiguousarray(frame.crop(redcap_region(centre, pitch=PITCH))),
            })
    print("  红罩卡 %d 张" % len(records))

    # ---- 标定：今天那条路在同机同刻的 3 张卡上（自检，不是基线重跑）----
    calibration = {}
    with Counting() as counting:
        began = time.perf_counter()
        today_ruler = margin_module.MaskedRuler(mask, faces, scales=NARROW)
        build_s = time.perf_counter() - began
        per_card = []
        check = []
        for record in records[:CALIBRATION_CARDS]:
            began = time.perf_counter()
            got = _verdict(today_ruler.scores(record["region"]))
            per_card.append(round(time.perf_counter() - began, 4))
            landed = by_slot[(record["frame"], record["index"])]
            check.append({"frame": record["frame"], "index": record["index"],
                          "mine": got,
                          "landed": {"name": landed["reading"]["name"],
                                     "score": landed["score"], "scale": landed["scale"]},
                          "delta": round(abs(got["score"] - landed["score"]), 9)})
        calibration = {"build_s": round(build_s, 4), "cards_s": per_card,
                       "median_card_s": round(statistics.median(per_card), 4),
                       "calls": counting.calls, "check": check,
                       "note": "同机同刻重跑今天的红罩路，%d 张 —— 标定用，不是基线重跑"
                               % CALIBRATION_CARDS}
    print("  标定（今天那条路 · 同机同刻 %d 张）：建尺子 %.3f s · 每张 %s（中位 %.3f s）"
          % (CALIBRATION_CARDS, calibration["build_s"], calibration["cards_s"],
             calibration["median_card_s"]))

    # ---- 粗尺子（自检）----
    with Counting() as counting:
        coarse = margin_module.MaskedRuler(small_mask, small_faces, scales=grid)
        coarse_build = counting.calls
    small_region = np.ascontiguousarray(records[0]["region"][::STRIDE, ::STRIDE])
    ranked = coarse.scores(small_region)
    where = _coarse_argmax(coarse, ranked[0][1], small_region)
    self_check = {"scores_top": [round(ranked[0][0], 9), ranked[0][1], ranked[0][2]],
                  "argmax": {"score": round(where["score"], 9), "x": where["x"],
                             "y": where["y"], "scale": where["scale"]},
                  "delta": round(abs(ranked[0][0] - where["score"]), 12)}
    print("  自检（粗层自己的峰）：%s" % json.dumps(self_check, ensure_ascii=False))

    # ---- 两个口径各跑一遍 ----
    results = {}
    for mode in LIBRARY_MODES:
        rounds = []
        rows = []
        calls = 0
        for round_index in range(ROUNDS):
            began = time.perf_counter()
            with Counting() as counting:
                per_round = []
                for record in records:
                    got = _one_card(coarse, record, faces, mask, mode, {})
                    landed = by_slot[(record["frame"], record["index"])]
                    per_round.append({
                        "frame": record["frame"], "index": record["index"],
                        "window": got["window"], "crop": got["crop"],
                        "library_scored": got["library_scored"],
                        "coarse_top": got["chosen"],
                        "landed_name": landed["reading"]["name"],
                        "landed_runner_up": landed.get("runner_up"),
                        "landed_scale": landed["scale"],
                        "landed_blocked_by": landed["blocked_by"],
                        "coarse_has_landed": any(c["name"] == landed["reading"]["name"]
                                                 for c in got["chosen"]),
                        "coarse_has_runner_up": any(c["name"] == landed.get("runner_up")
                                                    for c in got["chosen"]),
                        "landed_scale_in_window": any(abs(landed["scale"] - s) < 1e-9
                                                      for s in got["window"]),
                        "layered": got["got"],
                    })
            if round_index == 0:
                rows = per_round
                calls = counting.calls
            rounds.append(round(time.perf_counter() - began, 4))
        median = round(statistics.median(rounds), 4)
        before = [r for r in rows if r["landed_blocked_by"] is None]
        after = [r for r in rows if r["layered"]["verdict"] == "精确"]
        gained = [r for r in rows if r["landed_blocked_by"] is not None
                  and r["layered"]["verdict"] == "精确"]
        lost = [r for r in rows if r["landed_blocked_by"] is None
                and r["layered"]["verdict"] == "未知"]

        def bad(row, field):
            truth = truth_of.get((row["frame"], row["index"]))
            return truth is not None and row[field] != truth

        wrong_after = [r for r in after if bad(r, "layered")
                       and r["layered"]["name"] != truth_of[(r["frame"], r["index"])]]
        results[mode] = {
            "before": {"exact": len(before), "unknown": len(rows) - len(before)},
            "after": {"exact": len(after), "unknown": len(rows) - len(after)},
            "gained_n": len(gained), "lost_n": len(lost),
            "gained": [{"frame": r["frame"], "index": r["index"],
                        "landed": {"blocked_by": r["landed_blocked_by"],
                                   "name": r["landed_name"],
                                   "runner_up": r["landed_runner_up"]},
                        "layered": r["layered"],
                        "runner_up_in_topk": r["coarse_has_runner_up"]} for r in gained],
            "lost": [{"frame": r["frame"], "index": r["index"],
                      "landed": {"blocked_by": r["landed_blocked_by"]},
                      "layered": r["layered"]} for r in lost],
            "wrong_after_n": len(wrong_after),
            "wrong_rows": [{"frame": r["frame"], "index": r["index"],
                            "layered": r["layered"],
                            "truth": truth_of[(r["frame"], r["index"])]}
                           for r in wrong_after],
            "coarse_has_landed": sum(1 for r in rows if r["coarse_has_landed"]),
            "coarse_has_runner_up": sum(1 for r in rows if r["coarse_has_runner_up"]),
            "landed_scale_in_window": sum(1 for r in rows if r["landed_scale_in_window"]),
            "rounds_s": rounds, "median_s": median,
            "speedup_vs_landed_today": round(today_s / median, 3) if median else None,
            "speedup_vs_calibration": round(
                (calibration["build_s"] + len(rows) * calibration["median_card_s"]) / median,
                3) if median else None,
            "calls_one_pass": calls,
            "rows": rows,
        }
        print("  [%s] 判定 %d -> %d 精确 · 未知->精确 %d · 精确->未知 %d · 错名字 %d"
              % (mode, results[mode]["before"]["exact"], results[mode]["after"]["exact"],
                 results[mode]["gained_n"], results[mode]["lost_n"],
                 results[mode]["wrong_after_n"]))
        print("      粗层 top-3 含落盘赢家 %d/28 · 含落盘次高 %d/28 · 落盘那一档在窗里 %d/28"
              % (results[mode]["coarse_has_landed"], results[mode]["coarse_has_runner_up"],
                 results[mode]["landed_scale_in_window"]))
        print("      墙钟 %s -> 中位 %.3f s · 对落盘今天 %.1fx · 对标定今天 %.1fx · 一趟 %d 次"
              % (rounds, median, results[mode]["speedup_vs_landed_today"],
                 results[mode]["speedup_vs_calibration"], calls))

    windows = sorted({tuple(r["window"]) for r in results[LIBRARY_MODES[0]]["rows"]})
    payload = {
        "note": "乙：红罩那条路的多层细化（掩码内点抽样 + top-k 候选 + 覆盖整格 + 收位置窗）。"
                "只量不改。两个口径并排：全库（余量可与今天比）与 top-3（省得多但余量会被抬高）。",
        "design": {"stride": STRIDE, "sample_ratio": "1/%d" % (STRIDE * STRIDE),
                   "mask_pixels_full": int(mask.sum()),
                   "mask_pixels_coarse": int(small_mask.sum()),
                   "coarse_step": COARSE_STEP, "coarse_grid": list(grid),
                   "top_k": TOP_K, "position_half_px": POS_HALF,
                   "cell_half_steps": COARSE_STEP / 2.0 / 0.01,
                   "fine_windows_seen": [list(w) for w in windows],
                   "mask_box": list(box)},
        "baseline_source": {"file": "tests/帧样本/红罩/点名.json",
                            "red": ledger["totals"]["red"],
                            "exact": ledger["totals"]["exact"],
                            "unknown": ledger["totals"]["unknown"],
                            "build_ruler_s": cost["components"]["build_ruler_s"],
                            "one_red_card_s": cost["components"]["one_red_card_s"],
                            "today_s": round(today_s, 3), "reused": True},
        "lines": {"score": ROSTER_SOLID_MATCH_SCORE,
                  "margin": ROSTER_AMBIGUITY_MARGIN},
        "coarse_build_calls": coarse_build,
        "self_check": self_check,
        "calibration": calibration,
        "cards": len(records),
        "results": results,
        "seconds": round(time.perf_counter() - started, 3),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("总耗时 %.1f s" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
