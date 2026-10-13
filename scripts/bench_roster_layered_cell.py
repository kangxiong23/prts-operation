# -*- coding: utf-8 -*-
r"""甲：在四层骨架上补两个洞 —— ① 粗层的档范围铺到真档够得到 ② 细层窗覆盖粗层那一整格。

    .venv\Scripts\python.exe scripts\bench_roster_layered_cell.py

## 两个洞

**洞①「39% 的赢家档贴在区间两头、22 张坐在窗沿」**：11 档的 argmax 被区间截住了 ——
真峰在 1.05 下面时，argmax 只能落在 1.05 这一个点上。粗层如果也只在 1.05…1.15 里找，
它的「最优」就是被截断的那个位置，往下传的窗也只能围着被截断的位置转。
**补法**：粗层铺到**宽区间 0.95…1.20**（`曲线.json` 里那 26 档），粗层的 argmax 就落在
它真正该在的地方；往下每一层的窗都围着这个**没有被截断**的估计转。

**洞②「细层窗要覆盖粗层那一整格」**：粗层按**步长**采样，真值最多离最近的采样点半格。
所以第 k+1 层的窗必须 >= 第 k 层的格 —— 这是**保证够得到**的规则，不是调参调出来的：
    scale_half(k+1) = ceil(step(k) / 0.01)          档
    pos_half(k+1)   = ceil(100 / pixels(k)) + 2     px   （一格粗像素 + 两像素取整）

## 三个走法

| | 粗层区间 | 粗层步长 | 报出的分 |
| --- | --- | --- | --- |
| 甲-1 铺满·放开 | 0.95…1.20 | 0.03 | 窗里所有档（可能落到 1.05…1.15 之外 —— 那是**改链**） |
| 甲-2 铺满·锁 11 档 | 0.95…1.20 | 0.03 | 窗 ∩ 1.05…1.15；空了退回全 11 档 |
| 甲-3 不铺满·覆盖整格 | 1.05…1.15 | 0.03 | 窗 ∩ 1.05…1.15；空了退回全 11 档 |

真值一律是 `曲线.json` 的全分辨率 argmax（**不重跑**）：甲-1 用宽区间那 26 档的 argmax，
甲-2 / 甲-3 用 11 档的。今天那一侧照旧从 `曲线.json` 现算。
"""

from __future__ import annotations

import argparse
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
import bench_roster_layered as ref  # noqa: E402

SAMPLES = base.SAMPLES
OUT_DIR = SAMPLES / "先粗后细"
CURVES = OUT_DIR / "曲线.json"
READINGS = OUT_DIR / "读数.json"
OUT = OUT_DIR / "补洞.json"
PAIRS = REPO / "data" / "probe" / "cell_pairs.json"
NARROW = base.NARROW
WIDE = base.WIDE
ALL = tuple(sorted(set(NARROW + WIDE)))


class CountingTarget:
    """run() 里只数次数，不包装 cv2 —— 调用计数走 counter 列表。"""


def _grid(interval, step: float) -> tuple:
    lo, hi = min(interval), max(interval)
    count = int(round((hi - lo) / step))
    values = [round(lo + index * step, 2) for index in range(count + 1)]
    if values[-1] < hi - 1e-9:
        values.append(round(hi, 2))
    return tuple(sorted(set(values)))


def _window(center: float, half: int, interval) -> tuple:
    span = half * 0.01
    return tuple(s for s in interval if center - span - 1e-9 <= s <= center + span + 1e-9)


def _select(scales, allowed) -> tuple:
    inside = tuple(s for s in scales if any(abs(s - a) < 1e-9 for a in allowed))
    return inside if inside else tuple(allowed)


def _scale_half(step: float) -> int:
    return max(1, int(np.ceil(round(step / 0.01, 6))))


def _pos_half(pixels: int) -> int:
    return int(np.ceil(100.0 / pixels)) + 2


def plan_four(pixels, first_step: float, first_interval, allowed, lock_final) -> list:
    """按两条规则搭出四层。每层 = (pixels, step, interval, scale_half, pos_half, allowed, final)。"""
    plan = []
    for index, px in enumerate(pixels):
        final = index == len(pixels) - 1
        if index == 0:
            plan.append((px, first_step, first_interval, None, None, first_interval, False))
            continue
        step = first_step if index == 1 else 0.01
        interval = allowed if (final and lock_final) else None
        plan.append((px, 0.01, interval,
                     _scale_half(first_step if index == 1 else 0.01),
                     _pos_half(pixels[index - 1]),
                     allowed, final))
    return plan


def run(region, art, plan, counter: list) -> dict:
    """跑一遍多层。报出的分只来自最后一层。"""
    positions: dict = {}
    grid = None
    trace = []
    picked = None
    score = -2.0
    loc = (0.0, 0.0)
    searched_final = []
    for index, item in enumerate(plan):
        pixels, step, interval, scale_half, pos_half, allowed, final = item
        factor = pixels / float(region.shape[0])
        if pixels == 100:
            small_query, small_art = region, art
        else:
            small_query = cv2.resize(region, (pixels, pixels), interpolation=cv2.INTER_AREA)
            small_art = cv2.resize(
                art,
                (max(1, int(round(art.shape[1] * factor))),
                 max(1, int(round(art.shape[0] * factor)))),
                interpolation=cv2.INTER_AREA)
        if grid is None:
            grid = _grid(interval, step)
        rows = {}
        area = 0
        for scale in grid:
            height = int(round(small_query.shape[0] * scale))
            width = int(round(small_query.shape[1] * scale))
            if height > small_art.shape[0] or width > small_art.shape[1]:
                continue
            template = cv2.resize(small_query, (width, height), interpolation=cv2.INTER_LINEAR)
            if pos_half is None or not positions:
                block, x0, y0 = small_art, 0, 0
            else:
                anchor = ref._anchor(positions, scale)
                half = pos_half * factor
                x0 = max(0, int(round(anchor[0] * factor - half)))
                y0 = max(0, int(round(anchor[1] * factor - half)))
                x1 = min(int(small_art.shape[1]), int(round(anchor[0] * factor + width + half)))
                y1 = min(int(small_art.shape[0]), int(round(anchor[1] * factor + height + half)))
                block = np.ascontiguousarray(small_art[y0:y1, x0:x1])
            if block.shape[0] < height or block.shape[1] < width:
                print("    [guard] 裁出来的块 %s 装不下模板 %s —— 退回整张"
                      % (block.shape, (height, width)))
                block, x0, y0 = small_art, 0, 0
            counter[0] += 1
            area += int(block.shape[0]) * int(block.shape[1])
            result = cv2.matchTemplate(block, template, cv2.TM_CCOEFF_NORMED)
            _lo, hi, _loloc, hiloc = cv2.minMaxLoc(result)
            rows[scale] = (float(hi), (hiloc[0] + x0) / factor,
                           (hiloc[1] + y0) / factor)
        best_scale = max(rows, key=lambda key: (rows[key][0], -key))
        picked = best_scale
        score, x, y = rows[best_scale]
        loc = (x, y)
        trace.append({"pixels": pixels, "step": step, "scale_half": scale_half,
                      "pos_half": pos_half, "grid_size": len(rows),
                      "scales": sorted(rows), "picked": best_scale,
                      "area_px": area})
        positions = {key: (value[1], value[2]) for key, value in rows.items()}
        if index + 1 < len(plan):
            nxt = plan[index + 1]
            grid = sorted(set(_select(
                _window(best_scale, nxt[3], nxt[2] if nxt[2] else ALL), nxt[5])))
        else:
            searched_final = sorted(set(rows))
    return {"picked": picked, "score": score, "loc": loc, "trace": trace,
            "searched_final": searched_final}


def _plan_description(plan) -> list:
    out = []
    for pixels, step, interval, scale_half, pos_half, allowed, final in plan:
        out.append({"pixels": pixels, "factor": round(pixels / 100.0, 3), "step": step,
                    "scale_half": scale_half, "pos_half": pos_half,
                    "interval": list(interval) if interval else ("宽区间" if allowed is WIDE else "窄区间"),
                    "first_grid": list(_grid(interval, step)) if interval else None,
                    "final": final})
    return out


FAMILY = {
    "甲-D 对照：旧四层（11档步0.01 · ±3/±2/±1 · 位12/8/6）": (
        (12, 0.01, NARROW, None, None, NARROW, False),
        (25, 0.01, None, 3, 12, NARROW, False),
        (50, 0.01, None, 2, 8, NARROW, False),
        (100, 0.01, NARROW, 1, 6, NARROW, True),
    ),
    "甲-A 旧四层 + 细层加宽到 ±2（位12/8/6）": (
        (12, 0.01, NARROW, None, None, NARROW, False),
        (25, 0.01, None, 3, 12, NARROW, False),
        (50, 0.01, None, 2, 8, NARROW, False),
        (100, 0.01, NARROW, 2, 6, NARROW, True),
    ),
    "甲-B 旧四层 + 细层加宽到 ±3（位12/8/6）": (
        (12, 0.01, NARROW, None, None, NARROW, False),
        (25, 0.01, None, 3, 12, NARROW, False),
        (50, 0.01, None, 2, 8, NARROW, False),
        (100, 0.01, NARROW, 3, 6, NARROW, True),
    ),
    "甲-C 粗层铺满 26 档步0.01 + 细层±2": (
        (12, 0.01, WIDE, None, None, WIDE, False),
        (25, 0.01, None, 3, 12, WIDE, False),
        (50, 0.01, None, 2, 8, WIDE, False),
        (100, 0.01, NARROW, 2, 6, NARROW, True),
    ),
    "甲-E 粗层铺满 26 档步0.03 + 细层±2": (
        (12, 0.03, WIDE, None, None, WIDE, False),
        (25, 0.01, None, 3, 12, WIDE, False),
        (50, 0.01, None, 2, 8, WIDE, False),
        (100, 0.01, NARROW, 2, 6, NARROW, True),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-pairs", action="store_true")
    parser.add_argument("--no-measure", action="store_true")
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    blob = json.loads(CURVES.read_text(encoding="utf-8"))
    readings = json.loads(READINGS.read_text(encoding="utf-8"))
    cards = ref._truth_cards(blob)
    corpus, libraries = base._corpus()
    truth_map = base._truth_map()
    print("=== 真值（复用落盘）=== 曲线 %d 条 / %d 张卡  来源 tests/帧样本/先粗后细/曲线.json"
          % (len(blob["keys"]), len(cards)))
    print("  语料 %d 张卡" % len(corpus))
    for label, plan in FAMILY.items():
        print("  %s 的层：%s" % (label, json.dumps(_plan_description(plan), ensure_ascii=False)))

    results = {}
    wanted = [label for label in FAMILY if args.only in label] if args.only else list(FAMILY)
    print("  这一轮只跑：%s" % wanted)
    for label in wanted:
        plan = FAMILY[label]
        began = time.perf_counter()
        per_pair = []
        verdict_rows = []
        calls = 0
        for record in corpus:
            faces = libraries[record["library"]]
            key = "%s|%s|%d" % (record["batch"], record["frame"], record["index"])
            here = cards[key]
            by_entry = {}
            today_by_entry = {}
            for name, art in faces:
                rows = ref._narrow_rows(here[name])
                wide_rows = ref._narrow_rows(here[name], WIDE)
                if not rows:
                    continue
                narrow_scale = max(rows, key=lambda row: (row[1], -row[0]))[0]
                wide_scale = max(wide_rows, key=lambda row: (row[1], -row[0]))[0]
                counter = [0]
                got = run(record["region"], art, plan, counter)
                calls += counter[0]
                per_pair.append({
                    "batch": record["batch"], "frame": record["frame"],
                    "index": record["index"], "entry": name,
                    "narrow_scale": narrow_scale, "wide_scale": wide_scale,
                    "picked": got["picked"], "reported": got["score"],
                    "narrow_steps": int(round(abs(got["picked"] - narrow_scale) / 0.01)),
                    "wide_steps": int(round(abs(got["picked"] - wide_scale) / 0.01)),
                    "final_grid": got["searched_final"],
                    "layer_grids": [t["scales"] for t in got["trace"]],
                    "layer_sizes": [t["grid_size"] for t in got["trace"]],
                    "today_score": max(row[1] for row in rows),
                    "trace": got["trace"] if args.dump_pairs else [],
                })
                by_entry[name] = {"score": got["score"], "scale": got["picked"],
                                  "loc": got["loc"]}
                best_row = max(rows, key=lambda row: row[1])
                today_by_entry[name] = {"score": best_row[1], "scale": narrow_scale,
                                        "loc": (best_row[2], best_row[3])}
            group = record["frame"][1] if record["batch"] == "立绘两组" else None
            truth = truth_map.get((record["frame"], record["index"]))
            before = base._verdict(today_by_entry, NARROW, truth, group)
            after = base._verdict(by_entry, NARROW, truth, group)
            verdict_rows.append({"batch": record["batch"], "frame": record["frame"],
                                 "index": record["index"], "truth": truth,
                                 "before": before, "after": after})

        by_card: dict = {}
        for row in per_pair:
            by_card.setdefault((row["batch"], row["frame"], row["index"]), []).append(row)
        edge_narrow = edge_wide = on_edge = 0
        for rows in by_card.values():
            winner_today = max(rows, key=lambda r: r["today_score"])
            winner_report = max(rows, key=lambda r: r["reported"])
            if winner_today["narrow_scale"] in (min(NARROW), max(NARROW)):
                edge_narrow += 1
            if winner_today["wide_scale"] in (min(WIDE), max(WIDE)):
                edge_wide += 1
            truth_here = winner_today["wide_scale"] if (label.startswith("甲-C") or label.startswith("甲-E")) else winner_today["narrow_scale"]
            grid = winner_report["final_grid"]
            if grid and (abs(truth_here - min(grid)) < 1e-9 or abs(truth_here - max(grid)) < 1e-9):
                on_edge += 1

        truth_key = "wide_scale" if (label.startswith("甲-C") or label.startswith("甲-E")) else "narrow_scale"
        coverage = []
        for depth in range(4):
            hit = 0
            total = 0
            for r in per_pair:
                if len(r["layer_grids"]) <= depth:
                    continue
                total += 1
                if any(abs(r[truth_key] - s) < 1e-9 for s in r["layer_grids"][depth]):
                    hit += 1
            coverage.append({"depth": depth + 1, "hit": hit, "total": total})
        steps = [r["wide_steps"] if truth_key == "wide_scale" else r["narrow_steps"] for r in per_pair]
        gained = [r for r in verdict_rows if r["before"]["verdict"] == "未知" and r["after"]["verdict"] == "精确"]
        lost = [r for r in verdict_rows if r["before"]["verdict"] == "精确" and r["after"]["verdict"] == "未知"]
        wrong_new = [r for r in gained if r["after"]["wrong"] is True]
        results[label] = {
            "plan": _plan_description(plan),
            "pairs": len(per_pair),
            "scale_exact": sum(1 for s in steps if s == 0),
            "scale_within_1": sum(1 for s in steps if s <= 1),
            "scale_within_2": sum(1 for s in steps if s <= 2),
            "cards": len(verdict_rows),
            "before": {"exact": sum(1 for r in verdict_rows if r["before"]["verdict"] == "精确")},
            "after": {"exact": sum(1 for r in verdict_rows if r["after"]["verdict"] == "精确")},
            "gained_n": len(gained), "lost_n": len(lost), "wrong_new_n": len(wrong_new),
            "gained": gained[:10], "lost": lost[:10],
            "name_moved_n": sum(1 for r in verdict_rows if r["before"]["name"] != r["after"]["name"]),
            "coverage_by_layer": coverage,
            "edge_narrow_n": edge_narrow, "edge_wide_n": edge_wide, "on_final_edge_n": on_edge,
            "cards_total": len(by_card),
            "calls_per_pair": round(calls / max(1, len(per_pair)), 3),
            "seconds": round(time.perf_counter() - began, 3),
            "per_pair": per_pair if args.dump_pairs else [],
        }
        print("  [%s] pairs=%d 选档准=%d 判定 %d->%d 丢=%d 得=%d 新错=%d 贴两头(11)=%d 贴两头(宽)=%d 坐窗沿=%d 次/对=%.2f  %.1fs"
              % (label, results[label]["pairs"], results[label]["scale_exact"],
                 results[label]["before"]["exact"], results[label]["after"]["exact"],
                 results[label]["lost_n"], results[label]["gained_n"],
                 results[label]["wrong_new_n"], edge_narrow, edge_wide, on_edge,
                 results[label]["calls_per_pair"], results[label]["seconds"]))
        if args.dump_pairs:
            PAIRS.write_text(json.dumps({"design": label, "results": results[label]},
                                        ensure_ascii=False), encoding="utf-8")

    today = readings["cost_measured"]["甲的今天：全分辨率全档"]
    chain_s = today["median_s"] / 2
    measure = {}
    if not args.no_measure:
        picks = readings["measure_sample"]["picks"]
        sample = [corpus[i] for i in picks]
        print("=== 墙钟（真跑 8 张卡 / 88 条库，3 次取中位）今天复用 %.4f s ===" % chain_s)
        for label in wanted:
            plan = FAMILY[label]
            counter = [0]
            times = []
            for _ in range(3):
                began = time.perf_counter()
                counter[0] = 0
                for record in sample:
                    for _name, art in libraries[record["library"]]:
                        run(record["region"], art, plan, counter)
                times.append(time.perf_counter() - began)
            median = statistics.median(times)
            measure[label] = {"match_template_calls": counter[0],
                              "median_s": round(median, 4),
                              "rounds_s": [round(t, 4) for t in times],
                              "ratio": round(chain_s / median, 2)}
            print("  %-44s %5d 次  %8.3f s  %5.1fx"
                  % (label, counter[0], median, chain_s / median))

    payload = {
        "note": "甲：补洞（粗层铺满 + 细层覆盖整格）三个走法。只量不改。",
        "truth": {"file": "tests/帧样本/先粗后细/曲线.json", "pairs": len(blob["keys"])},
        "today": {"source": "tests/帧样本/先粗后细/读数.json 的 cost_measured（复用 · 折半）",
                  "chain_calls": today["match_template_calls"] // 2,
                  "chain_s": round(chain_s, 4)},
        "family": {label: _plan_description(plan) for label, plan in FAMILY.items()},
        "designs": results, "measure": measure,
        "seconds": round(time.perf_counter() - started, 3),
    }
    if not args.dump_pairs:
        OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("总耗时 %.1f s" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
