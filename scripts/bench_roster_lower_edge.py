# -*- coding: utf-8 -*-
r"""尺子**下沿那一刀**（只量不落）：把粗层的尺度区间往下沿放几档。

    .venv\Scripts\python.exe scripts\bench_roster_lower_edge.py

线索：`塞雷娅` 那一族的峰在 **1.01**，而线上区间从 **1.05** 起（`曲线.json` 里读到的）。
这一份把粗层的第一层网格往下铺（1.03 / 1.01 / 0.99），**后三层照 甲-A 那一行**，
报：判定账（丢 / 得 / 新错）· `塞雷娅` 那一族回没回来 · 墙钟倍数 ·
以及 `曲线.json` 覆盖的尺度范围（**只读，不重跑它**）。

**只量不落** —— 线上链的尺度区间一个字没动。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import bench_roster_coarse as base
import bench_roster_layered_cell as cell

OUT_DIR = base.SAMPLES / "先粗后细"
CURVES = OUT_DIR / "曲线.json"
READINGS = OUT_DIR / "读数.json"
OUT = OUT_DIR / "下沿.json"
NARROW = base.NARROW
FAMILY_NAME = "塞雷娅"


def _plan(low):
    """把 甲-A 那一行的第一层网格换成 low，其余三层照旧。"""
    return (
        (12, 0.01, low, None, None, low, False),
        (25, 0.01, None, 3, 12, low, False),
        (50, 0.01, None, 2, 8, low, False),
        (100, 0.01, low, 2, 6, low, True),
    )


def _low_grid(floor):
    values = []
    value = floor
    while value <= 1.1500001:
        values.append(round(value, 2))
        value += 0.01
    return tuple(values)


def main():
    started = time.perf_counter()
    blob = json.loads(CURVES.read_text(encoding="utf-8"))
    readings = json.loads(READINGS.read_text(encoding="utf-8"))
    grid = [float(s) for s in blob["grid"]]
    corpus, libraries = base._corpus()
    truth_map = base._truth_map()
    ref_plan = cell.FAMILY["甲-A 旧四层 + 细层加宽到 ±2（位12/8/6）"]
    print("=== 下沿那一刀 ===")
    print("  曲线.json 覆盖的尺度：%d 档 · %.2f … %.2f（**只读**，这一份没重跑它）"
          % (len(grid), min(grid), max(grid)))

    designs = {}
    for floor in (1.05, 1.03, 1.01, 0.99):
        label = "甲-A（下沿 1.05）" if abs(floor - 1.05) < 1e-9 else "下沿 %.2f" % floor
        designs[label] = _plan(_low_grid(floor))
    print("  这一轮跑：%s" % list(designs))

    results = {}
    for label, plan in designs.items():
        began = time.perf_counter()
        verdict_rows = []
        per_pair = []
        for record in corpus:
            faces = libraries[record["library"]]
            key = "%s|%s|%d" % (record["batch"], record["frame"], record["index"])
            # 旧链（窄区间，全分辨率全档）那一侧从 曲线.json 现算
            old_by_entry = {}
            new_by_entry = {}
            for name, art in faces:
                here = [row for row in blob["curves"][key + "|" + name]
                        if any(abs(float(row[0]) - t) < 1e-9 for t in NARROW)]
                if not here:
                    continue
                old_scale, old = base._best({float(r[0]): (float(r[1]), int(r[2]), int(r[3]))
                                            for r in here})
                old_by_entry[name] = {"score": old[0], "scale": old_scale,
                                      "loc": (old[1], old[2])}
                got = cell.run(record["region"], art, plan, [0])
                new_by_entry[name] = {"score": got["score"], "scale": got["picked"],
                                      "loc": got["loc"]}
                per_pair.append({"batch": record["batch"], "frame": record["frame"],
                                 "index": record["index"], "entry": name,
                                 "old_score": old[0], "old_scale": old_scale,
                                 "new_score": got["score"], "new_scale": got["picked"],
                                 "family": FAMILY_NAME in name})
            if not old_by_entry:
                continue
            group = record["frame"][1] if record["batch"] == "立绘两组" else None
            known = truth_map.get((record["frame"], record["index"]))
            before = base._verdict(old_by_entry, NARROW, known, group)
            after = base._verdict(new_by_entry, NARROW, known, group)
            verdict_rows.append({"batch": record["batch"], "frame": record["frame"],
                                 "index": record["index"], "truth": known, "group": group,
                                 "before": before, "after": after})
        gained = [r for r in verdict_rows if r["before"]["verdict"] == "未知"
                  and r["after"]["verdict"] == "精确"]
        lost = [r for r in verdict_rows if r["before"]["verdict"] == "精确"
                and r["after"]["verdict"] == "未知"]
        wrong_new = [r for r in gained if r["after"]["wrong"] is True]
        # 塞雷娅那一族：哪几格它成了 top-1、它的分过没过线
        family_cards = {}
        for row in verdict_rows:
            family = [p for p in per_pair
                      if (p["batch"], p["frame"], p["index"])
                      == (row["batch"], row["frame"], row["index"]) and p["family"]]
            if not family:
                continue
            family_cards["%s|%s|%d" % (row["batch"], row["frame"], row["index"])] = {
                "before_name": row["before"]["name"], "after_name": row["after"]["name"],
                "before_verdict": row["before"]["verdict"],
                "after_verdict": row["after"]["verdict"],
                "family_old_best": max(p["old_score"] for p in family),
                "family_new_best": max(p["new_score"] for p in family),
                "family_old_scale": max(family, key=lambda p: p["old_score"])["old_scale"],
                "family_new_scale": max(family, key=lambda p: p["new_score"])["new_scale"],
                "won_after": FAMILY_NAME in row["after"]["name"],
                "won_before": FAMILY_NAME in row["before"]["name"],
            }
        results[label] = {
            "plan": cell._plan_description(plan),
            "grid": list(plan[0][2]),
            "cards": len(verdict_rows),
            "before_exact": sum(1 for r in verdict_rows if r["before"]["verdict"] == "精确"),
            "after_exact": sum(1 for r in verdict_rows if r["after"]["verdict"] == "精确"),
            "lost_n": len(lost), "gained_n": len(gained), "wrong_new_n": len(wrong_new),
            "wrong_after_n": sum(1 for r in verdict_rows if r["after"]["wrong"] is True),
            "gained": gained[:20], "lost": lost[:20],
            "family_cards": family_cards,
            "seconds": round(time.perf_counter() - began, 3),
        }
        got = results[label]
        print("  [%s] 网格 %d 档（%.2f…%.2f）判定 %d -> %d · 丢 %d · 得 %d · 新错 %d · %.1fs"
              % (label, len(got["grid"]), min(got["grid"]), max(got["grid"]),
                 got["before_exact"], got["after_exact"], got["lost_n"], got["gained_n"],
                 got["wrong_new_n"], got["seconds"]))

    # ---- 墙钟：同一份采样上把每个走法真跑 3 次 ----
    measured = readings["cost_measured"]["甲的今天：全分辨率全档"]
    chain_s = measured["median_s"] / 2
    picks = readings["measure_sample"]["picks"]
    sample = [corpus[i] for i in picks]
    print("=== 墙钟（真跑 8 张卡 / %d 条库，3 次取中位）今天复用 %.4f s ==="
          % (sum(len(libraries[r["library"]]) for r in sample), chain_s))
    timing = {}
    for label, plan in designs.items():
        times = []
        counter = [0]
        for _ in range(3):
            began = time.perf_counter()
            counter[0] = 0
            for record in sample:
                for _name, art in libraries[record["library"]]:
                    cell.run(record["region"], art, plan, counter)
            times.append(time.perf_counter() - began)
        median = statistics.median(times)
        timing[label] = {"median_s": round(median, 4),
                         "rounds_s": [round(t, 4) for t in times],
                         "match_template_calls": counter[0],
                         "ratio_vs_today": round(chain_s / median, 2)}
        print("  %-22s %6d 次 %8.3f s %6.2fx" % (label, counter[0], median, chain_s / median))

    payload = {
        "note": "尺子下沿那一刀：粗层区间往下铺几档。只量不落 —— 线上区间一个字没动。",
        "truth": {"file": "tests/帧样本/先粗后细/曲线.json", "grid": grid,
                  "grid_min": min(grid), "grid_max": max(grid), "grid_n": len(grid)},
        "today": {"source": "tests/帧样本/先粗后细/读数.json 的 cost_measured（复用 · 折半）",
                  "chain_s": round(chain_s, 4)},
        "designs": results,
        "timing": timing,
        "seconds": round(time.perf_counter() - started, 3),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("总耗时 %.1f s" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
