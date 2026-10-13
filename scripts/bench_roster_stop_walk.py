# -*- coding: utf-8 -*-
r"""新扫法（扩到整段 · 近的先试 · 判据满足即停）对 穷举那条链 —— 逐张对比 + 代价。

    .venv\Scripts\python.exe scripts\bench_roster_stop_walk.py

穷举那一侧从 `曲线.json` 现算（26 档 0.95…1.20 的全分辨率取值域，与旧 roster_scores
逐位相等已核过 4030 对）；新那一侧走线上 `roster_scores`。**只读、不重跑基线。**

对比口径（用户裁定 6）：**（判定, 名字）必须一致 —— 丢 0 · 名字不许变**。
名字按**读出来的那个名字**比（未知就是未知，不拿"最像的那个"当名字）；
另外**单列**一栏"最像的那一条"的变化，因为它是粗层的位置窗造成的、与判定无关。
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
import ark.engine.vision.roster as R
from ark.engine.vision.roster import IDENTITY_SCALES, identity_is_exact

OUT_DIR = base.SAMPLES / "先粗后细"
CURVES = OUT_DIR / "曲线.json"
OUT = OUT_DIR / "停即扫.json"
NARROW = base.NARROW


def _name_of(value):
    payload = getattr(value, "payload", None)
    if payload is None or type(payload).__name__ != "Exact":
        return None
    return payload.value


def _reading(scores, truth, group):
    ranked = sorted(scores.items(), key=lambda item: (-item[1][0], item[0]))
    top_name, (top_score, _scale) = ranked[0]
    runner = ranked[1][1][0] if len(ranked) > 1 else -2.0
    exact = identity_is_exact(top_score, runner)
    if not exact:
        return {"verdict": "未知", "name": None, "top": top_score, "runner": runner,
                "margin": top_score - runner, "scale": _scale,
                "most_like": top_name}
    if truth is None and group is not None:
        wrong = not top_name.startswith("组" + group + "·")
    elif truth is None:
        wrong = None
    elif truth == "":
        wrong = True
    else:
        wrong = top_name != truth
    return {"verdict": "精确", "name": top_name, "top": top_score, "runner": runner,
            "margin": top_score - runner, "scale": _scale, "wrong": wrong,
            "most_like": top_name}


def main() -> int:
    started = time.perf_counter()
    blob = json.loads(CURVES.read_text(encoding="utf-8"))
    curves = blob["curves"]
    corpus, libraries = base._corpus()
    truth_map = base._truth_map()
    print("=== 新扫法 vs 穷举 ===")
    print("  语料 %d 张卡 · 尺度区间 %d 档 %.2f…%.2f"
          % (len(corpus), len(IDENTITY_SCALES), IDENTITY_SCALES[0], IDENTITY_SCALES[-1]))

    rows = []
    for record in corpus:
        faces = libraries[record["library"]]
        key = "%s|%s|%d" % (record["batch"], record["frame"], record["index"])
        old = {}
        old_scale = {}
        old_most = None
        for name, _art in faces:
            here = [r for r in curves[key + "|" + name]
                    if any(abs(float(r[0]) - t) < 1e-9 for t in IDENTITY_SCALES)]
            if not here:
                continue
            best = max(here, key=lambda r: (float(r[1]), -float(r[0])))
            old[name] = (float(best[1]), float(best[0]))
            old_scale[name] = float(best[0])
        if not old:
            continue
        began = time.perf_counter()
        got, trace = R._layered_scores(record["region"], faces)
        elapsed = time.perf_counter() - began
        new = {name: (score, scale) for score, name, scale in got}
        group = record["frame"][1] if record["batch"] == "立绘两组" else None
        known = truth_map.get((record["frame"], record["index"]))
        before = base._verdict({n: {"score": s, "scale": sc, "loc": (0, 0)}
                                for n, (s, sc) in old.items()}, NARROW, known, group)
        after = _reading(new, known, group)
        # 穷举的"最像的那一条"
        old_most = max(old, key=lambda n: (old[n][0], n))
        new_most = got[0][1]
        # 锚点 vs 穷举对**冠军条目**选中的那一档
        leader = trace["anchor"]["leader"]
        anchor = float(trace["anchor"]["scale"])
        rows.append({
            "batch": record["batch"], "frame": record["frame"], "index": record["index"],
            "redeploying": record["redeploying"], "truth": known, "group": group,
            "library": record["library"], "entries": len(faces),
            "old": {"verdict": before["verdict"], "name": before["name"],
                    "top": before["score"], "margin": before["margin"],
                    "scale": before["scale"], "most_like": old_most},
            "new": {"verdict": after["verdict"], "name": after["name"],
                    "top": round(after["top"], 9), "margin": round(after["margin"], 9),
                    "scale": after["scale"], "most_like": new_most,
                    "wrong": after.get("wrong")},
            "verdict_changed": before["verdict"] != after["verdict"],
            "name_changed": before["name"] != after["name"],
            "most_like_changed": old_most != new_most,
            "anchor": {"scale": anchor, "leader": leader,
                       "leader_old_scale": old_scale.get(leader),
                       "delta_steps": None if old_scale.get(leader) is None
                       else round(abs(anchor - old_scale[leader]) / 0.01)},
            "tried": len(trace["fine"]["tried"]),
            "stopped_at": trace["fine"]["stopped_at"],
            "satisfied": trace["fine"]["satisfied"],
            "seconds": round(elapsed, 4),
            "trace": trace,
        })

    lost = [r for r in rows if r["old"]["verdict"] == "精确" and r["new"]["verdict"] != "精确"]
    gained = [r for r in rows if r["old"]["verdict"] != "精确" and r["new"]["verdict"] == "精确"]
    moved = [r for r in rows if r["name_changed"]]
    wrong_new = [r for r in rows if r["new"].get("wrong") is True]
    tries = [r["tried"] for r in rows]
    satisfied = [r for r in rows if r["satisfied"]]
    deltas = [r["anchor"]["delta_steps"] for r in rows if r["anchor"]["delta_steps"] is not None]
    times = [r["seconds"] for r in rows]
    payload = {
        "note": "新扫法（扩到整段 · 近的先试 · 判据满足即停）对穷举那条链的逐张对比 + 代价。只读夹具。",
        "criterion": {"score_line": R.ROSTER_SOLID_MATCH_SCORE,
                      "factor": R.ROSTER_AMBIGUITY_FACTOR, "pad": R.ROSTER_AMBIGUITY_PAD},
        "interval": {"min": IDENTITY_SCALES[0], "max": IDENTITY_SCALES[-1],
                     "n": len(IDENTITY_SCALES)},
        "anchor": {"pixels": R._ANCHOR_PIXELS, "position_pixels": R._POSITION_PIXELS,
                   "position_half": R._ANCHOR_POSITION_HALF,
                   "fine_half": R._FINE_POSITION_HALF},
        "summary": {
            "cards": len(rows),
            "old_exact": sum(1 for r in rows if r["old"]["verdict"] == "精确"),
            "new_exact": sum(1 for r in rows if r["new"]["verdict"] == "精确"),
            "lost_n": len(lost), "gained_n": len(gained),
            "name_changed_n": len(moved),
            "most_like_changed_n": sum(1 for r in rows if r["most_like_changed"]),
            "wrong_new_n": len(wrong_new),
            "satisfied_n": len(satisfied),
            "tried_median": statistics.median(tries) if tries else None,
            "tried_mean": round(statistics.fmean(tries), 2) if tries else None,
            "tried_p90": sorted(tries)[int(0.9 * (len(tries) - 1))] if tries else None,
            "tried_max": max(tries) if tries else None,
            "tried_histogram": {str(k): sum(1 for t in tries if t == k)
                                for k in sorted(set(tries))},
            "anchor_delta_median": statistics.median(deltas) if deltas else None,
            "anchor_delta_histogram": {str(k): sum(1 for d in deltas if d == k)
                                       for k in sorted(set(deltas))},
            "seconds_median": round(statistics.median(times), 4) if times else None,
            "seconds_total": round(sum(times), 4),
        },
        "lost": lost[:20], "gained": gained[:20],
        "name_changed": moved[:30],
        "most_like_changed": [{"batch": r["batch"], "frame": r["frame"], "index": r["index"],
                               "old": r["old"]["most_like"], "new": r["new"]["most_like"],
                               "old_top": r["old"]["top"], "new_top": r["new"]["top"],
                               "old_verdict": r["old"]["verdict"],
                               "new_verdict": r["new"]["verdict"]}
                              for r in rows if r["most_like_changed"]][:60],
        "rows": rows,
        "seconds": round(time.perf_counter() - started, 3),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    s = payload["summary"]
    print("  判定：%d -> %d 精确 · 丢 %d · 得 %d" % (s["old_exact"], s["new_exact"],
                                                s["lost_n"], s["gained_n"]))
    print("  名字：读出来的名字变了 %d 张 · 「最像的那一条」变了 %d 张 · 新错 %d"
          % (s["name_changed_n"], s["most_like_changed_n"], s["wrong_new_n"]))
    print("  试档：中位 %s · 均值 %s · p90 %s · 最大 %s · 一次就停 %d 张"
          % (s["tried_median"], s["tried_mean"], s["tried_p90"], s["tried_max"],
             s["tried_histogram"].get("1", 0)))
    print("  锚点与穷举选中档之差（冠军条目）：中位 %s · 分布 %s"
          % (s["anchor_delta_median"], s["anchor_delta_histogram"]))
    print("  耗时：中位 %.3f s/卡 · 合计 %.1f s（%d 卡）"
          % (s["seconds_median"], s["seconds_total"], s["cards"]))
    print("总耗时 %.1f s" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
