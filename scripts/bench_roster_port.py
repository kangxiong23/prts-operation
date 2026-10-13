# -*- coding: utf-8 -*-
r"""甲-A 落进线上点名链之后的账：走法 · 判定账 · **逐张分差表** · guard 计数。

    .venv\Scripts\python.exe scripts\bench_roster_port.py

**只量不改** —— 这一份跑的时候链已经落了（`roster_scores` = 四层扫）。
真值一律复用落盘：旧链那一侧从 曲线.json 现算（它记的就是全分辨率全档的取值域，
与旧 roster_scores 逐位相等已经核过 4030 对），节拍复用 读数.json 的 cost_measured。

四件事：

1. **自证**：线上 `roster_scores` 与参考实现（bench_roster_layered_cell 的 甲-A 那一行）
   在全语料上**逐位相等** —— 落了的那条链就是量过的那条。
2. **判定账**：对着 曲线.json 那张旧链的表。
3. **逐张分差表**：每张卡 旧余量 − 新余量。**它整不整体向下，就是"甲-A 是不是拿搜索偏向
   换提速"那块证据。**
4. **guard 计数**：裁窗装不下模板 -> 退回整张 出现了几次（按模板大小归层）。
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
from ark.engine.vision.roster import roster_scores

OUT_DIR = base.SAMPLES / "先粗后细"
CURVES = OUT_DIR / "曲线.json"
READINGS = OUT_DIR / "读数.json"
OUT = OUT_DIR / "落地.json"
PLAN_LABEL = "甲-A 旧四层 + 细层加宽到 ±2（位12/8/6）"
NARROW = base.NARROW


def _old_rows(curve_rows):
    """旧链（全分辨率全档）在窄区间上的逐条目读数 —— 从 曲线.json 现算。"""
    out = {}
    for scale, score, lx, ly in curve_rows:
        s = float(scale)
        if any(abs(s - t) < 1e-9 for t in NARROW):
            out[s] = (float(score), int(lx), int(ly))
    return out


def _by_entry(rows):
    scale, (score, lx, ly) = base._best(rows)
    return {"score": score, "scale": scale, "loc": (lx, ly)}


def _layer_of(text):
    try:
        tail = text.split("装不下模板")[1].strip()
        height = int(tail.strip("()").split(",")[0].strip())
    except Exception:
        return "认不出"
    if height <= 30:
        return "第2层(25px)"
    if height <= 60:
        return "第3层(50px)"
    return "第4层(100px)"


def main():
    started = time.perf_counter()
    blob = json.loads(CURVES.read_text(encoding="utf-8"))
    readings = json.loads(READINGS.read_text(encoding="utf-8"))
    curves = blob["curves"]
    corpus, libraries = base._corpus()
    truth_map = base._truth_map()
    plan = cell.FAMILY[PLAN_LABEL]
    print("=== 甲-A 落地账 ===")
    print("  真值 曲线.json：%d 条 / 语料 %d 张卡" % (len(curves), len(corpus)))
    print("  走法（照 补洞.json 的 甲-A 那一行）：%s"
          % json.dumps(cell._plan_description(plan), ensure_ascii=False))

    guards = []
    real_print = getattr(cell, "print", print)
    cell.print = lambda *a, **k: guards.append(" ".join(str(x) for x in a))

    pairs = []
    ledger_rows = []
    calls_ref = 0
    mismatches = []
    old_differs = 0
    try:
        for record in corpus:
            faces = libraries[record["library"]]
            key = "%s|%s|%d" % (record["batch"], record["frame"], record["index"])
            old_by_entry = {}
            new_by_entry = {}
            ref_by_entry = {}
            for name, art in faces:
                here = curves[key + "|" + name]
                old_rows = _old_rows(here)
                if not old_rows:
                    continue
                old_by_entry[name] = _by_entry(old_rows)
                counter = [0]
                got = roster_scores(record["region"], [(name, art)])[0]
                ref = cell.run(record["region"], art, plan, counter)
                calls_ref += counter[0]
                if got[0] != ref["score"] or abs(got[2] - ref["picked"]) > 1e-12:
                    mismatches.append({"key": key, "entry": name,
                                       "live": [got[0], got[2]],
                                       "ref": [ref["score"], ref["picked"]]})
                new_by_entry[name] = {"score": got[0], "scale": got[2], "loc": (0.0, 0.0)}
                ref_by_entry[name] = {"score": ref["score"], "scale": ref["picked"],
                                      "loc": ref["loc"]}
                if (abs(got[0] - old_by_entry[name]["score"]) > 1e-9
                        or abs(got[2] - old_by_entry[name]["scale"]) > 1e-12):
                    old_differs += 1
                pairs.append({"batch": record["batch"], "frame": record["frame"],
                              "index": record["index"], "entry": name,
                              "old_score": old_by_entry[name]["score"],
                              "new_score": got[0],
                              "old_scale": old_by_entry[name]["scale"],
                              "new_scale": got[2]})
            if not old_by_entry:
                continue
            # 分差表用**没四舍五入**的原值：_verdict 里的 score/margin 是 6 位小数，
            # 拿它比大小会把 +-4e-6 的舍入差记成"涨"。
            def _raw(table):
                ranked = sorted(table.items(), key=lambda item: (-item[1]["score"], item[0]))
                top = ranked[0][1]["score"]
                second = ranked[1][1]["score"] if len(ranked) > 1 else -2.0
                return top, top - second
            old_top, old_gap = _raw(old_by_entry)
            new_top, new_gap = _raw(new_by_entry)
            group = record["frame"][1] if record["batch"] == "立绘两组" else None
            known = truth_map.get((record["frame"], record["index"]))
            before = base._verdict(old_by_entry, NARROW, known, group)
            after = base._verdict(new_by_entry, NARROW, known, group)
            ref_v = base._verdict(ref_by_entry, NARROW, known, group)
            ledger_rows.append({
                "batch": record["batch"], "frame": record["frame"],
                "index": record["index"], "redeploying": record["redeploying"],
                "truth": known, "group": group,
                "before": before, "after": after, "reference": ref_v,
                "raw_old_top": old_top, "raw_new_top": new_top,
                "raw_old_margin": old_gap, "raw_new_margin": new_gap,
                "raw_delta_score": new_top - old_top,
                "raw_delta_margin": new_gap - old_gap,
                "delta_score": round(after["score"] - before["score"], 9),
                "delta_margin": round(after["margin"] - before["margin"], 9),
            })
    finally:
        cell.print = real_print

    gained = [r for r in ledger_rows
              if r["before"]["verdict"] == "未知" and r["after"]["verdict"] == "精确"]
    lost = [r for r in ledger_rows
            if r["before"]["verdict"] == "精确" and r["after"]["verdict"] == "未知"]
    wrong_new = [r for r in gained if r["after"]["wrong"] is True]
    name_moved = [r for r in ledger_rows if r["before"]["name"] != r["after"]["name"]]

    deltas = [r["raw_delta_margin"] for r in ledger_rows]
    up = sum(1 for d in deltas if d > 0.0)
    down = sum(1 for d in deltas if d < 0.0)
    flat = sum(1 for d in deltas if d == 0.0)
    score_deltas = [r["raw_delta_score"] for r in ledger_rows]
    raw_up = sum(1 for d in score_deltas if d > 0.0)
    raw_up_max = max(score_deltas) if score_deltas else None
    margin_up_max = max(deltas) if deltas else None
    s_down = sum(1 for d in score_deltas if d < 0.0)
    s_flat = sum(1 for d in score_deltas if d == 0.0)

    guard_by_layer = {}
    for text in guards:
        key = _layer_of(text)
        guard_by_layer[key] = guard_by_layer.get(key, 0) + 1

    measured = readings["cost_measured"]["甲的今天：全分辨率全档"]
    payload = {
        "note": "甲-A 落进线上链之后的账：自证 · 判定账 · 逐张分差表 · guard 计数。只量不改。",
        "plan": {"label": PLAN_LABEL, "layers": cell._plan_description(plan)},
        "truth": {"file": "tests/帧样本/先粗后细/曲线.json", "pairs": len(curves),
                  "cards": len(corpus)},
        "today": {"source": "tests/帧样本/先粗后细/读数.json 的 cost_measured（复用 · 折半）",
                  "chain_calls": measured["match_template_calls"] // 2,
                  "chain_s": round(measured["median_s"] / 2, 4)},
        "selfcheck": {
            "port_vs_reference_pairs": len(pairs),
            "port_vs_reference_mismatches": len(mismatches),
            "port_vs_reference_rows": mismatches[:10],
            "port_vs_old_chain_differing": old_differs,
            "reference_match_template_calls": calls_ref,
            "note": "线上 roster_scores 与 bench_roster_layered_cell 的 甲-A 参考实现在"
                    "全语料上逐位比 —— 不等 0 才算落的是量过的那条。",
        },
        "ledger": {
            "cards": len(ledger_rows),
            "before_exact": sum(1 for r in ledger_rows if r["before"]["verdict"] == "精确"),
            "after_exact": sum(1 for r in ledger_rows if r["after"]["verdict"] == "精确"),
            "gained_n": len(gained), "lost_n": len(lost), "wrong_new_n": len(wrong_new),
            "name_moved_n": len(name_moved), "name_moved": name_moved[:10],
            "gained": gained[:20], "lost": lost[:20],
            "wrong_before_n": sum(1 for r in ledger_rows if r["before"]["wrong"] is True),
            "wrong_after_n": sum(1 for r in ledger_rows if r["after"]["wrong"] is True),
        },
        "margin_table": {
            "cards": len(ledger_rows),
            "up": up, "down": down, "flat": flat,
            "median_delta_margin": round(statistics.median(deltas), 9) if deltas else None,
            "mean_delta_margin": round(statistics.fmean(deltas), 9) if deltas else None,
            "min_delta_margin": round(min(deltas), 9) if deltas else None,
            "max_delta_margin": round(max(deltas), 9) if deltas else None,
            "score_down": s_down, "score_flat": s_flat, "score_up": raw_up,
            "score_up_max": raw_up_max, "margin_up_max": margin_up_max,
            "raw_note": "分差与最高分都取**没四舍五入**的原值；_verdict 里那两个是 6 位小数，"
                        "拿它比大小会把 +-4e-6 的舍入差记成涨。",
            "median_delta_score": round(statistics.median(score_deltas), 9) if score_deltas else None,
            "min_delta_score": round(min(score_deltas), 9) if score_deltas else None,
            "rows": [{"batch": r["batch"], "frame": r["frame"], "index": r["index"],
                      "redeploying": r["redeploying"], "truth": r["truth"],
                      "old_name": r["before"]["name"], "new_name": r["after"]["name"],
                      "old_score": r["before"]["score"], "new_score": r["after"]["score"],
                      "old_margin": r["before"]["margin"], "new_margin": r["after"]["margin"],
                      "delta_score": r["raw_delta_score"],
                      "delta_margin": r["raw_delta_margin"],
                      "old_verdict": r["before"]["verdict"],
                      "new_verdict": r["after"]["verdict"]}
                     for r in ledger_rows],
        },
        "guards": {"total": len(guards), "by_layer": guard_by_layer,
                   "rows": guards[:40],
                   "note": "裁窗装不下模板 -> 退回整张（**不是报错**）。它让那一层没裁成窗、"
                           "墙钟变差。线上那条链与参考实现逐位相等，所以这个计数对两者都成立。"},
        "seconds": round(time.perf_counter() - started, 3),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    led = payload["ledger"]
    tab = payload["margin_table"]
    print("  自证：线上 vs 参考 %d 对 · 不等 %d ｜ 与旧链不同的对 %d / %d"
          % (len(pairs), len(mismatches), old_differs, len(pairs)))
    print("  判定账：%d -> %d 精确 · 丢 %d · 得 %d · 新错 %d · 改名 %d"
          % (led["before_exact"], led["after_exact"], led["lost_n"], led["gained_n"],
             led["wrong_new_n"], led["name_moved_n"]))
    print("  分差表（逐张 · 原值）：涨 %d · 落 %d · 平 %d ｜ 中位 %s · 最小 %s"
          % (tab["up"], tab["down"], tab["flat"], tab["median_delta_margin"],
             tab["min_delta_margin"]))
    print("  最高分（逐张 · 原值）：落 %d · 平 %d · 涨 %d ｜ 中位 %s · 最小 %s · 涨的最大 %s"
          % (tab["score_down"], tab["score_flat"], tab["score_up"],
             tab["median_delta_score"], tab["min_delta_score"], tab["score_up_max"]))
    print("  guard：%d 次 %s" % (len(guards), json.dumps(guard_by_layer, ensure_ascii=False)))
    print("总耗时 %.1f s" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
