# -*- coding: utf-8 -*-
r"""新意义下的标定（用户裁定 5）：报出来的分是"第一个满足判据的那一档上的分"。

    .venv\Scripts\python.exe scripts\bench_roster_calibration.py

**旧那句只能作废、必须重测**：规格里"真匹配 0.9618-0.9921 · 非匹配 <=0.5622 · 空档 0.3996"
是**全局最大分**上的口径；新扫法故意不算全局最大，所以那一组数在新口径下不存在。

这一份量三件事：

1. **两簇**：逐 (卡, 库条目) 对，按真值分 真匹配 / 非匹配，各落在什么区间、空档多宽；
2. **0.85 那条线**：它还在不在空档里、离两簇各多远；
3. **余量风险**（用户裁定 8）：每张卡 `余量 - 它自己那条要求的分差`，报"离判定线 0.02 以内"的。

分两套语料：**有逐位真值的那两帧**（`shot01` + `cd60`，与规格同源）与**全语料有真值的卡**。
"""
from __future__ import annotations

import json
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
OUT = OUT_DIR / "新标定.json"
TRUTH_FRAMES = ("shot01", "cd60")
OLD = {"hits": [0.9618, 0.9921], "misses_max": 0.5622, "gap": 0.3996,
       "source": "旧口径（全局最大分）· tests/unit/test_vision_roster.py 的 docstring"}


def _cluster(pairs):
    hits = sorted(s for s, tag in pairs if tag)
    misses = sorted(s for s, tag in pairs if not tag)
    return {
        "hits_n": len(hits), "misses_n": len(misses),
        "hits_min": hits[0] if hits else None, "hits_max": hits[-1] if hits else None,
        "misses_max": misses[-1] if misses else None,
        "gap": (hits[0] - misses[-1]) if hits and misses else None,
        "line_above_misses": (R.ROSTER_SOLID_MATCH_SCORE - misses[-1]) if misses else None,
        "line_below_hits": (hits[0] - R.ROSTER_SOLID_MATCH_SCORE) if hits else None,
        "hits": hits, "misses": misses,
    }


def main():
    started = time.perf_counter()
    corpus, libraries = base._corpus()
    truth_map = base._truth_map()
    print("=== 新标定 ===")
    print("  判据：最高 >= %.2f 并且 次高 <= %.2f x 最高 - %.2f（= 要求的分差 %.2f - 最高）"
          % (R.ROSTER_SOLID_MATCH_SCORE, R.ROSTER_AMBIGUITY_FACTOR,
             R.ROSTER_AMBIGUITY_FACTOR + R.ROSTER_AMBIGUITY_PAD - 1.0, 1.09))
    print("  区间 %d 档 %.2f...%.2f" % (len(IDENTITY_SCALES), IDENTITY_SCALES[0],
                                     IDENTITY_SCALES[-1]))

    truth_pairs = []
    all_pairs = []
    cards = []
    for record in corpus:
        faces = libraries[record["library"]]
        known = truth_map.get((record["frame"], record["index"]))
        got, trace = R._layered_scores(record["region"], faces)
        by_name = {name: (score, scale) for score, name, scale in got}
        in_truth_frames = record["frame"] in TRUTH_FRAMES
        for name, _art in faces:
            if known is None:
                continue
            tag = (known != "" and name == known)
            all_pairs.append((by_name[name][0], tag))
            if in_truth_frames:
                truth_pairs.append((by_name[name][0], tag))
        ranked = sorted(by_name.items(), key=lambda item: (-item[1][0], item[0]))
        top_name, (top, _scale) = ranked[0]
        runner = ranked[1][1][0] if len(ranked) > 1 else -2.0
        margin = top - runner
        required = 1.09 - top
        cards.append({
            "batch": record["batch"], "frame": record["frame"], "index": record["index"],
            "truth": known, "name": top_name, "top": round(top, 9),
            "runner": round(runner, 9), "margin": round(margin, 9),
            "required_margin": round(required, 9), "room": round(margin - required, 9),
            "exact": identity_is_exact(top, runner),
            "tried": len(trace["fine"]["tried"]),
        })

    minimum = _cluster(truth_pairs)
    whole = _cluster(all_pairs)
    known_cards = [c for c in cards if c["truth"] not in (None, "")]
    near = sorted((c for c in known_cards if abs(c["room"]) <= 0.02),
                  key=lambda c: abs(c["room"]))
    near5 = sorted((c for c in known_cards if abs(c["room"]) <= 0.05),
                   key=lambda c: abs(c["room"]))

    payload = {
        "note": "新意义下的标定：报出来的分是第一个满足判据的那一档上的分（不是全局最大分）。",
        "criterion": {"score": R.ROSTER_SOLID_MATCH_SCORE,
                      "factor": R.ROSTER_AMBIGUITY_FACTOR, "pad": R.ROSTER_AMBIGUITY_PAD,
                      "required_margin": "1.09 - 最高"},
        "interval": {"min": IDENTITY_SCALES[0], "max": IDENTITY_SCALES[-1],
                     "n": len(IDENTITY_SCALES)},
        "old_calibration": OLD,
        "truth_frames": {"frames": list(TRUTH_FRAMES), "cluster": minimum},
        "whole_corpus": {"cluster": whole},
        "line": {
            "score": R.ROSTER_SOLID_MATCH_SCORE,
            "in_gap_truth_frames": bool(
                minimum["misses_max"] is not None and minimum["hits_min"] is not None
                and minimum["misses_max"] < R.ROSTER_SOLID_MATCH_SCORE <= minimum["hits_min"]),
            "nearest_hit": minimum["hits_min"], "nearest_miss": minimum["misses_max"],
        },
        "cards": cards,
        "margin_risk": {
            "known_cards": len(known_cards),
            "within_002": [{"frame": c["frame"], "index": c["index"], "name": c["name"],
                            "truth": c["truth"], "top": c["top"], "margin": c["margin"],
                            "required": c["required_margin"], "room": c["room"],
                            "exact": c["exact"]} for c in near],
            "within_005": [{"frame": c["frame"], "index": c["index"], "name": c["name"],
                            "top": c["top"], "room": c["room"]} for c in near5],
            "note": "room = 余量 - 它自己那条要求的分差；负的就是被判定线咬掉的。",
        },
        "seconds": round(time.perf_counter() - started, 3),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    for label, block in (("有逐位真值那两帧", minimum), ("全语料有真值的卡", whole)):
        print("  [%s] 真匹配 %d 对 %.4f...%.4f | 非匹配 %d 对 <= %.4f | 空档 %.4f"
              % (label, block["hits_n"], block["hits_min"] or 0, block["hits_max"] or 0,
                 block["misses_n"], block["misses_max"] or 0, block["gap"] or 0))
    print("  0.85 落在空档里？%s（离非匹配 %+.4f · 离真匹配 %+.4f）"
          % (payload["line"]["in_gap_truth_frames"], minimum["line_above_misses"],
             minimum["line_below_hits"]))
    print("  旧口径（全局最大分）：真匹配 %.4f...%.4f · 非匹配 <=%.4f · 空档 %.4f"
          % (OLD["hits"][0], OLD["hits"][1], OLD["misses_max"], OLD["gap"]))
    print("  余量风险：有真值的卡 %d 张 | 离判定线 0.02 以内 %d 张 | 0.05 以内 %d 张"
          % (len(known_cards), len(near), len(near5)))
    for c in near[:10]:
        print("     %s#%d %s 最高 %.4f 余量 %.4f 要求 %.4f room %+.6f"
              % (c["frame"], c["index"], c["name"], c["top"], c["margin"],
                 c["required_margin"], c["room"]))
    print("总耗时 %.1f s" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
