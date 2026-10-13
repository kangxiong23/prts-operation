# -*- coding: utf-8 -*-
r"""判据换成用户刚裁定的那一条（**只量不落**）：读现成夹具就够。

    .venv\Scripts\python.exe scripts\bench_roster_new_criterion.py

新判据：

    最高 >= 0.85   并且   (1 - 次高) / (1 - 最高) >= 2 + 0.09 / (1 - 最高)

等价形式（两边同乘 (1 - 最高)，它是正的）：**次高 <= 2 x 最高 - 1.09**。
这一份**逐点核**一遍这个等价（不等就按"规划者给错了"报）。

要求的分差 = 1.09 - 最高（<= 0.24 < 旧线的 0.25，所以**判定账只可能松不可能紧**）。

夹具（都是现成的，**一个字节没动**）：

- 主路：`读数.json` 的 benefit_rows（350 张，今天那条链：score 与 margin 都记着）。
- 红罩路：`红罩/点名.json` 的 frames[].slots（28 张，score 与 runner_up_score 都记着）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

OUT_DIR = REPO / "tests" / "帧样本" / "先粗后细"
READINGS = OUT_DIR / "读数.json"
REDCAP = REPO / "tests" / "帧样本" / "红罩" / "点名.json"
OUT = OUT_DIR / "新判据.json"
SCORE_LINE = 0.85
OLD_MARGIN = 0.25
PAD = 0.09
FACTOR = 2.0


def new_passes(s1, s2, factor=FACTOR, pad=PAD):
    """新判据的两半 —— 与 <=> 次高 <= factor*最高 - (factor + pad - 1) 等价。"""
    if s1 < SCORE_LINE:
        return False
    return (1.0 - s2) / (1.0 - s1) >= factor + pad / (1.0 - s1)


def new_passes_closed(s1, s2, factor=FACTOR, pad=PAD):
    """等价形式（乘开的那一条）。"""
    if s1 < SCORE_LINE:
        return False
    return s2 <= factor * s1 - (factor + pad - 1.0)


def old_passes(s1, s2):
    return s1 >= SCORE_LINE and s1 - s2 >= OLD_MARGIN


def main():
    readings = json.loads(READINGS.read_text(encoding="utf-8"))
    redcap = json.loads(REDCAP.read_text(encoding="utf-8"))
    cards = []
    for row in readings["benefit_rows"]:
        s1 = row["before"]["score"]
        s2 = round(s1 - row["before"]["margin"], 9)
        cards.append({"path": "主路", "batch": row["batch"], "frame": row["frame"],
                      "index": row["index"], "truth": row["truth"],
                      "name": row["before"]["name"], "score": s1, "runner_up": s2,
                      "runner_up_name": None, "margin": round(s1 - s2, 9),
                      "wrong": row["before"]["wrong"],
                      "blocked_by": row["before"]["blocked_by"]})
    for frame in redcap["frames"]:
        for slot in frame["slots"]:
            if slot["redeploying"] is not True:
                continue
            s1 = slot["score"]
            s2 = slot["runner_up_score"]
            cards.append({"path": "红罩路", "batch": "红罩", "frame": frame["name"],
                          "index": slot["index"], "truth": slot.get("truth"),
                          "name": slot["reading"].get("name"),
                          "runner_up": s2, "runner_up_name": slot["runner_up"],
                          "score": s1, "margin": round(s1 - s2, 9),
                          "wrong": None, "blocked_by": slot["blocked_by"]})

    # ---- 0. 等价形式逐点核 ----
    mism = [c for c in cards
            if new_passes(c["score"], c["runner_up"])
            != new_passes_closed(c["score"], c["runner_up"])]
    print("=== 判据等价自检 ===")
    print("  逐点核 %d 张：两式不一致 %d 张" % (len(cards), len(mism)))

    old_exact = [c for c in cards if old_passes(c["score"], c["runner_up"])]
    new_exact = [c for c in cards if new_passes(c["score"], c["runner_up"])]
    lost = [c for c in old_exact if not new_passes(c["score"], c["runner_up"])]
    gained = [c for c in new_exact if not old_passes(c["score"], c["runner_up"])]

    print("=== 判定账（同一批读数，只换判据）===")
    print("  旧判据 %d 精确 -> 新判据 %d 精确 ｜ 丢 %d · 得 %d"
          % (len(old_exact), len(new_exact), len(lost), len(gained)))

    for c in new_exact:
        c["required_margin"] = round(1.09 - c["score"], 9)
        c["room"] = round(c["margin"] - c["required_margin"], 9)

    boundary = [c for c in cards if SCORE_LINE <= c["score"] < 0.95]
    boundary.sort(key=lambda c: c["score"])
    for c in boundary:
        c["required_margin"] = round(1.09 - c["score"], 9)
        c["room"] = round(c["margin"] - c["required_margin"], 9)
        c["old_verdict"] = "精确" if old_passes(c["score"], c["runner_up"]) else "未知"
        c["new_verdict"] = "精确" if new_passes(c["score"], c["runner_up"]) else "未知"

    # ---- d. 取值不敏感区间 ----
    base_ledger = frozenset(
        (c["frame"], c["index"], c["path"]) for c in new_exact)

    def ledger_of(factor, pad):
        return frozenset((c["frame"], c["index"], c["path"]) for c in cards
                         if new_passes(c["score"], c["runner_up"], factor, pad))

    pads = [round(0.0 + i * 0.0025, 4) for i in range(0, 121)]
    same_pad = [p for p in pads if ledger_of(FACTOR, p) == base_ledger]
    factors = [round(1.0 + i * 0.01, 4) for i in range(0, 201)]
    same_factor = [k for k in factors if ledger_of(k, PAD) == base_ledger]

    def contiguous(values, centre):
        if not values:
            return None
        values = sorted(values)
        chosen = None
        for value in values:
            if abs(value - centre) < 1e-12:
                chosen = value
        if chosen is None:
            return None
        low = high = chosen
        index = values.index(chosen)
        while index > 0 and abs(values[index - 1] - low) < 0.00251:
            index -= 1
            low = values[index]
        index = values.index(chosen)
        while index + 1 < len(values) and abs(values[index + 1] - high) < 0.00251:
            index += 1
            high = values[index]
        return [low, high]

    pad_range = contiguous(same_pad, PAD)
    factor_range = contiguous(same_factor, FACTOR)

    # 谁是那条刀口：在 (2.0, 0.09) 上离判定线最近的那几张
    for c in cards:
        if c["score"] >= SCORE_LINE:
            c["required_margin"] = round(1.09 - c["score"], 9)
            c["room"] = round(c["margin"] - c["required_margin"], 9)
    knife = sorted((c for c in cards if c["score"] >= SCORE_LINE
                    and c["room"] is not None), key=lambda c: abs(c["room"]))[:8]
    factor_neighbours = {}
    for k in (1.99, 2.01):
        differs = []
        for c in cards:
            here = new_passes(c["score"], c["runner_up"], FACTOR, PAD)
            there = new_passes(c["score"], c["runner_up"], k, PAD)
            if here != there:
                differs.append({"path": c["path"], "frame": c["frame"], "index": c["index"],
                                "name": c["name"], "score": c["score"],
                                "runner_up": c["runner_up"], "margin": c["margin"],
                                "required_here": round(1.09 - c["score"], 9),
                                "required_there": round((k - 1.0) * (1.0 - c["score"]) + PAD, 9),
                                "verdict_here": "精确" if here else "未知",
                                "verdict_there": "精确" if there else "未知"})
        factor_neighbours[str(k)] = differs
    pad_neighbours = {}
    for q in (0.0875, 0.0925):
        differs = [c["frame"] + "#" + str(c["index"]) for c in cards
                   if new_passes(c["score"], c["runner_up"], FACTOR, PAD)
                   != new_passes(c["score"], c["runner_up"], FACTOR, q)]
        pad_neighbours[str(q)] = differs

    redcap_six = [c for c in cards if c["path"] == "红罩路"
                  and c["blocked_by"] == "余量不够"]
    for c in redcap_six:
        c["required_margin"] = round(1.09 - c["score"], 9)
        c["room"] = round(c["margin"] - c["required_margin"], 9)
        c["new_verdict"] = "精确" if new_passes(c["score"], c["runner_up"]) else "未知"

    payload = {
        "note": "判据换成 (1-次高)/(1-最高) >= 2 + 0.09/(1-最高) 且 最高 >= 0.85 之后的判定账。只量不落。",
        "criterion": {"score_line": SCORE_LINE, "old_margin": OLD_MARGIN,
                      "factor": FACTOR, "pad": PAD,
                      "closed_form": "次高 <= 2 x 最高 - 1.09",
                      "required_margin": "1.09 - 最高"},
        "sources": {
            "主路": "tests/帧样本/先粗后细/读数.json 的 benefit_rows（350 张，今天那条链）",
            "红罩路": "tests/帧样本/红罩/点名.json 的 frames[].slots（28 张红罩卡）",
        },
        "equivalence": {"cards": len(cards), "mismatches": len(mism),
                        "rows": [{"frame": c["frame"], "index": c["index"]} for c in mism[:5]]},
        "a_lost": {"cards": len(cards), "old_exact": len(old_exact),
                   "new_exact": len(new_exact), "lost_n": len(lost), "gained_n": len(gained),
                   "lost": lost[:20]},
        "b_gained": [
            {"path": c["path"], "frame": c["frame"], "index": c["index"], "name": c["name"],
             "score": c["score"], "runner_up": c["runner_up"], "margin": c["margin"],
             "runner_up_name": c.get("runner_up_name"), "required_margin": c["required_margin"],
             "room": c["room"], "truth": c["truth"], "wrong": c["wrong"]}
            for c in gained],
        "c_boundary": {"range": [SCORE_LINE, 0.95], "n": len(boundary),
                       "rows": [{"path": c["path"], "frame": c["frame"], "index": c["index"],
                                 "name": c["name"], "score": c["score"],
                                 "runner_up": c["runner_up"], "margin": c["margin"],
                                 "required_margin": c["required_margin"], "room": c["room"],
                                 "old_verdict": c["old_verdict"],
                                 "new_verdict": c["new_verdict"]} for c in boundary]},
        "d_insensitive": {
            "factor_fixed_pad_scan": {"factor": FACTOR, "pad_center": PAD,
                                      "same_ledger_pads": pad_range,
                                      "n_tested": len(pads)},
            "pad_fixed_factor_scan": {"pad": PAD, "factor_center": FACTOR,
                                      "same_ledger_factors": factor_range,
                                      "n_tested": len(factors)},
            "note": "判定账**完全不动**的那一段（其他两项不变）。值由用户定，这一份只报区间、不改值。",
            "knife_edge_rows": [
                {"path": c["path"], "frame": c["frame"], "index": c["index"],
                 "name": c["name"], "score": c["score"], "margin": c["margin"],
                 "required_margin": c["required_margin"], "room": c["room"]}
                for c in knife],
            "factor_neighbour_differs": factor_neighbours,
            "pad_neighbour_differs": pad_neighbours,
        },
        "e_redcap_six": [{"path": c["path"], "frame": c["frame"], "index": c["index"],
                          "name": c["name"], "score": c["score"],
                          "runner_up_name": c["runner_up_name"],
                          "runner_up": c["runner_up"], "margin": c["margin"],
                          "required_margin": c["required_margin"], "room": c["room"],
                          "new_verdict": c["new_verdict"]} for c in redcap_six],
        "by_path": {
            "主路": {"cards": sum(1 for c in cards if c["path"] == "主路"),
                     "old_exact": sum(1 for c in old_exact if c["path"] == "主路"),
                     "new_exact": sum(1 for c in new_exact if c["path"] == "主路")},
            "红罩路": {"cards": sum(1 for c in cards if c["path"] == "红罩路"),
                       "old_exact": sum(1 for c in old_exact if c["path"] == "红罩路"),
                       "new_exact": sum(1 for c in new_exact if c["path"] == "红罩路")},
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=== a) 丢 / 得 ===")
    print("  丢 %d 张；新增精确 %d 张" % (len(lost), len(gained)))
    for c in gained:
        print("    + %s %s#%d %s：最高 %.6f · 次高 %.6f · 分差 %.6f · 要求 %.6f · 富余 %.6f"
              % (c["path"], c["frame"], c["index"], c["name"], c["score"], c["runner_up"],
                 c["margin"], c["required_margin"], c["room"]))
    print("=== 刀口：离判定线最近的 8 张（在 (2.0, 0.09) 上）===")
    for c in knife:
        print("    %s %s#%d %s 最高 %.6f 余量 %.6f 要求 %.6f 富余 %+.6f"
              % (c["path"], c["frame"], c["index"], c["name"], c["score"], c["margin"],
                 c["required_margin"], c["room"]))
    print("=== c) 边界（最高 落在 0.85-0.95 那一带）===")
    print("  这一带共 %d 张：%s" % (
        len(boundary), " ".join("%s#%d %s 最高%.4f 余量%.4f 要求%.4f 富余%+.6f -> %s" % (
            c["frame"], c["index"], c["old_verdict"], c["score"], c["margin"],
            c["required_margin"], c["room"], c["new_verdict"]) for c in boundary)
        or "（一张都没有）"))
    print("=== d) 取值不敏感区间 ===")
    print("  pad（因子固定 %.1f）：判定账不动的那一段 = %s（在 %d 个取值上扫）"
          % (FACTOR, pad_range, len(pads)))
    print("  因子（pad 固定 %.2f）：判定账不动的那一段 = %s（在 %d 个取值上扫）"
          % (PAD, factor_range, len(factors)))
    print("=== e) 红罩那 %d 张（旧判据下被余量线挡住）===" % len(redcap_six))
    for c in redcap_six:
        print("    %s#%d %s：最高 %.6f · 次高 %s %.6f · 要求 %.6f · 富余 %.6f -> %s"
              % (c["frame"], c["index"], c["name"], c["score"], c["runner_up_name"],
                 c["runner_up"], c["required_margin"], c["room"], c["new_verdict"]))
    print("=== 分路 ===")
    print("  %s" % json.dumps(payload["by_path"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
