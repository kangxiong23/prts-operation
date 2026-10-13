# -*- coding: utf-8 -*-
r"""红罩那条路（margin.MaskedRuler）的**判定账** —— 多层细化只在这里量判定，不量选档一致率。

    .venv\Scripts\python.exe scripts\bench_roster_layered_redcap.py

## 为什么这一份只做判定账

红罩卡走的是掩码那把尺子（取区不规则），它自己的全档曲线**没量过**，跑一遍太贵。
这一单对它只问一件事：**把尺度网格从 11 档收成粗层挑出来的 5 档之后，
那一帧的精确 / 未知还对不对得上**。

## 基线那一侧**复用落盘**，不重跑

tests/帧样本/红罩/点名.json —— 红罩 28 张：精确 22 · 未知 6（每格的分数 · 次高 ·
余量 · 被哪条线挡的都记着）。这一份只在那几张标定卡上重跑一遍**自己核对**
（分数要与落盘对上），其余全是新量。

## 一层粗、一层细

粗层：立绘与取区都缩到 25%（INTER_AREA / INTER_NEAREST），用同一把尺子的公式在
小图上把 11 档扫一遍 —— 挑出**一张卡**要精算的那一档。
细层：全分辨率，只扫**那一档 +-2 档**（5 档），报出的分只来自这一层。
位置那一轴这一份不动（`绝对线.json` 的 window 那几行是另一批语料上量过的，
只作为旁证列出来，不混进判定账）。
"""

from __future__ import annotations

import json
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
OUT = OUT_DIR / "红罩细化.json"
LEDGER = SAMPLES / "红罩" / "点名.json"
NARROW = base.NARROW
COARSE_FACTOR = 0.25
HALF = 2
SELF_CHECK_SLOTS = 3


def _verdict(ranked) -> dict:
    top_score, top_name, top_scale = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else -2.0
    margin = top_score - second
    exact = (top_score >= ROSTER_SOLID_MATCH_SCORE
             and margin >= ROSTER_AMBIGUITY_MARGIN)
    return {"name": top_name, "score": round(top_score, 6), "scale": top_scale,
            "margin": round(margin, 6), "verdict": "精确" if exact else "未知"}


def _scale_window(center: float, half: int = HALF) -> tuple:
    low = center - half * 0.01 - 1e-9
    high = center + half * 0.01 + 1e-9
    return tuple(s for s in NARROW if low <= s <= high)


def _shrink(image: np.ndarray, factor: float, nearest: bool) -> np.ndarray:
    return cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * factor))),
         max(1, int(round(image.shape[0] * factor)))),
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA,
    )


class Counting:
    """把 cv2.matchTemplate 包一层数次数 —— 与那条链比的账要分得开。"""

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


def _ledger_slots(ledger: dict) -> list:
    out = []
    for frame in ledger["frames"]:
        for slot in frame["slots"]:
            copy = dict(slot)
            copy["frame"] = frame["name"]
            out.append(copy)
    return out


def main() -> int:
    started = time.perf_counter()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    slots = _ledger_slots(ledger)
    print("=== 复用落盘（基线）===")
    print("  tests/帧样本/红罩/点名.json  red=%d 精确=%d 未知=%d"
          % (ledger["totals"]["red"], ledger["totals"]["exact"],
             ledger["totals"]["unknown"]))

    faces = load_cardfaces(base.SHARED_LIBRARY)
    box, mask = margin_module.identity_disc_region()
    print("  红罩取区 box=%s 掩码 %d px shape=%s  库 %d 条"
          % (list(box), int(mask.sum()), tuple(mask.shape), len(faces)))

    records = []
    for case in load_batch(SAMPLES / "红罩").cases:
        frame = frame_from_raw(case.raw())
        centres = bench.squad_card_centres(frame, pitch=PITCH)
        top = float(bench.squad_card_rect(0, PITCH).y)
        for index, (centre, _score) in enumerate(centres):
            if bench.redeploying(frame, centre, card_top=top) is not True:
                continue
            records.append({
                "frame": case.name, "index": index,
                "region": np.ascontiguousarray(
                    frame.crop(redcap_region(centre, pitch=PITCH))),
            })
    print("  语料里红罩卡 %d 张（落盘说 %d）" % (len(records), ledger["totals"]["red"]))

    by_slot = {(s["frame"], s["index"]): s for s in slots}
    self_check = []
    with Counting() as counting:
        ruler = margin_module.MaskedRuler(mask, faces, scales=NARROW)
        build_calls = counting.calls
        for record in records[:SELF_CHECK_SLOTS]:
            got = _verdict(ruler.scores(record["region"]))
            landed = by_slot[(record["frame"], record["index"])]
            self_check.append({
                "frame": record["frame"], "index": record["index"], "mine": got,
                "landed": {"name": landed["reading"]["name"],
                           "score": landed["score"], "scale": landed["scale"]},
                "score_delta": round(abs(got["score"] - landed["score"]), 9),
            })
    print("  自检（%d 张）：%s" % (SELF_CHECK_SLOTS,
          json.dumps(self_check, ensure_ascii=False)))

    small_mask = _shrink(mask.astype(np.uint8), COARSE_FACTOR, True) > 0
    small_faces = [(name, _shrink(art, COARSE_FACTOR, False)) for name, art in faces]
    coarse_rows = []
    windows: dict = {}
    fine_rulers: dict = {}
    with Counting() as counting:
        coarse_ruler = margin_module.MaskedRuler(small_mask, small_faces, scales=NARROW)
        coarse_build = counting.calls
        for record in records:
            small_region = _shrink(record["region"], COARSE_FACTOR, False)
            coarse = coarse_ruler.scores(small_region)
            picked = coarse[0][2]
            window = _scale_window(picked)
            windows[window] = windows.get(window, 0) + 1
            if window not in fine_rulers:
                fine_rulers[window] = margin_module.MaskedRuler(mask, faces, scales=window)
            got = _verdict(fine_rulers[window].scores(record["region"]))
            landed = by_slot[(record["frame"], record["index"])]
            coarse_rows.append({
                "frame": record["frame"], "index": record["index"],
                "coarse_pick": picked, "coarse_name": coarse[0][1],
                "coarse_score": round(coarse[0][0], 6),
                "window": list(window),
                "landed_scale": landed["scale"],
                "landed_in_window": any(abs(landed["scale"] - s) < 1e-9 for s in window),
                "landed": {"name": landed["reading"]["name"], "score": landed["score"],
                           "margin": landed["margin"],
                           "blocked_by": landed["blocked_by"]},
                "layered": got,
            })
        fine_calls = counting.calls - coarse_build

    before = [r for r in coarse_rows if r["landed"]["blocked_by"] is None]
    after = [r for r in coarse_rows if r["layered"]["verdict"] == "精确"]
    gained = [r for r in coarse_rows
              if r["landed"]["blocked_by"] is not None
              and r["layered"]["verdict"] == "精确"]
    lost = [r for r in coarse_rows
            if r["landed"]["blocked_by"] is None
            and r["layered"]["verdict"] == "未知"]
    truth_of = {(s["frame"], s["index"]): s.get("truth") for s in slots}

    def _is_wrong(name, key):
        truth = truth_of.get(key)
        return truth is not None and name != truth

    wrong_new = [r for r in gained
                 if _is_wrong(r["layered"]["name"], (r["frame"], r["index"]))]
    wrong_before = [r for r in before
                    if _is_wrong(r["landed"]["name"], (r["frame"], r["index"]))]
    wrong_after = [r for r in after
                   if _is_wrong(r["layered"]["name"], (r["frame"], r["index"]))]

    side = {}
    absolute = SAMPLES / "预处理" / "绝对线.json"
    if absolute.is_file():
        blob = json.loads(absolute.read_text(encoding="utf-8"))
        for row in blob["rows"]:
            key = "基线" if row["window"] is None else ("窗%d" % row["window"])
            if key in side:
                continue
            per_pair = row["ends"]["capped"]["per_pair"]
            passed = [p for p in per_pair
                      if p["winner_score"] >= ROSTER_SOLID_MATCH_SCORE
                      and p["margin"] >= ROSTER_AMBIGUITY_MARGIN]
            side[key] = {"pairs": len(per_pair), "exact": len(passed),
                         "unknown": len(per_pair) - len(passed)}
        side["corpus"] = "tests/帧样本/预处理（**不是**红罩那 28 张）"
        side["note"] = "这一栏是另一批语料上位置窗的读数，只作旁证，不混进判定账。"

    payload = {
        "note": "红罩那条路（MaskedRuler）的判定账：档从 11 收成粗层挑出来的 5 档。只量不改。",
        "baseline_source": {"file": "tests/帧样本/红罩/点名.json",
                            "red": ledger["totals"]["red"],
                            "exact": ledger["totals"]["exact"],
                            "unknown": ledger["totals"]["unknown"], "reused": True},
        "lines": {"score": ROSTER_SOLID_MATCH_SCORE,
                  "margin": ROSTER_AMBIGUITY_MARGIN},
        "coarse": {"factor": COARSE_FACTOR, "half_steps": HALF,
                   "mask_pixels": int(mask.sum()),
                   "coarse_build_calls": coarse_build,
                   "fine_rulers_built": len(fine_rulers)},
        "self_check": self_check,
        "cards": len(coarse_rows),
        "before": {"exact": len(before), "unknown": len(coarse_rows) - len(before)},
        "after": {"exact": len(after), "unknown": len(coarse_rows) - len(after)},
        "gained_n": len(gained), "lost_n": len(lost),
        "wrong_new_n": len(wrong_new), "wrong_before_n": len(wrong_before),
        "wrong_after_n": len(wrong_after),
        "landed_scale_in_window": sum(1 for r in coarse_rows if r["landed_in_window"]),
        "windows": {"+".join("%.2f" % s for s in w): n for w, n in windows.items()},
        "match_template_calls": {"coarse_build": coarse_build,
                                 "fine_build_and_scores": fine_calls,
                                 "today_28_cards_estimate": build_calls + 121 * len(records)},
        "rows": coarse_rows, "side_position_window": side,
        "seconds": round(time.perf_counter() - started, 3),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  判定账：%d -> %d 精确（落盘 %d）· 未知->精确 %d · 精确->未知 %d · 新错名字 %d"
          % (len(before), len(after), len(before), len(gained), len(lost), len(wrong_new)))
    print("  落盘那一档落在细层窗里的卡：%d/%d"
          % (payload["landed_scale_in_window"], len(coarse_rows)))
    print("  次数：粗尺子 %d · 细层 %d · 今天那 %d 张约 %d"
          % (coarse_build, fine_calls, len(records),
             build_calls + 121 * len(records)))
    print("总耗时 %.1f s" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
