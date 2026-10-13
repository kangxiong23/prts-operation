# -*- coding: utf-8 -*-
r"""两帧真跑：**一帧没有红罩卡**（备战席/shot01）· **一帧带红罩卡**（红罩/cd14）。

    .venv\Scripts\python.exe scripts\bench_roster_frame_paths.py

这一单的验收线：**两帧的中位墙钟都要 >= 10 倍**（对着旧口径 20.692 s 那个一帧）。
口径按裁定：**中位 + 三次全列 + 散度**。

同一帧还报**分路耗时**：干净卡那条（`roster_scores`）· 红罩卡那条（`MaskedRuler.scores`）·
建尺子（`MaskedRuler.__init__`）· 「是不是再部署中」（`redeploying`）· 其余（取区裁剪等）。
分路是把那几个函数各包一层记墙钟，**不是**另一条链。

**只 import 线上的入口**，不碰产品代码。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import ark.engine.vision.margin as margin_mod
import ark.engine.vision.roster as roster_mod
from ark.engine.device.adb import SQUAD_CARD_PITCH as PITCH
from ark.engine.vision.frame import frame_from_raw
from ark.offline.assets_gen.cardface import load_cardfaces
from ark.offline.bench.batch import load_batch

SAMPLES = REPO / "tests" / "帧样本"
OUT_DIR = SAMPLES / "先粗后细"
OUT = OUT_DIR / "两帧.json"
OLD_FRAME_S = 20.692
KINDS = ("clean", "disc", "build", "redeploy")
FRAMES = (("备战席", "shot01"), ("红罩", "cd14"))
ROUNDS = 3


class Counting:
    """把 cv2.matchTemplate 包一层数次数；顺便按调用方归路。"""

    def __init__(self):
        self.calls = 0
        self.by_path = {"clean": 0, "disc": 0}
        self.depth = 0
        self._real = cv2.matchTemplate

    def __enter__(self):
        counter = self

        def counted(source, template, method):
            counter.calls += 1
            if counter.depth:
                counter.by_path[counter.depth] = counter.by_path.get(counter.depth, 0) + 1
            return counter._real(source, template, method)

        cv2.matchTemplate = counted
        return self

    def __exit__(self, *_exc):
        cv2.matchTemplate = self._real
        return False


def main():
    started = time.perf_counter()
    faces = load_cardfaces(SAMPLES / "点名" / "编队页.json")
    real_box = roster_mod.roster_scores
    real_disc = margin_mod.MaskedRuler.scores
    real_build = margin_mod.MaskedRuler.__init__
    real_flag = roster_mod.redeploying
    stats = {kind: 0.0 for kind in KINDS}
    stats["clean_n"] = 0
    stats["disc_n"] = 0
    stats["build_n"] = 0
    counting = Counting()

    def box(region, faces_arg):
        counting.depth = "clean"
        began = time.perf_counter()
        out = real_box(region, faces_arg)
        stats["clean"] += time.perf_counter() - began
        stats["clean_n"] += 1
        counting.depth = 0
        return out

    def disc(self, region):
        counting.depth = "disc"
        began = time.perf_counter()
        out = real_disc(self, region)
        stats["disc"] += time.perf_counter() - began
        stats["disc_n"] += 1
        counting.depth = 0
        return out

    def build(self, *args, **kwargs):
        began = time.perf_counter()
        out = real_build(self, *args, **kwargs)
        stats["build"] += time.perf_counter() - began
        stats["build_n"] += 1
        return out

    def flag(frame, centre, *, card_top):
        began = time.perf_counter()
        out = real_flag(frame, centre, card_top=card_top)
        stats["redeploy"] += time.perf_counter() - began
        return out

    roster_mod.roster_scores = box
    margin_mod.MaskedRuler.scores = disc
    margin_mod.MaskedRuler.__init__ = build
    roster_mod.redeploying = flag

    report = {}
    for batch, name in FRAMES:
        cases = {case.name: case for case in load_batch(SAMPLES / batch).cases}
        if name not in cases:
            print("这一批里没有 %r：%s" % (name, sorted(cases)))
            return 2
        frame = frame_from_raw(cases[name].raw())
        rounds = []
        last = None
        for _ in range(ROUNDS):
            for key in KINDS:
                stats[key] = 0.0
            stats["clean_n"] = stats["disc_n"] = stats["build_n"] = 0
            with counting:
                counting.calls = 0
                counting.by_path = {"clean": 0, "disc": 0}
                began = time.perf_counter()
                got = roster_mod.read_squad_identities(frame, faces, pitch=PITCH)
                elapsed = time.perf_counter() - began
            last = {
                "elapsed": round(elapsed, 4),
                "calls": counting.calls,
                "calls_by_path": dict(counting.by_path),
                "split": {key: round(stats[key], 4) for key in KINDS},
                "split_n": {"clean": stats["clean_n"], "disc": stats["disc_n"],
                            "build": stats["build_n"]},
                "calls_per_slot": round(counting.calls / max(1, len(got)), 2),
            }
            rounds.append(last)
        times = [r["elapsed"] for r in rounds]
        median = round(statistics.median(times), 4)
        spread = round(max(times) / min(times), 3) if min(times) else None
        chosen = min(rounds, key=lambda r: abs(r["elapsed"] - median))
        split = chosen["split"]
        slots = []
        for index, value in enumerate(got):
            payload = getattr(value, "payload", None)
            slots.append({"index": index,
                          "payload": type(payload).__name__ if payload is not None else None,
                          "name": getattr(payload, "name", None)
                          if payload is not None else None})
        report[batch + "/" + name] = {
            "batch": batch, "frame": name, "cards": len(got), "library": len(faces),
            "rounds_s": times, "median_s": median, "spread_max_over_min": spread,
            "calls": rounds[0]["calls"],
            "calls_by_path": {
                "clean": [r["calls_by_path"].get("clean") for r in rounds],
                "disc": [r["calls_by_path"].get("disc") for r in rounds],
            },
            "ratio_vs_old_frame": round(OLD_FRAME_S / median, 3) if median else None,
            "split_s": split,
            "split_share": {key: round(split[key] / median, 4) for key in KINDS},
            "split_n": chosen["split_n"],
            "redcap_slots": chosen["split_n"]["disc"],
            "other_s": round(median - sum(split.values()), 4),
            "calls_per_slot": rounds[0]["calls_per_slot"],
            "slots": slots,
        }
        print("一帧 %s/%s：%d 卡位 x %d 条库" % (batch, name, len(got), len(faces)))
        print("  真跑三次 %s -> 中位 %.3f s（散度 %.2fx）· 调用 %d 次"
              % (times, median, spread, rounds[0]["calls"]))
        print("  对旧口径 %.3f s 的倍数 = %.2fx" % (OLD_FRAME_S, OLD_FRAME_S / median))
        print("  分路：干净 %.3f s (%d 次) · 红罩 %.3f s (%d 次) · 建尺子 %.3f s (%d 次)"
              " · 判再部署 %.3f s · 其他 %.3f s"
              % (split["clean"], chosen["split_n"]["clean"], split["disc"],
                 chosen["split_n"]["disc"], split["build"], chosen["split_n"]["build"],
                 split["redeploy"], median - sum(split.values())))
        print("  每格：" + " ".join("%d:%s" % (s["index"], s["name"] or s["payload"])
                                    for s in slots))

    payload = {
        "note": "两帧真跑（无红罩 / 带红罩）：三次全列 + 中位 + 对旧口径 20.692 s 的倍数 + 分路耗时。",
        "old_frame_s": OLD_FRAME_S,
        "old_frame_source": "tests/帧样本/先粗后细/一帧.json 的真跑中位（已裁定口径）",
        "pitch": PITCH,
        "rounds": ROUNDS,
        "frames": report,
        "seconds": round(time.perf_counter() - started, 3),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("总耗时 %.1f s" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
