# -*- coding: utf-8 -*-
r"""丙：**真跑一帧** —— 点名那条链在 shot01 上的墙钟与 matchTemplate 次数。

    .venv\Scripts\python.exe scripts\bench_roster_frame.py

上一单那个「一帧 21.125 s」是我按落盘的每次 14.549 ms **折出来的**，不是真跑。
这一份真跑：走线上那一个入口 `roster.read_squad_identities`（红罩卡照样走红罩那条路），
数真次数、量真墙钟（3 次取中位），与折算并排，**定下以后统一用哪个口径**。

**一个字节都不改**：只 import 线上的入口，不碰任何产品代码。
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

from ark.engine.device.adb import SQUAD_CARD_PITCH as PITCH  # noqa: E402
from ark.engine.vision.frame import frame_from_raw  # noqa: E402
from ark.engine.vision.roster import read_squad_identities  # noqa: E402
from ark.offline.assets_gen.cardface import load_cardfaces  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402

SAMPLES = REPO / "tests" / "帧样本"
OUT_DIR = SAMPLES / "先粗后细"
OUT = OUT_DIR / "一帧.json"
READINGS = OUT_DIR / "读数.json"
BATCH = "备战席"
FRAME = "shot01"
ROUNDS = 3


class Counting:
    """把 cv2.matchTemplate 包一层数次数。"""

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


def main() -> int:
    started = time.perf_counter()
    cases = {case.name: case for case in load_batch(SAMPLES / BATCH).cases}
    if FRAME not in cases:
        print("这一批里没有 %r：%s" % (FRAME, sorted(cases)))
        return 2
    frame = frame_from_raw(cases[FRAME].raw())
    faces = load_cardfaces(SAMPLES / "点名" / "编队页.json")

    rounds = []
    calls = 0
    got = None
    for _ in range(ROUNDS):
        with Counting() as counting:
            began = time.perf_counter()
            got = read_squad_identities(frame, faces, pitch=PITCH)
            elapsed = time.perf_counter() - began
        calls = counting.calls
        rounds.append(round(elapsed, 4))
    median = round(statistics.median(rounds), 4)

    readings = json.loads(READINGS.read_text(encoding="utf-8"))
    reuse = readings["cost_measured"]["甲的今天：全分辨率全档"]
    per_call = reuse["median_s"] / reuse["match_template_calls"]
    folded = round(len(got) * len(faces) * 11 * per_call, 4)

    slots = []
    for index, value in enumerate(got):
        payload = getattr(value, "payload", None)
        name = getattr(payload, "name", None) if payload is not None else None
        slots.append({"index": index, "shape": type(value).__name__,
                      "payload": type(payload).__name__ if payload is not None else None,
                      "name": name})

    payload_out = {
        "note": "丙：真跑一帧（点名那条链的线上入口 read_squad_identities）。只量不改。",
        "batch": BATCH, "frame": FRAME,
        "cards": len(got), "library": len(faces),
        "match_template_calls": calls,
        "rounds_s": rounds, "median_s": median,
        "folded_estimate": {
            "seconds": folded,
            "source": "tests/帧样本/先粗后细/读数.json 的 cost_measured（复用）",
            "formula": "卡位 %d x 库 %d x 11 档 x 每次 %.6f s" % (
                len(got), len(faces), per_call),
        },
        "slots": slots,
        "seconds": round(time.perf_counter() - started, 3),
    }
    OUT.write_text(json.dumps(payload_out, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("一帧 %s/%s：%d 卡位 x %d 条库" % (BATCH, FRAME, len(got), len(faces)))
    print("  真跑 %s 次 / 中位 %.3f s（%s）" % (calls, median, rounds))
    print("  折算（复用 读数.json）%d 次 / %.3f s" % (len(got) * len(faces) * 11, folded))
    print("  真跑/折算 = %.3f" % (median / folded if folded else 0.0))
    print("  每格：" + " ".join(
        "%d:%s" % (s["index"], s["name"] or s["payload"]) for s in slots))
    print("总耗时 %.1f s" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
