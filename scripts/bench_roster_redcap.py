
r"""备战席点名**接上红罩卡**那一单的读数表。

    .venv\Scripts\python.exe scripts\bench_roster_redcap.py [--limit K]

## 这一单问什么

再部署中的红罩卡原来一律报「未知」；这一单把它接上（D-全局-34）—— **只换取区，
判据一个字节没动**。这份表把**同一批真帧**上的读数、真值、干净卡的逐位对照与**代价**
一起落下来。

## 口径

- **入口**：`roster.read_squad_identities`（今天那两个入口之一，**一个都没另开**）；
- **取区**：`redcap_region` —— 环带以内的圆盘、挖掉中央秒数那一条（形状住
  `margin.IDENTITY_DISC_*`）；干净卡走身份区那个 100×100 的框，**照旧**；
- **统计量**：同一块取区上、同一个「逐通道减均值的相关」（`margin.MaskedRuler` 掩码
  全开时逐位复现 `roster_scores`）。**不反解** —— 罩子对这条判据天生透明（D-全局-28）；
- **真值**：`红罩/取材.json` 那 4 对 + `预处理/取材.json` 落在同一帧上的那 1 对
  （`cd23_6`）。两份都是"同一张卡后来干净了那一帧"读出来的，不是目视。
- **每一格把次高分 · 余量 · 被哪条线挡的一起落下来**：判定仍由那两条线算，而
  `_ranking` 走的是**同一块取区 · 同一个打分函数 · 同一个分支**，并与入口
  `read_squad_identities` 的读数**逐位核对**（`_crosscheck`，对不上就报错）——
  所以**挪线可以离线重算**，不必把整批重跑。

## 「改动前」怎么量（这份表里最短的一条路）

**同一段代码、把 `redeploying` 钉成 `False`**：那条路就是接红罩之前那条
（每一格都走身份区 + `roster_scores`）。于是同一份进程里跑两遍 —— 一遍真的、
一遍钉住的 —— 干净卡**逐位**比、墙钟**同机**比。**没有另改一份实现去比。**

## 代价那几条是怎么量的

- **建尺子 / 一张红罩卡 / 一张干净卡**：同一块画面上各跑 3 次取中位（单线程）；
- **整帧**：那 8 帧各跑一遍真的与钉住的那一遍，墙钟之差就是这一单要的账。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import ark.engine.vision.roster as roster_module  # noqa: E402
from ark.contracts import Exact, Valued  # noqa: E402
from ark.engine.device.adb import SQUAD_CARD_PITCH  # noqa: E402
from ark.engine.vision import bench, margin  # noqa: E402
from ark.engine.vision.frame import Frame, frame_from_raw  # noqa: E402
from ark.engine.vision.roster import (  # noqa: E402
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
    identity_region,
    read_squad_identities,
    redcap_region,
    roster_scores,
)
from ark.offline.assets_gen.cardface import load_cardfaces  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402

BATCH = REPO / "tests" / "帧样本" / "红罩"
HARVEST = BATCH / "取材.json"
PREPROCESS = REPO / "tests" / "帧样本" / "预处理" / "取材.json"
OUT = BATCH / "点名.json"

PITCH = SQUAD_CARD_PITCH
"""卡距。它住设备层，视觉层不许 import 设备层，所以照调用方的样子传进来。"""


def _value(value) -> dict:
    """一个槽值 → 可写进 JSON 的几项（**未知不带分数**这一条也在这里落）。"""
    if not isinstance(value, Valued):
        return {"shape": type(value).__name__, "payload": None, "name": None, "score": None}
    payload = value.payload
    return {
        "shape": "Valued",
        "payload": type(payload).__name__,
        "name": payload.value if isinstance(payload, Exact) else None,
        "score": value.confidence,
    }


def _truth() -> dict[tuple[str, int], str]:
    """红罩卡的**逐位真值**：两份取材表的 `under_the_cap`。"""
    found: dict[tuple[str, int], str] = {}
    for path in (HARVEST, PREPROCESS):
        record = json.loads(path.read_text(encoding="utf-8"))
        for pair in record["pairs"]:
            frame = Path(pair["capped"]["file"]).stem.split("_")[0]
            found[(frame, int(pair["capped"]["index"]))] = pair["under_the_cap"]
    return found


def _run(frame: Frame, faces) -> tuple:
    return read_squad_identities(frame, faces, pitch=PITCH)


def _run_as_before(frame: Frame, faces) -> tuple:
    """**同一段代码**，把 `redeploying` 钉成 `False` —— 那就是接红罩之前那条路。"""
    real = roster_module.redeploying
    roster_module.redeploying = lambda *args, **kwargs: False
    try:
        return read_squad_identities(frame, faces, pitch=PITCH)
    finally:
        roster_module.redeploying = real


def _ranking(frame: Frame, faces, centres, top: float) -> list:
    """**逐卡把整列分数留下来**（最高分 · 次高分 · 余量）—— 与 `read_squad_identities`
    同一块取区、同一个打分函数、同一个分支。

    **它不另立判据**：判定仍按那两条线算（`ROSTER_SOLID_MATCH_SCORE` /
    `ROSTER_AMBIGUITY_MARGIN`），而且与入口的读数**逐位核对**（`_crosscheck`，
    对不上就报错）。记它的理由只有一个：**线一挪，判定要能离线重算** —— 以前只落了
    「精确 / 未知」，没落次高分与余量，所以挪一次线就得把整批重跑一遍。
    """
    redcap = None
    rows = []
    for index, (centre, _) in enumerate(centres):
        if bench.redeploying(frame, centre, card_top=top) is True:
            if redcap is None:
                redcap = roster_module._redcap_scorer(faces)
            region = np.ascontiguousarray(frame.crop(redcap_region(centre, pitch=PITCH)))
            scored = redcap(region)
        else:
            region = np.ascontiguousarray(frame.crop(identity_region(centre, pitch=PITCH)))
            scored = roster_scores(region, faces)
        best = scored[0]
        second = scored[1] if len(scored) > 1 else (-2.0, None, None)
        rows.append({
            "index": index,
            "name": best[1],
            "score": float(best[0]),
            "scale": None if best[2] is None else float(best[2]),
            "runner_up": second[1],
            "runner_up_score": None if second[1] is None else float(second[0]),
            "margin": float(best[0]) - float(second[0]),
        })
    return rows


def _blocked_by(row: dict) -> str | None:
    """这一格**被哪一条线挡的**（判定过了就返回 None）；两条都没够就两条都写。"""
    short = []
    if row["score"] < ROSTER_SOLID_MATCH_SCORE:
        short.append("分数不够")
    if row["margin"] < ROSTER_AMBIGUITY_MARGIN:
        short.append("余量不够")
    return " + ".join(short) or None


def _crosscheck(after, rows) -> None:
    """入口的读数与逐卡整列**逐位核对** —— 这就是「表里的判定 = 线上那条链」的凭据。"""
    if len(after) != len(rows):
        raise AssertionError("卡数对不上：%d vs %d" % (len(after), len(rows)))
    for value, row in zip(after, rows):
        payload = value.payload
        if isinstance(payload, Exact):
            if payload.value != row["name"]:
                raise AssertionError("名字对不上：%r vs %r" % (payload.value, row["name"]))
            if abs(float(value.confidence) - row["score"]) > 1e-9:
                raise AssertionError("分数对不上：%r vs %r" % (value.confidence, row["score"]))
        elif value.confidence is not None:
            raise AssertionError("未知带了分数：%r" % (value,))


def _match_calls(work) -> int:
    """跑一遍 `work()`，数它调了几次 `cv2.matchTemplate`。

    **这份账比墙钟稳**：同一段代码在这台机器上墙钟能差 1.5 倍（三次测量里干净卡是
    1.41 / 1.70 / 2.12 s），而次数是确定的。
    """
    import cv2

    real = cv2.matchTemplate
    counted = [0]

    def spy(*args, **kwargs):
        counted[0] += 1
        return real(*args, **kwargs)

    cv2.matchTemplate = spy
    try:
        work()
    finally:
        cv2.matchTemplate = real
    return counted[0]


def _timings(frame: Frame, faces, *, rounds: int = 3) -> dict:
    """拆项：建一把尺子 / 一张红罩卡 / 一张干净卡（单线程、各跑 `rounds` 次取中位）。"""
    centres = bench.squad_card_centres(frame, pitch=PITCH)
    top = float(bench.squad_card_rect(0, PITCH).y)
    _box, mask = margin.identity_disc_region()
    clean_region = np.ascontiguousarray(frame.crop(identity_region(centres[0][0], pitch=PITCH)))
    red_region = np.ascontiguousarray(frame.crop(redcap_region(centres[0][0], pitch=PITCH)))
    ruler = margin.MaskedRuler(mask, faces)
    calls = {
        "one_clean_card": _match_calls(lambda: roster_scores(clean_region, faces)),
        "build_ruler": _match_calls(lambda: margin.MaskedRuler(mask, faces)),
        "one_red_card": _match_calls(lambda: ruler.scores(red_region)),
    }
    builds, disc, box_scores, flags = [], [], [], []
    for _ in range(rounds):
        started = time.perf_counter()
        ruler = margin.MaskedRuler(mask, faces)
        builds.append(time.perf_counter() - started)
        started = time.perf_counter()
        ruler.scores(red_region)
        disc.append(time.perf_counter() - started)
        started = time.perf_counter()
        roster_scores(clean_region, faces)
        box_scores.append(time.perf_counter() - started)
        started = time.perf_counter()
        bench.redeploying(frame, centres[0][0], card_top=top)
        flags.append(time.perf_counter() - started)
    median = statistics.median
    return {
        "calls": calls,
        "calls_note": (
            "`cv2.matchTemplate` 次数 —— **这份账比墙钟稳**（墙钟同代码差 1.5 倍）。"
            "红罩卡与干净卡**同为 %d 次**；多花的是墙钟：掩码那把尺子在每个"
            "（条目, 尺度）上多做几处整张相关图的逐元素运算。**每帧第一次用红罩那条"
            "路 +%d 次**（建尺子，只建一次；没有红罩卡的帧不建）。"
            % (calls["one_red_card"], calls["build_ruler"])
        ),
        "build_ruler_s": round(median(builds), 4),
        "one_red_card_s": round(median(disc), 4),
        "one_clean_card_s": round(median(box_scores), 4),
        "redeploying_s": round(median(flags), 5),
        "note": (
            "同一块画面上各跑 %d 次取中位（单线程）。**每张红罩卡多花** = 一张红罩卡 −"
            " 一张干净卡；**有红罩卡的帧再多花**一次建尺子。" % rounds
        ),
    }


def main(argv: list[str]) -> int:
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 0

    batch = load_batch(BATCH)
    frames = {case.name: frame_from_raw(case.raw()) for case in batch.cases}
    faces = load_cardfaces()
    truth = _truth()

    components = _timings(frames["cd14"], faces)

    names = sorted(frames)
    if limit:
        names = names[:limit]

    output: list[dict] = []
    totals = {"red": 0, "exact": 0, "unknown": 0, "truth": 0, "truth_ok": 0,
              "clean": 0, "clean_changed": 0}
    blocked: dict = {}
    seconds_after = seconds_before = 0.0
    for name in names:
        frame = frames[name]
        top = float(bench.squad_card_rect(0, PITCH).y)
        centres = bench.squad_card_centres(frame, pitch=PITCH)
        started = time.perf_counter()
        after = _run(frame, faces)
        after_s = time.perf_counter() - started
        started = time.perf_counter()
        before = _run_as_before(frame, faces)
        before_s = time.perf_counter() - started
        rows = _ranking(frame, faces, centres, top)
        _crosscheck(after, rows)
        seconds_after += after_s
        seconds_before += before_s
        slots = []
        for index, (value, was, (centre, _)) in enumerate(zip(after, before, centres)):
            flag = bench.redeploying(frame, centre, card_top=top)
            row = rows[index]
            entry = {
                "index": index,
                "centre_x": round(float(centre), 3),
                "redeploying": flag,
                "reading": _value(value),
                "score": round(row["score"], 6),
                "scale": None if row["scale"] is None else round(row["scale"], 6),
                "runner_up": row["runner_up"],
                "runner_up_score": (None if row["runner_up_score"] is None
                                    else round(row["runner_up_score"], 6)),
                "margin": round(row["margin"], 6),
                "blocked_by": _blocked_by(row),
                "as_before": _value(was),
                "truth": truth.get((name, index)),
            }
            if entry["blocked_by"] is not None:
                blocked[entry["blocked_by"]] = blocked.get(entry["blocked_by"], 0) + 1
            if flag is True:
                totals["red"] += 1
                if entry["reading"]["payload"] == "Exact":
                    totals["exact"] += 1
                else:
                    totals["unknown"] += 1
                if entry["truth"] is not None:
                    totals["truth"] += 1
                    totals["truth_ok"] += int(entry["reading"]["name"] == entry["truth"])
            else:
                totals["clean"] += 1
                if value != was:
                    totals["clean_changed"] += 1
            slots.append(entry)
        output.append(
            {
                "name": name,
                "cards": len(after),
                "seconds_after": round(after_s, 3),
                "seconds_as_before": round(before_s, 3),
                "slots": slots,
            }
        )
        print("  %-6s %2d cards  after %6.2f s  before %6.2f s"
              % (name, len(after), after_s, before_s))

    report = {
        "note": (
            "红罩卡接上今天那条入口之后的读数表：**只换取区**，判据 · 线 · 取区形状一个字节"
            "没动（D-全局-34）。真值来自两份取材表的 `under_the_cap`；「改动前」= **同一"
            "段代码把 `redeploying` 钉成 `False`**（那就是接红罩之前那条路）。"
        ),
        "batch": "tests/帧样本/红罩",
        "harvest": "tests/帧样本/红罩/取材.json",
        "library": "tests/帧样本/点名/编队页.json",
        "entry": "ark.engine.vision.roster.read_squad_identities",
        "region": {
            "red_capped": {
                "who": "roster.redcap_region",
                "box_in_card": list(margin.IDENTITY_DISC_BOX),
                "pixels": int(margin.identity_disc_mask().sum()),
                "shape": (
                    "环带以内的圆盘 − 中央秒数那一条；形状住 margin.IDENTITY_DISC_RADIUS /"
                    " IDENTITY_DISC_STRIP_HALF（一处定义）"
                ),
            },
            "clean": {
                "who": "roster.identity_region",
                "box_in_card": [52, 76, 100, 100],
                "pixels": 100 * 100,
            },
            "note": (
                "两块取区**分开记、不互相顶替**：哪一格用哪块由 `bench.redeploying` 说。"
                "两条路**共用**同一条线（ROSTER_SOLID_MATCH_SCORE /"
                " ROSTER_AMBIGUITY_MARGIN）。"
            ),
        },
        "criterion": {
            "score": ROSTER_SOLID_MATCH_SCORE,
            "margin": ROSTER_AMBIGUITY_MARGIN,
            "note": (
                "读得出 ⇒ 精确（分数就是定下它的那个）；读不出 ⇒ 未知，**不带分数**。"
                "**每一格都把次高分（runner_up / runner_up_score）· 余量（margin）·"
                " 被哪条线挡的（blocked_by）一起落下来** —— 以前只落了判定，所以挪一次线"
                "就要把整批重跑。"
            ),
        },
        "blocked_by": {
            "counts": blocked,
            "note": "仍未知的那些卡被哪一条线挡的（分数不够 / 余量不够 / 两条都没够）——"
                    "**这两样现在分得开**；判定与入口读数逐位核对过（_crosscheck）。",
        },
        "cost": {
            "components": components,
            "frames": {
                "after_s": round(seconds_after, 3),
                "as_before_s": round(seconds_before, 3),
                "delta_s": round(seconds_after - seconds_before, 3),
                "frames": len(names),
                "note": (
                    "同一份进程里跑两遍：一遍真的、一遍把 `redeploying` 钉成"
                    " `False`（= 接红罩之前那条路）。差就是这一单在**这一批帧**上的账。"
                ),
            },
        },
        "totals": totals,
        "frames": output,
    }
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print()
    print("=== 代价（拆项）===")
    print("  建一把尺子            %.3f s / 帧（只在真有红罩卡时）" % components["build_ruler_s"])
    print("  一张红罩卡            %.3f s" % components["one_red_card_s"])
    print("  一张干净卡            %.3f s" % components["one_clean_card_s"])
    print("  redeploying（环剖面） %.5f s" % components["redeploying_s"])
    print("  → 每张红罩卡多花 %.3f s；有红罩卡的帧再多花 %.3f s"
          % (components["one_red_card_s"] - components["one_clean_card_s"],
             components["build_ruler_s"]))
    print()
    print("=== 整批 ===")
    print("  %d 帧合计：真的 %.2f s / 钉住的 %.2f s → %+.2f s"
          % (len(names), seconds_after, seconds_before, seconds_after - seconds_before))
    print("  红罩卡 %d：精确 %d / 未知 %d；有真值 %d 张，名字对上的 %d"
          % (totals["red"], totals["exact"], totals["unknown"],
             totals["truth"], totals["truth_ok"]))
    print("  仍未知的卡被哪条线挡的：%s" % (blocked or "没有"))
    print("  干净卡 %d：与改动前**逐位相同** %d（不同的 %d）"
          % (totals["clean"], totals["clean"] - totals["clean_changed"], totals["clean_changed"]))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
