# -*- coding: utf-8 -*-
r"""第六单那张网：**红罩那一刀 · 每卡位×每条几何 · 复用路自己的分线 · 画面没变就跳过**。

    .venvScriptspython.exe scriptsench_roster_geometry.py

素材是 `data/precheck/cd/` 那一整段 —— **同一局**真机战斗每 3 秒一张、连截 60 张
（`cd01…cd60`，gitignored 临时产物，但就在盘上）。它盖住的 8 帧与
`tests/帧样本/红罩/` 里那 8 个裁剪**横带逐位相同**（脚本第一步就核这件事）。

五块：

1. **逐张对比**（验收线）：每一帧每一个卡位，线上那条路（复用几何 / 跳过的沿用）
   与"全分辨率跑满"那条路（`_layered_scores(coarse_accept=False)` / 掩码尺子整张滑）
   比 **（判定, 名字）** —— 丢 0 · 名字不许变 · 新错 0。
2. **红罩那一刀**：同一批红罩卡上，整张滑（`scores_and_positions`）与分层
   （`_masked_layered_scores`）的墙钟，外加**逐对**（判定, 名字）核对。
3. **复用那条路自己的分线**：把每个有几何的格子、每条库**在它自己的对位上**的分按
   "是不是真匹配"分两簇，报最低/最高/空档（线要坐在空档里）。
4. **画面没变就跳过**：跨帧**逐位相同**的比例（干净/红罩分开）· 每帧跳了几格 ·
   跳过的格沿用的判定有没有出过错。
5. **代价与账**：首帧 / 后续帧 · 升级账（前名/后名/分数）· 几何抖动的收敛 ·
   掩码尺子改成每局一次省掉多少 · 素材体量。

第七单加的三块：

6. **甲1 去重**：锚点层按模板尺寸去重（干净那条路 18 档 → 5 个尺寸；掩码那条路
   25 像素层 18 → 9、50 像素层 18 → 12）。省下的时间由「网」那一节的逐张对比兜底
   （判定与名字必须逐张相同）。
7. **甲2 尺子懒建**：那把掩码尺子不再一建就建满 —— 它现在砸在**第一次出现红罩卡
   那一帧**上。`ruler_spike` 报那一帧凸起多少、它之后各帧中位多少、懒建到底建了多少。
   参照（整张滑）那条路**用自己那把尺子**，两把不混。
8. **甲3 按层拆解**：每格的墙钟拆到锚点层 / 位置层 / 精算层 / 复用（掩码那条路同一套
   键）。它是「还能不能再压」唯一的依据。
"""
from __future__ import annotations

import hashlib
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

import bench_roster_coarse as base
import ark.engine.vision.roster as R
from ark.engine.device.adb import SQUAD_CARD_PITCH as PITCH
from ark.engine.vision import bench
from ark.engine.vision.frame import Canvas, Frame
from ark.engine.vision.roster import identity_region, identity_is_exact, redcap_region
from ark.engine.vision import margin as margin_mod
from ark.offline.assets_gen.cardface import load_cardfaces

SAMPLES = REPO / "tests" / "帧样本"
OUT_DIR = SAMPLES / "先粗后细"
OUT = OUT_DIR / "几何复用.json"
CD = REPO / "data" / "precheck" / "cd"
SHARED_LIBRARY = SAMPLES / "点名" / "编队页.json"
TRUTH = base._truth_map()
FIXTURE_FRAMES = ("cd14", "cd15", "cd22", "cd23", "cd27", "cd28", "cd37", "cd59")


def full_frame(path: Path) -> Frame:
    """`data/precheck/cd/cdNN.png` → 一整帧（2800×1260，RGB）。"""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit("读不到 " + str(path))
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return Frame(canvas=Canvas(width=rgb.shape[1], height=rgb.shape[0]), pixels=rgb)


def name_of(value) -> str | None:
    payload = getattr(value, "payload", None)
    if payload is None or type(payload).__name__ != "Exact":
        return None
    return payload.value


def verdict_of(ranked, known):
    top = ranked[0][0]
    runner = ranked[1][0] if len(ranked) > 1 else -2.0
    if not identity_is_exact(top, runner):
        return {"verdict": "未知", "name": None, "top": top, "runner": runner}
    name = ranked[0][1]
    wrong = None if known is None else (known != "" and name != known)
    return {"verdict": "精确", "name": name, "top": top, "runner": runner,
            "most_like": name, "wrong": wrong}


def reference(frame, faces, centres, ruler):
    """**全分辨率跑满**那条路：干净卡走 `coarse_accept=False` 的扫法，红罩卡整张滑。"""
    card_top = float(bench.squad_card_rect(0, PITCH).y)
    out = []
    for index, (centre, _) in enumerate(centres):
        red = bench.redeploying(frame, centre, card_top=card_top) is True
        rect = redcap_region(centre, pitch=PITCH) if red else identity_region(centre, pitch=PITCH)
        region = np.ascontiguousarray(frame.crop(rect))
        began = time.perf_counter()
        if red:
            rows = ruler.scores_and_positions(region)
            ranked = tuple((score, name, scale) for score, name, scale, _p in rows)
        else:
            ranked, _trace = R._layered_scores(region, faces, coarse_accept=False)
        elapsed = time.perf_counter() - began
        known = TRUTH.get((frame_name, index))
        # **红罩那一刀**要用同一份读数：整张滑那一趟在上面已经付过钱了，不再算第二遍。
        out.append({"index": index, "red": red, "seconds": round(elapsed, 4),
                    "ranked": ranked if red else None, **verdict_of(ranked, known)})
    return out


def corpus_phase() -> dict:
    """全语料：粗层接受到底把答案改没改。"""
    corpus, libraries = base._corpus()
    truth = base._truth_map()
    rows = []
    for record in corpus:
        if record["redeploying"] is True:
            continue
        faces = libraries[record["library"]]
        region = record["region"]
        began = time.perf_counter()
        coarse, trace = R._layered_scores(region, faces, coarse_accept=True)
        coarse_s = time.perf_counter() - began
        began = time.perf_counter()
        full, _ = R._layered_scores(region, faces, coarse_accept=False)
        full_s = time.perf_counter() - began
        known = truth.get((record["frame"], record["index"]))
        layer = trace["fine"]["layer"] if trace["coarse"] else None
        rows.append({
            "batch": record["batch"], "frame": record["frame"], "index": record["index"],
            "layer": layer,
            "coarse": verdict_of(coarse, known), "full": verdict_of(full, known),
            "coarse_seconds": round(coarse_s, 4), "full_seconds": round(full_s, 4),
            "tried": len(trace["fine"]["tried"]),
            "coarse_top": None if trace["coarse"] is None else trace["coarse"]["top"],
            "coarse_runner": None if trace["coarse"] is None else trace["coarse"]["runner"],
        })
    same = [r for r in rows if r["coarse"]["verdict"] == r["full"]["verdict"]
            and r["coarse"]["name"] == r["full"]["name"]]
    differ = [r for r in rows if r not in same]
    layers = {}
    for row in rows:
        key = "全分辨率" if row["layer"] is None else row["layer"]
        layers[key] = layers.get(key, 0) + 1
    return {
        "cards": len(rows),
        "layers": layers,
        "same": len(same),
        "differ": len(differ),
        "differ_rows": differ[:40],
        "rows": rows,
    }


def measure_material() -> dict:
    """素材体量：这份网**依赖一个 gitignored 路径**，体量要报出来。"""
    files = sorted(CD.glob("cd*.png"))
    total = sum(path.stat().st_size for path in files)
    return {"path": "data/precheck/cd", "files": len(files), "bytes": total,
            "mib": round(total / 1048576.0, 2)}


def main() -> int:
    started = time.perf_counter()
    faces = load_cardfaces(SHARED_LIBRARY)
    names = sorted(p.stem for p in CD.glob("cd*.png"))
    print("=== 素材 ===")
    material = measure_material()
    print("  同一局那一整段：%d 帧（%s…%s）· %s MiB（%s，**gitignored**）"
          % (len(names), names[0], names[-1], material["mib"], material["path"]))
    frames: dict[str, Frame] = {}
    layout = []
    for name in names:
        frame = full_frame(CD / (name + ".png"))
        frames[name] = frame
        centres = bench.squad_card_centres(frame, pitch=PITCH)
        layout.append({"frame": name, "cards": len(centres),
                       "centres": [round(float(c), 1) for c, _ in centres]})
    counts = [row["cards"] for row in layout]
    print("  卡位数：%s" % {n: counts.count(n) for n in sorted(set(counts))})
    print("  卡位数变过的相邻对：%s"
          % [(layout[i - 1]["frame"], layout[i]["frame"])
             for i in range(1, len(layout)) if layout[i]["cards"] != layout[i - 1]["cards"]])
    # 与夹具同源核对（红罩那 8 帧）
    from ark.offline.bench.batch import load_batch
    fixture = {case.name: case for case in load_batch(SAMPLES / "红罩").cases}
    manifest = json.loads((SAMPLES / "红罩" / "index.json").read_text(encoding="utf-8"))
    band_x, band_y = manifest["crops"][0]["origin"]
    band_w, band_h = manifest["crops"][0]["size"]
    identical = True
    for name in FIXTURE_FRAMES:
        mine = frames[name].pixels[band_y:band_y + band_h, band_x:band_x + band_w]
        theirs = base.frame_from_raw(fixture[name].raw()).pixels[
            band_y:band_y + band_h, band_x:band_x + band_w]
        same = mine.shape == theirs.shape and bool((mine == theirs).all())
        identical = identical and same
        print("  夹具同源 %s（横带 %s）：%s" % (name, mine.shape, same))
    print("  8 帧横带全同：%s" % identical)

    # ---- 掩码尺子：每局建一次的那把（这一步的账单独列）----
    assets = R.SquadGeometry()
    began = time.perf_counter()
    ruler = assets.ruler(faces)
    ruler_build = time.perf_counter() - began
    print("=== 建尺子（一局一次） %.3f s ===" % ruler_build)
    print("  尺子用得上的档：%s" % (ruler.usable_scales,))
    # **一把尺子若一次建满全分辨率那一套要多久**（用户裁定 2 要的那个对照数）：
    # 拿一把**用完就丢**的量一下 —— 两把尺子一起留在内存里会把这一局的足迹撑到
    # 两倍（全分辨率那套的通道图一把就有几百 MB），那会让后面所有墙钟都变味。
    _box, _mask = margin_mod.identity_disc_region()
    _scratch = margin_mod.MaskedRuler(_mask, faces)
    began = time.perf_counter()
    _ = _scratch._prepared
    full_build = time.perf_counter() - began
    del _scratch
    print("  一把尺子若一次建满全分辨率那一套：%.3f s（量完就丢，不占着内存）" % full_build)

    # ---- 逐帧：线上那条路 vs 全分辨率跑满 ----
    print("=== 逐张对比（同一局 %d 帧） ===" % len(names))
    per_frame = []
    diffs = []
    cards = 0
    knife = {"full_s": [], "layered_s": [], "pairs": [], "differ": []}
    # (真值口径的标签, 参照口径的标签, 分, 帧, 卡位, 谁, 真值, 参照给的名字)
    clusters: list[tuple[int, int, float, str, int, str, str, str | None]] = []
    identical_pixels = {"干净": [0, 0], "红罩": [0, 0]}
    previous: dict[tuple[int, str], str] = {}
    skip_errors = []
    for name in names:
        frame = frames[name]
        global frame_name
        frame_name = name
        centres = bench.squad_card_centres(frame, pitch=PITCH)
        card_top = float(bench.squad_card_rect(0, PITCH).y)
        trace: list = []
        began = time.perf_counter()
        values = R.read_squad_identities(frame, faces, pitch=PITCH, assets=assets,
                                         trace=trace)
        online_s = time.perf_counter() - began
        # **参照那一条在线上那一条之后跑、而且共用同一把尺子**（用户裁定 2）：
        # 共用是为了不把内存撑成两倍（见上面那段）；线上先跑是为了让"懒建那一笔"
        # 落在**第一次出现红罩卡那一帧的线上时间里** —— 参照先跑就会替它把这笔付掉，
        # 那个凸起就看不见了。
        ref = reference(frame, faces, centres, assets.ruler(faces))
        online = []
        for index, value in enumerate(values):
            known = TRUTH.get((name, index))
            got = {"index": index, "name": name_of(value),
                   "verdict": "精确" if name_of(value) is not None else "未知",
                   "top": None if value.confidence is None else float(value.confidence)}
            if got["verdict"] == "精确" and known:
                got["wrong"] = got["name"] != known
            online.append(got)
            cards += 1
            row = {"frame": name, "index": index, "truth": known,
                   "red": ref[index]["red"], "ref": ref[index], "online": got,
                   "trace": trace[index] if index < len(trace) else None}
            if (ref[index]["verdict"], ref[index]["name"]) != (got["verdict"], got["name"]):
                diffs.append(row)
            # 跳过的格：沿用的判定与"全分辨率跑满"比过没有
            if row["trace"] is not None and row["trace"].get("how") == "跳过":
                if (ref[index]["verdict"], ref[index]["name"]) != (got["verdict"], got["name"]):
                    skip_errors.append(row)
        # ---- 红罩那一刀：整张滑 vs 分层（逐对核对）----
        for index, (centre, _) in enumerate(centres):
            if bench.redeploying(frame, centre, card_top=card_top) is not True:
                continue
            region = np.ascontiguousarray(frame.crop(redcap_region(centre, pitch=PITCH)))
            full = ref[index]["ranked"]          # 整张滑那一趟在 reference 里算过了
            full_s = ref[index]["seconds"]
            began = time.perf_counter()
            layered, walk = R._masked_layered_scores(region, ruler)
            layered_s = time.perf_counter() - began
            knife["full_s"].append(full_s)
            knife["layered_s"].append(layered_s)
            pair = {"frame": name, "index": index,
                    "full": verdict_of(full, TRUTH.get((name, index))),
                    "layered": verdict_of(layered, TRUTH.get((name, index))),
                    "layer": walk["fine"]["layer"], "tried": len(walk["fine"]["tried"]),
                    "full_s": round(full_s, 4), "layered_s": round(layered_s, 4)}
            knife["pairs"].append(pair)
            if (pair["full"]["verdict"], pair["full"]["name"]) != (
                    pair["layered"]["verdict"], pair["layered"]["name"]):
                knife["differ"].append(pair)
        # ---- 复用那条路自己的分线 + 逐位相同 ----
        for index, (centre, _) in enumerate(centres):
            red = bench.redeploying(frame, centre, card_top=card_top) is True
            rect = (redcap_region(centre, pitch=PITCH) if red
                    else identity_region(centre, pitch=PITCH))
            region = np.ascontiguousarray(frame.crop(rect))
            kind = "红罩" if red else "干净"
            digest = hashlib.blake2b(region.tobytes(), digest_size=10).hexdigest()
            key = (index, kind)
            if key in previous:
                identical_pixels[kind][1] += 1
                if previous[key] == digest:
                    identical_pixels[kind][0] += 1
            previous[key] = digest
            stored = assets.slot_of(index)
            known = TRUTH.get((name, index))
            if stored is None or known is None or known == "":
                continue
            box = ((float(R._margin().IDENTITY_DISC_BOX[0]), float(R._margin().IDENTITY_DISC_BOX[1]))
                   if red else (float(R.IDENTITY_BOX_X), float(R.IDENTITY_BOX_Y)))
            reference_name = ref[index]["name"]     # 全分辨率跑满那条路给的名字（None = 未知）
            reading = R._reuse_scores(region, faces, stored, box, red=red, ruler=ruler)
            for score, who, _scale, _p in reading:
                clusters.append((
                    1 if who == known else 0,
                    1 if (reference_name is not None and who == reference_name) else 0,
                    score, name, index, who, known, reference_name,
                ))
        per_frame.append({
            "frame": name, "cards": len(centres),
            "online_seconds": round(online_s, 4),
            "reference_seconds": round(sum(r["seconds"] for r in ref), 4),
            "reference": ref,
            "red_slots": sum(1 for r in ref if r["red"]),
            "skips": sum(1 for t in trace if t.get("how") == "跳过"),
            "geometry": {
                str(index): [len(slot),
                             round(min(g.scale for g in slot.values()), 3),
                             round(max(g.scale for g in slot.values()), 3)]
                for index in range(12)
                if (slot := assets.slot_of(index)) is not None
            },
            "slots": [{"index": t.get("index"), "red": t.get("red"),
                       "how": t.get("how"), "layer": t.get("layer"),
                       "top": t.get("top"), "runner": t.get("runner"),
                       "runner_at_geometry": t.get("runner_at_geometry"),
                       "reused": t.get("reused"),
                       "upgrade": bool(t.get("upgrade")),
                       "cost": t.get("cost"),
                       "d_jitter": t.get("geometry", {}).get("d_offset")}
                      for t in trace],
        })
        print("  %s 卡 %2d · 线上 %6.3f s · 跑满 %6.3f s · 复用 %d · 升级 %d · 首帧 %d · 跳过 %d"
              % (name, len(centres), online_s, per_frame[-1]["reference_seconds"],
                 sum(1 for t in trace if t.get("how") == "复用"),
                 sum(1 for t in trace if t.get("upgrade")),
                 sum(1 for t in trace if t.get("how") in ("首帧", "红罩首帧")),
                 per_frame[-1]["skips"]))
    summary = assets.summary()
    lost = [d for d in diffs if d["ref"]["verdict"] == "精确" and d["online"]["verdict"] != "精确"]
    gained = [d for d in diffs if d["ref"]["verdict"] != "精确" and d["online"]["verdict"] == "精确"]
    renamed = [d for d in diffs
               if d["ref"]["verdict"] == "精确" and d["online"]["verdict"] == "精确"
               and d["ref"]["name"] != d["online"]["name"]]
    wrong = [d for d in diffs if d["online"].get("wrong") is True]
    print("=== 网 ===")
    print("  卡 %d · 丢 %d · 得 %d · 改名 %d · 新错 %d"
          % (cards, len(lost), len(gained), len(renamed), len(wrong)))
    print("  资产：%s" % summary)

    # ---- 甲2：懒建之后，那把尺子的凸起落在哪一帧（用户裁定 2）----
    red_frames = [row["frame"] for row in per_frame if row["red_slots"]]
    first_red = red_frames[0] if red_frames else None
    first_red_row = next((row for row in per_frame if row["frame"] == first_red), None)
    after_first = [row["online_seconds"] for row in per_frame
                   if first_red is not None and row["frame"] > first_red]
    ruler_spike = {
        "first_red_frame": first_red,
        "first_red_online_s": (None if first_red_row is None
                               else first_red_row["online_seconds"]),
        "after_first_median_s": (round(statistics.median(after_first), 4)
                                 if after_first else None),
        "spike_s": (None if (not after_first or first_red_row is None) else round(
            first_red_row["online_seconds"] - statistics.median(after_first), 4)),
        "built_cells": ruler.built_cells,
        "red_frames": red_frames,
    }
    print("=== 甲2 尺子懒建：凸起落在哪一帧 ===")
    print("  第一次出现红罩卡那一帧 %s · 线上 %.4f s · 它之后各帧中位 %.4f s · 凸起 %s s"
          % (ruler_spike["first_red_frame"], ruler_spike["first_red_online_s"],
             ruler_spike["after_first_median_s"], ruler_spike["spike_s"]))
    print("  懒建建了多少：%s" % ruler_spike["built_cells"])

    # ---- 升级自愈（用户裁定 5）----
    slot_upgrades: dict[int, list] = {}
    for row in assets.upgrades:
        slot_upgrades.setdefault(int(row["index"]), []).append(row)
    upgrade_ledger = {
        "total": len(assets.upgrades),
        "slots": len(slot_upgrades),
        "per_slot": {str(key): len(value) for key, value in sorted(slot_upgrades.items())},
        "repeats": {str(key): [row["after"] for row in value]
                    for key, value in sorted(slot_upgrades.items()) if len(value) > 1},
        "max_per_slot": max((len(value) for value in slot_upgrades.values()), default=0),
    }
    print("=== 升级自愈（用户裁定 5）===")
    print("  升级 %d 次 · 涉及 %d 个卡位 · 同一卡位最多 %d 次 · 重复升级的卡位 %s"
          % (upgrade_ledger["total"], upgrade_ledger["slots"],
             upgrade_ledger["max_per_slot"], upgrade_ledger["repeats"]))

    # ---- 甲3：每格成本按层拆解（线上那条路自己的足迹）----
    cost_rows = [t["cost"] for row in per_frame for t in row["slots"]
                 if t and t.get("cost")]
    layer_cost = {}
    for key in sorted({k for row in cost_rows for k in row}):
        vals = [row[key] for row in cost_rows if key in row]
        layer_cost[key] = {"n": len(vals),
                           "median_s": round(statistics.median(vals), 5),
                           "mean_s": round(statistics.fmean(vals), 5)}
    print("=== 甲3 每格成本按层拆解（线上那条路，%d 格有足迹） ===" % len(cost_rows))
    for key, row in layer_cost.items():
        print("  %-10s n=%d 中位 %.5f s 均值 %.5f s"
              % (key, row["n"], row["median_s"], row["mean_s"]))

    # ---- 红罩那一刀 ----
    knife["median_full_s"] = round(statistics.median(knife["full_s"]), 4)
    knife["median_layered_s"] = round(statistics.median(knife["layered_s"]), 4)
    knife["speedup"] = round(knife["median_full_s"] / max(knife["median_layered_s"], 1e-9), 2)
    print("=== 红罩那一刀（%d 张红罩卡） ===" % len(knife["pairs"]))
    print("  整张滑 中位 %.4f s · 分层 中位 %.4f s · %.2f× · 判定/名字不同的 %d 对"
          % (knife["median_full_s"], knife["median_layered_s"], knife["speedup"],
             len(knife["differ"])))

    # ---- 复用路两簇：**两种口径都报** ----
    def clusters_of(which: int) -> dict:
        truth = [s for _t, r, s, *_ in clusters if (r if which else _t) == 1]
        false = [s for _t, r, s, *_ in clusters if (r if which else _t) == 0]
        return {
            "true_n": len(truth), "false_n": len(false),
            "true_min": min(truth) if truth else None,
            "true_median": statistics.median(truth) if truth else None,
            "false_max": max(false) if false else None,
            "false_median": statistics.median(false) if false else None,
            "gap": (min(truth) - max(false)) if truth and false else None,
            "in_gap": bool(truth and false and max(false) < R.ROSTER_REUSE_MATCH_SCORE
                           < min(truth)),
            "worst_true": sorted(
                ({"frame": f, "index": i, "who": w, "score": s, "reference": ref_name}
                 for _t, r, s, f, i, w, _k, ref_name in clusters
                 if (r if which else _t) == 1), key=lambda row: row["score"])[:8],
            "worst_false": sorted(
                ({"frame": f, "index": i, "who": w, "score": s, "reference": ref_name}
                 for _t, r, s, f, i, w, _k, ref_name in clusters
                 if (r if which else _t) == 0), key=lambda row: -row["score"])[:8],
        }

    truth_side = clusters_of(0)
    reference_side = clusters_of(1)
    clusters_out = {
        "pairs": len(clusters),
        "line": R.ROSTER_REUSE_MATCH_SCORE,
        "by_truth": truth_side,
        "by_reference": reference_side,
    }
    print("=== 复用那条路自己的分线（线 %.2f） ===" % clusters_out["line"])
    for label, side in (("真值口径", truth_side), ("参照口径", reference_side)):
        print("  %s：真 n=%d 最低 %.6f 中位 %.6f · 非 n=%d 最高 %.6f · 空档 %s · 线坐进空档 %s"
              % (label, side["true_n"], side["true_min"], side["true_median"],
                 side["false_n"], side["false_max"],
                 "%.6f" % side["gap"] if side["gap"] is not None else "—", side["in_gap"]))

    # ---- 画面没变就跳过 ----
    skips_out = {
        "identical": {kind: {"same": hit, "pairs": total,
                             "ratio": round(hit / total, 4) if total else None}
                      for kind, (hit, total) in identical_pixels.items()},
        "skipped_total": summary["counts"]["跳过"],
        "per_frame": [{"frame": row["frame"], "cards": row["cards"], "skips": row["skips"]}
                      for row in per_frame],
        "skip_errors": skip_errors,
    }
    print("=== 画面没变就跳过 ===")
    for kind, row in skips_out["identical"].items():
        print("  逐位相同 %s：%d/%d = %s" % (kind, row["same"], row["pairs"], row["ratio"]))
    print("  总共跳过 %d 格 · 每帧 %s" % (skips_out["skipped_total"],
          [row["skips"] for row in skips_out["per_frame"]]))
    print("  沿用判定出错：%d" % len(skip_errors))

    # ---- 粗层接受（全语料） ----
    print("=== 粗层接受（全语料，干净卡） ===")
    corpus = corpus_phase()
    print("  卡 %d · 锚点层接受 %d · 全分辨率 %d · 与跑满不同 %d"
          % (corpus["cards"], corpus["layers"].get("锚点层", 0),
             corpus["layers"].get("全分辨率", 0), corpus["differ"]))

    online_times = [row["online_seconds"] for row in per_frame]
    ref_times = [row["reference_seconds"] for row in per_frame]
    payload = {
        "note": "第六单：红罩那一刀（锚点收窄）· 资产改成每卡位×每条 · 复用路自己的分线 ·"
                " 画面没变就整格跳过。对「全分辨率跑满」那条路的逐张对比 + 代价。"
                "素材是同一局那 60 帧（data/precheck/cd，**gitignored**）。",
        "criterion": {"score_line": R.ROSTER_SOLID_MATCH_SCORE,
                      "coarse_line": R.ROSTER_COARSE_ACCEPT_SCORE,
                      "reuse_line": R.ROSTER_REUSE_MATCH_SCORE,
                      "factor": R.ROSTER_AMBIGUITY_FACTOR, "pad": R.ROSTER_AMBIGUITY_PAD},
        "geometry": {"jitter": R._GEOMETRY_JITTER,
                     "interval": [R.IDENTITY_SCALES[0], R.IDENTITY_SCALES[-1],
                                  len(R.IDENTITY_SCALES)],
                     "per": "每卡位 × 每条库"},
        "sequence": {"source": "data/precheck/cd", "frames": len(names),
                     "fixture_frames_identical": identical, "layout": layout,
                     "material": material},
        "net": {"cards": cards, "lost": len(lost), "gained": len(gained),
                "renamed": len(renamed), "wrong": len(wrong),
                "diff_rows": diffs[:40], "lost_rows": lost, "wrong_rows": wrong,
                "renamed_rows": renamed},
        "assets": summary,
        "knife": knife,
        "reuse_clusters": clusters_out,
        "skips": skips_out,
        "corpus": corpus,
        "times": {
            "ruler_build_s": round(ruler_build, 4),
            "first_frame_s": online_times[0],
            "first_frame_reference_s": ref_times[0],
            "other_frames_median_s": round(statistics.median(online_times[1:]), 4),
            "other_frames_mean_s": round(statistics.fmean(online_times[1:]), 4),
            "other_frames_max_s": max(online_times[1:]),
            "other_frames_argmax": per_frame[1:][int(np.argmax(online_times[1:]))]["frame"],
            "online_total_s": round(sum(online_times), 4),
            "reference_total_s": round(sum(ref_times), 4),
            "online_median_s": round(statistics.median(online_times), 4),
            "reference_median_per_card_s": round(
                statistics.median([r["seconds"] for row in per_frame
                                   for r in row["reference"]]), 4),
            "reference_median_clean_s": round(statistics.median(
                [r["seconds"] for row in per_frame for r in row["reference"]
                 if not r["red"]]), 4),
            "reference_median_red_s": round(statistics.median(
                [r["seconds"] for row in per_frame for r in row["reference"]
                 if r["red"]]), 4),
            "red_frames": sum(1 for row in per_frame if row["red_slots"]),
            "rebuilds": assets.rebuilds,
        },
        "ruler_spike": ruler_spike,
        "layer_cost": layer_cost,
        "full_build_s": round(full_build, 4),
        "upgrade_ledger": upgrade_ledger,
        "coarse": {"cards": corpus["cards"], "layers": corpus["layers"],
                   "differ": corpus["differ"]},
        "frames": per_frame,
        "seconds": round(time.perf_counter() - started, 3),
    }
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print("落盘 %s（%.1f s）" % (OUT, payload["seconds"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
