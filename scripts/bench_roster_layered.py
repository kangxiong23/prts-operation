# -*- coding: utf-8 -*-
r"""多层细化：一层比一层细 —— 图先缩小再放大回来 · 档距先粗后细 · 位置先粗后细。

    .venv\Scripts\python.exe scripts\bench_roster_layered.py            # 全量 + 时间
    .venv\Scripts\python.exe scripts\bench_roster_layered.py --smoke    # 只跑一小撮自检

## 这一单问什么

点名字那条链今天一帧约 15 s，上线用不了。开销规律（上一单实测）：

    一次调用的代价 ~ 搜索位置数 x 模板像素数        （缩图是平方级的便宜）

所以要压墙钟，就得**让精算那一层只看很少的位置、很少的档**，而把「该看哪儿」交给
前面几层粗的去找。这一份就是**设计并量**这件事：层的划分（几层 · 每层图缩到多大 ·
每层试哪些档 · 每层位置搜多宽），以及它买到了多少速度、丢了多少选档。

**只量，不改产品**。报出的分**只**来自最后一层（全分辨率 100 那一次 matchTemplate），
前面几层只用来挑档 / 挑位置。

## 真值从哪来

tests/帧样本/先粗后细/曲线.json —— 4030 条**全分辨率**取值域曲线（26 档 0.95…1.20），
上一单落下的缓存收进版本库。**本单不重跑基线**：真值读它，节拍的今天那一侧读
读数.json 的 cost_measured。

## 一个字节都不改

线 · 尺度区间 · 取区形状 · roster_scores · 库 · 模型 全不动。
"""

from __future__ import annotations

import argparse
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

import bench_roster_coarse as base  # noqa: E402

SAMPLES = base.SAMPLES
OUT_DIR = SAMPLES / "先粗后细"
CURVES = OUT_DIR / "曲线.json"
READINGS = OUT_DIR / "读数.json"
OUT = OUT_DIR / "细化.json"
LAYER_CACHE = REPO / "data" / "probe" / "layered_cache.json"
NARROW = base.NARROW
"""今天那条链的档 —— 1.05 … 1.15（11 档）。多层细化只在它上面找。"""
GRID_STEP = 0.01


class Layer:
    """一层。pixels = 这一层把图缩到多大（100 = 全分辨率）。

    scale_half：这一层试的档 = 上一层挑中的那一档 +- scale_half 档（None = 整个区间）。
    pos_half：这一层的位置窗 = 上一层在同一档上给的位置 +- pos_half 全分辨率像素
              （None = 整张立绘）。**位置是按档各自带的**，不是全层一个。
    """

    def __init__(self, pixels: int, scale_half=None, pos_half=None) -> None:
        self.pixels = int(pixels)
        self.scale_half = scale_half
        self.pos_half = pos_half

    def as_dict(self) -> dict:
        return {"pixels": self.pixels, "scale_half": self.scale_half,
                "pos_half": self.pos_half,
                "factor": round(self.pixels / 100.0, 3),
                "covers_all_scales": self.scale_half is None}


def _design(label: str, layers) -> dict:
    return {"label": label, "layers": tuple(layers)}


DESIGNS = (
    _design("保真 · 50→100 · 全 11 档 · 位±12", (Layer(50), Layer(100, None, 12))),
    _design("保真 · 25→100 · 全 11 档 · 位±16", (Layer(25), Layer(100, None, 16))),
    _design("两层 · 15→100 · 档±1 · 位±16", (Layer(15), Layer(100, 1, 16))),
    _design("两层 · 12→100 · 档±1 · 位±12", (Layer(12), Layer(100, 1, 12))),
    _design("两层 · 15→100 · 档±2 · 位±24", (Layer(15), Layer(100, 2, 24))),
    _design("三层 · 12→50→100 · 档±3/±1 · 位±12/±6",
            (Layer(12), Layer(50, 3, 12), Layer(100, 1, 6))),
    _design("三层 · 12→40→100 · 档±4/±1 · 位±10/±6",
            (Layer(12), Layer(40, 4, 10), Layer(100, 1, 6))),
    _design("三层 · 15→50→100 · 档±4/±2 · 位±16/±8",
            (Layer(15), Layer(50, 4, 16), Layer(100, 2, 8))),
    _design("三层 · 25→50→100 · 档±3/±1 · 位±12/±6",
            (Layer(25), Layer(50, 3, 12), Layer(100, 1, 6))),
    _design("四层 · 12→25→50→100 · 档±3/±2/±1 · 位±12/±8/±6",
            (Layer(12), Layer(25, 3, 12), Layer(50, 2, 8), Layer(100, 1, 6))),
)
BY_LABEL = {d["label"]: d for d in DESIGNS}


def _scale_window(interval, center: float, half) -> list:
    if half is None:
        return list(interval)
    span = half * GRID_STEP
    return [s for s in interval if center - span - 1e-9 <= s <= center + span + 1e-9]


def _anchor(positions: dict, scale: float):
    """上一层的**同一档**给的位置；那一档缺了就用最近的档。"""
    for key, value in positions.items():
        if abs(key - scale) < 1e-9:
            return value
    if not positions:
        return None
    return positions[min(positions, key=lambda key: abs(key - scale))]


def run_layers(region: np.ndarray, art: np.ndarray, layers, counter: list) -> dict:
    """**真的跑**一条多层搜索。返回最后一层报出的分 / 档 / 位置，以及各层的足迹。"""
    trace = []
    positions: dict = {}
    cand = list(NARROW)
    picked = None
    score = -2.0
    loc = (0.0, 0.0)
    for index, layer in enumerate(layers):
        factor = layer.pixels / float(region.shape[0])
        if layer.pixels == 100:
            small_query = region
            small_art = art
        else:
            small_query = cv2.resize(region, (layer.pixels, layer.pixels),
                                     interpolation=cv2.INTER_AREA)
            small_art = cv2.resize(
                art,
                (max(1, int(round(art.shape[1] * factor))),
                 max(1, int(round(art.shape[0] * factor)))),
                interpolation=cv2.INTER_AREA,
            )
        rows: dict = {}
        area = 0
        for scale in cand:
            height = int(round(small_query.shape[0] * scale))
            width = int(round(small_query.shape[1] * scale))
            if height > small_art.shape[0] or width > small_art.shape[1]:
                continue
            template = cv2.resize(small_query, (width, height),
                                  interpolation=cv2.INTER_LINEAR)
            if layer.pos_half is None or not positions:
                block = small_art
                x0 = y0 = 0
            else:
                anchor = _anchor(positions, scale)
                half = layer.pos_half * factor
                x0 = max(0, int(round(anchor[0] * factor - half)))
                y0 = max(0, int(round(anchor[1] * factor - half)))
                x1 = min(int(small_art.shape[1]),
                         int(round(anchor[0] * factor + width + half)))
                y1 = min(int(small_art.shape[0]),
                         int(round(anchor[1] * factor + height + half)))
                block = np.ascontiguousarray(small_art[y0:y1, x0:x1])
            counter[0] += 1
            area += int(block.shape[0]) * int(block.shape[1])
            result = cv2.matchTemplate(block, template, cv2.TM_CCOEFF_NORMED)
            _lo, hi, _loloc, hiloc = cv2.minMaxLoc(result)
            rows[scale] = (float(hi), (hiloc[0] + x0) / factor,
                           (hiloc[1] + y0) / factor)
        best_scale = max(rows, key=lambda key: (rows[key][0], -key))
        picked = best_scale
        score, x, y = rows[best_scale]
        loc = (x, y)
        trace.append({"layer": index, "pixels": layer.pixels,
                      "scale_half": layer.scale_half, "pos_half": layer.pos_half,
                      "scales_tried": len(rows), "position_window": layer.pos_half,
                      "picked_scale": best_scale, "picked_score": score,
                      "area_px": area})
        positions = {key: (value[1], value[2]) for key, value in rows.items()}
        if index + 1 < len(layers):
            nxt = layers[index + 1]
            cand = _scale_window(NARROW, best_scale, nxt.scale_half)
    return {"picked": picked, "score": score, "loc": loc, "trace": trace}


def _truth_cards(blob: dict) -> dict:
    """按 (批次|帧|卡位) 归拢曲线；每条 = [(尺度, 分数, x, y), ...]。"""
    out: dict = {}
    for key in blob["keys"]:
        batch, frame, index, entry = key.split("|")
        out.setdefault(batch + "|" + frame + "|" + index, {})[entry] = blob["curves"][key]
    return out


def _narrow_rows(rows, interval=NARROW):
    return [(float(s), float(sc), int(x), int(y)) for s, sc, x, y in rows
            if any(abs(float(s) - t) < 1e-9 for t in interval)]


def _run_design(design: dict, corpus: list, libraries: dict, cards: dict,
                limit=None, dump=False) -> dict:
    """全语料逐对跑一遍。真值 = 曲线.json 里的全分辨率 argmax。"""
    per_pair = []
    verdict_rows = []
    truth_map = base._truth_map()
    calls = 0
    area = 0
    checked = 0
    for record in corpus:
        faces = libraries[record["library"]]
        key = "%s|%s|%d" % (record["batch"], record["frame"], record["index"])
        here = cards[key]
        by_entry = {}
        today_by_entry = {}
        for name, art in faces:
            rows = _narrow_rows(here[name])
            if not rows:
                continue
            true_scale = max(rows, key=lambda row: (row[1], -row[0]))[0]
            true_score = max(row[1] for row in rows)
            counter = [0]
            got = run_layers(record["region"], art, design["layers"], counter)
            calls += counter[0]
            trace = got["trace"]
            entry = {
                "batch": record["batch"], "frame": record["frame"],
                "index": record["index"], "entry": name,
                "true_scale": true_scale, "true_score": true_score,
                "picked": got["picked"], "reported": got["score"],
                "scale_steps": int(round(abs(got["picked"] - true_scale) / GRID_STEP)),
                "l0_picked": trace[0]["picked_scale"],
                "l0_steps": int(round(abs(trace[0]["picked_scale"] - true_scale) / GRID_STEP)),
                "calls": counter[0],
                "trace": trace if dump else [],
            }
            per_pair.append(entry)
            by_entry[name] = {"score": got["score"], "scale": got["picked"],
                              "loc": got["loc"]}
            today_by_entry[name] = {
                "score": true_score, "scale": true_scale,
                "loc": (max(rows, key=lambda row: row[1])[2],
                        max(rows, key=lambda row: row[1])[3])}
            checked += 1
            if limit is not None and checked >= limit:
                break
        group = record["frame"][1] if record["batch"] == "立绘两组" else None
        truth = truth_map.get((record["frame"], record["index"]))
        before = base._verdict(today_by_entry, NARROW, truth, group)
        after = base._verdict(by_entry, NARROW, truth, group)
        verdict_rows.append({
            "batch": record["batch"], "frame": record["frame"],
            "index": record["index"], "redeploying": record["redeploying"],
            "truth": truth, "group": group,
            "before": before, "after": after,
            "scale_moved": before["scale"] != after["scale"],
            "name_moved": before["name"] != after["name"],
            "changed": (before["verdict"] != after["verdict"]
                        or before["name"] != after["name"]),
        })
        if limit is not None and checked >= limit:
            break
    exact = [r for r in per_pair if r["scale_steps"] == 0]
    winners = {}
    for row in per_pair:
        card = (row["batch"], row["frame"], row["index"])
        if card not in winners or row["true_score"] > winners[card]["true_score"]:
            winners[card] = row
    winner_rows = list(winners.values())
    gained = [r for r in verdict_rows if r["before"]["verdict"] == "未知"
              and r["after"]["verdict"] == "精确"]
    lost = [r for r in verdict_rows if r["before"]["verdict"] == "精确"
            and r["after"]["verdict"] == "未知"]
    wrong_new = [r for r in gained if r["after"]["wrong"] is True]
    return {
        "label": design["label"],
        "layers": [layer.as_dict() for layer in design["layers"]],
        "pairs": len(per_pair),
        "calls_total": calls,
        "calls_per_pair": round(calls / max(1, len(per_pair)), 3),
        "scale_exact": len(exact),
        "scale_within_1": sum(1 for r in per_pair if r["scale_steps"] <= 1),
        "scale_within_2": sum(1 for r in per_pair if r["scale_steps"] <= 2),
        "l0_exact": sum(1 for r in per_pair if r["l0_steps"] == 0),
        "l0_within_1": sum(1 for r in per_pair if r["l0_steps"] <= 1),
        "winner_pairs": len(winner_rows),
        "winner_scale_exact": sum(1 for r in winner_rows if r["scale_steps"] == 0),
        "winner_scale_within_1": sum(1 for r in winner_rows if r["scale_steps"] <= 1),
        "winner_scale_within_2": sum(1 for r in winner_rows if r["scale_steps"] <= 2),
        "cards": len(verdict_rows),
        "before": {"exact": sum(1 for r in verdict_rows
                                if r["before"]["verdict"] == "精确"),
                   "unknown": sum(1 for r in verdict_rows
                                  if r["before"]["verdict"] == "未知")},
        "after": {"exact": sum(1 for r in verdict_rows
                               if r["after"]["verdict"] == "精确"),
                  "unknown": sum(1 for r in verdict_rows
                                 if r["after"]["verdict"] == "未知")},
        "gained": gained[:20], "gained_n": len(gained),
        "lost": lost[:20], "lost_n": len(lost),
        "wrong_new": wrong_new[:20], "wrong_new_n": len(wrong_new),
        "wrong_before_n": sum(1 for r in verdict_rows
                              if r["before"]["wrong"] is True),
        "wrong_after_n": sum(1 for r in verdict_rows
                             if r["after"]["wrong"] is True),
        "name_moved_n": sum(1 for r in verdict_rows if r["name_moved"]),
        "scale_moved_n": sum(1 for r in verdict_rows if r["scale_moved"]),
        "scale_steps_histogram": _histogram(per_pair),
        "per_pair": per_pair if dump else [],
        "lost_rows": [{"frame": r["frame"], "index": r["index"],
                       "before": r["before"], "after": r["after"]} for r in lost[:6]],
    }


def _histogram(per_pair) -> dict:
    out: dict = {}
    for row in per_pair:
        key = str(row["scale_steps"])
        out[key] = out.get(key, 0) + 1
    return out


def _measure(designs, corpus: list, libraries: dict, picks, rounds=3) -> dict:
    """真跑：固定那几张卡，每种走法整个执行一遍，数真次数、量真墙钟（取中位）。"""
    sample = [corpus[i] for i in picks]
    out = {}
    for design in designs:
        counter = [0]
        times = []
        for _ in range(rounds):
            started = time.perf_counter()
            counter[0] = 0
            for record in sample:
                for _name, art in libraries[record["library"]]:
                    run_layers(record["region"], art, design["layers"], counter)
            times.append(time.perf_counter() - started)
        out[design["label"]] = {
            "cards": len(sample),
            "entries": sum(len(libraries[r["library"]]) for r in sample),
            "match_template_calls": counter[0],
            "median_s": round(statistics.median(times), 4),
            "rounds_s": [round(value, 4) for value in times],
        }
    return out


def _measure_frame(design, corpus: list, libraries: dict, batch: str,
                   frame: str, per_call_s: float, rounds=3) -> dict:
    """一帧的账：这一帧**所有卡位 x 所有库条目**整个执行一遍。"""
    sample = [r for r in corpus if r["batch"] == batch and r["frame"] == frame]
    counter = [0]
    times = []
    for _ in range(rounds):
        started = time.perf_counter()
        counter[0] = 0
        for record in sample:
            for _name, art in libraries[record["library"]]:
                run_layers(record["region"], art, design["layers"], counter)
        times.append(time.perf_counter() - started)
    entries = sum(len(libraries[r["library"]]) for r in sample)
    return {"batch": batch, "frame": frame, "cards": len(sample), "entries": entries,
            "match_template_calls": [counter[0]],
            "median_s": round(statistics.median(times), 4),
            "rounds_s": [round(value, 4) for value in times],
            "today_calls": entries * 11,
            "today_s": round(entries * 11 * per_call_s, 4)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--designs", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-measure", action="store_true")
    parser.add_argument("--dump-pairs", action="store_true")
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--out", default="")
    parser.add_argument("--frame-only", action="store_true")
    parser.add_argument("--designs-from", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    blob = json.loads(CURVES.read_text(encoding="utf-8"))
    readings = json.loads(READINGS.read_text(encoding="utf-8"))
    cards = _truth_cards(blob)
    print("=== 真值（复用落盘）===")
    print("  曲线 %d 条 / %d 张卡   来源 tests/帧样本/先粗后细/曲线.json" % (
        len(blob["keys"]), len(cards)))

    corpus, libraries = base._corpus()
    print("  语料 %d 张卡（红罩 %d）" % (
        len(corpus), sum(1 for r in corpus if r["redeploying"] is True)))

    chosen = DESIGNS
    if args.index >= 0:
        chosen = (DESIGNS[args.index],)
    if args.designs:
        wanted = [name.strip() for name in args.designs.split(",") if name.strip()]
        chosen = tuple(BY_LABEL[name] for name in wanted)
    if args.smoke:
        chosen = chosen[:1]
        args.limit = args.limit or 60

    results = {}
    backup = {}
    if args.designs_from:
        backup = json.loads(Path(args.designs_from).read_text(encoding="utf-8"))["designs"]
    for design in chosen:
        if args.frame_only:
            continue
        if backup:
            results[design["label"]] = backup[design["label"]]
            got = results[design["label"]]
            print("  [%s] 复用已算好的逐对结果（%s）" % (design["label"], args.designs_from))
            continue
        began = time.perf_counter()
        got = _run_design(design, corpus, libraries, cards,
                          limit=args.limit or None, dump=args.dump_pairs)
        got["seconds"] = round(time.perf_counter() - began, 3)
        results[design["label"]] = got
        print("  [%s] pairs=%d 选档准=%d/%d 卡判定 %d/%d 次数/对=%.2f  %.1fs" % (
            design["label"], got["pairs"], got["scale_exact"], got["pairs"],
            got["after"]["exact"], got["cards"], got["calls_per_pair"],
            got["seconds"]))
        if LAYER_CACHE.is_file():
            LAYER_CACHE.unlink()
        LAYER_CACHE.write_text(json.dumps({"designs": results}, ensure_ascii=False),
                               encoding="utf-8")

    measured = readings["cost_measured"]["甲的今天：全分辨率全档"]
    per_call_s = measured["median_s"] / measured["match_template_calls"]
    reuse = {
        "source_file": "tests/帧样本/先粗后细/读数.json（本单**复用**，没有重跑）",
        "source_key": "cost_measured 的 甲的今天：全分辨率全档",
        "config_calls": measured["match_template_calls"],
        "config_median_s": measured["median_s"],
        "config_note": "那一趟把同一件事做了两遍（粗那一层的 pixels=100 就是精那一层），"
                       "所以次数与墙钟都要折半才是今天那条链",
        "chain_calls": measured["match_template_calls"] // 2,
        "chain_median_s": round(measured["median_s"] / 2, 4),
        "chain_calls_per_pair": 11,
        "per_call_s": round(per_call_s, 6),
    }
    measure = {}
    frame = {}
    if not args.smoke and not args.no_measure:
        picks = readings["measure_sample"]["picks"]
        print("=== 墙钟（真跑，取中位）===")
        if not args.frame_only:
            measure = _measure(chosen, corpus, libraries, picks, rounds=3)
            for label, got in measure.items():
                print("  %-42s %5d 次  %8.3f s" % (
                    label, got["match_template_calls"], got["median_s"]))
        for design in chosen:
            got = _measure_frame(design, corpus, libraries, "备战席", "shot01",
                                 per_call_s, rounds=3)
            frame[design["label"]] = got
            print("  一帧 %s：%d 卡位 x %d 条 -> %d 次 / %.3f s（今天同一帧 %d 次 / %.3f s）"
                  % (got["frame"], got["cards"], got["entries"] // max(1, got["cards"]),
                     got["match_template_calls"][0], got["median_s"],
                     got["today_calls"], got["today_s"]))
    payload = {
        "note": "多层细化：层表 + 选档一致率 + 判定账 + 速度。报出的分只来自最后一层（全分辨率）。只量不改。",
        "truth": {"file": "tests/帧样本/先粗后细/曲线.json",
                  "pairs": len(blob["keys"]), "cards": len(cards)},
        "reuse_today": reuse,
        "designs": results,
        "measure": measure,
        "measure_frame": frame,
    }
    if not args.smoke:
        target = Path(args.out) if args.out else OUT
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print("总耗时 %.1f s" % (time.perf_counter() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
