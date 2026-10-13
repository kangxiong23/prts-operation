# -*- coding: utf-8 -*-
r"""先粗后细能不能同时买到「尺子铺得更宽」与「更快」—— **只量，不改**。

    .venv\Scripts\python.exe scripts\bench_roster_coarse.py

## 这一单问什么

点名那条链的判据，内层是「一块身份区 × 一条库 × 一档尺度 → 一张相关图」：
`cv2.resize(query, s)` 然后 `cv2.matchTemplate(art, tpl, TM_CCOEFF_NORMED)`，取 max。
尺度档数 × 库条数 × 卡位数 就是点名那一笔账。现在尺度只扫 1.05–1.15（11 档），
而下界那一刀截掉了峰（组2 三位峰在 1.01 / 1.02 / 1.02）。

用户点的思路：**先用少量像素粗挑最好的倍率，再精算**（起点 225 = 15x15 个像素）。
粗可以在三个地方做 —— **档（尺度）· 位置 · 像素**。

**硬约束：报出来的分必须仍然来自全分辨率那一次精算。** 粗探只用来挑档 / 挑位置，
不许改报出的分数。所以这一份量的不是「换一条链」，而是「换一条**找**那条链该在哪里
算的搜索路径」—— 找到的档与位置对不对，就是它成不成。

## 一个字节都不改

线（ROSTER_SOLID_MATCH_SCORE / ROSTER_AMBIGUITY_MARGIN）· 尺度区间 · 取区形状 ·
那条链（roster_scores）· 库 · 模型 全不动。这里只把那条链**内层那一步**按尺度
一档一档重跑，取值域比线上宽，用来离线重算任何区间。重跑在线上那 11 档上的最大值
必须与 roster_scores 给的分**逐位相等**（crosscheck），对不上就报错。

## 测的是什么取区

**干净卡那块框**（roster.identity_region 100x100，roster_scores）—— 那就是这条链
本来的样子。红罩卡那一格换的是**统计量**（margin.MaskedRuler 的掩码相关），但
**尺度网格与 resize 那一步是共用的**，而它更贵，所以这一份量出来的收益对它是**下界**。
红罩那条路自己的曲线没量（见汇报 ③）。

## 五张表

1. **粗探准不准** —— 粗探挑的档 / 位置 与 全分辨率峰 的差（差几档各多少对）；
2. **报出的分逐位相同** —— 粗探挑档 + 精算 vs 全档全像素那条链，逐位比（全语料 + 真跑样本）；
3. **代价** —— 匹配次数 · 立绘像素量 · 墙钟（真跑，取中位），外加「每多扫一档多花多少」；
4. **收益与风险** —— 区间铺到覆盖 1.01 及以下之后，逐张报未知→精确 · 新出现的错名字；
5. **最少要多少个像素** —— 像素数 → 选档正确率 → 代价，一条曲线。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.engine.device.adb import SQUAD_CARD_PITCH  # noqa: E402
from ark.engine.vision import bench  # noqa: E402
from ark.engine.vision.frame import frame_from_raw  # noqa: E402
from ark.engine.vision.roster import (  # noqa: E402
    IDENTITY_SCALES,
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
    identity_region,
    roster_scores,
)
from ark.offline.assets_gen.cardface import load_cardfaces  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402

SAMPLES = REPO / "tests" / "帧样本"
OUT_DIR = SAMPLES / "先粗后细"
OUT = OUT_DIR / "读数.json"
PITCH = SQUAD_CARD_PITCH
SHARED_LIBRARY = SAMPLES / "点名" / "编队页.json"

NARROW = tuple(IDENTITY_SCALES)
"""今天的区间：1.05 … 1.15（11 档）。"""
WIDE = tuple(round(0.95 + index * 0.01, 2) for index in range(26))
"""宽区间：**只把下界往下铺**到 0.95（上界顺带铺到 1.20）—— 0.95 … 1.20，26 档。"""
CACHE = WIDE
"""取值域 = 宽区间。任何区间都从这一份离线重算。"""
BATCHES = ("备战席", "红罩", "余量", "立绘两组")
SWEEP_PIXELS = (8, 10, 15, 20, 25, 30, 40, 50, 75, 100)
"""第 5 张表：从少到多各试一档。100 = 全分辨率（就是今天那条链）。"""


class Config:
    """一种「先粗后细」的走法：粗在哪几处粗 · 精算留多宽。"""

    def __init__(self, label, mode, pixels, step, window, stride, pos_half):
        self.label = label
        self.mode = mode            # 窄 / 宽
        self.pixels = pixels        # 粗探那一层的像素边长（100 = 全分辨率）
        self.step = step            # 粗档网格的步长（None = 就用区间那 11/26 档）
        self.window = window        # 精算的档窗半宽（None = 整个区间；0 = 只精算挑中那一档）
        self.stride = stride        # 粗探的位置步长（1 = 逐像素）
        self.pos_half = pos_half    # 精算的位置窗半宽 px（None = 整张立绘）

    @property
    def interval(self):
        return NARROW if self.mode == "窄" else WIDE

    @property
    def grid(self):
        return self.interval if self.step is None else coarse_grid(self.interval, self.step)


CONFIGS = (
    Config("甲的今天：全分辨率全档", "窄", 100, None, None, 1, None),
    Config("只粗档 步0.05·精算±2档", "窄", 100, 0.05, 2, 1, None),
    Config("只粗档 步0.05·精算±3档", "窄", 100, 0.05, 3, 1, None),
    Config("只粗位置 步4·精算±32px", "窄", 100, None, None, 4, 32),
    Config("只粗像素 15x15·精算挑中那档", "窄", 15, None, 0, 1, None),
    Config("只粗像素 15x15·精算±1档", "窄", 15, None, 1, 1, None),
    Config("粗像素15x15 + 粗档0.05±2", "窄", 15, 0.05, 2, 1, None),
    Config("粗像素15x15 + 粗位置±32px", "窄", 15, None, 0, 1, 32),
    Config("粗档0.05±2 + 粗位置±32px", "窄", 100, 0.05, 2, 4, 32),
    Config("三处一起粗：15x15 + 0.05±2 + ±16px", "窄", 15, 0.05, 2, 1, 16),
    Config("宽区间 + 全分辨率全档", "宽", 100, None, None, 1, None),
    Config("宽区间 + 三处一起粗", "宽", 15, 0.05, 2, 1, 16),
    Config("宽区间 + 粗像素15x15·精算挑中那档", "宽", 15, None, 0, 1, None),
)


def coarse_grid(interval, step: float) -> tuple:
    """一个区间上的粗档网格（含两端）。端点是 0.01 的整数倍时，粗点全落在细网格上。"""
    lo, hi = min(interval), max(interval)
    count = int(round((hi - lo) / step))
    grid = [round(lo + index * step, 2) for index in range(count + 1)]
    if grid[-1] < hi - 1e-9:
        grid.append(round(hi, 2))
    return tuple(sorted(set(grid)))


def _truth_map() -> dict:
    """已落盘的逐位真值：(帧, 卡位) -> 干员名；**空串表示那一格不是干员**。缺键 = 没真值。"""
    found: dict = {}

    def take(path: Path) -> None:
        record = json.loads(path.read_text(encoding="utf-8"))
        for pair in record["pairs"]:
            name = pair["under_the_cap"]
            for side in ("capped", "clean"):
                node = pair[side]
                frame = node.get("frame") or Path(node["file"]).stem.split("_")[0]
                found[(frame, int(node["index"]))] = name

    take(SAMPLES / "红罩" / "取材.json")
    take(SAMPLES / "预处理" / "取材.json")

    harvest = json.loads((SAMPLES / "点名" / "取材.json").read_text(encoding="utf-8"))
    for item in harvest["harvest"]:
        stem = Path(item["file"]).stem
        for index, name in enumerate(item["identity_right_to_left"]):
            found[(stem, index)] = "" if name == "未知" else name

    bench_harvest = json.loads((SAMPLES / "备战席" / "取材.json").read_text(encoding="utf-8"))
    for item in bench_harvest["harvest"]:
        stem = Path(item["file"]).stem
        for index in item.get("not_operator") or []:
            found.setdefault((stem, index), "")
    return found


def _corpus() -> tuple:
    """全部夹具逐张：每张卡落一条记录（批次 · 帧 · 卡位 · 中心 · 红罩标记 · 库）。"""
    libraries = {
        "shared": load_cardfaces(SHARED_LIBRARY),
        "twins": load_cardfaces(SAMPLES / "立绘两组" / "库1.json")
        + load_cardfaces(SAMPLES / "立绘两组" / "库2.json"),
    }
    records: list = []
    for batch_name in BATCHES:
        for case in load_batch(SAMPLES / batch_name).cases:
            frame = frame_from_raw(case.raw())
            centres = bench.squad_card_centres(frame, pitch=PITCH)
            if not centres:
                continue
            top = float(bench.squad_card_rect(0, PITCH).y)
            key = "twins" if batch_name == "立绘两组" else "shared"
            for index, (centre, _) in enumerate(centres):
                flag = bench.redeploying(frame, centre, card_top=top)
                records.append({
                    "batch": batch_name,
                    "frame": case.name,
                    "index": index,
                    "centre": float(centre),
                    "redeploying": None if flag is None else bool(flag),
                    "library": key,
                    "region": np.ascontiguousarray(
                        frame.crop(identity_region(centre, pitch=PITCH))),
                })
    return records, libraries


def _full_curve(region: np.ndarray, art: np.ndarray, scales) -> dict:
    """那条链内层那一步：逐档 resize + matchTemplate，记分与最优位置。"""
    rows = {}
    for scale in scales:
        height = int(round(region.shape[0] * scale))
        width = int(round(region.shape[1] * scale))
        if height > art.shape[0] or width > art.shape[1]:
            continue
        tpl = cv2.resize(region, (width, height), interpolation=cv2.INTER_LINEAR)
        result = cv2.matchTemplate(art, tpl, cv2.TM_CCOEFF_NORMED)
        _lo, hi, _loloc, hiloc = cv2.minMaxLoc(result)
        rows[scale] = (float(hi), int(hiloc[0]), int(hiloc[1]))
    return rows


def _coarse_setup(region: np.ndarray, art: np.ndarray, pixels: int):
    """把查询与立绘按同一个因子缩下去 —— 粗探就在这两块小的上做。"""
    factor = pixels / float(region.shape[0])
    small_query = cv2.resize(region, (pixels, pixels), interpolation=cv2.INTER_AREA)
    small_art = cv2.resize(
        art,
        (max(1, int(round(art.shape[1] * factor))),
         max(1, int(round(art.shape[0] * factor)))),
        interpolation=cv2.INTER_AREA,
    )
    return small_query, small_art, factor


def _best(rows: dict):
    scale = max(rows, key=lambda key: (rows[key][0], -key))
    return scale, rows[scale]


def _fine_window(config: Config, picked: float) -> set:
    """精算要扫哪几档：挑中那一档周围 ±window 档（None = 整个区间）。"""
    if config.window is None:
        return set(config.interval)
    half = config.window * 0.01
    return {s for s in config.interval if picked - half - 1e-9 <= s <= picked + half + 1e-9}


def _crop_rect(art, region, scale, loc, half):
    height = int(round(region.shape[0] * scale))
    width = int(round(region.shape[1] * scale))
    x0 = max(0, int(round(loc[0] - half)))
    y0 = max(0, int(round(loc[1] - half)))
    x1 = min(int(art.shape[1]), int(round(loc[0] + width + half)))
    y1 = min(int(art.shape[0]), int(round(loc[1] + height + half)))
    return x0, y0, x1, y1


def _run_pipeline(region: np.ndarray, art: np.ndarray, config: Config,
                  counter: list) -> dict:
    """**真的跑一遍**「粗探挑档挑位置 + 全分辨率精算」，数次数、量像素量、回报出的分。

    报出的分**只**来自这一步里的全分辨率 matchTemplate（粗探那几刀只用来挑档 / 挑位置）。
    """
    calls = [0]

    def _match(a, b):
        calls[0] += 1
        return cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)

    small_query, small_art, factor = _coarse_setup(region, art, config.pixels)
    area_coarse = 0
    coarse = {}
    for scale in config.grid:
        height = int(round(small_query.shape[0] * scale))
        width = int(round(small_query.shape[1] * scale))
        if height > small_art.shape[0] or width > small_art.shape[1]:
            continue
        tpl = cv2.resize(small_query, (width, height), interpolation=cv2.INTER_LINEAR)
        result = _match(small_art, tpl)
        area_coarse += int(small_art.shape[0]) * int(small_art.shape[1])
        if config.stride > 1:
            block = result[:: config.stride, :: config.stride]
            spot = np.unravel_index(int(np.argmax(block)), block.shape)
            y = int(spot[0]) * config.stride
            x = int(spot[1]) * config.stride
            coarse[scale] = (float(result[y, x]), x, y)
        else:
            _lo, hi, _loloc, hiloc = cv2.minMaxLoc(result)
            coarse[scale] = (float(hi), int(hiloc[0]), int(hiloc[1]))
    picked, (cscore, cx, cy) = _best(coarse)
    fine = sorted(_fine_window(config, picked))
    estimate = (cx / factor, cy / factor)
    best = -2.0
    best_loc = None
    area_fine = 0
    for scale in fine:
        height = int(round(region.shape[0] * scale))
        width = int(round(region.shape[1] * scale))
        if height > art.shape[0] or width > art.shape[1]:
            continue
        tpl = cv2.resize(region, (width, height), interpolation=cv2.INTER_LINEAR)
        if config.pos_half is None:
            block = art
            x0 = y0 = 0
        else:
            x0, y0, x1, y1 = _crop_rect(art, region, scale, estimate, config.pos_half)
            block = np.ascontiguousarray(art[y0:y1, x0:x1])
        area_fine += int(block.shape[0]) * int(block.shape[1])
        result = _match(block, tpl)
        _lo, hi, _loloc, hiloc = cv2.minMaxLoc(result)
        if float(hi) > best:
            best = float(hi)
            best_loc = (int(hiloc[0]) + x0, int(hiloc[1]) + y0)
    counter[0] += calls[0]
    return {"picked": picked, "estimate": estimate, "fine": fine,
            "reported": best, "reported_loc": best_loc,
            "calls_coarse": len(config.grid), "calls_fine": len(fine),
            "calls": calls[0], "area_coarse": area_coarse, "area_fine": area_fine}


CACHE_FILE = REPO / "data" / "probe" / "coarse_cache.json"
"""取值域曲线那一份缓存（gitignored 的 scratch）—— 第一次跑花二十几分钟，之后秒开。
它只是**同一条链的中间结果**，每次仍然要用 roster_scores 逐位核一遍（_crosscheck）。"""


def _cache_key(batch: str, frame: str, index: int, name: str) -> str:
    return "%s|%s|%d|%s" % (batch, frame, index, name)


def _runs(corpus: list, libraries: dict, *, reuse: bool) -> dict:
    """取值域曲线：**这就是那条链**逐档重跑（crosscheck 核对到逐位）。"""
    wanted = [(_cache_key(r["batch"], r["frame"], r["index"], name), r, name)
              for r in corpus for name, _art in libraries[r["library"]]]
    if reuse and CACHE_FILE.is_file():
        try:
            blob = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
            if set(blob["keys"]) == {key for key, _r, _n in wanted}:
                flat = blob["curves"]
                data: dict = {}
                for key, record, name in wanted:
                    rows = {float(scale): (float(score), int(lx), int(ly))
                            for scale, score, lx, ly in flat[key]}
                    data.setdefault(
                        (record["batch"], record["frame"], record["index"]), {})[name] = rows
                print("  从缓存读入（%d 条曲线）" % len(flat))
                return data
            print("  缓存对不上这一批语料，重算")
        except Exception as problem:
            print("  缓存读不动（%r），重算" % (problem,))
    data = {}
    for record in corpus:
        region = record["region"]
        faces = libraries[record["library"]]
        data[(record["batch"], record["frame"], record["index"])] = {
            name: _full_curve(region, art, CACHE) for name, art in faces
        }
    keys = []
    curves = {}
    for record in corpus:
        here = data[(record["batch"], record["frame"], record["index"])]
        for name, _art in libraries[record["library"]]:
            key = _cache_key(record["batch"], record["frame"], record["index"], name)
            keys.append(key)
            curves[key] = [[scale, rows[0], rows[1], rows[2]]
                           for scale, rows in sorted(here[name].items())]
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps({"grid": list(CACHE), "keys": keys, "curves": curves},
                   ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    print("  曲线缓存写到 %s" % CACHE_FILE)
    return data


def _crosscheck(corpus: list, libraries: dict, data: dict) -> dict:
    """线上那 11 档上的最大值必须与 roster_scores 逐位相等 —— 不相等就说明缓存不是那条链。"""
    bad = []
    checked = 0
    for record in corpus:
        faces = libraries[record["library"]]
        live = roster_scores(record["region"], faces)
        here = data[(record["batch"], record["frame"], record["index"])]
        checked += len(faces)
        for score, name, _scale in live:
            mine = max(here[name][s][0] for s in NARROW if s in here[name])
            if score != mine:
                bad.append({"batch": record["batch"], "frame": record["frame"],
                            "index": record["index"], "entry": name,
                            "roster_scores": score, "curve_max": mine})
    return {"pairs": checked, "mismatches": len(bad), "rows": bad[:20],
            "all_equal": not bad}


def _accuracy(corpus: list, libraries: dict, data: dict, config: Config) -> dict:
    """全语料逐对：粗探挑的档 / 位置 对不对 · 报出的分与那条链逐位同不同。

    这里报出的分从取值域缓存里取（缓存已与 roster_scores 逐位核对过）—— 真跑的那一遍
    在 _measure 里，两者必须给同一个结论。
    """
    per_pair = []
    per_card = []
    calls_coarse = calls_fine = 0
    area_coarse = area_fine = 0
    for record in corpus:
        faces = libraries[record["library"]]
        here = data[(record["batch"], record["frame"], record["index"])]
        card_all = True
        scores = {}
        for name, art in faces:
            rows = here[name]
            true_scale, true_row = _best({s: rows[s] for s in config.interval if s in rows})
            small_query, small_art, factor = _coarse_setup(record["region"], art, config.pixels)
            call_names = []
            if config.pixels == 100 and config.stride == 1:
                coarse = {s: rows[s] for s in config.grid if s in rows}
            else:
                coarse = {}
                for scale in config.grid:
                    height = int(round(small_query.shape[0] * scale))
                    width = int(round(small_query.shape[1] * scale))
                    if height > small_art.shape[0] or width > small_art.shape[1]:
                        continue
                    tpl = cv2.resize(small_query, (width, height),
                                     interpolation=cv2.INTER_LINEAR)
                    result = cv2.matchTemplate(small_art, tpl, cv2.TM_CCOEFF_NORMED)
                    if config.stride > 1:
                        block = result[:: config.stride, :: config.stride]
                        spot = np.unravel_index(int(np.argmax(block)), block.shape)
                        y = int(spot[0]) * config.stride
                        x = int(spot[1]) * config.stride
                        coarse[scale] = (float(result[y, x]), x, y)
                    else:
                        _lo, hi, _loloc, hiloc = cv2.minMaxLoc(result)
                        coarse[scale] = (float(hi), int(hiloc[0]), int(hiloc[1]))
            calls_coarse += len(config.grid)
            area_coarse += len(config.grid) * int(small_art.shape[0]) * int(small_art.shape[1])
            picked, (cscore, cx, cy) = _best(coarse)
            fine = sorted(_fine_window(config, picked))
            calls_fine += len(fine)
            height = int(round(record["region"].shape[0] * true_scale))
            width = int(round(record["region"].shape[1] * true_scale))
            if config.pos_half is None:
                area_fine += len(fine) * int(art.shape[0]) * int(art.shape[1])
            else:
                area_fine += len(fine) * (height + 2 * config.pos_half) * (
                    width + 2 * config.pos_half)
            if config.pos_half is None:
                # 整张立绘：精算那一遍就是这条链在 fine 那几档上的最大值。
                reported = max(rows[s][0] for s in fine if s in rows)
            else:
                # 位置窗那一刀**真跑**：把立绘裁到窗口里再 matchTemplate。
                # 落在窗口里的位置，值与整张图上同位置**逐位相同**（TM_CCOEFF_NORMED
                # 只取决于立绘那一块），所以这一趟给出的就是线上会报出来的那个数。
                reported = -2.0
                estimate = (cx / factor, cy / factor)
                for scale in fine:
                    height = int(round(record["region"].shape[0] * scale))
                    width = int(round(record["region"].shape[1] * scale))
                    if height > art.shape[0] or width > art.shape[1]:
                        continue
                    tpl_ = cv2.resize(record["region"], (width, height),
                                      interpolation=cv2.INTER_LINEAR)
                    x0, y0, x1, y1 = _crop_rect(art, record["region"], scale,
                                                estimate, config.pos_half)
                    block = np.ascontiguousarray(art[y0:y1, x0:x1])
                    result = cv2.matchTemplate(block, tpl_, cv2.TM_CCOEFF_NORMED)
                    reported = max(reported, float(result.max()))
            chain = true_row[0]
            distance = max(abs(cx / factor - true_row[1]), abs(cy / factor - true_row[2]))
            scale_ok = true_scale in fine
            pos_ok = config.pos_half is None or distance <= config.pos_half + 1e-9
            per_pair.append({
                "batch": record["batch"], "frame": record["frame"],
                "index": record["index"], "entry": name,
                "true_scale": true_scale, "picked_scale": picked,
                "scale_steps": int(round(abs(true_scale - picked) / 0.01)),
                "distance_px": round(distance, 2),
                "scale_ok": bool(scale_ok), "pos_ok": bool(pos_ok),
                "peak_inside": bool(scale_ok and pos_ok),
                "reported": reported, "chain": chain,
                "delta": reported - chain,
                "equal_within_1e9": bool(abs(reported - chain) <= 1e-9),
                "bit_identical": bool(reported == chain),
            })
            scores[name] = reported
            if reported != chain:
                card_all = False
        per_card.append({"batch": record["batch"], "frame": record["frame"],
                         "index": record["index"],
                         "all_entries_bit_identical": card_all,
                         "score": max(scores.values())})
    bad = [r for r in per_pair if not r["bit_identical"]]
    histogram: dict = {}
    for row in per_pair:
        histogram[str(row["scale_steps"])] = histogram.get(str(row["scale_steps"]), 0) + 1
    return {
        "label": config.label, "interval": config.mode,
        "interval_scales": len(config.interval), "coarse_grid_scales": len(config.grid),
        "pixels": config.pixels, "coarse_step": config.step,
        "fine_half_steps": config.window, "stride": config.stride,
        "position_half_px": config.pos_half, "pairs": len(per_pair),
        "scale_exact": sum(1 for r in per_pair if r["scale_steps"] == 0),
        "scale_within_1": sum(1 for r in per_pair if r["scale_steps"] <= 1),
        "scale_within_2": sum(1 for r in per_pair if r["scale_steps"] <= 2),
        "scale_steps_histogram": histogram,
        "scale_in_fine_window": sum(1 for r in per_pair if r["scale_ok"]),
        "position_in_window": sum(1 for r in per_pair if r["pos_ok"]),
        "peak_inside_window": sum(1 for r in per_pair if r["peak_inside"]),
        "bit_identical_when_peak_inside": sum(
            1 for r in per_pair if r["peak_inside"] and r["bit_identical"]),
        "equal_within_1e9": sum(1 for r in per_pair if r["equal_within_1e9"]),
        "max_abs_delta_when_peak_inside": round(max(
            [abs(r["delta"]) for r in per_pair if r["peak_inside"]] or [0.0]), 12),
        "bit_identical": sum(1 for r in per_pair if r["bit_identical"]),
        "cards": len(per_card),
        "cards_bit_identical": sum(1 for r in per_card if r["all_entries_bit_identical"]),
        "calls_coarse": calls_coarse, "calls_fine": calls_fine,
        "calls_total": calls_coarse + calls_fine,
        "area_coarse_px": area_coarse, "area_fine_px": area_fine,
        "bit_mismatch_count": len(bad), "bit_mismatch_rows": bad[:12],
    }


def _measure(corpus: list, libraries: dict, configs, picks) -> dict:
    """真跑：取固定的几张卡，把每种走法**整个执行一遍**，数真次数、量真墙钟（取中位）。"""
    sample = [corpus[i] for i in picks]
    out = {}
    for config in configs:
        counter = [0]
        rounds = []
        reported = {}
        for _ in range(3):
            started = time.perf_counter()
            reported = {}
            counter[0] = 0
            for record in sample:
                faces = libraries[record["library"]]
                for name, art in faces:
                    got = _run_pipeline(record["region"], art, config, counter)
                    reported[(record["batch"], record["frame"], record["index"], name)] = got
            rounds.append(time.perf_counter() - started)
        out[config.label] = {
            "cards": len(sample), "entries": sum(
                len(libraries[r["library"]]) for r in sample),
            "match_template_calls": counter[0],
            "median_s": round(statistics.median(rounds), 4),
            "rounds_s": [round(value, 4) for value in rounds],
        }
    return out


def _measure_bit_identity(corpus: list, libraries: dict, data: dict, config: Config,
                          picks) -> dict:
    """真跑那一遍里报出的分，与那条链逐位比 —— 这是「报出的分不许变」的直接证据。"""
    sample = [corpus[i] for i in picks]
    bad = []
    count = 0
    for record in sample:
        faces = libraries[record["library"]]
        here = data[(record["batch"], record["frame"], record["index"])]
        for name, art in faces:
            got = _run_pipeline(record["region"], art, config, [0])
            chain = max(here[name][s][0] for s in config.interval if s in here[name])
            count += 1
            if got["reported"] != chain:
                bad.append({"frame": record["frame"], "index": record["index"],
                            "entry": name, "reported": got["reported"], "chain": chain,
                            "picked": got["picked"], "fine": got["fine"]})
    return {"config": config.label, "pairs": count, "mismatches": len(bad),
            "rows": bad[:12], "all_equal": not bad}


def _verdict(rows_by_entry: dict, interval, truth, group):
    """一张卡上的判定 —— **读者按线算**（最高分过线 而且 余量拉开）。"""
    ranked = sorted(rows_by_entry.items(),
                    key=lambda item: (-item[1]["score"], item[0]))
    top_name, top = ranked[0]
    second_score = ranked[1][1]["score"] if len(ranked) > 1 else -2.0
    margin = top["score"] - second_score
    exact = (top["score"] >= ROSTER_SOLID_MATCH_SCORE
             and margin >= ROSTER_AMBIGUITY_MARGIN)
    short = []
    if top["score"] < ROSTER_SOLID_MATCH_SCORE:
        short.append("分数不够")
    if margin < ROSTER_AMBIGUITY_MARGIN:
        short.append("余量不够")
    if truth is None and group is not None:
        # 双胞胎那一批没有真值：卡属于哪一组，读到另一组的条目就一定是错的
        wrong = bool(exact) and not top_name.startswith("组" + group + "·")
    elif truth is None:
        wrong = None
    elif truth == "":
        wrong = bool(exact)
    else:
        wrong = bool(exact) and top_name != truth
    return {"name": top_name, "score": round(top["score"], 6),
            "scale": top["scale"], "margin": round(margin, 6),
            "verdict": "精确" if exact else "未知", "wrong": wrong,
            "blocked_by": " + ".join(short) or None}


def _benefit(corpus: list, libraries: dict, data: dict, truth: dict) -> dict:
    """第 4 张表：区间铺得更宽之后，逐张报 未知→精确 · 新出现的错名字。"""
    rows = []
    for record in corpus:
        faces = libraries[record["library"]]
        here = data[(record["batch"], record["frame"], record["index"])]
        # **两张分数表**：一张只取今天那 11 档，一张取宽区间那 26 档。
        # 一个库条目对一块取区的分数与库里还有谁无关，所以两张表都从同一份缓存里取。
        narrow_by_entry = {}
        wide_by_entry = {}
        for name, _art in faces:
            for target, interval in ((narrow_by_entry, NARROW), (wide_by_entry, WIDE)):
                rows_here = {s: here[name][s] for s in interval if s in here[name]}
                scale, (score, lx, ly) = _best(rows_here)
                target[name] = {"score": score, "scale": scale, "loc": (lx, ly)}
        group = None
        if record["batch"] == "立绘两组":
            group = record["frame"][1]
        key = (record["frame"], record["index"])
        known = truth.get(key)
        before = _verdict(narrow_by_entry, NARROW, known, group)
        after = _verdict(wide_by_entry, WIDE, known, group)
        rows.append({
            "batch": record["batch"], "frame": record["frame"],
            "index": record["index"], "redeploying": record["redeploying"],
            "truth": known, "group": group,
            "before": before, "after": after,
            "changed": before["verdict"] != after["verdict"] or before["name"] != after["name"],
        })
    gained = [r for r in rows if r["before"]["verdict"] == "未知"
              and r["after"]["verdict"] == "精确"]
    lost = [r for r in rows if r["before"]["verdict"] == "精确"
            and r["after"]["verdict"] == "未知"]
    wrong_new = [r for r in gained if r["after"]["wrong"] is True]
    wrong_before = [r for r in rows if r["before"]["wrong"] is True]
    wrong_after = [r for r in rows if r["after"]["wrong"] is True]
    return {
        "cards": len(rows),
        "before": {"exact": sum(1 for r in rows if r["before"]["verdict"] == "精确"),
                   "unknown": sum(1 for r in rows if r["before"]["verdict"] == "未知")},
        "after": {"exact": sum(1 for r in rows if r["after"]["verdict"] == "精确"),
                  "unknown": sum(1 for r in rows if r["after"]["verdict"] == "未知")},
        "gained_unknown_to_exact": gained,
        "lost_exact_to_unknown": lost,
        "wrong_new": wrong_new,
        "wrong_before": wrong_before,
        "wrong_after": wrong_after,
        "truth_cells_used": sum(1 for r in rows if r["truth"] is not None),
        "rows": rows,
    }


def _sweep(corpus: list, libraries: dict, data: dict, window: int) -> dict:
    """第 5 张表：像素数 → 选档正确率 → 代价（选档正确率在全语料上算）。"""
    out = {}
    for pixels in SWEEP_PIXELS:
        config = Config("p%d" % pixels, "窄", pixels, None, window, 1, None)
        got = _accuracy(corpus, libraries, data, config)
        out[str(pixels)] = {
            "pixels": pixels,
            "matches_the_full_resolution_peak": got["scale_exact"],
            "pairs": got["pairs"],
            "scale_within_1": got["scale_within_1"],
            "bit_identical": got["bit_identical"],
            "calls_total": got["calls_total"],
            "calls_fine": got["calls_fine"],
            "area_fine_px": got["area_fine_px"],
            "calls_coarse": got["calls_coarse"],
            "area_coarse_px": got["area_coarse_px"],
        }
    return out


def main() -> int:
    started = time.perf_counter()

    corpus, libraries = _corpus()
    print("=== 语料 ===")
    for batch_name in BATCHES:
        rows = [r for r in corpus if r["batch"] == batch_name]
        if not rows:
            continue
        print("  %-6s %3d 张（红罩 %d）  库 %d 条"
              % (batch_name, len(rows), sum(1 for r in rows if r["redeploying"] is True),
                 len(libraries[rows[0]["library"]])))
    print("  合计 %d 张卡" % len(corpus))

    print()
    print("=== 取值域曲线（%d 档 0.95…1.20） ===" % len(CACHE))
    marked = time.perf_counter()
    data = _runs(corpus, libraries, reuse="--fresh" not in sys.argv)
    print("  跑完 %.1f s" % (time.perf_counter() - marked))

    cross = _crosscheck(corpus, libraries, data)
    print("  逐位核对（缓存 vs roster_scores）：%d 对 · 不等 %d"
          % (cross["pairs"], cross["mismatches"]))
    if not cross["all_equal"]:
        raise AssertionError("缓存不是那条链：%s" % cross["rows"][:3])

    truth = _truth_map()
    print("  已落盘真值 %d 格" % len(truth))

    print()
    print("=== 表 1 / 2：粗探准不准 · 报出的分逐位同不同 ===")
    tables = {}
    for config in CONFIGS:
        got = _accuracy(corpus, libraries, data, config)
        tables[config.label] = got
        print("  %-38s 对%5d 选档准%5d 窗内%5d 峰在窗内%5d 逐位同%5d 1e-9内%5d"
              % (config.label, got["pairs"], got["scale_exact"],
                 got["scale_in_fine_window"], got["peak_inside_window"],
                 got["bit_identical"], got["equal_within_1e9"]))

    print()
    print("=== 表 2b：真跑一遍，报出的分与那条链逐位比 ===")
    picks = list(range(0, len(corpus), max(1, len(corpus) // 8)))[:8]
    live_check = [_measure_bit_identity(corpus, libraries, data, config, picks)
                  for config in (CONFIGS[0], CONFIGS[4], CONFIGS[9], CONFIGS[11])]
    for row in live_check:
        print("  %-38s 对 %4d · 不等 %d" % (row["config"], row["pairs"], row["mismatches"]))

    print()
    print("=== 表 3：代价（真跑 %d 张卡，各跑 3 次取中位） ===" % len(picks))
    measured = _measure(corpus, libraries, CONFIGS, picks)
    for config in CONFIGS:
        got = measured[config.label]
        table = tables[config.label]
        print("  %-38s 真次数%7d 墙钟%8.3f s  解析次数%8d"
              % (config.label, got["match_template_calls"], got["median_s"],
                 table["calls_total"]))

    print()
    print("=== 表 4：区间铺宽（%.2f…%.2f）之后的收益与风险 ==="
          % (min(WIDE), max(WIDE)))
    benefit = _benefit(corpus, libraries, data, truth)
    print("  卡 %d：精确 %d→%d · 未知 %d→%d"
          % (benefit["cards"], benefit["before"]["exact"], benefit["after"]["exact"],
             benefit["before"]["unknown"], benefit["after"]["unknown"]))
    print("  未知→精确 %d 张 · 精确→未知 %d 张 · 新出现的错名字 %d 张（宽区间下错名字共 %d 张）"
          % (len(benefit["gained_unknown_to_exact"]),
             len(benefit["lost_exact_to_unknown"]),
             len(benefit["wrong_new"]), len(benefit["wrong_after"])))
    for row in benefit["gained_unknown_to_exact"]:
        print("      %-6s #%-2d 读成 %-14s %s 余量 %s 真值 %s 错? %s"
              % (row["frame"], row["index"], row["after"]["name"],
                 row["after"]["score"], row["after"]["margin"],
                 row["truth"] if row["truth"] is not None else "(没真值)",
                 row["after"]["wrong"]))

    print()
    print("=== 表 5：最少要多少个像素 ===")
    sweep = {}
    for window in (1, 2):
        sweep["精算±%d 档" % window] = _sweep(corpus, libraries, data, window)
        print("  --- 精算窗口 ±%d 档 ---" % window)
        for key, row in sweep["精算±%d 档" % window].items():
            print("      %3sx%-3s=%5d px  选档准 %5d/%5d  逐位同 %5d  精算档数/对 %.2f"
                  % (key, key, row["pixels"], row["matches_the_full_resolution_peak"],
                     row["pairs"], row["bit_identical"],
                     row["calls_fine"] / float(row["pairs"])))

    payload = {
        "note": ("先粗后细能不能同时买到「尺子铺得更宽」与「更快」——**只量不改**。"
                 "报出的分全部来自全分辨率那一次精算；粗探只用来挑档 / 挑位置。"),
        "corpus": "tests/帧样本/{备战席,红罩,余量,立绘两组}",
        "region": "roster.identity_region（干净卡那块 100x100 的框）+ roster_scores",
        "caveat": ("测的是干净卡那条路（roster_scores）。红罩卡那一格换的是统计量"
                   "（margin.MaskedRuler 的掩码相关），尺度网格与 resize 那一步是共用的 ——"
                   "所以这一份量出来的收益对红罩那条路是**下界**；它自己的曲线没量。"),
        "interval_today": list(NARROW),
        "interval_wide": list(WIDE),
        "criterion": {"score": ROSTER_SOLID_MATCH_SCORE,
                      "margin": ROSTER_AMBIGUITY_MARGIN,
                      "note": "判定由读者按这两条线算 —— 这一份只落分数与余量。"},
        "truth_cells": len(truth),
        "crosscheck": cross,
        "tables": tables,
        "live_bit_identity": live_check,
        "cost_measured": measured,
        "measure_sample": {"picks": picks, "cards": len(picks)},
        "benefit_wider_interval": {key: value for key, value in benefit.items()
                                   if key != "rows"},
        "benefit_rows": benefit["rows"],
        "pixel_sweep": sweep,
        "seconds": round(time.perf_counter() - started, 2),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                   encoding="utf-8", newline="\n")
    print()
    print("耗时 %.1f s" % payload["seconds"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
