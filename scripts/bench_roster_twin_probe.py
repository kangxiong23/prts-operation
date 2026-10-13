
r"""塞雷娅那一对在干净卡取区上为什么只有 0.83 —— 乙：只找原因。

    .venv\Scripts\python.exe scripts\bench_roster_twin_probe.py

## 这一单问什么

组2 编队页切出来的「塞雷娅」那块 与 组2 备战席上她那张卡，在**干净卡取区**
（identity_region 那个 100x100 的框）上像度 0.8309；同一位干员在**组1**那一对上是 0.9875。
这一单**只找原因**：差在哪（机制）· 是不是这一套皮肤的特例 · 若是普遍原因会影响哪些卡。

## 一个字节都不改

线（ROSTER_SOLID_MATCH_SCORE / ROSTER_AMBIGUITY_MARGIN）· 取区形状 · 那条链
（roster_scores）· 尺度区间 · 库 · 模型 全不动。这里只把那条链**内层那一步**按尺度
一档一档重跑，把曲线拿出来 —— 重跑出来的最大值必须与 roster_scores 给的分**逐位相等**
（_crosscheck_curve），对不上就报错，所以它不是另写的一份匹配。

## 三个量的意思

- **尺度曲线**：cv2.resize(query, s) 与 cv2.matchTemplate(art, tpl, TM_CCOEFF_NORMED)
  的 max，逐档记分与最优位置。IDENTITY_SCALES 那 11 档上的最大值就是线上那个分；
  另外在 0.80…1.40（步长 0.01）上再跑一遍，看**峰落在哪、现在的区间有没有把峰截掉**。
- **最优位置**：cv2.minMaxLoc 给的左上角（立绘坐标）。期望位置 = 身份框折进立绘
  （(52, 76) * s）—— 残差就是"偏了多少像素"。
- **对照图**：不猜，落图给人看（构图 · 姿态 · 立绘在卡里的位置 · 有没有东西压在上面）。

## 落哪

tests/帧样本/立绘两组/塞雷娅/ ：读数.json + 两张对照图 + README.md（怎么重建）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.engine.device.adb import SQUAD_CARD_PITCH  # noqa: E402
from ark.engine.vision import bench  # noqa: E402
from ark.engine.vision.frame import frame_from_raw  # noqa: E402
from ark.engine.vision.roster import (  # noqa: E402
    IDENTITY_BOX_X,
    IDENTITY_BOX_Y,
    IDENTITY_SCALES,
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
    find_squad_card,
    identity_region,
    roster_scores,
    squad_card_scores,
)
from ark.offline.assets_gen.cardface import load_cardfaces  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402

BATCH = REPO / "tests" / "帧样本" / "立绘两组"
OUT = BATCH / "塞雷娅"
PITCH = SQUAD_CARD_PITCH
GROUPS = ("1", "2")
WIDE = tuple(round(0.80 + index * 0.01, 3) for index in range(61))
FONT = None
"""对照图上的字要中文 —— PIL 的默认位图字体画不了汉字，所以找一份系统字体；找不到就
**不画字**（图还是留得下的），不要为了它去装依赖。"""
for _candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
                   "C:/Windows/Fonts/simsun.ttc"):
    if Path(_candidate).is_file():
        try:
            FONT = ImageFont.truetype(_candidate, 15)
        except Exception:
            FONT = None
        break


def _text(draw, xy, message, fill=(230, 230, 230)) -> None:
    """画一行字 —— 没有字体就不画（图的其它部分不受影响）。"""
    if FONT is not None:
        draw.text(xy, message, fill=fill, font=FONT)
"""大网格：0.80 … 1.40，步长 0.01 —— 只为看峰落在哪、区间有没有截掉它。"""


def _png(path: Path, array: np.ndarray) -> None:
    """非 ASCII 路径上 cv2.imwrite 会**静默失败**，所以走 imencode + tofile。"""
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError("编码失败 " + str(path))
    buffer.tofile(str(path))


def _curve(query: np.ndarray, art: np.ndarray, scales) -> dict:
    """那条链**内层那一步**按尺度重跑一遍：逐档记分 + 最优位置。"""
    rows = {}
    for scale in scales:
        height = int(round(query.shape[0] * scale))
        width = int(round(query.shape[1] * scale))
        if height > art.shape[0] or width > art.shape[1]:
            continue
        tpl = cv2.resize(query, (width, height), interpolation=cv2.INTER_LINEAR)
        result = cv2.matchTemplate(art, tpl, cv2.TM_CCOEFF_NORMED)
        _lo, hi, _loloc, hiloc = cv2.minMaxLoc(result)
        rows["%.2f" % scale] = {
            "score": round(float(hi), 6),
            "loc": [int(hiloc[0]), int(hiloc[1])],
        }
    return rows


def _peak(curve: dict) -> tuple:
    key = max(curve, key=lambda k: curve[k]["score"])
    return float(key), curve[key]


def _slot(frame, label: str, faces) -> tuple:
    """这一条落在第几张卡 —— 单条那条路就一招：某一条库过线。
    组2 的塞雷娅过不了线（0.8309），所以退一步用**同一批分数的 argmax**
    （squad_card_scores 是公开的，它就是这条路判据本身）。"""
    scored = squad_card_scores(frame, label, faces, pitch=PITCH)
    return int(scored[0][1]), float(scored[0][0]), scored


def _pair(faces, frame, label: str) -> dict:
    """一位干员在**这一组**上的那一对：编队页那块立绘 与 她那张卡。"""
    slot, top, _ranked = _slot(frame, label, faces)
    centres = bench.squad_card_centres(frame, pitch=PITCH)
    art = next(block for name, block in faces if name == label)
    region = np.ascontiguousarray(frame.crop(identity_region(centres[slot][0], pitch=PITCH)))
    card = np.ascontiguousarray(frame.crop(bench.squad_card_rect(slot, PITCH)))

    fine = _curve(region, art, IDENTITY_SCALES)
    wide = _curve(region, art, WIDE)
    peak_scale, peak = _peak(wide)
    at_min = fine["%.2f" % IDENTITY_SCALES[0]]
    at_max = fine["%.2f" % IDENTITY_SCALES[-1]]
    best_at = max(fine, key=lambda k: fine[k]["score"])

    expected = [IDENTITY_BOX_X * float(best_at), IDENTITY_BOX_Y * float(best_at)]
    loc = fine[best_at]["loc"]
    ranked = roster_scores(region, faces)
    online = {name: (score, scale) for score, name, scale in ranked}
    luma = region.astype(np.float32).mean(axis=2)
    return {
        "library_entry": label,
        "slot": slot,
        "slot_top_score": round(top, 6),
        "online_score": round(online[label][0], 6),
        "online_scale": online[label][1],
        "curve_best_scale": float(best_at),
        "curve_max": round(max(row["score"] for row in fine.values()), 6),
        "best_loc": loc,
        "expected_loc": [round(value, 2) for value in expected],
        "residual_px": [round(loc[0] - expected[0], 2), round(loc[1] - expected[1], 2)],
        "residual_norm": round(float(np.hypot(loc[0] - expected[0], loc[1] - expected[1])), 2),
        "widest_peak_scale": peak_scale,
        "widest_peak_score": round(peak["score"], 6),
        "peak_on_grid_edge": peak_scale in (WIDE[0], WIDE[-1]),
        "interval_edge_scores": {"1.05": round(at_min["score"], 6),
                                 "1.15": round(at_max["score"], 6)},
        "above_interval_peak": round(
            max(row["score"] for key, row in wide.items()
                if not (IDENTITY_SCALES[0] - 1e-9 <= float(key) <= IDENTITY_SCALES[-1] + 1e-9)),
            6),
        "box_luma_mean": round(float(luma.mean()), 3),
        "box_luma_std": round(float(luma.std()), 3),
        "box_rgb_mean": [round(float(value), 2) for value in region.reshape(-1, 3).mean(axis=0)],
        "ranking_top": [
            {"entry": name, "score": round(score, 6)}
            for score, name, _scale in ranked[:4]
        ],
        "curve_11": fine,
        "curve_wide": wide,
        "_art": art,
        "_card": card,
        "_region": region,
    }


def _draw_pair(pairs: dict) -> np.ndarray:
    """组1 / 组2 各一行：编队页那块立绘 | 备战席那张卡 | 立绘上画出最优匹配框。"""
    scale = 0.55
    art_h = int(480 * scale)
    art_w = int(214 * scale)
    card_h = int(214 * scale)
    card_w = int(204 * scale)
    gap = 12
    width = art_w + card_w + art_w + gap * 4
    height = (art_h + gap) * len(GROUPS) + gap
    canvas = Image.new("RGB", (width, height), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    for row, group in enumerate(GROUPS):
        item = pairs[group]
        y = gap + row * (art_h + gap)
        art_small = np.asarray(Image.fromarray(item["_art"]).resize((art_w, art_h)))
        card_small = np.asarray(Image.fromarray(item["_card"]).resize((card_w, card_h)))
        canvas.paste(Image.fromarray(art_small), (gap, y))
        draw.rectangle([gap, y, gap + art_w - 1, y + art_h - 1], outline=(90, 90, 100))
        x2 = gap + art_w + gap
        canvas.paste(Image.fromarray(card_small), (x2, y))
        box = [IDENTITY_BOX_X * scale * (204 / 204), IDENTITY_BOX_Y * scale,
               (IDENTITY_BOX_X + 100) * scale, (IDENTITY_BOX_Y + 100) * scale]
        draw.rectangle([x2 + box[0], y + box[1], x2 + box[2], y + box[3]],
                       outline=(255, 210, 0), width=2)
        x3 = x2 + card_w + gap
        canvas.paste(Image.fromarray(art_small), (x3, y))
        draw.rectangle([x3, y, x3 + art_w - 1, y + art_h - 1], outline=(90, 90, 100))
        loc = item["best_loc"]
        draw.rectangle([x3 + loc[0] * scale * (art_w / 214.0) - 1,
                        y + loc[1] * scale * (art_h / 480.0) - 1,
                        x3 + (loc[0] * (art_w / 214.0) + 100 * item["curve_best_scale"]) * scale,
                        y + (loc[1] * (art_h / 480.0) + 100 * item["curve_best_scale"]) * scale],
                       outline=(0, 255, 120), width=2)
        _text(draw, (gap + 4, y + 4), "组" + group + " " + item["library_entry"]
                  + "  %.4f @ %.2f  残差 %.1f px"
                  % (item["online_score"], item["curve_best_scale"], item["residual_norm"]),
                  fill=(255, 255, 255))
    return np.asarray(canvas)[:, :, ::-1].copy()


def _draw_boxes(pairs: dict) -> np.ndarray:
    """三块放大并排：组1 的身份框 | 组2 的身份框 | 两块之差（绝对差）。"""
    zoom = 3
    size = 100 * zoom
    gap = 16
    canvas = Image.new("RGB", (size * 3 + gap * 4, size + gap * 2 + 22), (24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    boxes = {group: pairs[group]["_region"] for group in GROUPS}
    diff = np.abs(boxes["1"].astype(np.int16) - boxes["2"].astype(np.int16)).astype(np.uint8)
    for column, (name, block) in enumerate(
        (("组1 " + pairs["1"]["library_entry"], boxes["1"]),
         ("组2 " + pairs["2"]["library_entry"], boxes["2"]),
         ("两块绝对差", diff))
    ):
        x = gap + column * (size + gap)
        big = np.asarray(Image.fromarray(block).resize((size, size), Image.NEAREST))
        canvas.paste(Image.fromarray(big), (x, gap))
        draw.rectangle([x, gap, x + size - 1, gap + size - 1], outline=(90, 90, 100))
        if column < 2:
            item = pairs[GROUPS[column]]
            _text(draw, (x, gap + size + 4),
                      name + "  均亮 %.1f 对比 %.1f  分 %.4f @ %.2f"
                      % (item["box_luma_mean"], item["box_luma_std"],
                         item["online_score"], item["curve_best_scale"]),
                      fill=(230, 230, 230))
        else:
            _text(draw, (x, gap + size + 4),
                      "均差 %.1f（像素）—— 两套皮肤本来就不一样" % float(diff.mean()),
                      fill=(230, 230, 230))
    return np.asarray(canvas)[:, :, ::-1].copy()


def _draw_curves(pairs: dict, every: dict) -> np.ndarray:
    """两张小图：左边塞雷娅 组1 vs 组2（大网格）；右边这一批每一位的大网格峰位。"""
    width, height = 1180, 430
    canvas = Image.new("RGB", (width, height), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)

    def axes(x0, y0, w, h, title):
        draw.rectangle([x0, y0, x0 + w, y0 + h], outline=(90, 90, 100))
        for score in (0.25, 0.5, 0.75, 1.0):
            y = y0 + h - int(h * score)
            draw.line([x0, y, x0 + w, y], fill=(48, 48, 56))
            _text(draw, (x0 - 30, y - 6), "%.2f" % score, fill=(150, 150, 160))
        for scale in (0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40):
            x = x0 + int(w * (scale - 0.80) / 0.60)
            draw.line([x, y0, x, y0 + h], fill=(48, 48, 56))
            _text(draw, (x - 14, y0 + h + 4), "%.2f" % scale, fill=(150, 150, 160))
        _text(draw, (x0, y0 - 20), title, fill=(240, 240, 240))
        left = x0 + int(w * (IDENTITY_SCALES[0] - 0.80) / 0.60)
        right = x0 + int(w * (IDENTITY_SCALES[-1] - 0.80) / 0.60)
        draw.rectangle([left, y0, right, y0 + h], outline=(70, 90, 130))
        _text(draw, (left + 2, y0 + 2), "尺度区间", fill=(120, 160, 220))
        line = y0 + h - int(h * ROSTER_SOLID_MATCH_SCORE)
        draw.line([x0, line, x0 + w, line], fill=(200, 90, 90))
        _text(draw, (x0 + w - 74, line - 14), "分数线 0.85", fill=(220, 130, 130))

    def poly(rows, x0, y0, w, h, color):
        keys = sorted(rows, key=float)
        points = []
        for key in keys:
            x = x0 + int(w * (float(key) - 0.80) / 0.60)
            y = y0 + h - int(h * min(1.0, rows[key]["score"]))
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=2)

    axes(60, 40, 480, 320, "塞雷娅那一对：分 vs 尺度（大网格 0.80…1.40）")
    poly(pairs["1"]["curve_wide"], 60, 40, 480, 320, (90, 200, 255))
    poly(pairs["2"]["curve_wide"], 60, 40, 480, 320, (255, 170, 60))
    _text(draw, (70, 46), "组1 %.4f" % pairs["1"]["online_score"], fill=(90, 200, 255))
    _text(draw, (70, 62), "组2 %.4f" % pairs["2"]["online_score"], fill=(255, 170, 60))
    for item, color in ((pairs["1"], (90, 200, 255)), (pairs["2"], (255, 170, 60))):
        key = "%.2f" % item["widest_peak_scale"]
        x = 60 + int(480 * (item["widest_peak_scale"] - 0.80) / 0.60)
        y = 40 + 320 - int(320 * min(1.0, item["curve_wide"][key]["score"]))
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], outline=color, width=3)
        _text(draw, (x - 18, y - 20), "峰 %.2f" % item["widest_peak_scale"], fill=color)

    axes(660, 40, 480, 320, "这一批每一位：大网格上的峰位（组1 / 组2）")
    for group, color in (("1", (90, 200, 255)), ("2", (255, 170, 60))):
        for operator, item in every[group].items():
            x = 660 + int(480 * (item["widest_peak_scale"] - 0.80) / 0.60)
            y = 40 + 320 - int(320 * min(1.0, item["widest_peak_score"]))
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], outline=color, width=2)
            if item["online_score"] < 0.95:
                _text(draw, (x - 16, y + 5), operator, fill=color)
    return np.asarray(canvas)[:, :, ::-1].copy()


def main() -> int:
    batch = load_batch(BATCH)
    frames = {case.name: frame_from_raw(case.raw()) for case in batch.cases}
    faces1 = load_cardfaces(BATCH / "库1.json")
    faces2 = load_cardfaces(BATCH / "库2.json")
    faces = list(faces1) + list(faces2)
    if len({name for name, _ in faces}) != len(faces):
        raise ValueError("标签撞了：同名两条会让「他在第几张卡」没有唯一答案")
    operators = [name.split("·", 1)[1] for name, _ in faces1]

    every = {group: {} for group in GROUPS}
    payload = {
        "note": "塞雷娅那一对为什么只有 0.83：乙，只找原因。线 · 取区形状 · 那条链 · "
                "尺度区间 · 库 · 模型一个字节没动。",
        "batch": "tests/帧样本/立绘两组",
        "library": ["tests/帧样本/立绘两组/库1.json", "tests/帧样本/立绘两组/库2.json"],
        "entry": "ark.engine.vision.roster.roster_scores（只调用；曲线是它内层那一步逐档重跑，"
                 "最大值与它逐位核对）",
        "region": {"who": "roster.identity_region", "box_in_card": [52, 76, 100, 100]},
        "criterion": {"score": ROSTER_SOLID_MATCH_SCORE,
                      "margin": ROSTER_AMBIGUITY_MARGIN},
        "pairs": {},
        "every_operator": {},
        "crosscheck": {},
    }

    for group in GROUPS:
        frame = frames["组" + group + "_备战席"]
        for operator in operators:
            label = "组" + group + "·" + operator
            item = _pair(faces, frame, label)
            every[group][operator] = item
            if operator == "塞雷娅":
                payload["pairs"][group] = {key: value for key, value in item.items()
                                    if not key.startswith("_")}
        payload["every_operator"][group] = {
            operator: {
                "library_entry": item["library_entry"],
                "slot": item["slot"],
                "online_score": item["online_score"],
                "online_scale": item["online_scale"],
                "widest_peak_scale": item["widest_peak_scale"],
                "widest_peak_score": item["widest_peak_score"],
                "residual_norm": item["residual_norm"],
                "above_interval_peak": item["above_interval_peak"],
                "interval_edge_scores": item["interval_edge_scores"],
                "box_luma_mean": item["box_luma_mean"],
                "box_luma_std": item["box_luma_std"],
            }
            for operator, item in every[group].items()
        }

    # 核对：曲线在 IDENTITY_SCALES 上的最大值必须与线上那条链给的分**逐位相等**。
    cross = {}
    for group in GROUPS:
        for operator, item in every[group].items():
            ok = item["online_score"] == item["curve_max"]
            cross[item["library_entry"]] = {
                "roster_scores": item["online_score"],
                "curve_max": item["curve_max"],
                "equal": bool(ok),
            }
    payload["crosscheck"] = {
        "note": "曲线是那条链内层那一步逐档重跑；它在 IDENTITY_SCALES 上的最大值 "
                "必须与 roster_scores 给的分逐位相等 —— 不相等就说明这不是同一条链。",
        "all_equal": all(row["equal"] for row in cross.values()),
        "rows": cross,
    }

    slots = {}
    for group in GROUPS:
        frame = frames["组" + group + "_备战席"]
        for operator in operators:
            label = "组" + group + "·" + operator
            slots[label] = find_squad_card(frame, label, faces, pitch=PITCH)
    payload["crosscheck"]["find_squad_card"] = slots
    payload["crosscheck"]["find_squad_card_note"] = (
        "单条那条路的卡位（恰有一张卡过分数线才给下标，否则 None）—— 它只吃分数，"
        "与余量线无关。")

    # 跨批对照：同一位干员在**另一批**（红罩 / 余量，库是 点名/编队页.json）上的读数。
    other = {}
    redcap = BATCH.parent / "红罩" / "点名.json"
    payload["cross_batch"] = {
        "note": "红罩那一批（库 = 点名/编队页.json，与这一批**不同源**）里，名字相同的"
                "那些卡的干净卡取区读数 —— 用来分开「这个人在这套立绘上不好认」与"
                "「这个人在这条链上就是低」。",
        "overlap_operators": [],
        "redcap_batch": {},
    }
    try:
        landed = json.loads(redcap.read_text(encoding="utf-8"))
        for frame in landed["frames"]:
            for slot in frame["slots"]:
                reading = slot["reading"]
                name = reading.get("name") or slot.get("truth")
                if name is None:
                    continue
                other.setdefault(name, []).append({
                    "frame": frame["name"], "index": slot["index"],
                    "score": slot.get("score"), "margin": slot.get("margin"),
                    "redeploying": slot["redeploying"],
                })
        payload["cross_batch"]["overlap_operators"] = sorted(
            {name.split("·", 1)[1] for name, _ in faces1}
            & {name.split("·", 1)[-1] for name in other})
        payload["cross_batch"]["redcap_batch"] = {
            name: other[name] for name in sorted(other)}
    except Exception as problem:  # 那一批正在被另一条单重写
        payload["cross_batch"]["error"] = repr(problem)

    # 对照图
    OUT.mkdir(parents=True, exist_ok=True)
    raw_pairs = {group: every[group]["塞雷娅"] for group in GROUPS}
    _png(OUT / "对照_那一对.png", _draw_pair(raw_pairs))
    _png(OUT / "对照_身份框.png", _draw_boxes(raw_pairs))
    _png(OUT / "曲线.png", _draw_curves(raw_pairs, every))
    (OUT / "读数.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8", newline="\n",
    )

    print("=== 塞雷娅那一对 ===")
    for group in GROUPS:
        item = payload["pairs"][group]
        print("  组%s 卡位 %d  线上 %.4f @ %.2f  大网格峰 %.2f = %.4f  区间外最高 %.4f"
              % (group, item["slot"], item["online_score"], item["curve_best_scale"],
                 item["widest_peak_scale"], item["widest_peak_score"],
                 item["above_interval_peak"]))
        print("     最优位置 %s 期望 %s 残差 %s (%s px)  框内均亮 %.1f 对比 %.1f"
              % (item["best_loc"], item["expected_loc"], item["residual_px"],
                 item["residual_norm"], item["box_luma_mean"], item["box_luma_std"]))
        print("     这一格的前四名 %s" % item["ranking_top"])
    print("=== 每位（组2）===")
    for operator, item in payload["every_operator"]["2"].items():
        print("  %-8s 线上 %.4f @ %.2f  大网格峰 %.2f = %.4f  区间外最高 %.4f  残差 %.1f px"
              % (operator, item["online_score"], item["online_scale"],
                 item["widest_peak_scale"], item["widest_peak_score"],
                 item["above_interval_peak"], item["residual_norm"]))
    print("=== 逐位核对 ===")
    print("  全部相等：%s" % payload["crosscheck"]["all_equal"])
    print("  跨批重叠的干员：%s" % payload["cross_batch"]["overlap_operators"])
    print("wrote", OUT / "读数.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
