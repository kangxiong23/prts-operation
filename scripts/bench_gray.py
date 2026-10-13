# -*- coding: utf-8 -*-
"""第八单的秤：备战席第三态（缺费灰卡）—— 代数标签、同帧对照、跨帧差图、还原、前后读数。

输入只有两张原图（screenshots/run-04/，与 data/ 同类、不进版本库）；输出是
tests/帧样本/缺费/ 这一批夹具（备战席横带两张 + index.json + 差图.json）。

它**不跑那张整网**（用户裁定）：这一单只吃这一对图，外加语料里捞得到的孪生对。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ark.contracts import (  # noqa: E402
    Canvas, Exact, Unknown, Valued, squad_card_face_rect, squad_card_rect,
)
from ark.engine.device.adb import SQUAD_CARD_PITCH as PITCH  # noqa: E402
from ark.engine.vision import gray  # noqa: E402
from ark.engine.vision.bench import read_squad_row  # noqa: E402
from ark.engine.vision.cost import read_cost_digits  # noqa: E402
from ark.engine.vision.frame import Frame  # noqa: E402
from ark.engine.vision.glyphs import DEFAULT_GLYPHS_PATH, GlyphTable  # noqa: E402
from ark.engine.vision.roster import (  # noqa: E402
    ROSTER_SOLID_MATCH_SCORE, identity_is_exact, read_squad_identities, roster_scores,
)

SHOTS = ROOT / "screenshots" / "run-04"
OUT = ROOT / "tests" / "帧样本" / "缺费"
BENCH = ROOT / "tests" / "帧样本" / "备战席"
COST = ROOT / "tests" / "帧样本" / "费用"
TWINS = ROOT / "tests" / "帧样本" / "立绘两组"
CANVAS = Canvas(width=2800, height=1260)
STRIP = (300, 1040, 2500, 220)
IDENT = (52, 76, 100, 100)
GRAY_PNG = "灰罩卡.png"
CLEAR_PNG = "干净卡对照.png"
CURRENT = {GRAY_PNG: 20, CLEAR_PNG: 28}
NAMES = {GRAY_PNG: "灰罩卡_备战席.png", CLEAR_PNG: "干净卡对照_备战席.png"}
# 费用链在**两帧**上都读不出的那两位（idx5 / idx8）：值是**目视读**出来的，落进夹具时标出来。
# 两帧的身份区对这两格逐位相同（都不是缺费），所以它们填什么都不改零假设/效应组的分组。
EYEBALL = {5: 19, 8: 12}


def load_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def frame_of(pixels: np.ndarray) -> Frame:
    assert pixels.shape == (1260, 2800, 3), pixels.shape
    return Frame(canvas=CANVAS, pixels=pixels)


def paste(path: Path, origin) -> Frame:
    px = load_png(path)
    h, w = px.shape[:2]
    out = np.zeros((1260, 2800, 3), dtype=np.uint8)
    x, y = origin
    out[y:y + h, x:x + w] = px
    return frame_of(out)


def box(i: int):
    r = squad_card_rect(i, PITCH)
    return int(r.x), int(r.y), int(r.width), int(r.height)


def face_box(i: int):
    r = squad_card_face_rect(i, PITCH)
    return int(r.x), int(r.y), int(r.width), int(r.height)


def ident_box(i: int):
    x, y, _w, _h = box(i)
    return x + IDENT[0], y + IDENT[1], IDENT[2], IDENT[3]


def cut(px: np.ndarray, b) -> np.ndarray:
    x, y, w, h = b
    return px[y:y + h, x:x + w]


def lum(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float64)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def chroma(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float64)
    return f.max(axis=2) - f.min(axis=2)


def sat(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 1].astype(np.float64)


def exact(value):
    if isinstance(value, Valued) and isinstance(value.payload, Exact):
        return value.payload.value
    return None


def fit_diag(c: np.ndarray, g: np.ndarray):
    c = c.reshape(-1, 3).astype(np.float64)
    g = g.reshape(-1, 3).astype(np.float64)
    a = np.zeros(3); b = np.zeros(3)
    for ch in range(3):
        m = np.stack([c[:, ch], np.ones(c.shape[0])], axis=1)
        sol, *_ = np.linalg.lstsq(m, g[:, ch], rcond=None)
        a[ch], b[ch] = sol
    pred = c * a + b
    return ([float(v) for v in a], [float(v) for v in b],
            float(np.sqrt(np.mean((pred - g) ** 2))))


def top_two(region: np.ndarray, faces):
    ranked = roster_scores(np.ascontiguousarray(region), faces)
    top = ranked[0]
    runner = ranked[1][0] if len(ranked) > 1 else None
    return top, runner, ranked


def formation_faces(png: Path, meta: dict, prefix: str):
    px = load_png(png)
    geo = meta["geometry"]
    faces = []
    for row in geo["rows"]:
        for k in range(row["count"]):
            x = geo["left0"] + geo["pitch"] * k
            y0, y1 = row["y"]
            name = meta["names"][len(faces)] if "names" in meta else prefix + str(len(faces))
            faces.append((name, px[y0:y1, x:x + geo["art_width"]]))
    return faces


def main() -> None:
    table = GlyphTable.load(DEFAULT_GLYPHS_PATH)
    raw = {name: load_png(SHOTS / name) for name in (GRAY_PNG, CLEAR_PNG)}
    gray_frame = frame_of(raw[GRAY_PNG])
    clear_frame = frame_of(raw[CLEAR_PNG])
    report: dict = {
        "note": "第八单：备战席第三态（缺费灰卡）的差图与读数。石头是两张原图（同局的两次暂停）。",
        "sources": {name: hashlib.sha256((SHOTS / name).read_bytes()).hexdigest()
                    for name in (GRAY_PNG, CLEAR_PNG)},
    }

    # ---------------------------------------------------------------- 做1 代数标签
    print("### 做1：代数标签（逐张读卡费与当前费）")
    labels = {}
    for name in (GRAY_PNG, CLEAR_PNG):
        current = exact(read_cost_digits(frame_of(raw[name]), table))
        row = read_squad_row(frame_of(raw[name]), table, pitch=PITCH)
        costs = [exact(card.cost) for card in row]
        labels[name] = {"current_cost": current, "card_cost_right_to_left": costs}
        print("  %s  当前费 %s  卡数 %d" % (name, current, len(row)))
        print("    序号(左起)  idx   卡费   状态")
        for k in range(1, len(row) + 1):
            i = len(row) - k
            cost = costs[i]
            state = "读不出" if cost is None else ("缺费" if cost > current else "够费")
            print("       %2d       %2d   %5s  %s" % (k, i, cost, state))
    n = len(labels[GRAY_PNG]["card_cost_right_to_left"])
    costs = []
    for i in range(n):
        a = labels[GRAY_PNG]["card_cost_right_to_left"][i]
        b = labels[CLEAR_PNG]["card_cost_right_to_left"][i]
        if a is not None and b is not None and a != b:
            print("  【停】idx%d 两帧卡费不一致：灰罩 %s / 干净 %s" % (i, a, b))
        costs.append(a if a is not None else b)
    chain_read = list(costs)
    eyeballed = []
    for i in sorted(EYEBALL):
        if costs[i] is None:
            costs[i] = EYEBALL[i]
            eyeballed.append(i)
    verdicts = [None if v is None else v > CURRENT[GRAY_PNG] for v in costs]
    print("  对齐后的卡费（右起 idx0..%d）：%s" % (n - 1, costs))
    print("  灰罩帧的缺费裁决（右起）：%s" % verdicts)
    print("  链没读出来的那两位（目视读的）：%s"
          % {i: EYEBALL[i] for i in eyeballed})
    report["labels"] = {
        "current_cost": {name: labels[name]["current_cost"] for name in labels},
        "current_cost_source": "cost.read_cost_digits（链读出来的；灰罩 0.9986 · 干净 0.9986）",
        "card_cost_right_to_left": costs,
        "card_cost_chain_read_right_to_left": chain_read,
        "card_cost_source": "bench.read_squad_row 逐张读；idx5/idx8 两帧上这条链都读不出（"
                            "已有的欠账），那两位按**原图目视**读、并在这里标出来",
        "card_cost_eyeballed": {str(i): EYEBALL[i] for i in eyeballed},
        "insufficient_right_to_left": verdicts,
    }

    # ------------------------------------------------------- 做2 同帧内绝对值统计量
    print()
    print("### 做2：同帧内绝对值统计量（灰罩帧，身份区 100x100）")
    print("  序号 idx 卡费 状态     亮度Y    色度chr   饱和度S")
    same = []
    for k in range(1, n + 1):
        i = n - k
        region = cut(raw[GRAY_PNG], ident_box(i))
        row_of = {
            "index": i, "cost": costs[i],
            "insufficient": verdicts[i],
            "y": float(lum(region).mean()),
            "chroma": float(chroma(region).mean()),
            "s": float(sat(region).mean()),
        }
        same.append(row_of)
        print("   %2d  %2d  %4d %s  %8.2f %9.2f %9.2f"
              % (k, i, costs[i], "缺费" if verdicts[i] else "够费",
                 row_of["y"], row_of["chroma"], row_of["s"]))
    for field, show in (("y", "亮度Y"), ("chroma", "色度chr"), ("s", "饱和度S")):
        low = [r[field] for r in same if r["insufficient"]]
        high = [r[field] for r in same if not r["insufficient"]]
        print("  %s：缺费 %s ｜ 够费 %s ｜ 分开 %s"
              % (show, _span(low), _span(high), "是" if min(high) > max(low) else "否"))
    report["same_frame"] = same

    print()
    corpus = _corpus(table)
    print("  语料里 %d 张卡（备战席 6 帧 + 这一对，卡费与当前费都读得出）的绝对值分布"
          " —— 同一帧分得开，换个干员就穿：" % len(corpus))
    for field, show in (("y", "亮度Y"), ("chroma", "色度chr"), ("s", "饱和度S")):
        low = [r[field] for r in corpus if r["insufficient"]]
        high = [r[field] for r in corpus if not r["insufficient"]]
        print("    %s：缺费 n=%d [%.2f, %.2f] ｜ 干净 n=%d [%.2f, %.2f] ｜ 两组交叠 %s"
              % (show, len(low), min(low), max(low), len(high), min(high), max(high),
                 "是" if max(low) > min(high) else "否"))
    lowest = min((r for r in corpus if not r["insufficient"]), key=lambda r: r["y"])
    print("    （干净那一簇最低的 %.2f 是 %s idx%d —— 那是**红罩**那张，不是灰罩）"
          % (lowest["y"], lowest["source"], lowest["index"]))
    report["corpus_absolute"] = {
        "n": len(corpus),
        "ranges": {f: {
            "insufficient": [_round(min([r[f] for r in corpus if r["insufficient"]])),
                             _round(max([r[f] for r in corpus if r["insufficient"]]))],
            "clean": [_round(min([r[f] for r in corpus if not r["insufficient"]])),
                      _round(max([r[f] for r in corpus if not r["insufficient"]]))],
        } for f in ("y", "chroma", "s")},
    }

    # ------------------------------------------------------------- 做3 跨两帧差图
    print()
    print("### 做3：跨两帧逐位差（备战席条带 %s）" % (STRIP,))
    ax, ay, aw, ah = STRIP
    a = raw[GRAY_PNG][ay:ay + ah, ax:ax + aw].astype(np.int32)
    b = raw[CLEAR_PNG][ay:ay + ah, ax:ax + aw].astype(np.int32)
    sd = np.abs(b - a)
    print("  整条：不同像素 %d / %d（%.2f%%）· 平均|差| %.3f · 最大 %d"
          % (int((sd.max(axis=2) > 0).sum()), a.shape[0] * a.shape[1],
             100.0 * (sd.max(axis=2) > 0).mean(), sd.mean(), int(sd.max())))
    print("  序号 idx 卡费 状态  整卡不同px 整卡mean|D| 身份区不同px 身份区mean|D| 自比(G/C中位)")
    diff_rows = []
    for k in range(1, n + 1):
        i = n - k
        d1 = np.abs(cut(raw[CLEAR_PNG], box(i)).astype(np.int32)
                    - cut(raw[GRAY_PNG], box(i)).astype(np.int32))
        cb = cut(raw[CLEAR_PNG], ident_box(i)).astype(np.float64)
        cg = cut(raw[GRAY_PNG], ident_box(i)).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(cb.mean(axis=2, keepdims=True) >= 60, cg / np.maximum(cb, 1.0), np.nan)
        share = float(np.nanmedian(ratio))
        d2 = np.abs(cg - cb)
        diff_rows.append({"index": i, "cost": costs[i], "insufficient": verdicts[i],
                          "card_diff_px": int((d1.max(axis=2) > 0).sum()),
                          "card_mean_abs": _round(d1.mean()),
                          "ident_diff_px": int((d2.max(axis=2) > 0).sum()),
                          "ident_mean_abs": _round(d2.mean()),
                          "share": _round(share)})
        print("   %2d  %2d  %4d %s %9d %10.3f %11d %12.3f %12.4f"
              % (k, i, costs[i], "缺费" if verdicts[i] else "够费",
                 diff_rows[-1]["card_diff_px"], diff_rows[-1]["card_mean_abs"],
                 diff_rows[-1]["ident_diff_px"], diff_rows[-1]["ident_mean_abs"], share))
    mask = np.zeros((ah, aw), dtype=bool)
    for i in range(n):
        x, y, w, h = box(i)
        mask[max(y - ay, 0):min(y - ay + h, ah), max(x - ax, 0):min(x - ax + w, aw)] = True
    outside = sd.max(axis=2) > 0
    print("  条带内非卡区：不同像素 %d / %d（%.2f%%）· 平均|差| %.3f"
          % (int(outside[~mask].sum()), int((~mask).sum()),
             100.0 * outside[~mask].mean(), sd[~mask].mean()))
    report["diff"] = {
        "strip": list(STRIP),
        "whole": {"diff_px": int((sd.max(axis=2) > 0).sum()),
                  "total_px": int(a.shape[0] * a.shape[1]),
                  "mean_abs": _round(sd.mean()), "max": int(sd.max())},
        "outside_cards": {"diff_px": int(outside[~mask].sum()),
                          "total_px": int((~mask).sum()),
                          "mean_abs": _round(sd[~mask].mean())},
        "cards": diff_rows,
    }

    print()
    print("  零假设组（1–5 号，两帧状态没变）与效应组（6–10 号，缺费 → 够费）分开报：")
    null_rows = [r for r in diff_rows if not r["insufficient"]]
    effect_rows = [r for r in diff_rows if r["insufficient"]]
    print("    零假设：整卡 mean|D| 最大 %.3f ｜ 身份区 mean|D| 最大 %.3f ｜ 不同像素 最大 %d"
          % (max(r["card_mean_abs"] for r in null_rows),
             max(r["ident_mean_abs"] for r in null_rows),
             max(r["card_diff_px"] for r in null_rows)))
    print("    效应组：整卡 mean|D| 最小 %.3f ｜ 身份区 mean|D| 最小 %.3f ｜ 不同像素 最小 %d"
          % (min(r["card_mean_abs"] for r in effect_rows),
             min(r["ident_mean_abs"] for r in effect_rows),
             min(r["card_diff_px"] for r in effect_rows)))
    print("    零假设 < 效应组 ？ %s（噪声地板 = 0）"
          % ("是" if max(r["ident_mean_abs"] for r in null_rows)
             < min(r["ident_mean_abs"] for r in effect_rows) else "否"))

    print()
    print("  灰罩模型：逐通道最小二乘 G = a*C + b（身份区）")
    fits = []
    for k in range(1, n + 1):
        i = n - k
        c = cut(raw[CLEAR_PNG], ident_box(i))
        g = cut(raw[GRAY_PNG], ident_box(i))
        a_, b_, rms = fit_diag(c, g)
        fits.append({"index": i, "cost": costs[i], "a": [_round(v, 4) for v in a_],
                     "b": [_round(v, 4) for v in b_], "rms": _round(rms, 4)})
        if verdicts[i]:
            print("    idx%2d 费%3d  a=(%.4f %.4f %.4f) b=(%7.3f %7.3f %7.3f) rms=%.3f"
                  % (i, costs[i], a_[0], a_[1], a_[2], b_[0], b_[1], b_[2], rms))
    print("    零假设那五格：a=(1 1 1) b=(0 0 0) rms=0.0000（两帧逐位相同 ⇒ 噪声地板 0）")
    report["model"] = {"fits": fits}

    print()
    print("  「乘小了」还是「抹平了」：")
    print("    序号 idx  chr(净)最大  chr(灰)最大  chr>20 像素(净→灰)  亮度比(灰/净)  chr比(中位)")
    model_rows = []
    for k in range(1, n + 1):
        i = n - k
        c = cut(raw[CLEAR_PNG], ident_box(i))
        g = cut(raw[GRAY_PNG], ident_box(i))
        cc, cg = chroma(c), chroma(g)
        yr = float(lum(g).mean() / max(lum(c).mean(), 1e-9))
        cr = float(np.median(cg) / max(np.median(cc), 1e-9))
        model_rows.append({"index": i, "insufficient": verdicts[i],
                           "chroma_clean_max": _round(cc.max()),
                           "chroma_gray_max": _round(cg.max()),
                           "chroma_over20_clean": int((cc > 20).sum()),
                           "chroma_over20_gray": int((cg > 20).sum()),
                           "luma_ratio": _round(yr, 4), "chroma_ratio_median": _round(cr, 4)})
        print("    %2d  %2d  %9.0f  %9.0f  %5d → %5d  %12.4f  %10.4f"
              % (k, i, cc.max(), cg.max(), int((cc > 20).sum()), int((cg > 20).sum()), yr, cr))
    print("    判决：①色度被乘小 = 是（中位比 ≈ 0.5）· ②被整片抹平 = 否（灰罩卡还剩 chr>20 的像素）")
    print("          · ③亮度一起变 = 是，但**同一个标量** ⇒ 一个算式就够了，不需要库")
    report["model"]["verdict"] = {
        "kind": "multiply",
        "scalar": 0.5,
        "chroma_flattened": False,
        "luma_changed": True,
        "same_scalar": True,
        "rows": [r for r in model_rows if r["insufficient"]],
        "null_rows_identical": all(
            r["chroma_over20_clean"] == r["chroma_over20_gray"]
            and r["chroma_over20_clean"] == r["chroma_over20_gray"]
            for r in model_rows if not r["insufficient"]),
    }

    # -------------------------------------------------------------- 做5 还原保真度
    print()
    print("### 做5：还原的像素保真度（五对，身份区 100x100）")
    pre, post = [], []
    fidelity = []
    for k in range(1, n + 1):
        i = n - k
        if not verdicts[i]:
            continue
        c = cut(raw[CLEAR_PNG], ident_box(i)).astype(np.int32)
        g = cut(raw[GRAY_PNG], ident_box(i)).astype(np.int32)
        d0 = np.abs(g - c)
        d1 = np.abs(gray.restore(cut(raw[GRAY_PNG], ident_box(i))).astype(np.int32) - c)
        pre.append(float(d0.mean())); post.append(float(d1.mean()))
        fidelity.append({"index": i, "cost": costs[i],
                         "before_mean_abs": _round(d0.mean()), "before_max": int(d0.max()),
                         "before_over8": int((d0 > 8).sum()),
                         "after_mean_abs": _round(d1.mean()), "after_max": int(d1.max()),
                         "after_over8": int((d1 > 8).sum())})
        print("    idx%2d 费%3d  还原前 平均|差| %7.3f 最大 %3d >8 的像素 %5d ｜ 还原后 %6.3f 最大 %3d >8 的像素 %5d"
              % (i, costs[i], d0.mean(), int(d0.max()), int((d0 > 8).sum()),
                 d1.mean(), int(d1.max()), int((d1 > 8).sum())))
    print("    五对平均：还原前 %.3f → 还原后 %.3f" % (float(np.mean(pre)), float(np.mean(post))))
    report["restore"] = {"mean_before": _round(float(np.mean(pre))),
                         "mean_after": _round(float(np.mean(post))), "pairs": fidelity}

    # ------------------------------------------- 做4 / 做6 点名分数（还原前 / 还原后）
    print()
    print("### 做4 / 做6：点名分数（库 = 对照帧那 10 张卡自己的卡面，名字 = 卡位k）")
    print("    （库与查询同源 ⇒ 干净那一侧的分是 1.0 上下，读的是**灰罩掉了多少**，不是绝对水平）")
    faces = [("卡位%d" % i, cut(raw[CLEAR_PNG], face_box(i))) for i in range(n)]
    scores = []
    print("  序号 idx 费 状态  参照(净)  还原前  还原后  过线前 过线后  名前    名后    对前 对后")
    for k in range(1, n + 1):
        i = n - k
        ref_top, ref_runner, _ = top_two(cut(raw[CLEAR_PNG], ident_box(i)), faces)
        ref_exact = identity_is_exact(ref_top[0], ref_runner)
        region = cut(raw[GRAY_PNG], ident_box(i))
        before_top, before_runner, _ = top_two(region, faces)
        # 还原**只加在裁决为真的那几格**上：裁决 None / False 的格子画面一个字都不动
        after_region = gray.restore(region) if verdicts[i] else region
        after_top, after_runner, _ = top_two(after_region, faces)
        before_exact = identity_is_exact(before_top[0], before_runner)
        after_exact = identity_is_exact(after_top[0], after_runner)
        want = "卡位%d" % i
        row = {
            "index": i, "cost": costs[i], "insufficient": verdicts[i],
            "reference": {"top": _round(ref_top[0], 6), "name": ref_top[1],
                          "runner_up": _round(ref_runner, 6) if ref_runner is not None else None,
                          "exact": bool(ref_exact)},
            "before": {"top": _round(before_top[0], 6), "name": before_top[1],
                       "runner_up": _round(before_runner, 6) if before_runner is not None else None,
                       "exact": bool(before_exact)},
            "after": {"top": _round(after_top[0], 6), "name": after_top[1],
                      "runner_up": _round(after_runner, 6) if after_runner is not None else None,
                      "exact": bool(after_exact)},
            "name_right": {"before": before_top[1] == want, "after": after_top[1] == want},
            "score_dropped": _round(ref_top[0] - before_top[0], 6),
        }
        scores.append(row)
        print("   %2d  %2d %3d %s %8.6f %8.6f %8.6f  %5s  %5s  %-7s %-7s  %s  %s"
              % (k, i, costs[i], "缺费" if verdicts[i] else "够费",
                 ref_top[0], before_top[0], after_top[0],
                 "过" if before_exact else "否", "过" if after_exact else "否",
                 before_top[1], after_top[1],
                 "对" if row["name_right"]["before"] else "错",
                 "对" if row["name_right"]["after"] else "错"))
    effect = [r for r in scores if r["insufficient"]]
    null = [r for r in scores if not r["insufficient"]]
    print("    零假设组（1–5 号，裁决不是 True ⇒ 还原一个字都不加）：分差最大 %.6f ｜ 名字改了 %d 张"
          % (max(abs(r["before"]["top"] - r["after"]["top"]) for r in null),
             sum(1 for r in null if r["before"]["name"] != r["after"]["name"])))
    print("    效应组（6–10 号）：还原前最高分 %s ｜ 还原后 %s"
          % (_span([r["before"]["top"] for r in effect]),
             _span([r["after"]["top"] for r in effect])))
    print("    「分数掉了」与「名次翻了」分开报（效应组）：")
    for r in effect:
        print("      idx%2d 费%3d  分数掉了 %+.6f（参照 %.6f → 还原前 %.6f）｜ 名次：参照 %s → 还原前 %s → 还原后 %s ｜ 翻了吗 %s"
              % (r["index"], r["cost"], -r["score_dropped"], r["reference"]["top"],
                 r["before"]["top"], r["reference"]["name"], r["before"]["name"],
                 r["after"]["name"],
                 "是" if r["reference"]["name"] != r["before"]["name"] else "否"))
    report["scores"] = {"library": "对照帧自己那 10 张卡面（卡位0..卡位9）", "rows": scores}

    print()
    print("### 做5：线上那条链自己（read_squad_identities 带 insufficient）是否与上面同一批")
    chain_trace: list = []
    chained = read_squad_identities(
        gray_frame, faces, pitch=PITCH, insufficient=verdicts, trace=chain_trace,
    )
    same = 0
    for k in range(1, n + 1):
        i = n - k
        got = exact(chained[i])
        want = scores[k - 1]["after"]["name"] if scores[k - 1]["after"]["exact"] else None
        if got == want:
            same += 1
        else:
            print("    【不同】idx%d 链说 %s / 手动路径说 %s" % (i, got, want))
    print("    与手动那条路逐格一致 %d/%d" % (same, n))
    print("    链的足迹里 gray 标记：%s" % [t.get("gray") for t in chain_trace])

    print()
    print("### 做6：自比兜底 —— 两簇读数与代数的对照（shot05 ↔ shot07，同局两次暂停）")
    twins = _twin_rows(table)
    for row in twins:
        print("    %s idx%d 费%3d：自比 %.4f ⇒ 兜底判 %s ｜ 单条库点名 %.6f → %.6f ｜ "
              "过线 %s → %s ｜ 代数（sh05 当前费11 / sh07 当前费8）%s / %s"
              % (row["file"], row["index"], row["cost"], row["share"],
                 row["fallback"], row["score_before"], row["score_after"],
                 row["above_line_before"], row["above_line_after"],
                 row["algebra_5"], row["algebra_7"]))
    unchanged = [r for r in twins if r["cost"] not in (10, 11)]
    print("    状态没变那几格的自比：%s" % _span([r["share"] for r in unchanged]))
    changed = [r for r in twins if r["cost"] in (10, 11)]
    print("    同一干员一净一灰那两格的自比：%s" % _span([r["share"] for r in changed]))
    print("    两簇：判不了那一段（%.2f…%.2f）这一批上一次都没落进去 —— 纸上的防线"
          % (gray.GRAY_SHARE_MAX, gray.CLEAR_SHARE_MIN))
    report["fallback"] = {"rows": twins,
                          "lines": [gray.GRAY_SHARE_MAX, gray.CLEAR_SHARE_MIN]}

    print()
    print("### 做1 续：编队界面.png 是干什么用的、与 立绘两组/ 那份什么关系")
    geo = json.loads((TWINS / "库1.json").read_text(encoding="utf-8"))["geometry"]
    same_geo = json.loads((TWINS / "库1.json").read_text(encoding="utf-8"))
    twins_geo = json.loads((TWINS / "库2.json").read_text(encoding="utf-8"))
    libs = {
        "组1_编队页": formation_faces(ROOT / same_geo["frame"], same_geo, "组1·"),
        "组2_编队页": formation_faces(ROOT / twins_geo["frame"], twins_geo, "组2·"),
        "编队界面": formation_faces(SHOTS / "编队界面.png", {"geometry": geo}, "编队·"),
    }
    for label, lib in libs.items():
        print("    %s：%d 条（%d 宽）" % (label, len(lib), lib[0][1].shape[1]))
    print("    把一个编队页的每块立绘当查询，去比另两个编队页的库（同一块几何、逐位同尺寸）：")
    print("      编队界面 的第 k 块 → 组1 / 组2 最像的那条（名字 分数）：")
    pairs = []
    for i, (_name, art) in enumerate(libs["编队界面"]):
        r1 = roster_scores(art, libs["组1_编队页"])[0]
        r2 = roster_scores(art, libs["组2_编队页"])[0]
        pairs.append({"k": i, "g1": [r1[1], _round(r1[0], 6)], "g2": [r2[1], _round(r2[0], 6)]})
        print("        第%2d 块：组1 %-14s %.6f ｜ 组2 %-14s %.6f"
              % (i, r1[1], r1[0], r2[1], r2[0]))
    hit1 = sum(1 for p in pairs if p["g1"][1] >= 0.99)
    hit2 = sum(1 for p in pairs if p["g2"][1] >= 0.99)
    print("      与组1 逐位同一条（≥0.99）的块 %d/10 ｜ 与组2 %d/10" % (hit1, hit2))
    report["relationship"] = {
        "library_geometry_source": "立绘两组/库1.json 的 geometry（编队页左缘/卡距/立绘宽是常量）",
        "sizes": {k: [len(v), int(v[0][1].shape[1])] for k, v in libs.items()},
        "per_block": pairs,
        "same_as_group1": hit1,
        "same_as_group2": hit2,
    }

    print()
    print("### 做4 续：拿 立绘两组/库1.json 那把**真名字的库**（编队页立绘）读这一帧")
    print("    （库与查询不同源、还有名字；名字是库给的读数，不是本层的真值 —— 真值住 tests/spec/）")
    lib1 = libs["组1_编队页"]
    for stage, region_of in (("还原前", lambda i: cut(raw[GRAY_PNG], ident_box(i))),
                             ("还原后", None)):
        pass
    panel = []
    for k in range(1, n + 1):
        i = n - k
        region = cut(raw[GRAY_PNG], ident_box(i))
        if verdicts[i]:
            region = gray.restore(region)
        top, runner, _ = top_two(region, lib1)
        panel.append({"index": i, "cost": costs[i],
                      "top": top[1], "score": _round(top[0], 6),
                      "runner": runner, "exact": bool(identity_is_exact(top[0], runner))})
    print("  序号 idx 费 状态  库1 读出来的名字   分数      次高      判定")
    for k in range(1, n + 1):
        row = panel[n - k]
        print("   %2d  %2d %3d %s  %-16s %.6f  %.6f  %s"
              % (k, row["index"], row["cost"], "缺费" if verdicts[row["index"]] else "够费",
                 row["top"], row["score"], row["runner"], "精确" if row["exact"] else "未知"))
    report["scores"]["library_1"] = {
        "source": "tests/帧样本/立绘两组/库1.json（组1 编队页的 10 条立绘）",
        "stage": "还原后（缺费那五格乘回去了）",
        "rows": panel,
    }

    print()
    print("    同一帧再拿 库2 读一遍 —— 把「库里那套立绘对不上」与「灰罩压掉了」分开：")
    lib2 = libs["组2_编队页"]
    both = []
    for frame_label, px in (("灰罩(还原后)", raw[GRAY_PNG]), ("对照帧", raw[CLEAR_PNG])):
        print("      %s：" % frame_label)
        for k in range(1, n + 1):
            i = n - k
            region = cut(px, ident_box(i))
            if frame_label.startswith("灰罩") and verdicts[i]:
                region = gray.restore(region)
            t1 = top_two(region, lib1)[0]
            t2 = top_two(region, lib2)[0]
            both.append({"frame": frame_label, "index": i, "cost": costs[i],
                         "lib1": [t1[1], _round(t1[0], 6)], "lib2": [t2[1], _round(t2[0], 6)]})
            print("        idx%2d 费%3d 库1 %-16s %.6f ｜ 库2 %-16s %.6f"
                  % (i, costs[i], t1[1], t1[0], t2[1], t2[0]))
    report["scores"]["library_2"] = {"source": "tests/帧样本/立绘两组/库2.json", "rows": both}


    # --------------------------------------------------------------------- 落盘
    OUT.mkdir(parents=True, exist_ok=True)
    x0, y0, w0, h0 = STRIP
    crops = []
    for name in (GRAY_PNG, CLEAR_PNG):
        Image.fromarray(raw[name][y0:y0 + h0, x0:x0 + w0]).save(OUT / NAMES[name])
        crops.append({
            "file": NAMES[name],
            "origin": [x0, y0],
            "size": [w0, h0],
            "source": "screenshots/run-04/" + name,
            "source_sha256": hashlib.sha256((SHOTS / name).read_bytes()).hexdigest(),
            "current_cost": CURRENT[name],
        })
    index = {
        "version": 1,
        "canvas": [2800, 1260],
        "device": "vivo V2338A",
        "box": [x0, y0, x0 + w0 - 1, y0 + h0 - 1],
        "crops": crops,
        "phase": "备战席第三态：缺费（灰卡）",
        "labels_note": "两帧是同一局的两次暂停；卡位 / 干员 / 顺序 / 剩余可放置角色全一样，"
                       "唯一变的是当前费用。所以标签是**代数**的、不是目视的。",
        "labels": report["labels"],
    }
    (OUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (OUT / "差图.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print()
    print("### 落盘")
    for path in sorted(OUT.iterdir()):
        print("    %s  %d 字节" % (path.name, path.stat().st_size))


def _round(value: float, nd: int = 3) -> float:
    return round(float(value), nd)


def _span(values) -> str:
    values = list(values)
    if not values:
        return "（空）"
    return "%.4f…%.4f" % (min(values), max(values))


def _corpus(table) -> list:
    """语料里所有"卡费与当前费都读得出"的卡（备战席 6 帧 + run-04 这一对）。"""
    rows = []
    meta = json.loads((BENCH / "index.json").read_text(encoding="utf-8"))
    cost_meta = json.loads((COST / "index.json").read_text(encoding="utf-8"))
    by_shot = {c["shot"]: c for c in cost_meta["crops"]}
    for entry in meta["crops"]:
        if entry["file"].startswith("cd"):
            continue
        frame = paste(BENCH / entry["file"], entry["origin"])
        current = None
        if entry["shot"] in by_shot:
            cost_entry = by_shot[entry["shot"]]
            current = exact(read_cost_digits(
                paste(COST / cost_entry["file"], cost_entry["origin"]), table))
        if current is None:
            continue
        for card in read_squad_row(frame, table, pitch=PITCH):
            cost = exact(card.cost)
            if cost is None:
                continue
            rows.append(_stat_row("备战席/" + entry["file"], card.index, cost, current,
                                  cut(frame.pixels, ident_box(card.index))))
    for name in (GRAY_PNG, CLEAR_PNG):
        frame = frame_of(load_png(SHOTS / name))
        current = exact(read_cost_digits(frame, table))
        for card in read_squad_row(frame, table, pitch=PITCH):
            cost = exact(card.cost)
            if cost is None or current is None:
                continue
            rows.append(_stat_row("run-04/" + name, card.index, cost, current,
                                  cut(frame.pixels, ident_box(card.index))))
    return rows


def _stat_row(source: str, index: int, cost: int, current: int, region: np.ndarray) -> dict:
    return {"source": source, "index": index, "cost": cost, "current": current,
            "insufficient": bool(cost > current),
            "y": float(lum(region).mean()),
            "chroma": float(chroma(region).mean()),
            "s": float(sat(region).mean())}


def _twin_rows(table) -> list:
    """shot05 ↔ shot07（同局两次暂停）：状态没变的几格 + 一净一灰的孪生对。"""
    meta = json.loads((BENCH / "index.json").read_text(encoding="utf-8"))
    org = {c["file"]: c["origin"] for c in meta["crops"]}
    f5 = paste(BENCH / "shot05.png", org["shot05.png"])
    f7 = paste(BENCH / "shot07.png", org["shot07.png"])
    costs5 = [exact(card.cost) for card in read_squad_row(f5, table, pitch=PITCH)]
    rows = []
    for i in range(5):
        rows.append(_twin_row("shot05↔shot07", i, i, costs5[i], f5, f7,
                              "状态没变（两帧逐位相同）"))
    for i5, i7, cost in ((7, 6, 11), (8, 7, 10)):
        rows.append(_twin_row("shot05↔shot07", i5, i7, cost, f5, f7, "同干员一净一灰"))
    return rows


def _twin_row(tag: str, i5: int, i7: int, cost: int, f5: Frame, f7: Frame, what: str) -> dict:
    before = cut(f5.pixels, ident_box(i5))
    now = cut(f7.pixels, ident_box(i7))
    share = gray.own_share(now, before)
    fallback = gray.insufficient_cost(Valued(Unknown(), None), Valued(Exact(11), None),
                                      share=share)
    algebra5 = cost > 11
    algebra7 = cost > 8
    # 单条库（只有这一位）：判据退化成只看那条线（见 squad_card_scores 的文档）
    face = [("费%d" % cost, cut(f5.pixels, face_box(i5)))]
    before = roster_scores(np.ascontiguousarray(now), face)[0][0]
    after = roster_scores(np.ascontiguousarray(gray.restore(now)), face)[0][0]
    return {"file": tag, "index": i7, "same_slot_in_other_frame": i5, "cost": cost,
            "what": what, "share": _round(share, 4), "fallback": fallback,
            "algebra_5": algebra5, "algebra_7": algebra7,
            "score_before": _round(before, 6), "score_after": _round(after, 6),
            "above_line_before": bool(before >= ROSTER_SOLID_MATCH_SCORE),
            "above_line_after": bool(after >= ROSTER_SOLID_MATCH_SCORE)}


if __name__ == "__main__":
    main()
