r"""A2c：红罩反解之后的「灰」，能不能靠像素变换救回来 —— **只换预处理**。

    .venv\Scripts\python.exe scripts\bench_redcap_preprocess.py

## 这一单问什么

反解出来的卡看着发灰，而像素匹配比的是绝对像素。所以把**参考图**（库立绘）与／或
**查询图**（反解之后那块取区）先做一次像素变换，看分数会不会上去。**没有任何一种有用
也是答案。**

## 这一单不换什么

**不换尺子、不换取区、不动线上那条链**：`read_squad_identities` /
`find_squad_card` / `squad_card_scores` / `roster_scores` /
`squad_card_centres` 一个字节没动，红罩卡仍然是未知（用户裁定）。读数的统计量还是
ark.engine.vision.margin.MaskedRuler —— 与 roster_scores 同一个（逐通道减均值的零均值
归一化相关），只是收「一块形状 + 一个掩码」。这个脚本**不接线上**。

## 样本

tests/帧样本/预处理/：data/precheck/cd/（一局真机、每 3 秒一张、cd01..cd60 六十帧）里
**所有**满足 pair_ok 的「同一张卡一次罩着、一次干净」配对 —— **44 对 · 39 张卡片裁剪 ·
32 个源帧**。裁剪只裁**那一张卡**（204x214 原生像素、一个都不缩）。

## 三件必须写在读数旁边的事

- **半径 62**：排掉倒计时圆盘那一圈。60 太小（规划者量到：取区原点在 x 上 ±1 px 时那条
  读数摆动 0.274、判定翻 2/4），62 摆 0.021、64 摆 0.013；再大只是白扔像素。**所有行用
  同一个半径**。倒计时数字在半径 45 以内，早就被排掉了，不为它扩大掩码。
- **取区原点**：**这一帧像素读出来的卡心**（squad_card_centres）减 102（卡宽/2）——
  不是常量网格。实测同一个卡位在 39 张裁剪上是同一个数；本批实测卡心与常量网格**最大
  差 1 px**（卡位 4/6/8/9 各 -1，0/1 为 0；本批没有卡位 10）。配对两端这个原点是同一个
  数（44/44），所以两块的像素是对齐的。
- **模型**：fit_redcap 只吃 取材.json 里那两个标定对（cd14#1 -> cd15#1、cd22#9 -> cd23#9）；
  **这一单没有重新拟合**。

## 标尺行（"三个通道乘同一个倍数、加同一个偏置"那一行）

那条尺子先减均值、再按起伏大小归一化，所以**整体仿射在数学上会被约掉**。规划者要的是
"这一行必须一位不变"。**实测：做不到** —— 而这些行只是**同一量级的样本**，没有哪一行是
这条链的"下限"（下面最后一行比其中一行还大）。把它拆开：

| 行 | max 差 |
| --- | --- |
| 只查询图 +16（不夹，掩码内是 3…210） | **1.314e-4** |
| 只参考图 +16（库立绘里有 151682 个纯白像素被夹） | ~2.0e-3 |
| 两边 0.8x+25（**一个像素都不夹**，只剩取整） | ~8.9e-4 |
| 填零对照 vs 填零+整数平移（整数平移、不夹不取整） | ~2.5e-4 |

**最小的是"只查询图"那一行（1.314e-4），比"填零+整数平移"（2.487e-4）还小** —— 所以没有
哪一行是下限，它们都是同一个量级的样本。这个量级的来源是 **cv2.resize**：把整块画面加一个
整数常数再放大、与先放大再加那个常数，差 **±1 个灰度级**（1.05…1.15 十一档全部如此；这份
读数落在"预处理.json" 的 invariance 里）。尺子是**先 resize 再减均值归一化**的，那 ±1 就
进了分子分母。**不是实验里有别的东西在动**：手写一份同公式的 float32 实现复现出同样的数
（差 1e-7），差别整个来自那次 resize。

所以"一位不变"这条门在**不换尺子**的前提下做不到。所有行都经过同一次 resize，所以下面
每一行**并排比**仍然成立（这个量级 ~1e-4…2e-3 远小于那些行自己动的 0.02…0.35）。

## 三条对照：红罩的账与掩码的账要分开

同一批 44 对、同一条链（MaskedRuler，同一个原点、同一个半径 62），只把**喂进去的那一端**
换掉，再拿**线上那条链**（roster_scores，整个身份区 10000 px、不掩码）当第三条：

| 对照 | 取区 | 链 |
| --- | --- | --- |
| 罩着·掩码 | 反解之后那块取区，552 px | MaskedRuler |
| 干净·掩码 | 干净那一端同一块取区，552 px | MaskedRuler |
| 干净·不掩码 | 干净那一端的整个身份区，10000 px | roster_scores |

**这三条一分开，"红罩的账"与"掩码的账"就归位了**：红罩对细节几乎不留痕（干净端掩码与
罩着端掩码相差极小），真正扔掉信息的是**掩码本身**（552 / 10000 = 5.5%）—— 它在相关之前
就扔掉了，像素变换补不回来。这一单"没有一种预处理有用"照留，**理由换成这条**。
**这不是"红罩卡认不出来"**：真值那一半读得出来，而且两端都过线。

## 罩子不是一块均匀的滤镜

同一张卡上，**不同取区的有效压缩不一样**（逐通道仿射 `capped = a * clean + b` 的斜率）：
身份区约 **0.15**、整卡立绘区（模型自己那个）**0.205**、顶上那条里的**职业图标区
0.43…0.61**。三者各自的取区与出处落在 `预处理.json` 的 `compression` 里。**它对分数没有
影响** —— 那把尺子对逐通道仿射天生免疫（就是上面那条标尺行量的事）。这一栏是记事实。

## 它落的那一份

tests/帧样本/预处理/预处理.json **逐字节可复现**：没有时间戳、没有绝对路径、没有随机量。
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.engine.vision.frame import frame_from_raw  # noqa: E402
from ark.engine.vision.margin import MaskedRuler, free_mask  # noqa: E402
from ark.engine.vision.redcap import fit_redcap, undo_redcap  # noqa: E402
from ark.engine.vision.roster import (  # noqa: E402
    IDENTITY_BOX_HEIGHT,
    IDENTITY_BOX_WIDTH,
    IDENTITY_BOX_X,
    IDENTITY_BOX_Y,
    IDENTITY_SCALES,
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
    roster_scores,
)
from ark.offline.assets_gen.cardface import load_cardfaces  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402

BATCH = REPO / "tests" / "帧样本" / "预处理"
HARVEST = BATCH / "取材.json"
OUT = BATCH / "预处理.json"

BOX = (IDENTITY_BOX_X, IDENTITY_BOX_Y, IDENTITY_BOX_WIDTH, IDENTITY_BOX_HEIGHT)
"""身份区（卡内 (52, 76, 100, 100)）—— 与线上那条链**同一个框**、同一层常量。"""

CARD = (214, 204, 3)
"""一整张备战席卡。"""

RADIUS = 62
"""排掉倒计时圆盘的半径。**所有行用这一个数**，不许一处一个。

它比 bench.RING_OUTER（60，环带外缘）大 2 px：60 太贴着环，取区原点抖 1 px 读数就
摆动 0.274、判定翻 2/4；62 摆 0.021、64 摆 0.013。再大只是白扔像素。
"""

AFFINE_LIFT = (1.0, 16.0)
"""标尺行：三个通道乘同一个倍数、加同一个偏置（**整体抬亮度**）。"""

AFFINE_COMPRESS = (0.8, 25.0)
"""另一条标尺行：**整体压对比**（0.8*x + 25）。它**不夹任何一个像素**（0->25、255->229），
所以它的摆动只含 uint8 取整那一份；上面那条抬亮度会夹住纯白。两条并排，夹住与取整各自
值多少就分得开。"""

FILL = 0.0
FILL_SHIFT = 16
"""取证行用的两个数：掩码外填成 0.0，再整体加 16（整数、不夹）。"""


def _affine(image: np.ndarray, a: float, b: float) -> np.ndarray:
    """逐像素 a*x + b，夹回 0…255（**图像变换必然要夹**：写回 uint8 就只有这条路）。"""
    return np.clip(np.rint(image.astype(np.float64) * a + b), 0.0, 255.0).astype(np.uint8)


def _shift(image: np.ndarray, b: int) -> np.ndarray:
    """整数平移，**不夹** —— 只要有一个像素会越界就响亮报错（取证行要的就是"恰好"）。"""
    lifted = image.astype(np.int16) + int(b)
    if lifted.min() < 0 or lifted.max() > 255:
        raise ValueError(f"整数平移 {b} 会把像素推到 0…255 之外：这条取证行要的是不夹")
    return lifted.astype(np.uint8)


def _saturate(image: np.ndarray, lam: float) -> np.ndarray:
    """降饱和：灰 + lam * (原色 - 灰)，灰是**该像素三个通道的算术平均**。

    lam = 1 一位不变（不动）；lam = 0 就是全灰度（三个通道都变成那个平均值）。
    """
    as_float = image.astype(np.float64)
    grey = as_float.mean(axis=2, keepdims=True)
    return np.clip(np.rint(grey + lam * (as_float - grey)), 0.0, 255.0).astype(np.uint8)


def _fill(image: np.ndarray, mask: np.ndarray, value: float) -> np.ndarray:
    """把**掩码外**的像素填成一个常数。

    尺子本来也会在 resize **之后**把它们乘 0；这一行只是**提前**填掉，好让「掩码里那
    几个像素」成为整块画面里唯一在动的东西。它是取证用的，不是一条方法。
    """
    return np.where(mask[:, :, None], image, np.uint8(round(value)))


def _transform(image: np.ndarray, spec: dict | None, mask: np.ndarray | None) -> np.ndarray:
    """一条变换。spec is None 就是「什么都不做」。"""
    if spec is None:
        return image
    kind = spec["kind"]
    if kind == "affine":
        return _affine(image, float(spec["a"]), float(spec["b"]))
    if kind == "sat":
        return _saturate(image, float(spec["lam"]))
    if kind == "grey":
        return _saturate(image, 0.0)
    if kind == "fill":
        return _fill(image, mask, float(spec["fill"]))
    if kind == "fill_shift":
        return _shift(_fill(image, mask, float(spec["fill"])), int(spec["b"]))
    raise ValueError(f"没见过的变换 {kind!r}")


def _chroma_rms(pixels: np.ndarray) -> float:
    """一块 (N, 3) 的**色度**起伏：逐像素减掉它自己三个通道的平均，再取 rms。"""
    as_float = pixels.astype(np.float64)
    centred = as_float - as_float.mean(axis=1, keepdims=True)
    return float(np.sqrt((centred ** 2).mean()))


def _per_channel_slope(capped: np.ndarray, clean: np.ndarray, mask: np.ndarray) -> list[float]:
    """逐通道最小二乘 `capped = a * clean + b` 的斜率 `a` —— **只在这块掩码上**。

    与 A2b 职业图标那一栏同一个估计量（那边不掩码；这里要掩掉逐帧在变的倒计时图形）。
    """
    capped_pixels = capped[mask].astype(np.float64)
    clean_pixels = clean[mask].astype(np.float64)
    out: list[float] = []
    for channel in range(3):
        x = clean_pixels[:, channel] - clean_pixels[:, channel].mean()
        y = capped_pixels[:, channel] - capped_pixels[:, channel].mean()
        out.append(float((x * y).sum() / (x * x).sum()))
    return out


def _per_channel_std_ratio(
    capped: np.ndarray, clean: np.ndarray, mask: np.ndarray
) -> list[float]:
    """逐通道 `std(capped) / std(clean)`（同一块掩码上）—— 同一个「压缩」的另一种量法。"""
    capped_pixels = capped[mask].astype(np.float64)
    clean_pixels = clean[mask].astype(np.float64)
    return [
        float(capped_pixels[:, channel].std(ddof=1) / clean_pixels[:, channel].std(ddof=1))
        for channel in range(3)
    ]


def _rows(levels: dict) -> list[dict]:
    """这一单要并排的那些行 —— 顺序就是报告里的顺序。"""
    lam = float(levels["lambda_star"])
    slope = float(levels["slope"])
    why_star = ("lam* = 中位( S^2 / (S^2 + N^2) )，S = 干净卡取区的色度 rms、"
                "N = 反解后误差的色度 rms —— 该保留多少颜色那个 Wiener 收缩比")
    why_slope = "lam = 拟合出来的罩子斜率 —— 颜色在罩子底下还剩多少那个数"
    why_zero = "lam = 0，全灰：只比明暗"
    lift = {"kind": "affine", "a": AFFINE_LIFT[0], "b": AFFINE_LIFT[1]}
    press = {"kind": "affine", "a": AFFINE_COMPRESS[0], "b": AFFINE_COMPRESS[1]}
    fill_base = {"kind": "fill", "fill": FILL}
    fill_shift = {"kind": "fill_shift", "fill": FILL, "b": FILL_SHIFT}

    def sat(value: float) -> dict:
        return {"kind": "sat", "lam": float(value)}

    grey = {"kind": "grey"}
    return [
        {"name": "对照", "query": None, "library": None,
         "why": "什么都不做：反解之后一个像素不改。"},
        {"name": "标尺-整体仿射（两边）", "query": lift, "library": lift,
         "why": "三个通道乘同一个倍数 1.0、加同一个偏置 +16；两边都做。期望一位不变。"},
        {"name": "标尺-整体仿射（只参考图）", "query": None, "library": lift,
         "why": "同上，只做库立绘那一侧 —— 把动的来源是哪一侧分开。"},
        {"name": "标尺-整体仿射（只查询图）", "query": lift, "library": None,
         "why": "同上，只做反解那块取区那一侧。"},
        {"name": "标尺-整体压对比（两边）", "query": press, "library": press,
         "why": "三个通道乘同一个倍数 0.8、加同一个偏置 +25；这一条**一个像素都不夹**，"
                "所以它的摆动只含取整那一份。"},
        {"name": "取证-填零对照（只查询图）", "query": fill_base, "library": None,
         "why": "取证行：掩码外先填成一个常数、什么都不做。它是下一条行的对照。"},
        {"name": "取证-填零+整数平移（只查询图）", "query": fill_shift, "library": None,
         "compare_to": 5,
         "why": "取证行：掩码外填成常数、再整体加 16（整数、不夹）。那块画面**恰好**是"
                "上一条 + 16，所以这一条**必须一位不变**。"},
        {"name": "降饱和-只参考图 lam=lam*", "query": None, "library": sat(lam),
         "why": why_star + "；只降库立绘那一侧。"},
        {"name": "降饱和-只参考图 lam=slope", "query": None, "library": sat(slope),
         "why": why_slope + "；只降库立绘那一侧。"},
        {"name": "降饱和-只参考图 lam=0", "query": None, "library": sat(0.0),
         "why": why_zero + "；只降库立绘那一侧。"},
        {"name": "降饱和-只查询图 lam=lam*", "query": sat(lam), "library": None,
         "why": why_star + "；只降反解那块取区那一侧。"},
        {"name": "降饱和-只查询图 lam=slope", "query": sat(slope), "library": None,
         "why": why_slope + "；只降反解那块取区那一侧。"},
        {"name": "降饱和-只查询图 lam=0", "query": sat(0.0), "library": None,
         "why": why_zero + "；只降反解那块取区那一侧。"},
        {"name": "降饱和-两边 lam=lam*", "query": sat(lam), "library": sat(lam),
         "why": why_star + "；两边都降。"},
        {"name": "降饱和-两边 lam=slope", "query": sat(slope), "library": sat(slope),
         "why": why_slope + "；两边都降。"},
        {"name": "降饱和-两边 lam=0", "query": sat(0.0), "library": sat(0.0),
         "why": why_zero + "；两边都降。"},
        {"name": "全灰度", "query": grey, "library": grey,
         "why": "两边都转灰度：灰 = 该像素三个通道的算术平均（等权，不是亮度加权）；"
                "**不另做对比度归一化** —— 那把尺子自己就减均值、除能量，整体仿射整个"
                "会被约掉（就是上面那条标尺行量的事）。丢掉的只有颜色。"},
    ]


def _passes(score: float, margin: float) -> bool:
    """判定 —— **读者按线算**，不落盘。

    score >= ROSTER_SOLID_MATCH_SCORE 而且 margin >= ROSTER_AMBIGUITY_MARGIN。
    落一个布尔下来，线一挪就有一份旧值躺在文件里，而它读起来跟新值一模一样。
    """
    return score >= ROSTER_SOLID_MATCH_SCORE and margin >= ROSTER_AMBIGUITY_MARGIN


def _count_passes(entries) -> int:
    """一列读数里有几块过线 —— 只给控制台与说明文字用，不进落盘。"""
    return sum(1 for entry in entries if _passes(entry["score"], entry["margin"]))


def main(argv: list[str]) -> int:
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 0
    dry = "--dry" in argv

    batch = load_batch(BATCH)
    harvest = json.loads(HARVEST.read_text(encoding="utf-8"))
    pairs = list(harvest["pairs"])
    if limit:
        order: list[str] = []
        for pair in pairs:
            if pair["capped"]["key"] not in order:
                order.append(pair["capped"]["key"])
        keep = set(order[:limit]) | {
            pair["capped"]["key"] for pair in pairs if pair["set"] == "calibration"
        }
        pairs = [pair for pair in pairs if pair["capped"]["key"] in keep]

    blocks: dict[str, np.ndarray] = {}
    for case in batch.cases:
        frame = frame_from_raw(case.raw())
        x0, y0 = int(case.rect.x), int(case.rect.y)
        blocks[case.name] = np.ascontiguousarray(
            frame.pixels[y0 : y0 + CARD[0], x0 : x0 + CARD[1]]
        )

    def block(file_name: str) -> np.ndarray:
        return blocks[Path(file_name).stem]

    crop_hash = {
        name: hashlib.sha256(image.tobytes()).hexdigest()[:12]
        for name, image in blocks.items()
    }
    capped_image_of = [crop_hash[Path(pair["capped"]["file"]).stem] for pair in pairs]
    clean_image_of = [crop_hash[Path(pair["clean"]["file"]).stem] for pair in pairs]

    def identity_hash(file_name: str) -> str:
        """另一个去重键：**那把尺子真正看到的那块身份区画面**（卡内 BOX）。"""
        card = blocks[Path(file_name).stem]
        return hashlib.sha256(
            np.ascontiguousarray(
                card[BOX[1] : BOX[1] + BOX[3], BOX[0] : BOX[0] + BOX[2]]
            ).tobytes()
        ).hexdigest()[:12]

    capped_identity_of = [identity_hash(pair["capped"]["file"]) for pair in pairs]
    clean_identity_of = [identity_hash(pair["clean"]["file"]) for pair in pairs]
    combo_of = [
        capped + "/" + clean for capped, clean in zip(capped_image_of, clean_image_of)
    ]
    distinct_combos = len(set(combo_of))

    def image_groups(keys: list[str], names: list[str]) -> list[dict]:
        """互不相同的画面各算一块 —— 同一个键就是同一时段里逐字节相同的那一张。"""
        seen: dict[str, dict] = {}
        for key, name in zip(keys, names):
            entry = seen.setdefault(key, {"key": key, "example": name, "pairs": 0})
            entry["pairs"] += 1
        return list(seen.values())

    def distinct_stats(entries: list[dict], keys: list[str]) -> tuple[dict, int]:
        """按互不相同的画面算：几块 · 余量中位（每块取第一次出现的读数）。

        **过线数不落盘**（返回值里只有块数与中位余量）；第二个返回值只给控制台那一行。"""
        picked: dict[str, dict] = {}
        for entry, key in zip(entries, keys):
            picked.setdefault(key, entry)
        margins = [entry["margin"] for entry in picked.values()]
        return {
            "distinct_images": len(picked),
            "median_margin_distinct_images": round(statistics.median(margins), 6),
        }, _count_passes(picked.values())

    capped_groups = image_groups(
        capped_image_of, [Path(pair["capped"]["file"]).stem for pair in pairs]
    )
    clean_groups = image_groups(
        clean_image_of, [Path(pair["clean"]["file"]).stem for pair in pairs]
    )

    def box(image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            image[BOX[1] : BOX[1] + BOX[3], BOX[0] : BOX[0] + BOX[2]]
        )

    calibration = [pair for pair in pairs if pair["set"] == "calibration"]
    if len(calibration) != 2:
        raise ValueError(f"标定对要是 2 个（红罩那批的那两个），收到 {len(calibration)}")
    model = fit_redcap(
        [
            (block(pair["capped"]["file"]), block(pair["clean"]["file"]))
            for pair in calibration
        ]
    )
    mask = free_mask(BOX, radius=RADIUS)
    faces = load_cardfaces()

    queries = [box(undo_redcap(block(pair["capped"]["file"]), model)) for pair in pairs]
    cleans = [box(block(pair["clean"]["file"])) for pair in pairs]
    capped_boxes = [box(block(pair["capped"]["file"])) for pair in pairs]

    invariance = {
        "question": "整体仿射在数学上会被尺子约掉；这一栏量它到底动不动。",
        "probe": (
            "把第一块取区的掩码外填 0，整块加 16（整数、不夹），再和填 0 那块一起喂"
            "cv2.resize(INTER_LINEAR) —— 看放大之后是否仍然差 16。"
        ),
        "resize_deviation_u8": [],
    }
    probe_filled = np.where(mask[:, :, None], queries[0], np.uint8(0))
    probe_lifted = (probe_filled.astype(np.int16) + FILL_SHIFT).astype(np.uint8)
    for scale in IDENTITY_SCALES:
        side = int(round(queries[0].shape[0] * scale))
        low = cv2.resize(probe_filled, (side, side), interpolation=cv2.INTER_LINEAR)
        high = cv2.resize(probe_lifted, (side, side), interpolation=cv2.INTER_LINEAR)
        invariance["resize_deviation_u8"].append(
            int(np.abs(high.astype(np.int16) - low.astype(np.int16) - FILL_SHIFT).max())
        )
    invariance["resize_shift_exact"] = all(value == 0 for value in invariance["resize_deviation_u8"])
    invariance["library_white_pixels"] = int(sum(int((art == 255).sum()) for _, art in faces))
    invariance["verdict"] = (
        "resize 不是平移等变的（每档 ±1 个灰度级），所以不换尺子就做不到一位不变。"
        "这个量级的样本从 1.314e-4 到 2.055e-3 —— **没有哪一行是下限**"
        "（「标尺-整体仿射（只查询图）」比「取证-填零+整数平移」还小）。"
        "所有行都经过同一次 resize，并排比仍然成立。"
    )

    errors = []
    signals: list[float] = []
    noises: list[float] = []
    for pair, query, clean, capped in zip(pairs, queries, cleans, capped_boxes):
        difference = query[mask].astype(np.float64) - clean[mask].astype(np.float64)
        rms = np.sqrt((difference ** 2).mean(axis=0))
        std = clean[mask].astype(np.float64).std(axis=0)
        signal = _chroma_rms(clean[mask])
        noise = _chroma_rms(difference)
        signals.append(signal)
        noises.append(noise)
        errors.append(
            {
                "key": pair["capped"]["key"],
                "clean_key": pair["clean"]["key"],
                "slot": pair["capped"]["index"],
                "rms": [round(float(value), 6) for value in rms],
                "std": [round(float(value), 6) for value in std],
                "ratio": [round(float(rms[i] / std[i]), 6) for i in range(3)],
                "std_solved": [
                    round(float(value), 6)
                    for value in query[mask].astype(np.float64).std(axis=0)
                ],
                "std_capped": [
                    round(float(value), 6)
                    for value in capped[mask].astype(np.float64).std(axis=0)
                ],
                "chroma_signal": round(signal, 6),
                "chroma_noise": round(noise, 6),
                "chroma_snr": round(signal / noise, 6),
            }
        )

    lambdas = [
        entry["chroma_signal"] ** 2
        / (entry["chroma_signal"] ** 2 + entry["chroma_noise"] ** 2)
        for entry in errors
    ]
    levels = {
        "slope": float(model.slope),
        "lambda_star": statistics.median(lambdas),
        "lambda_star_min": min(lambdas),
        "lambda_star_max": max(lambdas),
        "chroma_signal_median": statistics.median(signals),
        "chroma_noise_median": statistics.median(noises),
    }
    slope_values: list[float] = []
    ratio_values: list[float] = []
    for capped, clean in zip(capped_boxes, cleans):
        slope_values.extend(_per_channel_slope(capped, clean, mask))
        ratio_values.extend(_per_channel_std_ratio(capped, clean, mask))

    margin_report = REPO / "tests" / "帧样本" / "余量" / "余量.json"
    if not margin_report.is_file():
        raise FileNotFoundError(
            f"读不到 {margin_report} —— 职业图标区那一栏（A2b 乙组 5）要从它里面取，"
            "少了就不能往这一份里抄一个转手的数"
        )
    icon_slopes = [
        value
        for evidence in json.loads(margin_report.read_text(encoding="utf-8"))["class_icon"][
            "evidence"
        ]
        for value in evidence["slope"]
    ]
    compression = {
        "question": "罩子不是一块均匀的滤镜：同一张卡上，不同取区的有效压缩不一样。",
        "estimator": (
            "逐通道最小二乘 capped = a * clean + b 的斜率 a（与 A2b 职业图标那一栏同一个"
            "估计量），外加同一个「压缩」的另一种量法 std(capped)/std(clean)。"
        ),
        "regions": [
            {
                "name": "整卡立绘区",
                "region": (
                    "整张卡 · redcap._countdown_mask（半径 60 的倒计时圆盘 + 顶上 47 px），"
                    "每对 22752 px"
                ),
                "pairs": "红罩那批 2 个标定对",
                "values": {"slope": model.slope},
                "source": "本份 model.slope（fit_redcap）",
            },
            {
                "name": "身份区（这把尺子的取区）",
                "region": "卡内 (52, 76, 100, 100) 里排掉倒计时圆盘之后那 552 px",
                "pairs": "本批 44 对 · 逐通道 132 个值",
                "values": {
                    "slope_median": round(statistics.median(slope_values), 6),
                    "slope_min": round(min(slope_values), 6),
                    "slope_max": round(max(slope_values), 6),
                    "std_ratio_median": round(statistics.median(ratio_values), 6),
                    "std_ratio_min": round(min(ratio_values), 6),
                    "std_ratio_max": round(max(ratio_values), 6),
                },
                "source": "本脚本算的（scripts/bench_redcap_preprocess.py）",
            },
            {
                "name": "职业图标区",
                "region": "卡内 (44, 0, 62, 47) —— 顶上那一条里的职业图标块，2914 px，不掩码",
                "pairs": "红罩那批 4 对 · 逐通道 12 个值",
                "values": {
                    "slope_min": round(min(icon_slopes), 6),
                    "slope_max": round(max(icon_slopes), 6),
                    "slope": [round(value, 4) for value in icon_slopes],
                },
                "source": (
                    "tests/帧样本/余量/余量.json 的 class_icon.evidence[].slope（A2b 乙组 5）"
                ),
            },
        ],
        "note": (
            "它对分数没有影响：那把尺子对逐通道仿射天生免疫（就是「标尺-整体仿射」那几行量"
            "的事）。这一栏是记事实，不是一条方法。三个数各自的取区不一样，别互相替代。"
        ),
    }

    rows_spec = _rows(levels)

    rulers: dict[str, MaskedRuler] = {}

    def ruler_for(spec: dict | None) -> MaskedRuler:
        key = json.dumps(spec, sort_keys=True)
        if key not in rulers:
            arts = faces if spec is None else tuple(
                (name, _transform(art, spec, None)) for name, art in faces
            )
            rulers[key] = MaskedRuler(mask, arts)
        return rulers[key]

    cache: dict[tuple[str, bytes], tuple[float, str, float]] = {}

    def top_two(ranked) -> tuple[float, str, float]:
        """一列降序的读数 → （最高分, 名字, 次高分）。判据的两半就是前两个。"""
        runner = ranked[1][0] if len(ranked) > 1 else -2.0
        return float(ranked[0][0]), ranked[0][1], float(runner)

    def score(name: str, ruler: MaskedRuler, region: np.ndarray) -> tuple[float, str, float]:
        key = (name, region.tobytes())
        if key not in cache:
            cache[key] = top_two(ruler.scores(region))
        return cache[key]

    plain_ruler = ruler_for(None)
    controls: list[dict] = []
    control_console: list[dict] = []
    """控制台那一行要的过线数 —— **不落盘**，所以单独攒一份。"""

    def add_control(
        name: str,
        regions: list[np.ndarray],
        keys: list[str],
        identity_keys: list[str],
        chain,
        region: str,
        how: str,
    ) -> None:
        per_pair = []
        for pair, region_pixels in zip(pairs, regions):
            got, got_name, runner = chain(region_pixels)
            margin = got - runner
            per_pair.append(
                {
                    "key": pair["capped"]["key"],
                    "capped_frame": pair["capped"]["frame"],
                    "clean_frame": pair["clean"]["frame"],
                    "slot": pair["capped"]["index"],
                    "truth": pair["under_the_cap"],
                    "set": pair["set"],
                    "score": round(got, 6),
                    "name": got_name,
                    "runner_up": round(runner, 6),
                    "margin": round(margin, 6),
                }
            )
        margins = [entry["margin"] for entry in per_pair]
        whole, whole_passes = distinct_stats(per_pair, keys)
        identity, identity_passes = distinct_stats(per_pair, identity_keys)
        controls.append(
            {
                "name": name,
                "region": region,
                "chain": how,
                "per_pair": per_pair,
                "median_score": round(statistics.median([e["score"] for e in per_pair]), 6),
                "median_margin": round(statistics.median(margins), 6),
                "of": len(per_pair),
                **whole,
                "identity_region": identity,
            }
        )
        control_console.append(
            {
                "name": name,
                "region": region,
                "of": len(per_pair),
                "median_margin": round(statistics.median(margins), 6),
                "median_score": round(statistics.median([e["score"] for e in per_pair]), 6),
                "passes": _count_passes(per_pair),
                "passes_distinct_images": whole_passes,
                "distinct_images": whole["distinct_images"],
                "median_margin_distinct_images": whole["median_margin_distinct_images"],
                "identity_passes": identity_passes,
                "identity_images": identity["distinct_images"],
                "identity_margin": identity["median_margin_distinct_images"],
            }
        )

    add_control(
        "罩着·掩码",
        queries,
        capped_image_of,
        capped_identity_of,
        lambda region: score("对照", plain_ruler, region),
        "身份区里排掉倒计时圆盘之后那 552 px",
        "MaskedRuler（与 roster_scores 同一个统计量，换一块取区）",
    )
    add_control(
        "干净·掩码",
        cleans,
        clean_image_of,
        clean_identity_of,
        lambda region: score("干净·掩码", plain_ruler, region),
        "与上面同一块 552 px（这一端没盖罩子）",
        "MaskedRuler",
    )
    add_control(
        "干净·不掩码",
        cleans,
        clean_image_of,
        clean_identity_of,
        lambda region: top_two(roster_scores(region, faces)),
        "干净那一端的整个身份区 100×100 = 10000 px，不掩码",
        "roster_scores（线上那条链）",
    )

    # 第三条必须把 取材.json 的 clean_* 复现出来 —— 它记的就是这一栏。
    unmasked_clean = controls[2]["per_pair"]
    harvest_gap = {
        "max_abs_score_delta": max(
            abs(entry["score"] - pair["clean_score"])
            for entry, pair in zip(unmasked_clean, pairs)
        ),
        "max_abs_margin_delta": max(
            abs(entry["margin"] - pair["clean_margin"])
            for entry, pair in zip(unmasked_clean, pairs)
        ),
        "note": (
            "「干净·不掩码」这一行必须复现 取材.json 的 clean_score / clean_margin —— "
            "它记的就是线上那条链在这一整块上的读数。对不上就是这两处口径不同。"
        ),
    }
    if max(harvest_gap["max_abs_score_delta"], harvest_gap["max_abs_margin_delta"]) > 1e-3:
        raise ValueError(
            "「干净·不掩码」复现不出 取材.json 的 clean_*：差 "
            f"{harvest_gap['max_abs_score_delta']:.3e} / "
            f"{harvest_gap['max_abs_margin_delta']:.3e} —— 两处口径不一样了"
        )

    clean_masked_by_key = {entry["key"]: entry for entry in controls[1]["per_pair"]}
    carry = [
        {
            "key": entry["key"],
            "slot": entry["slot"],
            "capped_margin": entry["margin"],
            "clean_masked_margin": clean_masked_by_key[entry["key"]]["margin"],
            "delta": round(entry["margin"] - clean_masked_by_key[entry["key"]]["margin"], 6),
        }
        for entry in controls[0]["per_pair"]
    ]
    picked_carry: dict[str, float] = {}
    for entry, key in zip(carry, capped_image_of):
        picked_carry.setdefault(key, entry["delta"])
    carry_deltas = [entry["delta"] for entry in carry]
    carry_summary = {
        "question": "逐对余量差 = 罩着那一端（掩码）减 干净那一端（掩码）—— 红罩留下的账。",
        "per_pair": carry,
        "of": len(carry),
        "min": min(carry_deltas),
        "max": max(carry_deltas),
        "median": round(statistics.median(carry_deltas), 6),
        "distinct_images": len(picked_carry),
        "median_distinct_images": round(statistics.median(list(picked_carry.values())), 6),
    }

    rows = []
    for spec in rows_spec:
        ruler = ruler_for(spec["library"])
        per_pair = []
        for pair, query in zip(pairs, queries):
            region = _transform(query, spec["query"], mask)
            got, name, runner = score(spec["name"], ruler, region)
            margin = got - runner
            per_pair.append(
                {
                    "key": pair["capped"]["key"],
                    "capped": pair["capped"]["frame"],
                    "clean": pair["clean"]["frame"],
                    "slot": pair["capped"]["index"],
                    "truth": pair["under_the_cap"],
                    "set": pair["set"],
                    "score": round(got, 6),
                    "name": name,
                    "runner_up": round(runner, 6),
                    "margin": round(margin, 6),
                    "_exact": got,
                }
            )
        margins = [entry["margin"] for entry in per_pair]
        held = [entry["margin"] for entry in per_pair if entry["set"] == "validation"]
        combos: dict[str, dict] = {}
        for entry, combo in zip(per_pair, combo_of):
            combos.setdefault(combo, entry)
        images: dict[str, dict] = {}
        for entry, key in zip(per_pair, capped_image_of):
            images.setdefault(key, entry)
        by_slot = {}
        for slot in sorted({entry["slot"] for entry in per_pair}):
            values = [entry["margin"] for entry in per_pair if entry["slot"] == slot]
            by_slot[str(slot)] = {
                "pairs": len(values),
                "median_margin": round(statistics.median(values), 6),
            }
        rows.append(
            {
                "name": spec["name"],
                "params": {"query": spec["query"], "library": spec["library"]},
                "why": spec["why"],
                "compare_to": spec.get("compare_to", 0),
                "per_pair": per_pair,
                "median_margin": round(statistics.median(margins), 6),
                "median_margin_validation": round(statistics.median(held), 6) if held else None,
                "distinct_images": len(images),
                "median_margin_distinct_images": round(
                    statistics.median([entry["margin"] for entry in images.values()]), 6
                ),
                "distinct_combos": len(combos),
                "median_margin_distinct_combos": round(
                    statistics.median([entry["margin"] for entry in combos.values()]), 6
                ),
                "median_score": round(statistics.median([e["score"] for e in per_pair]), 6),
                "of": len(per_pair),
                "by_slot": by_slot,
            }
        )

    for row in rows:
        base = rows[row["compare_to"]]
        deltas = [
            abs(entry["_exact"] - other["_exact"])
            for entry, other in zip(row["per_pair"], base["per_pair"])
        ]
        row["delta_vs_compare"] = {
            "compare_to": base["name"],
            "max_abs": max(deltas) if deltas else 0.0,
            "pairs_not_identical": sum(
                1
                for entry, other in zip(row["per_pair"], base["per_pair"])
                if entry["_exact"] != other["_exact"]
            ),
            "of": len(deltas),
        }
    for row in rows:
        for entry in row["per_pair"]:
            entry.pop("_exact")

    report = {
        "note": (
            "A2c：只换预处理。尺子还是 MaskedRuler（与 roster_scores 同一个统计量），取区还是"
            "身份区里排掉倒计时圆盘之后的那一块，线上那条链一个字节没动。44 对配对来自"
            "data/precheck/cd 的 60 帧连拍，判据沿用 pair_ok。**红罩对这把尺子几乎不留痕，"
            "贵的是掩码本身**（见 controls 与 compression 两栏）。判「够 / 不够」不在这份文件里。"
        ),
        "batch": "tests/帧样本/预处理",
        "harvest": "tests/帧样本/预处理/取材.json",
        "library": harvest["library"],
        "criterion": {
            "score": ROSTER_SOLID_MATCH_SCORE,
            "margin": ROSTER_AMBIGUITY_MARGIN,
            "note": (
                "**这一份不落判定**：每一条读数只有 score / runner_up / margin。"
                "「过没过线」由读者按 ark.engine.vision.roster.ROSTER_SOLID_MATCH_SCORE"
                " 与 ROSTER_AMBIGUITY_MARGIN 现算 —— 落一个布尔下来，线一挪就有一份旧值"
                "躺在文件里，而它读起来跟新值一模一样。这两条线是**生成时**的值，"
                "单测拿它们与本文件对照，常量一改就会红，逼着重跑。"
            ),
        },
        "mask": {
            "box": list(BOX),
            "radius": RADIUS,
            "pixels": int(mask.sum()),
            "of_box": BOX[2] * BOX[3],
            "origin": (
                "这一帧像素读出来的卡心（squad_card_centres）减 102（卡宽/2）；"
                "横向取整，纵向是契约层 squad_card_rect 的 y=1045。不是常量网格。"
            ),
            "origin_source": "tests/帧样本/预处理/index.json 里每条裁剪的 origin",
            "radius_source": (
                "bench.RING_OUTER = 60 是环带外缘，但取区原点抖 1 px 时读数就摆 0.274"
                "（判定翻 2/4）；62 摆 0.021、64 摆 0.013。所以取 62 —— 所有行同一个数。"
            ),
            "digits": "倒计时数字在半径 45 以内，早就被这个圆盘排掉了，不为它扩大掩码。",
        },
        "model": {
            "fit_from": [
                pair["capped"]["key"] + " -> " + pair["clean"]["key"] for pair in calibration
            ],
            "slope": model.slope,
            "offset": list(model.offset),
            "residual": model.residual,
            "pixels": model.pixels,
            "parameters": model.parameters,
            "note": "这一单没有重新拟合：还是红罩那批两个标定对定下来的四个参数。",
        },
        "statistic": {
            "chroma": "逐像素减掉它自己三个通道的平均，再取 rms —— 颜色那一份。",
            "slope": levels["slope"],
            "lambda_star": levels["lambda_star"],
            "lambda_star_min": levels["lambda_star_min"],
            "lambda_star_max": levels["lambda_star_max"],
            "chroma_signal_median": levels["chroma_signal_median"],
            "chroma_noise_median": levels["chroma_noise_median"],
            "levels": {
                "lambda_star": "S^2/(S^2+N^2)，逐对算完取中位 —— 该保留多少颜色",
                "slope": "罩子斜率 —— 颜色在罩子底下还剩多少",
                "zero": "全灰",
            },
        },
        "pairs_note": (
            "clean_score / clean_runner_up / clean_margin 是**线上那条链**（roster_scores）在"
            "**干净那一端的整个身份区（100×100 = 10000 px、不掩码）**上的读数 —— "
            "**不是**这把尺子（掩码 552 px）在干净端的读数。后者在同一行条目下的 clean_masked，"
            "另见 controls 里「干净·掩码」那一整行。两者不是一回事，别混着用。"
        ),
        "pairs": [
            {
                "key": pair["capped"]["key"],
                "capped_frame": pair["capped"]["frame"],
                "clean_frame": pair["clean"]["frame"],
                "slot": pair["capped"]["index"],
                "truth": pair["under_the_cap"],
                "set": pair["set"],
                "clean_score": pair["clean_score"],
                "clean_runner_up": pair["clean_runner_up"],
                "clean_margin": pair["clean_margin"],
                "clean_read_by": (
                    "roster_scores（线上那条链）：整个身份区 100×100 = 10000 px、不掩码"
                ),
                "clean_masked": {
                    "score": clean_masked_by_key[pair["capped"]["key"]]["score"],
                    "runner_up": clean_masked_by_key[pair["capped"]["key"]]["runner_up"],
                    "margin": clean_masked_by_key[pair["capped"]["key"]]["margin"],
                    "read_by": (
                        "MaskedRuler（这把尺子）：身份区排掉倒计时圆盘之后那 552 px"
                    ),
                },
                "capped_read": pair["capped_read"],
                "own_affine_rms": pair["own_affine_rms"],
            }
            for pair in pairs
        ],
        "invariance": invariance,
        "redundancy": {
            "note": (
                "44 对不是 44 份独立画面：同一个时段的裁剪逐字节相同。判定与中位余量按 44 对"
                "给（规划者要的），**同时**给一份按互不相同的画面算的 —— 后者才是「几块画面、"
                "几块过线」。%d/44 里有很大一份是那几个重复时段的贡献。**去重有两个键**："
                "整块卡面（dedup_key）与那把尺子真正看到的身份区（dedup_key_identity_region）"
                "—— 报数时必须说清用的是哪一个，两者在干净端给出不同的块数。"
                % _count_passes(controls[0]["per_pair"])
            ),
            "dedup_key": (
                "裁剪整块画面（204×214 uint8 RGB）的 sha256 前 12 位 —— 同一时段里逐字节"
                "相同，所以同一个键。capped_groups / clean_groups 列出每一块与它被用了几次。"
            ),
            "crops": len(blocks),
            "distinct_crops": len(set(crop_hash.values())),
            "pairs": len(pairs),
            "distinct_pair_images": distinct_combos,
            "capped_images": len(capped_groups),
            "clean_images": len(clean_groups),
            "capped_groups": capped_groups,
            "clean_groups": clean_groups,
            "dedup_key_identity_region": (
                "另一个键：卡内 100×100 那块身份区（那把尺子真正看到的画面）的 sha256 "
                "前 12 位 —— 罩着端两个键给一样的块数，干净端不一样。"
            ),
            "capped_images_identity_region": len(set(capped_identity_of)),
            "clean_images_identity_region": len(set(clean_identity_of)),
            "by_slot": {
                str(slot): sum(1 for pair in pairs if pair["capped"]["index"] == slot)
                for slot in sorted({pair["capped"]["index"] for pair in pairs})
            },
        },
        "error": errors,
        "error_note": (
            "error 那一栏量的是**反解后的**误差（反解那块取区 vs 干净那块，逐对逐通道）："
            "rms = 逐像素差的 rms，std = **干净那张**在同一批掩码像素上的 std，比值 = rms/std。"
            "它只落在顶层 error，**不在任何一行里** —— 每行自己的误差比这一版删掉了：那个口径"
            "含均值差，与那把尺子的偏置免疫对不上，会把人往错的方向带。"
        ),
        "controls": controls,
        "delta_capped_minus_clean_masked": carry_summary,
        "compression": compression,
        "baseline": rows[0],
        "rows": rows,
    }

    print("=== 三条对照（同一批 44 对；只换喂进去的那一端、或换成线上那条链）===")
    for control in control_console:
        print(
            "%-12s passes %2d/%2d  median_margin %7.4f  median_score %7.4f  "
            "| 整卡 %2d/%2d margin %7.4f | 身份区 %2d/%2d margin %7.4f | %s"
            % (
                control["name"], control["passes"], control["of"],
                control["median_margin"], control["median_score"],
                control["passes_distinct_images"], control["distinct_images"],
                control["median_margin_distinct_images"],
                control["identity_passes"], control["identity_images"],
                control["identity_margin"], control["region"],
            )
        )
    print(
        "罩着-干净（掩码）逐对余量差：min %.4f max %.4f median %.4f | 去重 median %.4f (%d 块)"
        % (
            carry_summary["min"], carry_summary["max"], carry_summary["median"],
            carry_summary["median_distinct_images"], carry_summary["distinct_images"],
        )
    )
    print("compression:", "; ".join(
        "%s %s" % (region["name"], region["values"]) for region in compression["regions"]
    ))
    print()
    for row in rows:
        print(
            "%-28s passes %2d/%2d  median_margin %7.4f  median_score %7.4f  "
            "vs[%s] max|d| %.3e  not_identical %d/%d"
            % (
                row["name"], _count_passes(row["per_pair"]), row["of"],
                row["median_margin"],
                row["median_score"], row["delta_vs_compare"]["compare_to"],
                row["delta_vs_compare"]["max_abs"],
                row["delta_vs_compare"]["pairs_not_identical"],
                row["delta_vs_compare"]["of"],
            )
        )
    print(
        "redundancy: crops %d/%d  capped images %d  clean images %d  pair combos %d"
        " | clean-unmasked vs 取材 max|d| %.2e / %.2e"
        % (
            len(set(crop_hash.values())), len(blocks), len(capped_groups),
            len(clean_groups), distinct_combos,
            harvest_gap["max_abs_score_delta"], harvest_gap["max_abs_margin_delta"],
        )
    )
    print("distinct resized regions scored:", len({key[1] for key in cache}),
          "| cells:", len(cache))
    if dry:
        return 0
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print("mask pixels", int(mask.sum()), "| model slope", round(model.slope, 6),
          "| lambda*", round(levels["lambda_star"], 6),
          "| chroma S/N", round(levels["chroma_signal_median"], 3), "/",
          round(levels["chroma_noise_median"], 3))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
