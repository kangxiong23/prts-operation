r"""A3：让「一条绝对线」成立 —— 把真值的最差分抬到错解的最好分之上。

    .venv\Scripts\python.exe scripts\bench_redcap_identity.py [--jobs N] [--limit K] [--dry]

## 这一单问什么

现在判「认出来」要**两个**条件：最高分过 ROSTER_SOLID_MATCH_SCORE **而且**
余量拉开 ROSTER_AMBIGUITY_MARGIN。这一单量的是：能不能把余量那条去掉 ——
让**真值的最差分**稳定地高于**错解的最好分**，于是一条绝对线就够。

**验收的尺子**（用户定）：

    比值 = (100% - 真值最低分) / (100% - 次高最高分)      比值 < 1 才叫分得开

两个端点**各自来自哪一对、哪一个库条目**都要写出来；样本量报**互不相同的画面有几块**。

## 这一单不换什么

read_squad_identities / find_squad_card / squad_card_scores / roster_scores /
squad_card_centres **一个字节没动**（线上那条链接不接由 ④ 裁）。取区原点与掩码半径
在所有并排的行里**用的是同一个口径**；tests/帧样本/红罩/ 与 tests/帧样本/余量/
只读。

## 两把尺子

| 后端 | 统计量 | 取区 | 谁在用 |
| --- | --- | --- | --- |
| masked | 逐通道减均值的掩码相关（TM_CCORR + 预存 sum(t) / sum(t^2)） | 一块形状 + 一个掩码 | 掩码那一族（与 margin.MaskedRuler 同一份公式） |
| full | TM_CCOEFF_NORMED（线上那条链，roster_scores） | 整个身份区 100x100 = 10000 px | 不掩码那一族 |

**自证**：masked 全开时与 roster_scores 必须逐位对上（自检写在 self_check 里，
差 > 1e-5 就响亮报错），掩码那一族与 margin.MaskedRuler 对上。所以并排的不是两条链。

**为什么要自己写一份**：margin.MaskedRuler.scores() 只给「每个条目的最大值」，
而这一单要的是**同一批相关图上的第二个量** —— 收位置之后的**窗口内最大值**。
自己写一份才能一次算图、多窗复用（这是这一单跑得完的原因）。

## 四条口径

- **半径 62**：排掉倒计时圆盘。所有掩码行同一个数。
- **取区原点**：**这一帧像素读出来的卡心**（squad_card_centres）减 102，纵向是契约层
  的 y = 1045 —— 与 预处理.json 同一批原点。取区原点敏感性那一栏抖的是这个框。
- **去重键**：「互不相同的画面」这一版按**读数完全相同**（真值分 + 错解分）算 ——
  它给出罩着端 11 块、干净端 8 块（与 像素匹配诊断.md §5 对上）；画面键
  （整卡 / 身份区）另行并排报。
- **库立绘存 uint8**：TM_CCORR 两边同型即可，比分与 /255 完全等价，内存小四倍 ——
  这一单要同时活好几把尺子。

## 并排的东西

1. **基线**：不掩码（线上那条链）与掩码 552 px 两种取区，两个端点 + 出处 + 样本量；
2. **收位置**：第一遍自由搜索定下胜者，第二遍**所有条目共用胜者那个落点**，
   窗口 ±R（R = 0/1/2/4/8，在库立绘的画布坐标里取窗）。**胜者的分数一位不变**
   （它本来就在自己的最优上）—— 对不上就报错；
3. **锐度**：把两边的锐度磨到可比（只磨参考图）与「只磨不平移」的对照（两边同核低通，
   不位移）；σ 扫三档；
4. **分布 / 统计**：不逐像素比的那一族（逐通道直方图 · 粗网格分布）；
5. **取区**：把掩码扔掉的像素换回来（半径 45 那一族），外加半径 45/55/62/70 与
   取区原点 ±1 px 的摆动；
6. **罩着端与干净端**：上面每一行都同时报两组；
7. **锐度那一份值多少**：在某个条目的对齐处做有界亚像素重搜（±0.5 px，
   0.5 px 格），与「同二阶矩的对称核、不位移」的对照并排 —— 把对齐收益与插值软化分开。

**判「够 / 不够」不在这份文件里。** 读数落在 tests/帧样本/预处理/绝对线.json。
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.engine.vision.frame import frame_from_raw  # noqa: E402
from ark.engine.vision.margin import free_mask  # noqa: E402
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
OUT = BATCH / "绝对线.json"

BOX = (IDENTITY_BOX_X, IDENTITY_BOX_Y, IDENTITY_BOX_WIDTH, IDENTITY_BOX_HEIGHT)
CARD = (214, 204, 3)
RADIUS = 62
WINDOWS = (0, 1, 2, 4, 8)
SIGMAS = (0.0, 0.5, 1.0, 2.0)
ALIGN_OFFSETS = ((-0.5, -0.5), (-0.5, 0.0), (-0.5, 0.5), (0.0, -0.5), (0.0, 0.0),
                 (0.0, 0.5), (0.5, -0.5), (0.5, 0.0), (0.5, 0.5))
BLUR_KERNEL = np.array([0.125, 0.75, 0.125], dtype=np.float32)
"""与 0.5 px 双线性平移**同二阶矩**的对称三点核（2a = 0.25 -> a = 0.125）。不位移。

它就是诊断文档 §3 第 4 步要的那个对照：亚像素重搜的收益里有多少只是「把画面磨软了」。
"""


def _sha(image: np.ndarray, length: int = 12) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()[:length]


def _blur_u8(image: np.ndarray, sigma: float) -> np.ndarray:
    """对称高斯低通，**不位移** —— 落在 uint8 上（库立绘与取区都是 uint8）。"""
    if sigma <= 0.0:
        return np.ascontiguousarray(image)
    as_float = cv2.GaussianBlur(image.astype(np.float32), (0, 0), float(sigma))
    return np.ascontiguousarray(np.clip(np.rint(as_float), 0.0, 255.0).astype(np.uint8))


def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, width, height = (int(value) for value in box)
    return np.ascontiguousarray(image[y0 : y0 + height, x0 : x0 + width])


def _windowed_max(plane: np.ndarray, centre: tuple[int, int], radius: int) -> float:
    """一张相关图上，以 centre = (x, y) 为中心、边长 2R+1 的窗里取最大。

    窗在**库立绘的画布坐标**里取。条目那档尺度比胜者大时，它的图会短一截，落点可能
    落在图外 —— 那就**裁到最近的合法落点**（贴边）。胜者自己的落点一定在图里，
    所以它这一档是恒等变换（这一条就是「胜者分数一位不变」）。
    """
    height, width = plane.shape
    x0 = min(max(int(centre[0]), 0), width - 1)
    y0 = min(max(int(centre[1]), 0), height - 1)
    rows = slice(max(0, y0 - radius), min(height, y0 + radius + 1))
    cols = slice(max(0, x0 - radius), min(width, x0 + radius + 1))
    return float(plane[rows, cols].max())


class MaskedRuler:
    """掩码那一族：一块 (取区形状, 掩码)，**一次算图、多窗复用**。

    与 margin.MaskedRuler 同一份公式（逐通道减均值、按能量归一化、TM_CCORR 分子），
    但多两件事：① 给出**胜者的落点**（取窗要用它）；② 给出一列窗口内的最大值，
    所以「自由搜索」与「收位置」共用同一批相关图 —— 一行只算一次。
    """

    def __init__(self, mask: np.ndarray, faces, *, art_sigma: float = 0.0) -> None:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2:
            raise TypeError("掩码要是 (高, 宽) 的布尔数组")
        if int(mask.sum()) < 16:
            raise ValueError("掩码只有 %d 个像素：太少，量不出端点" % int(mask.sum()))
        if not faces:
            raise ValueError("库是空的")
        self.mask_shape = (int(mask.shape[0]), int(mask.shape[1]))
        self.mask_pixels = int(mask.sum())
        self.scales = tuple(IDENTITY_SCALES)
        self._mask_u8 = np.ascontiguousarray(mask.astype(np.uint8) * 255)
        self._prepared: list[tuple] = []
        for name, art in faces:
            if art.dtype != np.uint8 or art.ndim != 3 or art.shape[2] != 3:
                raise TypeError("库里 %r 的立绘要是 (高, 宽, 3) 的 uint8 RGB" % name)
            blurred = _blur_u8(art, art_sigma)
            square = np.ascontiguousarray(blurred.astype(np.float32) ** 2)
            levels: list[tuple] = []
            for scale in self.scales:
                height = int(round(self.mask_shape[0] * scale))
                width = int(round(self.mask_shape[1] * scale))
                if height > blurred.shape[0] or width > blurred.shape[1]:
                    continue
                small = cv2.resize(
                    self._mask_u8, (width, height), interpolation=cv2.INTER_NEAREST
                )
                layer = small > 127
                pixels = float(layer.sum())
                if pixels < 16:
                    continue
                layer_f = np.ascontiguousarray(layer.astype(np.float32))
                layer_u8 = np.ascontiguousarray(layer.astype(np.uint8))
                channels: list[np.ndarray] = []
                energy: np.ndarray | None = None
                for index in range(3):
                    channel = np.ascontiguousarray(blurred[:, :, index])
                    square_channel = np.ascontiguousarray(square[:, :, index])
                    image_sum = cv2.matchTemplate(channel, layer_u8, cv2.TM_CCORR)
                    image_square = cv2.matchTemplate(square_channel, layer_f, cv2.TM_CCORR)
                    # 这一个减法**必须**在 float64 里做：两项都近似 n * mean^2，差值才是方差，
                    # float32 在这里会把有效位吃光。算完降回 float32 存 —— 那时只剩相对误差
                    # ~6e-8，够。
                    term = image_square.astype(np.float64) - (
                        image_sum.astype(np.float64) ** 2 / pixels
                    )
                    energy = term if energy is None else energy + term
                    channels.append(np.ascontiguousarray(image_sum))
                levels.append(
                    (
                        scale,
                        height,
                        width,
                        blurred,
                        layer_u8,
                        pixels,
                        channels,
                        np.ascontiguousarray(np.maximum(energy, 0.0).astype(np.float32)),
                    )
                )
            self._prepared.append((name, levels))

    def cells(self, region: np.ndarray, windows=WINDOWS) -> tuple[list, dict]:
        """一块取区 -> （每个条目一行的读数, 每个窗口一张排名表）。"""
        source = np.ascontiguousarray(region)
        rows: list[dict] = []
        for name, levels in self._prepared:
            best = -2.0
            best_scale = self.scales[0]
            best_plane = None
            for scale, height, width, art_rgb, layer_u8, pixels, channels, energy in levels:
                query = cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)
                query = np.ascontiguousarray(query)
                masked = np.ascontiguousarray(query * layer_u8[:, :, None])
                numerator = cv2.matchTemplate(art_rgb, masked, cv2.TM_CCORR)
                query_energy = 0.0
                for index in range(3):
                    channel = np.ascontiguousarray(masked[:, :, index])
                    as_float = channel.astype(np.float32)
                    mean = float(as_float.sum()) / pixels
                    numerator -= np.float32(mean) * channels[index]
                    query_energy += float(np.dot(as_float.ravel(), as_float.ravel())) - (
                        pixels * mean * mean
                    )
                if query_energy <= 0.0:
                    continue
                denominator = np.float32(np.sqrt(query_energy)) * np.sqrt(energy)
                plane = numerator / (denominator + np.float32(1e-8))
                # 掩码太小的时候有的窗里立绘的起伏是 0（energy == 0），分母就塌了 —— 不算数
                plane[energy <= 0.0] = -2.0
                value = float(plane.max())
                if value > best:
                    best = value
                    best_scale = scale
                    best_plane = plane
            rows.append({"name": name, "score": best, "scale": best_scale, "plane": best_plane})
        ranked = sorted(rows, key=lambda item: (-item["score"], item["name"]))
        for row in rows:
            loc = np.unravel_index(int(row["plane"].argmax()), row["plane"].shape)
            row["loc"] = (int(loc[1]), int(loc[0]))
        winner = ranked[0]
        out = {int(radius): [] for radius in windows}
        for radius in windows:
            table = [
                {
                    "name": row["name"],
                    "score": _windowed_max(row["plane"], winner["loc"], int(radius)),
                    "scale": row["scale"],
                }
                for row in rows
            ]
            table.sort(key=lambda item: (-item["score"], item["name"]))
            out[int(radius)] = table
        return ranked, {"windows": out, "winner": winner}


class FullRuler:
    """不掩码那一族：线上那条链（TM_CCOEFF_NORMED），同样一次算图、多窗复用。"""

    def __init__(self, faces, *, art_sigma: float = 0.0) -> None:
        if not faces:
            raise ValueError("库是空的")
        self.faces = tuple((name, _blur_u8(art, art_sigma)) for name, art in faces)

    def cells(self, region: np.ndarray, windows=WINDOWS) -> tuple[list, dict]:
        source = np.ascontiguousarray(region)
        rows: list[dict] = []
        for name, art in self.faces:
            best = -2.0
            best_scale = IDENTITY_SCALES[0]
            best_plane = None
            for scale in IDENTITY_SCALES:
                height = int(round(source.shape[0] * scale))
                width = int(round(source.shape[1] * scale))
                resized = cv2.resize(source, (width, height), interpolation=cv2.INTER_LINEAR)
                plane = np.array(
                    cv2.matchTemplate(art, resized, cv2.TM_CCOEFF_NORMED), dtype=np.float64
                )
                value = float(plane.max())
                if value > best:
                    best = value
                    best_scale = scale
                    best_plane = plane
            rows.append({"name": name, "score": best, "scale": best_scale, "plane": best_plane})
        ranked = sorted(rows, key=lambda item: (-item["score"], item["name"]))
        for row in rows:
            loc = np.unravel_index(int(row["plane"].argmax()), row["plane"].shape)
            row["loc"] = (int(loc[1]), int(loc[0]))
        winner = ranked[0]
        out = {int(radius): [] for radius in windows}
        for radius in windows:
            table = [
                {
                    "name": row["name"],
                    "score": _windowed_max(row["plane"], winner["loc"], int(radius)),
                    "scale": row["scale"],
                }
                for row in rows
            ]
            table.sort(key=lambda item: (-item["score"], item["name"]))
            out[int(radius)] = table
        return ranked, {"windows": out, "winner": winner}


def _spec_key(spec: dict) -> str:
    return json.dumps(
        {key: spec[key] for key in ("mode", "radius", "box", "art_sigma")}, sort_keys=True
    )


_STATE: dict = {"key": None, "ruler": None}
_FACES: dict = {"value": None}


def _faces():
    """卡面库只从盘上读一次 —— 每个任务重新 load 一遍会把这一单的时间吃掉一大半。"""
    if _FACES["value"] is None:
        _FACES["value"] = load_cardfaces()
    return _FACES["value"]


def _ruler_for(spec: dict, faces):
    key = _spec_key(spec)
    if _STATE["key"] != key:
        if spec["mode"] == "masked":
            mask = free_mask(tuple(spec["box"]), radius=int(spec["radius"]))
            _STATE["ruler"] = MaskedRuler(mask, faces, art_sigma=float(spec["art_sigma"]))
        else:
            _STATE["ruler"] = FullRuler(faces, art_sigma=float(spec["art_sigma"]))
        _STATE["key"] = key
    return _STATE["ruler"]


_RESULTS: dict = {}


def _task(payload: dict) -> dict:
    """一个 (规格, 端, 对) 的活：返回那一端 11 个库条目的读数 + 各档窗。

    **同一个规格下取区逐字节相同就复用** —— 44 对里罩着端只有 19 块互不相同的身份区、
    干净端只有 8 块（同一时段的连拍逐字节一样）。尺子是确定性的，喂一样的字节就是一样的读数。
    """
    cache_key = (payload["spec_key"], payload["bytes"])
    if cache_key in _RESULTS:
        return _RESULTS[cache_key]
    spec = payload["spec"]
    region = np.frombuffer(payload["bytes"], dtype=np.uint8).reshape(tuple(spec["shape"]))
    region = _blur_u8(region, float(spec["query_sigma"]))
    ruler = _ruler_for(spec, _faces())
    ranked, detail = ruler.cells(region, payload["windows"])
    out = {
        "key": payload["key"],
        "end": payload["end"],
        "ranked": [
            (float(row["score"]), row["name"], float(row["scale"])) for row in ranked
        ],
        "windows": {
            str(radius): [(float(e["score"]), e["name"], float(e["scale"])) for e in table]
            for radius, table in detail["windows"].items()
        },
        "winner": {
            "name": detail["winner"]["name"],
            "score": float(detail["winner"]["score"]),
            "scale": float(detail["winner"]["scale"]),
            "loc": [int(detail["winner"]["loc"][0]), int(detail["winner"]["loc"][1])],
        },
    }
    _RESULTS[cache_key] = out
    return out


def _prepare_specs() -> list[dict]:
    def spec(name, mode, **rest):
        base = {
            "name": name,
            "mode": mode,
            "radius": RADIUS,
            "box": list(BOX),
            "art_sigma": 0.0,
            "query_sigma": 0.0,
            "shape": [BOX[3], BOX[2], CARD[2]],
        }
        base.update(rest)
        return base

    # **七条规格**，把七件【做】全覆盖了，就这些。少了哪几档见 sensitivity 的说明：
    # 这台机器上「互不相同的身份区画面」一块要 ~1.4 s（matchTemplate 是瓶颈，
    # 见 run_timings 那一段的取值），所以档数就是时间。
    return [
        spec("基线-掩码552", "masked"),
        spec("基线-不掩码10000", "full"),
        spec("取区-环带回填·半径45", "masked", radius=45),
        spec("取区原点·x-1", "masked", box=[BOX[0] - 1, BOX[1], BOX[2], BOX[3]]),
        spec("取区原点·x+1", "masked", box=[BOX[0] + 1, BOX[1], BOX[2], BOX[3]]),
        spec("锐度-磨参考图·sigma1", "masked", art_sigma=1.0),
        spec("锐度-只磨不平移·sigma1", "masked", art_sigma=1.0, query_sigma=1.0),
    ]


def _num(value) -> str:
    return "%.4f" % float(value)


def _fmt(endpoint: dict) -> str:
    return "%s(%s %s#%d)" % (
        _num(endpoint["score"]),
        endpoint["pair"],
        endpoint["entry"],
        endpoint["slot"],
    )


def _how(spec: dict, radius) -> str:
    if radius is not None:
        return (
            "第一遍自由搜索定下胜者，第二遍所有条目共用胜者那个落点，窗 ±%d px"
            "（在库立绘的画布坐标里取窗，逐条目按它自己那档尺度）" % radius
        )
    if spec["mode"] == "full":
        return "线上那条链：整个身份区 100x100 = 10000 px，不掩码；库里每条在整块立绘里滑着找位置"
    return "掩码 %d px（身份区排掉倒计时圆盘），库里每条在整块立绘里滑着找位置" % int(
        free_mask(tuple(spec["box"]), radius=int(spec["radius"])).sum()
    )


def _end_summary(
    per: list[dict], identity_hashes: list[str], card_hashes: list[str], library: int
) -> dict:
    min_truth = min(per, key=lambda e: (e["truth_score"], e["key"]))
    max_wrong = min(per, key=lambda e: (-e["wrong_score"], e["key"]))
    readings = {(e["truth_score"], e["wrong_score"]) for e in per}
    margins = [e["margin"] for e in per]
    return {
        "pairs": len(per),
        "min_truth": {
            "score": min_truth["truth_score"],
            "pair": min_truth["key"],
            "frame": min_truth["key"].split("#")[0],
            "slot": min_truth["slot"],
            "entry": min_truth["truth"],
            "set": min_truth["set"],
        },
        "max_wrong": {
            "score": max_wrong["wrong_score"],
            "pair": max_wrong["key"],
            "frame": max_wrong["key"].split("#")[0],
            "slot": max_wrong["slot"],
            "entry": max_wrong["wrong_name"],
        },
        "ratio": round((1.0 - min_truth["truth_score"]) / (1.0 - max_wrong["wrong_score"]), 6),
        "gap": round(min_truth["truth_score"] - max_wrong["wrong_score"], 6),
        "max_truth": round(max(e["truth_score"] for e in per), 6),
        "median_margin": round(statistics.median(margins), 6),
        "min_margin": round(min(margins), 6),
        # **不落判定**：这一份只落分数与余量（truth_score / wrong_score / margin），
        # 判定由读者按 ark.engine.vision.roster 的两条线算 —— 落一个布尔下来，
        # 线一挪就有一份旧值躺在文件里，而它读起来跟新值一模一样。
        "distinct_readings": len(readings),
        "distinct_identity_images": len(set(identity_hashes)),
        "distinct_card_images": len(set(card_hashes)),
        "library": library,
    }


def _sensitivity(rows: list[dict], by_name: dict) -> dict:
    window = {}
    for row in rows:
        if row["window"] is None:
            continue
        window.setdefault(row["spec"], {})[row["window"]] = {
            side: {
                "min_truth": row["ends"][side]["min_truth"]["score"],
                "max_wrong": row["ends"][side]["max_wrong"]["score"],
                "ratio": row["ends"][side]["ratio"],
                "median_margin": row["ends"][side]["median_margin"],
            }
            for side in ("capped", "clean")
        }
    sigma = {}
    for value in SIGMAS:
        for kind, prefix in (("磨参考图", "锐度-磨参考图"), ("只磨不平移", "锐度-只磨不平移")):
            key = prefix + "·sigma%g" % value
            row = by_name.get(key)
            if row is None:
                continue
            sigma.setdefault(kind, {})[value] = {
                side: {
                    "ratio": row["ends"][side]["ratio"],
                    "min_truth": row["ends"][side]["min_truth"]["score"],
                    "max_wrong": row["ends"][side]["max_wrong"]["score"],
                }
                for side in ("capped", "clean")
            }
    baseline_sigma = {
        side: {
            "ratio": by_name["基线-掩码552"]["ends"][side]["ratio"],
            "min_truth": by_name["基线-掩码552"]["ends"][side]["min_truth"]["score"],
            "max_wrong": by_name["基线-掩码552"]["ends"][side]["max_wrong"]["score"],
        }
        for side in ("capped", "clean")
    }
    radius = {}
    for spec_name, label in (
        ("取区-环带回填·半径45", "45"),
        ("取区-半径55", "55"),
        ("基线-掩码552", "62"),
        ("取区-半径70", "70"),
    ):
        if spec_name not in by_name:
            continue
        row = by_name[spec_name]
        radius[label] = {
            side: {
                "pixels": int(
                    free_mask(
                        tuple(row["params"]["box"]), radius=int(row["params"]["radius"])
                    ).sum()
                ),
                "ratio": row["ends"][side]["ratio"],
                "median_margin": row["ends"][side]["median_margin"],
            }
            for side in ("capped", "clean")
        }
    origin = {}
    for spec_name, label in (
        ("取区原点·x-1", "-1"),
        ("基线-掩码552", "0"),
        ("取区原点·x+1", "+1"),
    ):
        if spec_name not in by_name:
            continue
        row = by_name[spec_name]
        origin[label] = {
            side: {
                "ratio": row["ends"][side]["ratio"],
                "median_margin": row["ends"][side]["median_margin"],
                "median_truth_score": round(
                    statistics.median(
                        [e["truth_score"] for e in row["ends"][side]["per_pair"]]
                    ),
                    6,
                ),
            }
            for side in ("capped", "clean")
        }
    return {
        "note": (
            "四档自变量的摆动：收位置那个窗、低通的强度、掩码半径、取区原点。同一个口径才可并排。"
            "**强度只扫了 sigma = 0（基线）与 1.0**，半径只扫了 62（基线）与 45 —— "
            "这块机器上一档要 ~40 s（互不相同的身份区画面 27 块 x 1.4 s），档数就是时间，"
            "所以先给两档；要更多档就加 specs。"
        ),
        "window": window,
        "sigma": sigma,
        "sigma_baseline": baseline_sigma,
        "radius": radius,
        "box_origin": origin,
        "window_median_margin": {
            row["spec"] + "±%d" % row["window"]: {
                side: row["ends"][side]["median_margin"] for side in ("capped", "clean")
            }
            for row in rows
            if row["window"] is not None
        },
        "radius_ratio": {
            label: {side: radius[label][side]["ratio"] for side in ("capped", "clean")}
            for label in radius
        },
        "box_origin_ratio": {
            label: {side: origin[label][side]["ratio"] for side in ("capped", "clean")}
            for label in origin
        },
    }


def _feature_hist(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """逐通道直方图（32 桶，L1 归一），三通道接起来 —— **丢掉空间布局**，保留颜色。"""
    out = []
    for index in range(3):
        histogram = np.bincount(
            (image[:, :, index].astype(np.int32) * bins // 256).ravel(), minlength=bins
        ).astype(np.float64)
        out.append(histogram / max(histogram.sum(), 1.0))
    return np.concatenate(out)


def _feature_grid(image: np.ndarray, side: int = 8) -> np.ndarray:
    """缩到 side x side 的逐格逐通道均值（INTER_AREA）—— **丢掉细节与锐度**，保留粗布局。"""
    small = cv2.resize(image, (side, side), interpolation=cv2.INTER_AREA)
    return small.astype(np.float64).reshape(-1)


def _correlate(left: np.ndarray, right: np.ndarray) -> float:
    a = left - left.mean()
    b = right - right.mean()
    denominator = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    if denominator <= 0.0:
        return 0.0
    return float((a * b).sum() / denominator)


def _distribution(pairs, blocks, faces, model) -> dict:
    """不逐像素比的那一族：同一个「零均值单位模相关」尺子，两种取法各一行。"""
    families = [
        {
            "name": "直方图-逐通道32桶",
            "extract": _feature_hist,
            "drops": "丢掉全部空间布局（放哪都一样）；保留颜色与灰度分布",
            "region": "查询 = 身份区那块 100x100；参考 = 库立绘整块 480x214",
        },
        {
            "name": "粗网格-8x8逐格均值",
            "extract": _feature_grid,
            "drops": "丢掉细节与锐度（缩到 8x8 就是平均）；保留粗布局与颜色，两边视场不同",
            "region": "查询 = 身份区那块缩到 8x8；参考 = 库立绘整块缩到 8x8",
        },
    ]
    out = []
    for family in families:
        library = [family["extract"](art) for _, art in faces]
        ends = {}
        for side in ("capped", "clean"):
            per = []
            for pair in pairs:
                card = blocks[Path(pair[side]["file"]).stem]
                if side == "capped":
                    card = undo_redcap(card, model)
                query = family["extract"](_crop(card, BOX))
                truth = pair["under_the_cap"]
                scores = [
                    (round(_correlate(query, art), 6), name)
                    for art, (name, _) in zip(library, faces)
                ]
                truth_score = [s for s, name in scores if name == truth][0]
                wrong_score, wrong_name = max(
                    [(s, name) for s, name in scores if name != truth],
                    key=lambda item: item[0],
                )
                per.append(
                    {
                        "key": pair["capped"]["key"],
                        "slot": pair["capped"]["index"],
                        "set": pair["set"],
                        "truth": truth,
                        "truth_score": truth_score,
                        "wrong_score": wrong_score,
                        "wrong_name": wrong_name,
                        "margin": round(truth_score - wrong_score, 6),
                        "pair_ratio": (
                            round((1.0 - truth_score) / (1.0 - wrong_score), 6)
                            if wrong_score < 1.0
                            else None
                        ),
                    }
                )
            ends[side] = _end_summary(
                per,
                [_sha(_crop(blocks[Path(pair[side]["file"]).stem], BOX)) for pair in pairs],
                [_sha(blocks[Path(pair[side]["file"]).stem]) for pair in pairs],
                len(faces),
            )
            ends[side]["per_pair"] = per
        out.append(
            {**{key: family[key] for key in ("name", "drops", "region")}, "ends": ends}
        )
    return {
        "question": "用不逐像素比的那一族，同一批 44 对能不能把真值的最差分抬到错解的最好分之上。",
        "statistic": "两个特征向量的零均值单位模相关（与那把尺子同一个族：减均值、除能量）",
        "warning": (
            "仓库里同族试过两次都没分开：红色占比（红罩卡 0.73-0.85 对干净卡最高 0.74）、"
            "亮度（命中组 56-216 对未中组 52-143，区间重叠）。它们答的问题不完全相同，"
            "只当警示。"
        ),
        "families": out,
    }


def _sharpness(pairs, blocks, model, faces, results) -> dict:
    """锐度那一份值多少：在**不掩码那一行的胜者**的对齐处做有界亚像素重搜，
    与「同二阶矩的对称核、不位移」的对照并排。"""
    lookup = {
        side: results[("基线-不掩码10000", side)] for side in ("capped", "clean")
    }
    per_pair = []
    for side in ("capped", "clean"):
        for pair, item in zip(pairs, lookup[side]):
            winner = item["winner"]
            name = winner["name"]
            scale = winner["scale"]
            base = float(winner["score"])
            art = dict(faces)[name]
            card = blocks[Path(pair[side]["file"]).stem]
            if side == "capped":
                card = undo_redcap(card, model)
            region = _crop(card, BOX)
            height = int(round(region.shape[0] * scale))
            width = int(round(region.shape[1] * scale))
            resized = cv2.resize(region, (width, height), interpolation=cv2.INTER_LINEAR)
            locate = (int(winner["loc"][0]), int(winner["loc"][1]))
            best_align = base
            best_offset = [0.0, 0.0]
            for dx, dy in ALIGN_OFFSETS:
                matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
                warped = cv2.warpAffine(
                    resized,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                plane = np.array(
                    cv2.matchTemplate(art, warped, cv2.TM_CCOEFF_NORMED), dtype=np.float64
                )
                value = _windowed_max(plane, locate, 1)
                if value > best_align:
                    best_align = float(value)
                    best_offset = [float(dx), float(dy)]
            softened = cv2.sepFilter2D(
                resized.astype(np.float32),
                -1,
                BLUR_KERNEL,
                BLUR_KERNEL,
                borderType=cv2.BORDER_REPLICATE,
            )
            softened = np.ascontiguousarray(
                np.clip(np.rint(softened), 0.0, 255.0).astype(np.uint8)
            )
            blur_plane = np.array(
                cv2.matchTemplate(art, softened, cv2.TM_CCOEFF_NORMED), dtype=np.float64
            )
            blur_score = _windowed_max(blur_plane, locate, 1)
            align_gain = best_align - base
            blur_gain = float(blur_score) - base
            per_pair.append(
                {
                    "key": pair["capped"]["key"],
                    "end": side,
                    "entry": name,
                    "scale": scale,
                    "base": round(base, 6),
                    "align_best": round(best_align, 6),
                    "align_offset": best_offset,
                    "align_gain": round(align_gain, 6),
                    "blur_score": round(float(blur_score), 6),
                    "blur_gain": round(blur_gain, 6),
                    "share": round(blur_gain / align_gain, 4) if align_gain > 1e-9 else None,
                }
            )
    summary = {}
    for side in ("capped", "clean"):
        block = [e for e in per_pair if e["end"] == side]
        aligns = [e["align_gain"] for e in block]
        blurs = [e["blur_gain"] for e in block]
        shares = [e["share"] for e in block if e["share"] is not None]
        summary[side] = {
            "pairs": len(block),
            "align_gain_min": round(min(aligns), 6),
            "align_gain_max": round(max(aligns), 6),
            "align_gain_median": round(statistics.median(aligns), 6),
            "align_gain_positive": sum(1 for value in aligns if value > 1e-9),
            "blur_gain_min": round(min(blurs), 6),
            "blur_gain_max": round(max(blurs), 6),
            "blur_gain_median": round(statistics.median(blurs), 6),
            "blur_gain_positive": sum(1 for value in blurs if value > 1e-9),
            "share_median": round(statistics.median(shares), 4) if shares else None,
            "share_max": round(max(shares), 4) if shares else None,
            "share_of": len(shares),
            "share_note": (
                "share 只在「对齐确实多拿了分」（align_gain > 0）的那些对上算；"
                "对齐收益为 0 的那些对没有分母。负的 blur_gain 表示低通本身在扣分。"
            ),
        }
    return {
        "question": (
            "把一个条目的对齐再往细里搜，能不能多拿分？多拿的那一份里有多少只是"
            "「把画面磨软了」？"
        ),
        "seed": (
            "种子 = 「基线-不掩码10000」那一行（线上那条链，TM_CCOEFF_NORMED）在每对每端的"
            "胜者：用它的（条目, 尺度, 落点）。不是全库重排 —— 这一栏量的是**一个条目**的"
            "对齐收益，两个端点那一栏在 rows 里。"
        ),
        "align_offsets_px": [list(offset) for offset in ALIGN_OFFSETS],
        "align_method": (
            "把该尺度下的查询用 warpAffine（双线性、边界复制）平移 (dx, dy) 之后与**那条库"
            "立绘**做 TM_CCOEFF_NORMED，在胜者落点的 ±1 px 窗里取最大；取 9 个偏移里的最好"
            "一个（0.5 px 格）。"
        ),
        "blur_kernel": [0.125, 0.75, 0.125],
        "blur_method": (
            "对照：把同一个模板用一个**同二阶矩的对称三点核** [0.125, 0.75, 0.125] 低通，"
            "**不位移**，在同一个窗里取最大。它拿到的就是插值软化那一份（诊断文档 §3 第 4 步）。"
        ),
        "per_pair": per_pair,
        "capped": summary["capped"],
        "clean": summary["clean"],
        "value_of_sharpness": (
            "锐度（软化）那一份 = blur_gain，单位是分数；align_gain 是它的上界。"
            "share = blur_gain / align_gain（对齐收益里有多少是软化）。"
        ),
    }


def _self_check(pairs, blocks, model, faces) -> dict:
    """两把自写的尺子必须对上原来那两把 —— 对不上就是并排的不是同一条链。"""
    from ark.engine.vision.margin import MaskedRuler as ReferenceMasked

    pair = pairs[0]
    card = blocks[Path(pair["capped"]["file"]).stem]
    query = _crop(undo_redcap(card, model), BOX)
    mask = free_mask(BOX, radius=RADIUS)
    mine = MaskedRuler(mask, faces)
    reference = ReferenceMasked(mask, faces)
    mine_scores = {row["name"]: row["score"] for row in mine.cells(query, (0,))[0]}
    reference_scores = {name: score for score, name, _ in reference.scores(query)}
    masked_gap = max(abs(mine_scores[name] - reference_scores[name]) for name in mine_scores)
    clean = _crop(blocks[Path(pair["clean"]["file"]).stem], BOX)
    full = FullRuler(faces)
    full_scores = {row["name"]: row["score"] for row in full.cells(clean, (0,))[0]}
    roster = {name: score for score, name, _ in roster_scores(clean, faces)}
    full_gap = max(abs(full_scores[name] - roster[name]) for name in full_scores)
    masked_full = MaskedRuler(np.ones_like(mask), faces)
    cross_gap = max(
        abs(row["score"] - roster[row["name"]]) for row in masked_full.cells(clean, (0,))[0]
    )
    if max(masked_gap, full_gap) > 1e-5:
        raise ValueError(
            "自检没对上：masked 差 %.3e、full 差 %.3e —— 并排的不是同一条链"
            % (masked_gap, full_gap)
        )
    return {
        "masked_vs_margin_masked": round(masked_gap, 12),
        "full_vs_roster_scores": round(full_gap, 12),
        "masked_all_open_vs_roster_scores": round(cross_gap, 12),
        "note": (
            "masked 与 margin.MaskedRuler 同一块取区上的逐条目最大差；full 与 roster_scores "
            "的逐条目最大差；第三行验的是「掩码全开时 masked 复现线上那条链」。"
        ),
    }


def main(argv: list[str]) -> int:
    # 默认串行：这一台机器上 ProcessPoolExecutor 起不来（沙箱不许开命名管道，
    # multiprocessing 报 WinError 5）。--jobs N 留着，换了环境就能用。
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 1
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
    calibration = [pair for pair in pairs if pair["set"] == "calibration"]
    if len(calibration) != 2:
        raise ValueError("标定对要是 2 个，收到 %d" % len(calibration))
    model = fit_redcap(
        [
            (
                blocks[Path(pair["capped"]["file"]).stem],
                blocks[Path(pair["clean"]["file"]).stem],
            )
            for pair in calibration
        ]
    )
    faces = _faces()
    card_of = {
        side: [_sha(blocks[Path(pair[side]["file"]).stem]) for pair in pairs]
        for side in ("capped", "clean")
    }

    self_check = _self_check(pairs, blocks, model, faces)

    specs = _prepare_specs()
    windows_rows = ("基线-掩码552", "基线-不掩码10000", "取区-环带回填·半径45")

    regions: dict[str, dict] = {}
    for spec in specs:
        box = tuple(spec["box"])
        regions[spec["name"]] = {
            "capped": [
                _blur_u8(
                    _crop(undo_redcap(blocks[Path(pair["capped"]["file"]).stem], model), box),
                    float(spec["query_sigma"]),
                )
                for pair in pairs
            ],
            "clean": [
                _blur_u8(
                    _crop(blocks[Path(pair["clean"]["file"]).stem], box),
                    float(spec["query_sigma"]),
                )
                for pair in pairs
            ],
            "identity": {
                side: [
                    _sha(_crop(blocks[Path(pair[side]["file"]).stem], box))
                    for pair in pairs
                ]
                for side in ("capped", "clean")
            },
        }

    import time as _time

    results: dict[tuple, list] = {}
    timings: list[tuple[str, float, int]] = []
    for spec in specs:
        want = list(WINDOWS) if spec["name"] in windows_rows else [0]
        payloads = []
        for side in ("capped", "clean"):
            for index, pair in enumerate(pairs):
                payloads.append(
                    {
                        "spec": spec,
                        "spec_key": _spec_key(spec),
                        "windows": want,
                        "bytes": regions[spec["name"]][side][index].tobytes(),
                        "key": pair["capped"]["key"],
                        "end": side,
                    }
                )
        started = _time.perf_counter()
        if jobs > 1:
            with ProcessPoolExecutor(max_workers=int(jobs)) as pool:
                got = list(pool.map(_task, payloads, chunksize=2))
        else:
            got = [_task(payload) for payload in payloads]
        timings.append((spec["name"], _time.perf_counter() - started, len(_RESULTS)))
        for item in got:
            results.setdefault((spec["name"], item["end"]), []).append(item)

    rows = []
    for spec in specs:
        name = spec["name"]
        variants = [None] + list(WINDOWS) if name in windows_rows else [None]
        for radius in variants:
            label = name if radius is None else "%s · 收位置窗±%d" % (name, radius)
            row = {
                "name": label,
                "spec": name,
                "window": radius,
                "how": _how(spec, radius),
                "params": {
                    "mode": spec["mode"],
                    "box": list(spec["box"]),
                    "radius": spec["radius"],
                    "art_sigma": spec["art_sigma"],
                    "query_sigma": spec["query_sigma"],
                    "window_px": radius,
                },
                "ends": {},
            }
            for side in ("capped", "clean"):
                per = []
                for pair, item in zip(pairs, results[(name, side)]):
                    table = item["ranked"] if radius is None else item["windows"][str(radius)]
                    truth = pair["under_the_cap"]
                    truth_scores = [e[0] for e in table if e[1] == truth]
                    if len(truth_scores) != 1:
                        raise ValueError("库里的真值 %r 在读数里不是恰一条" % truth)
                    truth_score = float(truth_scores[0])
                    wrong = [(e[0], e[1]) for e in table if e[1] != truth]
                    wrong_score, wrong_name = max(wrong, key=lambda item2: item2[0])
                    per.append(
                        {
                            "key": pair["capped"]["key"],
                            "slot": pair["capped"]["index"],
                            "set": pair["set"],
                            "truth": truth,
                            "truth_score": round(truth_score, 6),
                            "wrong_score": round(wrong_score, 6),
                            "wrong_name": wrong_name,
                            "margin": round(truth_score - wrong_score, 6),
                            # 胜者（最高分那个条目）**不round**：收位置那几行要证它一位不变
                            "winner_score": float(table[0][0]),
                            "winner_name": table[0][1],
                            "pair_ratio": (
                                round((1.0 - truth_score) / (1.0 - wrong_score), 6)
                                if wrong_score < 1.0
                                else None
                            ),
                        }
                    )
                row["ends"][side] = _end_summary(
                    per, regions[name]["identity"][side], card_of[side], len(faces)
                )
                row["ends"][side]["per_pair"] = per
                row["ends"][side]["winner_is_truth"] = all(
                    e["truth_score"] >= e["wrong_score"] for e in per
                )
            rows.append(row)

    by_name = {row["name"]: row for row in rows}
    baseline = by_name["基线-掩码552"]
    for row in rows:
        if row["spec"] == "基线-不掩码10000":
            row["baseline_ref"] = "基线-不掩码10000"
        elif row["window"] is None:
            row["baseline_ref"] = "基线-掩码552"
        else:
            row["baseline_ref"] = row["spec"]

    # 验收要「每一行都带改前 / 改后两态」—— 把 baseline_ref 那一行的数与差内联进来，
    # 一份文件里每一行都是自足的两态对照，不用去别处查。
    for row in rows:
        reference = by_name[row["baseline_ref"]]
        row["vs_baseline"] = {
            "baseline": row["baseline_ref"],
            "ends": {
                side: {
                    "min_truth": reference["ends"][side]["min_truth"]["score"],
                    "max_wrong": reference["ends"][side]["max_wrong"]["score"],
                    "ratio": reference["ends"][side]["ratio"],
                    "median_margin": reference["ends"][side]["median_margin"],
                    "delta_ratio": round(
                        row["ends"][side]["ratio"] - reference["ends"][side]["ratio"], 6
                    ),
                    "delta_median_margin": round(
                        row["ends"][side]["median_margin"]
                        - reference["ends"][side]["median_margin"],
                        6,
                    ),
                }
                for side in ("capped", "clean")
            },
            "note": (
                "改前 = baseline_ref 那一行（同一把尺子、同一批 44 对、同一个取区原点口径）；"
                "改后 = 本行。两态就在同一行里。"
            ),
        }

    distribution = _distribution(pairs, blocks, faces, model)
    sharpness = _sharpness(pairs, blocks, model, faces, results)
    sensitivity = _sensitivity(rows, by_name)

    report = {
        "note": (
            "A3：一条绝对线成不成立。比值 = (1 - 真值最低分) / (1 - 次高最高分)，< 1 才叫"
            "分得开。判「够 / 不够」不在这份文件里。线上那条链一个字节没动。"
        ),
        "batch": "tests/帧样本/预处理",
        "harvest": "tests/帧样本/预处理/取材.json",
        "library": harvest["library"],
        "criterion": {
            "ratio": "(1 - 真值最低分) / (1 - 次高最高分)；< 1 才叫分得开，越小越安全",
            "truth": "一对一个真值：干净那一端的身份区喂线上那条链读出来的那个名字",
            "wrong": "同一块取区上，库里除了真值以外得分最高的那一条",
            "endpoint_provenance": "两个端点各自记：哪一对（key / 卡位 / 帧）、哪个库条目",
            "sample_size": "互不相同的画面有几块（读数键 / 整卡画面键 / 身份区画面键）",
            "verdict": (
                "**这一份不落判定**：逐对只有 truth_score / wrong_score / margin。"
                "「过没过线」由读者按 ark.engine.vision.roster.ROSTER_SOLID_MATCH_SCORE"
                " 与 ROSTER_AMBIGUITY_MARGIN 现算 —— 落一个布尔下来，线一挪就有一份旧值"
                "躺在文件里，而它读起来跟新值一模一样。"
            ),
        },
        "chains": {
            "masked": (
                "逐通道减均值的掩码相关（TM_CCORR + 预存 sum(t) / sum(t^2)）——"
                "与 ark.engine.vision.margin.MaskedRuler 同一份公式"
            ),
            "full": "TM_CCOEFF_NORMED —— 线上那条链 roster_scores",
        },
        "mask": {
            "box": list(BOX),
            "radius": RADIUS,
            "pixels": int(free_mask(BOX, radius=RADIUS).sum()),
            "of_box": BOX[2] * BOX[3],
            "origin": (
                "这一帧像素读出来的卡心（squad_card_centres）减 102；纵向是契约层"
                " squad_card_rect 的 y=1045。不是常量网格。"
            ),
        },
        "model": {
            "fit_from": [
                pair["capped"]["key"] + " -> " + pair["clean"]["key"] for pair in calibration
            ],
            "slope": model.slope,
            "offset": list(model.offset),
            "parameters": model.parameters,
            "pairs": model.pairs,
        },
        "self_check": self_check,
        "windows": list(WINDOWS),
        "baseline": baseline,
        "rows": rows,
        "distribution": distribution,
        "sharpness_share": sharpness,
        "sensitivity": sensitivity,
    }

    print("=== 基线：两个端点 ===")
    for side in ("capped", "clean"):
        end = baseline["ends"][side]
        print(
            "  %-6s 真值最低 %s  次高最高 %s  比值 %.4f  读数 %d 块 / 身份区 %d 块 / 整卡 %d 块"
            % (
                side,
                _fmt(end["min_truth"]),
                _fmt(end["max_wrong"]),
                end["ratio"],
                end["distinct_readings"],
                end["distinct_identity_images"],
                end["distinct_card_images"],
            )
        )
    print()
    print("=== 每一行（改前 = baseline_ref）===")
    for row in rows:
        capped = row["ends"]["capped"]
        clean = row["ends"]["clean"]
        print(
            "%-40s | 罩着 %s -> %s 比值 %6.4f | 干净 %s -> %s 比值 %6.4f"
            % (
                row["name"],
                _num(capped["min_truth"]["score"]),
                _num(capped["max_wrong"]["score"]),
                capped["ratio"],
                _num(clean["min_truth"]["score"]),
                _num(clean["max_wrong"]["score"]),
                clean["ratio"],
            )
        )
    print()
    print("=== 分布 / 统计那一族 ===")
    for family in distribution["families"]:
        print(
            "  %-16s 罩着比值 %6.4f 干净比值 %6.4f | %s"
            % (
                family["name"],
                family["ends"]["capped"]["ratio"],
                family["ends"]["clean"]["ratio"],
                family["drops"],
            )
        )
    print()
    print("=== 锐度那一份值多少（种子 = 不掩码那一行的胜者）===")
    for side in ("capped", "clean"):
        block = sharpness[side]
        share = block["share_median"]
        print(
            "  %-6s 对齐收益中位 %.6f（%.6f…%.6f）| 只磨不平移 %.6f | 占 %s"
            % (
                side,
                block["align_gain_median"],
                block["align_gain_min"],
                block["align_gain_max"],
                block["blur_gain_median"],
                "%.3f" % share if share is not None else "nan",
            )
        )
    print()
    print("=== 敏感性 ===")
    print("  窗口：", json.dumps(sensitivity["window_median_margin"], ensure_ascii=False))
    print("  半径：", json.dumps(sensitivity["radius_ratio"], ensure_ascii=False))
    print("  原点：", json.dumps(sensitivity["box_origin_ratio"], ensure_ascii=False))
    print()
    print("自检：", json.dumps(self_check, ensure_ascii=False))
    print()
    print("=== 每个规格的耗时（互不相同的取区块才真算）===")
    for name, elapsed, cached in timings:
        print("  %-32s %6.1fs  累计取区读数 %d" % (name, elapsed, cached))
    if dry:
        return 0
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
