"""备战席点名的规格符合性验收（`tests/spec/`，N 族）。

跑法（不需要 pytest）：

    $env:PYTHONPATH="src"
    .venv\\Scripts\\python.exe tests\\spec\\roster_spec.py

**这一套只用公开面**：`roster.read_squad_identities` / `roster_scores` / `identity_region` /
`redcap_region` / `squad_card_scores` / `find_squad_card` 与它们的常量 ·
`margin.identity_disc_region` / `margin.MaskedRuler` · `cardface.load_formation_page` /
`load_cardfaces` / `cut_formation_arts` / `FORMATION_*` 常量 ·
`bench.squad_card_centres` / `bench.redeploying` · `frame.frame_from_raw` ·
`device.adb.SQUAD_CARD_PITCH`。`_` 开头的内部名一个都不碰（三纪律之一）。

**真值是怎么来的**（这一点必须写在最前，因为这一层的真值**不能从像素推**）：

- **取材那两帧**（`shot01` · `cd60`）的真值在 `tests/帧样本/点名/取材.json` 里，
  是交付时给执行者的那份；
- **验收那四帧**（`shot05` · `shot26` · `cd01` · `cd08`）的真值由**规划者另写一份
  独立实现**算出来（脸区 100×100、滑位置、尺度 1.00–1.20 步长 0.05），
  **再逐张目视核过一遍**。两条独立实现给同一批结论，才算证据 —— 只跑一条链，
  量的就是那条链自己。真值与 `取材.json` **互不重叠**。

**红罩那 14 张**（三帧里）**不再期望是「未知」**：用户裁定的是**形态**
（读得出报精确、读不出来报未知，**不是**未采样），而"红罩必然未知"那一条
**已经被推翻**（D-全局-34）—— 红罩卡现在换**另一块取区**（环带以内的圆盘 − 中央
秒数那一条，`roster.redcap_region` + `margin.MaskedRuler`），所以它**读得出就是精确，
而且必须真的是那一位**。全量那一批实测 28 张红罩卡里 16 张读得出，而**真值已立的
5 张全部读对**（`红罩/点名.json` 的 `truth` 那一栏）—— 所以下面 `TRUTH` 里那几个
「未知」是**旧账**（写在"红罩一律读不出来"那一版上），这一套不再拿它们跟实测比。
见[视觉层/备战席点名] §红罩卡 · §两块取区。

## 这一套为什么值得一口气先扫完

这 12 条（N1–N12）要看的，是**同一批帧乘几套库**：10 帧配基库（N2–N7 · N11 · N12）＋
`cd01` 配孪生库（N8）＋`cd60` 配放大库（N9）＋`cd60` 配倒序库（N10）。一次全扫
在现在这条链上是**十几秒**（一帧 11–12 张卡 × 11 条库 × 11 档尺度 ≈ 1400 次
`matchTemplate`），13 次串起来就是几分钟 —— 而**这 13 次彼此没有依赖**。

所以 `_prewarm()` 在第一条要用到读数的检查之前，把这 13 次丢进一个**线程池**里
一起算：`cv2.matchTemplate` 会放开 GIL，实测墙钟掉到 **三分之一**（6 路就吃满了，
再往上加不再赚）。**这只是跑法，不是判据** —— 池子只填 `_READS` 缓存，所有断言、
所有证据行与串行跑**逐字一样**。想串行复现（读 traceback 时更清楚）：

    $env:ARK_SPEC_POOL="1"

**N13 与这个池子无关**：它只读一份规格文档与几个常量，不碰帧、不碰库，所以
那个「13 次」也没变（N13 是第 13 条检查，不是第 14 次全扫）。

**N14–N17 是后来的**（「某一位干员在第几张卡」那条路）。它们**不进 `_tasks()` 那个池子**，
各自开一次（N14 一批 · N15 一批 · N16 一批）—— 那几批彼此没有依赖，也都只读。
**N17 必须串行跑**：它要把 `cv2.matchTemplate` 换成计数包装器，池子里还有别的活在跑时，
那个计数器就不是它一个人的了。

它们为什么值这一笔钱：这条路是**唯一一条不扫全库的读数链**，而「它真的是单条」这件事
**行为测试看不出来** —— 扫完全库再挑出那一条，答案一模一样。所以一头**数次数**（N17），
一头从两头把结论包住：有人的卡上，分数必须与全扫**逐位相同**、而且只有那一张过线（N14 ·
N15）；没人的卡上，整列分数必须**够不到线**（N16）。两头一夹，
「谁在第几张卡」这件事在**全 10 帧**上是结论性地钉住的。

**N18 与它们不同族**：它是 N13 那一对的另一半 —— 同一批读数住在**两处**（规格与
`roster.py` 的模块文档），中间要有一条"逐个数对齐"的网。它只读文本，不碰帧。

**N19 是另一批夹具**（`tests/帧样本/立绘两组/`）：两组**同 10 位干员、同位置、不同
立绘**，库是两组合起来的 **20 条**，口径是「**立绘就是身份**」（用户裁定）—— 读出
同一个干员的**另一套**立绘算失败。它也不进任何池子，那两条带自己读。

**N20 跨到红罩那批**（`tests/帧样本/红罩/`）：真值已立的那 **5 张**红罩卡必须被
**线上那条链**读对 —— N4 与 N12 的红罩名单都长在备战席那批帧上，这 5 张原本**没有网**。
真值从 `红罩/点名.json` 的 `truth` 栏读（不手抄），那几帧也走"裁剪贴回画布"那条路。

**池子里只放纯函数**：帧、库、切割都在进池之前**串行**建好 —— `_FRAMES` /
`_FACES` 那两张表不是线程安全的。池子里的活只有 `read_squad_identities` 本身。
某一个任务炸了，异常被收下来挂在 `_BROKEN` 上，检查走到它时**原样抛出去**
（不重算，也不吞）。
"""

from __future__ import annotations

import importlib
import json
import os
import re
import struct
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _entry in (str(REPO / "src"), str(_HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

BATCH = REPO / "tests" / "帧样本" / "备战席"
ROSTER = REPO / "tests" / "帧样本" / "点名"
CANVAS_W, CANVAS_H = 2800, 1260

# ---------------------------------------------------------------------------
# 真值：逐位、下标 0 = 最右那张（与卡位那一层同一个口径）
#
# "未知" = 库里没有这个人（含装置卡）。**没有一个"确切值"是猜的** ——
# 见模块文档里"真值是怎么来的"。
#
# **红罩位上那几个「未知」是旧账**：它们写在"红罩一律读不出来"那一版上。
# 这一套不再拿它们跟实测比 —— 红罩位按新规则核（`_red_slot_complaint`：
# 读得出⇒精确且名字要有来路 · 读不出⇒未知），非红罩位**照旧逐位严比**。
# ---------------------------------------------------------------------------

TRUTH: Mapping[str, Sequence[str]] = {
    # —— 取材帧（执行者手里有）——
    "shot01": ["维什戴尔", "圣聆初雪", "乌尔比安", "未知", "未知", "斩业星熊",
               "未知", "歌蕾蒂娅", "未知", "新约能天使", "未知", "未知"],
    "cd60": ["斩业星熊", "维什戴尔", "圣聆初雪", "乌尔比安", "黍", "Mon3tr",
             "白面鸮", "歌蕾蒂娅", "未知", "新约能天使", "极境"],
    # —— 验收帧（执行者手里没有）——
    "shot05": ["未知", "斩业星熊", "未知", "维什戴尔", "歌蕾蒂娅", "未知", "未知", "未知", "未知"],
    "shot26": ["未知", "新约能天使", "未知", "斩业星熊", "歌蕾蒂娅", "乌尔比安"],
    "cd01": ["未知", "未知", "未知", "Mon3tr", "白面鸮", "未知", "未知", "未知", "未知"],
    "cd08": ["未知", "未知", "未知", "未知", "未知", "未知", "未知", "未知", "极境"],
}

# 夹具里那 10 帧（含执行者连帧都没拿到的 4 帧）—— 名字无关的那些检查要跑全量
ALL_FRAMES = ("shot01", "shot02", "shot05", "shot07", "shot21", "shot26",
              "cd01", "cd08", "cd20", "cd60")

# 装置卡：`shot01` 位 6（卡中心 x = 1449.5），它不是干员
NOT_OPERATOR = {"shot01": (6,)}

UNKNOWN = "未知"

# 池子开几路。6 路就吃满了实测那条曲线（8 · 12 · 16 一路只再赚几秒），
# 所以默认 8，机器小就少开。这一层**只读**，开多路不改变任何结果。
_DEFAULT_POOL = min(8, os.cpu_count() or 1)


def _pool_size() -> int:
    raw = os.environ.get("ARK_SPEC_POOL", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, _DEFAULT_POOL)


# ---------------------------------------------------------------------------
# 骨架（与 bench_row_spec / countdown_spec 同一张表）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    ident: str
    spec: str
    claim: str
    probe: Callable[[], str]


_LOADED: dict[str, object] = {}


def _mod(dotted: str):
    """import 一个还不一定存在的模块。不存在 = 这条检查红，不是检查炸了。"""
    if dotted not in _LOADED:
        try:
            _LOADED[dotted] = importlib.import_module(dotted)
        except BaseException as exc:  # noqa: BLE001
            raise AssertionError(f"`{dotted}` 还不存在：{type(exc).__name__}: {exc}") from None
    return _LOADED[dotted]


def _index(batch: Path = BATCH) -> dict:
    return json.loads((batch / "index.json").read_text(encoding="utf-8"))


def _entry(name: str, batch: Path = BATCH) -> dict:
    for item in _index(batch)["crops"]:
        if Path(item["file"]).stem == name:
            return item
    raise AssertionError(f"夹具里没有 {name}")


def _raw(name: str, batch: Path = BATCH) -> bytes:
    """把夹具裁剪**贴回** 2800×1260 的黑底，编成设备层 §4 定义的原始 RGBA 字节。

    这样 ROI 的取法仍然是真的在被测 —— 没有把 ROI 预先替读数链做好。

    `batch` 默认是备战席那一批；**立绘两组那两条横带也是裁剪**（原点 `(300, 1040)`、
    2500×220），所以 N19 走的是同一条贴回去的路。
    """
    item = _entry(name, batch)
    x, y = item["origin"]
    w, h = item["size"]
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    canvas[y : y + h, x : x + w] = np.asarray(Image.open(batch / item["file"]).convert("RGB"))
    rgba = np.dstack([canvas, np.full((CANVAS_H, CANVAS_W), 255, np.uint8)])
    return struct.pack("<4I", CANVAS_W, CANVAS_H, 1, 1) + rgba.tobytes()


_FRAMES: dict[tuple[str, str], object] = {}


def _frame(name: str, batch: Path = BATCH):
    key = (str(batch), name)
    if key not in _FRAMES:
        _FRAMES[key] = _mod("ark.engine.vision.frame").frame_from_raw(_raw(name, batch))
    return _FRAMES[key]


def _pitch() -> float:
    return _mod("ark.engine.device.adb").SQUAD_CARD_PITCH


def _manifest() -> dict:
    return json.loads((ROSTER / "编队页.json").read_text(encoding="utf-8"))


_FACES: tuple[tuple[str, np.ndarray], ...] | None = None


def _faces() -> tuple[tuple[str, np.ndarray], ...]:
    """库：编队页那一帧切出来的 11 块立绘，按 cut_order 配名字。"""
    global _FACES
    if _FACES is None:
        man = _manifest()
        page = _mod("ark.offline.assets_gen.cardface").load_formation_page(REPO / man["frame"])
        arts = _mod("ark.offline.assets_gen.cardface").cut_formation_arts(page)
        names = list(man["names"])
        if len(arts) != len(names):
            raise AssertionError(f"切出 {len(arts)} 块，名字有 {len(names)} 个")
        _FACES = tuple(zip(names, arts))
    return _FACES


# --- 三套派生库：缓存住，好让检查与预扫拿到**同一个对象**（缓存按对象身份认）---------


@cache
def _twin_faces() -> tuple[tuple[str, np.ndarray], ...]:
    """孪生库（N8）：真库**原样留着**，再各配一条横移 2 px、换个名字的孪生块。

    真库必须留着：这条要问的是"最像与次像几乎同分时会不会挑一个"，
    所以孪生那条得**跟它自己那份真条目**争，而不是在一堆孪生里当唯一候选。
    """
    real = _faces()
    return real + tuple((f"{name}的孪生", np.roll(art, 2, axis=1)) for name, art in real)


@cache
def _bigger_faces() -> tuple[tuple[str, np.ndarray], ...]:
    """放大库（N9）：整库放大 3% —— 还挪得动的那些必须照样认得出。"""
    import cv2

    return tuple(
        (name, cv2.resize(art, (int(art.shape[1] * 1.03), int(art.shape[0] * 1.03)),
                          interpolation=cv2.INTER_LINEAR))
        for name, art in _faces()
    )


@cache
def _reversed_faces() -> tuple[tuple[str, np.ndarray], ...]:
    """倒序库（N10）：只该按内容认人，不许沾位次。"""
    return tuple(reversed(_faces()))


_READS: dict[tuple[str, str], tuple[object, tuple]] = {}
_BROKEN: dict[tuple[str, str], BaseException] = {}
_PREWARMED = False


def _compute(name: str, faces: Sequence[tuple[str, np.ndarray]]) -> tuple:
    """池子里唯一的那件活：一帧 × 一套库 → 逐位读数。**纯函数。**"""
    return _mod("ark.engine.vision.roster").read_squad_identities(
        _frame(name), faces, pitch=_pitch()
    )


def _tasks() -> tuple[tuple[tuple[str, str], str, tuple], ...]:
    """13 次全扫，没有依赖，谁都用得上 —— 所以一口气排出来。"""
    base = _faces()
    out = [((name, "库"), name, base) for name in ALL_FRAMES]
    out.append((("cd01", "孪生库"), "cd01", _twin_faces()))
    out.append((("cd60", "放大库"), "cd60", _bigger_faces()))
    out.append((("cd60", "倒序库"), "cd60", _reversed_faces()))
    return tuple(out)


def _prewarm() -> None:
    tasks = _tasks()
    # 帧先进来串行建好：`_FRAMES` 那张表不是线程安全的，而且 `_raw` 要读盘。
    for _key, name, _faces_of in tasks:
        _frame(name)
    _mod("ark.engine.vision.roster")  # 先 import 掉，别让几路线程同时去 import
    with ThreadPoolExecutor(max_workers=_pool_size()) as pool:
        jobs = [(key, faces, pool.submit(_compute, name, faces)) for key, name, faces in tasks]
    for key, faces, job in jobs:
        broken = job.exception()
        if broken is None:
            _READS[key] = (faces, job.result())
        else:
            _BROKEN[key] = broken


def _ensure_prewarm() -> None:
    global _PREWARMED
    if not _PREWARMED:
        _PREWARMED = True
        _prewarm()


def _read(name: str, faces: Sequence[tuple[str, np.ndarray]] | None = None, tag: str = "库"):
    """走**设备层 §5 的重放路径**读一帧：字节 → 帧 → 点名链。同一组合只读一次。

    缓存按 `(帧, 库)` 记，而且记着**是哪一套库**：库是按对象身份认的，
    所以"同名的签子配了另一套库"只会重算，不会串味。
    """
    chosen = _faces() if faces is None else faces
    key = (name, tag)
    _ensure_prewarm()
    hit = _READS.get(key)
    if hit is not None and hit[0] is chosen:
        return hit[1]
    broken = _BROKEN.get(key)
    if broken is not None:
        raise broken
    _READS[key] = (chosen, _compute(name, chosen))
    return _READS[key][1]


def _payload_of(slot):
    """一个卡位读数 → 载荷（`Exact` / `Unknown`）。顺带挡掉那两个不该出现的形态。"""
    contracts = _mod("ark.contracts")
    if isinstance(slot, contracts.Unsampled):
        raise AssertionError("出现了「未采样」—— 这一层只产精确 / 未知")
    if isinstance(slot, getattr(contracts, "SampledEmpty", ())):
        raise AssertionError("出现了「已采样但为空」—— 这一层只产精确 / 未知")
    if not isinstance(slot, contracts.Valued):
        raise AssertionError(f"卡位的读数不是 Valued，是 {type(slot).__name__}")
    return slot.payload


def _names(slots) -> list[str]:
    contracts = _mod("ark.contracts")
    out: list[str] = []
    for slot in slots:
        payload = _payload_of(slot)
        out.append(str(payload.value) if isinstance(payload, contracts.Exact) else UNKNOWN)
    return out


def _compare(name: str) -> str:
    """逐位比对一帧，返回证据行；不一致就 AssertionError。

    **红罩卡那几位按新规则核**（`_red_slot_complaint`）：`TRUTH` 里那几个「未知」是
    "红罩一律读不出来"那一版写下的**旧账**，拿它们跟实测比，会把一次**读对了**判成错。
    其余每一位**照旧逐位严比** —— 红罩卡的名单从取材表来，剩下的一位都不放松。
    """
    want = list(TRUTH[name])
    got = _names(_read(name))
    if len(got) != len(want):
        raise AssertionError(f"{name}：读到 {len(got)} 个卡位，真值有 {len(want)} 位")
    covered = set(_covered().get(name, ()))
    wrong: list[str] = []
    for index in range(len(want)):
        if index in covered:
            complaint = _red_slot_complaint(name, index, got[index])
            if complaint is not None:
                wrong.append(f"#{index}（红罩卡）{complaint}")
            continue
        if want[index] != got[index]:
            wrong.append(f"#{index} 真值={want[index]} 读到={got[index]}")
    if wrong:
        raise AssertionError(f"{name}：{len(wrong)} 位不符 —— " + " · ".join(wrong))
    hits = sum(1 for value in want if value != UNKNOWN)
    exact = sum(1 for value in got if value != UNKNOWN)
    red = len(covered & set(range(len(want))))
    return (
        f"{name} {len(want)} 位逐位一致（读数给 {exact} 个确切值，其中 {hits} 位在真值表里"
        f"是确切值；{red} 位红罩卡按新规则核）"
    )


def _covered() -> dict[str, tuple[int, ...]]:
    """红罩卡的下标 —— 从夹具那份取材.json 读，**不手抄**。"""
    harvest = json.loads((BATCH / "取材.json").read_text(encoding="utf-8"))
    out: dict[str, tuple[int, ...]] = {}
    for item in harvest["harvest"]:
        out[Path(item["file"]).stem] = tuple(d["index"] for d in (item.get("redeploying") or []))
    return out


# ---------------------------------------------------------------------------
# 红罩卡那一格的**新规则**（D-全局-34）：读得出 ⇒ 精确 · 读不出 ⇒ 未知
#
# 真值只有一处来源：`红罩/点名.json` 的 `truth` 那一栏 —— 那里有 5 张红罩卡是
# **真值已立**的（`totals.truth = 5` / `truth_ok = 5`，全部读对）。**不手抄**。
# ---------------------------------------------------------------------------

REDCAP = REPO / "tests" / "帧样本" / "红罩"
"""红罩那一批夹具 —— 红罩卡的真值住在这里。"""

REDCAP_TABLE = REDCAP / "点名.json"
"""红罩那一批的读数表：`truth` 有值的那些位就是**真值已立**的红罩卡。"""


@cache
def _established_truth() -> dict[tuple[str, int], str]:
    """红罩卡里**真值已立**的那些位：`(帧, 下标) → 干员名`（一个都不手抄）。

    "红罩卡读得出 ⇒ 精确，而且必须真的是那一位"这条规则的上限那一半，就是拿它核的
    —— N4 与 N12 核形态与来路，N20 拿它核**这 5 张线上读得对不对**。表里一条都没有
    的话，这一半就没法核了，所以那是**报错**，不是"跳过"。
    """
    record = json.loads(REDCAP_TABLE.read_text(encoding="utf-8"))
    out: dict[tuple[str, int], str] = {}
    for page in record["frames"]:
        for slot in page["slots"]:
            name = slot.get("truth")
            if name:
                out[(str(page["name"]), int(slot["index"]))] = str(name)
    if not out:
        raise AssertionError(
            f"`{REDCAP_TABLE.name}` 里一条已立的真值都没有 —— 红罩卡那条规则的"
            "上限那一半（精确必须真的是那一位）就没法核了"
        )
    return out


@cache
def _library_names() -> frozenset[str]:
    """库里有哪几个名字 —— 红罩卡读出**库外**的名字就是假事实。"""
    return frozenset(name for name, _ in _faces())


def _red_slot_complaint(frame_name: str, index: int, got: str) -> str | None:
    """红罩卡那一位按**新规则**核一遍：不对给一句理由，对给 `None`。

    新规则（推翻"红罩必然未知"）：**读得出 ⇒ 精确 · 读不出 ⇒ 未知**。精确那一支的
    名字要有来路 —— **真值已立就必须真的是那一位**，真值没立至少也要**在库里**
    （库里没有的名字是假事实）。「未采样 / 已采样但为空」这两种形态由 `_payload_of`
    挡在门外：走到这里的读数一律是 `Valued`。
    """
    if got == UNKNOWN:
        return None
    truth = _established_truth().get((frame_name, index))
    if truth is not None and got != truth:
        return f"真值已立是「{truth}」，却读成「{got}」"
    if truth is None and got not in _library_names():
        return f"读成「{got}」，而库里没有这个名字 —— 那就是假事实"
    return None


# ---------------------------------------------------------------------------
# N 族
# ---------------------------------------------------------------------------


def check_the_roster_agrees_with_its_manifest() -> str:
    """名字表、切割顺序、块数三者自洽；而且写明"与战斗帧不同源"。"""
    man = _manifest()
    names = list(man["names"])
    if len(names) != len(set(names)):
        raise AssertionError(f"名字表里有重名：{sorted(names)}")
    arts = _mod("ark.offline.assets_gen.cardface").cut_formation_arts(
        _mod("ark.offline.assets_gen.cardface").load_formation_page(REPO / man["frame"])
    )
    if len(arts) != len(names):
        raise AssertionError(f"切出 {len(arts)} 块立绘，名字有 {len(names)} 个")
    rows = man["geometry"]["rows"]
    if man.get("same_run_as_battle_frames") is not False:
        raise AssertionError("这一批的库与战斗帧**不同源**，manifest 没写明")
    return (
        f"{len(names)} 个名字互不相同 ｜ 切出 {len(arts)} 块立绘"
        f"（{arts[0].shape[1]}×{arts[0].shape[0]}）｜ 两行 {[r['count'] for r in rows]} 张"
        f" ｜ 同源={man['same_run_as_battle_frames']}"
    )


def check_the_harvest_frames_read_exactly() -> str:
    return " ｜ ".join(_compare(name) for name in ("shot01", "cd60"))


def check_the_acceptance_frames_read_exactly() -> str:
    return " ｜ ".join(_compare(name) for name in ("shot05", "shot26", "cd01", "cd08"))


def check_a_covered_card_is_exact_or_unknown_never_unsampled() -> str:
    """红罩卡那一位的**新规则**：读得出 ⇒ 精确 · 读不出 ⇒ 未知。

    「红罩必然未知」已经被推翻（D-全局-34）：红罩卡换一块取区（环带以内的圆盘 −
    中央秒数那一条），全量那一批 28 张里 16 张读得出，而**真值已立的 5 张全部读对**
    （`红罩/点名.json`）。所以这一条不再要求它落未知，只要求**形态与来路**。

    **什么会让这一条红**：

    - 红罩卡读出**库里没有**的名字 —— 假事实，这一层最坏的一种；
    - 红罩卡读出的名字与**已立的真值**不一致（真值在 `红罩/点名.json` 里，不手抄）；
    - 红罩卡落成「未采样」/「已采样但为空」，或者一个不是 `Valued` 的形态
      （那两种形态由 `_payload_of` 挡着，真报出来就是形态出问题了）；
    - 三帧里一张红罩卡都数不出来（那是取材表烂了，不是它通过了）。

    **它不问"读出了几张"**：读得出几张是画面的事（那批 28 张 → 16 张），不是判据的
    事。判据只管形态，以及精确那一支的名字**有没有来路**。
    """
    covered = _covered()
    exact = unknown = 0
    for name, bad in sorted(covered.items()):
        if not bad:
            continue
        got = _names(_read(name))
        for index in bad:
            if index >= len(got):
                raise AssertionError(f"{name} 只读到 {len(got)} 个卡位，红罩下标却到 {index}")
            complaint = _red_slot_complaint(name, index, got[index])
            if complaint is not None:
                raise AssertionError(f"{name} 位 {index}（红罩卡）{complaint}")
            if got[index] == UNKNOWN:
                unknown += 1
            else:
                exact += 1
    frames = [n for n, b in covered.items() if b]
    if not frames:
        raise AssertionError("三帧里一张红罩卡都没有 —— 这一条量不出东西，不是它通过了")
    # `_payload_of` 那两行已经把「未采样 / 已采样但为空」挡在门外
    return (
        f"{len(frames)} 帧 · {exact + unknown} 张红罩卡：读得出 {exact} 张落精确"
        f"（名字有来路）· 读不出 {unknown} 张落未知（不带分数）· 形态不是未采样"
    )


def check_a_card_outside_the_library_is_unknown() -> str:
    """库里没有这个人（含装置卡）必须落未知，而且**不许**出现"装置"这种词。"""
    seen: list[str] = []
    for name in ("shot01", "cd60"):
        got = _names(_read(name))
        for index in NOT_OPERATOR.get(name, ()):
            if got[index] != UNKNOWN:
                raise AssertionError(f"{name} 位 {index} 是自带装置，却读成 {got[index]}")
            seen.append(f"{name}#{index}")
    vocabulary = {"装置", "召唤物", "未知", "unknown", "None"}
    for name in ("shot01", "cd60"):
        for value in _names(_read(name)):
            if value != UNKNOWN and value in vocabulary:
                raise AssertionError(f"{name} 读出了「{value}」—— 那是个词表外的东西，不是干员名")
    return f"{len(seen)} 个装置位全部落未知 ｜ 读数里没出现词表外的东西"


def check_no_two_cards_get_the_same_name() -> str:
    """一帧之内不许两个卡位给同一个名字（**不需要真值**，专抓"挑最像的那个"）。"""
    worst = 0
    for name in ALL_FRAMES:
        got = [v for v in _names(_read(name)) if v != UNKNOWN]
        if len(got) != len(set(got)):
            raise AssertionError(f"{name} 里同一个名字被点了两次：{got}")
        worst = max(worst, len(got))
    return f"{len(ALL_FRAMES)} 帧 · 每帧之内确切值两两不同（最多的一帧给了 {worst} 个名字）"


def check_an_empty_library_is_a_loud_error() -> str:
    """库是空的：**必须响亮报错**，不许静默吐一列未知。

    "空库"不是"库里没有这个人" —— 它是**没建库**。两者落成同一个样子，就是把一次
    配置错误伪装成一次正常的"认不出来"（同**D9**：表没建好 ≠ 这张卡没有倒计时）。
    这条原来写反了（我本来断言空库→全未知），是执行者那条实现更对。
    """
    roster = _mod("ark.engine.vision.roster")
    try:
        got = roster.read_squad_identities(_frame("cd60"), (), pitch=_pitch())
    except Exception as loud:  # noqa: BLE001 —— 要的就是"炸出来"，类型不钉死
        message = str(loud).strip()
        if not message:
            raise AssertionError(f"报错了但一个字都没说：{type(loud).__name__}") from None
        return f"空库 → {type(loud).__name__}：{message[:60]}"
    raise AssertionError(f"空库竟然没报错，吐了 {len(got)} 个卡位：{[str(v)[:28] for v in got[:2]]}")


def check_every_redeploying_card_is_read_by_the_new_rule() -> str:
    """凡"再部署中"的卡位，身份必须**读得出 ⇒ 精确 · 读不出 ⇒ 未知** —— **跑全 10 帧**。

    "哪一张在再部署中"不借 `取材.json`（那是交付时给执行者的，用在它身上就成了
    "按帧按下标查表"的漏洞），而是**问倒计时那条已经验收过的链**（`read_squad_row`
    的 `redeploying`，R 族与 D 族各自钉过）。所以这条检查**帧是执行者从没拿到的
    4 帧也在里面**，靠硬编码过不去。判不了的（`None`）跳过。

    它与 N4 是**同一枚硬币的两面**：同一条新规则，但两边的红罩名单**各自独立**
    （一边来自取材表、一边来自倒计时那条链），所以两条互为对方的旁证。
    红罩卡换取区之后**不再"一律未知"** —— 原来那条结论（凡再部署中必落未知）已经作废。

    **什么会让这一条红**：倒计时那条链说在再部署中的卡位上，身份读出了**库里没有**
    的名字、或者与已立真值不一致（那两条都由 `_red_slot_complaint` 核）；两条链数出来
    的卡位数不一样也当场红。
    """
    bench = _mod("ark.engine.vision.bench")
    glyphs = _mod("ark.engine.vision.glyphs").GlyphTable.load(
        REPO / "assets" / "features" / "glyphs"
    )
    total = 0
    frames_hit = 0
    exact = unknown = 0
    for name in ALL_FRAMES:
        cards = bench.read_squad_row(_frame(name), glyphs, pitch=_pitch())
        got = _names(_read(name))
        if len(cards) != len(got):
            raise AssertionError(f"{name}：倒计时链读 {len(cards)} 个卡位，点名读 {len(got)} 个")
        flagged = 0
        for card, value in zip(cards, got):
            if card.redeploying is not True:
                continue
            flagged += 1
            complaint = _red_slot_complaint(name, card.index, value)
            if complaint is not None:
                raise AssertionError(f"{name} 位 {card.index} 在再部署中（红罩）：{complaint}")
            if value == UNKNOWN:
                unknown += 1
            else:
                exact += 1
        total += flagged
        frames_hit += 1 if flagged else 0
    return (
        f"{len(ALL_FRAMES)} 帧 · {frames_hit} 帧上有再部署中的卡 · 共 {total} 张："
        f"读得出 {exact} 张落精确 · 读不出 {unknown} 张落未知（形态不是未采样）"
    )



def check_the_margin_line_really_bites() -> str:
    """把一条**孪生条目**塞进库：最像与次像几乎同分 → 必须落未知。

    这一条盯的是判据里"次高要退开"那半条（`required_runner_up`）—— 只看最高分时，
    孪生条目的最高分照样过线，判据会**挑一个**。孪生块 = 真块横移 2 px、换个名字。
    """
    got = _names(_read("cd01", faces=_twin_faces(), tag="孪生库"))
    if got[4] != UNKNOWN:
        raise AssertionError(f"cd01 位 4 本该落未知，却读成 {got[4]}（孪生条目没被余量挡住）")
    return "孪生库 · cd01 位 4 落未知（基线里它是白面鸮，余量 0.6360）"


def check_the_scale_step_really_searches() -> str:
    """尺度那一步真的在扫：库整体放大 3% 之后，还挪得动的那些仍必须认得出。

    `214 ÷ 204 = 1.049` 那个尺度关系一变，能认出来的当前提是链在**扫尺度**；
    只试 1.00 的链会整帧落空。**只对"放大 3% 之后仍落在区间里"的卡断言** ——
    贴着上界的那些本来就会被推出区间，不该拿来判这条。所以筛法是
    `基线上的尺度 × 1.03 ≤ IDENTITY_SCALE_MAX`：**区间一挪它自动跟着走**，不写死数。

    **基线尺度是按"一条库、一档尺度"问回来的**，没重跑一遍全库扫：`roster_scores`
    是**一条一条算**的，条目之间互不影响，所以这样量到的尺度与全库扫出来的
    **是同一个数**，而花的钱是 1/11。名字本身来自 `_read("cd60")`（缓存里已有）。
    """
    roster = _mod("ark.engine.vision.roster")
    contracts = _mod("ark.contracts")
    bench = _mod("ark.engine.vision.bench")
    frame = _frame("cd60")
    pitch = _pitch()
    library = dict(_faces())
    scale_max = float(_attr("IDENTITY_SCALE_MAX"))
    baseline: list[tuple[str, float]] = []
    for (centre, _), slot in zip(bench.squad_card_centres(frame, pitch=pitch), _read("cd60")):
        payload = _payload_of(slot)
        if not isinstance(payload, contracts.Exact):
            baseline.append((UNKNOWN, 0.0))
            continue
        name = str(payload.value)
        if name not in library:
            raise AssertionError(f"点名读出了「{name}」，库里却没有这一条")
        region = np.ascontiguousarray(frame.crop(roster.identity_region(centre, pitch=pitch)))
        _score, got_name, scale = roster.roster_scores(region, ((name, library[name]),))[0]
        baseline.append((got_name, scale))
    movable = [
        (index, name, scale)
        for index, (name, scale) in enumerate(baseline)
        if name != UNKNOWN
        and scale * 1.03 <= scale_max
        and TRUTH["cd60"][index] != UNKNOWN
    ]
    if len(movable) < 3:
        raise AssertionError(f"只有 {len(movable)} 张还挪得动，这条检查试不出东西")

    got = _names(_read("cd60", faces=_bigger_faces(), tag="放大库"))
    wrong = [(i, n, got[i]) for i, n, _ in movable if got[i] != n]
    if wrong:
        detail = " · ".join(f"#{i} 基线={n} 读成={g}" for i, n, g in wrong)
        raise AssertionError(f"放大 3% 之后认不出来了：{detail}")
    return (
        f"{len(movable)} 张放大 3% 后仍落在区间里（尺度 "
        f"{sorted({round(s, 2) for _, _, s in movable})}，上界 {scale_max:.2f}）"
        f"，放大 3% 后仍全部认出"
    )


def check_the_library_order_does_not_matter() -> str:
    """库倒序：读数逐位不变（它只该按内容认人，不许沾位次）。"""
    want = _names(_read("cd60"))
    got = _names(_read("cd60", faces=_reversed_faces(), tag="倒序库"))
    if got != want:
        diff = [(i, want[i], got[i]) for i in range(len(want)) if want[i] != got[i]]
        raise AssertionError(f"换了个顺序结果就变了：{diff}")
    return f"库倒序 · {len(want)} 位读数逐位不变"


def check_only_two_forms_come_out() -> str:
    """这一层**只产两种形态**：精确 / 未知。没有「未采样」、也没有「已采样但为空」。"""
    contracts = _mod("ark.contracts")
    exact = unknown = 0
    for name in ALL_FRAMES:
        for slot in _read(name):
            payload = _payload_of(slot)
            if isinstance(payload, contracts.Exact):
                if not isinstance(payload.value, str) or not payload.value:
                    raise AssertionError(f"{name} 的精确值不是个名字：{payload.value!r}")
                exact += 1
            elif isinstance(payload, contracts.Unknown):
                unknown += 1
            else:
                raise AssertionError(f"{name} 出现了第三种载荷：{type(payload).__name__}")
    return f"{len(ALL_FRAMES)} 帧 · 精确 {exact} 格 · 未知 {unknown} 格 · 没有第三种形态"


# ---------------------------------------------------------------------------
# N13：规格那张几何表 ↔ 产卡面那一侧的常量
#
# **这一条不进池子**：它只读一份文档与几个常量，没有帧、没有库、没有匹配，
# 快到看不见。它补的是**上一道闸门不看的那一半** —— 产卡面那侧的
# `_check_geometry` 管「清单 ↔ 常量」，这一条管「规格文档 ↔ 常量」。两张网都要有，
# 因为**执行者读的是文档**：文档写错一个数，下一个实现的人就照着错的数写。
# ---------------------------------------------------------------------------

CARD_FACE_DOC = REPO / "docs" / "10-规格" / "资产生成器" / "卡面与指纹.md"
"""《资产生成器/卡面与指纹》§1 末那张几何表所在的文档。

**提成模块级名字是为了能把它指到一份副本上**：这一条有一半的价值在"文档那一侧
被改坏也会红"，而验证那一半只能靠换个路径指过去。
"""

_GEOMETRY_SECTION = "### 编队页的卡片坐标（实测，原生 2800×1260）"
"""那张几何表所在的那一节（§1 末那个三级标题，整行逐字）。

**行集合那一条断言按它切段**：从这一行往下，到下一个以 `#` 开头的行（或文件末）。
"""

_GEOMETRY_ROW_ANCHORS = ("卡左缘", "卡距", "卡里那块", "上行", "下行")
"""那张表每一行的行首（第一个单元格的前缀）—— **拿它当锚点找行，不按行号**。

行号会随文档增删而漂，锚点不会。这一节里的**数据行必须恰好是这五个**：
多一行（文档声明了一个代码里没有的几何量）与少一行（锚点改了名）都当场红。
"""

_LABEL_PATTERN = re.compile(r"（\s*(\d+)\s*张\s*）")
"""行标签里的张数：`（6 张）` 那种 —— 张数写在**标签**里，不在正文里。"""

_HEIGHT = re.compile(r"(\d+)\*{0,2}\s*高")
"""「数 高」那种写法 —— `\\*{0,2}` 是容 markdown 的强调记号：表里写的是 `**480** 高`。"""

_GEOMETRY_PATTERNS: tuple[tuple[str, tuple[tuple[re.Pattern[str], str], ...]], ...] = (
    ("卡左缘", (
        (re.compile(r"(\d+)\s*\+\s*(\d+)\s*k"), "「数 + 数 k」那种写法"),
    )),
    ("卡距", (
        (re.compile(r"(\d+)\s*·\s*(\d+)\s*·\s*(\d+)"), "三个数用 `·` 连起来的写法"),
    )),
    ("卡里那块", (
        (re.compile(r"(\d+)\s*宽"), "「数 宽」"),
        (re.compile(r"(\d+)\s*px"), "「数 px」"),
    )),
    ("上行", (
        (re.compile(r"(\d+)\s*[–—-]\s*(\d+)"), "「数–数」那种跨度写法"),
        (_HEIGHT, "「数 高」"),
    )),
    ("下行", (
        (re.compile(r"(\d+)\s*[–—-]\s*(\d+)"), "「数–数」那种跨度写法"),
        (_HEIGHT, "「数 高」"),
        (re.compile(r"行距\s*(\d+)"), "「行距 数」"),
    )),
)
"""每一行**按语义**读那几个数：每种写法在它那一格里必须**恰好命中一次**。

**为什么按行写死，而不是"取单元格里的整数"**：后者要配一条**文档里没写过的元约定**
才排得掉散文里的数字（`（`k` 从 0 数）` 里那个 `0`），而谁都不知道自己踩了那条约定；
一条会因为一句散文换了说法就红的检查，迟早会被开白名单或者被删掉 ——
**假警报把红变成噪声，比漏一条更坏**。

按行认之后，红的两种情形也分得开了：**写法变了**（模式命中不到、或不止一次）与
**数不一样**（命中了、值对不上）。两种都当场红，但报的是两件事。

**这些模式是照着这张表现在的写法写的**（`480` 那个数被 `**` 夹着，`_HEIGHT` 容了它）。
写法真变了就该红 —— 那是要人来决定"这算不算同一件事"的时候，不是可以安静过去的事。
"""

_SEPARATOR_CELL = re.compile(r":?-{2,}:?")


def _is_separator(cells: Sequence[str]) -> bool:
    """`| --- | --- |` 那种分隔行（表头就贴在它上面那一行）。"""
    return bool(cells) and all(_SEPARATOR_CELL.fullmatch(cell) for cell in cells)


def _cell_numbers(
    anchor: str, cell: str, patterns: Sequence[tuple[re.Pattern[str], str]]
) -> list[int]:
    """一个单元格 → 按 `patterns` **按语义**读出来的整数（按出现顺序）。

    每种写法**恰好命中一次**：命中不到或命中多次都当场红，报的是"这一行的**写法**
    变了"，与"数不一样"分开报。
    """
    numbers: list[int] = []
    for pattern, shape in patterns:
        hits = list(pattern.finditer(cell))
        if len(hits) != 1:
            raise AssertionError(
                f"「{anchor}」那一行不再按被钉住的那种写法报数："
                f"没找到{shape}（`{pattern.pattern}`）正好一次，命中 {len(hits)} 次 —— "
                f"要么换了个写法，要么这一格里没有那个数。单元格是 {cell!r}"
            )
        match = hits[0]
        numbers.extend(int(match.group(group)) for group in range(1, pattern.groups + 1))
    return numbers


def _section_lines() -> list[str]:
    """那一节的行：从 `_GEOMETRY_SECTION` 那一行往下，到下一个 `#` 行或文件末。"""
    lines = CARD_FACE_DOC.read_text(encoding="utf-8").splitlines()
    for position, line in enumerate(lines):
        if line.strip() == _GEOMETRY_SECTION:
            break
    else:
        raise AssertionError(
            f"{CARD_FACE_DOC.name} 里找不到标题为「{_GEOMETRY_SECTION}」的那一节："
            "标题改了名，这一条按节切段就切不出来"
        )
    section: list[str] = []
    for line in lines[position + 1 :]:
        if line.startswith("#"):
            break
        section.append(line)
    return section


def _geometry_rows() -> dict[str, tuple[str, str]]:
    """那一节的几何表 → `{行首锚点: (行标签, 正文单元格)}`。

    **数据行恰好就是那五个锚点**（表头行与 `| --- |` 分隔行排掉之后）：
    多一行是"文档声明了一个代码里没有的几何量"，少一行是"锚点改了名"，
    两种都当场红 —— 少了会**安静地少比一行**，那是这一条最坏的烂法。
    """
    table = [line.strip() for line in _section_lines() if line.strip().startswith("|")]
    if not table:
        raise AssertionError(
            f"{CARD_FACE_DOC.name} 那一节里一张表都没有：这一条的出处就是那张几何表"
        )
    data: list[list[str]] = []
    for line in table:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if _is_separator(cells):
            if data:  # 分隔行上面那一行是表头（`| 项 | 实测 |`），表头不算数据行
                data.pop()
            continue
        data.append(cells)
    found: dict[str, tuple[str, str]] = {}
    for cells in data:
        if len(cells) < 2:
            raise AssertionError(
                f"{CARD_FACE_DOC.name} 那一节里有一行只有 {len(cells)} 格：{cells!r} —— "
                "那张表是「项 | 实测」两列"
            )
        head = cells[0]
        for anchor in _GEOMETRY_ROW_ANCHORS:
            if head.startswith(anchor):
                if anchor in found:
                    raise AssertionError(
                        f"{CARD_FACE_DOC.name} 那一节里「{anchor}」那一行出现了两次："
                        "同一行两处各说一套，比不出来是哪一套"
                    )
                found[anchor] = (head, cells[1])
                break
    extra = [
        head
        for head in (cells[0] for cells in data)
        if not any(head.startswith(anchor) for anchor in _GEOMETRY_ROW_ANCHORS)
    ]
    missing = [anchor for anchor in _GEOMETRY_ROW_ANCHORS if anchor not in found]
    if extra:
        tail = f"；另外少了 {'、'.join(missing)} 那一行" if missing else ""
        raise AssertionError(
            f"{CARD_FACE_DOC.name} 那一节多了一行「{extra[0]}」："
            "文档声明了一个代码里没有的几何量 —— 要么把那行删掉，"
            f"要么给代码加常量、再来改这条检查{tail}"
        )
    if missing:
        raise AssertionError(
            f"{CARD_FACE_DOC.name} 那一节里找不到 {'、'.join(missing)} 那一行："
            "行首改了名，这条检查就瞎了一半（这一节的数据行应当恰好是那五个锚点）"
        )
    return found


def _constant(cardface: object, name: str) -> Any:
    """读产卡面那一侧的一个公开常量。

    **名字还不存在要是一条能看懂的 AssertionError，不是 `AttributeError`**：
    `FORMATION_CARD_WIDTH` 与 `FORMATION_GAP` 是纠正清单 5 才加的，这一条要它们
    才能把规格那张表钉住 —— 直接取属性的话，"常量还没落地"会伪装成"检查写错了"。
    """
    value = getattr(cardface, name, None)
    if value is None:
        raise AssertionError(
            f"`{name}` 还不存在：这一条要它才能把规格那张表钉住 —— 纠正清单 5 加的就是它"
        )
    return value


def check_the_geometry_table_matches_the_constants() -> str:
    """《卡面与指纹》§1 末那一节那张几何表里的**每一个数**都必须等于产卡面那一侧的常量。

    **为什么这一条在 N 族里独一份**：别的条看的都是"跑起来对不对"，这一条看的是
    **文档与代码说不说同一件事**。这个坑栽过一次：下行下缘写成 `1146`（真值 `1122`，
    多吞了卡下面那条队名栏），而**清单与常量当时两边同错**，所以产卡面那侧的
    `_check_geometry` 一声不吭 —— 那张表与常量之间当时**一点检查都没有**。

    期望值**一律从常量算出来**，一个都不手抄（手抄的数字会静默过期）。表里的
    派生值也一起钉：`480`（行高）、`505`（行距）、`48`（右边那条底纹的宽）。

    红有三种，报的是三件事：**写法变了**（那一行不再按被钉住的写法报数）、
    **数不一样**（写法和常量对不上）、**这一节的行变了**（多一行是"文档声明了一个
    代码里没有的几何量"，少一行是"锚点改了名"）。
    """
    cardface = _mod("ark.offline.assets_gen.cardface")
    left0 = _constant(cardface, "FORMATION_LEFT0")
    pitch = _constant(cardface, "FORMATION_PITCH")
    card_width = _constant(cardface, "FORMATION_CARD_WIDTH")
    gap = _constant(cardface, "FORMATION_GAP")
    art_width = _constant(cardface, "FORMATION_ART_WIDTH")
    rows = tuple(_constant(cardface, "FORMATION_ROWS"))
    if len(rows) != 2 or any(len(row) != 3 for row in rows):
        raise AssertionError(f"`FORMATION_ROWS` 要是两行 `(上缘, 下缘, 张数)`，收到 {rows!r}")
    (t0, b0, n0), (t1, b1, n1) = rows

    # 表的行序：每一行正文里要报的数（按出现顺序），以及它等于常量里的哪个式子。
    # **顺序必须与 `_GEOMETRY_PATTERNS` 抓出来的顺序一致** —— 下面有一条自检钉它。
    want: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = (
        ("卡左缘", (
            ("FORMATION_LEFT0（卡左缘）", left0),
            ("FORMATION_PITCH（`271k` 那个系数）", pitch),
        )),
        ("卡距", (
            ("FORMATION_PITCH", pitch),
            ("FORMATION_CARD_WIDTH", card_width),
            ("FORMATION_GAP", gap),
        )),
        ("卡里那块", (
            ("FORMATION_ART_WIDTH", art_width),
            ("FORMATION_CARD_WIDTH - FORMATION_ART_WIDTH（右边那条底纹）",
             card_width - art_width),
        )),
        ("上行", (
            ("FORMATION_ROWS[0] 上缘", t0),
            ("FORMATION_ROWS[0] 下缘", b0),
            ("FORMATION_ROWS[0] 高（下缘 - 上缘）", b0 - t0),
        )),
        ("下行", (
            ("FORMATION_ROWS[1] 上缘", t1),
            ("FORMATION_ROWS[1] 下缘", b1),
            ("FORMATION_ROWS[1] 高（下缘 - 上缘）", b1 - t1),
            ("行距（下行上缘 - 上行上缘）", t1 - t0),
        )),
    )
    # 张数写在**行标签**的括号里（`（6 张）`），不在正文里 —— 也要一起钉住
    counted: tuple[tuple[str, int], ...] = (("上行", n0), ("下行", n1))

    if tuple(anchor for anchor, _ in want) != tuple(
        anchor for anchor, _ in _GEOMETRY_PATTERNS
    ):
        raise AssertionError(
            "这条检查自己写歪了：期望值那张表与 `_GEOMETRY_PATTERNS` 的行对不上 —— "
            "对不上就会**安静地少比一行**"
        )
    patterns_of = dict(_GEOMETRY_PATTERNS)

    cells = _geometry_rows()
    body_checked = 0
    label_checked = 0
    for anchor, columns in want:
        _label, body = cells[anchor]
        numbers = _cell_numbers(anchor, body, patterns_of[anchor])
        if len(numbers) != len(columns):
            raise AssertionError(
                f"「{anchor}」那一行按被钉住的那些写法读出来 {len(numbers)} 个数 {numbers}，"
                f"而期望值那一列有 {len(columns)} 个 —— 要么这一行换了写法"
                "（多写或少写了一个量，例如把别的行的内容搬了过来），"
                f"要么这条检查自己写歪了。单元格是 {body!r}"
            )
        for position, ((source, expected), number) in enumerate(zip(columns, numbers), 1):
            if number != expected:
                raise AssertionError(
                    f"「{anchor}」那一行第 {position} 个数：文档写 {number}，"
                    f"{source} 算出来是 {expected} —— 执行者读的是文档，"
                    "文档与代码不一致，下一个实现的人会照着文档写出错的实现"
                )
        body_checked += len(numbers)
    for anchor, count in counted:
        numbers = _cell_numbers(
            anchor, cells[anchor][0], ((_LABEL_PATTERN, "「（数 张）」那种写法"),)
        )
        if numbers != [count]:
            raise AssertionError(
                f"「{anchor}」那一行的行标签里写的张数是 {numbers}，"
                f"而 `FORMATION_ROWS` 里是 {count}：张数写在标签里，"
                "少了它这一行只钉住 y、钉不住几张"
            )
        label_checked += len(numbers)

    return (
        f"{CARD_FACE_DOC.name} 那一节那张几何表 {len(want)} 行 · "
        f"{body_checked} 个正文数 + "
        f"{label_checked} 个行标签里的张数 = {body_checked + label_checked} 个数，"
        f"逐个等于产卡面那一侧的常量（左缘 {left0} · 卡距 {pitch} · 卡宽 {card_width} · "
        f"缝 {gap} · 立绘宽 {art_width} · 两行 {list(rows)}）"
    )


# ---------------------------------------------------------------------------
# N14–N17：单条查询那条路（「某一位干员在第几张卡」）
#
# **这一族只用公开面**：`roster.squad_card_scores` · `roster.find_squad_card` ·
# `roster.ROSTER_SOLID_MATCH_SCORE` · `roster.IDENTITY_SCALES`，加上本文件已有的
# 帧、库、真值。它**不从实现反推** —— 要断言的那些话都是从「这条路该是什么」来的：
# 只答两件事、不产否定存在、代价不随库长、判据那条线是全扫那条线。
#
# **两头的夹法（这是这一族的核心）**：
#   - 「认错人」的保证**不再是代数**了。从前的判据是「过一条固定的分线 **并且** 余量过一条
#     固定的余量线」，于是能从 `1 − 余量线 < 分线` **推**出"别人到不了线"；现在那半条换成了
#     `次高 <= required_runner_up(最高)`（随分数走），那个推论**不成立**。
#     分线的依据从"代数"变成**实测的非匹配天花板**（读数住《备战席点名》§单条查询，
#     两处抄本由 N18 对齐）—— 这是**弱化**，要如实说。
#   - 所以这一族的两个头都得自己量：**没人的卡**上整列够不到线（N16），
#     以及**全扫说未知的红罩卡**上单条那条路指到谁（N17）。
# ---------------------------------------------------------------------------


def _attr(name: str):
    """`ark.engine.vision.roster` 上的一个**公开**名字；没有就是一句明确的红。"""
    module = _mod("ark.engine.vision.roster")
    if not hasattr(module, name):
        raise AssertionError(f"`ark.engine.vision.roster.{name}` 还不存在")
    return getattr(module, name)


def _pool_map(jobs: Mapping[Any, Callable[[], Any]]) -> dict[Any, Any]:
    """一批彼此没有依赖的活丢进同一个池子；谁炸了就把它原样抛出去。"""
    with ThreadPoolExecutor(max_workers=_pool_size()) as pool:
        submitted = [(key, pool.submit(job)) for key, job in jobs.items()]
    out: dict[Any, Any] = {}
    for key, job in submitted:
        broken = job.exception()
        if broken is not None:
            raise broken
        out[key] = job.result()
    return out


def _card_regions(frame_name: str) -> list[np.ndarray]:
    """一帧每个卡位的身份区，**次序与 `squad_card_centres` 同一套**（0 = 最右那张）。"""
    frame = _frame(frame_name)
    pitch = _pitch()
    roster = _mod("ark.engine.vision.roster")
    return [
        np.ascontiguousarray(frame.crop(roster.identity_region(centre, pitch=pitch)))
        for centre, _ in _mod("ark.engine.vision.bench").squad_card_centres(
            frame, pitch=pitch
        )
    ]


def _exact_slots(frame_name: str) -> list[tuple[int, str, float]]:
    """全扫给的那些确切值：`(卡下标, 名字, 置信度)`。"""
    contracts = _mod("ark.contracts")
    out: list[tuple[int, str, float]] = []
    for index, slot in enumerate(_read(frame_name)):
        payload = _payload_of(slot)
        if isinstance(payload, contracts.Exact):
            out.append((index, str(payload.value), float(slot.confidence)))
    return out


def check_the_single_path_is_the_same_chain() -> str:
    """**同一条链**：全扫把 `n` 定在第 `i` 张卡，单条查 `n` 必须
    ①第 `i` 张上给的分数与那条置信度**逐位相同** ②而且**只有第 `i` 张过线**。

    这是整族的地基。分数对不上，就说明这条路自己另跑了一遍匹配 —— 那样"两条路一致"
    就不再是同一件事说的，下面那些结论也跟着塌。②那一半同样是判据的内容：
    「至多一张」不成立时它就该答「没能指出」，而不是随便挑一张。
    """
    roster = _mod("ark.engine.vision.roster")
    line = float(_attr("ROSTER_SOLID_MATCH_SCORE"))
    exact = {frame_name: _exact_slots(frame_name) for frame_name in ALL_FRAMES}
    wanted = {
        (frame_name, entry[1]): entry[2]
        for frame_name, entries in exact.items()
        for entry in entries
    }
    if not wanted:
        raise AssertionError("10 帧里一个确切值都没读出来 —— 那不是这条路的事，是上游塌了")
    columns = _pool_map(
        {
            key: (
                lambda f=key[0], n=key[1]: roster.squad_card_scores(
                    _frame(f), n, _faces(), pitch=_pitch()
                )
            )
            for key in wanted
        }
    )
    for (frame_name, name), confidence in sorted(wanted.items()):
        column = columns[(frame_name, name)]
        over = [(score, index) for score, index in column if score >= line]
        if len(over) != 1:
            raise AssertionError(
                f"{frame_name} 查「{name}」：过线的卡有 {len(over)} 张 {over} —— "
                "只有唯一一张过线时才敢给下标，多一张或少一张它都该答「没能指出」"
            )
        where = [index for index, entry_name, _ in exact[frame_name] if entry_name == name]
        if len(where) != 1:
            raise AssertionError(f"{frame_name} 的全扫把「{name}」定在了 {len(where)} 张卡上")
        if over[0][1] != where[0]:
            raise AssertionError(
                f"{frame_name} 查「{name}」指到第 {over[0][1]} 张卡，全扫定的是第 {where[0]} 张"
            )
        if over[0][0] != confidence:
            raise AssertionError(
                f"{frame_name} 查「{name}」在第 {where[0]} 张卡上给 {over[0][0]!r}，"
                f"全扫那条置信度是 {confidence!r} —— 不是同一个数，就说明不是同一条链"
            )
    return (
        f"{len(ALL_FRAMES)} 帧 · {len(wanted)} 个（帧, 名字）对："
        "分数逐位等于全扫那条，而且只有它自己那张卡过线"
    )


def check_the_single_path_answers_on_the_frames_it_never_saw() -> str:
    """**逐位点名**：把那四帧验收帧上库里的每一条逐个问一遍 ——「在第 k 张卡」还是
    「没能指出」，必须与真值逐位对上。

    **只用验收帧**：那四帧的真值不在 `取材.json` 里（执行者手里没有），
    所以这一条不是"照表填"能过的。

    **红罩卡那一格按新规则**：真值表里那几个「未知」是旧账（红罩卡现在读得出），
    所以"没人"那一半只在**真值立得住**的位上问 —— 非红罩位照旧严（真值说未知就是
    没人），红罩位上**只有 `红罩/点名.json` 里真值已立的那几位**算数。真值未立的
    红罩位上答出了下标**不算假事实**（那里可能真的有人，只是这一批没立真值）。

    **什么会让这一条红**：全扫在最上面那几位上与真值（含已立的红罩真值）对不上；
    真值里有的那位，单条那条路指到别人头上或答不出来；真值里没有的名字，单条那条路
    指到了一张**非红罩**的卡上（那就是零假事实那一面被捅穿了）。
    """
    roster = _mod("ark.engine.vision.roster")
    unseen = ("shot05", "shot26", "cd01", "cd08")
    names = [name for name, _ in _faces()]
    answers = _pool_map(
        {
            (frame_name, name): (
                lambda f=frame_name, n=name: roster.find_squad_card(
                    _frame(f), n, _faces(), pitch=_pitch()
                )
            )
            for frame_name in unseen
            for name in names
        }
    )
    established = _established_truth()
    covered = _covered()
    loose = 0
    for frame_name in unseen:
        truth = list(TRUTH[frame_name])
        got = _names(_read(frame_name))
        red = set(covered.get(frame_name, ()))
        for index in range(len(truth)):
            if index in red:
                complaint = _red_slot_complaint(frame_name, index, got[index])
            elif truth[index] != got[index]:
                complaint = f"真值={truth[index]} 读到={got[index]}"
            else:
                complaint = None
            if complaint is not None:
                raise AssertionError(
                    f"{frame_name} 的全扫在这一位上不对（先修上游）：#{index} {complaint}"
                )
        # 「有人」只算真值立得住的那几位：非红罩位的真值 + 红罩位里真值已立的那几个
        firm = {
            index: truth[index]
            for index in range(len(truth))
            if truth[index] != UNKNOWN and index not in red
        }
        for (truth_frame, index), value in established.items():
            if truth_frame == frame_name:
                firm[index] = value
        for name in names:
            where = [index for index, value in firm.items() if value == name]
            answer = answers[(frame_name, name)]
            if len(where) == 1:
                if answer != where[0]:
                    raise AssertionError(
                        f"{frame_name} 查「{name}」给 {answer!r}，真值说他在第 {where[0]} 张卡"
                    )
            elif len(where) > 1:
                raise AssertionError(
                    f"{frame_name} 的真值里「{name}」同时出现在第 {where} 张卡上 —— "
                    "这一条自己写歪了（一位干员一张卡）"
                )
            elif answer is not None and answer not in red:
                raise AssertionError(
                    f"{frame_name} 查「{name}」指到第 {answer} 张卡，"
                    f"而真值里这一帧没有「{name}」，那一位也不是真值未立的红罩卡"
                    " —— 这就是零假事实那一面"
                )
            elif answer is not None:
                loose += 1
    return (
        f"{len(unseen)} 帧（执行者没拿到的）· 逐个问了 {len(unseen) * len(names)} 次 · "
        "与立得住的真值逐位一致（有人的给对下标；没人且那一位不是红罩卡的一律「没能指出」）"
        f" ｜ 真值未立的红罩卡上答出了下标 {loose} 次（那几位没有真值可比）"
    )


def check_no_name_can_reach_the_line_on_a_card_with_nobody_on_it() -> str:
    """**认不出一个不存在的名字**：全扫落未知的每一张卡，**用那条链在这一格上用的
    取区**量整列，看单条那条路会不会在这张卡上认出一个名字。

    **量的是哪一块取区，这一版变了**：红罩卡现在换圆盘那一块（§两块取区），所以拿
    干净的 100×100 方框去量它，量出来的**不是**那条路会看到的数（上一版就是这么量的
    —— 它在红罩卡上量错了地方）。干净卡照旧：那块方框就是链上用的那一块。

    两半**分开断言**，因为两条线把关的方式不同：

    - **干净卡**：全扫说未知 ⇒ 整列最高分够不到线（`≥ 线` 当场红）—— 与上一版逐字一样；
    - **红罩卡**：未知还有第二种来路 —— **分够而余量不够**（判据同时看两半）。这种卡上
      分线是够得到的，而单条那条路**只看分线**，所以它会**多指出一张卡**（`roster.py`
      §两路 把这写明为"至多比全扫多指出一张"，不是假事实）。这一条于是问**结论**：
      那条路指到的那张卡**不能是别人** —— 指到另一张卡才是假事实。

    **什么会让这一条红**：干净卡上整列够到了线；红罩卡上单条那条路把这一格的名字
    指到了**另一张卡**；或者十帧里一张未知卡都没有（那是上游塌了，不是它通过了）。
    """
    roster = _mod("ark.engine.vision.roster")
    bench = _mod("ark.engine.vision.bench")
    line = float(_attr("ROSTER_SOLID_MATCH_SCORE"))
    faces = _faces()
    pitch = _pitch()
    regions = {frame_name: _card_regions(frame_name) for frame_name in ALL_FRAMES}
    targets = [
        (frame_name, index)
        for frame_name in ALL_FRAMES
        for index, value in enumerate(_names(_read(frame_name)))
        if value == UNKNOWN
    ]
    if not targets:
        raise AssertionError("10 帧里一张未知卡都没有 —— 那不是这条路的事，是上游塌了")
    # 哪几格是红罩卡：判据与链上**同一处**（`bench.redeploying`），不借取材表
    red: dict[tuple[str, int], float] = {}
    card_top = float(_mod("ark.contracts").squad_card_rect(0, pitch).y)
    for frame_name in ALL_FRAMES:
        frame = _frame(frame_name)
        for index, (centre, _) in enumerate(bench.squad_card_centres(frame, pitch=pitch)):
            if bench.redeploying(frame, centre, card_top=card_top) is True:
                red[(frame_name, index)] = float(centre)
    clean_targets = [key for key in targets if key not in red]
    red_targets = [key for key in targets if key in red]
    columns = _pool_map(
        {
            key: (lambda f=key[0], i=key[1]: roster.roster_scores(regions[f][i], faces))
            for key in clean_targets
        }
    )
    worst = (-2.0, "", -1)
    for frame_name, index in clean_targets:
        best, who, _ = columns[(frame_name, index)][0]
        if best > worst[0]:
            worst = (best, frame_name, index)
        if best >= line:
            raise AssertionError(
                f"{frame_name} 第 {index} 张卡全扫落的是未知（干净卡），可是整列里「{who}」"
                f"给到 {best:.4f} ≥ 线 {line} —— 单条那条路在这张卡上会认出一个名字，"
                "而这张卡上没有人"
            )
    risky: list[str] = []
    ruler = None  # 尺子只在真遇到红罩卡时才建（与链上同一个做法：建一把要 ~1 s）
    for frame_name, index in red_targets:
        region = np.ascontiguousarray(
            _frame(frame_name).crop(roster.redcap_region(red[(frame_name, index)], pitch=pitch))
        )
        if ruler is None:
            margin_module = _mod("ark.engine.vision.margin")
            _box, disc_mask = margin_module.identity_disc_region()
            ruler = margin_module.MaskedRuler(disc_mask, faces)
        ranked = ruler.scores(region)
        best, who = ranked[0][0], ranked[0][1]
        runner = ranked[1][0] if len(ranked) > 1 else -2.0
        # 单条那条路**只看分线**，所以"分够、而判据不成立"的这一批，正是它会多指出一张的地方。
        # 判据走 `roster.identity_is_exact` —— 它是判据的**唯一定义**，这里不另写一遍。
        if not (best >= line and not roster.identity_is_exact(best, runner)):
            continue
        answer = roster.find_squad_card(_frame(frame_name), who, faces, pitch=pitch)
        if answer is not None and answer != index:
            raise AssertionError(
                f"{frame_name} 第 {index} 张卡（红罩卡）全扫落未知，可是圆盘那一块上"
                f"「{who}」给到 {best:.4f} ≥ 线、次高 {runner:.4f} 却没退到 "
                f"{roster.required_runner_up(best):.4f} 以下；单条那条路只看分线，"
                f"它把「{who}」指到了第 {answer} 张卡 —— 指到别人头上才是假事实"
            )
        risky.append(
            f"{frame_name}#{index} {who} {best:.4f}/次高 {runner:.4f} → "
            + ("没能指出" if answer is None else f"指到他自己（第 {answer} 张卡）")
        )
    clean_line = (
        f"干净卡 {len(clean_targets)} 张整列都够不到线（最险 {worst[0]:.4f} < {line}，"
        f"{worst[1]} 第 {worst[2]} 张卡）"
        if clean_targets
        else "干净卡一张都没有"
    )
    return (
        f"{len(ALL_FRAMES)} 帧 · {len(targets)} 张未知卡（干净 {len(clean_targets)} · "
        f"红罩 {len(red_targets)}）：{clean_line} ｜ 红罩卡上分线够得到的有 {len(risky)} 张，"
        "单条那条路一张都没指到别人头上"
        + ("：" + " · ".join(risky) if risky else "")
    )


def check_the_single_path_cost_does_not_follow_the_library() -> str:
    """**代价不随库长增长**：数 `cv2.matchTemplate` 的次数 —— 库 11 条与库 110 条必须
    **同值**，而且**有上界** `卡位 × 档数 × 库长`。顺带钉住判据那半条线的**形状**。

    **"等于 `卡位 × 尺度档数`"那条已经死了**：单条那条路现在也走三层扫法、判据满足即停
    （实测 12 张卡 × 18 档 = 216，实调 **376**），所以绝对公式不再是它该有的样子；
    活下来的是"有上界"与"不随库长变"两条 —— 那才是"单条"这两个字的意思。

    **形状，不是那条已经退役的代数推论。** 从前这里钉的是 `1 − 余量线 < 分线`，而那半条
    已经换成随分数走的 `次高 <= required_runner_up(最高)` —— 那个推论**不成立了**。
    现在钉它的两条性质：在最高分 1.0 处**连续**（=> `1 − 垫片`，没有特例），
    以及随最高分**单调**。**"分线为什么够"不在这里** —— 那已经换成实测的非匹配天花板，
    住《备战席点名》§单条查询（两处抄本由 N18 对齐）。

    **这一条必须串行跑**：计数器是全局的，池子里有别的活在跑时它就不是一个人的了。
    而它也是这一族里唯一**不能被行为测试替代**的一条 —— 扫完全库再挑出那一条，
    答案一模一样，只有次数看得见。
    """
    cv2 = _mod("cv2")
    roster = _mod("ark.engine.vision.roster")
    line = float(_attr("ROSTER_SOLID_MATCH_SCORE"))
    factor = float(_attr("ROSTER_AMBIGUITY_FACTOR"))
    pad = float(_attr("ROSTER_AMBIGUITY_PAD"))
    if abs(roster.required_runner_up(1.0) - (1.0 - pad)) > 1e-12:
        raise AssertionError(
            f"required_runner_up(1.0) = {roster.required_runner_up(1.0):.6f}，"
            f"而不是 1 − 垫片 = {1.0 - pad:.6f} —— 顶部出现了一个特例"
        )
    if factor <= 0:
        raise AssertionError(f"那半条线的斜率是 {factor} —— 不随最高分单调就说不通")
    if not roster.required_runner_up(line) < line:
        raise AssertionError(
            f"在分线 {line} 处它要求次高 ≤ {roster.required_runner_up(line):.4f}，"
            f"并不比分线低 —— 过了分线就什么都不用退，那半条线等于没有"
        )
    frame_name = "shot01"
    name = "维什戴尔"
    truth_index = list(TRUTH[frame_name]).index(name)
    faces = _faces()
    fat = faces + tuple(
        (f"{entry_name}（副本{copy_index}）", art)
        for copy_index in range(1, 10)
        for entry_name, art in faces
    )
    original = cv2.matchTemplate
    counter = {"calls": 0}

    def counting(*args, **kwargs):
        counter["calls"] += 1
        return original(*args, **kwargs)

    measured: list[tuple[int, int, int | None]] = []
    cv2.matchTemplate = counting
    try:
        for library in (faces, fat):
            counter["calls"] = 0
            got = roster.find_squad_card(
                _frame(frame_name), name, library, pitch=_pitch()
            )
            measured.append((len(library), counter["calls"], got))
    finally:
        cv2.matchTemplate = original

    cards = len(_names(_read(frame_name)))
    scales = len(_attr("IDENTITY_SCALES"))
    for size, calls, got in measured:
        # 上界照《备战席点名》§单条查询 那条写：`卡位 × 档数 × 库长`。
        # 绝对公式（`卡位 × 尺度`）已经死了，所以这里只问"有没有越界"。
        ceiling = cards * scales * size
        if calls > ceiling:
            raise AssertionError(
                f"库 {size} 条：`matchTemplate` 调了 {calls} 次，越过上界 "
                f"卡位 {cards} × 档数 {scales} × 库长 {size} = {ceiling}"
            )
        if got != truth_index:
            raise AssertionError(
                f"库 {size} 条：查到第 {got} 张卡，真值说第 {truth_index} 张"
            )
    if measured[0][1] != measured[1][1]:
        raise AssertionError(
            f"库 11 条与库 110 条的匹配次数不同：{measured[0][1]} 对 {measured[1][1]} —— "
            "随库长增长就不叫单条"
        )
    return (
        f"{frame_name} 查「{name}」：库 11 条与库 110 条都是 {measured[0][1]} 次匹配"
        f"（= 卡位 {cards} × 尺度 {scales}）｜ 判据在 1.0 处连续、"
        f"在分线 {line} 处要求次高 ≤ {roster.required_runner_up(line):.4f}"
    )


# ---------------------------------------------------------------------------
# N18：同一批读数住在两处 —— 逐个数对齐（N13 那一对的另一半）
# ---------------------------------------------------------------------------


ROSTER_SOURCE = REPO / "src" / "ark" / "engine" / "vision" / "roster.py"
"""`roster.py` —— 它模块文档 §两路 里写着同一批读数。"""

ROSTER_DOC = REPO / "docs" / "10-规格" / "视觉层" / "备战席点名.md"
"""《视觉层/备战席点名》§单条查询 —— 那批读数的另一份抄本，**以这一份为准**（画面事实住规格）。"""

_READING_FRAGMENTS: tuple[tuple[str, str], ...] = (
    ("真匹配那一簇", r"真匹配 \*\*([\d.]+)–([\d.]+)\*\*（(\d+) 对）"),
    ("非匹配那一簇", r"非匹配 \*\*≤ ([\d.]+)\*\*（(\d+) 对）"),
    ("空档", r"—— 空档 \*\*([\d.]+)\*\*"),
    ("样本量", r"两帧 (\d+) 张卡、(\d+) 个 \(卡, 条目\) 对"),
    ("分线", r"`ROSTER_SOLID_MATCH_SCORE = ([\d.]+)`"),
)
"""要逐个数对齐的那几组读数，以及它们在**两处**都长什么样。

**写法是判据的一部分**（与 N13 那张几何表同一个道理）：这几句一改，这一条就红 ——
那不是它坏了，是它在说"两处要一起改"。
"""


def _readings(text: str, where: str) -> dict[str, tuple[str, ...]]:
    """把一处文本里的那几组读数抠出来；哪一句找不到、或找到不止一处，当场报。"""
    out: dict[str, tuple[str, ...]] = {}
    for label, pattern in _READING_FRAGMENTS:
        found = re.findall(pattern, text)
        if len(found) != 1:
            raise AssertionError(
                f"{where} 里「{label}」那一句找到 {len(found)} 处，而不是一处 —— "
                "这一条按那几句的写法逐个数对齐，写法变了它就对不上了"
            )
        out[label] = found[0] if isinstance(found[0], tuple) else (found[0],)
    return out


def check_the_readings_are_the_same_in_the_source_and_the_spec() -> str:
    """同一批读数住在**两处**（`roster.py` 模块文档 §两路 与 规格 §单条查询），中间要有网。

    这一条与 N13 是同一对：N13 管「规格那张几何表 ↔ 产卡面那一侧的常量」，
    这一条管「规格那段读数 ↔ 视觉层那份模块文档」。**两处一起改才算改完** ——
    只改一处，下一个人读到的就是两个不一样的数（`tests/spec/README.md` 纪律 3
    那条"静默过期"说的就是这个）。

    **两处的标题不同名是有意的**：规格那边的 `§两路` 说的是**人点名 / 自动点名**，
    而 `roster.py` 那边的 `§两路` 说的是**全扫 / 单条**（自动那条路内部的两个入口）。
    同名不同义会让"以规格为准"变成一句没法执行的话，所以读数那一节单开了一个标题。
    """
    source = _readings(ROSTER_SOURCE.read_text(encoding="utf-8"), ROSTER_SOURCE.name)
    spec = _readings(ROSTER_DOC.read_text(encoding="utf-8"), ROSTER_DOC.name)
    drift = [label for label, _ in _READING_FRAGMENTS if source[label] != spec[label]]
    if drift:
        detail = " · ".join(
            f"「{label}」代码写 {source[label]}，规格写 {spec[label]}" for label in drift
        )
        raise AssertionError(
            f"{len(drift)} 处对不上：{detail} —— 这两处要一起改，而且**以规格为准**"
            "（画面事实住规格）；只改一处，下一个人读到的就是两个不一样的数"
        )
    hits = int(source["真匹配那一簇"][2])
    misses = int(source["非匹配那一簇"][1])
    pairs = int(source["样本量"][1])
    if hits + misses != pairs:
        raise AssertionError(
            f"两簇的对数加起来是 {hits + misses}，而样本量写的是 {pairs} —— "
            "这一条自己写歪了（三处抄本互相不闭合）"
        )
    return (
        f"{ROSTER_SOURCE.name} 与 {ROSTER_DOC.name} 的 {len(_READING_FRAGMENTS)} 组读数逐位相同"
        f" ｜ 真匹配 {hits} 对 + 非匹配 {misses} 对 = {pairs} 对 ｜ "
        f"空档 {source['空档'][0]} · 线 {source['分线'][0]}"
    )


# ---------------------------------------------------------------------------
# N19：立绘两组 —— 「立绘就是身份」
#
# **这一条不进任何池子**：它自己那两帧自己读（与 N14–N17 同一个做法）。它问的是一件
# **新的事**：同一个干员的**两套立绘同时在库里**（20 条）时，读数会不会挑错那一条。
#
# **两条带共用一张卡位表**：两组的人员与位置完全相同（用户给的数据就是这样挑的），
# 所以真值只有**一张**十位的表 —— 不同立绘的那两条是**孪生对**。
# ---------------------------------------------------------------------------

ART_BATCH = REPO / "tests" / "帧样本" / "立绘两组"
"""立绘两组那一批：两组各一张编队页（整帧）＋一条备战席横带（裁剪，见 `index.json`）。"""

ART_GROUPS: tuple[int, ...] = (1, 2)
"""两组的号：库是 `库N.json`、横带是 `组N_备战席.png`、标签是 `组N·干员名`。"""

ART_TRUTH: tuple[str, ...] = (
    "维什戴尔", "圣聆初雪", "荒芜拉普兰德", "温蒂", "塞雷娅",
    "凯尔希·思衡托", "白面鸮", "歌蕾蒂娅", "凛御银灰", "极境",
)
"""**卡位（0 = 最右那张）↔ 干员名**，两条带共用。

**规划者目视给的**（与那一批的真值同一个来路：人读，不从像素推 —— 游戏在编队页卡上
印的是汉字，而这一层不上通用 OCR）。横带上**从左到右**是这张表**倒过来**。
"""


@cache
def _art_libraries() -> dict[int, tuple[tuple[str, np.ndarray], ...]]:
    """两组的库各 10 条：本组编队页上那 10 格立绘，标签是 `组N·干员名`（`库N.json`）。"""
    cardface = _mod("ark.offline.assets_gen.cardface")
    return {
        group: tuple(cardface.load_cardfaces(ART_BATCH / f"库{group}.json"))
        for group in ART_GROUPS
    }


def _art_regions() -> dict[tuple[int, int], np.ndarray]:
    """两条带每个卡位的**身份区** —— 次序与 `squad_card_centres` 同一套（0 = 最右那张）。

    这一批的卡**全是干净卡**（`取材.json` 的 `cards_are_clean`），所以链上走的也是
    这块方框（`roster.identity_region`），不需要分两条路。
    """
    roster = _mod("ark.engine.vision.roster")
    bench = _mod("ark.engine.vision.bench")
    pitch = _pitch()
    out: dict[tuple[int, int], np.ndarray] = {}
    for group in ART_GROUPS:
        frame = _frame(f"组{group}_备战席", ART_BATCH)
        for index, (centre, _) in enumerate(bench.squad_card_centres(frame, pitch=pitch)):
            region = frame.crop(roster.identity_region(centre, pitch=pitch))
            out[(group, index)] = np.ascontiguousarray(region)
    return out


def check_the_illustration_is_the_identity() -> str:
    """**N19** —— 立绘两组：同一个干员的**另一套立绘**同时在库里，读数必须认自己那一套。

    这一批是**两组同 10 位干员、同位置、不同立绘**，库是两组合起来的 **20 条**，口径是
    「**立绘就是身份**」（用户裁定）：读出同一个干员的另一套立绘**算失败**。所以这一条
    同时钉三件事：

    1. **自己那一套认得出**：每条带 10 张卡里**至少 9 张**落精确，而落精确的那几位必须
       正好是**这一条带自己那一组**的那一条（真值见 `ART_TRUTH`）—— 孪生那一条赢一次
       就算失败；
    2. **只有孪生那一条在库里的那 10 条**（把库缩到**另一组**），**20 张卡一张都不许
       落精确**：判据不另写一遍 —— 调 `roster.identity_is_exact`，**它不成立**才算未知，
       成立就成了假事实。**不写成"两个量分别都没够到"**：判据挪一档，那一档就可能变成
       "分够、次高也退开了"，而判据问的从来不是"两个量都低"，是"**没有同时过线**"；
    3. **孪生一次都不许当第一**：20 条库下面，20 张卡的第一名一个都不是另一组那一条。

    判据不在这条检查里另写一遍：一律调 `roster.identity_is_exact(最高, 次高)` ——
    **它是判据的唯一定义**（扫法的停止条件与最后的判定都调它），所以线或那条公式挪一档，
    这一条自动跟着走，不许红。读数走 `roster.roster_scores`（这一批全是干净卡，
    与 `read_squad_identities` 在干净卡上走的是同一条）。

    **什么会让这一条红**：某张卡落精确而第一名是**另一组**的同名条目（立绘张冠李戴）；
    某条带上精确的少于 9 张；把库缩到另一组之后**有卡两半同时过线**；库不是 20 条、
    标签不成对、两条带的卡位数与真值位数对不上；或者某组的库里的干员与真值表不是同一批
    （那就不是孪生对，这一条问的那件事也不成立）。
    """
    roster = _mod("ark.engine.vision.roster")
    line = float(_attr("ROSTER_SOLID_MATCH_SCORE"))
    libraries = _art_libraries()
    library = tuple(entry for group in ART_GROUPS for entry in libraries[group])
    labels = [name for name, _ in library]
    want_labels = len(ART_GROUPS) * len(ART_TRUTH)
    if len(labels) != want_labels or len(set(labels)) != want_labels:
        raise AssertionError(
            f"库有 {len(labels)} 条、{len(set(labels))} 个不同的标签 —— 两组各 "
            f"{len(ART_TRUTH)} 条、合起来 {want_labels} 条，而且标签要互不相同"
            "（同名两条就没有唯一答案）"
        )
    crews = {
        group: frozenset(name.split("·", 1)[-1] for name, _ in libraries[group])
        for group in ART_GROUPS
    }
    if any(len(crew) != len(ART_TRUTH) for crew in crews.values()):
        raise AssertionError(
            f"每一组的库要恰好 {len(ART_TRUTH)} 位干员，数出来是 "
            f"{[len(c) for c in crews.values()]}"
        )
    if len(set(crews.values())) != 1 or crews[ART_GROUPS[0]] != frozenset(ART_TRUTH):
        raise AssertionError(
            f"两组的干员不是同一批（或与真值表不是同一批）："
            f"组1 {sorted(crews[ART_GROUPS[0]])} · 组2 {sorted(crews[ART_GROUPS[1]])} —— "
            "不成孪生对，这一条问的「另一套立绘」就不成立了"
        )

    regions = _art_regions()
    expected = {
        (group, index) for group in ART_GROUPS for index in range(len(ART_TRUTH))
    }
    if set(regions) != expected:
        raise AssertionError(
            f"两条带数出来的卡位是 {sorted(regions)}，而真值表是 10 位 × 2 条带 —— "
            "位数对不上就比不了"
        )
    columns = _pool_map(
        {
            key: (lambda r=region: roster.roster_scores(r, library))
            for key, region in regions.items()
        }
    )

    def twin_of(group: int, index: int) -> str:
        return f"组{3 - group}·{ART_TRUTH[index]}"

    per_group: dict[int, int] = {group: 0 for group in ART_GROUPS}
    for group in ART_GROUPS:
        for index in range(len(ART_TRUTH)):
            ranked = columns[(group, index)]
            best, who = ranked[0][0], ranked[0][1]
            runner = ranked[1][0] if len(ranked) > 1 else -2.0
            if who == twin_of(group, index):
                raise AssertionError(
                    f"组{group} 第 {index} 张卡的第一名是「{who}」—— 那是**另一组**的同名条目。"
                    "立绘就是身份：读出另一套立绘算失败"
                )
            if roster.identity_is_exact(best, runner):
                want = f"组{group}·{ART_TRUTH[index]}"
                if who != want:
                    raise AssertionError(
                        f"组{group} 第 {index} 张卡落精确，认的却是「{who}」，"
                        f"而这一位真值是「{want}」（孪生那一条在库里，立绘就是身份）"
                    )
                per_group[group] += 1
    for group in ART_GROUPS:
        if per_group[group] < len(ART_TRUTH) - 1:
            raise AssertionError(
                f"组{group} 那条带上只有 {per_group[group]}/{len(ART_TRUTH)} 张落精确 —— "
                f"同一批库下至少要 {len(ART_TRUTH) - 1} 张认得出（认得出几张是画面的事，"
                "少到 8 张就说明这条链在自己那一组上已经不稳了）"
            )

    # 库缩到**另一组**那 10 条：卡上自己那套立绘不在库里，只剩孪生那一条
    foreign = _pool_map(
        {
            key: (
                lambda r=region, faces=tuple(libraries[3 - key[0]]): roster.roster_scores(r, faces)
            )
            for key, region in regions.items()
        }
    )
    worst_score = (-2.0, None)
    worst_margin = (-2.0, None)
    for key in sorted(foreign):
        ranked = foreign[key]
        best = ranked[0][0]
        runner = ranked[1][0] if len(ranked) > 1 else -2.0
        margin = best - runner
        if best > worst_score[0]:
            worst_score = (best, key)
        if margin > worst_margin[0]:
            worst_margin = (margin, key)
        # **判据不成立**才落未知 —— 照线上那一条问，不写成"两个量分别都没够到"：
        # 判据问的从来不是"两个量都低"，是"**没有同时过线**"。
        if roster.identity_is_exact(best, runner):
            raise AssertionError(
                f"库缩到另一组之后，组{key[0]} 第 {key[1]} 张卡上**两半都过**：孪生那一条"
                f"给到 {best:.4f} ≥ 线 {line}，次高 {runner:.4f} 也退到了 "
                f"{roster.required_runner_up(best):.4f} 以下"
                "（那张卡上只有**孪生**那一条在库里）—— 那就是拿另一套立绘认出了本人，"
                "成了一个假事实"
            )
    return (
        f"{len(ART_GROUPS)} 组 × {len(ART_TRUTH)} 张卡 · {len(library)} 条库：精确 "
        f"{sum(per_group.values())}/{len(ART_GROUPS) * len(ART_TRUTH)}"
        + "（" + " · ".join(
            f"组{group} {per_group[group]}/{len(ART_TRUTH)}" for group in ART_GROUPS
        ) + "）"
        + "，第一名一次都不是孪生 ｜ 缩到另一组的 10 条："
        f"{len(ART_GROUPS) * len(ART_TRUTH)} 张**两半都没同时过**（最险的分 "
        f"{worst_score[0]:.4f} · 最险的次高差 {worst_margin[0]:.4f}，"
        f"分线 {line} · 在分线处要求次高 ≤ {roster.required_runner_up(line):.4f}）"
    )


# ---------------------------------------------------------------------------
# N20：真值已立的那 5 张红罩卡，线上那条链必须**读对**
#
# N4 / N12 的红罩名单都长在**备战席**那批帧上；而"真值已立"的 5 张住在**红罩**那批帧上
# （cd14 / cd22 / cd23 / cd27 / cd37）—— 所以"5 张全部读对"那句话原本**没有网**。
# 这一条把它变成机器检查：真值从 `红罩/点名.json` 的 `truth` 栏读（不手抄），读数走
# **线上那条链**（`read_squad_identities`，红罩那几格自己会换圆盘取区）。
# ---------------------------------------------------------------------------


def check_the_red_cards_with_established_truth_are_read_correctly() -> str:
    """**N20** —— 红罩卡里**真值已立**的那 5 张，线上那条链必须**读对**。

    N4 与 N12 都够不到这一块：那两条的红罩名单来自**备战席**那批（取材表 / 倒计时链），
    而真值已立的 5 张住在**红罩**那批帧上。这一条把那句"5 张全部读对"变成机器检查 ——
    真值从 `红罩/点名.json` 的 `truth` 栏读（**不手抄**），读数走**线上那条链**
    （`read_squad_identities`：`redeploying` 是 `True` 的那几格它自己会换圆盘取区）。

    **它只射这 5 张**：那几帧上其余的红罩卡这一版**没有真值**，规格证不了名字对不对
    （那些位属于 N4 / N12 那一半：形态与来路），所以不在这条的射程里。

    **什么会让这一条红**：真值已立的某一张读成了**别人**（假事实）；或者读不出来
    （落未知，或落成未采样 / 已采样但为空）；或者 `红罩/点名.json` 里一张已立的真值
    都没有（那这张网就没有被测物了）；或者某帧的卡位数比真值那个下标还少。
    """
    contracts = _mod("ark.contracts")
    roster = _mod("ark.engine.vision.roster")
    truth = _established_truth()
    frames = {name: _frame(name, REDCAP) for name in sorted({f for f, _ in truth})}
    readings = _pool_map(
        {
            name: (
                lambda frame=frames[name]: roster.read_squad_identities(
                    frame, _faces(), pitch=_pitch()
                )
            )
            for name in frames
        }
    )
    lines: list[str] = []
    for (frame_name, index), want in sorted(truth.items()):
        slots = readings[frame_name]
        if index >= len(slots):
            raise AssertionError(
                f"{frame_name} 只读到 {len(slots)} 个卡位，而真值立在第 {index} 张卡上"
            )
        payload = _payload_of(slots[index])
        if not isinstance(payload, contracts.Exact):
            raise AssertionError(
                f"{frame_name} 第 {index} 张卡（红罩卡）真值已立是「{want}」，"
                f"却读不出来（落 `{type(payload).__name__}`）—— 真值已立的那几张必须读得出"
            )
        got = str(payload.value)
        if got != want:
            raise AssertionError(
                f"{frame_name} 第 {index} 张卡（红罩卡）读成「{got}」，真值是「{want}」—— "
                "读出别人就是假事实"
            )
        lines.append(f"{frame_name}#{index} {got} {float(slots[index].confidence):.4f}")
    return (
        f"{len(frames)} 帧（红罩那批）· {len(truth)} 张真值已立的红罩卡全部读对："
        + " ｜ ".join(lines)
    )


CHECKS: tuple[Check, ...] = (
    Check("N1", "资产/卡面与指纹 §1 末（编队页几何那张表）+ 视觉层/备战席点名 §两路",
          "名字表 · 切割顺序 · 块数三者自洽，而且写明库与战斗帧不同源",
          check_the_roster_agrees_with_its_manifest),
    Check("N2", "帧样本/点名/取材.json + 视觉层/备战席点名 §落法",
          "取材那两帧逐位一致（执行者手里只有这两帧）",
          check_the_harvest_frames_read_exactly),
    Check("N3", "视觉层/备战席点名 §已实测 + §判据的形状",
          "验收那四帧逐位一致 —— 它们没进取材，是这一层主要那道网",
          check_the_acceptance_frames_read_exactly),
    Check("N4", "视觉层/备战席点名 §红罩卡 · §两块取区（D-全局-34 推翻「红罩必然未知」）+ §落法",
          "红罩卡：读得出 ⇒ 精确（名字要有来路）· 读不出 ⇒ 未知 · 绝不许未采样或库外的名字",
          check_a_covered_card_is_exact_or_unknown_never_unsampled),
    Check("N5", "视觉层/备战席点名 §落法（比不中 → 未知）+ §装置那一张长什么样",
          "库里没有这个人（含装置卡）落未知，而且不许出现词表外的东西",
          check_a_card_outside_the_library_is_unknown),
    Check("N6", "视觉层/备战席点名 §不许「挑最像的那个」",
          "一帧之内不许两个卡位给同一个名字",
          check_no_two_cards_get_the_same_name),
    Check("N7", "视觉层/备战席点名 §落法 + 识别方法 §已定（D9 那条同款：没建好 ≠ 没有）",
          "库是空的时候必须响亮报错，不许静默吐一列未知",
          check_an_empty_library_is_a_loud_error),
    Check("N8", "视觉层/备战席点名 §判据的形状（同时看最高分与余量）",
          "最像与次像几乎同分时必须落未知 —— 余量那条线真的在起作用",
          check_the_margin_line_really_bites),
    Check("N9", "视觉层/备战席点名 §已实测 后果 1（同一尺度让判据退化成找位置）",
          "库整体放大 3% 之后，还挪得动的那些仍认得出 —— 尺度那一步真的在扫",
          check_the_scale_step_really_searches),
    Check("N10", "视觉层/备战席点名 §每片都点名，不许靠推理顺序",
          "库倒序之后读数逐位不变",
          check_the_library_order_does_not_matter),
    Check("N11", "视觉层/备战席点名 §落法（这一层只产两种形态）",
          "读出来的只有精确与未知两种，没有未采样",
          check_only_two_forms_come_out),
    Check("N12", "视觉层/备战席点名 §红罩卡（红罩 ⟺ 再部署中）+ §两块取区",
          "凡再部署中的卡位：读得出 ⇒ 精确（名字要有来路）· 读不出 ⇒ 未知 —— 跑全 10 帧，含执行者没拿到的 4 帧",
          check_every_redeploying_card_is_read_by_the_new_rule),
    Check("N13", f"资产/卡面与指纹 §1 末（`{_GEOMETRY_SECTION}` 那一节的那张表）",
          "规格那张几何表里的每一个数都等于产卡面那一侧的常量 —— 执行者读的是文档，不是常量",
          check_the_geometry_table_matches_the_constants),
    Check("N14", "视觉层/备战席点名 §单条查询：某一位在第几张卡",
          "单条查一位给的分数与全扫那条置信度逐位相同，而且只有他自己那张卡过线",
          check_the_single_path_is_the_same_chain),
    Check("N15", "视觉层/备战席点名 §单条查询 + §每片都点名，不许靠推理顺序 + §两块取区",
          "四帧验收帧上逐个问库里的每一条：与**立得住的**真值逐位一致（真值未立的红罩卡不算），没人的一律答不出来",
          check_the_single_path_answers_on_the_frames_it_never_saw),
    Check("N16", "视觉层/备战席点名 §不许「挑最像的那个」+ §单条查询 + §落法（比不中 → 未知）+ §两块取区",
          "全扫落未知的卡上，用链上那一格的取区量：干净卡整列够不到线，红罩卡也不会把名字指到别人头上",
          check_no_name_can_reach_the_line_on_a_card_with_nobody_on_it),
    Check("N17", "任务单【做】1 + 视觉层/备战席点名 §单条查询（代价与库长无关）",
          "库 11 条与库 110 条的匹配次数同值且 = 卡位 × 尺度，另加「1 − 余量线 < 分线」",
          check_the_single_path_cost_does_not_follow_the_library),
    Check("N18", f"视觉层/备战席点名 §单条查询（判据与读数）+ `{ROSTER_SOURCE.name}` 模块文档 §两路",
          "那几组读数在两处逐位相同 —— 改一处不改另一处就会红",
          check_the_readings_are_the_same_in_the_source_and_the_spec),
    Check("N19", "帧样本/立绘两组/取材.json · 库1.json · 库2.json（「立绘就是身份」：用户裁定）",
          "两组同 10 位干员、不同立绘：自己那一套认得出 ≥9/10，缩到另一组则 20 张全未知，孪生一次都不当第一",
          check_the_illustration_is_the_identity),
    Check("N20", "帧样本/红罩/点名.json 的 truth 栏 + 视觉层/备战席点名 §红罩卡 · §两块取区",
          "真值已立的那 5 张红罩卡，线上那条链（read_squad_identities）全部读对 —— 读成别人 / 读不出来都红",
          check_the_red_cards_with_established_truth_are_read_correctly),
)


def main() -> int:
    failures: list[tuple[Check, str]] = []
    print(
        f"备战席点名的规格符合性验收 —— {len(CHECKS)} 条"
        f"（13 次全扫 + 单条那条路 + 立绘两组 + 红罩那 5 张，池子 {_pool_size()} 路）\n"
    )
    for check in CHECKS:
        try:
            evidence = check.probe()
        except AssertionError as bad:
            failures.append((check, str(bad)))
            print(f"  ✗ {check.ident}  {check.claim}")
            print(f"      出处 {check.spec}")
            print(f"      证据 {bad}")
        except BaseException as broken:  # noqa: BLE001
            failures.append((check, f"{type(broken).__name__}: {broken}"))
            print(f"  ‼ {check.ident}  {check.claim}")
            print(f"      出处 {check.spec}")
            print(f"      检查本身炸了 {type(broken).__name__}: {broken}")
        else:
            print(f"  ✓ {check.ident}  {check.claim}")
            print(f"      {evidence}")

    print()
    if failures:
        print(f"不成立 {len(failures)} / {len(CHECKS)}：")
        for check, why in failures:
            print(f"  - {check.ident}（{check.spec}）{why}")
        return 1
    print(f"全部成立（{len(CHECKS)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
