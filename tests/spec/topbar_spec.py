"""顶部两个数（生命 · 敌数）读数链的规格符合性验收 —— **由规划者维护，不住在 tests/unit/**。

与 `tests/unit/` 的分工：

| 目录 | 谁写 | 验证什么 |
| --- | --- | --- |
| `tests/unit/` | 执行者 | "实现按我的设计工作" |
| **本文件** | **规划者** | **"规格说的那些话成立"** |

## 两条纪律

1. **只用公开面。** 只 import 视觉层与离线层**对外承诺**的那些名字。碰 `_` 开头的
   内部名，验收就绑死在实现上了。
2. **每条检查都带规格出处。** 没有出处的检查是**发明规范**；出处对不上的检查是
   **发现了规格洞**。两种都要能在报告里看出来。

## 它是规划者的工具，不进任务消息

D-全局-22：`tests/spec/` 那套检查**只归规划者用**。执行者该建什么，从**规格文档**
里读；把它当"活清单"交出去，等于给它一道有答案的题——它会朝"让这些检查变绿"
过拟合，而不是朝"把这件事做对"。

## 跑法（不需要 pytest）

    python tests/spec/topbar_spec.py

pytest 外壳是 `test_topbar_spec.py`，它只是把下面那张表逐条变成用例。

## 真值从哪来

`tests/帧样本/顶部两个数/` 的 35 张原生裁剪，规划者**逐帧目视确认过**两个数。
真值表在下面 `TRUTH`。**执行者不许读本文件。**

## 35 帧是「一批顺手抓的常规帧」，不是难例集

覆盖到的是：敌数一侧 1 位与 2 位、`1` 这个窄字形在首位与末位、生命 `1` 与 `3`。
**没覆盖到**：两位数的生命、3 位数的敌数——那两条的窗口是按度量推的，见
《视觉层/顶部两个数》§尚未定。
"""

from __future__ import annotations

import importlib
import json
import struct
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "帧样本" / "顶部两个数"
COST_FIXTURES = REPO / "tests" / "帧样本" / "费用"
GLYPH_DIR = REPO / "assets" / "features" / "glyphs"

CANVAS_W, CANVAS_H = 2800, 1260


# ---------------------------------------------------------------------------
# 真值：规划者逐帧目视确认（35 帧）
# 帧名 → (已击杀数, 敌人总数, 蓝门血量)
# ---------------------------------------------------------------------------

TRUTH: Mapping[str, tuple[int, int, int]] = {
    "shot01": (0, 8, 1), "shot02": (1, 8, 1), "shot03": (1, 8, 1), "shot04": (1, 8, 1),
    "shot05": (3, 8, 1), "shot06": (3, 8, 1), "shot07": (3, 8, 1), "shot08": (3, 8, 1),
    "shot09": (3, 8, 1),
    "shot16": (0, 22, 1), "shot17": (0, 22, 1), "shot18": (0, 22, 1), "shot19": (0, 22, 1),
    "shot20": (0, 22, 1), "shot21": (0, 22, 1), "shot22": (0, 22, 1), "shot23": (0, 22, 1),
    "shot24": (2, 22, 1), "shot25": (3, 22, 1),
    "shot26": (14, 22, 1), "shot27": (14, 22, 1), "shot28": (14, 22, 1), "shot29": (14, 22, 1),
    "shot30": (15, 22, 1), "shot31": (17, 22, 1), "shot32": (17, 22, 1),
    "shot33": (19, 22, 1), "shot34": (20, 22, 1), "shot35": (20, 22, 1),
    "shot36": (21, 22, 1), "shot37": (22, 22, 1),
    "m2_view": (40, 41, 3), "m2x_pview": (40, 41, 3), "raw_a": (21, 41, 3), "raw_b": (25, 41, 3),
}

# ---------------------------------------------------------------------------
# 实测几何（原生 2800×1260，见《视觉层/顶部两个数》§每个区域必须知道的事实）
# ---------------------------------------------------------------------------

ENEMY_UNION = (1236, 38, 1348, 67)
"""敌数字形的紧外框并集：x0 y0 x1 y1，含两端。"""
LIFE_UNION = (1599, 38, 1618, 67)
"""生命字形的紧外框并集（`1` 11×28 · `3` 20×30 的并集）。"""

ENEMY_ANCHOR = 1292.5
"""敌数那串东西的水平锚点（实测：`0/8`/`3/8`/`14/22`/`22/22` 都是这个中心）。"""
LIFE_ANCHOR = 1608.5
"""生命数字的水平锚点（`1` 中心 1609.0 · `3` 中心 1608.5）。"""

# 文档那张表里属于本文件的两行 —— 值必须与 `cost_spec.DOC_RECTS` 逐字相同
DOC_WINDOWS: Mapping[str, tuple[int, int, int, int]] = {
    "enemy_count": (1200, 33, 1385, 72),
    "life_digits": (1574, 33, 1643, 72),
}

PINNED: Mapping[str, int] = {
    "MAX_ENEMY_DIGITS": 3,
    "MAX_LIFE_DIGITS": 2,
    "LIFE_INK_THRESHOLD": 150,
}
"""文档钉住的那三个数（位数上限两个 + 生命那条墨线）。三处缝的线**不钉**。"""

# 一处实测的墨迹坐标，用来造"挡住某一位"的合成帧（原生绝对像素，含两端）
SEP_SHOT01 = (1287, 38, 1303, 67)
"""`shot01` 那个分隔符的紧外框：抹掉它就等于"分隔符认不出来"。"""
DIGIT4_SHOT26 = (1257, 38, 1279, 67)
"""`shot26`（`14/22`）里那个 `4`：换成实心块就等于"这一位对不上任何字形"。"""


# ---------------------------------------------------------------------------
# 骨架
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    ident: str
    spec: str
    claim: str
    probe: Callable[[], str]


_LOADED: dict[str, object] = {}


def _mod(name: str):
    """按需 import；缺模块时报成"这条链还不存在"，而不是炸得看不清。"""
    if name not in _LOADED:
        try:
            _LOADED[name] = importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001 —— 缺模块就是这批检查的红法
            raise AssertionError(f"`{name}` 还不存在：{type(exc).__name__}: {exc}") from None
    return _LOADED[name]


def _index() -> dict:
    return json.loads((FIXTURES / "index.json").read_text(encoding="utf-8"))


def _entries() -> list[dict]:
    return list(_index()["crops"])


def _entry(name: str) -> dict:
    for e in _entries():
        if e["file"].rsplit(".", 1)[0] == name:
            return e
    raise AssertionError(f"夹具里没有 {name}")


def _canvas(name: str) -> np.ndarray:
    """把这一帧的裁剪贴回一张 2800×1260 的黑底画布。

    这样 ROI 的取法仍然是真的在被测 —— 没有把 ROI 预先替读数链做好。
    """
    e = _entry(name)
    x, y = e["origin"]
    w, h = e["size"]
    crop = np.asarray(Image.open(FIXTURES / e["file"]).convert("RGB"))
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    canvas[y : y + h, x : x + w] = crop
    return canvas


def _raw_from_canvas(canvas: np.ndarray) -> bytes:
    rgba = np.dstack([canvas, np.full((CANVAS_H, CANVAS_W), 255, np.uint8)])
    return struct.pack("<4I", CANVAS_W, CANVAS_H, 1, 1) + rgba.tobytes()


def _patch(x0: int, y0: int, x1: int, y1: int, block: np.ndarray) -> tuple:
    return ((x0, y0, x1, y1), block)


def _blank(x0: int, y0: int, x1: int, y1: int) -> tuple:
    """把这块矩形涂成黑（"那里没有东西"）。"""
    return _patch(x0, y0, x1, y1, np.zeros((y1 - y0 + 1, x1 - x0 + 1, 3), np.uint8))


def _window(name: str) -> tuple[int, int, int, int]:
    """**文档**那个窗口（含两端）。合成帧一律按文档的坐标造。

    不按 `ui.rects` 造：那是被测实现自己的值，拿它去定位合成帧，就等于
    "实现把窗口挪到哪、我就跟着挪到哪"——那样 T1 之外谁也抓不住窗口挪错。
    """
    return DOC_WINDOWS[name]


GLYPH_BOXES: Mapping[str, tuple[str, int, int, int, int]] = {
    "1": ("shot26", 1240, 34, 1254, 72),
    "3": ("shot05", 1259, 34, 1282, 72),
}
"""能从既有帧里整块搬的真字形（框比紧外框宽一圈，对齐时按墨收回来）。"""


def _paste_glyph(ch: str, left: int, top: int) -> tuple:
    """把一个真字形**整块**搬到 `(left, top)`（对齐的是它的**墨**，不是整块）。

    **整块原样搬，不按阈值裁墨**：裁墨会把抗锯齿那一圈切掉、把字搬瘦 ——
    实测那样搬过来的 `0` 对字形表只得到 0.914（原帧上同一个字形 0.947），
    离 `SOLID_MATCH_SCORE = 0.90` 只剩 0.014。摆位置时按墨的紧外框对齐，
    所以多带一圈背景不影响它落在哪。
    """
    frame_name, x0, y0, x1, y1 = GLYPH_BOXES[ch]
    block = _canvas(frame_name)[y0 : y1 + 1, x0 : x1 + 1].copy()
    ys, xs = np.nonzero(block.min(axis=2) >= 128)
    assert len(xs) > 0, f"字形 {ch!r} 的框 {GLYPH_BOXES[ch]} 里没有墨"
    px, py = left - int(xs.min()), top - int(ys.min())
    return _patch(px, py, px + block.shape[1] - 1, py + block.shape[0] - 1, block)


def _raw(name: str, *patches: tuple) -> bytes:
    canvas = _canvas(name)
    for (x0, y0, x1, y1), block in patches:
        canvas[y0 : y1 + 1, x0 : x1 + 1] = block
    return _raw_from_canvas(canvas)


def _frame(raw: bytes):
    return _mod("ark.engine.vision.frame").frame_from_raw(raw)


def _patched(name: str, *patches: tuple):
    return _frame(_raw(name, *patches))


def _ui():
    return _mod("ark.engine.vision.layout").UI


def _glyphs():
    return _mod("ark.engine.vision.glyphs").GlyphTable.load(GLYPH_DIR)


def _topbar():
    return _mod("ark.engine.vision.topbar")


def _slot_value(value) -> str:
    """把一个槽值渲染成短字符串，只给证据用。"""
    contracts = _mod("ark.contracts")
    if isinstance(value, contracts.Unsampled):
        return "未采样"
    if isinstance(value, contracts.SampledEmpty):
        return "已采样但为空"
    if not isinstance(value, contracts.Valued):
        return f"不是槽值：{type(value).__name__}"
    payload = value.payload
    if isinstance(payload, contracts.Exact):
        return f"精确({payload.value})"
    if isinstance(payload, contracts.Interval):
        return f"区间[{payload.lo},{payload.hi}]"
    if isinstance(payload, contracts.Unknown):
        return "未知"
    return f"载荷 {type(payload).__name__}"


def _exact_of(value):
    """槽值是 `Valued(Exact(n))` 就返回 n，否则 None。"""
    contracts = _mod("ark.contracts")
    if isinstance(value, contracts.Valued) and isinstance(value.payload, contracts.Exact):
        return value.payload.value
    return None


def _is_unsampled(value) -> bool:
    return isinstance(value, _mod("ark.contracts").Unsampled)


# ---------------------------------------------------------------------------
# 走一遍基准集：一趟走完，三条检查共用（35 帧 × 14 MB，不许走三趟）
# ---------------------------------------------------------------------------

_WALK: dict | None = None


def _walk() -> dict:
    global _WALK
    if _WALK is not None:
        return _WALK

    batch = _mod("ark.offline.bench.batch").load_batch(FIXTURES)
    feed = _mod("ark.offline.bench.feed")
    frame_mod = _mod("ark.engine.vision.frame")
    topbar = _topbar()
    ui = _ui()
    table = _glyphs()

    order: list[str] = []
    paste_bad: list[str] = []
    life: dict[str, object] = {}
    enemy: dict[str, object] = {}
    for case, raw in feed.replay_frames(batch):
        order.append(case.name)
        frame = frame_mod.frame_from_raw(raw)
        # 贴回逐像素对不对：拿原帧的两个 ROI 直接比
        e = _entry(case.name)
        ox, oy = e["origin"]
        orig = np.asarray(Image.open(FIXTURES / e["file"]).convert("RGB"))
        for (x0, y0, x1, y1) in (ENEMY_UNION, LIFE_UNION):
            got = frame.pixels[y0 : y1 + 1, x0 : x1 + 1]
            want = orig[y0 - oy : y1 + 1 - oy, x0 - ox : x1 + 1 - ox]
            if not np.array_equal(got, want):
                paste_bad.append(f"{case.name}@{x0}")
        life[case.name] = topbar.read_life_digits(frame, table, ui=ui)
        enemy[case.name] = topbar.read_enemy_count(frame, table, ui=ui)

    _WALK = {
        "order": order,
        "paste_bad": paste_bad,
        "life": life,
        "enemy": enemy,
        "manifest_order": [e["file"].rsplit(".", 1)[0] for e in _entries()],
    }
    return _WALK


# ---------------------------------------------------------------------------
# 检查
# ---------------------------------------------------------------------------


def check_windows_are_the_documented_ones() -> str:
    """两个窗口的值必须与文档那张表逐字相同。

    出处：《视觉层/顶部两个数》§两个窗口。文档是**唯一**写值的地方，代码与
    规格各抄一份，抄漏一处就是静默漂移。
    """
    import cost_spec

    ui = _ui()
    bad = []
    for name, want in DOC_WINDOWS.items():
        if name not in ui.rects:
            bad.append(f"UI 常量里没有 {name}")
            continue
        r = ui.rects[name]
        got = (r.x, r.y, r.x + r.width - 1, r.y + r.height - 1)
        if got != want:
            bad.append(f"{name} 是 {got}，文档写的是 {want}")
        if cost_spec.DOC_RECTS.get(name) != want:
            bad.append(f"{name} 在 cost_spec.DOC_RECTS 里是 {cost_spec.DOC_RECTS.get(name)}")
    assert not bad, "；".join(bad)
    return f"{len(DOC_WINDOWS)} 个窗口与文档一致：{dict(DOC_WINDOWS)}"


def check_windows_hold_the_measured_glyphs_and_the_anchor() -> str:
    """窗口要装得下实测的字形并集，而且**窗口中心就是实测锚点**。

    出处：《视觉层/顶部两个数》§每个区域必须知道的事实（"两个数都是水平居中的"
    ——只读到 1 个字形时要拿窗口中心当判据，所以中心偏了判据就偏了）。
    """
    ui = _ui()
    bad = []
    for name, union, anchor in (
        ("enemy_count", ENEMY_UNION, ENEMY_ANCHOR),
        ("life_digits", LIFE_UNION, LIFE_ANCHOR),
    ):
        r = ui.rects[name]
        x0, y0, x1, y1 = union
        left, right, top, bottom = r.x, r.x + r.width - 1, r.y, r.y + r.height - 1
        if not (left <= x0 and x1 <= right):
            bad.append(f"{name} 的 x [{left},{right}] 装不下字形 [{x0},{x1}]")
        if not (top <= y0 and y1 <= bottom):
            bad.append(f"{name} 的 y [{top},{bottom}] 装不下字形 [{y0},{y1}]")
        center = r.x + r.width / 2.0
        if abs(center - anchor) > 1.0:
            bad.append(f"{name} 的窗口中心 {center} 离实测锚点 {anchor} 超过 1 px")
        if not (left >= 0 and top >= 0 and right <= CANVAS_W - 1 and bottom <= CANVAS_H - 1):
            bad.append(f"{name} 越出画布：{left},{top}..{right},{bottom}")
    assert not bad, "；".join(bad)
    return "两个窗口都装得下字形并集，中心与实测锚点相差不超过 1 px"


def check_bench_goes_through_the_replay_adapter() -> str:
    """35 帧经重放适配器逐帧吐出来，贴回画布后两个 ROI **逐像素**等于原帧。

    出处：《基准集》§3（裁剪要贴回画布，不许直接把裁剪当帧）+ §4（按批冻一份）。
    """
    walk = _walk()
    assert not walk["paste_bad"], f"贴回之后对不上的：{walk['paste_bad'][:6]}"
    assert walk["order"] == walk["manifest_order"], (
        f"吐出来的次序与清单不一致：{walk['order']} vs {walk['manifest_order']}"
    )
    assert len(walk["order"]) == len(TRUTH), (
        f"这一批该有 {len(TRUTH)} 帧，走了 {len(walk['order'])} 帧"
    )
    return f"{len(walk['order'])} 帧依次吐出、贴回逐像素一致、次序等于清单次序"


def check_life_reads_on_every_frame() -> str:
    """35 帧的生命都要读成**精确值**，而且值对。

    出处：《视觉层/顶部两个数》§每个区域必须知道的事实。这 35 个字形**全是粉的**
    `(255,120,120)` —— 共用白字那套墨迹口径会一个都切不出来，所以这条同时也是
    「取墨口径要跟着读数链走」那条的落地检查。
    """
    walk = _walk()
    bad = []
    for name, (_, _, want) in TRUTH.items():
        got = _exact_of(walk["life"].get(name))
        if got != want:
            bad.append(f"{name} 该是 {want}，读成 {_slot_value(walk['life'].get(name))}")
    assert not bad, "；".join(bad[:8])
    return f"{len(TRUTH)} 帧生命全部精确：{sorted({v[2] for v in TRUTH.values()})}"


def check_enemy_reads_on_every_frame() -> str:
    """35 帧的敌数都要读成**两个精确值**，而且值对。

    出处：《视觉层/顶部两个数》§每个区域必须知道的事实 + 《状态层/全局与HUD》
    §已击杀数 / §敌人总数（同一次读数里一起给）。这条同时守着"分隔符不是数字"：
    把它当一位数字读进去，`0/8` 会变成一个三位数。
    """
    walk = _walk()
    bad = []
    for name, (want_killed, want_total, _) in TRUTH.items():
        pair = walk["enemy"].get(name)
        if not isinstance(pair, tuple) or len(pair) != 2:
            bad.append(f"{name} 的敌数读数不是一对：{type(pair).__name__}")
            continue
        got_killed, got_total = _exact_of(pair[0]), _exact_of(pair[1])
        if (got_killed, got_total) != (want_killed, want_total):
            bad.append(
                f"{name} 该是 {want_killed}/{want_total}，读成 "
                f"{_slot_value(pair[0])}/{_slot_value(pair[1])}"
            )
    assert not bad, "；".join(bad[:8])
    return f"{len(TRUTH)} 帧敌数全部精确（{len({v[:2] for v in TRUTH.values()})} 种组合）"


def check_blank_life_window_is_unsampled() -> str:
    """生命窗口整块涂黑 → 未采样。

    出处：《视觉层/顶部两个数》§降级第一条（"整块没笔画"）。它与"认不出"是两件事：
    涂黑是"那里没有东西"，不是"有东西我不知道是什么"。
    """
    ui = _ui()
    frame = _patched("shot26", _blank(*_window("life_digits")))
    got = _topbar().read_life_digits(frame, _glyphs(), ui=ui)
    assert _is_unsampled(got), f"涂黑该报未采样，读成 {_slot_value(got)}"
    return "生命窗口整块涂黑 → 未采样"


def check_unreadable_digit_with_known_count_is_unknown() -> str:
    """**位数已经数对、只是某一位认不出** → **未知**（不是区间、也不是挑一个）。

    出处：《识别方法》§已定 +《视觉层/顶部两个数》§降级（T7 裁定：这三个字段的
    允许形态里**没有区间**，读数链不许替字段加一个）。生命窗口里放**两块**：左边
    是真实的 `1`，右边是一块实心块（实测对最像的数字只得 0.54）。位数是 2，但那
    一位认不出 —— 我们知道那里有一位数字，不知道它是几。
    """
    ui = _ui()
    solid = np.full((LIFE_UNION[3] - LIFE_UNION[1] + 1, 19, 3), 255, np.uint8)
    frame = _patched(
        "shot26",
        _blank(*_window("life_digits")),
        _paste_glyph("1", 1589, LIFE_UNION[1]),
        _patch(1604, LIFE_UNION[1], 1622, LIFE_UNION[3], solid),
    )
    got = _topbar().read_life_digits(frame, _glyphs(), ui=ui)
    contracts = _mod("ark.contracts")
    assert isinstance(got, contracts.Valued), f"该是有值，读成 {_slot_value(got)}"
    assert isinstance(got.payload, contracts.Unknown), (
        f"位数已定、某一位认不出该落未知，读成 {_slot_value(got)}"
    )
    return "两块、其中一块认不出 → 未知"


def check_single_unreadable_block_is_unknown() -> str:
    """**只有一块、而这一块认不出** → **未知**。

    出处：《视觉层/顶部两个数》§降级。只有一块且认不出时，**位数本身就没定下来**
    （可能是一位认不出的数字，也可能是两位被桥接成一块）。用这个 HUD 上真实的
    那个分隔符当"认不出的墨"（实测对最像的数字只得 0.63–0.78）。
    """
    ui = _ui()
    src = _canvas("shot01")
    block = src[SEP_SHOT01[1] : SEP_SHOT01[3] + 1, SEP_SHOT01[0] : SEP_SHOT01[2] + 1].copy()
    # 只搬墨、不搬它周围的背景：搬背景会把敌数那边的战场一起带过来。
    keep = block.max(axis=2) >= 150
    ink = np.zeros_like(block)
    ink[keep] = block[keep]
    x0 = int(LIFE_ANCHOR - ink.shape[1] / 2)
    frame = _patched(
        "shot01",
        _blank(*_window("life_digits")),
        _patch(x0, LIFE_UNION[1], x0 + ink.shape[1] - 1, LIFE_UNION[1] + ink.shape[0] - 1, ink),
    )
    got = _topbar().read_life_digits(frame, _glyphs(), ui=ui)
    contracts = _mod("ark.contracts")
    assert isinstance(got, contracts.Valued), f"该是有值，读成 {_slot_value(got)}"
    assert isinstance(got.payload, contracts.Unknown), (
        f"只有一块且认不出该落未知（位数没定下来），读成 {_slot_value(got)}"
    )
    return "只有一块且认不出 → 未知"


def check_missing_life_digit_does_not_shrink_the_number() -> str:
    """只读到 1 个字形、而它**不在窗口中心附近** → 缺了一位，报区间。

    出处：《视觉层/顶部两个数》§降级第三条。把真实的 `1` 往左挪 20 px，于是它
    偏左 → 缺的在右边 → `1?` → `[10,19]`。**不许塌成确切值 `1`** —— 那正是这条
    链最危险的一种假事实（与费用那条同型）。
    """
    ui = _ui()
    wx0, wy0, wx1, wy1 = _window("life_digits")
    canvas = _canvas("shot26")
    band = canvas[LIFE_UNION[1] : LIFE_UNION[3] + 1, wx0 : wx1 + 1].copy()
    shifted = np.zeros_like(band)
    shifted[:, : band.shape[1] - 20] = band[:, 20:]
    canvas[wy0 : wy1 + 1, wx0 : wx1 + 1] = 0
    canvas[LIFE_UNION[1] : LIFE_UNION[3] + 1, wx0 : wx1 + 1] = shifted
    got = _topbar().read_life_digits(_frame(_raw_from_canvas(canvas)), _glyphs(), ui=ui)
    contracts = _mod("ark.contracts")
    assert isinstance(got, contracts.Valued), f"该是有值，读成 {_slot_value(got)}"
    assert isinstance(got.payload, contracts.Unknown), (
        f"缺一位该落未知（这三个字段不收区间），**但绝不许报成一个更短的确切数**，"
        f"读成 {_slot_value(got)}"
    )
    return "只剩一块且偏左 → 未知，没有塌成 `1`"


def check_missing_separator_makes_both_numbers_unknown() -> str:
    """敌数：分隔符认不出来 → **两个数都报未知**。

    出处：《视觉层/顶部两个数》§每个区域必须知道的事实。抹掉分隔符那几列之后，
    剩下的两个数字块**无法分辨谁是谁** —— 报 `0/8` 只是碰巧对，而"碰巧对"不是读数。
    """
    ui = _ui()
    frame = _patched("shot01", _blank(*SEP_SHOT01))
    pair = _topbar().read_enemy_count(frame, _glyphs(), ui=ui)
    contracts = _mod("ark.contracts")
    assert isinstance(pair, tuple) and len(pair) == 2, f"敌数该给一对，收到 {type(pair).__name__}"
    bad = [
        f"{side} 读成 {_slot_value(value)}"
        for side, value in zip(("已击杀", "总数"), pair)
        if isinstance(value, contracts.Valued) and isinstance(value.payload, contracts.Exact)
    ]
    assert not bad, "分隔符没了还报确切值：" + "；".join(bad)
    return f"分隔符被抹掉 → {_slot_value(pair[0])} / {_slot_value(pair[1])}"


def check_one_blocked_digit_only_degrades_its_own_side() -> str:
    """敌数：一边的某一位被挡住 → **只让那一边降级**，另一边照常精确。

    出处：《视觉层/顶部两个数》§每个区域必须知道的事实（"两个数各自带自己的三态"）。
    `shot26` 是 `14/22`：把那个 `4` 换成实心块（实测对最像的数字只得 0.54，既不是
    数字、也不是分隔符），于是已击杀该落成区间、总数该仍是 22。
    """
    ui = _ui()
    block = np.full((DIGIT4_SHOT26[3] - DIGIT4_SHOT26[1] + 1, DIGIT4_SHOT26[2] - DIGIT4_SHOT26[0] + 1, 3), 255, np.uint8)
    frame = _patched("shot26", _patch(*DIGIT4_SHOT26, block))
    pair = _topbar().read_enemy_count(frame, _glyphs(), ui=ui)
    contracts = _mod("ark.contracts")
    assert isinstance(pair, tuple) and len(pair) == 2, f"敌数该给一对，收到 {type(pair).__name__}"
    killed, total = pair
    assert _exact_of(total) == 22, f"总数那一侧没被动过，该仍是 22，读成 {_slot_value(total)}"
    assert not (isinstance(killed, contracts.Valued) and isinstance(killed.payload, contracts.Exact)), (
        f"被挡住的那一位不许报确切值，读成 {_slot_value(killed)}"
    )
    return f"挡住一位 → 已击杀 {_slot_value(killed)}，总数 {_slot_value(total)}"


def check_the_two_readers_do_not_look_at_each_other() -> str:
    """两条读数各看各的窗口：抹掉一边，另一边照读。

    出处：《视觉层/顶部两个数》§两个窗口（两个常量各管一块）。这条挡住"从整屏
    找数字"那种写法 —— 它会读到窗口外面的东西，而窗口正是这一层唯一承诺的事。
    """
    ui = _ui()
    blank_life = _patched("shot26", _blank(*_window("life_digits")))
    blank_enemy = _patched("shot26", _blank(*_window("enemy_count")))
    topbar = _topbar()
    table = _glyphs()
    pair = topbar.read_enemy_count(blank_life, table, ui=ui)
    got_enemy = (_exact_of(pair[0]), _exact_of(pair[1]))
    got_life = _exact_of(topbar.read_life_digits(blank_enemy, table, ui=ui))
    assert got_enemy == (14, 22), f"抹掉生命窗口之后敌数该还是 14/22，读成 {got_enemy}"
    assert got_life == 1, f"抹掉敌数窗口之后生命该还是 1，读成 {got_life}"
    return "抹掉任一边，另一边的读数一字不变"


def check_pinned_constants_match_the_doc() -> str:
    """规格钉住的三个数，代码里必须相等。

    出处：《视觉层/顶部两个数》§每个区域必须知道的事实（"这几个数住在代码里，
    规格这边把它们钉住"）。**同一个数不许只在文档写一遍、代码写一遍而没人比对**
    —— 改一处忘一处就是静默漂移，`check_docs.py` 抓不到（它只看状态行 / 链接 /
    废词）。**三处缝那几条线不在表里**：它们是实现挑的，规格钉的是观测到的空档。
    """
    topbar = _topbar()
    bad = []
    for name, want in PINNED.items():
        got = getattr(topbar, name, None)
        if got != want:
            bad.append(f"{name} 该是 {want}，收到 {got!r}")
    assert not bad, "；".join(bad)
    return f"钉住的 {len(PINNED)} 个数与文档一致：{dict(PINNED)}"


def check_cost_chain_did_not_move() -> str:
    """改墨迹口径**不许把费用那条链挪走**。

    出处：《视觉层/顶部两个数》§每个区域必须知道的事实 + D-全局-24。生命是粉的、
    敌数是白的，两套取墨口径必须各走各的；把共用的那个改掉，代价会落在费用上。
    真值不在这里重抄，取自 `cost_spec.DIGITS`。
    """
    import cost_spec

    frame_mod = _mod("ark.engine.vision.frame")
    cost = _mod("ark.engine.vision.cost")
    ui = _ui()
    table = _glyphs()
    index = json.loads((COST_FIXTURES / "index.json").read_text(encoding="utf-8"))
    by_shot = {e["shot"]: e for e in index["crops"]}
    bad = []
    for shot in (1, 26, 35):
        e = by_shot[shot]
        x, y = e["origin"]
        w, h = e["size"]
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
        canvas[y : y + h, x : x + w] = np.asarray(Image.open(COST_FIXTURES / e["file"]).convert("RGB"))
        frame = frame_mod.frame_from_raw(_raw_from_canvas(canvas))
        got = _exact_of(cost.read_cost_digits(frame, table, ui=ui))
        if got != cost_spec.DIGITS[shot]:
            bad.append(f"shot{shot:02d} 该是 {cost_spec.DIGITS[shot]}，读成 {got}")
    assert not bad, "；".join(bad)
    return "费用在 shot01/26/35 上仍是 2 / 65 / 99"


CHECKS: list[Check] = [
    Check("T1", "《视觉层/顶部两个数》§两个窗口", "两个窗口与文档逐字一致", check_windows_are_the_documented_ones),
    Check("T2", "《视觉层/顶部两个数》§每个区域必须知道的事实", "窗口装得下字形，中心就是实测锚点", check_windows_hold_the_measured_glyphs_and_the_anchor),
    Check("T3", "《采集与验收/基准集》§3 §4", "35 帧经重放适配器贴回，逐像素一致、次序不变", check_bench_goes_through_the_replay_adapter),
    Check("T4", "《视觉层/顶部两个数》§每个区域必须知道的事实", "35 帧生命全部读成精确值", check_life_reads_on_every_frame),
    Check("T5", "《视觉层/顶部两个数》§每个区域必须知道的事实", "35 帧敌数全部读成两个精确值", check_enemy_reads_on_every_frame),
    Check("T6", "《视觉层/顶部两个数》§降级", "生命整块涂黑 → 未采样", check_blank_life_window_is_unsampled),
    Check("T7", "《识别方法》§已定 · 《顶部两个数》§降级", "位数已定、某一位认不出 → 未知", check_unreadable_digit_with_known_count_is_unknown),
    Check("T8", "《视觉层/顶部两个数》§降级", "缺一位 → 未知，不塌成更短的确切数", check_missing_life_digit_does_not_shrink_the_number),
    Check("T9", "《视觉层/顶部两个数》§每个区域必须知道的事实", "分隔符认不出 → 两个数都未知", check_missing_separator_makes_both_numbers_unknown),
    Check("T10", "《视觉层/顶部两个数》§每个区域必须知道的事实", "挡住一位只让那一边降级", check_one_blocked_digit_only_degrades_its_own_side),
    Check("T11", "《视觉层/顶部两个数》§两个窗口", "两条读数各看各的窗口", check_the_two_readers_do_not_look_at_each_other),
    Check("T12", "《视觉层/顶部两个数》§每个区域必须知道的事实 · D-全局-24", "费用那条链没被挪走", check_cost_chain_did_not_move),
    Check("T13", "《视觉层/顶部两个数》§降级", "只有一块且认不出 → 未知", check_single_unreadable_block_is_unknown),
    Check("T14", "《视觉层/顶部两个数》§每个区域必须知道的事实", "钉住的三个数与文档相等", check_pinned_constants_match_the_doc),
]


def main() -> int:
    failures = 0
    for check in CHECKS:
        try:
            evidence = check.probe()
        except AssertionError as bad:
            failures += 1
            print(f"红 {check.ident}  {check.claim}\n     出处 {check.spec}\n     证据 {bad}")
        except BaseException as broken:  # noqa: BLE001
            failures += 1
            print(f"炸 {check.ident}  {check.claim}\n     {type(broken).__name__}: {broken}")
        else:
            print(f"绿 {check.ident}  {evidence}")
    print()
    print(f"{len(CHECKS) - failures}/{len(CHECKS)} 条规格检查通过")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
