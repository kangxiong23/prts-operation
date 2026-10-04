"""费用读数链的规格符合性验收 —— **由规划者维护，不住在 tests/unit/**。

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

## 这批检查现在是红的

T3 还没做。**红的这些就是 T3 的活清单**——不是"实现有问题"，是"规格要求的东西还不存在"。

## 跑法（不需要 pytest）

    python tests/spec/cost_spec.py

pytest 外壳是 `test_cost_spec.py`，它只是把下面那张表逐条变成用例。

## 真值从哪来

`tests/帧样本/费用/` 的 31 张原生裁剪，规划者**逐帧目视确认过**读数与条填充。
真值表在下面 `DIGITS` / `FILLS` / `FULL`。**执行者不许读本文件**——`取材.json`
里那 11 帧才是给它的。
"""

from __future__ import annotations

import contextlib
import importlib
import itertools
import json
import os
import shutil
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
FIXTURES = REPO / "tests" / "帧样本" / "费用"
GLYPH_DIR = REPO / "assets" / "features" / "glyphs"

CANVAS_W, CANVAS_H = 2800, 1260

# ---------------------------------------------------------------------------
# 真值：规划者目视确认（27 帧战斗界面 + 4 帧 99/MAX，共 31 帧）
# ---------------------------------------------------------------------------

DIGITS: Mapping[int, int] = {
    1: 2, 2: 8, 3: 14, 4: 6, 5: 11, 6: 2, 7: 8, 8: 13, 9: 18,
    16: 10, 17: 2, 18: 7, 19: 11, 20: 16, 21: 2, 22: 7, 23: 11,
    24: 15, 25: 6, 26: 65, 27: 69, 28: 73, 29: 77, 30: 81,
    31: 85, 32: 89, 33: 93, 34: 97, 35: 99, 36: 99, 37: 99,
}
# 轨道 x 2589..2741，全长 153 px；下面是从左端起连续亮白的列数
FILLS: Mapping[int, int] = {
    1: 29, 2: 153, 3: 95, 4: 29, 5: 58, 6: 116, 7: 15, 8: 7, 9: 29,
    16: 153, 17: 66, 18: 0, 19: 153, 20: 153, 21: 44, 22: 29, 23: 146,
    24: 153, 25: 153, 26: 80, 27: 87, 28: 73, 29: 73, 30: 66,
    31: 87, 32: 0, 33: 22, 34: 66, 35: 0, 36: 0, 37: 0,
}
# 轨道整条中间灰 + 白 MAX = 满
FULL = frozenset({35, 36, 37})

TRACK_PX = 153
TRACK_X0, TRACK_X1 = 2589, 2741
DIGIT_UNION_X = (2676, 2766)

# 文档《视觉层/顶部HUD与数字》§区域与常量 里那张表，逐字抄下来（含两端）
DOC_RECTS: Mapping[str, tuple[int, int, int, int]] = {
    "cost_panel": (2585, 845, 2799, 972),
    "cost_icon": (2595, 873, 2658, 936),
    "cost_digits": (2666, 868, 2776, 936),
    "cost_bar": (2589, 945, 2741, 954),
    "deploy_limit_row": (2462, 968, 2755, 998),
    "enemy_count": (1200, 29, 1372, 76),
    "life_digits": (1585, 29, 1623, 76),
    "pause_button": (2594, 54, 2710, 130),
}


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
            raise AssertionError(
                f"`{name}` 还不存在：{type(exc).__name__}: {exc}"
            ) from None
    return _LOADED[name]


_TMP_SEQ = itertools.count()


@contextlib.contextmanager
def _writable_tmp():
    """建一个**能进得去**的临时目录。

    本机沙箱下 `tempfile.mkdtemp()` 建出来的目录**连建它的进程自己都进不去**：
    它用 `os.mkdir(path, 0o700)`，而受限令牌下那份 DACL 不继承父目录。
    这是**本机环境的事，不是被测实现的事** —— 所以在这里绕开，而不是让实现去迁就。
    实测：默认权限的 mkdir 能进，`0o700` 的进不去。
    """
    base = REPO / "data" / "_spec_tmp"
    base.mkdir(parents=True, exist_ok=True)
    dest = base / f"run{os.getpid()}_{next(_TMP_SEQ)}"
    dest.mkdir()
    try:
        yield dest
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def _index() -> dict:
    return json.loads((FIXTURES / "index.json").read_text(encoding="utf-8"))


def _entry(shot: int) -> dict:
    for e in _index()["crops"]:
        if e["shot"] == shot:
            return e
    raise AssertionError(f"夹具里没有 shot{shot:02d}")


def _crop(shot: int) -> np.ndarray:
    return np.asarray(Image.open(FIXTURES / _entry(shot)["file"]).convert("RGB"))


def _raw(shot: int, patch: tuple[tuple[int, int, int, int], np.ndarray] | None = None) -> bytes:
    """把夹具裁剪贴回 2800×1260 的黑底，编成设备层 §4 定义的原始 RGBA 字节。

    这样 ROI 的取法仍然是真的在被测 —— 没有把 ROI 预先替读数链做好。
    """
    e = _entry(shot)
    x, y = e["origin"]
    w, h = e["size"]
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    canvas[y : y + h, x : x + w] = _crop(shot)
    if patch is not None:
        (px, py, pw, ph), block = patch
        canvas[py : py + ph, px : px + pw] = block
    rgba = np.dstack([canvas, np.full((CANVAS_H, CANVAS_W), 255, np.uint8)])
    return struct.pack("<4I", CANVAS_W, CANVAS_H, 1, 1) + rgba.tobytes()


def _frame(shot: int, patch=None):
    frame = _mod("ark.engine.vision.frame")
    return frame.frame_from_raw(_raw(shot, patch))


def _ui():
    return _mod("ark.engine.vision.layout").UI


def _glyphs():
    glyphs = _mod("ark.engine.vision.glyphs")
    return glyphs.GlyphTable.load(GLYPH_DIR)


def _harvest() -> dict[int, int]:
    data = json.loads((FIXTURES / "取材.json").read_text(encoding="utf-8"))
    return {h["shot"]: h["cost"] for h in data["harvest"]}


def _digit_union(shot: int) -> tuple[int, int, int, int]:
    """该帧数字白像素的外接框（绝对像素）。窗口取得比 cost_digits 宽得多。"""
    a = np.asarray(Image.open(FIXTURES / _entry(shot)["file"]).convert("RGB")).astype(np.int16)
    ox, oy = _entry(shot)["origin"]
    sub = a[873 - oy : 932 - oy, 2662 - ox : 2801 - ox]
    ys, xs = np.nonzero(sub.min(axis=2) > 190)
    return (2662 + xs.min(), 873 + ys.min(), 2662 + xs.max(), 873 + ys.max())


# ---------------------------------------------------------------------------
# 夹具自检：真值表必须能从夹具像素重算出来
# ---------------------------------------------------------------------------


def check_truth_table_is_recomputable_from_the_fixtures() -> str:
    """这条查的不是 T3，是**本文件自己的真值表**。

    `FILLS` 与数字并集是规划者从原帧量出来的；这里用**夹具那条路径**（裁剪 + 原点）
    重算一遍——原点写错、夹具裁错窗口，都会在这里露出来，而不是在 T3 的读数上。
    """
    bad = []
    for shot, want in sorted(FILLS.items()):
        e = _entry(shot)
        ox, oy = e["origin"]
        crop = _crop(shot)
        row = crop[950 - oy, TRACK_X0 - ox : TRACK_X1 - ox + 1].astype(np.int16)
        bright = row.min(axis=1) > 190
        xs = np.nonzero(bright)[0]
        got = int(xs.max()) + 1 if len(xs) and xs[0] == 0 else 0
        if got != want:
            bad.append(f"shot{shot:02d}: 重算 {got} px，真值表写 {want} px")
    for shot in sorted(FULL):
        e = _entry(shot)
        ox, oy = e["origin"]
        crop = _crop(shot)
        row = crop[950 - oy, TRACK_X0 - ox : TRACK_X1 - ox + 1].astype(np.int16)
        bright = row.min(axis=1) > 190
        xs = np.nonzero(bright)[0]
        # 满态**有**亮像素（白 MAX 就写在轨道中间），但没有**自左端起**的亮白填充
        # —— 这正是它和「相位 0」在亮度阈值下分不开的原因
        if len(xs) and xs[0] == 0:
            bad.append(f"shot{shot:02d}: 标成「满」，但左端起有亮白填充 {len(xs)} 列")
        if not (row.min() > 110).all():
            bad.append(f"shot{shot:02d}: 标成「满」，但轨道区不是整条中间灰")
    for shot in sorted(DIGITS):
        x0, _, x1, _ = _digit_union(shot)
        if not (DIGIT_UNION_X[0] <= x0 and x1 <= DIGIT_UNION_X[1]):
            bad.append(f"shot{shot:02d}: 数字框 {x0}..{x1} 超出并集 {DIGIT_UNION_X}")
    assert not bad, "；".join(bad[:8])
    return (
        f"{len(FILLS)} 帧填充重算一致 · {len(FULL)} 帧满态无亮白填充 · "
        f"{len(DIGITS)} 帧数字框都落在并集 {DIGIT_UNION_X} 里"
    )


# ---------------------------------------------------------------------------
# 布局常量
# ---------------------------------------------------------------------------


def check_ui_layout_has_the_eight_rects() -> str:
    from ark.contracts import UiLayout

    ui = _ui()
    assert isinstance(ui, UiLayout), f"UI 要是 UiLayout，收到 {type(ui).__name__}"
    got = set(ui.rects)
    want = set(DOC_RECTS)
    assert got == want, f"多出来 {sorted(got - want)}；少了 {sorted(want - got)}"
    return f"{len(want)} 个矩形常量，与文档表一致"


def check_ui_layout_values_cover_the_documented_windows() -> str:
    ui = _ui()
    bad = []
    for name, (x0, y0, x1, y1) in DOC_RECTS.items():
        r = ui.rects[name]
        right, bottom = r.x + r.width - 1, r.y + r.height - 1
        if not (x0 - 12 <= r.x <= x0 and x1 <= right <= x1 + 12):
            bad.append(f"{name} x 是 [{r.x},{right}]，文档要覆盖 [{x0},{x1}]")
        if not (y0 - 12 <= r.y <= y0 and y1 <= bottom <= y1 + 12):
            bad.append(f"{name} y 是 [{r.y},{bottom}]，文档要覆盖 [{y0},{y1}]")
    assert not bad, "；".join(bad)
    return "8 个矩形都覆盖文档窗口，外扩不超过 12 px"


def check_bar_constant_is_the_track_itself() -> str:
    """轨道是**硬边**的东西（填充的起止是准的），所以这条常量不像软边那样外扩
    —— 它**就是**轨道本体，长度直接是 153，不需要再立一个标量常量。"""
    r = _ui().rects["cost_bar"]
    right = r.x + r.width - 1
    assert r.x <= TRACK_X0 and right >= TRACK_X1, (
        f"cost_bar x [{r.x},{right}] 装不下轨道 {TRACK_X0}..{TRACK_X1}"
    )
    assert r.x >= TRACK_X0 - 4 and right <= TRACK_X1 + 4, (
        f"cost_bar x [{r.x},{right}] 比轨道 {TRACK_X0}..{TRACK_X1} 宽太多"
        "——轨道是硬边，这条常量该就是轨道本体，不是搜索窗口"
    )
    assert r.width == TRACK_PX, (
        f"cost_bar 宽该是轨道全长 {TRACK_PX} px，收到 {r.width}"
    )
    return f"cost_bar x [{r.x},{right}] 就是轨道本体，宽 {r.width} px"


def check_spec_thresholds_are_pinned_to_the_code() -> str:
    """规格写死的判据阈值，代码里的常量必须相等。

    **为什么单立一条**：这几个数是这条链**全部的决策面**，而它们原本住在两个家里
    （文档写一遍、代码写一遍），**改一处忘一处就是静默漂移**，且没有任何守卫抓得到
    —— `check_docs.py` 只看状态行 / 链接 / 废词，不比对数值。
    所以规格这边把值钉住，代码跟着比。

    **只钉规格说过的数**：`SOLID_MATCH_SCORE` / `AMBIGUITY_MARGIN` 是实现的自由
    （规格没给刻度），**不进这条** —— 钉了就等于我替实现定分。
    """
    cost = _mod("ark.engine.vision.cost")
    pinned = {
        "CENTER_TOLERANCE": 8,      # 顶部HUD与数字 §每个区域必须知道的事实：±8
        "BAR_BRIGHT_MIN": 190,      # 同上：「亮白」阈值
        "BAR_FULL_MIN": 117,        # 同上：满态轨道最暗列下界
        "BAR_DARK_MAX": 70,         # 同上：空态轨道最亮列上界（录像是 68–69，67 覆盖不住）
        "MAX_DIGITS": 2,            # 同上：位数范围 1–2 位
    }
    bad = []
    for name, want in pinned.items():
        got = getattr(cost, name, None)
        if got is None:
            bad.append(f"{name} 不在 cost.py 的公开面上")
        elif got != want:
            bad.append(f"{name} 是 {got!r}，规格钉的是 {want!r}")
    assert not bad, "；".join(bad)
    return "规格钉住的 5 个数与代码一致：" + " · ".join(f"{k}={v}" for k, v in pinned.items())


def check_windows_stay_inside_the_canvas() -> str:
    """窗口是**含两端**的绝对像素，所以右缘最大只能是画布最后一列（2799）、
    下缘最大只能是最后一行（1259）。写 2800 就是"含了一个不存在的像素"——
    夹具的 `index.json` 与 `cost_panel` 都踩过这个（规划者自己写错）。"""
    ui = _ui()
    bad = []
    for name, r in sorted(ui.rects.items()):
        right, bottom = r.x + r.width - 1, r.y + r.height - 1
        if r.x < 0 or r.y < 0:
            bad.append(f"{name} 左上角越出画布：({r.x},{r.y})")
        if right > CANVAS_W - 1:
            bad.append(f"{name} 右缘 {right} 越出画布最后一列 {CANVAS_W - 1}")
        if bottom > CANVAS_H - 1:
            bad.append(f"{name} 下缘 {bottom} 越出画布最后一行 {CANVAS_H - 1}")
    assert not bad, "；".join(bad)
    return f"{len(ui.rects)} 个窗口都落在 {CANVAS_W}×{CANVAS_H} 画布内（含两端口径）"


def check_digit_window_holds_one_and_two_digit_unions() -> str:
    r = _ui().rects["cost_digits"]
    right = r.x + r.width - 1
    x0, x1 = DIGIT_UNION_X
    assert r.x <= x0, f"cost_digits 左缘 {r.x} 切掉了数字并集左端 {x0}"
    assert right >= x1, (
        f"cost_digits 右缘 {right} 切掉了数字并集右端 {x1}"
        "——两位数的末位会整块落在窗口外"
    )
    return f"cost_digits x [{r.x},{right}] 装得下并集 {x0}..{x1}"


# ---------------------------------------------------------------------------
# 画布：原始帧 → 不缩放的像素
# ---------------------------------------------------------------------------


def check_raw_frame_decodes_to_the_capture_canvas() -> str:
    frame = _frame(1)
    c = frame.canvas
    assert (c.width, c.height) == (CANVAS_W, CANVAS_H), (
        f"画布该是捕获分辨率 {CANVAS_W}×{CANVAS_H}，收到 {c.width}×{c.height}"
    )
    assert frame.pixels.shape == (CANVAS_H, CANVAS_W, 3), (
        f"像素该是 ({CANVAS_H}, {CANVAS_W}, 3)，收到 {frame.pixels.shape}"
    )
    return f"画布 {c.width}×{c.height}，像素 {frame.pixels.shape}，dtype {frame.pixels.dtype}"


def check_raw_frame_channel_order_is_rgb() -> str:
    frame = _frame(2)
    ox, oy = _entry(2)["origin"]
    crop = _crop(2)
    # 费用条填充在夹具里的位置：原生 (2620, 950) 是纯白
    got = tuple(int(v) for v in frame.pixels[950, 2620])
    want = tuple(int(v) for v in crop[950 - oy, 2620 - ox])
    assert got == want == (255, 255, 255), f"费用条填充处该是 (255,255,255)，收到 {got}"
    return f"通道序 RGBA 解出来是 RGB：{got}"


def check_canvas_is_not_scaled() -> str:
    frame = _frame(3)
    ox, oy = _entry(3)["origin"]
    crop = _crop(3)
    got = frame.pixels[oy : oy + crop.shape[0], ox : ox + crop.shape[1]]
    diff = int(np.abs(got.astype(int) - crop.astype(int)).max())
    assert diff == 0, f"裁剪贴回原点后应当逐像素相等，最大差 {diff}（缩放过就会这样）"
    return f"{crop.shape[1]}×{crop.shape[0]} 逐像素相等，最大差 0"


def check_roi_is_taken_from_the_layout_table() -> str:
    frame = _frame(26)
    roi = frame.crop(_ui().rects["cost_digits"])
    x0, y0, x1, y1 = _digit_union(26)
    r = _ui().rects["cost_digits"]
    assert roi.shape[:2] == (r.height, r.width), (
        f"裁出来该是 {r.height}×{r.width}，收到 {roi.shape[:2]}"
    )
    inner = roi[y0 - r.y : y1 - r.y + 1, x0 - r.x : x1 - r.x + 1]
    assert inner.size > 0 and (inner.min(axis=2) > 190).sum() > 100, (
        "ROI 里应当装着这一帧的数字白像素"
    )
    return f"ROI {r.width}×{r.height} 里装着数字并集 {x0},{y0}..{x1},{y1}"


# ---------------------------------------------------------------------------
# 字形表资产
# ---------------------------------------------------------------------------


def check_glyph_table_builds_from_the_harvest() -> str:
    gen = _mod("ark.offline.assets_gen.glyphs")
    ui = _ui()
    with _writable_tmp() as td:
        dest = td / "glyphs"
        gen.build_digit_glyphs(
            _harvest(), dest, crops_dir=FIXTURES, rect=ui.rects["cost_digits"]
        )
        table = _mod("ark.engine.vision.glyphs").GlyphTable.load(dest)
        got = tuple(sorted(table.digits))
        assert got == tuple(range(10)), f"取材帧该覆盖 0–9，收到 {got}"
    return f"从 {len(_harvest())} 帧取材，得到字形 {list(got)}"


def check_shipped_glyph_table_reproduces_from_the_harvest() -> str:
    gen = _mod("ark.offline.assets_gen.glyphs")
    glyphs = _mod("ark.engine.vision.glyphs")
    ui = _ui()
    assert GLYPH_DIR.is_dir(), f"出厂字形表不在 {GLYPH_DIR}"
    shipped = glyphs.GlyphTable.load(GLYPH_DIR)
    with _writable_tmp() as td:
        dest = td / "glyphs"
        gen.build_digit_glyphs(
            _harvest(), dest, crops_dir=FIXTURES, rect=ui.rects["cost_digits"]
        )
        again = glyphs.GlyphTable.load(dest)
    assert set(shipped.digits) == set(again.digits), (
        f"出厂表 {sorted(shipped.digits)} 与重现出来的 {sorted(again.digits)} 不一样"
    )
    for d in shipped.digits:
        a, b = np.asarray(shipped.templates[d]), np.asarray(again.templates[d])
        assert a.shape == b.shape and (a == b).all(), (
            f"字形 {d} 对不上：出厂 {a.shape} vs 重现 {b.shape}"
            "——出厂表必须能从取材帧重现，否则它就不是取材帧产出的"
        )
    return f"{len(shipped.digits)} 个字形都能从取材帧逐像素重现"


def check_glyph_templates_are_actually_glyph_images() -> str:
    """A2 只比「重建 == 出厂」，**两边都错也一致** —— 所以另立这条。

    真出过一次：归一化给了 0..1 浮点，量化时忘了乘 255，模板被压成 0/1 两档；
    比对靠相关系数（尺度无关），D1 与 A2 照样全绿，**存进版本库的却不是字形图**。
    所以这里直接查模板长什么样：笔画得是亮的、灰度得是连续的。
    """
    table = _glyphs()
    bad = []
    for d in sorted(table.digits):
        t = np.asarray(table.templates[d])
        bright = int((t >= 200).sum())
        uniq = len(np.unique(t))
        if bright < 200:
            bad.append(f"字形 {d} 只有 {bright} 个像素 ≥200（该是笔画）")
        if uniq < 16:
            bad.append(f"字形 {d} 只有 {uniq} 个不同灰度（被压成几档了）")
    assert not bad, "；".join(bad)
    return (
        f"{len(table.digits)} 个模板都是连续灰度的字形图"
        f"（笔画像素 {min(int((np.asarray(table.templates[d]) >= 200).sum()) for d in table.digits)}"
        f"–{max(int((np.asarray(table.templates[d]) >= 200).sum()) for d in table.digits)}，"
        f"灰度档数 ≥16）"
    )


# ---------------------------------------------------------------------------
# 数字读数
# ---------------------------------------------------------------------------


def check_cost_digits_read_correctly_on_every_sample() -> str:
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Valued

    table = _glyphs()
    ui = _ui()
    bad = []
    for shot, want in sorted(DIGITS.items()):
        slot = cost.read_cost_digits(_frame(shot), table, ui=ui)
        if not isinstance(slot, Valued) or not isinstance(slot.payload, Exact):
            bad.append(f"shot{shot:02d}: 该读到 {want}，收到 {type(slot).__name__}")
            continue
        if slot.payload.value != want:
            bad.append(f"shot{shot:02d}: 该读到 {want}，读到 {slot.payload.value}")
    assert not bad, "；".join(bad[:8]) + (f"（共 {len(bad)} 帧不对）" if len(bad) > 8 else "")
    return f"{len(DIGITS)} 帧全部读对（1 位 8 帧、2 位 23 帧）"


def check_digits_are_read_not_remembered() -> str:
    """粘贴对抗：把 shot01 的数字区贴到 shot26 的面板上，读数该是 shot01 的值。

    按帧号记答案的实现在这条上必死。
    """
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Valued

    ui = _ui()
    r = ui.rects["cost_digits"]
    block = _crop(1)[r.y - _entry(1)["origin"][1] : r.y - _entry(1)["origin"][1] + r.height,
                    r.x - _entry(1)["origin"][0] : r.x - _entry(1)["origin"][0] + r.width]
    frame = _frame(26, patch=((r.x, r.y, r.width, r.height), block))
    slot = cost.read_cost_digits(frame, _glyphs(), ui=ui)
    assert isinstance(slot, Valued) and isinstance(slot.payload, Exact), (
        f"贴过去之后该读到 2，收到 {type(slot).__name__}"
    )
    assert slot.payload.value == 2, (
        f"贴过去的是 shot01 的数字（2），读到 {slot.payload.value}"
        "——按帧号记答案就会这样"
    )
    return f"shot01 的数字贴到 shot26 的面板上，读到 {slot.payload.value}"


def _last_glyph_cut(shot: int) -> tuple[int, int]:
    """末位字形在 ROI 里的相对起止列：返回 (间隙起点, 末位中点)。"""
    r = _ui().rects["cost_digits"]
    ox, oy = _entry(shot)["origin"]
    block = _crop(shot)[r.y - oy : r.y - oy + r.height, r.x - ox : r.x - ox + r.width]
    cols = (block.min(axis=2) > 190).any(axis=0)
    runs, start = [], None
    for i, on in enumerate(cols):
        if not on and start is None:
            start = i
        elif on and start is not None:
            runs.append((start, i - 1))
            start = None
    inner = [g for g in runs if g[0] > 0 and g[1] < len(cols) - 1]
    assert inner, f"shot{shot} 该是两位数字，中间该有一个列间隙"
    last_on = int(np.nonzero(cols)[0].max())
    gap = inner[-1][0]
    return gap, (gap + last_on) // 2


def check_missing_last_digit_is_not_a_shorter_number() -> str:
    """**整块**拿掉末位字形 —— 不许报成一个更短的确切数。

    这是这条链最危险的一种假事实：位数是"数有几个字形"数出来的，所以**缺一位的
    两位数会长得像一个一位数**。实测：真的一位数字形中心在 **2720–2722**，
    而缺了末位的 `1` 中心在 **2690**（偏左 31 px）—— 所以"居中于 x≈2721"
    这条事实**不只是描述，是判据**：偏了就是缺位。
    """
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Interval, Unknown, Unsampled, Valued

    ui = _ui()
    r = ui.rects["cost_digits"]
    ox, oy = _entry(16)["origin"]
    block = _crop(16)[r.y - oy : r.y - oy + r.height, r.x - ox : r.x - ox + r.width].copy()
    gap, _ = _last_glyph_cut(16)
    block[:, gap:] = 0  # shot16 的真值是 10；末位整块没了
    slot = cost.read_cost_digits(
        _frame(16, patch=((r.x, r.y, r.width, r.height), block)), _glyphs(), ui=ui
    )
    if isinstance(slot, Valued) and isinstance(slot.payload, Exact):
        raise AssertionError(
            f"末位整块没了，却报成一个更短的确切数 Exact({slot.payload.value})"
            "——真值是 10，这就是假事实"
        )
    if isinstance(slot, Valued) and isinstance(slot.payload, Interval):
        p = slot.payload
        assert 10 <= p.lo and p.hi <= 19, f"`1?` 该落在 [10,19] 里，收到 [{p.lo},{p.hi}]"
        return f"末位整块没了 → 区间 [{p.lo},{p.hi}]"
    assert isinstance(slot, (Unsampled,)) or (
        isinstance(slot, Valued) and isinstance(slot.payload, Unknown)
    ), f"该报区间或未知，收到 {type(slot).__name__}"
    return f"末位整块没了 → {type(slot).__name__}，没有给出更短的确切数"


def check_partly_occluded_last_digit_is_an_interval() -> str:
    """末位**被遮掉一半**（还剩一点笔画）→ 该是区间，不许猜整数。"""
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Interval, Unknown, Unsampled, Valued

    ui = _ui()
    r = ui.rects["cost_digits"]
    ox, oy = _entry(16)["origin"]
    block = _crop(16)[r.y - oy : r.y - oy + r.height, r.x - ox : r.x - ox + r.width].copy()
    _, mid = _last_glyph_cut(16)
    block[:, mid:] = 0
    slot = cost.read_cost_digits(
        _frame(16, patch=((r.x, r.y, r.width, r.height), block)), _glyphs(), ui=ui
    )
    if isinstance(slot, Valued) and isinstance(slot.payload, Exact):
        raise AssertionError(
            f"末位只剩一半，却报成确切数 Exact({slot.payload.value})——真值是 10"
        )
    if isinstance(slot, Valued) and isinstance(slot.payload, Interval):
        p = slot.payload
        assert 10 <= p.lo and p.hi <= 19, f"`1?` 该落在 [10,19] 里，收到 [{p.lo},{p.hi}]"
        return f"末位被遮一半 → 区间 [{p.lo},{p.hi}]"
    assert isinstance(slot, (Unsampled,)) or (
        isinstance(slot, Valued) and isinstance(slot.payload, Unknown)
    ), f"该报区间或未知，收到 {type(slot).__name__}"
    return f"末位被遮一半 → {type(slot).__name__}"


def check_missing_first_digit_is_not_a_shorter_number() -> str:
    """**首位**整块缺失 —— 同样不许报成一个更短的确切数。

    这一路的余量比末位缺失**小得多**：末位缺失时剩下的字形偏左 16 px 开外，
    而首位缺失时剩下的字形只偏 **15.0 px**（31 帧里最小的三帧是 shot05/19/23，
    都是 `11`——两个 `1` 都窄，去掉一个剩下的那个离中心最近）。
    所以文档把判据收到 **±8 px**：真读数全在 ±1.5 以内，两边各留 6.5 px。
    """
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Interval, Unknown, Unsampled, Valued

    ui = _ui()
    r = ui.rects["cost_digits"]
    bad, checked = [], 0
    for shot in sorted(DIGITS):
        if DIGITS[shot] < 10:
            continue
        ox, oy = _entry(shot)["origin"]
        block = _crop(shot)[r.y - oy : r.y - oy + r.height, r.x - ox : r.x - ox + r.width].copy()
        cols = (block.min(axis=2) > 190).any(axis=0)
        runs, start = [], None
        for i, on in enumerate(cols):
            if on and start is None:
                start = i
            elif not on and start is not None:
                runs.append((start, i - 1))
                start = None
        if start is not None:
            runs.append((start, len(cols) - 1))
        if len(runs) != 2:
            continue
        block[:, : runs[0][1] + 1] = 0  # 首位整块拿掉
        checked += 1
        want = DIGITS[shot]
        slot = cost.read_cost_digits(
            _frame(shot, patch=((r.x, r.y, r.width, r.height), block)), _glyphs(), ui=ui
        )
        if isinstance(slot, Valued) and isinstance(slot.payload, Exact):
            bad.append(f"shot{shot:02d}: 首位没了却报 Exact({slot.payload.value})，真值 {want}")
            continue
        if isinstance(slot, Valued) and isinstance(slot.payload, Interval):
            p = slot.payload
            if not (p.lo <= want <= p.hi):
                bad.append(f"shot{shot:02d}: 区间 [{p.lo},{p.hi}] 不含真值 {want}")
            continue
        if isinstance(slot, (Unsampled,)) or (
            isinstance(slot, Valued) and isinstance(slot.payload, Unknown)
        ):
            continue
        bad.append(f"shot{shot:02d}: 收到 {type(slot).__name__}")
    assert not bad, "；".join(bad[:6])
    return f"{checked} 帧涂掉首位，都报区间且都含真值，没有一帧塌成更短的确切数"


def check_occluded_digits_are_not_a_number() -> str:
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Interval, Unknown, Unsampled, Valued

    ui = _ui()
    r = ui.rects["cost_digits"]
    block = np.zeros((r.height, r.width, 3), np.uint8)
    frame = _frame(26, patch=((r.x, r.y, r.width, r.height), block))
    slot = cost.read_cost_digits(frame, _glyphs(), ui=ui)
    if isinstance(slot, Valued):
        assert isinstance(slot.payload, Unknown), (
            f"数字区整块涂黑了还给出 {type(slot.payload).__name__}"
            "——遮挡下报一个数就是假事实"
        )
        how = "有值但载荷是未知"
    else:
        assert isinstance(slot, Unsampled), (
            f"数字区整块涂黑了，收到 {type(slot).__name__}（该是未采样）"
        )
        how = "未采样"
    return f"涂黑后报「{how}」，没有给出数字"


# ---------------------------------------------------------------------------
# 费用条
# ---------------------------------------------------------------------------


def check_bar_fill_matches_on_every_sample() -> str:
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Valued

    ui = _ui()
    bad = []
    # 满态单独在 B2 里查：那里没有可观测的填充，读数是 Unknown 而不是 px
    samples = sorted(set(FILLS) - FULL)
    for shot in samples:
        want = FILLS[shot]
        slot = cost.read_cost_bar(_frame(shot), ui=ui)
        if not isinstance(slot, Valued) or not isinstance(slot.payload, Exact):
            bad.append(f"shot{shot:02d}: 该读到填充 {want} px，收到 {type(slot).__name__}")
            continue
        got = slot.payload.value
        if got != want:
            bad.append(f"shot{shot:02d}: 该 {want} px，读到 {got} px")
    assert not bad, "；".join(bad[:8]) + (f"（共 {len(bad)} 帧不对）" if len(bad) > 8 else "")
    return f"{len(samples)} 帧填充长度全对（0–153 px），满态那 {len(FULL)} 帧在 B2 里查"


def check_full_bar_is_not_read_as_empty() -> str:
    """满态与空态在「有没有自左端起的亮白填充」这个判据下长得一样——
    把满态读成 `Exact(0)` 就是假事实。"""
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Unknown, Unsampled, Valued

    ui = _ui()

    def read(shot):
        slot = cost.read_cost_bar(_frame(shot), ui=ui)
        assert isinstance(slot, Valued), f"shot{shot:02d}: 收到 {type(slot).__name__}"
        return slot.payload

    wrong = []
    for shot in sorted(FULL):
        p = read(shot)
        if isinstance(p, Exact):
            wrong.append(f"shot{shot:02d} 报成 Exact({p.value})")
        elif not isinstance(p, Unknown):
            wrong.append(f"shot{shot:02d} 报成 {type(p).__name__}")
    assert not wrong, (
        f"这些帧的轨道整条是中间灰 + 白 MAX（每列最亮 117–255），没有可观测的填充，"
        f"该报 Unknown：{'；'.join(wrong)}"
    )
    empty = []
    for shot in (18, 32):
        p = read(shot)
        if not (isinstance(p, Exact) and p.value == 0):
            empty.append(f"shot{shot:02d} 报成 {type(p).__name__}"
                         + (f"({p.value})" if isinstance(p, Exact) else ""))
    assert not empty, (
        f"这些帧的轨道整条是暗的（每列最亮 47–67），该报 Exact(0)：{'；'.join(empty)}"
    )
    return f"满 {sorted(FULL)} 报 Unknown；空 [18, 32] 报 Exact(0)——两种状态分开了"


def check_bar_reports_the_track_length() -> str:
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Valued

    ui = _ui()
    slot = cost.read_cost_bar(_frame(3), ui=ui)
    assert isinstance(slot, Valued) and isinstance(slot.payload, Exact), (
        f"收到 {type(slot).__name__}"
    )
    got = slot.payload.value
    assert got == FILLS[3], f"shot03 该读 {FILLS[3]} px，收到 {got} px"
    assert ui.rects["cost_bar"].width == TRACK_PX, (
        f"轨道全长该是 {TRACK_PX} px（x {TRACK_X0}..{TRACK_X1}），"
        f"cost_bar 宽 {ui.rects['cost_bar'].width}"
    )
    return f"shot03 填充 {got} px；轨道全长 {TRACK_PX} px，不是搜索窗口的宽度"


# ---------------------------------------------------------------------------
# 判据的形状：不许是"单列"，也不许是"整条极值"
#
# 这一组不查"阈值对不对"，查的是**判据建立在什么量上**。做法是把夹具的
# `cost_bar` 里**一列**涂成另一个值，看状态会不会被这一列翻掉。
# ---------------------------------------------------------------------------

GATE_COLS: tuple[int, ...] = (0, 1, 2, 3)
"""第一级判据所在的那一段：窗口最左 4 列。扰动只打在这里。"""

FAR_COLS: tuple[int, ...] = (76, 116)
"""条中段的两列（x = 2665 / 2705），用来验「空」那一级的形状。"""

EMPTY_SHOTS: tuple[int, ...] = (18, 32)
"""整条暗的夹具（`FILLS` 是 0）。"""

BENEATH_BRIGHT = 189
"""刚好在「亮白」线（190）下面一格 —— 实测录像通道上第 0 列掉到 188–189。"""

ABOVE_DARK = 71
"""刚好在「空」的线（70）上面一格 —— 实测 `m15b` 的空条到 71。"""


def _column_patch(col: int, value: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """把 `cost_bar` 的第 `col` 列整列涂成一个常量 —— 扰动用的贴片。"""
    r = _ui().rects["cost_bar"]
    return ((r.x + col, r.y, 1, r.height), np.full((r.height, 1, 3), value, np.uint8))


def check_one_bright_column_cannot_invent_a_fill() -> str:
    """「有没有开始填」不许挂在一列上 —— 涂亮一列不许**伪造**出填充。

    填充是**自左端起**的，所以"第 1 列亮、第 0 列暗"根本不是填充。空条里
    **非最左**的任意一列被涂成亮白，都不许报出一个 > 0 的 `Exact` —— 那是假事实
    （契约层「让不知道在类型上没法被伪装成知道」）。
    """
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Valued

    ui = _ui()
    bad = []
    for shot in EMPTY_SHOTS:
        for col in GATE_COLS[1:]:
            slot = cost.read_cost_bar(_frame(shot, patch=_column_patch(col, 255)), ui=ui)
            if (
                isinstance(slot, Valued)
                and isinstance(slot.payload, Exact)
                and slot.payload.value > 0
            ):
                bad.append(f"shot{shot:02d} 第 {col} 列涂亮 → 报出填充 {slot.payload.value} px")
    assert not bad, (
        "空条只有非最左的一列被涂亮，就报出了一个并不存在的填充数"
        "（填充自左端起，第 1 列亮而第 0 列暗不是填充）：" + "；".join(bad)
    )
    return f"空条第 {list(GATE_COLS[1:])} 列各自涂亮，都没有伪造出填充"


def check_one_dark_column_cannot_erase_a_fill() -> str:
    """涂暗一列不许**抹掉**填充。

    填充态最左 4 列里任意一列被压到 **189**（刚好在「亮白」线下面），条体那
    九十多列还是亮的 —— 不许因此变成 `Unknown()` 或 `Exact(0)`。
    """
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Valued

    ui = _ui()
    bad = []
    for shot in (3, 6):
        want = FILLS[shot]
        for col in GATE_COLS:
            slot = cost.read_cost_bar(_frame(shot, patch=_column_patch(col, BENEATH_BRIGHT)), ui=ui)
            ok = (
                isinstance(slot, Valued)
                and isinstance(slot.payload, Exact)
                and slot.payload.value > 0
            )
            if not ok:
                got = (
                    type(slot.payload).__name__
                    if isinstance(slot, Valued)
                    else type(slot).__name__
                )
                bad.append(f"shot{shot:02d}（真值 {want} px）第 {col} 列压到 {BENEATH_BRIGHT} → 报成 {got}")
    assert not bad, "填充态只有一列被压暗，整个填充就丢了：" + "；".join(bad)
    return f"填充态最左 {len(GATE_COLS)} 列各自压到 {BENEATH_BRIGHT}，填充都还在"


def check_empty_is_not_decided_by_the_brightest_column() -> str:
    """「空」不许看整条**最亮**的那一列。

    空条**中段**任意一列被抬到 **71**（实测录像通道上"空"的上界），必须仍报
    `Exact(0)` —— 一列异常不该把状态翻成"读不出"。
    """
    cost = _mod("ark.engine.vision.cost")
    from ark.contracts import Exact, Valued

    ui = _ui()
    bad = []
    for shot in EMPTY_SHOTS:
        for col in FAR_COLS:
            slot = cost.read_cost_bar(_frame(shot, patch=_column_patch(col, ABOVE_DARK)), ui=ui)
            ok = (
                isinstance(slot, Valued)
                and isinstance(slot.payload, Exact)
                and slot.payload.value == 0
            )
            if not ok:
                got = (
                    type(slot.payload).__name__
                    if isinstance(slot, Valued)
                    else type(slot).__name__
                )
                bad.append(f"shot{shot:02d} 第 {col} 列抬到 {ABOVE_DARK} → 报成 {got}")
    assert not bad, (
        "一列异常就把空条翻掉了（拿整条最亮的那一列当判据，天生没有余量）："
        + "；".join(bad)
    )
    return f"空条第 {list(FAR_COLS)} 列各自抬到 {ABOVE_DARK}，都还报 Exact(0)"


CHECKS: tuple[Check, ...] = (
    Check("S0", "本文件的真值表（自检，不查 T3）",
          "31 帧的真值能从夹具像素重算出来",
          check_truth_table_is_recomputable_from_the_fixtures),
    Check("L1", "视觉层/顶部HUD与数字 §区域与常量",
          "UI 布局常量是那 8 个具名矩形", check_ui_layout_has_the_eight_rects),
    Check("L2", "视觉层/顶部HUD与数字 §区域与常量",
          "每个矩形的值覆盖文档窗口，外扩不超过 12 px",
          check_ui_layout_values_cover_the_documented_windows),
    Check("L3", "视觉层/顶部HUD与数字 §每个区域必须知道的事实",
          "cost_bar 就是轨道本体（硬边，不外扩）",
          check_bar_constant_is_the_track_itself),
    Check("L4", "视觉层/顶部HUD与数字 §每个区域必须知道的事实",
          "cost_digits 窗口装得下 1 位与 2 位数字的并集",
          check_digit_window_holds_one_and_two_digit_unions),
    Check("L5", "契约层 §2（含两端）+ 设备层 §4（画布 2800×1260）",
          "每个窗口都落在画布内，右缘不超过 2799",
          check_windows_stay_inside_the_canvas),
    Check("L6", "视觉层/顶部HUD与数字 §每个区域必须知道的事实",
          "规格写死的判据阈值，代码里的常量必须相等",
          check_spec_thresholds_are_pinned_to_the_code),
    Check("F1", "设备层 §4 + 契约层 §2",
          "原始帧解出捕获分辨率的画布，不缩放",
          check_raw_frame_decodes_to_the_capture_canvas),
    Check("F2", "设备层 §4",
          "原始帧的通道序是 RGBA", check_raw_frame_channel_order_is_rgb),
    Check("F3", "契约层 §2",
          "画布就是捕获分辨率本身，一个像素都没缩",
          check_canvas_is_not_scaled),
    Check("R1", "视觉层/顶部HUD与数字 §区域与常量",
          "ROI 从 UiLayout 的矩形裁出来", check_roi_is_taken_from_the_layout_table),
    Check("A1", "视觉层/识别方法 §已定：数字一律走固定字形模板",
          "从取材帧能生成覆盖 0–9 的字形表",
          check_glyph_table_builds_from_the_harvest),
    Check("A2", "资产生成器/特征库 + 视觉层/识别方法 §已定",
          "出厂字形表能从取材帧逐像素重现",
          check_shipped_glyph_table_reproduces_from_the_harvest),
    Check("A3", "视觉层/识别方法 §已定：数字一律走固定字形模板",
          "字形模板真的是字形图，不是被压成几档的数",
          check_glyph_templates_are_actually_glyph_images),
    Check("D1", "状态层/全局与HUD §当前费用",
          "31 帧的费用整数全部读对",
          check_cost_digits_read_correctly_on_every_sample),
    Check("D2", "契约层「这一层唯一的验收判据」",
          "数字是读出来的，不是按帧号记的", check_digits_are_read_not_remembered),
    Check("D3", "视觉层/顶部HUD与数字 §每个区域必须知道的事实（数字居中于 x≈2721）",
          "末位整块缺失时，不许报成一个更短的确切数",
          check_missing_last_digit_is_not_a_shorter_number),
    Check("D5", "视觉层/识别方法 §已查：逐位歧义降成区间",
          "末位被遮一半时降成区间，不猜整数",
          check_partly_occluded_last_digit_is_an_interval),
    Check("D6", "视觉层/顶部HUD与数字 §每个区域必须知道的事实（数字居中于 x≈2721）",
          "首位整块缺失时，也不许报成一个更短的确切数",
          check_missing_first_digit_is_not_a_shorter_number),
    Check("D4", "契约层 §1 + 状态层 README「这一层的验收标准」",
          "数字被遮挡时不给出数字", check_occluded_digits_are_not_a_number),
    Check("B1", "视觉层/顶部HUD与数字 §每个区域必须知道的事实",
          "31 帧的填充长度全部读对", check_bar_fill_matches_on_every_sample),
    Check("B2", "视觉层/顶部HUD与数字 §每个区域必须知道的事实",
          "满态报 Unknown，不许报成 Exact(0)", check_full_bar_is_not_read_as_empty),
    Check("B3", "视觉层/顶部HUD与数字 §每个区域必须知道的事实",
          "轨道全长是实测的 153 px，不是搜索窗口的宽度",
          check_bar_reports_the_track_length),
    Check("B4", "视觉层/顶部HUD与数字 §每个区域必须知道的事实（判据的形状）",
          "空条非最左的任意一列涂亮，不许伪造出填充",
          check_one_bright_column_cannot_invent_a_fill),
    Check("B5", "视觉层/顶部HUD与数字 §每个区域必须知道的事实（判据的形状）",
          "填充态最左 4 列里任意一列压到 189，不许丢掉填充",
          check_one_dark_column_cannot_erase_a_fill),
    Check("B6", "视觉层/顶部HUD与数字 §每个区域必须知道的事实（判据的形状）",
          "空条中段任意一列抬到 71，必须仍报 Exact(0)",
          check_empty_is_not_decided_by_the_brightest_column),
)


def main() -> int:
    failures: list[tuple[Check, str]] = []
    print(f"费用读数链规格符合性验收 —— {len(CHECKS)} 条\n")
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
    sys.exit(main())
