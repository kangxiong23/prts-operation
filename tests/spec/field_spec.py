"""场上我方干员那三条读数链的**规格符合性验收** —— 由规划者维护，不住在 tests/unit/。

分工、纪律、跑法见 [README](README.md) —— **不在这里重述**。**执行者不许读本文件。**

## 用例从哪来

全部来自**规格与用户裁定**，不从执行者的实现、汇报或它自己的单测反推：

- `docs/10-规格/状态层/我方干员.md` 那张字段表（**允许的形态是字段级的**）；
- `docs/10-规格/视觉层/场上我方干员.md`（§1 三样是什么 · §2 读亮区比例、不数格子 · §3 菱形是搜出来的图标）；
- 用户裁定：**线上只读条**（读数字是文字识别，比读条贵，不做线上通道）· 看不到≠满血 ·
  技力生效期间读不出来 · 弹药分界模糊给下界 · **不许拿验收帧当素材** · 漏检方向是假事实；
- `docs/10-规格/采集与验收/基准集.md` §2（原点+尺寸落在画布内 · 越出要拒绝）与 §3（**裁剪要贴回画布**）。

## 坐标这一条是踩过的坑

`load_batch` 吐出来的帧是**整张画布**（裁剪被贴回原处），所以取区必须用**画布坐标** ——
批次 `取材.json` 里 `crops[].window` 给的就是画布坐标。第一版拿裁剪坐标去开窗，
读到的是画布左上角那片黑边，于是**整批读数都是未采样而用例照样绿**（空过）。
所以下面每一条探针都带一句"这一批声明覆盖了什么，读数里就必须看得见什么"。

## 真值对比这一条的两处来源

真值**只从这里来**：`field_ops_truth.md`（规划者看图逐帧读出来的面板数与那个 `N/M` 读数框）。
**不读执行者 `取材.json` 里那份 `truth`** —— 那是它自己看图的产物，拿它当判据就是拿它的
读数验证它的读数。

- **F10 / F11**：跑在**执行者裁的真值批**上（帧与取区来自它的 `取材.json`，真值来自我这份表）；
- **F12**：跑在**规划者自己新采的 5 帧**上（`data/probe/sp_series/`，**执行者从没见过**）；
- **F16**：菱形那个布尔（在 / 不在）也由**我看图逐帧读**（`DIAMOND_PRESENT`，16 帧），
  取证图 `data/probe/diamond_sheet.png` —— 夹具 `取材.json` 的 `covers` 只是**抄本**。

**`look_now` 不进用例**：那一帧面板写满血，而"离屏幕中心最近的那根"读到 0.774，执行者列出的
候选里另有 4 根是满的 —— **哪一根配那行数，我定不了**。真值配对定不了的帧不能当判据，
它进 `verdict.md` 的存疑，不进这里。
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _entry in (str(REPO / "src"), str(_HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

BATCH = REPO / "tests" / "帧样本" / "场上干员"
TRUTH_BATCH = REPO / "tests" / "帧样本" / "场上干员真值"
READINGS = BATCH / "判据读数.json"
TRUTH_READINGS = TRUTH_BATCH / "判据读数.json"
DIAMOND_ASSET = REPO / "assets" / "features" / "ready_diamond" / "diamond.npz"
SP_SERIES = REPO / "data" / "probe" / "sp_series"
"""规划者自己采的那一批（执行者没见过的真值帧）。**gitignored**，缺席就跳过这一条。"""

_TAKING = json.loads((BATCH / "取材.json").read_text("utf-8"))
WINDOW_SIZE = tuple(float(v) for v in _TAKING["window_size"])
BAR_OFFSET = tuple(float(v) for v in _TAKING["bars_in_window"])
"""取区尺寸，以及**血条左上角在取区里**的偏移 —— 都从批次自己的 `取材.json` 读，不另抄一份。"""

BAR_COLUMNS = 146
"""血条在画布上有多少列 —— 批次 README 与 `判据读数.json` 的 `bar_pixels / 8` 都是这个数。"""

# 状态层那张字段表里，这三个字段**允许的形态**（其余形态出现就是越界）。
HEALTH_FORMS = frozenset({"exact", "interval", "lowerbound", "unsampled"})
SKILL_FORMS = frozenset({"exact", "interval", "unsampled"})
MEANING_FORMS = frozenset({"exact", "lowerbound", "unknown"})

# ---------------------------------------------------------------------------
# 独立真值（出自 field_ops_truth.md，规划者逐帧看图读的）
# ---------------------------------------------------------------------------

TRUTH_LIFE: dict[str, tuple[int, int]] = {
    "p00": (2351, 2867),
    "p01": (2322, 2867),
    "p02": (2867, 2867),
    "p03": (2839, 2867),
    "p04": (2839, 2867),
    "p05": (2810, 2867),
    "p06": (2781, 2867),
    "p07": (2753, 2867),
    "p08": (329, 1801),
    "p09": (329, 1801),
    "p10": (329, 1801),
    "p11": (329, 1801),
    "sp_now": (329, 1801),
}
"""真值批里**配对定得下来**的那 13 帧。`look_now` 不在这里（见文件头）。"""

TRUTH_SP: dict[str, tuple[int, int]] = {
    "sp_now": (16, 26),
    "s00": (3, 40),
    "s01": (13, 26),
    "s04": (6, 30),
}
"""技力那一侧的真值 —— 出自那个 `⚡N/M` / 电池 `N/M` 小读数框。"""

SP_SERIES_TRUTH: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "s00": ((1234, 2560), (3, 40)),
    "s01": ((1608, 1608), (13, 26)),
    "s02": ((2153, 2153), (13, 15)),
    "s03": ((2153, 2153), (13, 15)),
    "s04": ((773, 2560), (6, 30)),
}
"""规划者新采那 5 帧：`{帧名: (生命真值, 技力真值)}`。

`s02` / `s03` 曾经是存疑（面板 Mon3tr 满血、读数 13/15，而我在别处量到的那根条两样都对不上）。
**取区定成 (1447, 389, 240, 360) 之后两边都对上了**：生命 1.0000 = 2153/2153 ✓、
技力 0.86986 = 127/146 ≈ 13/15 ✓。所以它们进来了。"""

DIAMOND_PRESENT: dict[str, bool] = {
    "cd01_a": False,
    "cd01_b": False,
    "cd01_c": False,
    "cd12_a": False,
    "shot01_a": False,
    "shot21_a": False,
    "shot26_a": False,
    "shot31_a": False,
    "shot02_a": True,
    "shot04_a": True,
    "shot30_a": True,
    "shot22_a": True,
    "board_a": True,
    "board_b": True,
    "field_a": True,
    "field_b": True,
}
"""**规划者看图逐帧读出来的**「那个黄菱形在不在」—— 16 帧 = 8 在 / 8 不在。

「在」＝**黄色**菱形（中心一枚黑三叉）；「不在」＝画面里没有它。**最像它的两条邻居不是它**：
`cd01_c` / `cd12_a` 上是**灰色**菱形，`cd01_a` 上是**黄色圆环 + 数字 `3`**（倒计时那个图形）。
取证图 `data/probe/diamond_sheet.png`（16 格逐格裁出菱形邻域、×2 放大、画上链匹配到的落点）。

**它不来自执行者的 `取材.json`**（那份是它自己看图的产物）—— 两者相符由 F16 自己核出来：
我这张表是**主本**，夹具那份只是抄本。"""

TOLERANCE = 0.025
"""**条是 146 列**，1 列 = 0.0068。取 0.025 ≈ 3.7 列 —— 规划者自己量同一根条时，
填充的左缘会比取区声明的 (40, 320) 偏左 2–3 列（不同帧的条实例有自己的亚像素偏移），
所以"3 列以内"是这条用例该有的宽。**这个宽不是为执行者让的**：把 0.015 换成 0.025
之后，下面每一条的判定一个字都没变。"""


@dataclass(frozen=True)
class Check:
    """一条规格检查：能判通过/失败的最小单位。"""

    ident: str
    claim: str
    spec: str
    probe: Callable[[], None]


# ---------------------------------------------------------------------------
# 取工具
# ---------------------------------------------------------------------------


def _field():
    return importlib.import_module("ark.engine.vision.field")


def _contracts():
    return importlib.import_module("ark.contracts")


def _frame_mod():
    return importlib.import_module("ark.engine.vision.frame")


def _rect(*values: float):
    return _contracts().PixelRect(*values)


def _cases() -> list:
    batch = importlib.import_module("ark.offline.bench.batch").load_batch(BATCH)
    return list(batch.cases)


def _frames() -> dict:
    build = _frame_mod().frame_from_raw
    return {case.name: build(case.raw()) for case in _cases()}


def _window_of(name: str):
    """取区（**画布坐标**）：批次 `取材.json` 里那一条。"""
    for crop in _TAKING["crops"]:
        if crop["file"][:-4] == name:
            return _rect(*[float(v) for v in crop["window"]])
    raise KeyError(name)


def _bar_of(name: str):
    """血条那块矩形（**画布坐标**）= 取区原点 + 批次声明的偏移。"""
    window = _window_of(name)
    return _rect(
        window.x + BAR_OFFSET[0], window.y + BAR_OFFSET[1], float(BAR_COLUMNS), 8.0
    )


def _form(slot: object) -> str:
    """形态名：`Valued` 看它 payload 的类名，其余看槽自己的类名。全是小写。"""
    if type(slot).__name__ != "Valued":
        return type(slot).__name__.lower()
    return type(getattr(slot, "payload")).__name__.lower()


def _form_counts(slots: list) -> dict:
    counts: dict = {}
    for slot in slots:
        counts[_form(slot)] = counts.get(_form(slot), 0) + 1
    return counts


def _small_frame(pixels: np.ndarray | None = None):
    """一块 300×420 的小画布（取区落在 (40, 20)）—— 给空白与合成那两条用例用。"""
    contracts, frame_mod = _contracts(), _frame_mod()
    width, height = 300, 420
    if pixels is None:
        pixels = np.full((height, width, 3), 32, dtype=np.uint8)
    return frame_mod.Frame(pixels=pixels, canvas=contracts.Canvas(width, height))


def _small_window():
    return _rect(BAR_OFFSET[0], BAR_OFFSET[1] - 300.0, *WINDOW_SIZE)


def _column_colour(frame, bar, column: int) -> np.ndarray:
    """血条那一行里第 `column` 列的**中位颜色**（画布坐标）。"""
    top, left = int(bar.y), int(bar.x)
    rows = frame.pixels[top : top + int(bar.height), left + column]
    return np.median(rows, axis=0)


def _paint_bar(base_pixels: np.ndarray, filled_columns: int, fill_colour, track_colour):
    """在小画布上把血条重画成"恰好填了 `filled_columns` 列"，其余原样。"""
    pixels = base_pixels.copy()
    window = _small_window()
    left = int(window.x + BAR_OFFSET[0])
    top = int(window.y + BAR_OFFSET[1])
    for column in range(BAR_COLUMNS):
        colour = fill_colour if column < filled_columns else track_colour
        pixels[top : top + 8, left + column] = colour
    return _small_frame(pixels)


def _paint_skill(
    base_pixels: np.ndarray,
    filled_columns: int,
    fill_colour,
    track_colour,
    fade_columns: int = 0,
):
    """在小画布上画一根**技力条**：血条那一行整条填满（让链找得到条），技力那一行按参数画。

    技力那一行的位置**不照抄它的常量**：血条顶往下 8…20 行整段都铺上，无论那一行落在
    哪里读到的都是这一块。`fade_columns` 是**边界渐变**那几列（从填充色线性走到轨道色）。
    """
    pixels = base_pixels.copy()
    window = _small_window()
    left = int(window.x + BAR_OFFSET[0])
    hp_top = int(window.y + BAR_OFFSET[1])
    fill = np.asarray(fill_colour, dtype=float)
    track = np.asarray(track_colour, dtype=float)
    columns = []
    for column in range(BAR_COLUMNS):
        if column < filled_columns:
            columns.append(fill)
        elif column < filled_columns + fade_columns:
            step = (column - filled_columns + 1) / (fade_columns + 1)
            columns.append(np.round(fill + (track - fill) * step))
        else:
            columns.append(track)
    for column, colour in enumerate(columns):
        pixels[hp_top : hp_top + 8, left + column] = (110, 190, 228)
        pixels[hp_top + 8 : hp_top + 21, left + column] = colour.astype(np.uint8)
    return _small_frame(pixels)


def _f13_skill_fill_known() -> None:
    """**技力那一行也要能读亮区比例** —— 自己画一根已知填充的技力条，读数必须跟着走。

    `F4` 只画了**血条**那一行；技力那一行一条合成用例都没有，而这一格坏就坏在那里。
    轨道色取三种：暗的，以及两种**亮的**（上一版在亮底上会把整条判成填满）——
    三种轨道色与填充色的逐通道差都远大于任何一条容差线，所以"填着的列"没有歧义。

    绿条的含义按《视觉层/场上我方干员》§1 就是"充能中"，所以含义也要跟着对。
    """
    field = _field()
    fill = (150, 200, 90)
    for track in ((45, 45, 48), (140, 200, 230), (205, 115, 35)):
        for filled in (11, 73, 118):
            frame = _paint_skill(_small_frame().pixels, filled, fill, track)
            truth = filled / BAR_COLUMNS
            slot, meaning = field.read_skill(frame, _small_window())
            ok, how = _consistent(slot, truth)
            assert ok, (
                f"自己画的技力条（填 {filled} 列 = {truth:.4f}，轨道色 {track}）"
                f"读成 {how}（含义 {meaning!r}）"
            )
            assert _form(slot) != "exact" or abs(
                float(slot.payload.value) - truth
            ) <= TOLERANCE, f"填充 {filled} 列，精确值报成 {slot.payload.value}"
            assert _form(meaning) == "exact" and meaning.payload.value == "充能中", (
                f"绿条（{fill}）的含义该是「充能中」，读成 {meaning!r}"
            )


def _valued_values(slots: list) -> list:
    out = []
    for slot in slots:
        if _form(slot) == "exact":
            out.append(float(slot.payload.value))
    return out


def _truth_windows() -> dict:
    """真值批每一帧的取区（**执行者裁的**，画布坐标）—— 真值不从这里来。"""
    taking = json.loads((TRUTH_BATCH / "取材.json").read_text("utf-8"))
    return {
        crop["file"][:-4]: _rect(*[float(v) for v in crop["window"]])
        for crop in taking["crops"]
    }


def _truth_frames() -> dict:
    build = _frame_mod().frame_from_raw
    batch = importlib.import_module("ark.offline.bench.batch").load_batch(TRUTH_BATCH)
    return {case.name: build(case.raw()) for case in batch.cases}


def _series_frame(stem: str):
    """规划者新采那一批的某一帧（原始 `.raw`，带 16 字节头）。"""
    return _frame_mod().frame_from_raw((SP_SERIES / f"{stem}.raw").read_bytes())


def _consistent(slot: object, truth: float) -> tuple[bool, str]:
    """读数与真值相容吗。**看形态**：精确比差、区间要罩住、下界不许高过真值。"""
    form = _form(slot)
    if form == "exact":
        value = float(slot.payload.value)
        return abs(value - truth) <= TOLERANCE, f"精确 {value:.6f}"
    if form == "interval":
        lo, hi = float(slot.payload.lo), float(slot.payload.hi)
        return lo - TOLERANCE <= truth <= hi + TOLERANCE, f"区间 [{lo:.6f}, {hi:.6f}]"
    if form == "lowerbound":
        lo = float(slot.payload.lo)
        return lo <= truth + TOLERANCE, f"下界 {lo:.6f}"
    return False, f"{form}（真值 {truth:.6f} 那里有东西，不许落未采样）"


# ---------------------------------------------------------------------------
# 探针
# ---------------------------------------------------------------------------


def _f1_health_forms() -> None:
    """血条只许出字段表允许的形态，而且**这一批声明的九档填充要看得见**。"""
    field = _field()
    slots = [field.read_health(frame, _window_of(name)) for name, frame in _frames().items()]
    counts = _form_counts(slots)
    bad = sorted(set(counts) - HEALTH_FORMS)
    assert not bad, f"血条出现了字段表没收的形态：{bad}（全部：{counts}）"
    values = _valued_values(slots)
    assert values, f"整批一条「读到值」都没有（全部：{counts}）—— 夹具声明覆盖了九档填充，读数里得看得见"
    assert len(set(values)) > 1, (
        f"读到的值只有一种（{sorted(set(values))}）—— 夹具声明覆盖了九档不同的填充，"
        f"读数不该是一条直线"
    )


def _f2_skill_forms() -> None:
    """技力那个字段只许出「精确 / 区间 / 未采样」——**不许出下界或上界**。"""
    field = _field()
    sp_forms, fill_forms, mean_forms = [], [], []
    for name, frame in _frames().items():
        fill, meaning = field.read_skill(frame, _window_of(name))
        fill_forms.append(_form(fill))
        mean_forms.append(_form(meaning))
        sp_forms.append(_form(field.sp_of(fill, meaning)))
    assert set(mean_forms) <= MEANING_FORMS, (
        f"「技能是否生效中」出现了字段表没收的形态：{sorted(set(mean_forms))}"
    )
    bad = sorted(set(sp_forms) - SKILL_FORMS)
    assert not bad, (
        f"技力字段出现了字段表没收的形态：{bad}（技力那一行的形态是 {sorted(SKILL_FORMS)}）"
        f" ｜ 填充那一侧出现过：{sorted(set(fill_forms))} ｜ 含义那一侧：{sorted(set(mean_forms))}"
    )
    assert "unsampled" in sp_forms, (
        f"技力字段整批都没落过未采样（{sorted(set(sp_forms))}）—— 夹具声明里有生效中与弹药分格那几档"
    )


def _f3_blank_is_unsampled() -> None:
    """**看不到 ≠ 满血**：没有条的画面必须落未采样，不许给出一个比例。"""
    slot = _field().read_health(_small_frame(), _small_window())
    assert _form(slot) == "unsampled", f"空白画面读成了 {_form(slot)}：{slot!r} —— 看不到必须是未采样"


def _f4_known_fill() -> None:
    """**读亮区占总长的比例**：自己画一根已知填充的条，读数必须跟着走。"""
    field, frames = _field(), _frames()
    fill_colour = _column_colour(frames["cd01_a"], _bar_of("cd01_a"), 2)
    track_colour = _column_colour(frames["cd12_a"], _bar_of("cd12_a"), BAR_COLUMNS - 3)
    assert not np.array_equal(fill_colour, track_colour), (
        f"取样失败：填充色与轨道色一样（都是 {fill_colour}）—— 先查这条用例的取样列"
    )
    base = _small_frame()
    for fraction in (0.25, 0.5, 0.75):
        columns = round(fraction * BAR_COLUMNS)
        slot = field.read_health(_paint_bar(base.pixels, columns, fill_colour, track_colour), _small_window())
        form = _form(slot)
        assert form in {"exact", "interval"}, f"画了 {fraction} 却落成 {form}：{slot!r}"
        if form == "interval":
            lo, hi = float(slot.payload.lo), float(slot.payload.hi)
            assert lo - 0.06 <= fraction <= hi + 0.06, f"画了 {fraction}，区间给的是 [{lo}, {hi}]"
        else:
            value = float(slot.payload.value)
            assert abs(value - fraction) <= 0.06, (
                f"画了 {fraction}（{columns}/{BAR_COLUMNS} 列），读出来 {value}"
            )


def _f5_diamond_is_searched() -> None:
    """菱形是**搜出来的**：取区挪一小段，结论不许变、**报出来的落点也不许跟着挪**。

    落点是**画布坐标**，所以"图标在画布上没动"就意味着落点必须一模一样 —— 钉在小框里
    的那种做法给的是固定的**窗内**偏移，一挪窗落点就跟着挪。
    """
    field, frames = _field(), _frames()
    template = field.load_ready_diamond()

    def verdict(name: str, window) -> bool:
        slot = field.find_ready_diamond(frames[name], window, template)
        return bool(getattr(getattr(slot, "payload", None), "value", False))

    present = [name for name in frames if verdict(name, _window_of(name))]
    assert present, (
        "整批里一条「菱形在」都没搜到 —— 夹具声明 8 条有它；要么这条用例的坐标错了，"
        "要么搜索本身不成立"
    )
    for name in present:
        window = _window_of(name)
        moved = _rect(window.x - 20.0, window.y + 20.0, window.width, window.height)
        assert verdict(name, moved), (
            f"{name}：取区只挪 (−20, +20)（图标还在窗里）就找不到了 —— 那不是「搜」"
        )
        before = field.diamond_score(frames[name], window, template)
        after = field.diamond_score(frames[name], moved, template)
        assert (before[1], before[2]) == (after[1], after[2]), (
            f"{name}：取区挪了，报出来的落点也跟着挪（({before[1]}, {before[2]}) → "
            f"({after[1]}, {after[2]})）—— 落点是画布坐标，图标没动它就不该动"
        )


def _f6_no_digit_channel() -> None:
    """**线上只读条**：这一层不许把读数字那一套接进来（用户裁定）。"""
    source = (REPO / "src" / "ark" / "engine" / "vision" / "field.py").read_text("utf-8")
    imported: set = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    offenders = sorted(n for n in imported if "glyph" in n.lower() or "ocr" in n.lower())
    assert not offenders, f"这一层 import 了读数字那一套：{offenders} —— 线上只读条"


def _f7_readings_carry_truth() -> None:
    """判据读数必须带**与独立真值的逐条对比**（不是把这条链自己的读数聚成两簇）。

    真值对比落在**真值批**那份读数里（第一批那份是灵敏度与自聚类，另一回事）。
    逐条的意思是**每一帧一条**，不是只有一份汇总。
    """
    text = TRUTH_READINGS.read_text("utf-8")
    data = json.loads(text)
    frames = data.get("frames") or []
    assert frames, "真值批的判据读数里没有逐帧那一段 —— 只有汇总的话量不出是哪一帧偏了"
    missing = [
        frame.get("name")
        for frame in frames
        if frame.get("truth", {}).get("life") and "health_error" not in frame
    ]
    assert not missing, f"这些帧有生命真值却没有逐条误差：{missing}"
    assert "sp_errors" in data and data["sp_errors"].get("count"), (
        "技力那一侧的误差分布没有 —— 自聚类的空档量不出比例准不准："
        "一个系统性偏掉的读数照样能聚出漂亮的两簇"
    )
    assert any(marker in text for marker in ("真值", "面板", "误差")), (
        "读数里连一句「真值 / 面板 / 误差」都没有"
    )


def _f10_health_matches_truth() -> None:
    """血条读数与**独立真值**相容：精确的差在两列以内，区间必须罩住真值。"""
    field, frames, windows = _field(), _truth_frames(), _truth_windows()
    bad = []
    for name, (current, maximum) in TRUTH_LIFE.items():
        truth = current / maximum
        slot = field.read_health(frames[name], windows[name])
        ok, how = _consistent(slot, truth)
        if not ok:
            bad.append(f"{name} 真值 {current}/{maximum} = {truth:.6f}，读数 {how}")
    assert not bad, "血条读数与独立真值对不上：" + " ｜ ".join(bad)


def _f11_skill_matches_truth() -> None:
    """技力条**填充比例**与独立真值相容（真值来自那个 `N/M` 读数框）。

    查的是 `read_skill` 的**填充**那一侧，不是 `sp_of` —— 弹药 / 生效中那一档，
    `sp_of` 按规格就该落未采样，那是"读不出技力"，不是"比例读错了"。
    """
    field, frames, windows = _field(), _truth_frames(), _truth_windows()
    name = "sp_now"
    current, maximum = TRUTH_SP[name]
    truth = current / maximum
    fill, meaning = field.read_skill(frames[name], windows[name])
    ok, how = _consistent(fill, truth)
    assert ok, (
        f"{name}：真值 {current}/{maximum} = {truth:.6f}，填充读数 {how}（含义 {meaning!r}）"
    )


def _f12_new_frames_match_truth() -> None:
    """**执行者没见过的 3 帧**：血条与技力条都要与真值相容。

    这一批是规划者自己采的（`data/probe/sp_series/`），取区由规划者自己算
    （血条左上角 (1487, 709) 减 `bars_in_window`），执行者的夹具里没有它们。
    """
    if not (SP_SERIES / "s00.raw").exists():
        return  # 本机没采过就跳过（gitignored）
    field = _field()
    window = _rect(1447.0, 389.0, *WINDOW_SIZE)
    bad = []
    for name, (life, skill) in SP_SERIES_TRUTH.items():
        frame = _series_frame(name)
        life_truth = life[0] / life[1]
        life_slot = field.read_health(frame, window)
        ok, how = _consistent(life_slot, life_truth)
        if not ok:
            bad.append(f"{name} 生命 真值 {life[0]}/{life[1]} = {life_truth:.6f}，读数 {how}")
        skill_truth = skill[0] / skill[1]
        fill, meaning = field.read_skill(frame, window)
        ok, how = _consistent(fill, skill_truth)
        if not ok:
            bad.append(
                f"{name} 技力 真值 {skill[0]}/{skill[1]} = {skill_truth:.6f}，"
                f"填充读数 {how}（含义 {meaning!r}）"
            )
    assert not bad, "新采那 3 帧与真值对不上：" + " ｜ ".join(bad)


def _f8_template_not_from_batch() -> None:
    """图标模板**不许取自验收批的帧**。"""
    with np.load(DIAMOND_ASSET, allow_pickle=False) as asset:
        assert "source" in asset, f"模板资产里没有 source：{list(asset)}"
        raw = asset["source"]
        source = str(raw.item()) if raw.shape == () else str(raw.tolist())
    sources = {crop["source"] for crop in json.loads((BATCH / "index.json").read_text("utf-8"))["crops"]}
    assert source not in sources, f"模板取自验收批里的帧：{source}"


def _f9_window_out_of_canvas() -> None:
    """取区越出画布要**响亮地拒绝**，不许静默裁到边上。"""
    frame = next(iter(_frames().values()))
    width = frame.canvas.width
    outside = _rect(float(width) - 10.0, 0.0, *WINDOW_SIZE)
    try:
        _field().read_health(frame, outside)
    except Exception:  # noqa: BLE001 —— 报什么错不管，只要不是"照样读成功"
        return
    raise AssertionError("取区越出画布却照样读成功了 —— 那会把三样元素挪到别处再报一个数")


def _f16_diamond_verdicts_match_truth() -> None:
    """菱形的**判决**与规划者逐帧读出来的真值相符，而且**两条线坐在量出来的空档里**。

    **编号从 F16 起**：F14 / F15 已经随坏样本 `shot33_a` 一起退掉了（见 `field_verdict.md`
    §删掉了什么），**那两个号不许再用** —— 复用一个退掉的号，看历史记录的人会以为
    这是同一条检查。

    两个方向分开数：**误报**（画面里没有它却报「在」—— 假事实，最重）与**漏检**
    （画面里有它却报「不在」）。「认不出」**不算错**（它是安全方向），但也不许变成
    一条"什么都不判"的链 —— 所以另有一条：**分数落在空档两侧、判决却是「认不出」**，
    那就是把明明白白的读数丢掉了，要红。

    余量那一条**只写规则、不写死数**：两条线必须**严格坐在空档内部**
    （`max(不在) < ABSENT < SOLID < min(在)`）—— 线可以调，坐到空档外面不行。
    **那两条线现在没有推导出处**（这一批只量出空档），这件事记在
    `docs/10-规格/视觉层/场上我方干员.md` §3，不在这里重述。
    """
    field = _field()
    template = field.load_ready_diamond()
    frames = _frames()
    scores: dict[str, float] = {}
    unsampled: list[str] = []
    missed: list[str] = []
    false_positives: list[str] = []
    for name, want in DIAMOND_PRESENT.items():
        window = _window_of(name)
        score, _, _ = field.diamond_score(frames[name], window, template)
        scores[name] = score
        slot = field.find_ready_diamond(frames[name], window, template)
        payload = getattr(slot, "payload", None)
        if type(payload).__name__ != "Exact":
            unsampled.append(name)
            continue
        if bool(getattr(payload, "value", False)) == want:
            continue
        (missed if want else false_positives).append(name)

    present = [s for n, s in scores.items() if DIAMOND_PRESENT[n]]
    absent = [s for n, s in scores.items() if not DIAMOND_PRESENT[n]]
    assert len(present) == 8 and len(absent) == 8, (len(present), len(absent))
    assert not false_positives, (
        f"误报（画面里没有那个黄菱形，却报「在」）：{false_positives} —— 这是假事实那个方向"
    )
    assert not missed, f"漏检（画面里有那个黄菱形，却报「不在」）：{missed}"
    lost = [
        f"{name}（分 {scores[name]:.6f}）"
        for name in unsampled
        if scores[name] <= max(absent) or scores[name] >= min(present)
    ]
    assert not lost, f"分数在空档之外、判决却落「认不出」—— 明明白白的读数被丢掉了：{lost}"
    assert max(absent) < field.DIAMOND_ABSENT_SCORE < field.DIAMOND_SOLID_SCORE < min(present), (
        f"两条线没坐在空档里：不在最高 {max(absent):.6f} · 在最低 {min(present):.6f} · "
        f"线是 {field.DIAMOND_ABSENT_SCORE} / {field.DIAMOND_SOLID_SCORE}"
    )


CHECKS: tuple[Check, ...] = (
    Check("F1", "血条只出字段表允许的形态，且九档填充看得见", "状态层/我方干员 §字段表（血量那一行）", _f1_health_forms),
    Check("F2", "技力字段只出精确/区间/未采样，不许出下界", "状态层/我方干员 §字段表（技力那一行）", _f2_skill_forms),
    Check("F3", "没有条的画面落未采样，不是满", "状态层/我方干员 §血量：看不到≠满血", _f3_blank_is_unsampled),
    Check("F4", "已知填充的条读出来等于画的", "视觉层/场上我方干员 §2 读亮区占总长的比例", _f4_known_fill),
    Check("F5", "菱形是搜出来的，不是钉在小框里", "视觉层/场上我方干员 §3", _f5_diamond_is_searched),
    Check("F6", "线上只读条，不接读数字那一套", "用户裁定：文字识别比读条贵，不做线上通道", _f6_no_digit_channel),
    Check("F7", "判据读数带与独立真值的逐条对比", "任务单 A4【做】2/3 与【验收】", _f7_readings_carry_truth),
    Check("F8", "图标模板不取自验收批", "视觉层/识别方法 §已定：不许拿验收帧当素材", _f8_template_not_from_batch),
    Check("F9", "取区越出画布要拒绝", "采集与验收/基准集 §2（同源：不许替调用方编值）", _f9_window_out_of_canvas),
    Check("F10", "血条读数与独立真值相容（13 帧）", "field_ops_truth.md 生命表 · 视觉层/场上我方干员 §2", _f10_health_matches_truth),
    Check("F11", "技力填充与独立真值相容", "field_ops_truth.md 的 `16/26` · 状态层/我方干员 §技力", _f11_skill_matches_truth),
    Check("F12", "执行者没见过的 3 帧也与真值相容", "field_ops_truth.md §第二批真值帧（sp_series）", _f12_new_frames_match_truth),
    Check("F13", "自己画的技力条（含亮轨道）也读得出亮区比例", "视觉层/场上我方干员 §2 · §1（绿条 = 充能中）", _f13_skill_fill_known),
    Check("F16", "菱形的判决与规划者读的真值相符（16 帧），两条线坐在空档里", "视觉层/场上我方干员 §3 · 真值主本是本文件的 DIAMOND_PRESENT", _f16_diamond_verdicts_match_truth),
)
