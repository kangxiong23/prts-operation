"""顶部两个数的**第二张网** —— 专挑"那 35 帧上验不出来"的面。

它**故意不带 pytest 外壳**（文件名不匹配 `test_*.py`），所以 `pytest` 不收集它。
跑法：

    python tests/spec/topbar_extra_spec.py

## 为什么要有这一张

`topbar_spec.py` 那 12 条守着"规格说的那些话成立"。但它的骨架是**一批帧**，而
一批帧有它天生的盲区（[D-全局-22](../../docs/20-决策记录/全局.md) 就是为这个写的）：

| 那 35 帧的共同点 | 于是验不出来的面 |
| --- | --- |
| **每一张的原点、尺寸都一模一样**（`(1195,25)` / `455×55`） | 读数链可以**绕过 `ui.rects`**、把窗口坐标写死，照样全绿 |
| 生命只出现过 `1` 与 `3`，**都是单字形** | 两位数的生命（窗口够不够宽、缺位判据会不会误伤） |
| 敌数两侧**位数最多差 1** | 3 位数、以及"一侧 1 位一侧 3 位"（分隔符离串中心很远） |
| 窗口外面就是黑（裁剪贴回画布，四周是空的） | 窗口**外面真有亮东西**时会不会被读进来 |

## 合成帧怎么造

**用真墨**：字形从既有帧的敌数窗口里整块搬过来（原样像素，一个都不缩），按
**实测的字形宽度 + 间距**排到该在的位置，底部对齐到字形的基线 y=67。所以这不是
"画一个像数字的东西"，是**把真字形摆到那个位置上**。

**没有**覆盖到的面（写下来而不是假装覆盖了）：敌数一侧**少了一整位**（例如
`14/22` 里的 `4` 被挡掉）时，整串会变窄，而"该多宽"没有独立证据 —— 这一条
规格还没定，见《视觉层/顶部两个数》§尚未定。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import topbar_spec as base  # noqa: E402

CANVAS_W, CANVAS_H = base.CANVAS_W, base.CANVAS_H

# ---------------------------------------------------------------------------
# 真字形：都在敌数窗口里量到过。框给得比紧外框宽一圈，取墨时按阈值收回来。
# ---------------------------------------------------------------------------

GLYPH_SOURCES: dict[str, tuple[str, int, int, int, int]] = {
    "0": ("shot01", 1256, 34, 1285, 72),
    "1": ("shot26", 1240, 34, 1254, 72),
    "2": ("shot26", 1300, 34, 1322, 72),
    "3": ("shot05", 1259, 34, 1282, 72),
    "4": ("shot26", 1255, 34, 1281, 72),
    "8": ("shot01", 1304, 34, 1329, 72),
    "/": ("shot26", 1280, 34, 1300, 72),
}

BASELINE_Y = 67
"""字形的基线（35 帧实测：字形占 y 38..67）。合成时按它底部对齐。"""


def _glyph(ch: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """搬一个真字形：返回 `(整块像素, 墨的紧外框)`。

    **整块原样搬，不按阈值裁墨。** 按阈值裁会把抗锯齿的那一圈边切掉，字就瘦了
    —— 实测那样搬过来的 `0` 对字形表只得到 0.914（原帧上同一个字形是 0.947），
    离 `SOLID_MATCH_SCORE = 0.90` 只剩 0.014。摆位置时按**墨的紧外框**对齐，
    所以多带一圈背景不影响它落在哪。
    """
    frame_name, x0, y0, x1, y1 = GLYPH_SOURCES[ch]
    block = base._canvas(frame_name)[y0 : y1 + 1, x0 : x1 + 1].copy()
    ys, xs = np.nonzero(block.min(axis=2) >= 128)
    assert len(xs) > 0, f"字形 {ch!r} 的框 {GLYPH_SOURCES[ch]} 里没有墨"
    return block, (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _place(ch: str, left: int, bottom: int) -> tuple:
    """把某个字形的**墨**摆到 `左缘 = left`、`底缘 = bottom`。"""
    block, (ix0, _iy0, _ix1, iy1) = _glyph(ch)
    px, py = left - ix0, bottom - iy1
    return base._patch(px, py, px + block.shape[1] - 1, py + block.shape[0] - 1, block)


def _compose(text: str, center_x: float, gap: int) -> list[tuple]:
    """把一串字形按实测宽度与间距排好，**整串居中在 center_x**。"""
    glyphs = [(ch,) + _glyph(ch) for ch in text]
    widths = [ink[2] - ink[0] + 1 for _ch, _block, ink in glyphs]
    total = sum(widths) + gap * (len(widths) - 1)
    x = int(round(center_x - total / 2.0))
    patches = []
    for (ch, block, (ix0, _iy0, ix1, iy1)) in glyphs:
        px, py = x - ix0, BASELINE_Y - iy1
        patches.append(base._patch(px, py, px + block.shape[1] - 1, py + block.shape[0] - 1, block))
        x += (ix1 - ix0 + 1) + gap
    return patches


def _shifted_ui(dx: int):
    """两个窗口整体右移 dx，别的窗口不动。"""
    contracts = base._mod("ark.contracts")
    ui = base._ui()
    rects = {}
    for name, rect in ui.rects.items():
        if name in ("enemy_count", "life_digits"):
            rect = contracts.PixelRect(x=rect.x + dx, y=rect.y, width=rect.width, height=rect.height)
        rects[name] = rect
    return contracts.UiLayout(rects=rects)


def _canvas_shifted(name: str, dx: int) -> np.ndarray:
    """把这一帧的裁剪整体右移 dx 贴回画布。"""
    e = base._entry(name)
    x, y = e["origin"]
    crop = np.asarray(Image.open(base.FIXTURES / e["file"]).convert("RGB"))
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    canvas[y : y + crop.shape[0], x + dx : x + dx + crop.shape[1]] = crop
    return canvas


def _read_both(frame, ui=None):
    ui = ui if ui is not None else base._ui()
    table = base._glyphs()
    topbar = base._topbar()
    return (
        topbar.read_life_digits(frame, table, ui=ui),
        topbar.read_enemy_count(frame, table, ui=ui),
    )


# ---------------------------------------------------------------------------
# 检查
# ---------------------------------------------------------------------------


def check_readers_go_through_the_layout_parameter() -> str:
    """把两个窗口整体右移 200 px，读数要**跟着窗口走**。

    出处：《视觉层/顶部两个数》§两个窗口 + 派工单"ROI 一律从 `ui.rects` 取"。
    那一批 35 帧的原点、尺寸**完全一样**，所以一条把窗口坐标写死的实现也能全绿；
    这条把内容挪到**文档窗口之外**，写死的那条就什么都读不到。
    """
    dx = 200
    ui = _shifted_ui(dx)
    frame = base._frame(base._raw_from_canvas(_canvas_shifted("shot26", dx)))
    life, pair = _read_both(frame, ui)
    got = (base._exact_of(life), base._exact_of(pair[0]), base._exact_of(pair[1]))
    want = (1, 14, 22)
    assert got == want, (
        f"窗口右移 {dx} px 之后该读到 {want}，读到 {got}："
        "读数链没有按传进来的 `ui` 取 ROI（那一批帧的原点都一样，所以写死也能全绿）"
    )
    return f"窗口右移 {dx} px 之后仍读到 1 / 14 / 22"


def check_two_digit_life_is_read_as_two_digits() -> str:
    """两位数的生命：把真实的 `1` 与 `4` 居中摆到生命锚点上，要读成 `14`。

    出处：《视觉层/顶部两个数》§每个区域必须知道的事实（生命 1–2 位）+ §尚未定
    （真帧里一张两位数的生命都没有，所以窗口宽度与缺位判据是**推导**的）。
    这条把推导的那一面验一遍：**不许读成 `1`、也不许读成区间**。
    """
    patches = [base._blank(*base._window("life_digits"))] + _compose("14", base.LIFE_ANCHOR, 4)
    life, _ = _read_both(base._patched("shot26", *patches))
    got = base._exact_of(life)
    assert got == 14, f"该读成 14，读成 {base._slot_value(life)}"
    return "两位数生命 `14` 读成精确 14"


def check_three_digit_enemy_count_is_read() -> str:
    """敌数两侧各 3 位：`100/123`。

    出处：《视觉层/顶部两个数》§每个区域必须知道的事实（窗口按 3 位取）+ §尚未定。
    这条同时验窗口宽度够不够、以及整串居中之后还在不在窗口里。
    """
    patches = [base._blank(*base._window("enemy_count"))] + _compose("100/123", base.ENEMY_ANCHOR, 3)
    _, pair = _read_both(base._patched("shot26", *patches))
    got = (base._exact_of(pair[0]), base._exact_of(pair[1]))
    assert got == (100, 123), (
        f"该读成 100/123，读成 {base._slot_value(pair[0])}/{base._slot_value(pair[1])}"
    )
    return "3 位数的敌数 `100/123` 读成精确 (100, 123)"


def check_one_digit_against_three_digits() -> str:
    """一侧 1 位、一侧 3 位：`1/123` —— 这条**把"分隔符就是中间那个框"的取法否掉**。

    出处：《视觉层/顶部两个数》§每个区域必须知道的事实（分隔符要有它自己的判据）。
    整串居中在 1292.5 时，那个 `/` 的中心落在 **1270.5**，而串中心附近的框其实是
    左边那个 `1`（1296.5）。所以靠"哪个框离串中心近"认分隔符的实现，在这条上会
    把 `/` 当成数字、把 `1` 当成分隔符 —— 读出来的东西一定不是 `1/123`。
    """
    patches = [base._blank(*base._window("enemy_count"))] + _compose("1/123", base.ENEMY_ANCHOR, 3)
    _, pair = _read_both(base._patched("shot26", *patches))
    got = (base._exact_of(pair[0]), base._exact_of(pair[1]))
    assert got == (1, 123), (
        f"该读成 1/123，读成 {base._slot_value(pair[0])}/{base._slot_value(pair[1])}"
    )
    return "一侧 1 位一侧 3 位 `1/123` 读成精确 (1, 123)"


def check_bright_ink_outside_the_windows_is_ignored() -> str:
    """窗口**外面**的亮东西不许被读进来。

    出处：《视觉层/顶部两个数》§两个窗口（窗口正是这一层唯一承诺的事）。
    那一批帧贴回画布之后四周是黑的，所以"从整屏找数字"那种写法在那一批上也全绿。
    """
    e = base._entry("shot26")
    crop_x0 = e["origin"][0]
    left_of_life = base.LIFE_UNION[0] - 62
    right_of_enemy = 1390
    assert left_of_life < base._window("life_digits")[0], "那块要在生命窗口左边"
    assert right_of_enemy > base._window("enemy_count")[2], "那块要在敌数窗口右边"
    assert left_of_life > crop_x0 and right_of_enemy < crop_x0 + e["size"][0]
    frame = base._patched(
        "shot26",
        _place("8", left_of_life, BASELINE_Y),
        _place("8", right_of_enemy, BASELINE_Y),
    )
    life, pair = _read_both(frame)
    got = (base._exact_of(life), base._exact_of(pair[0]), base._exact_of(pair[1]))
    assert got == (1, 14, 22), (
        f"窗口外面那两个 `8` 该被无视，读数该还是 1 / 14 / 22，读到 {got}"
    )
    return "窗口外各贴一个 `8`，两个读数一字不变"


def check_blank_enemy_window_is_unsampled() -> str:
    """敌数窗口整块涂黑 → **两个都未采样**（不是未知）。

    出处：《视觉层/顶部两个数》§降级第一条。与生命那条互为对照：两边都要能分清
    "那里没有东西"与"有东西我不知道是什么"。
    """
    _, pair = _read_both(base._patched("shot26", base._blank(*base._window("enemy_count"))))
    assert all(base._is_unsampled(v) for v in pair), (
        f"涂黑该两个都报未采样，读成 {base._slot_value(pair[0])}/{base._slot_value(pair[1])}"
    )
    return "敌数窗口整块涂黑 → 两个都未采样"


def check_reads_are_pure_and_repeatable() -> str:
    """读数**不许改帧**，而且同一帧读两遍给一模一样的结果。

    出处：《视觉层/README》三条不能越的线之一（输出必须带三态）+ 离线重放的
    "同一批帧要能反复跑"。帧是只读的输入，读数链在它上面就地涂改会让下一帧、
    下一条链看到被改过的像素 —— 那是一种没有信号的假事实。
    """
    frame = base._patched("shot26")
    before = frame.pixels.tobytes()
    first = _read_both(frame)
    after = frame.pixels.tobytes()
    assert before == after, "读数链改动了帧的像素（帧是只读输入）"
    second = _read_both(frame)
    got = [
        (base._slot_value(a), base._slot_value(b[0]), base._slot_value(b[1]))
        for a, b in (first, second)
    ]
    assert got[0] == got[1], f"同一帧读两遍结果不一样：{got[0]} vs {got[1]}"
    return f"读两遍都是 {got[0]}，像素一个字节没动"


def check_life_digit_off_centre_to_the_right_is_unknown() -> str:
    """生命字形偏**右**（缺的在左边）→ **未知**，不许塌成确切值。

    出处：《视觉层/顶部两个数》§降级（缺一位必须判出来 —— 不判就会报一个更短的
    确切数）。`topbar_spec` 那条测的是偏左，这条测另一边：两条合起来说明那条判据
    是**两边都管**的。落法是未知（这三个字段不收区间）。
    """
    _block, (ix0, _iy0, ix1, _iy1) = _glyph("3")
    right = base._window("life_digits")[2]
    left = right - (ix1 - ix0 + 1) + 1
    frame = base._patched(
        "shot26",
        base._blank(*base._window("life_digits")),
        _place("3", left, BASELINE_Y),
    )
    life, _ = _read_both(frame)
    contracts = base._mod("ark.contracts")
    assert isinstance(life, contracts.Valued), f"该是有值，读成 {base._slot_value(life)}"
    assert isinstance(life.payload, contracts.Unknown), (
        f"字形偏右（缺的在左边）该落未知，**不许报成确切值 3**，"
        f"读成 {base._slot_value(life)}"
    )
    return "字形偏右 → 未知，没有塌成确切值 3"


def check_merged_pair_does_not_become_a_false_reading() -> str:
    """两位被桥接成一块 → 那一侧落**未知**，不许报出一个排除真值的读数。

    出处：《视觉层/顶部两个数》§降级。**这条不是纸上的顾虑**：数字与数字之间那条
    缝只有 **1–5 px** 宽，缝里的 `min(R,G,B)` 峰值实测到 **127**，而敌数的墨线是
    **128** —— **差一格**。所以"两位粘成一块"随时会发生。

    `shot26` 的甲是 `14`（两块，1242..1279）。把这两块涂成**一整块**（39 px 宽，
    比最宽的单个字形 26 px 还宽 13 px），于是甲只剩一块、而它认不出。真值是 14。
    落法必须是**未知** —— 报了 `[0,9]` 就是把真值排除在外（假事实）。
    """
    solid = np.full((base.LIFE_UNION[3] - base.LIFE_UNION[1] + 1, 39, 3), 255, np.uint8)
    frame = base._patched("shot26", base._patch(1242, base.LIFE_UNION[1], 1280, base.LIFE_UNION[3], solid))
    _, pair = _read_both(frame)
    contracts = base._mod("ark.contracts")
    assert base._exact_of(pair[1]) == 22, (
        f"乙那一侧没被动过，该仍是 22，读成 {base._slot_value(pair[1])}"
    )
    killed = pair[0]
    assert isinstance(killed, contracts.Valued), f"该是有值，读成 {base._slot_value(killed)}"
    assert isinstance(killed.payload, contracts.Unknown), (
        f"甲只剩一块、而且认不出，该落未知（位数没定下来），"
        f"读成 {base._slot_value(killed)} —— 报了确切值或区间就是把真值 14 排除在外"
    )
    return f"两块粘成一块 → 甲 {base._slot_value(killed)}，乙 精确(22)"


CHECKS: tuple[tuple[str, str, str, object], ...] = (
    ("X1", "《视觉层/顶部两个数》§两个窗口", "读数按传进来的 `ui` 取 ROI", check_readers_go_through_the_layout_parameter),
    ("X2", "《视觉层/顶部两个数》§尚未定", "两位数的生命读成精确值", check_two_digit_life_is_read_as_two_digits),
    ("X3", "《视觉层/顶部两个数》§尚未定", "两侧各 3 位的敌数读得出来", check_three_digit_enemy_count_is_read),
    ("X4", "《视觉层/顶部两个数》§每个区域必须知道的事实", "分隔符要有它自己的判据（一侧 1 位一侧 3 位）", check_one_digit_against_three_digits),
    ("X5", "《视觉层/顶部两个数》§两个窗口", "窗口外的亮东西不许读进来", check_bright_ink_outside_the_windows_is_ignored),
    ("X6", "《视觉层/顶部两个数》§降级", "敌数整块涂黑 → 两个都未采样", check_blank_enemy_window_is_unsampled),
    ("X7", "《视觉层/README》三条不能越的线", "读数不改帧、且可重复", check_reads_are_pure_and_repeatable),
    ("X8", "《视觉层/顶部两个数》§降级", "字形偏右 → 未知，不塌成确切值", check_life_digit_off_centre_to_the_right_is_unknown),
    ("X9", "《视觉层/顶部两个数》§降级", "两位粘成一块 → 未知，不报出排除真值的读数", check_merged_pair_does_not_become_a_false_reading),
)


def main() -> int:
    failures = 0
    for ident, spec, claim, probe in CHECKS:
        try:
            evidence = probe()
        except AssertionError as bad:
            failures += 1
            print(f"红 {ident}  {claim}\n     出处 {spec}\n     证据 {bad}")
        except BaseException as broken:  # noqa: BLE001
            failures += 1
            print(f"炸 {ident}  {claim}\n     {type(broken).__name__}: {broken}")
        else:
            print(f"绿 {ident}  {evidence}")
    print()
    print(f"{len(CHECKS) - failures}/{len(CHECKS)} 条补充检查通过")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
