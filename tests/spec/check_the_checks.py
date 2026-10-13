"""验证 `tests/spec/` 里的检查**不是永远绿的** —— 规划者的自检。

规划者写的验收套件有一个特有的风险：它描述的往往是**已经写好的**代码，所以
"绿"可能只说明检查太松，而不是代码太对。这个探针逐条制造反例（把公开面换成
坏的那一版），确认每条检查都会炸。

**它不属于 `pytest tests`**（文件名不匹配 `test_*.py`），跑法：

    python tests/spec/check_the_checks.py

现在覆盖三批：`contracts_spec.py`（契约层）· `topbar_spec.py`（顶部两个数）·
`field_spec.py`（场上我方干员的菱形那个布尔）。
**不是每条检查都有探针** —— 探针要有"把公开面换成坏的那一版"这个动作，有些检查
（例如"35 帧都读对"）的反例要伪造一整批帧，代价大于收益。**这一批挑的是"断言的是
形状、而不是某个值"的那些**：形状类断言最容易写成永远绿。

注意一个坑：`contracts_spec` 在**模块加载时**就把 `Interval` / `LowerBound` 这类
名字绑死了（`from ark.contracts import ...`）。所以打补丁要**同时**打到
`ark.contracts` 和 `contracts_spec` 两个模块上，只打前者会让检查"永远绿"——
那是探针的假象，不是检查的漏洞。
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "src"))
sys.path.insert(0, str(_HERE))

import ark.contracts as contracts  # noqa: E402
import ark.engine.vision.field as field_mod  # noqa: E402
import ark.engine.vision.topbar as topbar_mod  # noqa: E402
import contracts_spec as spec  # noqa: E402
import field_spec  # noqa: E402
import topbar_spec  # noqa: E402

_MODULES = (contracts, spec)


def _goes_red(name: str, check: Callable[[], str], patch: dict[str, object]) -> bool:
    """把 patch 盖到两个模块上，跑 check —— 期望它抛 AssertionError。"""
    saved = [
        (module, key, getattr(module, key))
        for module in _MODULES
        for key in patch
        if hasattr(module, key)
    ]
    for module in _MODULES:
        for key, value in patch.items():
            if hasattr(module, key):
                setattr(module, key, value)
    try:
        check()
    except AssertionError as bad:
        print(f"  ✓ {name} 会红：{str(bad)[:76]}")
        return True
    except BaseException as other:  # noqa: BLE001
        print(f"  ‼ {name} 炸错了类型：{type(other).__name__}: {other}")
        return False
    else:
        print(f"  ✗ {name} **永远绿** —— 换成坏实现它照样过")
        return False
    finally:
        for module, key, value in saved:
            setattr(module, key, value)


def _goes_red_in(
    module: object,
    name: str,
    check: Callable[[], str],
    patch: Mapping[str, object],
    *,
    must_mention: str | None = None,
) -> bool:
    """把 `patch` 盖到**某一个模块**上，跑 check —— 期望它抛 AssertionError。

    `must_mention` 用来钉住**是哪一条断言**炸的：一条检查里往往有几层断言，
    只验"会红"分不清是哪一层在起作用（外层先炸，里层就永远没人试过）。
    """
    saved = [(key, getattr(module, key)) for key in patch if hasattr(module, key)]
    for key, value in patch.items():
        if hasattr(module, key):
            setattr(module, key, value)
    try:
        check()
    except AssertionError as bad:
        text = str(bad)
        if must_mention is not None and must_mention not in text:
            print(f"  ‼ {name} 红了，但不是那一条：{text[:64]}")
            return False
        print(f"  ✓ {name} 会红：{text[:76]}")
        return True
    except BaseException as other:  # noqa: BLE001
        print(f"  ‼ {name} 炸错了类型：{type(other).__name__}: {other}")
        return False
    else:
        print(f"  ✗ {name} **永远绿** —— 换成坏实现它照样过")
        return False
    finally:
        for key, value in saved:
            setattr(module, key, value)


# --- G9：require() 永不拒绝 -------------------------------------------------
class _NeverRefuses:
    def __init__(self, view: object, *_: object) -> None:
        self.view = view

    def require(self, view: object) -> _NeverRefuses:
        return self


# --- G10：画布被归一化到 0–1 ------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _NormalizedCanvas:
    width: float
    height: float

    def __init__(self, width: float, height: float) -> None:
        object.__setattr__(self, "width", 1.0)
        object.__setattr__(self, "height", height / width)


# --- G11：把三种结果压成二值（只会说"可能 / 不可能"） ------------------------
class _TwoValued:
    def __init__(self, lo: float, *_: object) -> None:
        self.lo = lo

    def contains(self, other: object) -> bool:
        return self.intersects(other)

    def intersects(self, other: object) -> bool:
        hi = getattr(other, "hi", other)
        return self.lo <= hi


@dataclasses.dataclass(frozen=True)
class _BadInterval:
    lo: float
    hi: float


# --- G12：矩形收非正宽高 ----------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _NoPositivity:
    x: float
    y: float
    width: float
    height: float

    def __init__(self, x: float, y: float, width: float, height: float) -> None:
        for name, value in (("x", x), ("y", y), ("width", width), ("height", height)):
            object.__setattr__(self, name, value)


# --- U1：值收裸元组 ---------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _LooseLayout:
    rects: Mapping[str, object]

    def __init__(self, rects: Mapping[str, object]) -> None:
        object.__setattr__(self, "rects", dict(rects))


# --- U2：表可写 -------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _WritableLayout:
    rects: dict[str, object]

    def __init__(self, rects: Mapping[str, object]) -> None:
        object.__setattr__(self, "rects", dict(rects))


# --- U3：矩形被归一化 -------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _NormalizedRect:
    x: float
    y: float
    width: float
    height: float


@dataclasses.dataclass(frozen=True)
class _NormalizingLayout:
    rects: Mapping[str, object]

    def __init__(self, rects: Mapping[str, object]) -> None:
        scaled = {}
        for name, rect in rects.items():
            x, y, width, height = dataclasses.astuple(rect)
            scaled[name] = _NormalizedRect(
                x / 1920, y / 864, width / 1920, height / 864
            )
        object.__setattr__(self, "rects", MappingProxyType(scaled))


CASES: tuple[tuple[str, Callable[[], str], dict[str, object]], ...] = (
    ("G9  require() 放行错的视图", spec.check_projection_require_refuses_the_wrong_view, {"Projection": _NeverRefuses}),
    ("G10 画布被缩成归一化坐标", spec.check_canvas_is_the_capture_resolution, {"Canvas": _NormalizedCanvas}),
    ("G11 三值被压成二值", spec.check_interval_meets_a_threshold_in_three_ways, {"LowerBound": _TwoValued, "Interval": _BadInterval}),
    ("G12 矩形收非正宽高", spec.check_pixel_rect_is_corner_plus_size, {"PixelRect": _NoPositivity}),
    ("U1  UI 常量收裸元组", spec.check_ui_layout_is_a_table_of_named_rects, {"UiLayout": _LooseLayout}),
    ("U2  UI 常量表可写", spec.check_ui_layout_is_read_only, {"UiLayout": _WritableLayout}),
    ("U3  UI 常量被归一化", spec.check_ui_layout_values_are_absolute_pixels, {"UiLayout": _NormalizingLayout}),
)


# ---------------------------------------------------------------------------
# 顶部两个数那批（T15 / T16）：把两个读数入口换成坏的那一版
#
# 这两个入口在检查里是**每次调用现取**（`_topbar().read_life_digits(...)`），
# 所以盖在模块属性上就能生效，不用去改帧。
# ---------------------------------------------------------------------------


def _fake_interval_life(*_args: object, **_kwargs: object) -> object:
    """生命读数：一律报区间 —— T7 裁定说这三个字段不许有这个形态。"""
    return contracts.Valued(contracts.Interval(0, 9), 0.9)


def _fake_interval_enemy(*_args: object, **_kwargs: object) -> tuple:
    return contracts.Valued(contracts.Interval(0, 9), 0.9), contracts.Valued(
        contracts.Interval(0, 9), 0.9
    )


def _fake_unsampled_life(*_args: object, **_kwargs: object) -> object:
    """生命读数：一律报未采样 —— 有笔画的那几路不许落这个。"""
    return contracts.Unsampled()


def _fake_truncated_life(*_args: object, **_kwargs: object) -> object:
    """生命读数：一律报 `11` —— 三块超上限时"截断成两位"就是这么来的。"""
    return contracts.Valued(contracts.Exact(11), 0.9)


TOP_CASES: tuple[tuple[str, Callable[[], str], dict[str, object], str | None], ...] = (
    (
        "T15 退化族里冒出区间",
        topbar_spec.check_no_degradation_produces_an_interval,
        {"read_life_digits": _fake_interval_life, "read_enemy_count": _fake_interval_enemy},
        "没有区间",
    ),
    (
        "T15 有笔画的那几路落了未采样",
        topbar_spec.check_no_degradation_produces_an_interval,
        # **只盖生命那一个入口**：敌数那个换成区间假货的话，先炸的是"不产区间"
        # 那一条，这一层断言就永远没人试过（探针要一次只弄坏一处）。
        {"read_life_digits": _fake_unsampled_life},
        "混掉就是假事实",
    ),
    (
        "T16 超上限被截成一个确切数",
        topbar_spec.check_too_many_blocks_is_not_an_exact_number,
        {"read_life_digits": _fake_truncated_life},
        "静默改写成",
    ),
)


# ---------------------------------------------------------------------------
# 场上我方干员那批（F14）：菱形那个布尔
#
# F14 的断言是分层的：簇大小 → 两个方向 → "判决与线脱钩" → 两条线坐在空档里。
# 所以每个探针要**指定是哪一层炸的**（`must_mention`），否则外层先炸，
# 里层那几条断言就永远没人试过。
# ---------------------------------------------------------------------------


def _always_unsampled(*_args: object, **_kwargs: object) -> object:
    """菱形判决：一律「认不出」—— 分数与线都还是好的，只是什么都不判。"""
    return contracts.Unsampled()


def _diamond_absent_ceiling() -> float:
    """这一批「不在」那 8 条里**最高**的分 —— 量出来，不抄一个数。

    抬到它上面（哪怕只多一点点）「不在」那一簇就开始落「认不出」，所以这一个值
    是"这条线还有没有余量"的边界。
    """
    template = field_mod.load_ready_diamond()
    frames = field_spec._frames()
    return max(
        field_mod.diamond_score(frames[name], field_spec._window_of(name), template)[0]
        for name, present in field_spec.DIAMOND_PRESENT.items()
        if not present
    )


_DIAMOND_CEILING = _diamond_absent_ceiling()
"""模块加载时量一次（16 次模板匹配，代价可忽略）。"""

FIELD_CASES: tuple[
    tuple[str, object, Callable[[], str], dict[str, object], str | None], ...
] = (
    (
        "F16 真值两处对翻（一处在→不在、一处不在→在）",
        # `DIAMOND_PRESENT` 住在**规划者那份表**上，所以要盖 `field_spec`。
        field_spec,
        # 簇大小还是 8/8，所以只有**方向断言**能抓 —— 这正是要试的那一层。
        field_spec._f16_diamond_verdicts_match_truth,
        {
            "DIAMOND_PRESENT": {
                **field_spec.DIAMOND_PRESENT,
                "shot02_a": False,
                "cd01_a": True,
            }
        },
        "误报",
    ),
    (
        "F16 判决永远「认不出」（线与分数都还好）",
        field_mod,
        field_spec._f16_diamond_verdicts_match_truth,
        {"find_ready_diamond": _always_unsampled},
        "明明白白的读数被丢掉了",
    ),
    (
        "F16 「不在」那条线抬到空档上缘",
        field_mod,
        field_spec._f16_diamond_verdicts_match_truth,
        {"DIAMOND_ABSENT_SCORE": _DIAMOND_CEILING},
        "两条线没坐在空档里",
    ),
)


def _always_present(*_args: object, **_kwargs: object) -> object:
    """判决：一律「在」—— 拿去盖**黄**那条链，试「灰的帧上黄那条报在」那一条断言。"""
    return contracts.Valued(contracts.Exact(True), 0.99)


def _gray_absent_ceiling() -> float:
    """灰那条链「不在」那 16 条里**最高**的分 —— 量出来，不抄一个数。

    注意这 16 条**跨两批**（灰那批 2 条 + 场上干员那批 14 条），所以两批都要量 ——
    只量灰那批会得到一个小得多的数，那样一盖上去先炸的是"丢掉读数"那一条，
    "坐在空档里"那条就永远没人试过。
    """
    template = field_mod.load_close_diamond()
    gray, yellow = field_spec._gray_frames(), field_spec._frames()
    scores = []
    for name, present in field_spec.CLOSE_DIAMOND_PRESENT.items():
        if present:
            continue
        if name in gray:
            frame, window = gray[name], field_spec._gray_window_of(name)
        else:
            frame, window = yellow[name], field_spec._window_of(name)
        scores.append(field_mod.diamond_score(frame, window, template)[0])
    return max(scores)


_GRAY_CEILING = _gray_absent_ceiling()
"""模块加载时量一次（10 次模板匹配，代价可忽略）。"""

GRAY_CASES: tuple[
    tuple[str, object, Callable[[], str], dict[str, object], str | None], ...
] = (
    (
        "F17 灰的真值两处对翻（一处在→不在、一处不在→在）",
        field_spec,
        field_spec._f17_close_diamond_matches_truth_and_does_not_mix,
        {
            "CLOSE_DIAMOND_PRESENT": {
                **field_spec.CLOSE_DIAMOND_PRESENT,
                "cd26": False,
                "cd03": True,
            }
        },
        "误报",
    ),
    (
        "F17 灰那条链永远「认不出」",
        field_mod,
        field_spec._f17_close_diamond_matches_truth_and_does_not_mix,
        {"find_close_diamond": _always_unsampled},
        "明明白白的读数被丢掉了",
    ),
    (
        "F17 灰的线抬到灰自己空档的上缘",
        field_mod,
        field_spec._f17_close_diamond_matches_truth_and_does_not_mix,
        {"CLOSE_DIAMOND_ABSENT_SCORE": _GRAY_CEILING},
        "灰那两条线没坐在",
    ),
    (
        "F17 黄那条链被换成「永远报在」（跨色那一格）",
        field_mod,
        field_spec._f17_close_diamond_matches_truth_and_does_not_mix,
        {"find_ready_diamond": _always_present},
        "两个布尔混了",
    ),
    (
        "F18 灰那条链抄了黄那一对线",
        field_mod,
        field_spec._f18_close_diamond_is_its_own_asset,
        {
            "CLOSE_DIAMOND_ABSENT_SCORE": field_mod.DIAMOND_ABSENT_SCORE,
            "CLOSE_DIAMOND_SOLID_SCORE": field_mod.DIAMOND_SOLID_SCORE,
        },
        "抄了黄那一对线",
    ),
)


def main() -> int:
    print("把公开面换成坏的那一版，检查是否变红：")
    results = [_goes_red(name, check, patch) for name, check, patch in CASES]
    print()
    print("顶部两个数那批（盖在读数入口上）：")
    results += [
        _goes_red_in(topbar_mod, name, check, patch, must_mention=mention)
        for name, check, patch, mention in TOP_CASES
    ]
    print()
    print("场上我方干员那批（盖在判决入口与真值表上）：")
    results += [
        _goes_red_in(module, name, check, patch, must_mention=mention)
        for name, module, check, patch, mention in FIELD_CASES
    ]
    print()
    print("灰菱形那批（盖在灰的判决入口、灰的真值表、黄那条链上）：")
    results += [
        _goes_red_in(module, name, check, patch, must_mention=mention)
        for name, module, check, patch, mention in GRAY_CASES
    ]
    print()
    print(f"{sum(results)}/{len(results)} 条检查被证明会红")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
