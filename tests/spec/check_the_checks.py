"""验证 `contracts_spec.py` 里的检查**不是永远绿的** —— 规划者的自检。

规划者写的验收套件有一个特有的风险：它描述的往往是**已经写好的**代码，所以
"绿"可能只说明检查太松，而不是代码太对。这个探针逐条制造反例（把公开面换成
坏的那一版），确认每条检查都会炸。

**它不属于 `pytest tests`**（文件名不匹配 `test_*.py`），跑法：

    python tests/spec/check_the_checks.py

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
import contracts_spec as spec  # noqa: E402

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


def main() -> int:
    print("把公开面换成坏的那一版，检查是否变红：")
    results = [_goes_red(name, check, patch) for name, check, patch in CASES]
    print()
    print(f"{sum(results)}/{len(results)} 条检查被证明会红")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
