"""§1 槽：三态 × 载荷 × 置信度。

三态（未采样 / 已采样但为空 / 有值）是三支**不同的类型**，不是同一个类里的一个
枚举字段；载荷的**六种形态**同样是六个类型。于是「形态 = 精确 而值是 [15, 16]」
和「未采样 = 有值 + 未知」都写不出来。

- **载荷（形态）**：精确 / 区间 / 下界 / 上界 / 未知 / 二维区域，只有"有值"才带；
  二维区域的**形状**在 §2（geometry.Region），这里只把它收进词表；
- **置信度**：原语产生、槽携带；**只有"读"有**（算类由费用推算、记类来自部署
  记录，都没有识别过程），没有时写 None —— 那是"**不是读来的**"，不是
  "置信度为零"。

**区间 / 下界 / 上界 共用同一套操作**（包含 / 相交）：三个类型各给一对边界，
操作只写一遍（_HalfOpenInterval）。§2 的标量区间不是第四个类型，就是这套操作。

「未采样」是**字段级**的：一个原语的结果只有"有读数 / 没有读数"两种，它说不出
"未采样"。所以 Unsampled 不在载荷词表里 —— 只有字段组装层造得出它。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, TypeAlias, final

from ark.contracts.geometry import Region

__all__ = [
    "Exact",
    "Interval",
    "LowerBound",
    "UpperBound",
    "Unknown",
    "Payload",
    "PAYLOAD_TYPES",
    "Unsampled",
    "SampledEmpty",
    "Valued",
    "SlotValue",
    "SLOT_STATES",
]


def _is_number(value: object) -> bool:
    """实数（int 或 float）。bool 不算 —— 把 True 当 1 用只会掩盖错误。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_confidence(confidence: float | None) -> None:
    """置信度：原语算出的连续量，无量纲；只有"读"有。

    None 表示**这个字段不是识别来的**（算类由费用推算、记类来自我们自己的
    部署记录，都没有识别过程，也就没有置信度），不是"置信度等于零"。

    这里不校验取值范围：规格只说它是无量纲的连续量，没有给出上下界，
    擅自钉一个 [0, 1] 就是发明规范。
    """
    if confidence is None:
        return
    if not _is_number(confidence):
        raise TypeError(
            "置信度要么是数，要么是 None（该字段不是读来的），"
            f"收到 {type(confidence).__name__}"
        )


# 值类一律用 frozen dataclass，且**不打开 slots**：frozen + slots 的 dataclass
# 生成 __setattr__ 时拿的是被 slots 替换之前的那个类，于是给不存在的字段赋值
# 抛的是 "super(type, obj)" 那种 TypeError，而不是 FrozenInstanceError。
# 后者（它是 AttributeError 的子类）才是只读该有的报法。值的数量不大，
# 不值得为省一份 __dict__ 换来一个看不懂的错。


# ---------------------------------------------------------------------------
# 区间 / 下界 / 上界 共用的那套操作（§1 + §2）
# ---------------------------------------------------------------------------


class _HalfOpenInterval:
    """三段共用的操作：**包含**与**相交**只写一遍。

    下界与上界是"区间两端各自可开放"的两个具名形态，所以三者共用同一套操作
    （§1 原话）。每个类只要给出自己的一对边界：区间 (lo, hi)、下界 (lo, +∞)、
    上界 (-∞, hi)；None 表示那一端不设限。
    """

    __slots__ = ()

    def _bounds(self) -> tuple[int | float | None, int | float | None]:
        raise NotImplementedError

    def contains(self, other: int | float | _HalfOpenInterval) -> bool:
        """包含：区间 ⊇ 点 / 区间 ⊇ 区间（§2 的第一件操作）。

        用途是"哪一次读数更强 / 降级有没有发生"——降级就是区间变宽。
        对方开放着的那一端，必须我也开放，否则不算包含。
        """
        lo, hi = self._bounds()
        if isinstance(other, _HalfOpenInterval):
            other_lo, other_hi = other._bounds()
            lower_ok = lo is None or (other_lo is not None and lo <= other_lo)
            upper_ok = hi is None or (other_hi is not None and other_hi <= hi)
            return lower_ok and upper_ok
        if _is_number(other):
            lower_ok = lo is None or lo <= other
            upper_ok = hi is None or other <= hi
            return lower_ok and upper_ok
        raise TypeError(
            f"包含的另一头要么是一个点、要么是区间 / 下界 / 上界，"
            f"收到 {type(other).__name__}"
        )

    def intersects(self, other: int | float | _HalfOpenInterval) -> bool:
        """相交：两段**可能**指同一个值吗（交集非空，§2 的第二件操作）。

        端点相接算相交（下界 15 与上界 15 都含 15）——"开放"在这一层的意思是
        "不设限"，不是"不含端点"。
        """
        lo, hi = self._bounds()
        if isinstance(other, _HalfOpenInterval):
            other_lo, other_hi = other._bounds()
        elif _is_number(other):
            other_lo = other_hi = other
        else:
            raise TypeError(
                f"相交的另一头要么是一个点、要么是区间 / 下界 / 上界，"
                f"收到 {type(other).__name__}"
            )
        lower_ok = lo is None or other_hi is None or lo <= other_hi
        upper_ok = hi is None or other_lo is None or other_lo <= hi
        return lower_ok and upper_ok


# ---------------------------------------------------------------------------
# 载荷（形态）：六个类型，形态长在值自己身上
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class Exact:
    """精确：知道这个值是多少。

    **值域是 数 / 布尔 / 字符串**（§1 修正过一次）：《视觉层/原语》§1 里三条
    原语的「精确」不是数 —— exists 的**有 / 无**、classify 的**标签**、
    heading 的**四方位**。区间 / 下界 / 上界 / 二维区域另有其类，且**只收数**
    （二维区域收的是格坐标）。

    **已知的疣：bool 是 int 的子类，所以 Exact(True) == Exact(1) 成立。**
    接受它，不加字段去绕 —— 绕法（比如把布尔包一层）会让值不再等于它自己
    表达的那个东西，比这个疣更糟。真要区分"有/无"与"1"，那是字段的语义，
    不是载荷该管的事。
    """

    value: int | float | bool | str

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float, bool, str)):
            raise TypeError(
                f"Exact 的值只能是数、布尔或字符串，收到 {type(self.value).__name__}："
                "**非数也是合法的**（exists 的有/无、classify 的标签、heading 的"
                "方位），但区间 / 下界 / 上界 / 二维区域不是 —— 要表达区间请用 Interval"
            )


@final
@dataclass(frozen=True)
class Interval(_HalfOpenInterval):
    """区间：两端都封闭，lo <= hi，**只收数**。

    一端开放的情形由 LowerBound / UpperBound 承载（§1：「一端开放就是下界或
    上界」），所以这里不接受 None —— 否则同一个值就有两种写法，而"形态是
    载荷自己的类型"这条要的是唯一写法。
    """

    lo: int | float
    hi: int | float

    def __post_init__(self) -> None:
        if not _is_number(self.lo) or not _is_number(self.hi):
            raise TypeError(
                f"区间的两端都必须是数，收到 ({type(self.lo).__name__}, "
                f"{type(self.hi).__name__})：开放的一端请用 LowerBound / UpperBound"
            )
        if self.lo > self.hi:
            raise ValueError(f"区间的 lo 不许大于 hi：{self.lo} > {self.hi}")

    def _bounds(self) -> tuple[int | float | None, int | float | None]:
        return self.lo, self.hi


@final
@dataclass(frozen=True)
class LowerBound(_HalfOpenInterval):
    """下界：值不小于 lo（开口朝上），**只收数**。与区间共用同一套操作。"""

    lo: int | float

    def __post_init__(self) -> None:
        if not _is_number(self.lo):
            raise TypeError(f"下界必须是数，收到 {type(self.lo).__name__}")

    def _bounds(self) -> tuple[int | float | None, int | float | None]:
        return self.lo, None


@final
@dataclass(frozen=True)
class UpperBound(_HalfOpenInterval):
    """上界：值不大于 hi（开口朝下），**只收数**。与区间共用同一套操作。"""

    hi: int | float

    def __post_init__(self) -> None:
        if not _is_number(self.hi):
            raise TypeError(f"上界必须是数，收到 {type(self.hi).__name__}")

    def _bounds(self) -> tuple[int | float | None, int | float | None]:
        return None, self.hi


@final
@dataclass(frozen=True)
class Unknown:
    """未知：**那里确实有东西，但不知道它是什么**。

    这是"有值"的一种载荷，**不是三态的一支**。它和"已采样但为空"（那里没有
    东西）是两件事；混掉就是假事实（§1）。所以它没有值，也不该有值。
    """


Payload: TypeAlias = Exact | Interval | LowerBound | UpperBound | Unknown | Region
"""载荷词表：三态 = 有值 时，值就是这六种之一（Region 的形状在 §2）。"""

PAYLOAD_TYPES: Final[tuple[type, ...]] = (
    Exact,
    Interval,
    LowerBound,
    UpperBound,
    Unknown,
    Region,
)
"""词表的运行期形式，供 isinstance 与测试使用。"""


# ---------------------------------------------------------------------------
# 三态：三个类型
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class Unsampled:
    """未采样：还没有任何一条通道为它给出读数。

    - **只有字段组装层造得出它。** 原语说不出"未采样"——你调它，它就是采样
      过了；原语的未命中也不是缺席，而是读数的内容。所以这一支不在载荷词表里。
    - **"没调用"与"调了但没能执行"**（被暂停灰罩或特效盖住、前置不满足）
      在信息上等价，都落在这里，因此这里**不带"为什么"**：分成两个态只会
      制造一个假区别。反过来，"没能执行"绝不许冒充"未命中"。
    - 算类输入不全、记类还没有记录，也都落在这里（§1「一般形式」）。
    """


@final
@dataclass(frozen=True)
class SampledEmpty:
    """已采样但为空：看了，那里没有东西。

    **它不是"未知"。** 未知是"那里有东西，我不知道是什么"；这一支是"那里
    没有东西"。两者混掉就是假事实（§1）。

    置信度：原语的未命中只有分数够高时才配升格成这一支（分数低时字段只能
    报 Unsampled），所以升格所依据的那个分数留在这里；None = 来源不是读。
    """

    confidence: float | None

    def __post_init__(self) -> None:
        _check_confidence(self.confidence)


@final
@dataclass(frozen=True)
class Valued:
    """有值：那里确实有东西，值以载荷（形态）表达。

    - 载荷只能是 Payload 词表里的一种，别的（含三态本身）一律拒绝；
    - 这里**没有**形态字段：形态就是载荷的类型，所以构造不出「形态 = 精确
      而值是 [15, 16]」；
    - 置信度：原语算出的连续量，只有"读"有；None = 该字段不是识别来的；
    - 载荷与置信度都**没有默认值**：图省事的默认值正是"不知道伪装成知道"
      的入口。
    """

    payload: Payload
    confidence: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.payload, PAYLOAD_TYPES):
            raise TypeError(
                "有值的载荷只能是载荷词表里的一种，收到 "
                f"{type(self.payload).__name__}：三态不是载荷，"
                "「未知」才是"
            )
        _check_confidence(self.confidence)


SlotValue: TypeAlias = Unsampled | SampledEmpty | Valued
"""一个字段（槽）的取值：三态之一。"""

SLOT_STATES: Final[tuple[type, ...]] = (Unsampled, SampledEmpty, Valued)
"""三态只有这三支；载荷不是第四支。"""
