"""§3 刻度与时间态：逻辑帧 · 墙钟毫秒 · 时间态。

**两种刻度**混起来会产生一个**静默的错**：同一次 200 ms 的 ADB 延迟，常速下是
6 个游戏帧，慢放下只有 0.6 帧（时间基 §5.1）。这是**只有类型能拦住**的一类错 ——
本模块就是那道拦。

- **逻辑帧**：游戏时间，1 游戏秒 = 30 帧。轴 · 条件 · 快照页眉 · 观测迹只用
  它；它**不是从时钟读来的**，由费用推算，所以时钟端口里没有它。
- **墙钟毫秒**：真实时间。ADB 延迟 · 超时 · 判定节拍用它，只在记录里留痕，
  用来解释"为什么这一段墙钟很长"；
- **时间态**：常速 / 慢放 / 暂停（三值）。**类型在这一层，行为在控制层**：游戏在
  不在走是个物理事实，跨层都要说，所以它进契约层。

跨刻度一律拒绝（构造 / 全部二元运算 / 比较），同刻度照常。这里**不提供换算**：
换算取决于时间态（常速 / 慢放），必须由调用方显式带上。

**时间态与 §2 的视图是两个类型**：视图 = 时间态去掉「暂停」（投影只有两套，
没有"暂停投影"这种东西）。两个都留着，因为**类型不该允许不存在的取值**。
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from enum import Enum
from types import NotImplementedType
from typing import Final, final

__all__ = [
    "LOGICAL_FRAMES_PER_GAME_SECOND",
    "LogicalFrame",
    "WallClockMs",
    "TimeState",
]

LOGICAL_FRAMES_PER_GAME_SECOND: Final[int] = 30
"""1 游戏秒 = 30 逻辑帧（§3）。"""


def _crosses(a: _Scale, b: object) -> bool:
    """a 与 b 是不是两种**不同**的刻度。"""
    return isinstance(b, _Scale) and type(b) is not type(a)


class _Scale(int):
    """两种刻度的共同底座。**不是公开 API**，公开的是下面那两个子类。

    为什么值得进契约层：同一次 200 ms 的 ADB 延迟，常速下是 6 个游戏帧，
    慢放下只有 0.6 帧（时间基 §5.1）。混起来不报错，只算错 —— 这是只有
    类型能拦住的一类错。所以三条路都堵住：

    - 构造：LogicalFrame(WallClockMs(200)) 抛 TypeError；
    - 二元运算：加、减、乘、除、整除、取模、幂、位移、按位，只要一侧是另一个
      刻度就抛 TypeError（WallClockMs(200) * LogicalFrame(6) 也不例外）；
    - 比较：LogicalFrame(6) == WallClockMs(6) 是 False，小于 / 大于抛 TypeError。

    **裸 int 不拦**：写 LogicalFrame(6) + 1 时那个 1 没有刻度；
    "分得开"只在两个刻度之间才谈得上。加减是刻度内的量（结果仍留在这个刻度
    上），乘除取模等沿用 int 的结果。

    这里**不提供换算**。换算取决于时间态（常速 / 慢放），必须由调用方显式
    带上，不该藏在刻度类型里。
    """

    __slots__ = ()

    def __new__(cls, value: int) -> _Scale:
        if isinstance(value, _Scale) and type(value) is not cls:
            raise TypeError(
                f"{cls.__name__} 不能由 {type(value).__name__} 构造："
                "逻辑帧与墙钟毫秒是两个刻度，不许互换"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{cls.__name__} 只收整数，收到 {type(value).__name__}")
        return super().__new__(cls, value)

    # -- 算术 -------------------------------------------------------------

    def __add__(self, other: object) -> _Scale | NotImplementedType:
        if _crosses(self, other):
            return NotImplemented
        if isinstance(other, int):
            return type(self)(int(self) + int(other))
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: object) -> _Scale | NotImplementedType:
        if _crosses(self, other):
            return NotImplemented
        if isinstance(other, int):
            return type(self)(int(self) - int(other))
        return NotImplemented

    def __rsub__(self, other: object) -> _Scale | NotImplementedType:
        if _crosses(self, other):
            return NotImplemented
        if isinstance(other, int):
            return type(self)(int(other) - int(self))
        return NotImplemented

    # -- 比较 -------------------------------------------------------------

    def __eq__(self, other: object) -> bool | NotImplementedType:
        # 跨刻度不是"数值不等"而是"两回事"：1 帧与 1 毫秒本来就不相干，
        # 数值相同也不相等。裸 int 没有刻度，按数值比。
        if _crosses(self, other):
            return False
        if isinstance(other, int):
            return int(self) == int(other)
        return NotImplemented

    # 定义了 __eq__ 就得自己把 __hash__ 拿回来（否则它会变成 None）。
    __hash__ = int.__hash__

    def _compare(
        self, other: object, op: Callable[[int, int], bool]
    ) -> bool | NotImplementedType:
        if _crosses(self, other):
            return NotImplemented
        if isinstance(other, int):
            return op(int(self), int(other))
        return NotImplemented

    def __lt__(self, other: object) -> bool | NotImplementedType:
        return self._compare(other, operator.lt)

    def __le__(self, other: object) -> bool | NotImplementedType:
        return self._compare(other, operator.le)

    def __gt__(self, other: object) -> bool | NotImplementedType:
        return self._compare(other, operator.gt)

    def __ge__(self, other: object) -> bool | NotImplementedType:
        return self._compare(other, operator.ge)

    def __ne__(self, other: object) -> bool | NotImplementedType:
        # 必须自己写：int 自己有 __ne__，它不会回头问被覆盖过的 __eq__，
        # 于是 LogicalFrame(6) != WallClockMs(6) 会走成 6 != 6 而返回 False。
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


# 加、减、比较已经手写在 _Scale 里（它们有刻度语义：同刻度相减仍是同一刻度上
# 的量）。剩下的那些二元算子没有刻度语义，但**跨刻度同样必须拒绝** ——
# 漏掉哪一个，那一个就正好是"类型没拦住"的那条路，所以一个也不留。
_CROSS_SCALE_GUARDED_OPS: Final[tuple[str, ...]] = (
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__divmod__",
    "__pow__",
    "__lshift__",
    "__rshift__",
    "__and__",
    "__xor__",
    "__or__",
)


def _guarded_binary(name: str, *, reflected: bool) -> Callable[..., object]:
    """把一个 int 的二元算子包成"跨刻度就拒绝"，别的照旧交给 int。"""
    forward = getattr(int, name)

    def method(self: _Scale, other: object) -> object:
        if _crosses(self, other):
            return NotImplemented
        # 反射版就是把操作数换个位置 —— 两边都是整数时，这与 int 的 __rXXX__ 等价。
        return forward(other, self) if reflected else forward(self, other)

    method.__name__ = ("__r" + name[2:]) if reflected else name
    return method


def _seal_cross_scale_ops() -> None:
    for name in _CROSS_SCALE_GUARDED_OPS:
        setattr(_Scale, name, _guarded_binary(name, reflected=False))
        setattr(_Scale, "__r" + name[2:], _guarded_binary(name, reflected=True))


_seal_cross_scale_ops()


@final
class LogicalFrame(_Scale):
    """逻辑帧：游戏时间，整数帧号。

    轴、条件、快照页眉、观测迹都只用它。它**不是从时钟读来的**，由费用推算
    （时间基），所以时钟端口里没有它。
    """

    __slots__ = ()


@final
class WallClockMs(_Scale):
    """墙钟毫秒：真实时间，整数毫秒。

    ADB 延迟、超时、判定节拍用它。它只在记录里留痕，用来解释"为什么这一段
    墙钟很长"；轴上不许出现它。
    """

    __slots__ = ()


# ---------------------------------------------------------------------------
# 时间态：三值（常速 / 慢放 / 暂停）
# ---------------------------------------------------------------------------


class TimeState(Enum):
    """**时间态**：游戏在不在走，三种（常速 / 慢放 / 暂停）。

    - **类型在这一层，行为在控制层**：它在跨层到处出现（快照页眉、动作合法性、
      取像节拍），所以"是哪一种"必须是个说得清的值，而不是一个字符串；
    - **它与 §2 的视图是两个类型**：视图两值（常速 / 慢放），**视图 = 时间态去掉
      「暂停」**。两个都留着，是因为**类型不该允许不存在的取值** —— 拿三值的
      时间态去标投影，等于允许"暂停投影"这种东西存在，那正是这一层一路在拦的
      那类东西。反过来说，视图回答不了"游戏是不是停着"，那要问时间态。
    """

    NORMAL = "常速"
    SLOW = "慢放"
    PAUSED = "暂停"

    def __str__(self) -> str:
        """打印出来就是它自己那个词（常速 / 慢放 / 暂停）。

        不继承 str：那样 TimeState.NORMAL == "常速" 会成立，等于把两个刻度
        又混回一起（§3 一整条在防的就是这个）。这里只让 **打印** 好看、
        让报告里读得出是哪一个取值。
        """
        return self.value
