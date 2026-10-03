"""§3 刻度：逻辑帧与墙钟毫秒，类型上必须分得开。

为什么这值得测：同一次 200 ms 的 ADB 延迟，常速下是 6 个游戏帧，慢放下只有
0.6 帧。混起来不报错、只算错 —— 这些测试就是"类型必须拦住它"的现场。
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ark.contracts import (
    LOGICAL_FRAMES_PER_GAME_SECOND,
    LogicalFrame,
    TimeState,
    View,
    WallClockMs,
)


def test_the_two_scales_are_distinct_types() -> None:
    """两个刻度是两个类型，谁的 isinstance 都拦不住自己。"""
    frame = LogicalFrame(6)
    delay = WallClockMs(200)
    assert type(frame) is not type(delay)
    assert isinstance(frame, LogicalFrame) and not isinstance(frame, WallClockMs)
    assert isinstance(delay, WallClockMs) and not isinstance(delay, LogicalFrame)
    # 两者仍然是整数：刻度是整数刻度
    assert isinstance(frame, int) and isinstance(delay, int)
    assert issubclass(LogicalFrame, int) and issubclass(WallClockMs, int)


@pytest.mark.parametrize(
    "build",
    [LogicalFrame, WallClockMs],
    ids=["LogicalFrame", "WallClockMs"],
)
def test_a_scale_only_accepts_a_plain_integer(build: type) -> None:
    """刻度是整数：小数、字符串、None、bool 都不收。"""
    for bad in (1.5, "200", None, True, [200]):
        with pytest.raises(TypeError):
            build(bad)


def test_cross_scale_construction_is_refused() -> None:
    """一个刻度不能拿另一个刻度当输入构造 —— 这是最直接的那条混淆路。"""
    with pytest.raises(TypeError):
        LogicalFrame(WallClockMs(200))
    with pytest.raises(TypeError):
        WallClockMs(LogicalFrame(6))


@pytest.mark.parametrize(
    "op",
    [
        lambda delay, frame: delay + frame,
        lambda delay, frame: frame + delay,
        lambda delay, frame: delay - frame,
        lambda delay, frame: frame - delay,
        lambda delay, frame: delay < frame,
        lambda delay, frame: frame < delay,
        lambda delay, frame: delay <= frame,
        lambda delay, frame: frame <= delay,
        lambda delay, frame: delay > frame,
        lambda delay, frame: frame > delay,
        lambda delay, frame: delay >= frame,
        lambda delay, frame: frame >= delay,
        lambda delay, frame: delay * frame,
        lambda delay, frame: frame * delay,
        lambda delay, frame: delay / frame,
        lambda delay, frame: frame / delay,
        lambda delay, frame: delay // frame,
        lambda delay, frame: frame // delay,
        lambda delay, frame: delay % frame,
        lambda delay, frame: frame % delay,
        lambda delay, frame: delay**frame,
        lambda delay, frame: frame**delay,
        lambda delay, frame: delay & frame,
        lambda delay, frame: frame | delay,
        lambda delay, frame: delay ^ frame,
        lambda delay, frame: frame << delay,
        lambda delay, frame: delay >> frame,
        lambda delay, frame: divmod(delay, frame),
    ],
    ids=[
        "ms+frame",
        "frame+ms",
        "ms-frame",
        "frame-ms",
        "ms<frame",
        "frame<ms",
        "ms<=frame",
        "frame<=ms",
        "ms>frame",
        "frame>ms",
        "ms>=frame",
        "frame>=ms",
        "ms*frame",
        "frame*ms",
        "ms/frame",
        "frame/ms",
        "ms//frame",
        "frame//ms",
        "ms%frame",
        "frame%ms",
        "ms**frame",
        "frame**ms",
        "ms&frame",
        "frame|ms",
        "ms^frame",
        "frame<<ms",
        "ms>>frame",
        "divmod(ms,frame)",
    ],
)
def test_cross_scale_arithmetic_and_ordering_are_refused(
    op: Callable[[WallClockMs, LogicalFrame], object],
) -> None:
    """跨刻度的加减与大小比较一律 TypeError：静默的错在这里变成响的。"""
    with pytest.raises(TypeError):
        op(WallClockMs(200), LogicalFrame(6))


def test_cross_scale_equality_is_false_even_at_the_same_number() -> None:
    """数值相同也不相等：1 帧和 1 毫秒是两回事，不是同一个量。"""
    assert LogicalFrame(6) != WallClockMs(6)
    assert not (LogicalFrame(6) == WallClockMs(6))
    assert len({LogicalFrame(6), WallClockMs(6)}) == 2
    assert len({LogicalFrame(6), LogicalFrame(6)}) == 1


def test_same_scale_arithmetic_stays_on_that_scale() -> None:
    """同一个刻度内照常算，结果还是那个刻度；裸 int 没有刻度，不拦。"""
    total = LogicalFrame(4) + LogicalFrame(2)
    assert type(total) is LogicalFrame
    assert total == 6
    assert type(LogicalFrame(6) - LogicalFrame(2)) is LogicalFrame
    assert type(LogicalFrame(6) + 1) is LogicalFrame
    assert int(LogicalFrame(6) + 1) == 7
    assert type(1 + LogicalFrame(6)) is LogicalFrame
    assert type(sum([LogicalFrame(1), LogicalFrame(2)])) is LogicalFrame
    assert type(WallClockMs(200) + WallClockMs(50)) is WallClockMs
    assert LogicalFrame(6) == 6  # 裸 int 按数值比
    assert LogicalFrame(6) < 7
    assert WallClockMs(200) > 100


def test_non_additive_arithmetic_within_a_scale_defers_to_int() -> None:
    """加、减有刻度语义（结果仍留在这个刻度上）；其余算子沿用 int 的结果。"""
    assert type(LogicalFrame(6) + 1) is LogicalFrame
    assert LogicalFrame(6) * 2 == 12
    assert WallClockMs(200) // 3 == 66
    assert WallClockMs(200) / 2 == 100.0


def test_time_state_is_three_valued() -> None:
    """§3：时间态三值 —— 常速 / 慢放 / 暂停。它是跨层的物理事实，所以有类型。"""
    members = list(TimeState)
    assert len(members) == 3
    assert [member.value for member in members] == ["常速", "慢放", "暂停"]
    assert TimeState.NORMAL.value == "常速"
    assert TimeState.SLOW.value == "慢放"
    assert TimeState.PAUSED.value == "暂停"


def test_time_state_and_view_are_two_types() -> None:
    """视图 = 时间态去掉「暂停」：两个类型都留着，**类型不该允许不存在的取值**。

    拿三值的时间态去标投影，等于允许"暂停投影"这种东西存在。
    """
    assert TimeState is not View
    assert {view.value for view in View} == {"常速", "慢放"}
    assert "暂停" not in {view.value for view in View}
    assert {view.value for view in View} < {state.value for state in TimeState}
    # 两个类型之间没有构造路径：谁也不是谁的子类，也不能互相构造
    assert not issubclass(TimeState, View) and not issubclass(View, TimeState)
    with pytest.raises(ValueError):
        View(TimeState.PAUSED)  # 枚举里没有这一支：构造不出来，也就没有"暂停投影"
    with pytest.raises(ValueError):
        TimeState("快进")  # 不存在的取值同样构造不出来


def test_the_game_second_constant_is_thirty_frames() -> None:
    """§3：1 游戏秒 = 30 帧。这个常数是刻度自己的定义，不是换算函数。"""
    assert LOGICAL_FRAMES_PER_GAME_SECOND == 30
