"""§1 槽：三态 × 载荷 × 置信度。

对应《契约层》§1 与「这一层唯一的验收判据」：让"不知道"在类型上没法被伪装
成"知道"。

三条硬约束各有一组测试，名字以 test_a_ / test_b_ / test_c_ 开头，与任务里
那三条断言一一对应。
"""

from __future__ import annotations

import dataclasses

import pytest

from ark.contracts import (
    PAYLOAD_TYPES,
    SLOT_STATES,
    Exact,
    GridCoord,
    Interval,
    LowerBound,
    Region,
    SampledEmpty,
    Unknown,
    Unsampled,
    UpperBound,
    Valued,
)


# ---------------------------------------------------------------------------
# a. 「未采样」与「有值 + 未知」是两个不同的值
# ---------------------------------------------------------------------------


def test_a_unsampled_is_not_valued_unknown() -> None:
    """a. 两种"不知道"是两类不同的东西，不是同一个东西的两种写法。"""
    never_sampled = Unsampled()
    there_is_something_unknown = Valued(Unknown(), None)

    assert never_sampled != there_is_something_unknown
    assert type(never_sampled) is not type(there_is_something_unknown)
    assert not isinstance(there_is_something_unknown, Unsampled)
    assert not isinstance(never_sampled, Valued)
    # 两支各自都还在三态里，没有被顶掉
    assert isinstance(never_sampled, Unsampled | SampledEmpty | Valued)
    assert isinstance(there_is_something_unknown, Unsampled | SampledEmpty | Valued)


def test_a_no_construction_path_mixes_them() -> None:
    """a. 任何构造路径都造不出混淆：载荷位、置信度位、三态位都不收对方。"""
    with pytest.raises(TypeError):
        Valued(Unsampled(), None)  # 未采样不是载荷
    with pytest.raises(TypeError):
        Valued(Unknown(), Unsampled())  # 未采样不是置信度
    with pytest.raises(TypeError):
        Unsampled(Unknown())  # 未采样不收载荷
    with pytest.raises(TypeError):
        SampledEmpty(Unknown())  # 已采样但为空更不收载荷
    # 两个词表各自封闭：未采样不在载荷里，未知不在三态里
    assert Unsampled not in PAYLOAD_TYPES
    assert Unknown not in SLOT_STATES


# ---------------------------------------------------------------------------
# b. 构造不出「精确」而值是区间的东西
# ---------------------------------------------------------------------------


def test_b_exact_accepts_numbers_bools_and_strings() -> None:
    """值域已放宽（§1 修正过一次）：数 / 布尔 / 字符串都装得下。

    出处是《视觉层/原语》§1：七条原语里有三条的「精确」不是数 ——
    exists 的**有 / 无**、classify 的**标签**、heading 的**四方位**。
    """
    assert Exact(15).value == 15
    assert Exact(15.5).value == 15.5
    assert Exact(True).value is True
    assert Exact(False).value is False
    assert Exact("能天使").value == "能天使"
    assert Exact("右").value == "右"


def test_b_exact_still_rejects_payloads_states_and_containers() -> None:
    """b. 放宽值域不等于什么都收：载荷自己、三态、容器仍旧构造不出来。"""
    for bad in (
        Interval(15, 16),
        LowerBound(15),
        UpperBound(16),
        Unknown(),
        Unsampled(),
        SampledEmpty(0.9),
        Valued(Exact(15), None),
        [15, 16],
        (15, 16),
        {"v": 15},
        None,
        b"15",
    ):
        with pytest.raises(TypeError):
            Exact(bad)


def test_b_interval_and_bounds_still_reject_bool_and_string() -> None:
    """放宽的只有「精确」：区间 / 下界 / 上界仍然只收数，bool 也不算数。"""
    assert LowerBound(15).lo == 15
    assert UpperBound(16).hi == 16
    for bad_call in (
        lambda: Interval(True, 1),
        lambda: Interval(1, "16"),
        lambda: Interval(1, None),
        lambda: LowerBound(True),
        lambda: LowerBound("15"),
        lambda: UpperBound(True),
        lambda: UpperBound("16"),
    ):
        with pytest.raises(TypeError):
            bad_call()


def test_b_exact_bool_equals_int_and_that_is_a_known_wart() -> None:
    """已知的疣：bool 是 int 的子类，所以 Exact(True) == Exact(1)。接受，不绕。

    包一层布尔的绕法会让值不再等于它表达的那个东西，比这个疣更糟。
    """
    assert Exact(True) == Exact(1)
    assert len({Exact(True), Exact(1)}) == 1


def test_b_form_is_the_payload_own_type_not_a_parallel_field() -> None:
    """b. 形态不是并列字段：五个载荷各长各的字段，没有一个是"形态"。"""
    expected = {
        Exact: ["value"],
        Interval: ["lo", "hi"],
        LowerBound: ["lo"],
        UpperBound: ["hi"],
        Unknown: [],
        Region: ["center", "radius"],
    }
    for cls, names in expected.items():
        assert [f.name for f in dataclasses.fields(cls)] == names
        assert "form" not in {f.name for f in dataclasses.fields(cls)}
        assert "shape" not in {f.name for f in dataclasses.fields(cls)}
    # 有值把载荷整个装进来，自己只多一个置信度
    assert {f.name for f in dataclasses.fields(Valued)} == {"payload", "confidence"}


def test_b_interval_needs_two_closed_ends() -> None:
    """b. 区间两端都封闭且 lo <= hi；开放的一端另有下界/上界两个类型。"""
    assert (Interval(15, 16).lo, Interval(15, 16).hi) == (15, 16)
    assert Interval(15, 15).lo == 15  # 退化区间仍是一个区间
    with pytest.raises(ValueError):
        Interval(16, 15)
    with pytest.raises(TypeError):
        Interval(15, None)
    with pytest.raises(TypeError):
        Interval(None, 16)
    assert LowerBound(15).lo == 15
    assert UpperBound(16).hi == 16
    with pytest.raises(TypeError):
        LowerBound(None)
    with pytest.raises(TypeError):
        UpperBound(None)


# ---------------------------------------------------------------------------
# b. 区间 / 下界 / 上界 共用同一套操作（§2 的包含与相交）
# ---------------------------------------------------------------------------


def test_b_contains_a_point_on_closed_and_open_ends() -> None:
    """b. 包含一个点：闭合段卡两端，开放端不设限。"""
    closed = Interval(15, 16)
    at_least = LowerBound(15)
    at_most = UpperBound(16)

    assert closed.contains(15) is True and closed.contains(16) is True
    assert closed.contains(15.5) is True
    assert closed.contains(14.999) is False and closed.contains(16.001) is False

    assert at_least.contains(15) is True
    assert at_least.contains(1e9) is True  # 下界：上面不设限
    assert at_least.contains(14.999) is False

    assert at_most.contains(16) is True
    assert at_most.contains(-1e9) is True  # 上界：下面不设限
    assert at_most.contains(16.001) is False

    with pytest.raises(TypeError):
        closed.contains(True)
    with pytest.raises(TypeError):
        closed.contains("15")
    with pytest.raises(TypeError):
        closed.contains(Unsampled())


def test_b_contains_an_interval_on_open_ends() -> None:
    """b. 区间 ⊇ 区间：降级就是区间变宽 —— 对方开放着的那一端必须我也开放。"""
    closed = Interval(15, 16)
    at_least = LowerBound(15)
    at_most = UpperBound(16)

    assert closed.contains(closed) is True
    assert closed.contains(Interval(15.5, 15.6)) is True
    assert closed.contains(Interval(15, 17)) is False
    assert closed.contains(at_least) is False  # [15,16] ⊉ [15,∞)
    assert at_least.contains(closed) is True  # [15,∞) ⊇ [15,16]
    assert at_least.contains(Interval(20, 30)) is True
    assert at_most.contains(closed) is True  # (-∞,16] ⊇ [15,16]
    assert at_most.contains(at_least) is False


def test_b_intersects_on_closed_and_open_ends() -> None:
    """b. 相交：两段可能指同一个值吗。端点相接算相交（开放=不设限，不是不含端点）。"""
    assert Interval(15, 16).intersects(Interval(15.5, 16.5)) is True
    assert Interval(15, 16).intersects(Interval(16.5, 17)) is False
    assert Interval(15, 16).intersects(Interval(16, 17)) is True
    assert Interval(15, 16).intersects(15.5) is True  # 一个点也算
    assert LowerBound(15).intersects(UpperBound(20)) is True
    assert LowerBound(15).intersects(UpperBound(10)) is False
    with pytest.raises(TypeError):
        Interval(15, 16).intersects(object())


def test_b_the_three_forms_share_one_implementation() -> None:
    """b. 三件是三个类型，却共用同一套操作 —— 操作写在共同底座上，不是各写三遍。"""
    assert type(Interval(15, 16)) is not type(LowerBound(15))
    assert type(Interval(15, 16)) is not type(UpperBound(16))
    assert type(LowerBound(15)) is not type(UpperBound(16))
    bases = {cls.__mro__[1].__name__ for cls in (Interval, LowerBound, UpperBound)}
    assert bases == {"_HalfOpenInterval"}


# ---------------------------------------------------------------------------
# c. 只有"有值"才可能带载荷
# ---------------------------------------------------------------------------


def test_c_only_valued_has_a_payload() -> None:
    """c. 未采样与已采样但为空都没有载荷位；有值的载荷是必填的。"""
    assert not hasattr(Unsampled(), "payload")
    assert not hasattr(SampledEmpty(0.9), "payload")
    with pytest.raises(TypeError):
        Valued(confidence=0.5)  # 载荷没有默认值，不填就构造不出来
    with pytest.raises(AttributeError):
        Unsampled().payload = Unknown()  # FrozenInstanceError 也是 AttributeError


def test_c_payload_vocabulary_is_exactly_six() -> None:
    """c. 载荷词表就那六种；三态自身不是载荷。

    第六种是**二维区域**（Region，形状在 §2）：载荷的区分是"知道得多准"，
    而区域越小越准 —— 它是形态，不是值的种类。
    """
    assert set(PAYLOAD_TYPES) == {Exact, Interval, LowerBound, UpperBound, Unknown, Region}
    assert set(SLOT_STATES) == {Unsampled, SampledEmpty, Valued}
    assert not set(PAYLOAD_TYPES) & set(SLOT_STATES)


def test_d_unsampled_is_field_level_not_primitive_level() -> None:
    """d. 三态是字段级的：原语结果的形状里没有"未采样"这一支。

    原语的结果只有"有读数（载荷 + 分数 + 矩形）"与"没有读数"两种；矩形属于
    §2 几何，不在这里。所以这里能验的是：载荷词表里没有未采样，而未采样也
    不是"有值 + 未知"——它压根不带载荷。
    """
    assert Unsampled not in PAYLOAD_TYPES
    assert not issubclass(Unsampled, tuple(PAYLOAD_TYPES))
    assert not issubclass(Unknown, Unsampled)
    assert not issubclass(Unsampled, Unknown)


# ---------------------------------------------------------------------------
# 不许给槽加默认值：三态、载荷、置信度都必须显式说
# ---------------------------------------------------------------------------


def test_no_field_has_a_default() -> None:
    """图省事的默认值正是"不知道伪装成知道"的入口，一个也不许有。"""
    for cls in (*SLOT_STATES, *PAYLOAD_TYPES):
        for f in dataclasses.fields(cls):
            assert f.default is dataclasses.MISSING, f"{cls.__name__}.{f.name} 有默认值"
            assert f.default_factory is dataclasses.MISSING, (
                f"{cls.__name__}.{f.name} 有工厂默认值"
            )


def test_confidence_must_be_a_number_or_explicit_none() -> None:
    """置信度只有"读"有：None 是"不是读来的"，不是 0，也不许省略。"""
    assert Valued(Exact(3), 0.87).confidence == 0.87
    assert Valued(Exact(3), None).confidence is None
    assert SampledEmpty(0.9).confidence == 0.9
    assert SampledEmpty(None).confidence is None
    for bad in ("0.9", True, [0.9]):
        with pytest.raises(TypeError):
            Valued(Exact(3), bad)
        with pytest.raises(TypeError):
            SampledEmpty(bad)


def test_slot_values_are_immutable() -> None:
    """值是只读的：一份快照交出去以后，谁也不能回头把它改成另一个事实。"""
    valued = Valued(Exact(3), 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        valued.payload = Unknown()
    with pytest.raises(dataclasses.FrozenInstanceError):
        valued.confidence = 0.1
    with pytest.raises(dataclasses.FrozenInstanceError):
        SampledEmpty(0.5).confidence = 0.1


def test_values_are_hashable_and_equal_by_content() -> None:
    """三态本身可以进集合、可以按内容比较 —— 但两支不同的态永不相等。"""
    assert Unsampled() == Unsampled()
    assert Valued(Exact(3), 0.5) == Valued(Exact(3), 0.5)
    assert Valued(Exact(3), 0.5) != Valued(Exact(3), 0.6)
    assert len({Unsampled(), Unsampled(), Valued(Unknown(), None)}) == 2
