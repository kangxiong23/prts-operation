"""契约层的规格符合性验收 —— **由规划者维护，不住在 tests/unit/**。

与 `tests/unit/` 的分工：

| 目录 | 谁写 | 验证什么 |
| --- | --- | --- |
| `tests/unit/` | 执行者 | "实现按我的设计工作" |
| **本文件** | **规划者** | **"规格说的那些话成立"** |

分工不是形式。执行者写的测试**和它自己的实现共享同一套心智模型**——它测的是它
想到的那些面，想不到的那一面正好是漏洞所在（实例：`Exact` 的值域，48 条单元
测试全绿而硬伤照样在）。

## 两条纪律

1. **只用公开面。** 只 `from ark.contracts import ...`。碰 `_Scale` / `_crosses`
   这类内部名，验收就绑死在实现上了——内部名一改，验收跟着改，那它就不是验收。
2. **每条检查都带规格出处。** 没有出处的检查是**发明规范**；出处对不上的检查是
   **发现了规格洞**。两种情况都要能在报告里看出来。

## 跑法（不需要 pytest）

    python tests/spec/contracts_spec.py

pytest 外壳是 `test_contracts_spec.py`，它只是把下面那张表逐条变成用例。
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

# 独立跑得起来：不依赖 pytest 的 pythonpath 设置。
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ark.contracts import (  # noqa: E402
    LOGICAL_FRAMES_PER_GAME_SECOND,
    PAYLOAD_TYPES,
    SLOT_STATES,
    Exact,
    Interval,
    LogicalFrame,
    LowerBound,
    SampledEmpty,
    Unknown,
    Unsampled,
    UpperBound,
    Valued,
    WallClockMs,
)

MISSING = dataclasses.MISSING


@dataclass(frozen=True)
class Check:
    """一条规格检查。

    - `ident`：稳定编号，报告与 pytest 用例名都用它；
    - `spec`：规格出处，具体到节；
    - `claim`：规格里的那句话，一句话；
    - `probe`：返回**证据字符串**；不成立时抛 AssertionError，消息就是证据。
    """

    ident: str
    spec: str
    claim: str
    probe: Callable[[], str]


def _refuses(
    exc: type[BaseException], fn: Callable[..., object], *args: object, **kwargs: object
) -> str:
    """fn(*args, **kwargs) 必须抛 exc。返回实际抛出的类型名，作为证据。"""
    try:
        fn(*args, **kwargs)
    except exc as caught:
        return type(caught).__name__
    except BaseException as other:  # noqa: BLE001 —— 验收要把"抛错了"报清楚
        raise AssertionError(
            f"期望 {exc.__name__}，实际 {type(other).__name__}：{other}"
        ) from None
    raise AssertionError(f"期望 {exc.__name__}，实际**没抛**")


def check_public_surface() -> str:
    """契约层「这份文件负责什么」：它是**唯一被所有人 import** 的东西——
    所以它的公开面必须是一份**列得出来的名单**（`__all__`），不是一个会漂移的集合。

    这条同时是 T1.1b「拆叶子不改接口」的验收：T1.1 定下的那 15 个名字
    **一个都不许少、一个都不许改名**。后续的节往上加名字是允许的 ——
    名单只增不减，才是"接口稳定"该有的样子。
    """
    import ark.contracts as contracts

    from_t11 = {
        "LOGICAL_FRAMES_PER_GAME_SECOND",
        "LogicalFrame",
        "WallClockMs",
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
    }
    actual = set(contracts.__all__)
    lost = from_t11 - actual
    assert not lost, f"T1.1 定下的名字丢了或被改名了：{sorted(lost)}"
    for name in sorted(actual):
        assert getattr(contracts, name, None) is not None, f"__all__ 里的 {name} 取不到"
    return f"__all__ 共 {len(actual)} 个名字；T1.1 那 15 个一个不少"


# ---------------------------------------------------------------------------
# 契约层 §1 槽：三态 × 载荷 × 置信度
# ---------------------------------------------------------------------------


def check_two_vocabularies() -> str:
    assert len(SLOT_STATES) == 3, f"三态应当三支，实际 {len(SLOT_STATES)}"
    assert len(PAYLOAD_TYPES) == 6, f"载荷应当六支，实际 {len(PAYLOAD_TYPES)}"
    overlap = set(SLOT_STATES) & set(PAYLOAD_TYPES)
    assert not overlap, f"两套词表不许相交，交集 {overlap}"
    return (
        f"三态 {[c.__name__ for c in SLOT_STATES]}"
        f" ／ 载荷 {[c.__name__ for c in PAYLOAD_TYPES]}"
    )


def check_unsampled_is_not_valued_unknown() -> str:
    never = Unsampled()
    something = Valued(Unknown(), None)
    assert never != something, "两种'不知道'不许相等"
    assert type(never) is not type(something), "两种'不知道'不许是同一个类型"
    _refuses(TypeError, Valued, never, None)  # 未采样不是载荷
    _refuses(TypeError, Valued, Unknown(), never)  # 未采样不是置信度
    _refuses(TypeError, Unsampled, Unknown())  # 未采样不收载荷
    return "两个类型、互不相等；三条构造路径全被拒"


def check_exact_cannot_hold_a_payload() -> str:
    _refuses(TypeError, Exact, Interval(15, 16))
    _refuses(TypeError, Exact, LowerBound(15))
    _refuses(TypeError, Exact, UpperBound(16))
    _refuses(TypeError, Exact, Unknown())
    _refuses(TypeError, Exact, [15, 16])
    _refuses(TypeError, Exact, None)
    return "区间 / 下界 / 上界 / 未知 / 列表 / None 都进不了 Exact"


def check_form_is_not_a_parallel_field() -> str:
    banned = {"form", "shape", "kind", "形态", "取值形态"}
    hits = []
    for cls in PAYLOAD_TYPES:
        names = {f.name for f in dataclasses.fields(cls)}
        bad = names & banned
        if bad:
            hits.append(f"{cls.__name__} 上有 {bad}")
    assert not hits, "；".join(hits)
    return "五个载荷类上都没有 form / shape / kind / 形态 字段"


def check_unknown_is_a_payload_not_a_state() -> str:
    assert Unknown in PAYLOAD_TYPES, "「未知」必须是载荷"
    assert Unknown not in SLOT_STATES, "「未知」不许是三态的一支"
    payload = Valued(Unknown(), None).payload
    assert isinstance(payload, Unknown), "「未知」要能装进 Valued"
    return "Unknown ∈ 载荷，∉ 三态，且装得进 Valued"


def check_only_valued_has_a_payload() -> str:
    assert not hasattr(Unsampled(), "payload"), "未采样不许有载荷位"
    assert not hasattr(SampledEmpty(0.9), "payload"), "已采样但为空不许有载荷位"
    _refuses(TypeError, Valued, confidence=0.5)
    return "未采样 / 已采样但为空都没有 payload 位；Valued.payload 必填"


def check_interval_and_the_two_half_open_forms() -> str:
    _refuses(TypeError, Interval, 15, None)
    _refuses(TypeError, Interval, None, 16)
    _refuses(ValueError, Interval, 16, 15)
    assert LowerBound(15).lo == 15
    assert UpperBound(16).hi == 16
    assert type(Interval(15, 16)) is not type(LowerBound(15)), "三者必须是三个类型"
    return "区间两端必填且 lo≤hi；下界 / 上界各开一端；三者是三个类型"


def check_exact_value_domain() -> str:
    """出处不只是契约层 §1 —— 是 视觉层/原语.md §1 那张表。

    七条原语里有**三条的「精确」不是数**：`exists` 的有 / 无、`classify` 的标签、
    `heading` 的四方位。所以只收数的 Exact 表达不出这三条原语的答案。
    """
    evidence = []
    for label, value in (
        ("数（measure / count / event_time）", 15),
        ("布尔（exists 有/无）", True),
        ("字符串（classify 标签）", "能天使"),
        ("字符串（heading 四方位）", "右"),
    ):
        Exact(value)
        evidence.append(label)
    return "；".join(evidence)


def check_numeric_payloads_reject_bool_and_str() -> str:
    for bad in (True, "15"):
        _refuses(TypeError, Interval, bad, 16)
        _refuses(TypeError, LowerBound, bad)
        _refuses(TypeError, UpperBound, bad)
    return "区间 / 下界 / 上界 拒 bool 与 str（bool 是 int 子类，必须显式拒）"


def check_confidence_only_for_reads() -> str:
    assert Valued(Exact(3), None).confidence is None
    assert SampledEmpty(None).confidence is None
    assert Valued(Exact(3), 0.87).confidence == 0.87
    _refuses(TypeError, Valued, Exact(3), "high")
    _refuses(TypeError, SampledEmpty, True)
    assert not hasattr(Unsampled(), "confidence"), "未采样不许有置信度位"
    return "None = 不是读来的；非数被拒；未采样不带置信度"


def check_no_defaults_anywhere() -> str:
    for cls in (*SLOT_STATES, *PAYLOAD_TYPES):
        for field in dataclasses.fields(cls):
            assert field.default is MISSING, f"{cls.__name__}.{field.name} 有默认值"
            assert field.default_factory is MISSING, (
                f"{cls.__name__}.{field.name} 有工厂默认值"
            )
    return f"{len(SLOT_STATES) + len(PAYLOAD_TYPES)} 个类，无一个有默认值"


# ---------------------------------------------------------------------------
# 契约层 §3 刻度：两种，类型上必须分得开
# ---------------------------------------------------------------------------


def check_scales_are_distinct_types() -> str:
    assert type(LogicalFrame(6)) is not type(WallClockMs(6))
    assert not isinstance(WallClockMs(6), LogicalFrame)
    assert not isinstance(LogicalFrame(6), WallClockMs)
    return "两个刻度是两个类型，谁的 isinstance 都拦不住自己"


def check_cross_scale_construction_is_refused() -> str:
    _refuses(TypeError, LogicalFrame, WallClockMs(200))
    _refuses(TypeError, WallClockMs, LogicalFrame(6))
    return "一个刻度不能拿另一个刻度当输入构造"


def check_cross_scale_binary_ops_are_refused() -> str:
    """把 int 的二元算子逐个走一遍 —— 漏掉哪一个，那一个就是没拦住的路。"""
    import operator

    binary = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "//": operator.floordiv,
        "%": operator.mod,
        "**": operator.pow,
        "<<": operator.lshift,
        ">>": operator.rshift,
        "&": operator.and_,
        "^": operator.xor,
        "|": operator.or_,
        "divmod": divmod,
    }
    frame, wall = LogicalFrame(6), WallClockMs(200)
    for symbol, op in binary.items():
        _refuses(TypeError, op, frame, wall)
        _refuses(TypeError, op, wall, frame)
    return f"{len(binary)} 个二元算子 × 两个方向，跨刻度全拒"


def check_cross_scale_equality_and_order() -> str:
    assert (LogicalFrame(6) == WallClockMs(6)) is False
    assert (LogicalFrame(6) != WallClockMs(6)) is True
    assert len({LogicalFrame(6), WallClockMs(6)}) == 2
    _refuses(TypeError, lambda a, b: a < b, LogicalFrame(6), WallClockMs(200))
    _refuses(TypeError, lambda a, b: a >= b, LogicalFrame(6), WallClockMs(200))
    return "数值相同也不相等；大小比较抛 TypeError"


def check_game_second_is_thirty_frames() -> str:
    assert LOGICAL_FRAMES_PER_GAME_SECOND == 30, LOGICAL_FRAMES_PER_GAME_SECOND
    return "1 游戏秒 = 30 逻辑帧"


def check_time_state_is_three_valued() -> str:
    """§3：**时间态**三值（常速 / 慢放 / 暂停）。它与 §2 的**视图**是两个类型：
    视图两值（无暂停），时间态三值 —— **类型不该允许不存在的取值**。

    **判据用 `.value`，不用 `str(member)`**（修正过一次）：早先那一版用 `str()` 判，
    等于顺手要求了"打印出来就是那个词"，而规格没说过这条 ——
    那是**用检查发明规范**，正是这套脚本自己的头号纪律要防的事。
    """
    state, view = _need("TimeState"), _need("View")
    try:
        members = list(state)
    except TypeError:
        raise AssertionError(f"{state.__name__} 不可枚举；期望三个取值") from None
    assert len(members) == 3, f"时间态应当三个取值，实际 {len(members)}：{members}"
    values = [m.value for m in members]
    for want in ("常速", "慢放", "暂停"):
        assert want in values, f"时间态缺「{want}」：{values}"
    assert state is not view, "时间态与视图是同一个类型 —— 投影会因此允许「暂停」"
    assert not issubclass(state, str), (
        "时间态继承了 str：那会让 TimeState.NORMAL == '常速' 成立，"
        "等于把「分得开」松掉一格 —— §3 整节防的就是这个"
    )
    return f"时间态三值：{values}；未继承 str"


# ---------------------------------------------------------------------------
# 「这一层唯一的验收判据」与它派生出的两条
# ---------------------------------------------------------------------------


def check_the_one_acceptance_criterion() -> str:
    """§「这一层唯一的验收判据」：拿一份快照问"这里面哪几个数其实是猜的"——
    答案必须能在**类型**里直接看出来，不用读实现。"""
    snapshot = {
        "费用": Valued(Exact(37), 0.98),
        "敌人位置": Valued(Interval(2.5, 3.5), 0.61),
        "敌方类型": Valued(Unknown(), 0.44),
        "干员技能": Unsampled(),
        "备战席": SampledEmpty(0.95),
    }

    def verdict(slot: object) -> str:
        """只看类型，不看实现、不看任何字段的语义。"""
        if isinstance(slot, Unsampled):
            return "没采到"
        if isinstance(slot, SampledEmpty):
            return "看了，那里没有"
        payload = slot.payload  # type: ignore[union-attr]
        if isinstance(payload, Exact):
            return "知道"
        if isinstance(payload, Unknown):
            return "有东西，不知道是什么"
        return "只知道一个范围"

    got = {name: verdict(slot) for name, slot in snapshot.items()}
    assert got["费用"] == "知道", got
    assert got["敌人位置"] == "只知道一个范围", got
    assert got["敌方类型"] == "有东西，不知道是什么", got
    assert got["干员技能"] == "没采到", got
    assert got["备战席"] == "看了，那里没有", got
    return "；".join(f"{k}={v}" for k, v in got.items())


def check_promotion_needs_a_score() -> str:
    """§1：「原语的未命中，只有分数够高时才配升格成"已采样但为空"」——
    那个分数必须有落点，否则"够高"没有判据。"""
    assert hasattr(SampledEmpty(0.9), "confidence"), "升格所依据的分数没有落点"
    return "SampledEmpty 带置信度 —— 升格所依据的那个分数有落点"


# ---------------------------------------------------------------------------
# 契约层 §2 几何
# ---------------------------------------------------------------------------
#
# 下面几条要用到 §2 的**新公开名**。名字是规划者给的**建议**（见 T1.2 的补遗），
# 执行者改了名就报一声，这里改一处。找不到名字时报的是干净的 AssertionError，
# 不是 ImportError —— 报告里要能看出"是名字对不上"，而不是"检查炸了"。


def _need(name: str) -> object:
    """取公开面上的一个名字；没有就报出来，并附上当前 __all__。"""
    import ark.contracts as contracts

    found = getattr(contracts, name, None)
    if found is None:
        raise AssertionError(
            f"公开面里没有「{name}」；当前 __all__ = {sorted(contracts.__all__)}"
        )
    return found


def check_grid_coord_keeps_its_fraction() -> str:
    """§2：格坐标是连续的 —— 整数部分是格，小数部分是**格内偏移**。"""
    coord = _need("GridCoord")
    assert coord(5.4, 3.0) != coord(5.0, 3.0), "小数位被丢掉了：5.4 与 5.0 相等"
    assert coord(5.4, 3.0) != coord(5.5, 3.0), "小数位被丢掉了：5.4 与 5.5 相等"
    assert coord(5.0, 3.0) != coord(5.5, 3.0)
    return "5.4 / 5.0 / 5.5 互不相等 —— 小数位真的进了值"


def check_grid_and_pixel_cannot_impersonate_each_other() -> str:
    """§2：像素与格坐标**在类型上要分得开**——别让一个 (float, float) 既能当格又能当像素。"""
    grid, pixel = _need("GridCoord"), _need("Pixel")
    assert grid is not pixel, "格坐标与像素是同一个类型"
    g, p = grid(1.0, 1.0), pixel(1.0, 1.0)
    assert not isinstance(p, grid), "一个像素被当成了格坐标"
    assert not isinstance(g, pixel), "一个格坐标被当成了像素"
    assert g != p, "数值相同就相等了 —— 两者应当是两个刻度上的东西"
    return "两个类型、互不 isinstance、数值相同也不相等"


def check_interval_operations_are_shared() -> str:
    """§2：区间要提供「包含」与「相交」两件操作；
    §1：「下界」「上界」与它**共用同一套操作**。"""
    pairs = (
        (Interval, Interval(15, 16)),
        (LowerBound, LowerBound(15)),
        (UpperBound, UpperBound(16)),
    )
    for cls, obj in pairs:
        assert hasattr(obj, "contains"), f"{cls.__name__} 没有 contains"
        assert hasattr(obj, "intersects"), f"{cls.__name__} 没有 intersects"
    return "区间 / 下界 / 上界 三者都提供 contains 与 intersects"


def _needs_method(obj: object, name: str) -> None:
    """缺方法时给一条干净的 AssertionError，别让报告里出现看不懂的 AttributeError。"""
    if not hasattr(obj, name):
        raise AssertionError(f"{type(obj).__name__} 没有 {name}（见 G3）")


def check_contains_on_open_ends() -> str:
    """§2「包含」：区间 ⊇ 点 / 区间 ⊇ 区间。开放端的方向不许搞反。"""
    _needs_method(Interval(15, 16), "contains")
    assert Interval(15, 16).contains(15.5) is True
    assert Interval(15, 16).contains(16.5) is False
    assert LowerBound(15).contains(20) is True, "下界是开口朝上"
    assert LowerBound(15).contains(10) is False
    assert UpperBound(16).contains(10) is True, "上界是开口朝下"
    assert UpperBound(16).contains(20) is False
    return "三个形态的包含判定都对，开放端方向没搞反"


def check_intersects() -> str:
    """§2「相交」：两个区间是否可能指同一个值。"""
    _needs_method(Interval(15, 16), "intersects")
    assert Interval(15, 16).intersects(Interval(15.5, 16.5)) is True
    assert Interval(15, 16).intersects(Interval(16.5, 17)) is False
    assert LowerBound(15).intersects(UpperBound(20)) is True
    assert LowerBound(15).intersects(UpperBound(10)) is False
    return "相交判定对，含两个半开形态之间的一次相交"


def check_region_is_the_sixth_payload() -> str:
    """§2 + 视觉层/原语 §1：`locate` 的答案（二维区域）是**第六种载荷**。

    判据是「载荷的区分是知道得多准，值的种类是知道的是什么」——二维区域表达精度
    （区域越小越准），所以它是**形态**，不是值的种类。出处见 [契约层 §2]。
    """
    region = _need("Region")
    assert region in PAYLOAD_TYPES, "二维区域不在载荷词表里"
    assert len(PAYLOAD_TYPES) == 6, f"载荷应当六支，实际 {len(PAYLOAD_TYPES)}"
    return f"载荷六支：{[c.__name__ for c in PAYLOAD_TYPES]}"


def check_view_is_two_valued_and_excludes_pause() -> str:
    """§2 + [难例采集 §1]：投影有两套，那一维叫「**视图**」（常速 / 慢放）——
    **不是「时间态」**，后者是三值的、含暂停。两者混掉，投影就会去带一个它没有的值。"""
    view = _need("View")
    try:
        members = list(view)
    except TypeError:
        raise AssertionError(
            f"{view.__name__} 不可枚举；期望两个取值（常速 / 慢放）"
        ) from None
    assert len(members) == 2, f"视图应当两个取值，实际 {len(members)}：{members}"
    joined = " ".join(str(m) for m in members).lower()
    assert "暂停" not in joined and "pause" not in joined, f"暂停不是视图：{members}"
    return f"视图两个取值：{members}"


def check_projection_carries_its_view() -> str:
    """§2：**投影类型自带视图标识**——理由与 §3 同构，混用是只有类型能拦住的错。

    按**类型**判，不按字段名判：只要有一个字段的类型是 View 就算数。
    """
    projection, view = _need("Projection"), _need("View")
    fields = getattr(projection, "__dataclass_fields__", None)
    assert fields, f"{projection.__name__} 不是 dataclass，看不到它的字段"
    hit = [n for n, f in fields.items() if view.__name__ in str(f.type)]
    assert hit, (
        f"{projection.__name__} 的字段里没有一个是 {view.__name__}："
        f"{ {n: str(f.type) for n, f in fields.items()} }"
    )
    return f"{projection.__name__}.{hit[0]} 的类型是 {view.__name__}"


def check_projection_require_refuses_the_wrong_view() -> str:
    """§2「投影」：投影自带视图标识，而它的**落点是调用点的一次断言**（`require(view)`）。

    两套混用是**静默且系统性**的错：拿常速那套去算慢放下的部署点击或识别 ROI，
    错得看不出来。所以"要哪一套"必须在运行期被拒一次 —— 带标识却不拒，标识就是装饰。
    """
    projection, view = _need("Projection"), _need("View")
    members = list(view)
    assert len(members) == 2, f"视图应当两个取值：{members}"
    good, bad = members[0], members[1]
    made = projection(good, lambda point: point, lambda point: point)
    assert made.require(good) is made, "require() 没把这一套投影还回来"
    try:
        made.require(bad)
    except (TypeError, AssertionError, ValueError) as caught:
        return (
            f"require({good.value}) 放行；"
            f"require({bad.value}) 被拒（{type(caught).__name__}）"
        )
    raise AssertionError(
        f"要「{bad.value}」那一套，手里是「{good.value}」那一套，require() 却放行了 —— "
        "两套混用会静默出错"
    )


def check_canvas_is_the_capture_resolution() -> str:
    """§2「归一化画布」：那块画布的尺寸 = **捕获分辨率本身**（用户裁定）—— 不取一个
    固定尺寸、**不做缩放**。

    缩放会引入插值误差（这条链的判据是颜色与统计量、置信度要报出去），还会让
    "不吻合即拒绝"变成"先缩放再拒绝，等于已经适配过了"。
    """
    canvas = _need("Canvas")
    assert dataclasses.is_dataclass(canvas), f"{canvas.__name__} 不是 dataclass"
    made = canvas(1920, 864)
    flat = tuple(v for v in dataclasses.astuple(made) if isinstance(v, (int, float)))
    assert 1920 in flat and 864 in flat, f"画布没有原样保留捕获分辨率：{flat}"
    assert canvas(1920, 864) != canvas(1280, 720), "两种分辨率被当成了同一块画布"
    return f"画布原样保留 {flat}；没有缩放成某个固定尺寸"


def check_interval_meets_a_threshold_in_three_ways() -> str:
    """§2「区间进入条件判定 → 结果是三值」：位置是区间时，「敌人到第 3 格」的答案
    **不是真假二值**，而是**一定真 / 一定假 / 判不了**。

    技术上是区间与阈值求交的三种结果：包含 → 一定真；不相交 → 一定假；相交但不包含
    → 判不了。**这一层定"结果是三值"**；"判不了怎么办"归[控制层/暂停条件](../控制层/暂停条件.md) §4。
    """
    _needs_method(LowerBound(3), "contains")
    _needs_method(LowerBound(3), "intersects")
    threshold = LowerBound(3)
    sure_yes = threshold.contains(Interval(3.2, 4.0))
    sure_no = not threshold.intersects(Interval(1.0, 2.5))
    unsure = threshold.intersects(Interval(2.5, 3.5)) and not threshold.contains(
        Interval(2.5, 3.5)
    )
    assert sure_yes, "整段都在阈值之上，却没判成「一定真」"
    assert sure_no, "整段都在阈值之下，却没判成「一定假」"
    assert unsure, "跨在阈值两侧，却没判成「判不了」—— 它被压成了二值"
    return "一定真 / 一定假 / 判不了 三种结果都能区分出来，没有被压成二值"


def check_pixel_rect_is_corner_plus_size() -> str:
    """§2「像素矩形」：四个数是 **角 + 宽高**，不是 left/top/right/bottom —— 后者把
    四个数里塞进两个冗余约束，更容易写出自相矛盾的矩形。**宽高必须为正**。"""
    rect = _need("PixelRect")
    made = rect(1770, 600, 110, 60)
    assert (made.x, made.y, made.width, made.height) == (1770, 600, 110, 60), (
        f"四个数不是「角 + 宽高」：{dataclasses.astuple(made)}"
    )
    _refuses(ValueError, rect, 1770, 600, 0, 60)
    _refuses(ValueError, rect, 1770, 600, 110, -1)
    _refuses(TypeError, rect, 1770, 600, 110, "60")
    return "角 + 宽高；宽或高不为正、或不是数，都拒绝"


# ---------------------------------------------------------------------------
# 契约层 §2 UI 布局常量：设备层与视觉层共用的那张表
# ---------------------------------------------------------------------------


def check_ui_layout_is_a_table_of_named_rects() -> str:
    """§2「UI 布局常量」+ §5：设备层与视觉层都要读界面上那些固定矩形，而两者互不
    import —— 所以常量**住这一层**，类型与值都在这里。形状是「名字 → 像素矩形」的表。"""
    layout, rect = _need("UiLayout"), _need("PixelRect")
    sample = rect(1770, 600, 110, 60)
    made = layout({"费用读数": sample})
    table = made.rects
    assert isinstance(table, Mapping), f"UI 常量不是映射：{type(table).__name__}"
    assert table["费用读数"] == sample, "取出来的矩形不是放进去那个"
    _refuses(TypeError, layout, {"费用读数": (1770, 600, 110, 60)})
    _refuses(TypeError, layout, {1: sample})
    _refuses(TypeError, layout, {"": sample})
    return "一张「名字 → 像素矩形」的表；裸元组 / 非字符串键 / 空名字都进不来"


def check_ui_layout_is_read_only() -> str:
    """§2「UI 布局常量」：**交出去之后谁也不能改一个格子** —— 它是常量，不是一张谁都
    能写的草稿。常量能被就地改写，"常量"这两个字就不成立了。"""
    layout, rect = _need("UiLayout"), _need("PixelRect")
    made = layout({"费用读数": rect(1770, 600, 110, 60)})
    frozen = _refuses(dataclasses.FrozenInstanceError, setattr, made, "rects", {})
    try:
        made.rects["备战席"] = rect(0, 0, 1, 1)
    except TypeError as caught:
        return f"整张表是冻结的（{frozen}）；往表里加一条也被拒（{type(caught).__name__}）"
    raise AssertionError("UI 常量的表可以被改写 —— 那它就不是常量了")


def check_ui_layout_values_are_absolute_pixels() -> str:
    """§2「UI 布局常量」：坐标是**绝对像素**，与画布同一条纪律 —— 不缩放、不归一化。
    所以它跟画布一起定死在这台设备的分辨率上。"""
    layout, rect = _need("UiLayout"), _need("PixelRect")
    made = layout({"费用读数": rect(1770, 600, 110, 60)})
    (kept,) = made.rects.values()
    flat = tuple(v for v in dataclasses.astuple(kept) if isinstance(v, (int, float)))
    assert 1770 in flat and 110 in flat, f"像素矩形的值没原样保留：{flat}"
    assert all(v > 1 for v in flat), f"像素矩形的值像是归一化过的：{flat}"
    return f"矩形原样保留 {flat} —— 1920×864 下的绝对像素，不是 0–1"


# ---------------------------------------------------------------------------
# 契约层 §4 容器
# ---------------------------------------------------------------------------


def check_container_inventory() -> str:
    """§4 那张容器表，逐项都要在公开面上。"""
    wanted = {
        "槽集合": "SlotSet",
        "前提头": "Premise",
        "动作": "Action",
        "轴": "Axis",
        "记录": "Record",
        "观测迹条目": "TraceEntry",
        "成分表": "Composition",
        "存疑清单条目": "DoubtEntry",
        "快照": "Snapshot",
    }
    missing = []
    for chinese, py in wanted.items():
        try:
            _need(py)
        except AssertionError:
            missing.append(f"{chinese}（建议名 {py}）")
    assert not missing, "容器清单缺：" + "、".join(missing)
    return f"{len(wanted)} 样容器都在公开面上"


def check_trace_entry_holds_a_whole_slot() -> str:
    """§4：观测迹条目 = `帧号 + 字段名 + 槽`。
    **三态与形态取不出来单独一栏** —— 它们长在槽里。"""
    entry = _need("TraceEntry")
    fields = getattr(entry, "__dataclass_fields__", None)
    assert fields, f"{entry.__name__} 不是 dataclass"
    assert len(fields) == 3, f"观测迹条目应当是三元组，实际 {len(fields)} 栏：{sorted(fields)}"
    leaked = {n.lower() for n in fields} & {
        "state", "三态", "payload", "载荷", "form", "形态", "取值形态",
        "confidence", "置信度",
    }
    assert not leaked, f"出现了本该长在槽里的并列字段：{leaked}"
    return f"三元组 {sorted(fields)}；三态 / 形态 / 置信度都取不出单列"


def check_snapshot_has_a_header() -> str:
    """§4：快照 = **页眉** + 按块组织的槽集合 + 前提头指针。

    页眉是**补上的**（修正过一次）——[状态层/快照 ⓪] 说它有六项。
    """
    snapshot = _need("Snapshot")
    fields = getattr(snapshot, "__dataclass_fields__", None)
    assert fields, f"{snapshot.__name__} 不是 dataclass"
    assert len(fields) >= 3, f"快照至少三部分（页眉 / 槽集合 / 前提头指针），实际 {sorted(fields)}"
    return f"快照 {len(fields)} 部分：{sorted(fields)}"


def check_axis_is_a_subset_of_record() -> str:
    """§4 + 时序层 §1：轴 = 前提头 + 动作表；记录 = 前提头 + 实际动作表 + 观测迹。
    **「记录 ⊃ 轴」要在类型上成立**：两者都必须带前提头。"""
    axis, record, premise = _need("Axis"), _need("Record"), _need("Premise")
    af = getattr(axis, "__dataclass_fields__", None)
    rf = getattr(record, "__dataclass_fields__", None)
    assert af and rf, "轴或记录不是 dataclass"
    assert len(af) == 2, f"轴应当是两栏，实际 {sorted(af)}"
    assert len(rf) == 3, f"记录应当是三栏，实际 {sorted(rf)}"
    for label, fields in (("轴", af), ("记录", rf)):
        assert any(premise.__name__ in str(f.type) for f in fields.values()), (
            f"{label} 没有带前提头：{ {n: str(f.type) for n, f in fields.items()} }"
        )
    return f"轴 {sorted(af)} ⊂ 记录 {sorted(rf)}，两者都带前提头"


def check_action_is_frame_verb_args() -> str:
    """§4：动作 = `帧号 + 动词 + 参数…`。"""
    action = _need("Action")
    fields = getattr(action, "__dataclass_fields__", None)
    assert fields, f"{action.__name__} 不是 dataclass"
    assert len(fields) >= 3, f"动作至少三栏（帧号 / 动词 / 参数），实际 {sorted(fields)}"
    return f"动作 {len(fields)} 栏：{sorted(fields)}"


def check_snapshot_header_uses_time_state() -> str:
    """§4 + §3：快照页眉的「当前时间态」必须是 `TimeState`，不是裸字符串 ——
    裸字符串允许任何取值，那正是这一层要拦的东西。"""
    header, state = _need("SnapshotHeader"), _need("TimeState")
    fields = getattr(header, "__dataclass_fields__", None)
    assert fields, f"{header.__name__} 不是 dataclass"
    hit = [n for n, f in fields.items() if state.__name__ in str(f.type)]
    assert hit, (
        f"{header.__name__} 里没有一栏的类型是 {state.__name__}："
        f"{ {n: str(f.type) for n, f in fields.items()} }"
    )
    return f"{header.__name__}.{hit[0]} 的类型是 {state.__name__}"


def check_doubt_entry_has_four_fields() -> str:
    """§4：存疑清单条目 = 类别 + 涉及什么（字段 / 来源） + 严重度 + 涉及哪几帧。

    **八类的取值不在这层**（归状态层）—— 这里只查条目的形状。那一条靠
    "这一层不许出现八成员的枚举"这条粗判兜一下：它是个启发式，不是证明。
    """
    import enum

    import ark.contracts as contracts

    entry = _need("DoubtEntry")
    fields = getattr(entry, "__dataclass_fields__", None)
    assert fields, f"{entry.__name__} 不是 dataclass"
    assert len(fields) == 4, f"存疑清单条目应当四栏，实际 {sorted(fields)}"
    eight = [
        name
        for name, obj in vars(contracts).items()
        if isinstance(obj, type)
        and issubclass(obj, enum.Enum)
        and len(getattr(obj, "__members__", {})) == 8
    ]
    assert not eight, f"这一层出现了八成员的枚举，八类不该写死在这里：{eight}"
    return f"四栏 {sorted(fields)}；这一层没有八成员枚举"


# ---------------------------------------------------------------------------
# 契约层 §5 端口与错误
# ---------------------------------------------------------------------------


def _public_names(obj: object) -> list[str]:
    return sorted(n for n in dir(obj) if not n.startswith("_"))


def check_four_ports_exist() -> str:
    """§5：**端口有四个** —— 设备 · 时钟 · 样本 · 资产。"""
    wanted = {"设备": "DevicePort", "时钟": "ClockPort", "样本": "SamplePort", "资产": "AssetPort"}
    missing = []
    for chinese, py in wanted.items():
        try:
            _need(py)
        except AssertionError:
            missing.append(f"{chinese}（建议名 {py}）")
    assert not missing, "端口缺：" + "、".join(missing)
    return f"{len(wanted)} 个端口都在公开面上"


def check_device_port_methods() -> str:
    """§5 + 设备层 §1：**方法集就是轴上的动词表** —— 取一帧 · 四个动词 · 设备事实。"""
    port = _need("DevicePort")
    names = _public_names(port)
    wanted = (
        "grab_frame",
        "deploy",
        "retreat",
        "skill_on",
        "skill_off",
        "device_facts",
    )
    for want in wanted:
        assert want in names, f"设备端口没有 {want}；它有 {names}"
    assert len(names) == len(wanted), (
        f"设备端口应当是这 {len(wanted)} 个方法，实际 {names}"
    )
    return f"{len(wanted)} 个方法：{names}"


def check_device_verbs_take_grid_not_pixel() -> str:
    """§5 + 设备层 §1：动作方法的参数是**格坐标与朝向**，不是像素 ——
    像素在这个入口内部由投影算出（慢放是另一套投影，让每个调用方自己换算，
    等于把那条纪律复制到每个调用点）。"""
    port = _need("DevicePort")
    hints = {
        name: str(getattr(getattr(port, name), "__annotations__", {}))
        for name in ("deploy", "retreat", "skill_on", "skill_off")
    }
    bad = [name for name, hint in hints.items() if "Pixel" in hint]
    assert not bad, f"这些动作方法的签名里出现了像素：{bad} —— {hints}"
    return f"四个动词的签名都不提像素：{sorted(hints)}"


def check_clock_port_gives_wall_clock() -> str:
    """§5 + §3：时钟端口只产出**墙钟毫秒**。**逻辑帧不在这个端口里** ——
    它由费用推算，不是从时钟读来的。"""
    port = _need("ClockPort")
    names = _public_names(port)
    assert len(names) == 1, f"时钟端口应当只有一个方法，实际 {names}"
    lowered = names[0].lower()
    assert "wall" in lowered or "ms" in lowered, f"时钟端口的方法名看不出是墙钟：{names}"
    assert "frame" not in lowered, f"时钟端口里出现了逻辑帧：{names}"
    return f"一个方法：{names[0]}"


def check_sample_port_has_exactly_one_method() -> str:
    """§5：样本端口**方法集 = 一个方法**（落一份样本）——
    三层触发产出的是同一份东西，区别只在"谁触发 · 留几条 · 落在哪个池子里"，
    那些是 `offline/hardcase/` 的策略，不是边界。"""
    port = _need("SamplePort")
    names = _public_names(port)
    assert len(names) == 1, f"样本端口应当恰好一个方法，实际 {names}"
    return f"一个方法：{names[0]}"


def check_asset_port_has_only_the_projection_method() -> str:
    """§5：资产端口的**方法集由第一个用户层补**（补法规则：只加方法、不改端口
    本身）。现在有第一个：`projection` —— 它的用户层是设备层（§1 说像素在那个
    入口内部由投影算出），而投影值住在 `offline/`，`engine` 永远不 import 它。"""
    port = _need("AssetPort")
    names = _public_names(port)
    assert names == ["projection"], (
        f"资产端口现在应当只有 projection 这一个方法，实际 {names} —— "
        "再加方法要走补法规则"
    )
    return "一个方法：projection（补法规则的第一个）"


def check_error_model_is_three_without_partial() -> str:
    """§5：错误模型三种（拒绝 / 降级 / 中止）。
    **铁律：错误集里不许有"部分成功"这一支** —— 工具不报"我尽力了"，只报三态。"""
    import ark.contracts as contracts

    kinds = [_need("Refusal"), _need("Degradation"), _need("Abort")]
    assert len(set(kinds)) == 3, "三种错误必须是三个类型，不许合成一个"
    banned = [
        n
        for n in contracts.__all__
        if any(w in n.lower() for w in ("partial", "success", "succeed"))
    ]
    assert not banned, f"公开面上出现了「部分成功」这类名字：{banned}"
    return "拒绝 / 降级 / 中止 三个类型；公开面上没有「部分成功」"


CHECKS: tuple[Check, ...] = (
    Check("S0.1", "契约层「这份文件负责什么」", "公开面是一份列得出来的名单", check_public_surface),
    Check("S1.1", "契约层 §1", "三态三支、载荷六支，两套词表不相交", check_two_vocabularies),
    Check("S1.2", "契约层 §1", "「未采样」≠「有值 + 未知」，造不出混淆", check_unsampled_is_not_valued_unknown),
    Check("S1.3", "契约层 §1", "造不出「精确」而值是区间", check_exact_cannot_hold_a_payload),
    Check("S1.4", "契约层 §1", "形态不是并列字段，是载荷自己的类型", check_form_is_not_a_parallel_field),
    Check("S1.5", "契约层 §1", "「未知」是有值的一种载荷，不是三态的一支", check_unknown_is_a_payload_not_a_state),
    Check("S1.6", "契约层 §1", "只有「有值」才可能带载荷", check_only_valued_has_a_payload),
    Check("S1.7", "契约层 §1", "区间两端必填；下界/上界是它的两个半开形态", check_interval_and_the_two_half_open_forms),
    Check("S1.8", "契约层 §1 + 视觉层/原语 §1", "「精确」的值域是数/布尔/字符串", check_exact_value_domain),
    Check("S1.9", "契约层 §2", "区间/下界/上界只收数", check_numeric_payloads_reject_bool_and_str),
    Check("S1.10", "契约层 §1", "置信度只有「读」有，None = 不是读来的", check_confidence_only_for_reads),
    Check("S1.11", "契约层 §1", "三态与载荷的字段一个默认值都没有", check_no_defaults_anywhere),
    Check("S3.1", "契约层 §3", "两个刻度是两个类型", check_scales_are_distinct_types),
    Check("S3.2", "契约层 §3", "跨刻度构造被拒", check_cross_scale_construction_is_refused),
    Check("S3.3", "契约层 §3", "跨刻度二元运算被拒", check_cross_scale_binary_ops_are_refused),
    Check("S3.4", "契约层 §3", "跨刻度不相等、大小比较抛错", check_cross_scale_equality_and_order),
    Check("S3.5", "契约层 §3 + 时间基 §2", "1 游戏秒 = 30 逻辑帧", check_game_second_is_thirty_frames),
    Check("S3.6", "契约层 §3 + §2", "时间态三值，且与视图是两个类型", check_time_state_is_three_valued),
    Check("G1", "契约层 §2", "格坐标是连续的，小数位不丢", check_grid_coord_keeps_its_fraction),
    Check("G2", "契约层 §2", "像素与格坐标在类型上分得开", check_grid_and_pixel_cannot_impersonate_each_other),
    Check("G3", "契约层 §2 + §1", "区间 / 下界 / 上界 共用同一套操作", check_interval_operations_are_shared),
    Check("G4", "契约层 §2", "「包含」对开放端的方向没搞反", check_contains_on_open_ends),
    Check("G5", "契约层 §2", "「相交」判定正确", check_intersects),
    Check("G6", "契约层 §2 + 视觉层/原语 §1", "二维区域是第六种载荷", check_region_is_the_sixth_payload),
    Check("G7", "契约层 §2 + 采集与验收/难例采集 §1", "视图两值，且不含暂停", check_view_is_two_valued_and_excludes_pause),
    Check("G8", "契约层 §2", "投影类型自带视图标识", check_projection_carries_its_view),
    Check("G9", "契约层 §2", "require() 拒绝对不上的视图", check_projection_require_refuses_the_wrong_view),
    Check("G10", "契约层 §2", "画布就是捕获分辨率本身，不缩放", check_canvas_is_the_capture_resolution),
    Check("G11", "契约层 §2", "区间对阈值是三种结果，不是二值", check_interval_meets_a_threshold_in_three_ways),
    Check("G12", "契约层 §2", "像素矩形是「角 + 宽高」，宽高为正", check_pixel_rect_is_corner_plus_size),
    Check("U1", "契约层 §2 + §5", "UI 布局常量是「名字 → 像素矩形」的表", check_ui_layout_is_a_table_of_named_rects),
    Check("U2", "契约层 §2", "UI 布局常量交出去之后改不了", check_ui_layout_is_read_only),
    Check("U3", "契约层 §2", "UI 常量的值是绝对像素，不是归一化", check_ui_layout_values_are_absolute_pixels),
    Check("C1", "契约层 §4", "九样容器都在公开面上", check_container_inventory),
    Check("C2", "契约层 §4 + 时序层 §5", "观测迹条目是三元组，槽不许拆开", check_trace_entry_holds_a_whole_slot),
    Check("C3", "契约层 §4 + 状态层/快照 ⓪", "快照带页眉", check_snapshot_has_a_header),
    Check("C4", "契约层 §4 + 时序层 §1", "轴 ⊂ 记录，两者都带前提头", check_axis_is_a_subset_of_record),
    Check("C5", "契约层 §4", "动作 = 帧号 + 动词 + 参数", check_action_is_frame_verb_args),
    Check("C6", "契约层 §4", "存疑清单条目四栏，八类不在这层", check_doubt_entry_has_four_fields),
    Check("C7", "契约层 §4 + §3", "快照页眉的时间态是 TimeState，不是裸字符串", check_snapshot_header_uses_time_state),
    Check("P1", "契约层 §5", "四个端口都在公开面上", check_four_ports_exist),
    Check("P2", "契约层 §5 + 设备层 §1", "设备端口六个方法（取帧 · 四个动词 · 设备事实）", check_device_port_methods),
    Check("P3", "契约层 §5 + 设备层 §1", "四个动词收格坐标，不收像素", check_device_verbs_take_grid_not_pixel),
    Check("P4", "契约层 §5 + §3", "时钟端口只给墙钟毫秒", check_clock_port_gives_wall_clock),
    Check("P5", "契约层 §5", "样本端口恰好一个方法", check_sample_port_has_exactly_one_method),
    Check("P6", "契约层 §5（补法规则）", "资产端口只有 projection 这一个方法", check_asset_port_has_only_the_projection_method),
    Check("P7", "契约层 §5", "错误三种，没有「部分成功」", check_error_model_is_three_without_partial),
    Check("J1", "契约层「这一层唯一的验收判据」", "哪几个数是猜的，必须能从类型里看出来", check_the_one_acceptance_criterion),
    Check("J2", "契约层 §1", "未命中升格成「空」要有分数落点", check_promotion_needs_a_score),
)


def main() -> int:
    failures: list[tuple[Check, str]] = []
    print(f"契约层规格符合性验收 —— {len(CHECKS)} 条\n")
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
