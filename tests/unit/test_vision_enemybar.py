
"""敌人血条那条链的单元测试。

**不 import tests/spec**（D-全局-17）。这里测的是**这一条链自己的形状与边界**：
整张战场怎么一次筛完、两条分数线落在哪段空档里、三个出口各落什么形态、
我方那两根条怎么被排除、以及**每一档降级各落什么形态**。

## 这一份盯得最紧的是三件事

1. **「认不出」与「不在」分得开**。中间那一带落 `Unsampled()`，**绝不是
   `Exact(False)`** —— 把「认不出」读成「不在」正是《识别方法》「存在性搜索」
   那一行标了警告的漏检方向。
2. **两条硬负例必须报「不在」**：`shot26` 那根**发光的栏杆**与 `S5-0-0` 那几段
   **土红栈道边**。它们正是「判据不许只靠红扁条」那根骨头 —— 这两条用例要是红了，
   说明判据被放宽了，**那要停下来问，不许就地把它放过去**。加了红族之后这一条更要
   盯紧：`shot26` 里那块 8 列的红块**过得了色的门**，挡它的是红族的宽度门。
3. **排除我方条要判「重合」不判「交叠」**：敌人条可以整段压在我方条的行带上
   （`raw_b` 那根就压了 8 行、列还重了 17 列），按「交叠」剔会把真敌条剔掉。
4. **形状门只收紧**：矩形（每一列都亮着的那几行要装下整块 `RECT_FILL_MIN` 的像素）
   与内部均匀（那 8 行逐行中位色的极差 <= `RECT_ROW_SPREAD_MAX`）。
5. **比例的分母是按族查表的两档常量**（用户 10-15 的裁定）：橙族 -> 小怪档
   `MOB_FILL_COLUMNS = 123`、红族 -> boss 档 `BOSS_FILL_COLUMNS = 154` ——
   **分母不来自画面**（量轨道那一套整个退役）。这一条要是红了，先看那两根点名的条
   （`在_boss_红宽条` 的红条 115/154 = 0.7468 与 24 列那根 24/123 = 0.1951）。
6. **三支便宜的降级只降不升**：填充段**长过满条长** -> 条数按 `ceil` 降成下界、
   那一根不给比例；**条身行数与这一档对不上** -> 不给比例（防「橙条 boss」）；
   **填充段短于 `MIN_FILL_COLUMNS`** -> 条照报在、比例不给。

## 合成用例是在真帧之外造的

真帧里没有「中间带那一档」「填充段长过满条长那一档」「敌条短于 12 列」这几档，
它们如实写在 `tests/帧样本/敌人血条/README.md` 的「没覆盖到什么」里。所以那几支用
合成帧钉：一块空地色 + 画上去的一根条，**判据的形状与「谁画的」无关**，所以画出来
的那一支是真的走了一遍那条分支。**合成用例一条都不许混进真值**。

## 重活只跑一次

一帧解出来是 14 MB，9 条。所以**整批只跑一遍、装在模块级夹具里**，用例只读结果。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ark.contracts import (
    Exact,
    Interval,
    LowerBound,
    PixelRect,
    SlotValue,
    Unsampled,
    Valued,
)
from ark.engine.vision import enemybar
from ark.engine.vision.frame import CANVAS_2800x1260, Frame, frame_from_raw
from ark.offline.bench.batch import load_batch
from ark.offline.bench.feed import replay_frames

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "tests" / "帧样本" / "敌人血条"
HARVEST = BATCH / "取材.json"
REPORT = BATCH / "判据读数.json"

HOUND = "不在_cd01"
RAILING = "反例_shot26_栏杆"
BOARDWALK = "反例_S5_栈道边_上"

BASE = (96, 104, 96)
"""合成帧的地形底色：离 ENEMY_FILL 很远，所以筛出来只有**画上去的那根条**。"""


@pytest.fixture(scope="module")
def batch():
    return load_batch(BATCH)


@pytest.fixture(scope="module")
def harvest() -> dict:
    return json.loads(HARVEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report() -> dict:
    return json.loads(REPORT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def readings(batch, harvest) -> dict[str, dict]:
    """整批跑一遍：帧名 -> 三个出口 + 逐根 + 逐块。**只跑一次。**"""
    ours = {
        item["file"].rsplit(".", 1)[0]: [
            PixelRect(*rect) for rect in item.get("our_bars", [])
        ]
        for item in harvest["crops"]
    }
    out: dict[str, dict] = {}
    for case, raw in replay_frames(batch):
        frame = frame_from_raw(raw)
        result = enemybar.read_enemy_bars(frame, our_bars=ours[case.name])
        out[case.name] = {
            "result": result,
            "cores": enemybar.screened_cores(frame, our_bars=ours[case.name]),
        }
    return out


# ---------------------------------------------------------------------------
# 四类真值：逐帧判决
# ---------------------------------------------------------------------------


def _presence(entry) -> str:
    slot = entry["result"].presence
    if isinstance(slot, Unsampled):
        return "认不出"
    assert isinstance(slot, Valued), slot
    return "在" if slot.payload.value else "不在"


def test_every_crop_reads_the_way_the_picture_looks(readings, report) -> None:
    """**逐帧判决**：这一批 9 块，每一块链条读出来的话与 `取材.json` 里那一栏一致。"""
    truth = {item["name"]: item["truth"] for item in report["reads"]}
    assert len(truth) == 9
    for name, want in truth.items():
        assert _presence(readings[name]) == want, name


def test_both_directions_are_counted(readings, report) -> None:
    """**两个方向都点**：四类真值各自数一遍，「在」与「不在」都要各有几块。"""
    truth = {item["name"]: item["truth"] for item in report["reads"]}
    got = {name: _presence(readings[name]) for name in truth}
    assert sum(1 for name in truth if got[name] == "在") == 4
    assert sum(1 for name in truth if got[name] == "不在") == 5
    assert sum(1 for name in truth if got[name] == "认不出") == 0
    assert all(got[name] == truth[name] for name in truth)


def test_the_three_columns_match_the_recorded_reading(readings, report) -> None:
    """**三栏与落盘的那份读数逐条一致**：逐根 / 一堆 / 未采样各自数一遍。

    （这一层不另给一个计数：条数是那三栏算出来的，见下面「条数」那一组。）
    """
    recorded = {item["name"]: item for item in report["reads"]}
    for name, entry in readings.items():
        result = entry["result"]
        assert len(result.bars) == len(recorded[name]["bars"]), name
        assert len(result.clumps) == len(recorded[name]["clumps"]), name
        assert len(result.unsampled) == len(recorded[name]["unsampled"]), name


def test_the_railing_is_not_a_bar(readings) -> None:
    """**硬负例一**：`shot26` 那根发光的地形栏杆 —— 报「不在」。"""
    entry = readings[RAILING]
    assert _presence(entry) == "不在"
    assert entry["result"].bars == ()
    assert entry["result"].cores == ()


def test_the_boardwalk_edges_are_not_bars(readings) -> None:
    """**硬负例二**：`S5-0-0` 那几段土红栈道边 —— 报「不在」（上下两块都点）。"""
    for name in ("反例_S5_栈道边_上", "反例_S5_栈道边_下"):
        entry = readings[name]
        assert _presence(entry) == "不在", name
        assert entry["result"].bars == (), name


def test_a_full_health_enemy_has_no_bar(readings) -> None:
    """`cd01` / `cd30`：敌人站在画面上、一根条都没有 —— 报「不在」而不是「认不出」。"""
    for name in ("不在_cd01", "不在_cd30"):
        entry = readings[name]
        assert _presence(entry) == "不在", name
        assert entry["result"].bars == (), name
    assert _presence(readings[HOUND]) == "不在"


def test_the_present_crops_report_the_origin_of_their_bar(readings) -> None:
    """「在」那四块各自读出一件东西：**原点**都对得上源帧（条数那件事另有一组用例）。

    `在_raw_b_上` 那一块落的是**一堆**（130 列 > 123），所以它的落点读
    `clumps[0].region`；另外三块是**逐根**，读 `bars[0].fill_rect`。
    """
    for name, x, y in (
        ("在_raw_a_左", 1320, 517),
        ("在_raw_a_右", 1677, 480),
        ("在_raw_b_下", 1438, 634),
    ):
        bars = readings[name]["result"].bars
        assert len(bars) == 1, name
        assert (bars[0].fill_rect.x, bars[0].fill_rect.y) == (x, y), name
    clumps = readings["在_raw_b_上"]["result"].clumps
    assert len(clumps) == 1
    assert (clumps[0].region.x, clumps[0].region.y) == (1546, 481)



# ---------------------------------------------------------------------------
# 两条线落在哪段空档里
# ---------------------------------------------------------------------------


def test_the_two_colour_lines_sit_inside_the_measured_gap(report) -> None:
    """两条线落在**量出来的空档**里，两侧余量报得出来。

    全库 176 帧按松门（60）扫一遍，把所有「8 行高的橙块」的中位色距做成分布：
    真条 **1…12**（56 块）、非条那一堆从 **34** 起**密起来**、中间 13…33 **稀疏**
    （21 块）。所以线取 `SOLID_TOL=16` / `ABSENT_TOL=28` —— 离两簇各留 **4 与 6**。
    这一批里的四块全落在真条那一簇里（最大 3）。
    """
    assert enemybar.SOLID_TOL == 16
    assert enemybar.ABSENT_TOL == 28
    assert enemybar.SOLID_SCORE > enemybar.ABSENT_SCORE
    distances = [
        core["colour_distance"] for item in report["reads"] for core in item["cores"]
    ]
    assert len(distances) == 4, "这批该有四块（四块「在」各一块）"
    assert max(distances) <= 3


def test_the_three_exits_are_three_not_two() -> None:
    """**三个出口同款**：中间那一带落 `Unsampled()`，与「不在」在类型上分得开。

    够高（== ENEMY_FILL 本身）-> 在；中间（+22）-> 认不出；够低（+60）-> 不在。
    """
    solid = _frame_with_bar(300, 300, 60, 40, _shift(0))
    middle = _frame_with_bar(300, 300, 60, 40, _shift(22))
    away = _frame_with_bar(300, 300, 60, 40, _shift(60))
    assert _presence({"result": enemybar.read_enemy_bars(solid)}) == "在"
    assert _presence({"result": enemybar.read_enemy_bars(middle)}) == "认不出"
    assert _presence({"result": enemybar.read_enemy_bars(away)}) == "不在"
    # 够低那一支是**筛都没筛出来**（形状门的颜色门一起过的），不是「筛出来再否掉」。
    assert enemybar.read_enemy_bars(away).cores == ()
    assert enemybar.read_enemy_bars(middle).cores != ()


def test_a_middle_band_core_never_becomes_exact_false() -> None:
    """中间那一带的块**不许**落成 `Exact(False)` —— 那是漏检方向。"""
    slot = enemybar.read_enemy_bars(_frame_with_bar(400, 300, 40, 0, _shift(22))).presence
    assert isinstance(slot, Unsampled)
    assert not isinstance(slot, Valued)


# ---------------------------------------------------------------------------
# 填充比例：逐根，以及每一档降级
# ---------------------------------------------------------------------------


def test_a_clean_bar_reads_its_ratio_exactly() -> None:
    """填 60 列 -> 精确 **60/123**（分母是那一档的常量，不是画面上量到的那一截）。

    干净的那一根：**列数**与**比例**两条读数都落精确，各自带自己的余量。
    """
    frame = _frame_with_bar(300, 300, 60, 40, enemybar.ENEMY_FILL)
    bar = enemybar.read_enemy_bars(frame).bars[0]
    assert bar.fill_rect.width == 60
    assert isinstance(bar.columns.value, Valued)
    assert bar.columns.value.payload == Exact(60)
    assert bar.columns.margin is not None
    assert isinstance(bar.fill.value, Valued)
    assert bar.fill.value.payload == Exact(60 / enemybar.MOB_FILL_COLUMNS)



def test_a_full_bar_reads_one() -> None:
    """**填满那一档**（123 列）-> 精确 1.0。"""
    full = enemybar.MOB_FILL_COLUMNS
    frame = _frame_with_bar(300, 300, full, 0, enemybar.ENEMY_FILL)
    bar = enemybar.read_enemy_bars(frame).bars[0]
    assert bar.fill_rect.width == full
    assert isinstance(bar.fill.value, Valued)
    assert bar.fill.value.payload == Exact(1.0)



def test_the_denominator_comes_from_the_dial_not_from_the_picture() -> None:
    """**分母不来自画面**（用户 10-15 的裁定）：右边那一段亮不亮、多宽，都不影响比例。

    同一根**填 100 列**的条，右边画 0 列、40 列、200 列「没填到的那一段」——
    三次都必须读到 **100/123 = 0.8130**。老那一版把「填充 + 空轨」当整长，
    同一个画面能读出 0.7143…1.0 一串数，那正是这一版退役掉的东西。
    """
    want = Exact(100 / enemybar.MOB_FILL_COLUMNS)
    for track in (0, 40, 200):
        frame = _frame_with_bar(300, 300, 100, track, enemybar.ENEMY_FILL)
        bar = enemybar.read_enemy_bars(frame).bars[0]
        assert bar.fill_rect.width == 100, track
        assert isinstance(bar.fill.value, Valued), track
        assert bar.fill.value.payload == want, track



def test_a_fill_measured_across_a_blend_is_only_a_lower_bound() -> None:
    """**填充段中间被压过一截 -> 比例只给下界**（用户 10-15，接过老那一支的位置）。

    画一根填 60 列的条，正中间那 2 列换成**混色**（离填充色 58 > `SCREEN_TOL`，
    可它离上下那两块地形更远 —— 也就是 `_blended` 认得的那种「被半透明的东西压过」）。
    `_merge` 会把两块接回一根，而**填充段**那一趟容得下跨过 `FILL_GAP`（= 2）列，
    所以量到的填充段是 60 列、**严格那一段**只有 50：缝底下藏着什么看不见，
    只能说「**至少** 50/123」。这与我方血条那条链的 LowerBound 是同一套纪律。

    **两根读数的口径在这一支上刻意不同**（用户 10-15：「填充总列数 = 那个读得准的
    数」）：`columns`（这一段多宽）是 **60**（缝是压在条上的半透明东西，那两列算在
    里头），`fill`（填了多少）是 **≥50**（保守）。而且**两条读数各带自己的余量** ——
    缝那一列把 `columns` 的余量压到 0，`fill` 的余量量的是缝前那一段的门。
    """
    pixels = np.full((1260, 2800, 3), BASE, np.uint8)
    pixels[300:308, 300:360] = enemybar.ENEMY_FILL
    pixels[300:308, 350:352] = (160, 100, 60)
    frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    bar = enemybar.read_enemy_bars(frame).bars[0]
    assert bar.fill_rect.width == 60
    assert isinstance(bar.columns.value, Valued)
    assert bar.columns.value.payload == Exact(60)
    assert bar.columns.margin == 0.0, "缝那一列就在门上，余量该是 0"
    assert isinstance(bar.fill.value, Valued)
    assert bar.fill.value.payload == LowerBound(50 / enemybar.MOB_FILL_COLUMNS)
    assert bar.fill.margin is not None and bar.fill.margin > 0.0



def test_what_is_right_of_the_fill_is_never_counted() -> None:
    """**「没填到的那一段」一列都不算**：它亮也好暗也好，都改不了比例。

    右边画一段**比底色亮**的带子（老那一版靠「与地形差多少」去量它 ——
    `raw_b` 那张纯灰 177 的带子正是这一档）。这一版连看都不看它：
    比例只由填充段列数决定。
    """
    pixels = np.full((1260, 2800, 3), BASE, np.uint8)
    bright = np.clip(np.asarray(BASE, np.int16) + 81, 0, 255).astype(np.uint8)
    pixels[300:308, 300:360] = enemybar.ENEMY_FILL
    pixels[300:308, 360:400] = bright
    frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    bar = enemybar.read_enemy_bars(frame).bars[0]
    assert bar.fill_rect.width == 60
    assert isinstance(bar.fill.value, Valued)
    assert bar.fill.value.payload == Exact(60 / enemybar.MOB_FILL_COLUMNS)



def test_the_bar_body_is_measured_not_hard_coded() -> None:
    """**条身取量到的行数，不写死 8**（用户 10-14 的裁定 2 + 10-15 的跨档校验）。

    画一根 **10 行**的**红**条（boss 那一档就是 10 行）：填充 60 列。量到的条身该是
    **10 行**（不是写死的 8），比例 60/154 —— 这一根的量法**对得上它那一档**，
    所以比例照给（橙色的 10 行那一根不给，见下一条）。
    """
    pixels = np.full((1260, 2800, 3), BASE, np.uint8)
    pixels[300:310, 300:360] = enemybar.ENEMY_FILL_RED
    frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    region, top = enemybar._field(frame)
    core = enemybar._screen(region, top)[0]
    row, rows = enemybar._body_rows(region, core)
    assert (row - core.y, rows) == (0, 10)
    bar = enemybar.read_enemy_bars(frame).bars[0]
    assert bar.fill_rect.width == 60
    # 裁定 2：fill_rect 的高也是**量到的**（这条合成条是 10 行，不是写死的 8）。
    assert (bar.fill_rect.y, bar.fill_rect.height) == (300, 10)
    assert isinstance(bar.fill.value, Valued)
    assert bar.fill.value.payload == Exact(60 / enemybar.BOSS_FILL_COLUMNS)


def test_an_orange_bar_with_boss_rows_gets_no_ratio() -> None:
    """**跨档校验**（用户 10-15 那道便宜闸门）：橙条量到 10 行 -> 不给比例。

    橙族走小怪档（8 行），**量到 10 行比这一档更高**，就是「这一根的量法与它那一档
    对不上」—— 它可能是**橙色的 boss**（那一档是 154 列），拿 123 当分母会系统性
    报高。这一支**只降不升**：条照报「在」、填充列数照量，这一根落 `unsampled`
    （比例那一条读数落未采样）。

    **它只拦更高的那一头**（用户 10-15 的裁定 6）：矮一点的不拦。
    """
    pixels = np.full((1260, 2800, 3), BASE, np.uint8)
    pixels[300:310, 300:360] = enemybar.ENEMY_FILL
    frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    result = enemybar.read_enemy_bars(frame)
    assert result.bars == ()
    assert len(result.unsampled) == 1
    bar = result.unsampled[0]
    assert bar.fill_rect.height == 10
    assert isinstance(bar.fill.value, Unsampled)
    assert bar.columns.value.payload == Exact(60), "列数照量，读不出来的是「填了多少」"
    assert result.count.payload == Exact(1)
    assert enemybar.MOB_BODY_ROWS != 10
    assert enemybar.BOSS_BODY_ROWS == 10



def test_a_fill_longer_than_the_dial_is_more_than_one_bar() -> None:
    """**一道便宜的算术门**（用户 10-15）：填充段长过满条长 -> **这一坨是一堆**。

    画一根填 **130 列**的橙条：一根小怪的填充不可能长过 123 列，所以这一块落成
    **一堆**，它的**至少几根** = `ceil(130/123)` = **2** —— 不需要任何「把两根切开来」
    的能力。**这一堆里没有单根比例**（硬给就是编，「填了 130 列」这件事本身就说明
    它不止一根）。
    """
    frame = _frame_with_bar(300, 300, 130, 0, enemybar.ENEMY_FILL)
    result = enemybar.read_enemy_bars(frame)
    assert result.bars == (), "一堆里不许出现单根比例"
    assert len(result.clumps) == 1
    clump = result.clumps[0]
    assert clump.tier == enemybar.MOB_TIER
    assert clump.columns.value.payload == Exact(130)
    assert clump.least.value.payload == LowerBound(2)
    assert result.count.payload == LowerBound(2)



def test_the_measured_crops_show_the_fill_forms(readings) -> None:
    """真帧上这几支都出现过：**精确**（raw_a 那两根 84/123、raw_b 下面那根 57/123）·
    **一堆**（raw_b 上面那 130 列 > 123）· **未采样**（旧批这一批里没有——它由
    10-14 那两帧钉）。

    **下界**那一支由合成用例
    （`test_a_fill_measured_across_a_blend_is_only_a_lower_bound`）与 10-14 的互叠
    那两帧钉。`Interval` 那一档这一版**没有来路**（见判据模块开头「三样读法」）。
    """
    forms = {}
    for name, entry in readings.items():
        for bar in entry["result"].bars:
            forms.setdefault(name, set()).add(
                "Unsampled"
                if isinstance(bar.fill.value, Unsampled)
                else type(bar.fill.value.payload).__name__
            )
    assert forms["在_raw_a_左"] == {"Exact"}
    assert forms["在_raw_a_右"] == {"Exact"}
    assert forms["在_raw_b_下"] == {"Exact"}
    assert readings["在_raw_b_上"]["result"].bars == ()
    assert len(readings["在_raw_b_上"]["result"].clumps) == 1



def test_the_measured_fill_of_raw_a_is_the_picture_truth(readings) -> None:
    """`raw_a` 那两根的填充段都是 **84 列** -> 精确 **84/123 = 0.6829**。

    画面事实：那两根的填充段都是 84 列（x1320…x1403 的中位色 (220,109,57)；
    x1404 是 (87,102,83)，逐通道差 132 > `SCREEN_TOL`）。老那一版把这一支读成
    下界 / 区间（0.6512、0.677…0.683）—— 它们得先**量出**整长才敢给比例。
    这一版的分母是常量，所以两根都给**精确的同一个数**：它们本来就是同一档。
    """
    for name in ("在_raw_a_左", "在_raw_a_右"):
        bar = readings[name]["result"].bars[0]
        assert bar.fill_rect.width == 84, name
        assert isinstance(bar.fill.value, Valued), name
        assert bar.fill.value.payload == Exact(84 / enemybar.MOB_FILL_COLUMNS), name



def test_the_raw_b_top_core_is_a_clump_of_at_least_two(readings) -> None:
    """`raw_b` 上面那一块：填充段 **130 列 > 123** -> **一堆**（用户 10-15）。

    画面事实（用户裁定）：那 130 列是**两根的填充首尾相接**，真值是**两根**。
    链**不要求切开它**（用户原话：是否落成两根「有待讨论，要考虑实现难度」）——
    只要把它报成**一堆**：**区域 + 至少 2 根 + 填充总列数**；而且**这一堆里
    没有单根比例**（**不许给 130/123 = 1.06** 这种精确值，> 1 本来也不该出现）。

    用户那道便宜的算术门就是这一支：**填充总列数 > 满条长 ⇒ 这里不止一根**，
    `ceil(130/123)` = **2**。
    """
    result = readings["在_raw_b_上"]["result"]
    assert result.bars == ()
    assert len(result.cores) == 1 and len(result.clumps) == 1
    clump = result.clumps[0]
    assert clump.tier.name == "小怪" and clump.tier.full_columns == 123
    assert clump.columns.value.payload == Exact(130)
    assert clump.least.value.payload == LowerBound(2)
    assert clump.region.width == 130
    assert isinstance(result.count, Valued)
    assert isinstance(result.count.payload, LowerBound)
    assert result.count.payload.lo >= 2



def test_a_one_column_seam_does_not_split_the_core(readings) -> None:
    """`raw_b` 上面那一块**筛出来还是一块**：正中间那条 1 列宽的暗缝由 `_merge` 接回。

    接回之后它是 **130 列的一整块**（`cores` 也是 1 块）—— **接回不等于「它是一根
    条」**：130 > 123 说明这里**不止一根**（见上一条）。那条缝本身只是用户 10-15
    说的**分不开的边界候选**，**不许当精确切点**。
    """
    entry = readings["在_raw_b_上"]
    assert len(entry["result"].cores) == 1
    assert entry["result"].cores[0].fill_rect.width == 130



# ---------------------------------------------------------------------------
# 排除我方那两根条
# ---------------------------------------------------------------------------


def test_our_own_bar_next_to_an_enemy_bar_does_not_eat_it(readings, harvest) -> None:
    """**实机那种几何**：`raw_a` 那根敌条压在我方血条右端上（行重 3 行）-> 不许被剔。

    这一条要是红了，就说明排除变成了「按交叠剔」—— 那会把真敌条剔掉。
    """
    entry = readings["在_raw_a_右"]
    assert entry["result"].bars, "我方条把敌条吃掉了"
    kept = [core for core in entry["result"].cores if not core.masked]
    assert len(kept) == 1
    assert kept[0].fill_rect.x == 1677


def test_the_raw_b_overlap_is_not_a_coincidence() -> None:
    """`raw_b` 那根敌条的**行带整段**压在我方条上（8 行）、列还重 17 列 -> 仍不许被剔。

    这一块落的是**一堆**（130 列 > 123），所以落点读 `clumps[0].region`。
    """
    frame = _load_source_frame(BATCH, "在_raw_b_上")
    ours = [PixelRect(1417, 476, 146, 8)]
    result = enemybar.read_enemy_bars(frame, our_bars=ours)
    assert len(result.clumps) == 1
    assert result.clumps[0].region.x == 1546
    assert result.bars == ()
    assert all(not core.masked for core in result.cores)


def test_a_core_that_sits_on_our_bar_is_dropped() -> None:
    """一块**正好踩在我方血条那 8 行**上、列也在那一格里的橙块 -> 判成我们自己的。"""
    frame = _load_source_frame(BATCH, "在_raw_a_右")
    enemy = PixelRect(1677, 480, 146, 8)
    result = enemybar.read_enemy_bars(frame, our_bars=[enemy])
    assert any(core.masked for core in result.cores)
    assert all(bar.fill_rect.x != 1677 for bar in result.bars)


def test_the_skill_bar_slot_of_our_own_ui_is_also_ours() -> None:
    """我方**技力条**（生效中是橙的）也该被剔：它在那条血条下面第 8 行、共用左缘。"""
    frame = _load_source_frame(BATCH, "在_raw_a_右")
    health = PixelRect(1677, 480, 146, 8)
    result = enemybar.read_enemy_bars(frame, our_bars=[health])
    assert any(core.masked for core in result.cores)
    # 同一个位置再往上挪一条血条的高度：那一块就落在我方**技力条**那 8 行上了
    shifted = PixelRect(1677, 480 - enemybar.SKILL_ROW_OFFSET, 146, 8)
    again = enemybar.read_enemy_bars(frame, our_bars=[shifted])
    assert any(core.masked for core in again.cores)


def test_six_rows_away_is_not_our_bar() -> None:
    """行差 6 行（`raw_a` 那根敌条与我方血条的真实关系）-> **不许**判成我们的。"""
    frame = _load_source_frame(BATCH, "在_raw_a_右")
    result = enemybar.read_enemy_bars(frame, our_bars=[PixelRect(1677, 487, 146, 8)])
    assert result.cores, "这一帧本来就该筛出一块"
    assert all(not core.masked for core in result.cores)


def test_our_bars_must_be_rectangles() -> None:
    """`our_bars` 给的不是像素矩形 -> 抛异常（这一层不自己猜干员在哪）。"""
    frame = _load_source_frame(BATCH, "在_raw_a_右")
    with pytest.raises(TypeError):
        enemybar.read_enemy_bars(frame, our_bars=[(1, 2, 3, 4)])  # type: ignore[list-item]


def _bounds(fill: SlotValue) -> tuple[float, float]:
    """一个填充读数的两端：精确 -> (值, 值)；区间 -> (低, 高)；下界 -> (低, 1.0)。"""
    payload = fill.payload
    if isinstance(payload, Exact):
        return float(payload.value), float(payload.value)
    if isinstance(payload, Interval):
        return float(payload.lo), float(payload.hi)
    return float(payload.lo), 1.0


# ---------------------------------------------------------------------------
# 形状门与战场那一段
# ---------------------------------------------------------------------------


def test_a_fill_between_the_two_width_gates_is_a_short_bar() -> None:
    """两道宽度门之间的那一档：**存在性照报、列数照量，填了多少不给**。

    `MIN_CANDIDATE_COLUMNS`（7）是候选的门、`MIN_FILL_COLUMNS`（12）是够不够量
    比例的门 —— 11 落在中间，所以它是一条**条**，只是**填了多少读不出来**：
    这一根落 `unsampled` 那一栏（用户 10-15：「未采样只留给那一根确实有东西、
    但连填充都读不出来」）。
    """
    narrow = enemybar.MIN_FILL_COLUMNS - 1
    assert narrow >= enemybar.MIN_CANDIDATE_COLUMNS
    frame = _frame_with_bar(300, 300, narrow, 40, enemybar.ENEMY_FILL)
    result = enemybar.read_enemy_bars(frame)
    assert len(result.cores) == 1
    assert result.presence.payload == Exact(True)
    assert result.bars == ()
    assert [bar.fill_rect.x for bar in result.unsampled] == [300]
    assert isinstance(result.unsampled[0].fill.value, Unsampled)
    assert result.unsampled[0].columns.value.payload == Exact(narrow)
    assert result.count.payload == Exact(1)


def test_a_fill_below_the_candidate_gate_is_not_a_bar() -> None:
    """窄于 `MIN_CANDIDATE_COLUMNS` 的那一截连候选都不是（现成最短真条是 7 列）。"""
    frame = _frame_with_bar(
        300, 300, enemybar.MIN_CANDIDATE_COLUMNS - 1, 40, enemybar.ENEMY_FILL
    )
    result = enemybar.read_enemy_bars(frame)
    assert result.cores == ()
    assert _presence({"result": result}) == "不在"


def test_a_run_that_is_not_eight_rows_tall_is_not_a_bar() -> None:
    """高不是 8 行的「扁条」不当候选 —— 地貌那些红扁条正是这样被挡住的。"""
    pixels = np.full((1260, 2800, 3), BASE, np.uint8)
    pixels[300:305, 300:420] = enemybar.ENEMY_FILL
    frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    assert enemybar.read_enemy_bars(frame).cores == ()


def test_a_tall_blob_is_not_a_bar() -> None:
    """高过 `MAX_BAR_ROWS` 的橙块不当候选（贴图、红罩、警戒框都是这一类）。"""
    pixels = np.full((1260, 2800, 3), BASE, np.uint8)
    pixels[300:300 + enemybar.MAX_BAR_ROWS + 1, 300:400] = enemybar.ENEMY_FILL
    frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    assert enemybar.read_enemy_bars(frame).cores == ()


def test_the_top_hud_and_the_bottom_bench_are_outside_the_field() -> None:
    """顶部 HUD 与底部备战席都不在战场那一段里 —— 那里画的橙块不进这条链。"""
    for row in (enemybar.FIELD_TOP - 40, enemybar.FIELD_BOTTOM + 20):
        pixels = np.full((1260, 2800, 3), BASE, np.uint8)
        pixels[row : row + 8, 300:400] = enemybar.ENEMY_FILL
        frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
        assert enemybar.read_enemy_bars(frame).cores == (), row


def test_the_field_still_reads_a_bar_just_inside_it() -> None:
    """战场那一段的**下沿那一根**照样读得到（借底那 8 行是给量法用的，不是判据）。"""
    row = enemybar.FIELD_BOTTOM - enemybar.BAR_ROWS
    frame = _frame_with_bar(300, row, 60, 40, enemybar.ENEMY_FILL)
    result = enemybar.read_enemy_bars(frame)
    assert len(result.bars) == 1 and result.bars[0].fill_rect.y == row


# ---------------------------------------------------------------------------
# 两个入口说的是同一件事
# ---------------------------------------------------------------------------


def test_screened_cores_agrees_with_the_reader(readings) -> None:
    """`screened_cores` 与 `read_enemy_bars` 用的是**同一条筛法**（不许各写一份）。"""
    for entry in readings.values():
        assert entry["cores"] == entry["result"].cores


def test_a_frame_is_required() -> None:
    """给的不是一帧 -> 抛异常。"""
    with pytest.raises(TypeError):
        enemybar.read_enemy_bars("not a frame")  # type: ignore[arg-type]


def test_the_readings_file_records_the_ruler(report) -> None:
    """落盘的读数里记着**这一版用的是哪把尺子**（判据、三条门、排除的做法）。"""
    ruler = report["ruler"]
    assert ruler["colour"].startswith("逐像素离 ENEMY_FILL")
    assert str(enemybar.SCREEN_TOL) in ruler["colour"]
    assert ruler["exclusion"].endswith("剔掉")
    assert report["field"]["rows"] == [enemybar.FIELD_TOP, enemybar.FIELD_BOTTOM]


def test_bounds_helper_is_not_a_test() -> None:
    """`_bounds` 这个读比例两端的小工具本身也要对（下面几条用它）。"""
    assert _bounds(Valued(Exact(0.5), None)) == (0.5, 0.5)
    assert _bounds(Valued(Interval(0.2, 0.4), None)) == (0.2, 0.4)
    assert _bounds(Valued(LowerBound(0.2), None)) == (0.2, 1.0)


# ---------------------------------------------------------------------------
# 合成帧的搭法
# ---------------------------------------------------------------------------


def _shift(amount: int) -> tuple[int, int, int]:
    """把 ENEMY_FILL 往每个通道上加同一个数（用来造「中间带」与「够低」两档）。"""
    return tuple(min(255, channel + amount) for channel in enemybar.ENEMY_FILL)


def _frame_with_bar(
    x: int,
    y: int,
    fill: int,
    track: int,
    colour: tuple[int, int, int],
    *,
    rows: int = enemybar.BAR_ROWS,
) -> Frame:
    """一块空地色 + 画上去的一根条：填充 `fill` 列、**没填到的那一段** `track` 列。

    那一段的颜色比底色暗 52（离填充色很远）：它**一列都不算填充**（这一版量的是
    填充段，`track` 只是用来钉「画面右边那一段再亮也不算数」）。`rows` 是画多高，
    用来钉跨档校验（小怪 8 行 / boss 10 行）。
    """
    pixels = np.full((1260, 2800, 3), BASE, np.uint8)
    pixels[y : y + rows, x : x + fill] = colour
    base = np.asarray(BASE, np.int16)
    track_colour = np.clip(base - 52, 0, 255).astype(np.uint8)
    pixels[y : y + rows, x + fill : x + fill + track] = track_colour
    return Frame(canvas=CANVAS_2800x1260, pixels=pixels)


def _load_source_frame(batch_dir: Path, name: str) -> Frame:
    """从这一批里读回**贴回画布**的那一帧（走的是验收那条路）。"""
    batch = load_batch(batch_dir)
    for case in batch.cases:
        if case.name == name:
            return frame_from_raw(case.raw())
    raise AssertionError(name)


# ---------------------------------------------------------------------------
# 10-14 补的那一批：boss 的红宽条 · 快死的敌人 · 压住干员 · 互叠 · 满血
# ---------------------------------------------------------------------------
#
# 这一批的缺口与真值见 tests/帧样本/敌人血条-10-14/README.md。这里只钉**行为**：
# 哪几块报在 / 不在、条数是精确还是下界、分不开的那几根比例落什么形态 ——
# 另外钉住**那两根点名的条按两档常量读对了**（红条 115/154、24 列那根 24/123），
# 以及**跨档校验**（橙条 10 行不给比例）那一道便宜闸门。


BATCH_1014 = REPO / "tests" / "帧样本" / "敌人血条-10-14"
HARVEST_1014 = BATCH_1014 / "取材.json"
REPORT_1014 = BATCH_1014 / "判据读数.json"

RED_GATE = "不在_boss_红门"
"""`boss.png` 右上角那一**道红门**的裁剪 —— 红族最像条的两块都在它里面。"""



@pytest.fixture(scope="module")
def readings_1014() -> dict[str, dict]:
    """10-14 那一批整批跑一遍（7 块）。**只跑一次**（一帧 14 MB）。"""
    harvest = json.loads(HARVEST_1014.read_text(encoding="utf-8"))
    ours = {
        item["file"].rsplit(".", 1)[0]: [
            PixelRect(*rect) for rect in item.get("our_bars", [])
        ]
        for item in harvest["crops"]
    }
    out: dict[str, dict] = {}
    for case, raw in replay_frames(load_batch(BATCH_1014)):
        frame = frame_from_raw(raw)
        result = enemybar.read_enemy_bars(frame, our_bars=ours[case.name])
        out[case.name] = {"result": result, "frame": frame}
    return out


@pytest.fixture(scope="module")
def report_1014() -> dict:
    return json.loads(REPORT_1014.read_text(encoding="utf-8"))


def _red_distance(pixels: np.ndarray) -> np.ndarray:
    """一片像素离 ENEMY_FILL_RED 的逐通道最大差（红族那一支，同一个量纲）。"""
    return np.abs(
        pixels.astype(np.int16) - np.asarray(enemybar.ENEMY_FILL_RED, np.int16)
    ).max(axis=2)


def _colour_distance(pixels: np.ndarray) -> np.ndarray:
    """一片像素离 ENEMY_FILL 的逐通道最大差（与链同一个量纲）。"""
    return np.abs(
        pixels.astype(np.int16) - np.asarray(enemybar.ENEMY_FILL, np.int16)
    ).max(axis=2)


def _longest_vertical_run(mask: np.ndarray) -> int:
    """这一片里**最长的竖直连续**有多长（用来复核形状门挡的是哪一块）。"""
    best = 0
    for column in range(mask.shape[1]):
        run = 0
        for row in range(mask.shape[0]):
            run = run + 1 if mask[row, column] else 0
            best = max(best, run)
    return best


def test_the_10_14_crops_read_the_way_the_readings_file_says(
    readings_1014, report_1014
) -> None:
    """**逐帧判决**：这一批 7 块，链条读出来的话与 `判据读数.json` 里那一栏一致 ——
    三个出口、**条数的形态与那个数**、以及逐根的条数。"""
    recorded = {item["name"]: item for item in report_1014["reads"]}
    assert len(recorded) == 8
    for name, item in recorded.items():
        result = readings_1014[name]["result"]
        assert _presence(readings_1014[name]) == (
            "在" if item["presence"]["payload"]["value"] else "不在"
        ), name
        payload = item["count"]["payload"]
        assert len(result.bars) == len(item["bars"]), name
        assert len(result.clumps) == len(item["clumps"]), name
        assert len(result.unsampled) == len(item["unsampled"]), name
        assert len(result.bars) + len(result.unsampled) + sum(
            int(clump["least"]["value"]["payload"]["lo"]) for clump in item["clumps"]
        ) == (payload["value"] if payload["form"] == "exact" else payload["lo"]), name
        if payload["form"] == "exact":
            assert result.count.payload == Exact(payload["value"]), name
        else:
            assert result.count.payload == LowerBound(payload["lo"]), name


def test_both_directions_are_counted_in_the_10_14_batch(readings_1014) -> None:
    """两个方向都点了数：在 5 · 不在 3 · 认不出 0。

    （红族进来之前是 4 / 3 —— `在_快死的敌人` 那一块从「精确不在」翻了回来。）
    """
    tally = {"在": 0, "不在": 0, "认不出": 0}
    for entry in readings_1014.values():
        tally[_presence(entry)] += 1
    assert tally == {"在": 5, "不在": 3, "认不出": 0}


def test_a_full_health_frame_has_no_bar_at_all(readings_1014) -> None:
    """**全满血那一帧一根都不许有**（用户点名的卡住条件之一）。

    它旁边就有一块**离假阳性只差 1 行**的橙黄警戒线（见下一条），所以这一条不是
    「那里没东西」，是「有东西、但它过不了形状门」。
    """
    result = readings_1014["不在_全满血"]["result"]
    assert result.presence.payload == Exact(False)
    assert result.count.payload == Exact(0)
    assert result.bars == ()
    assert result.clumps == () and result.unsampled == ()
    assert result.cores == ()


def test_the_hazard_stripe_in_the_full_health_frame_is_one_row_too_tall(
    readings_1014,
) -> None:
    """全满血那一块**橙黄警戒线**：进得了颜色门、宽度也够，**只差 1 行**。

    量的是画面事实（那一带的像素），不是读数的转录：中位色距 <= `ABSENT_TOL`、
    竖着连成 **13 行**（`MAX_BAR_ROWS` 是 12）、宽 14 列（`MIN_FILL_COLUMNS` 是 12）。
    **它是这一批离假阳性最近的一块** —— 这一条红了说明形状门被放宽了，停下来问。
    """
    frame = readings_1014["不在_全满血"]["frame"]
    window = frame.pixels[700:716, 2358:2382]
    distance = _colour_distance(window)
    assert distance.min() <= enemybar.ABSENT_TOL, "那一带的颜色根本没进门"
    mask = distance <= enemybar.SCREEN_TOL
    assert _longest_vertical_run(mask) == 13
    assert _longest_vertical_run(mask) > enemybar.MAX_BAR_ROWS


def test_a_short_fill_is_present_but_its_ratio_is_unsampled(
    readings_1014,
) -> None:
    """**快死那一帧：条在，比例不给** —— 进门松、出口严（用户这一单的裁定 2）。

    画面事实：那一根的填充只有 **7 列**（右边是一长截深色空轨）。7 在
    `MIN_CANDIDATE_COLUMNS` 与 `MIN_FILL_COLUMNS` 之间，所以它是**候选**：
    存在性照报「在」（条确实在），比例落**未采样** —— 短填充给不出「填了多少」
    这句话该有的证据。它以前在 `_screen` 里就被丢掉，那一帧因此报出**精确的
    「不在」**（假事实）。
    """
    entry = readings_1014["在_快死的敌人"]
    result = entry["result"]
    assert result.presence.payload == Exact(True)
    assert result.count.payload == Exact(1)
    assert result.bars == ()
    assert len(result.unsampled) == 1
    assert result.cores != ()
    bar = result.unsampled[0]
    assert (bar.fill_rect.x, bar.fill_rect.y) == (1506, 481)
    assert isinstance(bar.fill.value, Unsampled)
    assert bar.columns.value.payload == Exact(7), "列数照量，读不出来的是「填了多少」"
    fill = entry["frame"].pixels[481:489, 1506:1630]
    runs = (_colour_distance(fill) <= enemybar.SCREEN_TOL).all(axis=0)
    leading = 0
    while leading < runs.size and runs[leading]:
        leading += 1
    assert leading == 7
    assert enemybar.MIN_CANDIDATE_COLUMNS <= leading < enemybar.MIN_FILL_COLUMNS


def test_the_red_wide_bar_is_read_by_the_red_family(readings_1014) -> None:
    """boss 那一帧的**红宽条**：橙族读不到它，**红族**读到，比例 = 115/154 = 0.7468。

    画布坐标下的画面事实：红条 x1562…1676 y1002…1011 —— **10 行**、填充 115 列、
    中位色 (197,41,40)；离 ENEMY_FILL_RED 的逐像素最大差 **14**（进得了门），
    离 ENEMY_FILL 的最小差 **59**（橙族读不到）。

    分母是**红族那一档的常量** `BOSS_FILL_COLUMNS = 154` —— 用户 10-15 认这个数，
    但按 (c) 记账：红族全库只有这一根真值，而且那次标定借过「量轨道」那条退役的路
    （那根红条自己**不满**，只有 115 列）。**运行时不再量轨道**。
    """
    result = readings_1014["在_boss_红宽条"]["result"]
    assert result.presence.payload == Exact(True)
    assert result.count.payload == Exact(3)
    assert [bar.fill_rect.x for bar in result.bars] == [1608, 1562, 1422]
    red = [bar for bar in result.bars if bar.fill_rect.x == 1562][0]
    assert red.fill_rect.width == 115
    # 裁定 2：fill_rect 的高是**量到的**条身行数。这一根画面上就是 10 行。
    assert (red.fill_rect.y, red.fill_rect.height) == (1002, 10)
    assert isinstance(red.fill.value, Valued)
    assert red.fill.value.payload == Exact(115 / enemybar.BOSS_FILL_COLUMNS)
    frame = readings_1014["在_boss_红宽条"]["frame"]
    body = frame.pixels[1002:1012, 1562:1677]
    assert body.shape == (10, 115, 3)
    to_red = _red_distance(body)
    assert to_red.min() == 0
    assert to_red.max() <= enemybar.SCREEN_TOL
    assert _colour_distance(body).min() > enemybar.SCREEN_TOL
    assert body.shape[1] >= enemybar.RED_MIN_COLUMNS



def test_the_twenty_four_column_bar_is_a_small_enemy_bar(readings_1014) -> None:
    """那根 **24 列**的：用户 10-14 裁定它是**一根小怪的血条**，不是假阳性。

    所以它上一版的 1.0 是错的 —— 分母是**小怪那一档**的 `MOB_FILL_COLUMNS = 123`：
    24/123 = **0.1951**（用户 10-14 要的「约五分之一」）。
    """
    bar = [
        item
        for item in readings_1014["在_boss_红宽条"]["result"].bars
        if item.fill_rect.x == 1608
    ][0]
    assert bar.fill_rect.width == 24
    assert isinstance(bar.fill.value, Valued)
    assert bar.fill.value.payload == Exact(24 / enemybar.MOB_FILL_COLUMNS)



def test_the_red_width_gate_still_leaves_the_boss_bar_in() -> None:
    """**红族的宽度门是「窄」那一步**：红门那些碎块比它窄得多，boss 那根比它宽得多。

    两根都是**红族**的整齐小矩形（每一列一样高、内部平），颜色也差不多 ——
    分开它们的只有宽度：红门那两块 18 / 23 列，boss 那根 115 列。
    """
    assert enemybar.RED_MIN_COLUMNS > 23
    assert enemybar.RED_MIN_COLUMNS < 115


def test_the_red_gate_blocks_are_not_bars(readings_1014) -> None:
    """**红门那一块里一个框都没有**（用户这一单的裁定 1；`boss.png` 整帧是红族的反例）。

    红门那两块是**过得了色的门**的：x2320 y330（18 列 x 10 行）离 `ENEMY_FILL_RED`
    最小 7、中位 10；x2373 y330（23 列 x 10 行）最小 8、中位 12。挡住它们的有两道：

    - 它们的**每一列并不是一样高**（8…9 / 7…9 行），所以「每一列都亮着的那几行」
      装不下整块（0.83 / 0.85 < `RECT_FILL_MIN`）—— 形状门；
    - 宽度 18 / 23 都低于红族的宽度门 `RED_MIN_COLUMNS`。

    搬进来的那一块（`不在_boss_红门.png`）里，链一个候选都没读出来。
    """
    entry = readings_1014[RED_GATE]
    result = entry["result"]
    assert result.presence.payload == Exact(False)
    assert result.count.payload == Exact(0)
    assert result.bars == ()
    assert result.cores == ()
    frame = entry["frame"]
    for x, y, width, height, low in ((2320, 330, 18, 10, 7), (2373, 330, 23, 10, 8)):
        window = frame.pixels[y : y + height, x : x + width]
        to_red = _red_distance(window)
        assert to_red.min() == low
        assert to_red.min() <= enemybar.SCREEN_TOL, (x, y)
        assert width < enemybar.RED_MIN_COLUMNS
        heights = (to_red <= enemybar.SCREEN_TOL).sum(axis=0)
        assert heights.min() != heights.max(), (x, y)


def test_a_blob_with_fringes_is_not_a_rectangle() -> None:
    """**矩形门唯一承重的那一块**：`raw_b` 里那块 9 列、色距 8 的假阳性。

    它是「颜色说它是条、形状说它不是」的那一类：离 `ENEMY_FILL` 最小 **5 <= SOLID_TOL**，
    可它每一列的竖直高度是 8…10 行不等 —— 毛刺占了整块 12% 的像素，
    「每一列都亮着的那几行」只装得下 0.88（线是 0.95）。
    以前它是被列的宽度门（12）顺带挡掉的；进门那道门放到 `MIN_CANDIDATE_COLUMNS`
    之后，挡住它的就是这一条。
    """
    frame = _load_source_frame(BATCH, "在_raw_b_上")
    window = frame.pixels[317:328, 2120:2129]
    distance = _colour_distance(window)
    assert distance.min() <= enemybar.SOLID_TOL
    mask = distance <= enemybar.SCREEN_TOL
    heights = mask.sum(axis=0)
    assert (int(heights.min()), int(heights.max())) == (8, 10)
    body = int(mask.all(axis=1).sum())
    assert body * window.shape[1] < enemybar.RECT_FILL_MIN * int(mask.sum())
    cores = enemybar.screened_cores(frame)
    assert all(
        not (core.fill_rect.x == 2120 and core.fill_rect.y == 317) for core in cores
    ), "这一块被读成候选了"


def test_the_two_hard_negatives_are_still_absent(readings) -> None:
    """**两条硬负例仍然是「不在」** —— 加了红族与形状门之后也要是。

    `shot26` 那根发光的栏杆里有一块 8 列、离**红族** 15 的整块：它**过得了色的门**，
    身体比是 1.01（形状门也拦不住），挡它的是红族的**宽度门**（8 < `RED_MIN_COLUMNS`）——
    也就是说这条硬负例这一版是宽度门守住的，**动那个数要先回头看它**。
    `S5-0-0` 那两段栈道边的高是 27 / 29 行，形状门在高度那一步就丢了。
    """
    for name in (RAILING, BOARDWALK, "反例_S5_栈道边_下"):
        entry = readings[name]
        assert entry["result"].presence.payload == Exact(False), name
        assert entry["result"].bars == (), name
        assert entry["result"].clumps == () and entry["result"].unsampled == (), name
        assert entry["result"].cores == (), name


def test_the_panel_side_of_the_pressed_operator_frame_has_no_bar(
    readings_1014,
) -> None:
    """**敌人介绍面板那一块**：整块都在战场那一段（100…1040 行）里，但一个候选都没有 ——
    它不是被「战场边界」排掉的（边界排不掉它），是颜色门排掉的。"""
    result = readings_1014["不在_压住干员面板"]["result"]
    assert result.cores == ()
    assert result.bars == ()
    assert result.presence.payload == Exact(False)
    assert result.count.payload == Exact(0)


def test_the_bar_pressed_against_our_operator_is_not_dropped(readings_1014) -> None:
    """**敌条压住我方干员的条**：那一根要读到，而且不许被当成我们自己的剔掉。

    它正踩在我方技力条那 8 行上，而我方血条的左缘在它左边 12 列（> `COL_SLACK`）——
    所以判「重合」不会把它吃掉。

    **那一根没被压住**（用户 10-15 的裁定）：真值就是 123 列 / 填充 30 列，
    比例 **30/123 = 0.2439**、**精确**、不给下界。老那一版只量到 42 列、给的是区间
    0.7143…0.8824 —— 根因是**半透明的轨道被底下那条我方条染了色**。这是
    「轨道不可信」的**第二条实证**（第一条是 `S8-3-4`）。
    """
    entry = readings_1014["在_压住干员"]
    result = entry["result"]
    assert result.presence.payload == Exact(True)
    assert result.count.payload == Exact(1)
    assert len(result.bars) == 1
    bar = result.bars[0]
    assert (bar.fill_rect.x, bar.fill_rect.y) == (1339, 701)
    assert bar.fill_rect.width == 30
    assert isinstance(bar.fill.value, Valued)
    assert bar.fill.value.payload == Exact(30 / enemybar.MOB_FILL_COLUMNS)
    assert all(core.masked is False for core in result.cores)



def test_the_overlap_frames_never_report_an_exact_count(readings_1014) -> None:
    """**叠在一起的那两帧：条数不许报精确**（用户 10-14 的裁定）。

    它们报的是「至少几根」，而且那个数**不许高报**。分不开的那几根**整坨落成一堆**
    （用户 10-15：「一堆」是一条正经的读数），**一堆里没有单根比例**。

    **两边不打架**（用户 10-15 的裁定 4）：下界 = 逐根 + 未采样 + 各堆的
    「至少几根」之和。
    """
    for name in ("在_互叠1", "在_互叠2"):
        result = readings_1014[name]["result"]
        assert result.presence.payload == Exact(True), name
        assert isinstance(result.count, Valued), name
        assert isinstance(result.count.payload, LowerBound), name
        assert result.clumps, name
        least = sum(int(clump.least.value.payload.lo) for clump in result.clumps)
        assert result.count.payload.lo == (
            len(result.bars) + len(result.unsampled) + least
        ), name


def test_a_clump_carries_its_total_fill_and_its_least_bars(readings_1014) -> None:
    """**要求 5①**：叠条那一坨落成「一堆」，而且带**填充总列数 + 至少几根**。

    真帧 `在_互叠1`：x1628 那 138 列与 x1808 那 48 列的两块**分不开**（同一条行带、
    填充挨着），所以它们**整坨**落成一堆 —— 区域是这两根填充段的外接框（1628 宽 228），
    填充总列数是两根**相加**（186），至少几根 = `ceil(186/123)` = **2**。

    这一堆的**填充总列数落的是下界**（组里有一条的右边没读到）—— 那正是用户要的
    「它自己带形态」，而「至少几根」跟着它算。
    """
    result = readings_1014["在_互叠1"]["result"]
    assert len(result.clumps) == 1
    clump = result.clumps[0]
    assert clump.tier == enemybar.MOB_TIER, "一堆要能看出是哪一档"
    assert (clump.region.x, clump.region.width) == (1628, 228)
    assert clump.columns.value.payload == LowerBound(186)
    assert clump.columns.margin is not None
    assert clump.least.value.payload == LowerBound(2)
    least = sum(int(item.least.value.payload.lo) for item in result.clumps)
    assert result.count.payload.lo == len(result.bars) + len(result.unsampled) + least



def test_the_first_batch_degrades_its_count_in_exactly_one_place(readings) -> None:
    """旧批 9 块：**8 块的条数精确、且等于三栏的根数**；只有 `在_raw_b_上` 降成
    下界 —— 它的填充段 130 列 > 123，说明那里**不止一根**（用户 10-15 的裁定）。

    也就是说：这一处降级**不是判据变松了**，是用户把那一块的真值从 1 根改成了 2 根。
    """
    for name, entry in readings.items():
        result = entry["result"]
        if name == "在_raw_b_上":
            assert isinstance(result.count.payload, LowerBound), name
            assert result.count.payload.lo >= 2, name
            assert len(result.clumps) == 1 and result.bars == (), name
            continue
        assert result.clumps == (), name
        assert isinstance(result.count.payload, Exact), name
        assert result.count.payload.value == len(result.bars) + len(result.unsampled), name



def test_two_cores_on_one_row_band_are_not_counted_twice() -> None:
    """合成：**同一条行带上、填充段被一段不混色的东西断开** -> 整坨落成**一堆**。

    右边那一块（x352 那 8 列）与左边那一根同带，而且左边那一根的**填充段那一趟**
    容得下跨过中间那 2 列 `FILL_GAP` —— 也就是说这两块**分不开**（`_touching`
    判的是**填充段自己的像素**：跨过去了就是相接）。同一条行带上两段填充这么并排，
    至少有一段是被截出来的一截 —— 所以「根数 = 2」是一个没有依据的精确主张。

    **要求 5②**：这一堆里**一根都不给单根比例**（`bars` 是空的）—— 硬给就是编。
    """
    pixels = np.full((1260, 2800, 3), BASE, np.uint8)
    track = np.clip(np.asarray(BASE, np.int16) - 52, 0, 255).astype(np.uint8)
    pixels[300:308, 300:350] = enemybar.ENEMY_FILL
    pixels[300:308, 350:352] = track
    pixels[300:308, 352:360] = enemybar.ENEMY_FILL
    frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    result = enemybar.read_enemy_bars(frame)
    assert result.bars == (), "一堆里不许出现单根比例"
    assert len(result.clumps) == 1
    clump = result.clumps[0]
    assert (clump.region.x, clump.region.width) == (300, 60)
    assert clump.columns.value.payload == Exact(68)
    assert clump.least.value.payload == LowerBound(1)
    assert result.count.payload == LowerBound(1)



def test_two_bars_on_different_bands_are_counted_exactly() -> None:
    """**分得开才给精确**：两根条各占一条行带（上下隔开），条数就是精确 2。"""
    pixels = np.full((1260, 2800, 3), BASE, np.uint8)
    track = np.clip(np.asarray(BASE, np.int16) - 52, 0, 255).astype(np.uint8)
    for row in (300, 420):
        pixels[row : row + 8, 300:340] = enemybar.ENEMY_FILL
        pixels[row : row + 8, 340:400] = track
    frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    result = enemybar.read_enemy_bars(frame)
    assert len(result.bars) == 2
    assert result.clumps == ()
    assert result.count.payload == Exact(2)


def test_two_families_never_share_a_clump() -> None:
    """**按族归组、不许跨族**（用户 10-15 的裁定 2）：橙条与红条挨着也各算一根。

    画一根 60 列的橙条，紧挨着它右边画一根 64 列的红条（**填充首尾相接**）。
    没有族门的话这两块会被接成一根（或者并成一组）—— 那样「至少几根」要除以哪一档
    的满条长就说不清了；有族门就是**两根**，各走各的档（橙 -> 小怪 123、
    红 -> boss 154）。
    """
    pixels = np.full((1260, 2800, 3), BASE, np.uint8)
    pixels[300:308, 300:360] = enemybar.ENEMY_FILL
    pixels[300:308, 360:424] = enemybar.ENEMY_FILL_RED
    frame = Frame(canvas=CANVAS_2800x1260, pixels=pixels)
    result = enemybar.read_enemy_bars(frame)
    assert result.clumps == (), "跨族并成一组了"
    assert [bar.fill_rect.x for bar in result.bars] == [300, 360]
    assert result.count.payload == Exact(2)


def test_the_clump_branch_is_load_bearing(monkeypatch, readings_1014) -> None:
    """**要求 5④ 红证**：把「一堆」这一支**整个拆掉**，本文件钉「一堆」的那几条用例
    必须变红 —— 拆了还绿，就说明它们钉的不是这一支。

    拆法：`_is_clump` 恒假。于是

    - `在_raw_b_上` 那 130 列的一坨**不再是一堆**：它被当成一根读，**条数报成精确
      1**（真值是至少 2）—— 正是这一支要挡的那种「假精确」；
    - `在_互叠1` 那两栏也塌成一栏（`clumps` 空），条数报成**精确 17**（真值是
      下界 17）—— 用户 10-14「叠在一起的条不许报精确」那条裁定就被推翻了。

    这里断言的是**拆掉之后的状态**，与文件里那几条用例的断言正好相反。
    """
    assert readings_1014["在_互叠1"]["result"].clumps
    monkeypatch.setattr(enemybar, "_is_clump", lambda members: False)
    raw_b = enemybar.read_enemy_bars(_load_source_frame(BATCH, "在_raw_b_上"))
    assert raw_b.clumps == ()
    assert len(raw_b.bars) == 1
    assert raw_b.count.payload == Exact(1), "拆掉之后真值 >= 2 的地方报成了精确 1"
    overlap = enemybar.read_enemy_bars(readings_1014["在_互叠1"]["frame"])
    assert overlap.clumps == ()
    assert len(overlap.bars) + len(overlap.unsampled) == 17
    assert overlap.count.payload == Exact(17), "分不开的那一帧被报成了精确"


def test_a_frame_with_nothing_in_it_counts_zero() -> None:
    """一块空地：三态是「不在」、条数是**精确 0**（不是未采样）。"""
    frame = Frame(
        canvas=CANVAS_2800x1260,
        pixels=np.full((1260, 2800, 3), BASE, np.uint8),
    )
    result = enemybar.read_enemy_bars(frame)
    assert result.presence.payload == Exact(False)
    assert result.count.payload == Exact(0)


def test_an_unsampled_frame_has_no_count_either() -> None:
    """认不出的那一帧，条数**也**是未采样 —— 不许拿 0 冒充「数过了，没有」。"""
    frame = _frame_with_bar(400, 300, 40, 0, _shift(22))
    result = enemybar.read_enemy_bars(frame)
    assert isinstance(result.presence, Unsampled)
    assert isinstance(result.count, Unsampled)


def test_the_readings_files_record_the_count_rule(report, report_1014) -> None:
    """两批的 `判据读数.json` 都要把**条数那一支的规则**记下来（它也是判据）。"""
    for payload in (report, report_1014):
        rule = payload["ruler"]["count"]
        assert "LowerBound" in rule and "Exact" in rule
        assert "不高报" in rule
        assert "条数" in rule
        assert "一堆" in rule
        assert "不打架" in rule


def test_the_readings_files_record_the_new_gates(report, report_1014) -> None:
    """**两档常量、跨档校验、短条、红族**这几支都落在尺子上 —— 判据变了，读数跟着变。"""
    for payload in (report, report_1014):
        ruler = payload["ruler"]
        assert "ENEMY_FILL_RED" in ruler["colour"]
        assert str(enemybar.MIN_CANDIDATE_COLUMNS) in ruler["shape"]
        assert str(enemybar.RED_MIN_COLUMNS) in ruler["shape"]
        assert str(enemybar.MIN_FILL_COLUMNS) in ruler["unsampled"]
        assert "未采样" in ruler["unsampled"]
        assert "clumps" in ruler["unsampled"]
        assert str(enemybar.RECT_FILL_MIN) in ruler["rectangle"]
        assert str(enemybar.RECT_ROW_SPREAD_MAX) in ruler["rectangle"]
        assert str(enemybar.MAX_BAR_ROWS) in ruler["body"]
        assert str(enemybar.MOB_BODY_ROWS) in ruler["body"]
        assert str(enemybar.BOSS_BODY_ROWS) in ruler["body"]
        assert "量到的行数" in ruler["body"]
        assert "更高" in ruler["body"], "跨档校验只拦更高的那一头"
        # 两档常量（分母）那一栏：两条都要写进去，并说清「画布宽 2800」与「未验」。
        dial = ruler["dial"]
        assert str(enemybar.MOB_FILL_COLUMNS) in dial
        assert str(enemybar.BOSS_FILL_COLUMNS) in dial
        assert "2800" in dial
        assert "未验" in dial
        # **两条读数各自的形态**都要落在尺子上（列数一条、比例一条）。
        assert "Exact" in ruler["columns"] and "LowerBound" in ruler["columns"]
        assert "FILL_GAP=" in ruler["columns"]
        assert "LowerBound" in ruler["fill"]
        assert "未采样" in ruler["fill"]
        assert "Exact" in ruler["fill"]
        # 「一堆」那一支的规则：判据（分不开 / 长过满条长）、按族归组、至少几根。
        assert "分不开" in ruler["clump"] and "满条长" in ruler["clump"]
        assert "不许跨族" in ruler["clump"]
        assert "至少几根" in ruler["clump"]
        assert "上界" in ruler["clump"]
        # 轨道那一套**整个退役**：尺子上不许再有那一栏；旧的 short 那一栏也并进
        # `unsampled` 了（同一条判据不写两遍）。
        assert "track" not in ruler
        assert "short" not in ruler


