"""编队页在人数不同（1–12）时稳不稳：**现有那条链**在 run-05 那两帧上的行为。

**这一单不新增链**：下面这些是**网**，钉住三件量出来的事实 ——

1. **卡位数是传进来的输入**：同一帧喂 (6,6) 与 (1,1) 链两个都收，切出来的块数跟着输入走
   ⇒ 编队页那条链**不产「有 / 空」这个判决**。⚠ 这条是**根因一的网，不是判据**：
   哪一天链开始从画面读卡位数（或张数对不上就报错），这条会红 —— 那时改它，
   并去改 `tests/帧样本/编队页人数/README.md` 的 §根因一。
2. **助战位在 x 2077 起**（第 7 列位置）：上行 <= 6 时**没有任何取区碰得到它**；
   取 (7,6) 时第 7 个取区正压在它上面 ⇒ 助战那一位的名字不会被读到。
3. **空位那一家族在立绘那一层分得开**（内部 <= 3.3546、与真卡 >= 77.3764），
   **在名字带那一层分不开**（空位 151–636 个白像素、真名字 361–2043，**重叠**）。
   这两条都是量出来的**画面事实**，**这一单没有拿它们立判据**。

夹具落在 `tests/帧样本/编队页人数/`（带原点 + 源帧 RAW sha256），加载走
`load_batch` 再**贴回画布**（与 `立绘两组` 同一套），所以 ROI 的取法仍然真的被测。
**不碰 `screenshots/`**：那目录随时会删。
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ark.engine.device.adb import SQUAD_CARD_PITCH
from ark.engine.vision.bench import squad_card_centres
from ark.engine.vision.frame import Frame, frame_from_raw
from ark.offline.assets_gen import roster_names as RN
from ark.offline.assets_gen.cardface import cut_formation_arts, load_formation_page
from ark.offline.bench.batch import load_batch

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "tests" / "帧样本" / "编队页人数"

QUCAI = json.loads((BATCH / "取材.json").read_text(encoding="utf-8"))
TRUTH = {entry["file"]: entry for entry in QUCAI["frames"]}
"""逐张真值（画面读出来的），按夹具里那个文件名取。"""

FULL_PAGES = ("编队1-0", "编队2-0")
BENCH_BANDS = ("编队1-1_备战席", "编队2-1_备战席")
ASSIST_X = 2077
"""助战位那一格的左缘（原生像素，目视量的）：它在第 7 列的格位上。"""


@pytest.fixture(scope="module")
def cases() -> dict:
    """夹具那四条（**没贴回画布**，每条带着自己的矩形与出处）。"""
    return {case.name: case for case in load_batch(BATCH).cases}


@pytest.fixture(scope="module")
def frames(cases: dict) -> dict[str, Frame]:
    """夹具那四条（**贴回画布之后**：裁剪之外是零）。"""
    return {name: frame_from_raw(case.raw()) for name, case in cases.items()}


@pytest.fixture(scope="module")
def pages(frames: dict[str, Frame]) -> dict[str, Frame]:
    """那两帧编队页（整帧，原生 2800×1260）。"""
    return {name: frames[name] for name in FULL_PAGES}


def _pressed(page: Frame, box: tuple[int, int, int, int], span: int) -> np.ndarray:
    """名字取区**按链那一套压深之后**那一块（`roster_names._cell` 同一个把戏）。"""
    x, y, width, height = box
    cell = np.ascontiguousarray(page.pixels[y : y + height, x : x + width]).copy()
    cell[0 : RN.PAD_ABOVE] = 0
    cell[RN.PAD_ABOVE + span :] = 0
    return cell


def _ink(cell: np.ndarray) -> int:
    """那一块里的**白像素**数（三通道都 >= 150）—— 名字与那句 `RHODES ISLAND` 都算。"""
    return int((cell >= 150).all(axis=2).sum())


def _mad(left: np.ndarray, right: np.ndarray) -> float:
    """两块立绘的**平均绝对差**（0–255 的 RGB 三通道）。"""
    return float(np.abs(left.astype(np.int16) - right.astype(np.int16)).mean())


# --- 夹具本身 ---------------------------------------------------------------


def test_the_batch_pastes_every_block_back_pixel_for_pixel(cases: dict) -> None:
    """四条都**贴回画布**，与直接解那一块 PNG 逐位相同 —— ROI 的取法真的在被测。"""
    assert sorted(cases) == sorted(FULL_PAGES + BENCH_BANDS)
    for name, case in cases.items():
        assert case.source and case.source_sha256
        assert len(case.source_sha256) == 64, name
        rect = case.rect
        got = frame_from_raw(case.raw()).pixels[
            rect.y : rect.y + rect.height, rect.x : rect.x + rect.width
        ]
        with Image.open(BATCH / (name + ".png")) as image:
            want = np.asarray(image.convert("RGB"))
        assert np.array_equal(got, want), name
    assert cases["编队1-0"].rect.x == 0 and cases["编队1-0"].rect.width == 2800
    for name in BENCH_BANDS:
        assert cases[name].rect.x == 300 and cases[name].rect.y == 1040
        assert cases[name].rect.width == 2500 and cases[name].rect.height == 220


def test_the_truth_file_says_which_slots_hold_a_card(pages: dict[str, Frame]) -> None:
    """真值那两栏（`someone` / `empty`）**加起来正好 12 格**，而且两帧的人数对得上画面。"""
    twelve = TRUTH["编队1-0.png"]
    two = TRUTH["编队2-0.png"]
    assert twelve["people"] == 12 and two["people"] == 2
    assert twelve["empty"] == [] and twelve["someone"] == list(range(12))
    assert two["someone"] == [0, 6]
    assert sorted(two["someone"] + two["empty"]) == list(range(12))
    assert twelve["assist"] == "空" and two["assist"] == "空"


# --- 根因一：卡位数是输入 ---------------------------------------------------


def test_a_wrong_card_count_is_accepted_silently_on_both_pages(pages: dict[str, Frame]) -> None:
    """**根因一的网**：同一帧喂两个不同的卡位数，链两个都收，块数跟着输入走。

    这不是判据 —— 编队页那条链**没有「有 / 空」这个判决**：每行几张由调用方给，
    空位照切、每块都得到一个身份位。两帧的常量那份示例（6 + 5）也都**不报错**：
    `_check_counts` 只留三条硬边界（两行 · 整数且至少一张 · 装得下画布），
    `_check_geometry` 核的是左缘/卡距/卡宽/缝/立绘宽/上下缘，**张数不在里面**。
    """
    full, sparse = pages["编队2-0"], pages["编队2-0"]
    assert len(cut_formation_arts(full, counts=(6, 6))) == 12
    assert len(cut_formation_arts(sparse, counts=(1, 1))) == 2
    assert len(RN.name_boxes((6, 6))) == 12
    assert len(RN.name_boxes((1, 1))) == 2
    for page in pages.values():
        assert cut_formation_arts(page, counts=(6, 5))
    assert TRUTH["编队2-0.png"]["people"] == 2, "画面数出来是 2 —— (6,6) 是超报"


def test_the_empty_slots_are_one_family_and_the_real_cards_are_far(pages: dict[str, Frame]) -> None:
    """**空位长什么样（立绘那一层）**：空位家族内部最大 3.3546，与真卡最小 77.3764。

    空档 **3.3546 … 77.3764（宽 74.0218）**。这是量出来的**画面事实**，不是判据 ——
    这一单没有拿它立线；谁要立，拿它当起点、并且先报规划者。
    """
    arts = cut_formation_arts(pages["编队2-0"], counts=(6, 6))
    empty = tuple(TRUTH["编队2-0.png"]["empty"])
    real = tuple(TRUTH["编队2-0.png"]["someone"])
    assert len(empty) == 10 and len(real) == 2
    inside = [_mad(arts[i], arts[j]) for i, j in combinations(empty, 2)]
    cross = [_mad(arts[i], arts[j]) for i in empty for j in real]
    assert len(inside) == 45 and len(cross) == 20
    assert max(inside) < 4.0, max(inside)
    assert min(cross) > 70.0, min(cross)
    assert min(cross) - max(inside) > 70.0


def test_the_name_strip_does_not_separate_empty_from_taken(pages: dict[str, Frame]) -> None:
    """**名字带那一层分不开**：空位格 151–636 个白像素，真名字 361–2043 —— **重叠**。

    空位那一格里那句 `RHODES ISLAND` 是真白字（压深之后还在），所以「名字带上有墨
    ⇒ 有卡」这条会把空位判成有人。这条也是**量出来的画面事实**，不是判据。
    """
    spans = [RN.NAME_SPANS[row] for row, count in enumerate((6, 6)) for _ in range(count)]
    taken = [
        _ink(_pressed(pages["编队1-0"], box, span))
        for box, span in zip(RN.name_boxes((6, 6)), spans)
    ]
    assert len(taken) == 12 and min(taken) > 0
    empty = [
        _ink(_pressed(pages["编队2-0"], RN.name_boxes((6, 6))[slot], spans[slot]))
        for slot in TRUTH["编队2-0.png"]["empty"]
    ]
    assert len(empty) == 10 and min(empty) > 0, "空位那一格也有白字"
    assert min(empty) < min(taken), "两簇重叠：空位里最高的那个比真名字里最低的那个还高"
    assert max(empty) > min(taken)


# --- 根因二：助战位不在链的视野里 -------------------------------------------


def test_the_assist_slot_is_outside_every_name_box_the_chain_can_make() -> None:
    """上行 <= 6 时**没有一个名字取区碰得到 x 2077 那一格**（助战位）。

    取 (7,6) 时第 7 个取区是 `(2096, 571, 214, 44)` —— 它压在那格上，而且那格里
    墨迹很足（压深之后 2171 个白像素，比任何一个真名字都多）。所以：**助战位一旦有人，
    那一位的名字不会被任何一行读到**；反过来，为了让链读到它把上行写成 7，
    那格里的 `SUPPORT UNIT` 会一起进拼图。**两条都没改，报规划者定。**
    """
    for counts in ((1, 1), (5, 5), (6, 6)):
        for x, _y, width, _height in RN.name_boxes(counts):
            assert x + width <= ASSIST_X, counts
    seventh = RN.name_boxes((7, 6))[6]
    assert seventh == (2096, 571, 214, 44)
    assert seventh[0] < ASSIST_X + 200 and seventh[0] + seventh[2] > ASSIST_X
    assert seventh[0] > ASSIST_X, "第 7 个取区比那一格的左缘还靠右一点（19 px）"


def test_the_two_pages_hold_the_same_empty_assist_tile(pages: dict[str, Frame]) -> None:
    """两帧的助战位是同**一张空占位**（不是同一份字节）：mean 0.004120 · max 5。

    所以这两帧都答不了「助战位有人的时候读得到吗」—— 报在 README §还差什么。
    """
    left = pages["编队1-0"].pixels[137:617, ASSIST_X : ASSIST_X + 253].astype(np.int16)
    right = pages["编队2-0"].pixels[137:617, ASSIST_X : ASSIST_X + 253].astype(np.int16)
    diff = np.abs(left - right)
    assert float(diff.mean()) < 0.05
    assert int(diff.max()) <= 8
    assert not np.array_equal(left, right), "是同一张占位，不是同一份字节"


# --- 名字库与备战席那条链 ---------------------------------------------------


def test_the_name_list_already_holds_every_name_printed_on_these_pages() -> None:
    """这一批 14 个名字**全在** `assets/roster/names.json` 里：库缺项 **0**。

    所以「名字库缺项」这条根因在这一批上**不成立** —— 一次 OCR 都不用为了补库。
    """
    index = RN.roster_index(RN.load_roster_names())
    names: list[str] = []
    for entry in QUCAI["frames"]:
        names.extend(entry.get("names_in_cut_order") or [])
    assert len(names) == 14
    for name in names:
        assert RN.decide(name, index) == (name, "精确"), name


def test_the_bench_chain_counts_its_cards_off_the_picture(frames: dict[str, Frame]) -> None:
    """备战席那条链**卡位数是数出来的**：12 人队 **7** 格 · 2 人队 **1** 格。

    这两条与画面数出来的一致，而且里没有一格是靠网格补的（全 `True`）。
    与编队页那条正好相反：仓里**已经有**「从画面读卡位数」这条做法，编队页那条没有。
    """
    got: dict[str, int] = {}
    for name in BENCH_BANDS:
        centres = squad_card_centres(frames[name], pitch=SQUAD_CARD_PITCH)
        assert centres and all(pixel for _x, pixel in centres), name
        got[name] = len(centres)
    assert got == {"编队1-1_备战席": 7, "编队2-1_备战席": 1}
    assert TRUTH["编队1-1_备战席.png"]["bench_slots"] == 7
    assert TRUTH["编队2-1_备战席.png"]["bench_slots"] == 1


def test_the_stored_sheet_is_the_deterministic_derivation_of_the_stored_frame(
    pages: dict[str, Frame],
) -> None:
    """仓里那张拼图 = 仓里那一帧按 (6,6) 拼出来的**逐字节同一张**。

    这样将来那份响应能钉在**这一张**拼图上：只留哈希证不了发出去的是哪一张，
    只留图证不了接口回了什么 —— 两样都留才叫可重放。
    """
    for name in FULL_PAGES:
        png = RN.sheet_png(RN.compose_sheet(pages[name], counts=(6, 6)))
        assert png == (BATCH / (name + "_拼图.png")).read_bytes(), name
        assert png == RN.sheet_png(RN.compose_sheet(pages[name], counts=(6, 6)))
        assert len(RN.sheet_digest(png)) == 32
    assert (BATCH / "编队1-0_拼图.png").stat().st_size > 0


# --- 那两次真调用（预算正好两次，响应在仓里） --------------------------------


def _response(name: str) -> str:
    """夹具里那一份**原始响应文本**（一字未改；单行、没有换行）。"""
    raw = (BATCH / (name + "_响应.json")).read_text(encoding="utf-8")
    assert chr(10) not in raw and chr(13) not in raw, "接口原文是单行"
    return raw


@pytest.fixture(scope="module")
def roster() -> dict:
    return RN.roster_index(RN.load_roster_names())


def _replay(name: str, counts: tuple[int, ...], roster: dict, *, strict: bool = True):
    """把夹具里那份响应喂进判决链（**网络一个字都不动**）。"""
    raw = _response(name)
    page = load_formation_page(BATCH / (name + ".png"))
    return RN.read_names(
        page, counts=counts, transport=lambda _png: raw, roster=roster,
        cache_dir=None, strict=strict,
    )


def test_the_twelve_page_reads_all_twelve_names_with_one_unknown(roster: dict) -> None:
    """**12 人那一帧**：12 格一格一词（没有合词、没有缺词），判决 精确 10 · 改过 1 · 未知 1。

    照画面真值：**对 11 · 错 0 · 认不出 1**（格 1，真值「余」，接口回「账」）。
    那一格是**认不出**，不是错、也不是「空着」；「司霆惊蛰」读成「司霆惊垫」落在**改过**。
    """
    reading = _replay("编队1-0", (6, 6), roster)
    assert reading.digest == RN.sheet_digest((BATCH / "编队1-0_拼图.png").read_bytes())
    assert reading.counts == {"精确": 10, "改过": 1, "未知": 1}
    assert len(reading.slots) == 12
    assert reading.slots[1].word == "账" and reading.slots[1].name is None
    truth = TRUTH["编队1-0.png"]["names_in_cut_order"]
    got = [item.name for item in reading.slots]
    assert [name for name in got if name] == [name for name in truth if name != "余"]
    assert reading.slots[7].how == "改过" and reading.slots[7].word == "司霆惊垫"
    assert reading.slots[7].name == "司霆惊蛰"


def test_the_two_page_refuses_when_the_card_count_is_overstated(roster: dict) -> None:
    """**2 人那一帧按 (6,6) 超报**时链**响亮地拒绝**，不编名字。

    接口只回了 2 个词（那 10 个空位**一个词都没回**），于是缺词的正好是那 10 个空位格 ——
    `strict=True` 抛 `RosterLayoutError`。所以「空位被认成一张卡」在名字链这一层是堵住的：
    空位落的是**缺词**，不是「未知」、也不是编出来的名字。
    """
    with pytest.raises(RN.RosterLayoutError) as bad:
        _replay("编队2-0", (6, 6), roster)
    assert list(bad.value.missing) == TRUTH["编队2-0.png"]["empty"]
    assert list(bad.value.extra) == []
    loose = _replay("编队2-0", (6, 6), roster, strict=False)
    assert loose.counts == {"精确": 2, "改过": 0, "未知": 10}
    assert loose.slots[0].name == "新约能天使"
    assert loose.slots[6].name == "THRM-EX"
    assert [item.word for item in loose.slots[1:6]] == [""] * 5


def test_a_single_character_name_read_wrong_can_never_be_decided(roster: dict) -> None:
    """**那一格「认不出」的根因**：库里有 **24 个单字名**。

    `decide` 的规矩是「编辑距离 <= 1 的候选**恰好一个**才采用」，而任意两个**单字**名
    的距离都是 1 ⇒ 单字名一旦读错一个字就是 24 条候选、**结构上不可判**。
    所以它不是阈值、也不是库缺项。库里单字名的条数一变，上面那个结论就要重算 —— 钉住它。
    """
    single = [name for name in RN.load_roster_names() if len(RN.normalize(name)) == 1]
    assert len(single) == 24, single
    assert RN.decide("余", roster) == ("余", "精确")
    assert RN.decide("账", roster) == (None, "未知")
    nearby = {name for name in single if RN.edit_distance(RN.normalize(name), "账") <= 1}
    assert len(nearby) == 24
