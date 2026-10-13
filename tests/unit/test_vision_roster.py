"""备战席点名那条链（「这张卡是谁」）的单元测试。

**不 import tests/spec**（D-全局-17），也不带验收的检查编号 —— 这里测的是**这条链
自己的形状与边界**：编队页怎么切成库、判据用的是哪个统计量、阈值落在哪段空档里、
以及**三档降级**（比中 / 库里没有这个人 / 装置卡 / 红罩卡）各落什么形态。

**这一层只产两种形态：精确 / 未知**（用户裁定）。所以下面每一条都在验
"**没给确切值**"，而不是"给了哪一个" —— 验收里"任何一个确切值认错就是没过"，
这一份把那条红线的每一侧都钉住。

## 真值从哪来，值型断言落在哪

- **库与名字**：`tests/帧样本/点名/编队页.json`（名字是规划者目视读出来的）；
- **逐位身份真值**：`tests/帧样本/点名/取材.json` 只给 **`shot01` · `cd60`**
  两帧，所以**值型的断言只落在这两帧上**；
- **红罩卡的下标**：`tests/帧样本/备战席/取材.json` 的 `redeploying`（那是倒计时
  那一层的取材表，这里只借它指出"哪几张是红罩"）。**备战席那一批的红罩卡没有身份
  真值可对**（那一批的真值表只覆盖 `shot01` · `cd60`，而它们一张红罩卡都没有）；
- **红罩卡的逐位真值**：`tests/帧样本/红罩/取材.json` 与 `预处理/取材.json` 的
  `under_the_cap` —— **5 张**。两份都是"拿同一张卡后来干净了那一帧喂同一条链
  读出来的"，不是目视。**值型断言只准落在这 5 张上。**

## 这一条链慢

一帧十几秒：`库条数 × 尺度档数 × 整块立绘的滑动搜索`。所以**重活跑一次、装在模块
级的夹具里**，用例只读结果；红罩那 14 张也只算一次。

## 单条查一位（这一单新开的那条路）

`find_squad_card` 回答「某一位在第几张卡」，**不扫全库**。它的判据是
`squad_card_scores`（公开），所以下面三处断言用的是**同一个读数**：两簇空档 ·
两路关系 · 零假事实面。

**代价那条钉的是调用次数，不是墙钟**：同代码在这台机器上能差 40%。次数用一个
计数的包装器量（`monkeypatch` 把 `cv2.matchTemplate` 换成"记一笔再转给真的那个"）
—— 它**没有**让实现绕过任何闸门，只是把调用数了一遍。

**「两张都过线」那一支在真帧上一次都没触发**，所以那一条用**合成的孪生卡**钉：
把卡 0 的身份区抄到卡 1 上，两格的判据输入一模一样。

## 合成用例是在**真帧之外**造的

尺度那条与 Pearson 那条用随机图（判据的形状与真帧无关）；"红罩卡从真入口上读得出"
那条用**只留最右那张卡的一帧**（`cd01` 的最右那张正好是红罩卡）—— 这样不必把整帧
跑完也能从**真入口**上验一次。

## 红罩卡那一组（D-全局-34 接上来的）

**它换的是取区，不是判据**：`redeploying` 说是 `True` 的格子走 `redcap_region` +
`margin.MaskedRuler`，其余照旧。所以那一组钉四件事：① 那 5 张有真值的读出真值、
② 读不出的落未知**而且没有分数**、③ 同一批帧上**干净卡逐位不变**（"改动前"按旧口径
重算）、④ **只有红罩卡那几格**走了新取区（量的是递给谁的哪一块，不是墙钟）。
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from ark.contracts import (
    Canvas,
    Exact,
    PixelRect,
    SlotValue,
    Unknown,
    Valued,
    squad_card_rect,
)
from ark.engine.device.adb import SQUAD_CARD_PITCH
from ark.engine.vision import margin
from ark.engine.vision.bench import redeploying, squad_card_centres
from ark.engine.vision.frame import Frame, frame_from_raw
from ark.engine.vision.roster import (
    IDENTITY_BOX_HEIGHT,
    IDENTITY_BOX_WIDTH,
    IDENTITY_BOX_X,
    IDENTITY_BOX_Y,
    IDENTITY_SCALES,
    IDENTITY_SCALE_MAX,
    IDENTITY_SCALE_MIN,
    IDENTITY_SCALE_STEP,
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
    find_squad_card,
    identity_region,
    read_squad_identities,
    redcap_region,
    roster_scores,
    squad_card_scores,
)
from ark.offline.assets_gen import cardface
from ark.offline.assets_gen.cardface import (
    FORMATION_ART_WIDTH,
    FORMATION_CARD_WIDTH,
    FORMATION_CARDS,
    FORMATION_GAP,
    FORMATION_LEFT0,
    FORMATION_MANIFEST,
    FORMATION_PITCH,
    FORMATION_ROWS,
    cut_formation_arts,
    load_cardfaces,
    load_formation_page,
)
from ark.offline.bench.batch import load_batch

REPO = Path(__file__).resolve().parents[2]
BATTLE_BATCH = REPO / "tests" / "帧样本" / "备战席"
NAME_TRUTH = REPO / "tests" / "帧样本" / "点名" / "取材.json"
BATTLE_HARVEST = BATTLE_BATCH / "取材.json"
PAGE = REPO / json.loads(FORMATION_MANIFEST.read_text(encoding="utf-8"))["frame"]

PITCH = SQUAD_CARD_PITCH
"""卡距。它住设备层的 `SQUAD_CARD_PITCH`，导入规则不许视觉层 import 设备层，
所以用例里也照调用方的样子把它传进来。"""

TRUTH_FRAMES = ("shot01", "cd60")
"""`点名/取材.json` 里**给了逐位身份真值**的那两帧。**值型断言只准落在它们身上。**"""

REDCAP_BATCH = REPO / "tests" / "帧样本" / "红罩"
"""红罩那一批（8 帧、一局）—— **红罩卡有身份真值**的那一批。"""

REDCAP_TRUTH_FRAMES = ("cd14", "cd22", "cd23", "cd27", "cd37")
"""那 8 帧里**有身份真值**的五帧（合起来 5 张红罩卡）。"""


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    """零均值归一化相关 —— 手算一份，用来钉住"链用的分数就是它"。"""
    left = first - first.mean()
    right = second - second.mean()
    return float(
        (left * right).sum() / np.sqrt((left * left).sum() * (right * right).sum())
    )


def _name_of(value: SlotValue) -> str | None:
    """这个槽值给出的**确切名字**；没给确切值就是 `None`。

    顺手把**形态**也钉住：身份层只产 `Valued`（没有「未采样」），载荷只有
    `Exact` 与 `Unknown` 两种。
    """
    assert isinstance(value, Valued), f"身份层只产 Valued，收到 {type(value).__name__}"
    payload = value.payload
    if isinstance(payload, Exact):
        return str(payload.value)
    assert isinstance(payload, Unknown), f"载荷只能是 Exact / Unknown，收到 {payload!r}"
    return None


def _recorded_truth() -> dict[str, list[str]]:
    """`取材.json` 的逐位真值 —— **只给 `shot01` · `cd60` 两帧**。"""
    return {
        Path(entry["file"]).stem: list(entry["identity_right_to_left"])
        for entry in json.loads(NAME_TRUTH.read_text(encoding="utf-8"))["harvest"]
    }


def _card_score(scored: tuple[tuple[float, int], ...], index: int) -> float:
    """单条那条路给**某一张卡**的分。

    它的返回是 `((分数, 卡下标), ...)`、按分数降序，所以这里要翻成"按下标查"。
    """
    return {card: score for score, card in scored}[index]


def _accepted(scored: tuple[tuple[float, int], ...]) -> list[int]:
    """这条路**接受**的那几张卡（过 `ROSTER_SOLID_MATCH_SCORE`）—— 按分数降序。"""
    return [card for score, card in scored if score >= ROSTER_SOLID_MATCH_SCORE]


def _red_card_slots() -> tuple[tuple[str, int], ...]:
    """**14 张红罩卡**：`(帧名, 卡下标)` —— 下标借自备战席那份取材表的 `redeploying`。

    取材表里只写了"这一帧的这几张在再部署中"，而**哪一张是红罩**由倒计时那一层的
    判据给（`redeploying` 为真）。这一批（`备战席`）的红罩卡**没有逐位真值**，
    所以它们只用在形态断言上 —— 有真值的那 5 张在 `红罩` 那一批里。
    """
    record = json.loads(BATTLE_HARVEST.read_text(encoding="utf-8"))
    found: list[tuple[str, int]] = []
    for entry in record["harvest"]:
        for item in entry.get("redeploying") or []:
            found.append((Path(entry["file"]).stem, int(item["index"])))
    return tuple(found)


def _rightmost(frame: Frame, cards: int) -> Frame:
    """只留最右 `cards` 张卡的一帧 —— 左缘往左的像素清成 0，跳变证据跟着没了。

    与 `lone_red_card` 同一个做法。**清到第 `cards - 1` 张卡的左缘**：清得不够，
    再往左那一格的跳变会活下来，卡位就不止 `cards` 个。
    """
    pixels = np.array(frame.pixels)
    pixels[:, : int(squad_card_rect(cards - 1, PITCH).x)] = 0
    return Frame(canvas=frame.canvas, pixels=pixels)


@pytest.fixture(scope="module")
def scratch() -> Iterator[Path]:
    """一个临时目录 —— **不用 `tmp_path`**：那个夹具在 Windows 上建的目录带 0o700
    的 DACL 且不可继承，pytest 再进去就 `WinError 5`。照仓库里已有的做法，在
    `tempfile.gettempdir()` 下面开一个 uuid 名的目录（默认权限）。"""
    root = Path(tempfile.gettempdir()) / f"ark-roster-{uuid.uuid4().hex[:12]}"
    root.mkdir()
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def page() -> Frame:
    """编队页那一帧（原生 2800×1260）。"""
    return load_formation_page(PAGE)


@pytest.fixture(scope="module")
def faces() -> tuple[tuple[str, np.ndarray], ...]:
    """**库**：编队页那一帧切出来、配好名字的 11 块立绘。"""
    return load_cardfaces()


@pytest.fixture(scope="module")
def frames() -> dict[str, Frame]:
    """备战席那 10 帧（贴回画布之后）。"""
    return {case.name: frame_from_raw(case.raw()) for case in load_batch(BATTLE_BATCH).cases}


@pytest.fixture(scope="module")
def identities(
    frames: dict[str, Frame], faces: tuple[tuple[str, np.ndarray], ...]
) -> dict[str, tuple[SlotValue, ...]]:
    """`shot01` + `cd60` 的逐卡位读数 —— 一帧十几秒，所以**只跑这两帧、跑一次**。"""
    return {
        name: read_squad_identities(frames[name], faces, pitch=PITCH)
        for name in TRUTH_FRAMES
    }


@pytest.fixture(scope="module")
def red_cards() -> tuple[tuple[str, int], ...]:
    """**14 张红罩卡**：`(帧名, 卡下标)` —— 下标借自备战席那份取材表的 `redeploying`。"""
    return _red_card_slots()


@pytest.fixture(scope="module")
def red_regions(
    frames: dict[str, Frame], red_cards: tuple[tuple[str, int], ...]
) -> tuple[tuple[str, int, np.ndarray], ...]:
    """那 14 张卡上的**身份区** —— 与链取的是**同一个框**（`identity_region`）。"""
    found: list[tuple[str, int, np.ndarray]] = []
    for name, index in red_cards:
        frame = frames[name]
        centre = squad_card_centres(frame, pitch=PITCH)[index][0]
        found.append(
            (name, index, np.array(frame.crop(identity_region(centre, pitch=PITCH))))
        )
    return tuple(found)


@pytest.fixture(scope="module")
def lone_red_card(
    frames: dict[str, Frame], faces: tuple[tuple[str, np.ndarray], ...]
) -> tuple[SlotValue, ...]:
    """**只留最右那张卡**的一帧 —— `cd01` 的最右那张正好是红罩卡。

    把左缘往左的像素清成 0，跳变证据就跟着没了，卡位只剩最右那一个 —— 于是
    "红罩卡从真入口上读得出"可以从**真入口**上验，不必把整帧（9 张卡、十几秒）
    跑完。
    """
    frame = frames["cd01"]
    pixels = np.array(frame.pixels)
    pixels[:, : int(squad_card_rect(0, PITCH).x)] = 0
    return read_squad_identities(
        Frame(canvas=frame.canvas, pixels=pixels), faces, pitch=PITCH
    )


@pytest.fixture(scope="module")
def single_scores(
    frames: dict[str, Frame], faces: tuple[tuple[str, np.ndarray], ...]
) -> dict[str, dict[str, tuple[tuple[float, int], ...]]]:
    """单条那条路在两帧上、对**每一位**的逐卡分数（一帧十几秒，所以跑一次）。

    它同时是这一单三处断言的**同一个读数**：两簇空档（【做】5）· 两路关系
    （【做】7）· 零假事实面（【做】8）。
    """
    return {
        name: {
            who: squad_card_scores(frames[name], who, faces, pitch=PITCH)
            for who, _ in faces
        }
        for name in TRUTH_FRAMES
    }


# ---------------------------------------------------------------------------
# 库与几何
# ---------------------------------------------------------------------------


def test_the_formation_page_is_cut_into_eleven_arts(page: Frame) -> None:
    arts = cut_formation_arts(page)
    assert len(arts) == FORMATION_CARDS == 11
    assert [art.shape[0] for art in arts] == [617 - 137] * 6 + [1122 - 642] * 5
    assert all(art.shape[1] == FORMATION_ART_WIDTH for art in arts)
    assert all(art.dtype == np.uint8 and art.ndim == 3 and art.shape[2] == 3 for art in arts)


def test_the_first_art_is_the_card_box_on_the_native_canvas(page: Frame) -> None:
    art = cut_formation_arts(page)[0]
    expected = page.pixels[137:617, FORMATION_LEFT0 : FORMATION_LEFT0 + FORMATION_ART_WIDTH]
    assert np.array_equal(art, expected)
    # 是**拷贝**，不是视图：库会被拿着到处走，留一个指向整帧的视图等于把整帧也拖着。
    assert not np.shares_memory(art, page.pixels)


def test_the_library_pairs_names_with_the_arts_in_cut_order(
    faces: tuple[tuple[str, np.ndarray], ...],
) -> None:
    recorded = json.loads(FORMATION_MANIFEST.read_text(encoding="utf-8"))
    assert [name for name, _ in faces] == recorded["names"]
    assert len(faces) == FORMATION_CARDS


def test_the_library_is_reproducible(faces: tuple[tuple[str, np.ndarray], ...]) -> None:
    again = load_cardfaces()
    assert [name for name, _ in again] == [name for name, _ in faces]
    for (_, first), (_, second) in zip(faces, again):
        assert np.array_equal(first, second)


def test_the_geometry_is_the_measured_one() -> None:
    assert FORMATION_LEFT0 == 470
    assert FORMATION_PITCH == 271
    assert FORMATION_CARD_WIDTH == 262
    assert FORMATION_GAP == 9
    assert FORMATION_ART_WIDTH == 214
    assert FORMATION_ROWS == ((137, 617, 6), (642, 1122, 5))
    assert FORMATION_CARDS == 11
    # 两侧同一像素尺度的那个比（备战席卡 204 宽）—— 身份区要搜尺度的由来。
    card_width = float(squad_card_rect(0, PITCH).width)
    assert FORMATION_ART_WIDTH / card_width == pytest.approx(1.049, abs=1e-3)


def test_a_foreign_canvas_is_refused(page: Frame) -> None:
    with pytest.raises(TypeError, match="一帧"):
        cut_formation_arts(page.pixels)
    small = Frame(canvas=Canvas(width=1920, height=864), pixels=np.zeros((864, 1920, 3), np.uint8))
    with pytest.raises(ValueError, match="不做分辨率适配"):
        cut_formation_arts(small)


def _declared_geometry() -> dict[str, object]:
    """夹具清单里声明的那份几何（六个键全在）。"""
    record = json.loads(FORMATION_MANIFEST.read_text(encoding="utf-8"))
    return dict(record["geometry"])


def _manifest_with_geometry(scratch: Path, geometry: object) -> Path:
    """把夹具清单抄一份、`geometry` 换成给定的那份 —— 单点验闸门用。

    只碰 `geometry`：`frame` 与 `names` 原样带过去，所以清单仍然指得着那一帧。
    """
    record = json.loads(FORMATION_MANIFEST.read_text(encoding="utf-8"))
    record["geometry"] = geometry
    path = scratch / f"编队页-{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return path


def test_a_manifest_that_disagrees_about_the_geometry_is_refused(scratch: Path) -> None:
    record = json.loads(FORMATION_MANIFEST.read_text(encoding="utf-8"))
    record["geometry"] = dict(record["geometry"], pitch=FORMATION_PITCH + 1)
    path = scratch / "编队页.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry"):
        load_cardfaces(path)


def test_a_manifest_that_disagrees_about_the_card_width_or_gap_is_refused(
    scratch: Path,
) -> None:
    """`card_width` 与 `gap` 这一轮进了「清单对常量」那一份 —— 以前这两个键
    **只被类型检查**，值本身没人对（于是 `250 + 21` 这种互相抵消的一对错值照样过）。"""
    for key, value in (("card_width", 250), ("gap", 21)):
        geometry = _declared_geometry()
        geometry[key] = value
        with pytest.raises(ValueError, match=key):
            load_cardfaces(_manifest_with_geometry(scratch, geometry))


def test_a_geometry_number_that_is_not_an_integer_is_refused(scratch: Path) -> None:
    """五个几何数都得是**整数像素**，一个都不能少。

    `271.0` 与 `271` 在 Python 里**比得出相等**（`262.0 == 262` 是 `True`），
    所以「清单对常量」那一层**接不住它** —— 只有类型那道接得住。红在哪一句上是有
    讲究的：这里必须是「要是整数像素」，不是「两处各说一套」。
    """
    for key in ("pitch", "left0", "card_width", "gap", "art_width"):
        geometry = _declared_geometry()
        geometry[key] = float(geometry[key])
        with pytest.raises(TypeError, match="要是整数像素"):
            load_cardfaces(_manifest_with_geometry(scratch, geometry))


def test_the_gate_checks_its_own_key_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    """闸门**对自己**也有一条检查：要求报全的键，必须与真正核过的键对得上。

    两个方向都堵，这里走**窄**的那一边：把 `_GEOMETRY_KEYS` 临时少一个 `art_width`。
    于是「报全」只按窄的清单要求（夹具六键齐全，过）、值那一段照样读
    `recorded["art_width"]`（过）—— 只剩这条自检能响。另一头（往 `_GEOMETRY_KEYS`
    里加一个没人核的键）由同一个集合比较兜住。

    这是 `monkeypatch` 的第二种用途：不是"把某个历史错值摆回来"，而是**把结构改坏、
    看闸门认不认**。它没有让实现绕过闸门 —— 改的是**被检查的那份清单**，闸门自己
    一行没动，而断言的是"它响了"。
    """
    monkeypatch.setattr(
        cardface,
        "_GEOMETRY_KEYS",
        tuple(key for key in cardface._GEOMETRY_KEYS if key != "art_width"),
    )
    with pytest.raises(ValueError, match="真正核过"):
        load_cardfaces(FORMATION_MANIFEST)


def test_the_two_rows_must_be_the_same_height(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**两行等高** —— 这条挡的是「清单与常量抄了同一个错」。

    历史：下行写的是 `(642, 1146, 5)`（**504** 高），而上行是 480 高。关键是
    **清单与常量两边都写 1146**，所以「清单对常量」那一层一声不吭 —— 当时全仓没有
    一点网抓得住它，多切进去的是卡下面那条队名栏。

    所以这里把**常量也改回那个错值**、清单跟着写。这是唯一能把那个错摆回来的办法：
    只改清单会先撞上「两处各说一套」，撞不到这一条。
    """
    wrong = ((137, 617, 6), (642, 1146, 5))
    monkeypatch.setattr(cardface, "FORMATION_ROWS", wrong)
    geometry = _declared_geometry()
    geometry["rows"] = [
        {"name": "上行", "y": [137, 617], "count": 6},
        {"name": "下行", "y": [642, 1146], "count": 5},
    ]
    with pytest.raises(ValueError, match="不一样高"):
        load_cardfaces(_manifest_with_geometry(scratch, geometry))


def test_a_card_width_that_cannot_hold_the_art_is_refused(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`FORMATION_ART_WIDTH < FORMATION_CARD_WIDTH` —— 立绘画是卡里挖掉右边那条
    底纹剩下的那块，它比整张卡宽就没有意义。

    与「两行等高」同一个形状：不变式**对常量算**，所以要真验它，得让**常量自己**
    不自洽、清单跟着写 —— 只改清单会先撞上「清单对常量」。
    """
    monkeypatch.setattr(cardface, "FORMATION_CARD_WIDTH", FORMATION_ART_WIDTH)
    monkeypatch.setattr(
        cardface, "FORMATION_GAP", FORMATION_PITCH - FORMATION_ART_WIDTH
    )
    geometry = _declared_geometry()
    geometry["card_width"] = FORMATION_ART_WIDTH
    geometry["gap"] = FORMATION_PITCH - FORMATION_ART_WIDTH
    with pytest.raises(ValueError, match="card_width"):
        load_cardfaces(_manifest_with_geometry(scratch, geometry))


def test_a_card_width_and_gap_that_do_not_add_up_to_the_pitch_are_refused(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`FORMATION_CARD_WIDTH + FORMATION_GAP == FORMATION_PITCH` —— 卡距就是
    「卡宽 + 缝」（`262 + 9 = 271`）。三个数里改坏一个，当场就报。"""
    monkeypatch.setattr(cardface, "FORMATION_GAP", FORMATION_GAP + 1)
    geometry = _declared_geometry()
    geometry["gap"] = FORMATION_GAP + 1
    with pytest.raises(ValueError, match="卡距"):
        load_cardfaces(_manifest_with_geometry(scratch, geometry))


def test_a_geometry_that_declares_only_some_keys_is_refused(scratch: Path) -> None:
    """**声明了就得报全** —— `{"rows": [...]}` 以前会**静默跳过**左缘/卡距/立绘宽，
    而「跳过」看起来与「通过」一模一样。"""
    with pytest.raises(KeyError, match="left0"):
        load_cardfaces(
            _manifest_with_geometry(scratch, {"rows": _declared_geometry()["rows"]})
        )


def test_a_manifest_that_declares_no_geometry_at_all_is_fine(scratch: Path) -> None:
    """「这份清单没有声明几何」该**跳过**，不是错 —— 与上一条是同一句话的两侧。"""
    record = json.loads(FORMATION_MANIFEST.read_text(encoding="utf-8"))
    record.pop("geometry")
    path = scratch / f"编队页-{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    assert len(load_cardfaces(path)) == FORMATION_CARDS


def test_a_manifest_that_lists_the_wrong_number_of_names_is_refused(scratch: Path) -> None:
    record = json.loads(FORMATION_MANIFEST.read_text(encoding="utf-8"))
    record["names"] = record["names"][:-1]
    path = scratch / "编队页.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="名字"):
        load_cardfaces(path)


def test_a_missing_manifest_is_loud(scratch: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_cardfaces(scratch / "没有这份清单.json")
    with pytest.raises(FileNotFoundError):
        load_formation_page(scratch / "没有这一帧.png")


# ---------------------------------------------------------------------------
# 判据：分数是什么、尺度扫哪几档、两条线落在哪
# ---------------------------------------------------------------------------


def test_the_match_score_is_the_same_pearson_as_the_glyph_chain() -> None:
    """`TM_CCOEFF_NORMED` 就是**把三通道拉平之后的 Pearson** —— 链用的判据。

    这一条把"分数是什么"钉死：`glyphs.similarity` 用的是同一个统计量（那边多一个
    ±2 px 的平移搜索），所以两条链的分数可以并排放。
    """
    rng = np.random.default_rng(3)
    art = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    template = art[13:33, 27:52].copy()
    result = cv2.matchTemplate(art, template, cv2.TM_CCOEFF_NORMED)
    other = template.astype(np.float64).ravel()
    # 自己跟自己比：正好 1.0 —— 这一位是精确的。
    assert float(result[13, 27]) == pytest.approx(1.0, abs=1e-5)
    assert float(result[13, 27]) == pytest.approx(
        _pearson(art[13:33, 27:52].astype(np.float64).ravel(), other), abs=1e-5
    )
    # 不是它的那些位置：与手算的 Pearson 对得上。容差按 **float32** 给 —— cv2 内部
    # 算的是 float32，而低相关的那些位置上分子是两个大数之差（灾难性相消），
    # 手算（float64）与它对不到 1e-6 那么紧。
    assert float(result[0, 0]) == pytest.approx(
        _pearson(art[0:20, 0:25].astype(np.float64).ravel(), other), abs=2e-3
    )


def test_the_scale_grid_is_the_measured_one() -> None:
    """区间是**拿真峰的分布定的**：实测真峰跨 **1.01…1.16**（`曲线.json` 的 26 档上，
    45 张有逐位真值的卡），两边各留一档 → **1.00…1.17（18 档）**。

    下沿为什么必须往下：`立绘两组` 的 `组2_备战席 #4`（`组2·塞雷娅`）峰在 **1.01**，
    而旧区间从 1.05 起 —— 它在旧区间里的最大分只有 **0.8309**，差一点过不了线。
    """
    assert IDENTITY_SCALE_MIN == 1.00
    assert IDENTITY_SCALE_MAX == 1.17
    assert IDENTITY_SCALE_STEP == 0.01
    assert IDENTITY_SCALES[0] == 1.00
    assert IDENTITY_SCALES[-1] == 1.17
    assert len(IDENTITY_SCALES) == 18
    assert list(IDENTITY_SCALES) == sorted(set(IDENTITY_SCALES))


def test_the_two_lines_sit_in_the_measured_gap() -> None:
    """规格那条实测：真匹配 **0.940–0.987**（余量 ≥ 0.392）、不是它 **≤ 0.575**
    （余量 ≤ 0.084）。两条线都必须落在这两段空档**里面**。"""
    assert 0.575 < ROSTER_SOLID_MATCH_SCORE < 0.940
    assert 0.084 < ROSTER_AMBIGUITY_MARGIN < 0.392


def test_a_template_taken_at_a_scale_is_found_at_that_scale() -> None:
    """合成的：立绘里 `100s × 100s` 那块缩成 100×100 当模板 —— 链要**按那个尺度**
    把它找回来，而且分数要**远远**高过"不是它"那一档（≤ 0.575）。"""
    rng = np.random.default_rng(11)
    art = rng.integers(0, 256, (480, 214, 3), dtype=np.uint8)
    for scale in (1.05, 1.10, 1.15):
        side = int(round(100 * scale))
        block = art[100 : 100 + side, 50 : 50 + side]
        query = np.ascontiguousarray(
            cv2.resize(block, (100, 100), interpolation=cv2.INTER_AREA)
        )
        ranked = roster_scores(query, [("甲", art)])
        assert len(ranked) == 1
        assert ranked[0][2] == pytest.approx(scale)
        assert ranked[0][0] > 0.70


def test_the_identity_region_is_the_measured_box() -> None:
    assert (IDENTITY_BOX_X, IDENTITY_BOX_Y, IDENTITY_BOX_WIDTH, IDENTITY_BOX_HEIGHT) == (
        52,
        76,
        100,
        100,
    )
    card = squad_card_rect(0, PITCH)
    centre = float(card.x) + float(card.width) / 2.0
    rect = identity_region(centre, pitch=PITCH)
    assert rect == PixelRect(
        float(card.x) + IDENTITY_BOX_X,
        float(card.y) + IDENTITY_BOX_Y,
        IDENTITY_BOX_WIDTH,
        IDENTITY_BOX_HEIGHT,
    )
    # 框必须**整块落在卡里** —— 越出去就不是"这张卡的身份区"了。
    assert rect.x >= card.x and rect.x + rect.width <= card.x + card.width
    assert rect.y >= card.y and rect.y + rect.height <= card.y + card.height


# ---------------------------------------------------------------------------
# 三档降级
# ---------------------------------------------------------------------------


def test_the_harvest_frames_are_read_exactly(
    identities: dict[str, tuple[SlotValue, ...]],
) -> None:
    """`shot01` · `cd60` 的**每一格**：给对的那些要给对，"库里没有"的那些要落未知。"""
    recorded = _recorded_truth()
    checked = 0
    for name in TRUTH_FRAMES:
        want = recorded[name]
        got = identities[name]
        assert len(got) == len(want), f"{name}：卡位数与取材表对不上"
        for index, value in enumerate(got):
            shown = _name_of(value)
            if want[index] == "未知":
                assert shown is None, f"{name}#{index} 落的是 {shown}，取材表说是未知"
            else:
                assert shown == want[index], f"{name}#{index} 读到 {shown}，真值是 {want[index]}"
            checked += 1
    assert checked == 23


def test_the_device_card_is_unknown_not_a_device(
    identities: dict[str, tuple[SlotValue, ...]],
) -> None:
    """`shot01` 位 6 是**装置卡**（画面上是准星）。库里没有它，所以落未知 ——
    **不许**落成"装置"（规格明写这一条）。"""
    value = identities["shot01"][6]
    assert _name_of(value) is None
    assert value.payload == Unknown()


def test_a_card_that_is_not_in_the_library_is_unknown(
    identities: dict[str, tuple[SlotValue, ...]],
) -> None:
    """`cd60` 位 8 是那一局队伍里的另一个人 —— 库与战斗帧**不同源**，这是常态。"""
    assert _name_of(identities["cd60"][8]) is None


def test_every_red_capped_card_scores_below_the_line(
    red_regions: tuple[tuple[str, int, np.ndarray], ...],
    faces: tuple[tuple[str, np.ndarray], ...],
) -> None:
    """**旧取区**（身份区那个方框）在那 14 张红罩卡上：没有一张过得了最高分那条线。

    这一条量的是 `roster_scores` 在**旧框**上的读数 —— 它就是「为什么要换取区」
    那条实测（旧框剩下的那四个角把所有条目一起抬起来）。**它不代表今天的入口怎么落**：
    入口现在对红罩卡走 `redcap_region`（见 §红罩卡那一组那几条）。
    """
    assert len(red_regions) == 14
    for name, index, region in red_regions:
        ranked = roster_scores(region, faces)
        best = ranked[0][0]
        margin = best - ranked[1][0]
        assert best < ROSTER_SOLID_MATCH_SCORE, f"{name}#{index} 最高分 {best:.4f} 过线了"
        assert margin < ROSTER_AMBIGUITY_MARGIN, f"{name}#{index} 余量 {margin:.4f} 够大了"
        # 规格那句：红罩卡实测全部 ≤ 0.371。
        assert best <= 0.371, f"{name}#{index} 最高分 {best:.4f} 越过 0.371"
        # **单条那条路也不可能接受它**：那条路的判据是"某一条库过线"，而这里
        # **每一条**都不过线 —— 所以这一条同时是「零假事实面」里红罩那一面。
        assert all(score < ROSTER_SOLID_MATCH_SCORE for score, _, _ in ranked), (
            f"{name}#{index} 有一条库过线了"
        )


def test_a_lone_red_capped_card_is_read_by_the_entry(
    lone_red_card: tuple[SlotValue, ...],
) -> None:
    """从**真入口**上再验一次：`cd01` 最右那张是红罩卡 —— 它**不再一律未知**。

    这一格只有一张卡，十几秒就够。**它的名字没有真值可对**（`cd01` 那一帧不在点名
    那份真值表里），所以这里钉的是**形态与那个分数**：给确切值、而且带分数。
    **真值那一条**在 `test_a_red_capped_card_gives_the_name_...` 上。

    **分数从 0.991018 变成 0.943155**（第六单，红罩那一刀）：那条路从前是"整张滑 18 档、
    报全局最大那一档的分"（1.10 档、0.991018），现在与干净卡那条路**同一条规矩** ——
    "报出来的分是**第一个满足判据的那一档**上的分"（这里 1.08 档、0.943155）。
    名字与判定一个都没变（第六单那张网：633 张 · 名字 0 变）。
    """
    assert len(lone_red_card) == 1
    value = lone_red_card[0]
    assert isinstance(value, Valued)
    assert isinstance(value.payload, Exact)
    assert value.payload.value == "斩业星熊"
    assert value.confidence == pytest.approx(0.943155, abs=1e-6)


def test_a_red_capped_card_gives_the_name_the_pair_tables_record(
    redcap_run: dict[str, tuple[Frame, tuple[SlotValue, ...], tuple[bool | None, ...]]],
    redcap_truth: dict[tuple[str, int], str],
) -> None:
    """红罩卡走**同一个入口**读出**确切名字**，而且那一格确实判成了"在再部署中"。

    真值来自两份取材表的 `under_the_cap`（不是目视）。**分数也要在**：它就在槽值里
    （`Valued(Exact(名字), 分数)`），而且它就是定下这次认出的那个分数。
    """
    assert len(redcap_truth) == 5
    for (name, index), expected in sorted(redcap_truth.items()):
        _frame, values, flags = redcap_run[name]
        assert flags[index] is True, f"{name}#{index} 没判成在再部署中"
        value = values[index]
        assert _name_of(value) == expected, f"{name}#{index}"
        assert value.confidence is not None, f"{name}#{index} 确切值没有分数"


def test_a_red_capped_card_that_does_not_match_lands_unknown_without_a_score(
    redcap_run: dict[str, tuple[Frame, tuple[SlotValue, ...], tuple[bool | None, ...]]],
) -> None:
    """红罩卡里**读不出的那些**落**未知**，而且**没有分数** —— 不为它另开形态。

    **这一单之后这五帧上一张都不剩**（第五单重录）。从前是 **4 张**：它们的分在
    0.9655–0.9762、余量 0.1980–0.2040，旧那条**固定**余量线 0.25 把它们挡住了。
    判据换成"余量随分数走"（`次高 <= 2 × 最高 - 1.09`）之后，0.97 那一带要求的分差
    只有 0.10 —— 它们**全都够**，所以现在每一张红罩卡都读得出来。
    **形态那一半照样钉着**：下面的循环要求每一张红罩卡都是 `Valued(Exact)` 而且带分数
    （`Unknown` 不带分数由 §落法那一条钉）。
    """
    unknown = [
        (name, index)
        for name, (_frame, values, flags) in redcap_run.items()
        for index, flag in enumerate(flags)
        if flag is True and _name_of(values[index]) is None
    ]
    assert unknown == [], unknown
    for name, (_frame, values, flags) in redcap_run.items():
        for index, flag in enumerate(flags):
            if flag is not True:
                continue
            value = values[index]
            assert isinstance(value, Valued), f"{name}#{index}"
            assert value.payload != Unknown(), f"{name}#{index} 红罩卡没读出来"
            assert value.confidence is not None, f"{name}#{index} 精确值没有分数"


def test_the_clean_cards_of_those_frames_are_untouched(
    redcap_run: dict[str, tuple[Frame, tuple[SlotValue, ...], tuple[bool | None, ...]]],
    redcap_clean_reference: dict[tuple[str, int], SlotValue],
) -> None:
    """**同一帧上干净卡的读数逐位不变** —— "改动前"按旧口径重算，**逐位**比。

    比的是**相等的槽值**（`Valued` 是冻结数据类，分数按浮点逐位相等），不是"名字一样"：
    干净卡一个像素都没换到红罩那条路上。那五帧上一共 34 格干净卡。
    """
    compared = 0
    for name, (_frame, values, flags) in redcap_run.items():
        for index, flag in enumerate(flags):
            if flag is True:
                continue
            assert values[index] == redcap_clean_reference[(name, index)], f"{name}#{index}"
            compared += 1
    assert compared == 34, compared


def test_only_the_red_capped_slots_take_the_new_region(
    monkeypatch: pytest.MonkeyPatch,
    redcap_frames: dict[str, Frame],
    faces: tuple[tuple[str, np.ndarray], ...],
) -> None:
    """**只在红罩卡上**走新取区 —— 量的是"谁收到哪一块"，不是墙钟。

    干净卡那条扫法（`_layered_scores`）与掩码那把尺子的分层入口
    （`MaskedRuler.rows`）各包一层记一笔：`cd14` 那一帧（6 红罩 / 4 干净）上，
    掩码那把只该在 **6 个格子**上开工、形状是圆盘取区那一块；干净那条只该收到 **4** 次、
    形状是身份区那个框。

    **数"格子"数的是锚点层那一次**（`pixels=_ANCHOR_PIXELS`）：红罩那一格一层一次调用
    （第六单把"整张滑"换成了三层），所以按调用次数数会数成十几 —— 一个格子只算一次锚点层。

    （从前包的是 `roster_scores` 与 `MaskedRuler.scores`；第五单换成
    `_layered_scores` 与 `scores_and_positions`；第六单红罩那条换成 `rows` 的三层。）
    """
    from ark.engine.vision import margin
    import ark.engine.vision.roster as roster_module

    seen: dict[str, list[tuple[int, ...]]] = {"box": [], "disc": []}
    slots = {"disc": 0}
    real_box = roster_module._layered_scores
    real_disc = margin.MaskedRuler.rows

    def box_spy(region: np.ndarray, library, **kwargs):
        seen["box"].append(tuple(region.shape))
        return real_box(region, library, **kwargs)

    def disc_spy(self, region: np.ndarray, **kwargs):
        seen["disc"].append(tuple(region.shape))
        if kwargs.get("pixels") == roster_module._ANCHOR_PIXELS:
            slots["disc"] += 1
        return real_disc(self, region, **kwargs)

    monkeypatch.setattr(roster_module, "_layered_scores", box_spy)
    monkeypatch.setattr(margin.MaskedRuler, "rows", disc_spy)
    values = read_squad_identities(redcap_frames["cd14"], faces, pitch=PITCH)
    assert len(values) == 10
    box = margin.IDENTITY_DISC_BOX
    assert slots["disc"] == 6, seen
    assert len(seen["box"]) == 4, seen
    assert set(seen["disc"]) == {(box[3], box[2], 3)}
    assert set(seen["box"]) == {(IDENTITY_BOX_HEIGHT, IDENTITY_BOX_WIDTH, 3)}


def test_every_entry_is_compared_on_one_region() -> None:
    """每一对（卡, 库条目）都在**同一块取区**上比：框与掩码**只有一处定义**、形状对得上。

    **取区与库、与卡位都无关**：它只由卡心与卡距定 —— 所以同一张卡上的每条库、以及
    单条那条路上的那一条，看的都是同一块。而它**不是**身份区那个框（两块分开记）。
    """
    from ark.engine.vision import margin

    box, mask = margin.identity_disc_region()
    left = redcap_region(1000.0, pitch=PITCH)
    right = redcap_region(2500.0, pitch=PITCH)
    assert (left.width, left.height) == (right.width, right.height) == (box[2], box[3])
    assert mask.shape == (box[3], box[2])
    assert int(mask.sum()) == 3986
    assert right.x - left.x == pytest.approx(1500.0)
    assert left.y == pytest.approx(float(squad_card_rect(0, PITCH).y) + box[1])
    assert box[:2] != (IDENTITY_BOX_X, IDENTITY_BOX_Y)




@pytest.fixture(scope="module")
def redcap_frames() -> dict[str, Frame]:
    """红罩那一批的 8 帧（贴回画布之后）。"""
    return {case.name: frame_from_raw(case.raw()) for case in load_batch(REDCAP_BATCH).cases}


@pytest.fixture(scope="module")
def redcap_truth() -> dict[tuple[str, int], str]:
    """红罩卡的**逐位真值**：`(帧名, 卡下标) -> 名字` —— 两份取材表的 `under_the_cap`。

    `红罩/取材.json` 那 4 对，加 `预处理/取材.json` 里落在同一帧上的那一对（`cd23_6`）
    —— 两份都是"同一张卡后来干净了那一帧"读出来的。
    """
    found: dict[tuple[str, int], str] = {}
    for path in (
        REDCAP_BATCH / "取材.json",
        REPO / "tests" / "帧样本" / "预处理" / "取材.json",
    ):
        record = json.loads(path.read_text(encoding="utf-8"))
        for pair in record["pairs"]:
            frame = Path(pair["capped"]["file"]).stem.split("_")[0]
            if frame in REDCAP_TRUTH_FRAMES:
                found[(frame, int(pair["capped"]["index"]))] = pair["under_the_cap"]
    return found


@pytest.fixture(scope="module")
def redcap_run(
    redcap_frames: dict[str, Frame], faces: tuple[tuple[str, np.ndarray], ...]
) -> dict[str, tuple[Frame, tuple[SlotValue, ...], tuple[bool | None, ...]]]:
    """那五帧上**今天的入口**的读数 + 每一格"在不在再部署中"（跑一次，用例只读结果）。"""
    out: dict[str, tuple[Frame, tuple[SlotValue, ...], tuple[bool | None, ...]]] = {}
    for name in REDCAP_TRUTH_FRAMES:
        frame = redcap_frames[name]
        top = float(squad_card_rect(0, PITCH).y)
        values = read_squad_identities(frame, faces, pitch=PITCH)
        flags = tuple(
            redeploying(frame, centre, card_top=top)
            for centre, _ in squad_card_centres(frame, pitch=PITCH)
        )
        out[name] = (frame, values, flags)
    return out


@pytest.fixture(scope="module")
def redcap_clean_reference(
    redcap_run: dict[str, tuple[Frame, tuple[SlotValue, ...], tuple[bool | None, ...]]],
    faces: tuple[tuple[str, np.ndarray], ...],
) -> dict[tuple[str, int], SlotValue]:
    """同一批帧上**干净卡**的"改动前"值 —— 按**旧口径**重算一遍。

    旧口径 = 身份区那个框 + `roster_scores` + 那两条线（接红罩那条路之前，
    `read_squad_identities` 对每一格做的就是这些）。带一个按取区字节的缓存：那五帧的
    干净卡里有好几块逐字节相同。
    """
    cache: dict[bytes, tuple[tuple[float, str, float], ...]] = {}
    out: dict[tuple[str, int], SlotValue] = {}
    for name, (frame, _values, flags) in redcap_run.items():
        centres = squad_card_centres(frame, pitch=PITCH)
        for index, flag in enumerate(flags):
            if flag is True:
                continue
            region = np.ascontiguousarray(
                frame.crop(identity_region(centres[index][0], pitch=PITCH))
            )
            key = region.tobytes()
            if key not in cache:
                cache[key] = roster_scores(region, faces)
            ranked = cache[key]
            best, entry, _ = ranked[0]
            runner = ranked[1][0] if len(ranked) > 1 else -2.0
            if best >= ROSTER_SOLID_MATCH_SCORE and best - runner >= ROSTER_AMBIGUITY_MARGIN:
                out[(name, index)] = Valued(Exact(entry), best)
            else:
                out[(name, index)] = Valued(Unknown(), None)
    return out


def test_the_chain_only_produces_the_two_shapes(
    identities: dict[str, tuple[SlotValue, ...]],
) -> None:
    for name, cards in identities.items():
        for index, value in enumerate(cards):
            assert isinstance(value, Valued), f"{name}#{index} 是 {type(value).__name__}"
            assert isinstance(value.payload, (Exact, Unknown)), f"{name}#{index}"


# ---------------------------------------------------------------------------
# 单条查一位：判据 · 两簇空档 · 两路关系 · 零假事实面
# ---------------------------------------------------------------------------


def test_the_line_alone_separates_the_two_clusters(
    single_scores: dict[str, dict[str, tuple[tuple[float, int], ...]]],
) -> None:
    """**只靠一个分数的判据自己站得住** —— 「库里只有一条」那个退化情形的读数。

    库里只有一条时没有次高分可比，判据退化成只看最高分（`read_squad_identities`
    里那句注释）。这里把它**量出来**：逐 **(卡, 库条目)** 对分两簇 ——

    - **真匹配**：那张卡的真值就是这一条；
    - **非匹配**：那张卡的真值不是这一条（含真值未知的那些卡）。

    **读数每一次都是现算的**（用户裁定 8）：`single_scores` 夹具拿 `squad_card_scores`
    在 `shot01` + `cd60` 两帧上现跑（23 张卡 × 11 条 = 253 对），断言只看**这条链这一次
    给出来的关系**，不跟任何录下来的数比 —— 录下来的数会随扫法改动而变，而"跟旧读数比"
    正是断言坏掉的那种形态（用户裁定 10 同一条：改成**形状断言**并写明为什么）。
    所以下面钉的是**四条形状**，不是四个数：

    1. 两簇的**条数**是结构性的（16 / 237），钉住；
    2. **非匹配天花板 < 线 < 真匹配地板** —— 判据自己站得住；
    3. **空档够宽**（> 0.2），两簇不重叠；
    4. **线偏向「正确」那一侧**：它离真匹配那簇比离非匹配那簇近 ——
       认错是假事实，认不出只是未知。

    **十一档时代那份读数（真匹配 0.9618–0.9921 · 非匹配 ≤ 0.5622 · 空档 0.3996）
    只能当历史**（用户裁定 8）：它是尺度区间还是 `1.05…1.15`、余量还是固定 `0.25`、
    报出来的分还是**全局最大分**的时候量的，今天的链复现不出来。历史与今天的对照在
    `tests/帧样本/先粗后细/README.md`，**不在这个文件里**。

    真匹配那一头比十一档时代**低**：单条那条路只有一条库、没有次高可比，判据退化成
    "过 0.85 就算"，所以**第一个过线的那一档**就是答案 —— 报出来的分是**第一个过线的分**，
    不是全局最大分（与用户裁定 9 同一条口径）。
    """
    want = _recorded_truth()
    hits: list[float] = []
    misses: list[float] = []
    for name in TRUTH_FRAMES:
        for who, scored in single_scores[name].items():
            for index, truth in enumerate(want[name]):
                score = _card_score(scored, index)
                (hits if truth == who else misses).append(score)
    assert (len(hits), len(misses)) == (16, 237)
    ceiling, floor, line = max(misses), min(hits), ROSTER_SOLID_MATCH_SCORE
    assert ceiling < line < floor, (ceiling, line, floor)
    assert floor - ceiling > 0.2, (ceiling, floor)
    # 线偏向「正确」那一侧：离真匹配那簇更近 —— 认错是假事实，认不出只是未知。
    assert line - ceiling > floor - line, (ceiling, line, floor)


def test_the_single_path_and_the_full_scan_give_the_same_number(
    identities: dict[str, tuple[SlotValue, ...]],
    single_scores: dict[str, dict[str, tuple[tuple[float, int], ...]]],
) -> None:
    """**地基**：同一个 (名字, 卡)，两条路给的是**同一个数**。

    全扫那条路的数就是 `Valued(Exact(名字), 置信度)` 里的**置信度** —— 那正是
    "定下这次读数的那个分数"。所以这一条不必另算一遍全扫：全扫已经把它交回来了。
    """
    checked = 0
    for name in TRUTH_FRAMES:
        for index, value in enumerate(identities[name]):
            if not isinstance(value.payload, Exact):
                continue
            who = str(value.payload.value)
            assert _card_score(single_scores[name][who], index) == value.confidence, (
                f"{name}#{index} {who}：单条给 {_card_score(single_scores[name][who], index)}，"
                f"全扫给 {value.confidence}"
            )
            checked += 1
    assert checked == 16


def test_no_library_entry_reaches_the_line_on_a_card_the_full_scan_called_unknown(
    identities: dict[str, tuple[SlotValue, ...]],
    single_scores: dict[str, dict[str, tuple[tuple[float, int], ...]]],
) -> None:
    """**"单条不可能接受那张卡"的直接形式**：全扫落未知的卡上，没有一条库够得到线。

    它**不是**"全扫落未知"的推论：全扫落未知有两个原因（最高分不过线 / 余量不够），
    只有第一个才让这句话成立。所以这里是真的量一遍，不是引一句。
    """
    checked = 0
    for name in TRUTH_FRAMES:
        for index, value in enumerate(identities[name]):
            if _name_of(value) is not None:
                continue
            for who, scored in single_scores[name].items():
                score = _card_score(scored, index)
                assert score < ROSTER_SOLID_MATCH_SCORE, (
                    f"{name}#{index}：全扫说未知，单条查 {who} 却给了 {score:.4f}"
                )
            checked += 1
    assert checked == 7


def test_the_two_paths_never_disagree_about_who_is_on_a_card(
    identities: dict[str, tuple[SlotValue, ...]],
    single_scores: dict[str, dict[str, tuple[tuple[float, int], ...]]],
) -> None:
    """**不许认错人**：单条说「n 在 k」，全扫在 k 上只能是 **n 或未知**。

    真帧上 16 位各自只有一张过线，所以这里同时钉住了"全扫说未知而单条说 n"
    是允许的（那只是单条更敢说），而"全扫说是别人"是不允许的。
    """
    checked = 0
    for name in TRUTH_FRAMES:
        for who, scored in single_scores[name].items():
            for index in _accepted(scored):
                shown = _name_of(identities[name][index])
                assert shown in (who, None), (
                    f"{name}#{index}：单条说「{who} 在这」，全扫说是 {shown}"
                )
                checked += 1
    assert checked == 16


def test_the_single_path_does_not_miss_what_the_full_scan_named(
    identities: dict[str, tuple[SlotValue, ...]],
    single_scores: dict[str, dict[str, tuple[tuple[float, int], ...]]],
) -> None:
    """**不许漏**：全扫给了确切 n 的那张卡，单条查 n 必须**指回同一张卡**。"""
    checked = 0
    for name in TRUTH_FRAMES:
        for index, value in enumerate(identities[name]):
            who = _name_of(value)
            if who is None:
                continue
            assert _accepted(single_scores[name][who]) == [index], (
                f"{name}#{index}：全扫说是 {who}，单条查 {who} 指到 "
                f"{_accepted(single_scores[name][who])}"
            )
            checked += 1
    assert checked == 16


def test_the_single_path_accepts_at_most_one_card(
    single_scores: dict[str, dict[str, tuple[tuple[float, int], ...]]],
) -> None:
    """**不许指两张**：同一帧里，单条查 n 接受的卡**至多一张**。

    真帧上这条一次都没触发；"两张都过线"那一支由下面那条合成的孪生卡钉着。
    """
    for name in TRUTH_FRAMES:
        for who, scored in single_scores[name].items():
            assert len(_accepted(scored)) <= 1, f"{name} {who}：{_accepted(scored)}"


def test_the_single_path_never_accepts_a_card_that_is_someone_else(
    single_scores: dict[str, dict[str, tuple[tuple[float, int], ...]]],
) -> None:
    """**零假事实面**：真值已知、而且**不是这个名字**的卡上，单条查询一次都不接受。

    两帧的**每一张卡**都在这里过一遍，覆盖三类：真值是真名但不是 n 的（装置卡 ·
    不同源那一位都落这一类 —— 它们的真值在取材表里是「未知」，而"未知"不等于任何一个
    库名字）、以及全扫落未知的那些。

    **前置**：这两帧**一张红罩卡都没有**，所以"未知"的每一张都**确实不在库里**，
    而不是"可能就是他、只是没读出来"。红罩那一面在 `test_every_red_capped_card_...`
    里量（那边是"每一条库都不过线"，所以单条那条路也接受不了）。
    """
    red_frames = {name for name, _ in _red_card_slots()}
    assert not (red_frames & set(TRUTH_FRAMES)), f"真值帧里出现了红罩卡：{red_frames}"

    want = _recorded_truth()
    accepted = 0
    for name in TRUTH_FRAMES:
        for who, scored in single_scores[name].items():
            for index in _accepted(scored):
                assert want[name][index] == who, (
                    f"{name}#{index} 真值是 {want[name][index]}，单条查 {who} 却接受了它"
                )
                accepted += 1
    assert accepted == 16


def test_find_squad_card_is_the_judge_plus_the_line(
    frames: dict[str, Frame],
    faces: tuple[tuple[str, np.ndarray], ...],
    single_scores: dict[str, dict[str, tuple[tuple[float, int], ...]]],
) -> None:
    """真入口与判据的关系：`find_squad_card` == 「判据里过线的**恰一张**」。

    在**只留最右两张卡**的真帧上把每一位都走一遍 —— 便宜，而且它同时是
    "零张 / 一张"两支。**期望值不另算**：清掉的是卡 1 左缘往左的像素，卡 0 与卡 1
    的身份区一个像素没动，所以它们的分数就是夹具里整帧那两个数。
    """
    for name in TRUTH_FRAMES:
        sliced = _rightmost(frames[name], 2)
        centres = squad_card_centres(sliced, pitch=PITCH)
        assert len(centres) == 2
        # 期望值是从**整帧**那份矩阵里取的，所以这两格的判据输入必须一个像素没动 ——
        # 抄歪了的话下面每一条都会对不上，而原因会看起来像"入口错了"。
        for centre, _ in centres:
            assert np.array_equal(
                np.array(sliced.crop(identity_region(centre, pitch=PITCH))),
                np.array(frames[name].crop(identity_region(centre, pitch=PITCH))),
            )
        for who, _ in faces:
            hits = [
                index
                for index in (0, 1)
                if _card_score(single_scores[name][who], index)
                >= ROSTER_SOLID_MATCH_SCORE
            ]
            assert find_squad_card(sliced, who, faces, pitch=PITCH) == (
                hits[0] if len(hits) == 1 else None
            ), who


def test_the_single_path_refuses_when_two_cards_reach_the_line(
    frames: dict[str, Frame], faces: tuple[tuple[str, np.ndarray], ...]
) -> None:
    """**不许指两张**：两格过线就落 `None` —— 证据不够时不许挑一张。

    真帧上这条一次都没触发（16 位各自只有一张过线），所以这里**造**一个：把卡 0 的
    身份区抄到卡 1 上。两格的**判据输入一模一样**，于是两张都过线。
    """
    frame = frames["shot01"]
    assert find_squad_card(frame, "维什戴尔", faces, pitch=PITCH) == 0
    centres = [centre for centre, _ in squad_card_centres(frame, pitch=PITCH)]
    first = identity_region(centres[0], pitch=PITCH)
    second = identity_region(centres[1], pitch=PITCH)
    pixels = np.array(frame.pixels)
    pixels[
        int(second.y) : int(second.y) + IDENTITY_BOX_HEIGHT,
        int(second.x) : int(second.x) + IDENTITY_BOX_WIDTH,
    ] = pixels[
        int(first.y) : int(first.y) + IDENTITY_BOX_HEIGHT,
        int(first.x) : int(first.x) + IDENTITY_BOX_WIDTH,
    ]
    twin = Frame(canvas=frame.canvas, pixels=pixels)
    assert len(squad_card_centres(twin, pitch=PITCH)) == len(centres)
    scored = squad_card_scores(twin, "维什戴尔", faces, pitch=PITCH)
    # 这一条同时是前置：抄歪了像素，两格的分就不会一样，红在这里而不是红在下一行。
    assert _accepted(scored) == [0, 1]
    assert find_squad_card(twin, "维什戴尔", faces, pitch=PITCH) is None


def test_the_single_path_cost_does_not_grow_with_the_library(
    frames: dict[str, Frame],
    faces: tuple[tuple[str, np.ndarray], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**次数与库长无关这条已经死了**（用户裁定 8）：这一版**停不停取决于库里有什么** ——
    判据是"这一张卡上最高与次高之间的关系"，库不一样、停在哪一档就不一样。

    死的只有这一条。**还真的**有两件：

    - **次数有上界** `卡位 × 档数 × 库长`（粗两层 + 每一次全分辨率调用都各算一次）；
    - **每个条目被同样对待** —— 同一张卡上，所有条目在同一个档上各算一次，没有谁被跳过、
      也没有谁被多算（这正是这里能拿"库 2 条 vs 11 条"比的原因）。

    **两个库长给同一个数**这件事仍然成立：`squad_card_scores` 只吃**一条**库
    （名字对上的那一条），所以库长在这里只改变"取出来的是哪一条"，不改变算多少。
    钉的是次数、不是墙钟 —— 同代码在这台机器上能差 40%。
    """
    calls: list[int] = []
    real = cv2.matchTemplate

    def counted(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(cv2, "matchTemplate", counted)
    frame = frames["shot01"]
    target = next(entry for entry in faces if entry[0] == "维什戴尔")
    fillers = [entry for entry in faces if entry[0] != target[0]]
    counts = []
    for size in (2, 11):
        calls.clear()
        squad_card_scores(
            frame, "维什戴尔", [target] + fillers[: size - 1], pitch=PITCH
        )
        counts.append(len(calls))
    cards = len(squad_card_centres(frame, pitch=PITCH))
    # 库 2 条与 11 条给**同一个数**：单条那条路只吃一条库，与库里还有谁无关。
    assert counts[0] == counts[1]
    # 上界：每个卡位至多 `档数`（粗两层）+ `档数`（全分辨率每一档各一次）次调用。
    assert counts[0] <= cards * len(IDENTITY_SCALES) * 3


def test_a_name_that_is_not_in_the_library_is_loud(
    frames: dict[str, Frame], faces: tuple[tuple[str, np.ndarray], ...]
) -> None:
    """**拿错库是配置错，不是「他不在」** —— 静默吐一个「没能指出」就是把一次
    配置错伪装成一次正常读数。报错时要把库里有哪几个名字打出来。
    """
    with pytest.raises(KeyError, match="库里没有"):
        find_squad_card(frames["shot01"], "不存在的干员", faces, pitch=PITCH)
    with pytest.raises(KeyError, match="库里没有"):
        squad_card_scores(frames["shot01"], "不存在的干员", faces, pitch=PITCH)


def test_a_duplicated_name_in_the_library_is_loud(
    frames: dict[str, Frame], faces: tuple[tuple[str, np.ndarray], ...]
) -> None:
    """同名两条：一位干员一条 —— 两条同名会让「他在第几张卡」没有唯一答案。"""
    target = next(entry for entry in faces if entry[0] == "维什戴尔")
    with pytest.raises(ValueError, match="维什戴尔"):
        squad_card_scores(frames["shot01"], "维什戴尔", [target, target], pitch=PITCH)


def test_the_single_path_has_the_same_guards(
    frames: dict[str, Frame], faces: tuple[tuple[str, np.ndarray], ...]
) -> None:
    """闸门与全扫那条路同一套（帧 · 库 · 卡距），另加名字那一条。"""
    frame = frames["shot01"]
    with pytest.raises(TypeError, match="一帧"):
        find_squad_card(frame.pixels, "维什戴尔", faces, pitch=PITCH)
    with pytest.raises(TypeError, match="卡距"):
        find_squad_card(frame, "维什戴尔", faces, pitch="207.5")
    with pytest.raises(TypeError, match="卡距"):
        find_squad_card(frame, "维什戴尔", faces, pitch=True)
    with pytest.raises(ValueError, match="卡距"):
        find_squad_card(frame, "维什戴尔", faces, pitch=0)
    with pytest.raises(ValueError, match="空"):
        find_squad_card(frame, "维什戴尔", (), pitch=PITCH)
    with pytest.raises(TypeError, match="名字"):
        find_squad_card(frame, 7, faces, pitch=PITCH)
    with pytest.raises(TypeError, match="名字"):
        find_squad_card(frame, "", faces, pitch=PITCH)


# ---------------------------------------------------------------------------
# 闸门
# ---------------------------------------------------------------------------


def test_an_empty_library_is_loud(frames: dict[str, Frame]) -> None:
    """空库不是「库里没有这个人」，是**没建库** —— 报错，不静默吐一列未知。"""
    centre = squad_card_centres(frames["shot01"], pitch=PITCH)[0][0]
    region = np.array(frames["shot01"].crop(identity_region(centre, pitch=PITCH)))
    with pytest.raises(ValueError, match="空"):
        read_squad_identities(frames["shot01"], (), pitch=PITCH)
    with pytest.raises(ValueError, match="空"):
        roster_scores(region, [])


def test_a_malformed_face_is_loud(frames: dict[str, Frame]) -> None:
    centre = squad_card_centres(frames["shot01"], pitch=PITCH)[0][0]
    region = np.array(frames["shot01"].crop(identity_region(centre, pitch=PITCH)))
    good = np.zeros((480, 214, 3), dtype=np.uint8)
    with pytest.raises(TypeError, match="立绘块"):
        roster_scores(region, [("甲", good.astype(np.float32))])
    with pytest.raises(TypeError, match="立绘块"):
        roster_scores(region, [("甲", np.zeros((480, 214), dtype=np.uint8))])
    with pytest.raises(TypeError, match="名字"):
        roster_scores(region, [("", good)])
    with pytest.raises(TypeError, match="条要是"):
        roster_scores(region, [("甲", good, good)])
    with pytest.raises(TypeError, match="身份区"):
        roster_scores(region[:, :, 0], [("甲", good)])


def test_a_face_too_small_for_any_template_is_loud(frames: dict[str, Frame]) -> None:
    """库块**连最小那一档都装不下**才报错 —— 第五单改的口径。

    区间扩到 `1.00…1.17` 之后，"某几档装不下"是**常态**（`余量` 那条 **186 宽**的横带
    就装不下 1.16/1.17）：那几档**跳过并记进名单**，不是把整块取区判死。
    只有"一档都装不下"（库块比一整块备战席卡还小）才是配置错。
    110×110 从前会报错，现在不报 —— 它只是用不上 1.11 以上那几档。
    """
    centre = squad_card_centres(frames["shot01"], pitch=PITCH)[0][0]
    region = np.array(frames["shot01"].crop(identity_region(centre, pitch=PITCH)))
    with pytest.raises(ValueError, match="一档都装不下"):
        roster_scores(region, [("小", np.zeros((90, 90, 3), dtype=np.uint8))])
    # 110×110 装得下 1.00…1.10 那几档 —— **不报错**，走那几档。
    ranked = roster_scores(region, [("小", np.zeros((110, 110, 3), dtype=np.uint8))])
    assert len(ranked) == 1


def test_a_bad_pitch_is_loud(
    frames: dict[str, Frame], faces: tuple[tuple[str, np.ndarray], ...]
) -> None:
    with pytest.raises(TypeError, match="卡距"):
        read_squad_identities(frames["shot01"], faces, pitch="207.5")
    with pytest.raises(TypeError, match="卡距"):
        read_squad_identities(frames["shot01"], faces, pitch=True)
    with pytest.raises(ValueError, match="卡距"):
        read_squad_identities(frames["shot01"], faces, pitch=0)
    with pytest.raises(TypeError, match="一帧"):
        read_squad_identities(frames["shot01"].pixels, faces, pitch=PITCH)


def test_a_duplicated_face_kills_the_margin(
    frames: dict[str, Frame], faces: tuple[tuple[str, np.ndarray], ...]
) -> None:
    """库里同一条出现两次（换个名字）—— 余量正好塌成 **0**，判据**不认**。

    这正是"只看最高分会把没认出来的那些也挑一个"那条：最高分还是 0.98，可是
    最像的两个一样像，证据就不够了。
    """
    centre = squad_card_centres(frames["shot01"], pitch=PITCH)[0][0]
    region = np.array(frames["shot01"].crop(identity_region(centre, pitch=PITCH)))
    assert roster_scores(region, faces)[0][1] == "维什戴尔"
    doubled = [(faces[0][0], faces[0][1]), ("冒名", faces[0][1])]
    ranked = roster_scores(region, doubled)
    assert ranked[0][0] == ranked[1][0]
    assert ranked[0][0] - ranked[1][0] < ROSTER_AMBIGUITY_MARGIN


def test_a_frame_with_no_cards_has_no_identities(
    frames: dict[str, Frame], faces: tuple[tuple[str, np.ndarray], ...]
) -> None:
    blank = Frame(
        canvas=frames["shot01"].canvas,
        pixels=np.zeros_like(frames["shot01"].pixels),
    )
    assert read_squad_identities(blank, faces, pitch=PITCH) == ()


# ---------------------------------------------------------------------------
# 双立绘那一组：**20 条库**（两组编队页各 10 条，同一干员两套立绘算两条）
# ---------------------------------------------------------------------------


TWIN_BATCH = REPO / "tests" / "帧样本" / "立绘两组"
"""run-03 那四张落成的夹具：两组，每组「一张编队页 + 一张备战席横带」。"""

TWIN_LIBS = (TWIN_BATCH / "库1.json", TWIN_BATCH / "库2.json")
"""两组的库各 10 条（就是两份编队页清单）；合起来 20 条。"""

TWIN_GROUPS = ("1", "2")

TWIN_SLOTS_OF_GROUP_1 = (1, 8, 7, 2, 5, 6, 4, 3, 0, 9)
"""组1 带单条那条路量到的卡位（按库顺序，0 = 最右那张）—— 是个**双射**。"""


def _group_of(label: str) -> str:
    """标签 `组N·干员名` 里那一组。"""
    return label.split("·", 1)[0]


def _operator_of(label: str) -> str:
    """标签里那位干员 —— **`split("·", 1)`**：`凯尔希·思衡托` 自己就带一个 `·`。

    这不是吹毛求疵：按 `·` 全劈开再取第 2 段，会把 `凯尔希·思衡托` 读成
    `凯尔希`，而"双胞胎"那一列去找 `组2·凯尔希` 就找不着了 —— 一个只在
    一位干员身上发作、别处全都正常的错。
    """
    return label.split("·", 1)[1]


@pytest.fixture(scope="module")
def twin_pages() -> dict[str, Frame]:
    """两组编队页那一帧（**整帧**，原生 2800×1260）。"""
    return {
        group: load_formation_page(TWIN_BATCH / f"组{group}_编队页.png")
        for group in TWIN_GROUPS
    }


@pytest.fixture(scope="module")
def twin_frames() -> dict[str, Frame]:
    """双立绘那一批四条（贴回画布之后）。"""
    return {
        case.name: frame_from_raw(case.raw()) for case in load_batch(TWIN_BATCH).cases
    }


@pytest.fixture(scope="module")
def twin_faces() -> tuple[tuple[str, np.ndarray], ...]:
    """**20 条库**：两组编队页各 10 条（`库1.json` 在前、`库2.json` 在后）。"""
    return load_cardfaces(TWIN_LIBS[0]) + load_cardfaces(TWIN_LIBS[1])


def _full_scan(region, faces):
    """**整段跑满**那条路（`coarse_accept=False`）—— 与掩码尺子整张滑可比的那一个。"""
    from ark.engine.vision import roster as roster_module

    return roster_module._layered_scores(region, faces, coarse_accept=False)[0]


def _twin_table(
    frame: Frame,
    faces: tuple[tuple[str, np.ndarray], ...],
    centres: list[float],
    rect,
    scorer,
) -> dict[str, list[float]]:
    """一帧 × 一块取区 → `{标签: [10 张卡的分]}`（20 条库、10 张卡）。"""
    rows: dict[str, list[float]] = {name: [] for name, _ in faces}
    for centre in centres:
        block = np.ascontiguousarray(frame.crop(rect(centre, pitch=PITCH)))
        for score, name, _scale in scorer(block):
            rows[name].append(float(score))
    return rows


def _twin_reading(
    rows: dict[str, list[float]], labels: list[str], index: int, wanted: frozenset[str]
) -> dict[str, object]:
    """一张卡在**某一批库条目**下的 最高 · 次高 · 余量 · 判定 —— 就是线上那两条线。"""
    ranked = sorted(
        ((rows[name][index], name) for name in labels if name in wanted),
        key=lambda item: -item[0],
    )
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else (-2.0, None)
    gap = top[0] - second[0]
    return {
        "top": top[1],
        "score": top[0],
        "second": second[1],
        "second_score": second[0],
        "margin": gap,
        "exact": top[0] >= ROSTER_SOLID_MATCH_SCORE and gap >= ROSTER_AMBIGUITY_MARGIN,
    }


@pytest.fixture(scope="module")
def twin_tables(
    twin_frames: dict[str, Frame], twin_faces: tuple[tuple[str, np.ndarray], ...]
) -> dict[str, dict[str, object]]:
    """两条带 × **两块取区**的 20×10 分数表 —— 重活，**跑一次**（约两分钟）。

    **两套取区都跑**：不给 `redeploying` 替这一单挑。这一批的卡全是干净卡
    （没有红罩），所以这一条实际是**在干净卡上比两套取区**。
    """
    _box, mask = margin.identity_disc_region()
    ruler = margin.MaskedRuler(mask, twin_faces)
    tables: dict[str, dict[str, object]] = {}
    for group in TWIN_GROUPS:
        frame = twin_frames[f"组{group}_备战席"]
        centres = [float(centre) for centre, _ in squad_card_centres(frame, pitch=PITCH)]
        tables[group] = {
            "cards": len(centres),
            # **两边都用"整段跑满"那条路**：干净卡这一边不能带"粗层直接接受" ——
            # 它报出来的是**锚点层**那个分，与掩码尺子整张滑出来的分不是一个统计量，
            # 拿来比"两套取区的差"就把**取区的差**与**扫法的差**混在一起了。
            "clean": _twin_table(
                frame, twin_faces, centres, identity_region,
                lambda block: _full_scan(block, twin_faces),
            ),
            "redcap": _twin_table(
                frame, twin_faces, centres, redcap_region,
                lambda block: tuple(ruler.scores(block)),
            ),
        }
    return tables


def test_a_five_by_five_page_gives_ten_arts_not_eleven(
    twin_pages: dict[str, Frame],
) -> None:
    """这一局的编队页是 **5 + 5**（10 人），仓里那份示例是 6 + 5（11 人）——
    两帧的左缘 · 卡距 · 卡宽 · 立绘宽 · 两行上下缘**都同值**，只有每行几张不同。"""
    for position, group in enumerate(TWIN_GROUPS):
        arts = cut_formation_arts(twin_pages[group], counts=(5, 5))
        assert len(arts) == 10
        assert {art.shape for art in arts} == {(617 - 137, FORMATION_ART_WIDTH, 3)}
        assert all(art.dtype == np.uint8 and art.ndim == 3 for art in arts)
        # 张数是从**清单**走下来的：那两份清单各给 10 条
        assert len(load_cardfaces(TWIN_LIBS[position])) == 10


def test_the_row_counts_come_from_the_manifest_and_move_the_cut(page: Frame) -> None:
    """张数是**每一帧自己的数**：同一个框上换张数，切出来的块跟着换。

    11 块里上行那 5 块与 5 + 5 的前 5 块**逐位相同**，后 5 块就是下行那 5 块 ——
    换张数不动任何一块的内容，只改"切几刀"。
    """
    eleven = cut_formation_arts(page)
    ten = cut_formation_arts(page, counts=(5, 5))
    assert (len(eleven), len(ten)) == (11, 10)
    assert all(np.array_equal(a, b) for a, b in zip(ten[:5], eleven[:5]))
    assert all(np.array_equal(a, b) for a, b in zip(ten[5:], eleven[6:]))


def test_a_manifest_whose_counts_do_not_match_its_names_is_refused(scratch: Path) -> None:
    """清单里那行张数**就是真正切几刀**：拿示例那份 11 个名字配一份 5 + 5 的几何，
    报的是「名字与块数对不上」（切出来确实只有 10 块），不是「张数不许变」。"""
    geometry = _declared_geometry()
    geometry["rows"] = [
        {"name": "上行", "y": [137, 617], "count": 5},
        {"name": "下行", "y": [642, 1122], "count": 5},
    ]
    with pytest.raises(ValueError, match="切出 10 块"):
        load_cardfaces(_manifest_with_geometry(scratch, geometry))


def test_a_row_that_cannot_fit_the_canvas_is_refused(page: Frame) -> None:
    """张数不跟常量比，但它有一条硬边界：**这么多张要落得进画布**。"""
    with pytest.raises(ValueError, match="装不下"):
        cut_formation_arts(page, counts=(9, 5))
    with pytest.raises(ValueError, match="装不下"):
        cut_formation_arts(page, counts=(5, 9))
    with pytest.raises(ValueError, match="至少一张"):
        cut_formation_arts(page, counts=(0, 5))
    with pytest.raises(ValueError, match="2 行"):
        cut_formation_arts(page, counts=(5,))
    with pytest.raises(TypeError, match="整数张数"):
        cut_formation_arts(page, counts=(5.0, 5))
    with pytest.raises(TypeError, match="整数张数"):
        cut_formation_arts(page, counts=(True, 5))


def test_each_group_manifest_gives_ten_entries_with_unique_labels() -> None:
    """两组各 10 条，标签是 `组N·干员名` —— **20 条里一条都不重名**。

    重名不是小事：`_entry_for` 明说「同名两条会让『他在第几张卡』没有唯一答案」。
    """
    first = load_cardfaces(TWIN_LIBS[0])
    second = load_cardfaces(TWIN_LIBS[1])
    assert len(first) == len(second) == 10
    labels = [name for name, _ in first + second]
    assert len(set(labels)) == 20
    assert [_group_of(name) for name, _ in first] == ["组1"] * 10
    assert [_group_of(name) for name, _ in second] == ["组2"] * 10
    assert [_operator_of(name) for name, _ in first] == [
        _operator_of(name) for name, _ in second
    ]


def test_the_two_groups_are_two_different_skins_of_the_same_ten() -> None:
    """双胞胎**是两张不同的画**（同一个干员的另一套立绘），不是同一条抄两遍。

    平均绝对差 21.7–42.4（0–255 的灰度）—— 差得不多不少：同一款美术、同一个尺度，
    只是换了皮肤。
    """
    first = load_cardfaces(TWIN_LIBS[0])
    second = load_cardfaces(TWIN_LIBS[1])
    for (_, left), (_, right) in zip(first, second):
        assert left.shape == right.shape
        assert not np.array_equal(left, right)
        assert float(np.abs(left.astype(int) - right.astype(int)).mean()) > 10.0


def test_the_batch_says_which_frame_is_the_library_and_which_are_the_cards(
    twin_frames: dict[str, Frame], twin_pages: dict[str, Frame]
) -> None:
    """出处与角色关系都写在夹具里：**哪张当库、哪张当卡**，而且**没有 facts**。"""
    record = json.loads((TWIN_BATCH / "取材.json").read_text(encoding="utf-8"))
    assert record["facts"] is None
    assert [entry["name"] for entry in record["groups"]] == ["组1", "组2"]
    for entry in record["groups"]:
        assert (TWIN_BATCH / entry["formation_page"]).is_file()
        assert (TWIN_BATCH / entry["battle_band"]).is_file()
        manifest = json.loads((TWIN_BATCH / entry["library"]).read_text(encoding="utf-8"))
        assert Path(manifest["frame"]).name == entry["formation_page"]
        assert [_operator_of(name) for name in manifest["names"]] == (
            entry["names_in_cut_order"]
        )
        assert len(entry["names_in_cut_order"]) == 10
    batch = load_batch(TWIN_BATCH)
    assert sorted(case.name for case in batch.cases) == sorted([
        "组1_编队页", "组1_备战席", "组2_编队页", "组2_备战席",
    ])
    assert all(case.source and case.source_sha256 for case in batch.cases)
    # 编队页是**整帧**：贴回画布之后必须与直接解 PNG 逐位相同
    for group in TWIN_GROUPS:
        assert np.array_equal(
            twin_frames[f"组{group}_编队页"].pixels, twin_pages[group].pixels
        )


def test_under_the_twenty_entry_library_every_card_reads_its_own_group(
    twin_tables: dict[str, dict[str, object]],
    twin_faces: tuple[tuple[str, np.ndarray], ...],
) -> None:
    """**20 条库下，每一张卡读到的都是本组那一条** —— 40 张（两带 × 两取区）无例外。

    这就够把「读出**另一套立绘**」按 0 条报：另一套立绘**只能是另一组那条**，
    而 40 张的 top-1 一条都不落在另一组上。而且十条库各指一位（是个双射）——
    "同一个干员的两条库"没有互相顶替。
    """
    labels = [name for name, _ in twin_faces]
    for group in TWIN_GROUPS:
        for region in ("clean", "redcap"):
            rows = twin_tables[group][region]
            operators = []
            for index in range(twin_tables[group]["cards"]):
                reading = _twin_reading(rows, labels, index, frozenset(labels))
                assert _group_of(reading["top"]) == f"组{group}"
                operators.append(_operator_of(reading["top"]))
            assert len(set(operators)) == 10


def test_the_twin_never_wins_and_never_moves_the_score(
    twin_tables: dict[str, dict[str, object]],
    twin_faces: tuple[tuple[str, np.ndarray], ...],
) -> None:
    """**双胞胎的代价只有一个方向**：条目不变、分数**逐位不变**，塌的只有余量。

    分数为什么不变：一个库条目对一块取区的分数**与库里还有谁无关**，所以"库里多一条
    双胞胎"改不了最高分，只能把次高抬起来 —— 余量因此变小，判定只会往**未知**倒。
    """
    labels = [name for name, _ in twin_faces]
    for group in TWIN_GROUPS:
        own = frozenset(name for name in labels if _group_of(name) == f"组{group}")
        for region in ("clean", "redcap"):
            rows = twin_tables[group][region]
            for index in range(twin_tables[group]["cards"]):
                full = _twin_reading(rows, labels, index, frozenset(labels))
                alone = _twin_reading(rows, labels, index, own)
                assert full["top"] == alone["top"]
                assert full["score"] == alone["score"]
                assert full["margin"] <= alone["margin"] + 1e-12
                twin = ("组2·" if group == "1" else "组1·") + _operator_of(alone["top"])
                assert rows[twin][index] < alone["score"]


def test_with_only_the_other_groups_ten_nothing_reaches_either_line(
    twin_tables: dict[str, dict[str, object]],
    twin_faces: tuple[tuple[str, np.ndarray], ...],
) -> None:
    """**卡上那一套不在库里**那一档（库里只有它的双胞胎）：两带 × 两取区 × 10 张
    **全部**落未知 —— 最高分与余量**分别**都没够到线，所以一条假事实都没有。

    实测两个量：最高分 ≤ 0.8000（线 0.85）· 余量 ≤ 0.2019（线 0.25）。
    """
    labels = [name for name, _ in twin_faces]
    for group in TWIN_GROUPS:
        other = frozenset(name for name in labels if _group_of(name) != f"组{group}")
        for region in ("clean", "redcap"):
            rows = twin_tables[group][region]
            for index in range(twin_tables[group]["cards"]):
                reading = _twin_reading(rows, labels, index, other)
                assert not reading["exact"]
                assert reading["score"] < ROSTER_SOLID_MATCH_SCORE
                assert reading["margin"] < ROSTER_AMBIGUITY_MARGIN


def test_the_disc_region_buys_score_with_margin(
    twin_tables: dict[str, dict[str, object]],
    twin_faces: tuple[tuple[str, np.ndarray], ...],
) -> None:
    """两套取区的差：**条目一个都不换**（top-1 逐张相同）、分数互有高低，
    而余量**整体变小**，判定只会从精确倒向未知 —— 从不让一张卡从未知变成精确。"""
    labels = [name for name, _ in twin_faces]
    for group in TWIN_GROUPS:
        own = frozenset(name for name in labels if _group_of(name) == f"组{group}")
        for wanted in (own, frozenset(labels)):
            box_total = 0.0
            disc_total = 0.0
            for index in range(twin_tables[group]["cards"]):
                box = _twin_reading(twin_tables[group]["clean"], labels, index, wanted)
                disc = _twin_reading(twin_tables[group]["redcap"], labels, index, wanted)
                assert disc["top"] == box["top"]
                box_total += box["margin"]
                disc_total += disc["margin"]
                assert not (box["exact"] is False and disc["exact"] is True)
            assert disc_total < box_total


@pytest.fixture(scope="module")
def twin_slots(
    twin_tables: dict[str, dict[str, object]],
    twin_faces: tuple[tuple[str, np.ndarray], ...],
) -> dict[tuple[str, str], int | None]:
    """单条那条路：(带, 库条目) → 卡位（**恰有一张**过线才给下标）。"""
    labels = [name for name, _ in twin_faces]
    found: dict[tuple[str, str], int | None] = {}
    for group in TWIN_GROUPS:
        rows = twin_tables[group]["clean"]
        for name in labels:
            hits = [
                index
                for index in range(twin_tables[group]["cards"])
                if rows[name][index] >= ROSTER_SOLID_MATCH_SCORE
            ]
            found[(group, name)] = hits[0] if len(hits) == 1 else None
    return found


def test_the_single_entry_path_points_each_operator_at_one_card(
    twin_slots: dict[tuple[str, str], int | None],
    twin_faces: tuple[tuple[str, np.ndarray], ...],
) -> None:
    """**20 条库下「有没有指到另一条上」**：没有。

    每一条带上**本组那 10 条**里每条指得出的都**互不撞**；**另一组那 10 条一条都
    指不出**（最高分全在 0.85 线下）。所以没有"双胞胎把某位干员指到别人那张卡上"
    这种事 —— 另一套立绘在单条那条路上**静默**，不产假卡位。

    "各指一张"不是十条都成立：组2 带上**组2·塞雷娅**只有 0.8309（线 0.85），
    它指不出来 —— 那是**分数**不够，不是撞车，也不是被双胞胎顶掉。
    """
    labels = [name for name, _ in twin_faces]
    operators = [_operator_of(name) for name in labels if _group_of(name) == "组1"]
    assert len(operators) == 10
    for group in TWIN_GROUPS:
        own = [name for name in labels if _group_of(name) == f"组{group}"]
        other = [name for name in labels if _group_of(name) != f"组{group}"]
        mine = [slot for slot in (twin_slots[(group, name)] for name in own)
                if slot is not None]
        assert len(mine) >= 9
        assert len(set(mine)) == len(mine)
        assert [twin_slots[(group, name)] for name in other] == [None] * 10
    assert [
        twin_slots[("1", name)] for name in labels if _group_of(name) == "组1"
    ] == list(TWIN_SLOTS_OF_GROUP_1)
    # 同一位干员的**两套立绘**、**两条带**：至少 9 位在两处都指得出，而且卡位一致
    agreed = 0
    for operator in operators:
        left = twin_slots[("1", f"组1·{operator}")]
        right = twin_slots[("2", f"组2·{operator}")]
        if left is not None and right is not None:
            assert left == right
            agreed += 1
    assert agreed >= 9

