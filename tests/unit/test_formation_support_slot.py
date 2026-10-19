"""助战干员那一格（右上那一格）：**它也要进本局干员库**（用户给的裁定）。

这一份量三件事，都是**画面事实**：

1. 那一格的框是**量出来的**：`SUPPORT_BOX = (2081, 125, 259, 505)` —— 在
   `助战.png`（那一格有人）与 `编队1-0.png`（那一格空着）上量橙色像素的包围盒，
   **两帧同值**。它与卡片几何的两条关系（上下各外扩 12 px · 跨在第 7 列的格位上，
   而左缘**不**落在 `470 + 271k` 上）由 `_check_support_box` 钉着。
2. 库最后那一条叫 `SUPPORT_NAME`（「助战」）：那一格**不印名字**，所以用固定占位名
   （用户裁定：「助战」作为助战干员的统一的名字）。
3. 空着 vs 有人**在画面上分得开**（整框平均绝对差 **0.000336** vs **58.689**；
   中央 80×80 白像素 **1821 / 饱和 0.35** vs **399 / 饱和 32.72**）——
   **但这一版没有把它接成判据**（用户裁定：要动判据先报）。有没有人由调用方/清单说，
   与「每行几张」同一个做法。

## 那三件事各自的网在哪

**框是量出来的**（第 1 件）：`test_the_box_is_measured_again_from_the_pixels` **真的回画面上
再量一遍** —— 取区里橙色像素（`R > 170 且 50 < G < 170 且 B < 100`）的包围盒，三帧
（有人 + 两张空着）各扫一次，逐帧等于 `SUPPORT_BOX`。上一版这里只有一条**字面量相等**
（`C.SUPPORT_BOX == (2081, 125, 259, 505)`）—— 那只是把常量抄了一遍，**没有回画面上量**；
`_check_support_box` 挡的是**自洽关系**（形状不对会响），挡不住一组数同时错。
这个网是本轮补的。

**「助战」这个字符串**（第 2 件）两边都要挡，它们**不是同一件事**：

- `test_the_placeholder_name_is_not_an_operator_name`：它**不是**闭集里的干员名
  （`decide("助战", roster_index(闭集))` 落「未知」）—— 这就是**不许靠名字认它**的由来；
  同时它**必须照样在库里有一条**（这一格的存在性不依赖名字是不是真名）；
- `test_a_manifest_that_already_names_the_support_slot_is_refused`：把它当**干员名**
  写进 `names` ⇒ 报错（一个位置不许报两次）。
- `test_the_slot_survives_when_the_name_is_not_in_the_closed_set`：**反方向**——
  名字不是真名**不是**把这一格丢掉的理由：同一帧、同一份清单，库里照样有那一条。

**默认那一条路一个字节没变**：`test_the_default_library_matches_its_pinned_fingerprint`
钉住 11 条与整库指纹（**现算**，不是抄一份清单）；另有一条
`test_no_manifest_but_the_new_fixture_declares_a_support_slot` 保证**除新夹具外**
没有第二份清单被加了这一项。**「与改动前逐位相同」本身**由探针 `data/probe/s21_default.py`
跑过（拿 `git show HEAD:` 落下来的旧模块跑同一份清单，指纹 `03f16a4d…ace864` 两边相同）——
那条证据在 `data/` 里（gitignored），**它不是用例**，这里不假装它是。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ark.engine.vision.frame import frame_from_raw
from ark.offline.assets_gen import cardface as C
from ark.offline.assets_gen.cardface import cut_support_art, load_cardfaces, load_formation_page
from ark.offline.bench.batch import load_batch

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "tests" / "帧样本" / "助战格"
MANIFEST = BATCH / "编队页.json"
SIZES = REPO / "tests" / "帧样本" / "编队页人数"
QUCAI = json.loads((BATCH / "取材.json").read_text(encoding="utf-8"))
BOOK = json.loads(MANIFEST.read_text(encoding="utf-8"))
NAMES = tuple(BOOK["names"])


@pytest.fixture(scope="module")
def taken() -> np.ndarray:
    """`助战.png` 那一格（**有人**）切出来的那一块。"""
    return cut_support_art(load_formation_page(BATCH / "助战.png"))


@pytest.fixture(scope="module")
def empty() -> dict[str, np.ndarray]:
    """两帧助战位**空着**的，切出来的那两块。"""
    return {
        name: cut_support_art(load_formation_page(SIZES / (name + ".png")))
        for name in ("编队1-0", "编队2-0")
    }


def test_the_box_is_measured_and_sits_off_the_card_grid() -> None:
    """框是**量出来的**：上下各外扩 12 px · 跨在第 7 列的格位上 · 左缘**不在**列网格上。"""
    x, y, width, height = C.SUPPORT_BOX
    assert C.SUPPORT_BOX == (2081, 125, 259, 505)
    assert C.SUPPORT_NAME == "助战"
    top, bottom = C.FORMATION_ROWS[0][0], C.FORMATION_ROWS[0][1]
    assert (y, y + height - 1) == (top - 12, bottom + 12)
    seventh = C.FORMATION_LEFT0 + C.FORMATION_PITCH * 6
    assert x < seventh < x + width
    assert (x - C.FORMATION_LEFT0) % C.FORMATION_PITCH == 256, "它不在列网格上"
    assert x + width <= 2800 and y + height <= 1260
    C._check_support_box()


@pytest.mark.parametrize(
    "bad",
    [
        (2081, 124, 259, 505),   # 上缘少 1 px（不是外扩 12）
        (2081, 125, 259, 504),   # 下缘少 1 px
        (2096, 125, 259, 505),   # 左缘挪到第 7 列的格位 = 落在列网格上
        (2400, 125, 259, 505),   # 整块跑出画布
    ],
)
def test_a_broken_box_is_caught(bad, monkeypatch: pytest.MonkeyPatch) -> None:
    """把常量改坏一条，`_check_support_box` 当场报 —— 网是打的。"""
    monkeypatch.setattr(C, "SUPPORT_BOX", bad)
    with pytest.raises(ValueError):
        C._check_support_box()


def test_the_block_is_the_whole_measured_box_and_it_is_a_copy(taken: np.ndarray) -> None:
    """块就是框里那一整块：形状 `(505, 259, 3)`、逐位等于帧上那一块、而且是**拷贝**。"""
    frame = load_formation_page(BATCH / "助战.png")
    x, y, width, height = C.SUPPORT_BOX
    assert taken.shape == (height, width, 3) == (505, 259, 3)
    assert taken.dtype == np.uint8
    assert np.array_equal(taken, frame.pixels[y : y + height, x : x + width])
    assert not np.shares_memory(taken, frame.pixels), "块要是拷贝，不是帧的视图"


def test_the_library_gains_one_trailing_entry_named_for_the_support_slot() -> None:
    """`support: true` 的清单 → 库是 **10 + 1 条**，最后一条叫「助战」，块就是那一格。"""
    faces = load_cardfaces(MANIFEST)
    assert len(faces) == 11
    assert [name for name, _ in faces[:-1]] == list(NAMES)
    assert faces[-1][0] == C.SUPPORT_NAME == "助战"
    assert faces[-1][1].shape == (505, 259, 3)
    for _name, art in faces[:-1]:
        assert art.shape == (480, 214, 3)


def test_the_flag_can_be_overridden_by_the_caller() -> None:
    """`support=` 是**调用方**说的：同一个清单 `False` 就是 10 条（那一格空着的时候）。"""
    assert len(load_cardfaces(MANIFEST, support=False)) == 10
    assert len(load_cardfaces(MANIFEST, support=True)) == 11
    with pytest.raises(TypeError):
        load_cardfaces(MANIFEST, support="yes")  # type: ignore[arg-type]


def test_a_manifest_that_already_names_the_support_slot_is_refused() -> None:
    """清单的 `names` 里已经写着「助战」时**报错**：那是这一条的位置，不许报两次。

    这一条钉的是**第一种失效**：把占位名当成一个真的干员名（写进 `names`，于是它跑进
    "名字 → 干员"那一套里）。**改之前这一份清单会被接受**：旧模块照切 11 条，第 11 条
    的名字就叫「助战」，与前面那 10 个真名混在一张表里，一声不吭。
    """
    record = dict(BOOK)
    record["names"] = list(NAMES) + [C.SUPPORT_NAME]
    record["geometry"] = dict(BOOK["geometry"])
    record["geometry"]["rows"] = [
        {"name": "上行", "y": [137, 617], "count": 5},
        {"name": "下行", "y": [642, 1122], "count": 6},
    ]
    scratch = REPO / "data" / "probe" / "pytest_scratch" / "support_name_collision"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / "编队页.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + chr(10),
                    encoding="utf-8", newline=chr(10))
    with pytest.raises(ValueError):
        load_cardfaces(path)


def test_the_old_library_is_untouched_by_this_change() -> None:
    """**默认那一条路一个字节没变**：仓里那份示例清单还是 11 条、块还是 `(480, 214, 3)`。"""
    faces = load_cardfaces()
    assert len(faces) == 11
    assert C.SUPPORT_NAME not in [name for name, _ in faces]
    assert all(art.shape == (480, 214, 3) for _name, art in faces)


def test_the_empty_slot_and_the_taken_slot_are_far_apart(taken, empty) -> None:
    """空着 vs 有人**在画面上分得开**（量出来的画面事实，**不是判据**）。

    整框平均绝对差：两块空占位之间 **0.000336**，空占位与真立绘之间 **58.689**；
    中央 80×80 里：空着 **1821** 个白像素、平均饱和 **0.35**（那个加号是纯白），
    有人 **399** 个白像素、平均饱和 **32.72**。**这一版没有拿它立线。**
    """
    left, right = empty["编队1-0"], empty["编队2-0"]
    assert float(np.abs(left.astype(np.int16) - right.astype(np.int16)).mean()) < 0.01
    assert float(np.abs(taken.astype(np.int16) - left.astype(np.int16)).mean()) > 50.0

    def centre(block):
        mid = block[block.shape[0] // 2 - 40:block.shape[0] // 2 + 40,
                    block.shape[1] // 2 - 40:block.shape[1] // 2 + 40].astype(np.int16)
        sat = mid.max(axis=2) - mid.min(axis=2)
        return int((mid >= 180).all(axis=2).sum()), float(sat.mean())

    taken_white, taken_sat = centre(taken)
    empty_white, empty_sat = centre(left)
    assert (taken_white, round(taken_sat, 2)) == (399, 32.72)
    assert (empty_white, round(empty_sat, 2)) == (1821, 0.35)
    assert empty_white > 4 * taken_white
    assert QUCAI["助战那一格"]["这一帧"] == "有人"


def test_the_frame_lands_back_on_the_canvas() -> None:
    """整帧搬进仓：贴回画布之后与直接解那一块 PNG 逐位相同，出处与原点都在清单里。"""
    cases = {case.name: case for case in load_batch(BATCH).cases}
    assert list(cases) == ["助战"]
    case = cases["助战"]
    assert case.rect.x == 0 and case.rect.y == 0
    assert (case.rect.width, case.rect.height) == (2800, 1260)
    assert case.source == "screenshots/run-05/助战.png"
    assert len(case.source_sha256) == 64
    assert np.array_equal(
        frame_from_raw(case.raw()).pixels, load_formation_page(BATCH / "助战.png").pixels
    )

# --- 这一单补的网：框回画面上再量一遍 ---------------------------------------


_ORANGE = "R > 170 且 50 < G < 170 且 B < 100"


def _orange_mask(pixels: np.ndarray) -> np.ndarray:
    """取区里"橙色"那一族像素（`SUPPORT_BOX` 的 docstring 写的那三条）。"""
    data = pixels.astype(np.int16)
    return (
        (data[:, :, 0] > 170)
        & (data[:, :, 1] > 50)
        & (data[:, :, 1] < 170)
        & (data[:, :, 2] < 100)
    )


def _measure_box(pixels: np.ndarray) -> tuple[int, int, int, int]:
    """**那一条量法**：先在一行上扫橙色列定左右，再在左缘那一列上扫橙色行定上下。

    与 `data/probe/r06_scan.py` 同一条（这一单把量法写进用例，不让它只活在探针里）。
    扫描行 `y = 300` 落在框内、扫描列取左缘 + 1，两者都在 `SUPPORT_BOX` 里面，
    所以这一条**不预设框的答案**，只预设"去哪儿扫"。
    """
    mask = _orange_mask(pixels)
    columns = [x for x in range(2020, 2450) if mask[300, x]]
    assert columns, "y=300 这一行上一个橙色像素都没有：那条框不在这儿了"
    left, right = columns[0], columns[-1]
    rows = [y for y in range(100, 700) if mask[y, left + 1]]
    assert rows, "左缘那一列上没有橙色：这不是一条竖线"
    top, bottom = rows[0], rows[-1]
    return (left, top, right - left + 1, bottom - top + 1)


def test_the_box_is_measured_again_from_the_pixels() -> None:
    """**重跑一遍量法**：三帧（有人 + 两张空着）各量一次，逐帧等于 `SUPPORT_BOX`。

    这一条是"那个框是量出来的"这句的网：常量改了、或者画面换了，这里就红。
    上一版这件事只在 docstring 与夹具 README 里写着，一个字都没核过。

    它**顺带**钉住另一件事：三帧同值 ⇒ 那个框是**那一格的版式**，与那一格有没有人无关。
    """
    frames = {
        "有人": BATCH / "助战.png",
        "空着（编队1-0）": SIZES / "编队1-0.png",
        "空着（编队2-0）": SIZES / "编队2-0.png",
    }
    measured = {tag: _measure_box(load_formation_page(path).pixels) for tag, path in frames.items()}
    assert measured["有人"] == C.SUPPORT_BOX == (2081, 125, 259, 505)
    assert set(measured.values()) == {C.SUPPORT_BOX}, measured


def test_each_edge_of_the_box_is_a_real_edge() -> None:
    """框的四条边**各指得出画面** —— 这一条钉的是「包围盒」这个说法本身。

    | 边 | 画面上的样子 | 为什么要分开说 |
    | --- | --- | --- |
    | 左 | 一条**连续**竖线，从 125 到 629 一行不缺 | 量法就是靠它定上下的 |
    | 右 | 2337–2339 三列都有 300 个以上橙色像素 | 有抗锯齿，最外一列会断（圆角） |
    | 上 | 125 之上（100–124）**一个橙色都没有**，125 是第一行 | 圆角让这一行只有两三个像素 |
    | 下 | 622–629 每一行都有 200 个以上橙色像素，630 起一个都没有 | 同上 |

    两帧都验（有人 + 空着）：**框与那一格有没有人无关**这件事在这里再落一次。
    """
    for path in (BATCH / "助战.png", SIZES / "编队1-0.png"):
        mask = _orange_mask(load_formation_page(path).pixels)
        x, y, width, height = C.SUPPORT_BOX
        left_rows = [row for row in range(y, y + height) if mask[row, x + 1]]
        assert left_rows == list(range(y, y + height)), path.name
        for column in (x + width - 3, x + width - 2, x + width - 1):
            assert int(mask[y : y + height, column].sum()) >= 300, (path.name, column)
        assert int(mask[100:y, 2020:2450].sum()) == 0, path.name
        assert int(mask[y, 2020:2450].sum()) >= 1, path.name
        assert int(mask[y + height : y + height + 5, 2020:2450].sum()) == 0, path.name
        for row in range(y + height - 8, y + height):
            assert int(mask[row, 2020:2450].sum()) >= 200, (path.name, row)

# --- 这一单补的网：「助战」这个字符串的两侧 ---------------------------------


def test_the_placeholder_name_is_not_an_operator_name() -> None:
    """「助战」**不是**干员名 —— 但这**不影响**它在本局库里有那一条。

    两件事一起钉：

    1. 它不在全量闭集里，`decide` 也判不出它（落「未知」）—— 所以**不许靠名字认它**；
    2. 它照样在库里有一条（`load_cardfaces(清单, support=True)` 的最后一条），
       块就是那一格 —— 这一格的存在性**不依赖**名字是不是真名。
    """
    from ark.offline.assets_gen.roster_names import decide, load_roster_names, roster_index

    closed = load_roster_names()
    assert C.SUPPORT_NAME not in closed, "闭集里居然有「助战」：那它就不是占位名了"
    assert decide(C.SUPPORT_NAME, roster_index(closed)) == (None, "未知")
    assert not any(name == C.SUPPORT_NAME for name in closed)
    faces = load_cardfaces(MANIFEST)
    assert [name for name, _ in faces].count(C.SUPPORT_NAME) == 1


def test_the_slot_survives_when_the_name_is_not_in_the_closed_set() -> None:
    """**反方向**：名字不是真名**不是**把这一格丢掉的理由（改之前这一格根本进不了库）。

    这是第二种失效（"因为它不是干员名就把这一格丢掉"）的网：同一帧、同一份清单，
    `support=True` 时库里照样有那一条 —— 它的存在**不经过**任何"名字 → 干员"的匹配。
    如果哪一天有人给 `load_cardfaces` 加一道"名字必须在闭集里"的闸，这一条会红。
    """
    from ark.offline.assets_gen.roster_names import load_roster_names

    closed = load_roster_names()
    faces = load_cardfaces(MANIFEST, support=True)
    names = [name for name, _ in faces]
    assert names[-1] == C.SUPPORT_NAME
    assert names[-1] not in closed
    assert all(name in closed for name in names[:-1]), "那 10 个真名要在闭集里，否则这一条不比出差别"
    assert faces[-1][1].shape == (505, 259, 3)

def _saturation(block: np.ndarray) -> float:
    data = block.astype(np.int16)
    return float((data.max(axis=2) - data.min(axis=2)).mean())


def _centre(block: np.ndarray) -> tuple[int, float]:
    mid = block[block.shape[0] // 2 - 40 : block.shape[0] // 2 + 40,
                block.shape[1] // 2 - 40 : block.shape[1] // 2 + 40].astype(np.int16)
    return int((mid >= 180).all(axis=2).sum()), float((mid.max(axis=2) - mid.min(axis=2)).mean())


def test_the_numbers_recorded_in_the_fixture_are_the_numbers_on_the_picture() -> None:
    """`取材.json` 里那几个数**现算一遍**与它比 —— 抄本与画面不许各说一套。

    这一条是**抄本的网**：`src/.../cardface.py` 的 docstring、夹具 README、`取材.json`
    三处写着同一批数，而真值只有这里现算的这一份。改常量、换帧、或者谁顺手改了 JSON，
    都会红。
    """
    recorded = QUCAI["助战那一格"]
    taken = cut_support_art(load_formation_page(BATCH / "助战.png"))
    empty_one = cut_support_art(load_formation_page(SIZES / "编队1-0.png"))
    empty_two = cut_support_art(load_formation_page(SIZES / "编队2-0.png"))

    white, saturation = _centre(taken)
    assert recorded["有人的那几个数（框的中央 80×80）"] == {
        "平均饱和": round(_saturation(taken), 2),
        "中央80白像素": white,
        "中央80平均饱和": round(saturation, 2),
    }
    assert (white, round(saturation, 2)) == (399, 32.72)

    for tag, block in (("编队1-0.png", empty_one), ("编队2-0.png", empty_two)):
        empty_white, empty_saturation = _centre(block)
        assert recorded["空着的那几个数（中央 80×80）"][tag] == {
            "平均饱和": round(_saturation(block), 2),
            "中央80白像素": empty_white,
            "中央80平均饱和": round(empty_saturation, 2),
        }
        assert (empty_white, round(empty_saturation, 2)) == (1821, 0.35)

    gap_empty = float(
        np.abs(empty_one.astype(np.int16) - empty_two.astype(np.int16)).mean()
    )
    gap_taken = float(np.abs(taken.astype(np.int16) - empty_one.astype(np.int16)).mean())
    assert recorded["两块的差（整框平均绝对差）"] == {
        "空 vs 空（两块空占位之差）": round(gap_empty, 6),
        "空 vs 有人（空占位与真立绘之差）": round(gap_taken, 3),
    }
    assert (round(gap_empty, 6), round(gap_taken, 3)) == (0.000336, 58.689)
    assert recorded["box"] == list(C.SUPPORT_BOX)
    assert recorded["库里那一条的名字"] == C.SUPPORT_NAME

def test_the_default_library_matches_its_pinned_fingerprint() -> None:
    """默认那一条路**一个字节没变**：11 条 · 无「助战」· 整库指纹等于钉住的那个值。

    指纹是 `sha256(每条的名字 + 0 字节 + 像素字节 + 0 字节)`（按库的次序）。它是
    **现算**的：改了默认路（多切一块、少切一块、块的内容动了）立刻红。
    """
    import hashlib

    faces = load_cardfaces()
    digest = hashlib.sha256()
    for name, art in faces:
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(art.tobytes())
        digest.update(b"\x00")
    assert len(faces) == 11
    assert C.SUPPORT_NAME not in [name for name, _ in faces]
    assert digest.hexdigest() == (
        "03f16a4d38cf8e6aff1f563b1c31e27f9d14decca56f7a5bf57f8a1b35ace864"
    ), "默认那一条路的库变了"


def test_no_manifest_but_the_new_fixture_declares_a_support_slot() -> None:
    """除新夹具外**没有任何清单**带 `support` —— 加了就是改口径（要报规划者）。

    扫的是仓里每一份"像清单"的 JSON（顶层同时有 `frame` 与 `names`），不按文件名猜。
    """
    offenders = []
    for path in sorted((REPO / "tests" / "帧样本").rglob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict) or "frame" not in record or "names" not in record:
            continue
        if "support" in record and path != MANIFEST:
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == [], offenders

