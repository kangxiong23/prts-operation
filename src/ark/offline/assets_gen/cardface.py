"""从编队页那一帧切出「库」：本局队伍的卡面。

《资产生成器/卡面与指纹》§1 管这一步。库的形态是**一串 `(名字, 立绘块)`**，
按编队页的切割顺序排好 —— 这一层只负责**切**（`cut_formation_arts`），
**不认名字**：名字由调用方配。这一批里名字是**规划者目视读出来的**，因为游戏在
卡上印的是汉字，而《视觉层/识别方法》已定"数字一律走固定字形模板、不上通用 OCR"。

**几何是实测的，而且是在原生画布上量的**（《卡面与指纹》§1 末那张表）：

| 项 | 值 |
| --- | --- |
| 卡左缘 | `FORMATION_LEFT0 + FORMATION_PITCH * k`（`k` 从 0 数） |
| 卡距 · 卡宽 · 缝 | `FORMATION_PITCH` · `FORMATION_CARD_WIDTH` · `FORMATION_GAP` |
| 卡里那块**立绘画** | `FORMATION_ART_WIDTH` 宽（右边那条卡底纹 = 卡宽 − 立绘宽） |
| 上行 / 下行 | 两行的**上下缘**住 `FORMATION_ROWS`，而**每行几张**是清单那一帧自己的数，左对齐、两行等高 |

**它不进 `UiLayout`**（用户裁定）：`UiLayout` 是**战斗界面**的地址表，而编队页
**只在 `启动` 那一刻用一次**，线上识别链一次都不读它。所以这组常量住在
**产卡面那一侧**（就是这里），与 `SQUAD_CARD_PITCH` 同一个做法。

**为什么读 PNG 要走 `cv2.imdecode(np.fromfile(...))`**：这一批的路径里有汉字
（`编队演示——干员卡片识别（原生）.png`），而 `cv2.imread` 在非 ASCII 路径上
**静默给 None** —— 那不是"读不到"，是"读到一半变成了别的东西"。

**分辨率不做适配**（《概念设计共识》§5.4 第 1 条）：这组坐标是在原生画布上量的，
别的画布一律报错 —— 按比例缩着用会**看起来仍然在工作**，而切出来的立绘块整块偏掉。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from ark.engine.vision.frame import CANVAS_2800x1260, Frame
from ark.offline.assets_gen.roster_names import load_roster_names

__all__ = [
    "cut_formation_arts",
    "cut_support_art",
    "load_formation_page",
    "load_cardfaces",
    "FORMATION_LEFT0",
    "FORMATION_PITCH",
    "FORMATION_CARD_WIDTH",
    "FORMATION_GAP",
    "FORMATION_ART_WIDTH",
    "FORMATION_ROWS",
    "FORMATION_CARDS",
    "SUPPORT_NAME",
    "SUPPORT_BOX",
    "FORMATION_MANIFEST",
    "MANIFEST_NAME",
    "REPO",
]


REPO = Path(__file__).resolve().parents[4]
"""仓库根：`src/ark/offline/assets_gen/` 往上四级（与 `countdown.py` 算的是同一个根）。"""

MANIFEST_NAME = "编队页.json"
"""清单的文件名。这一轮**库的落盘格式还没定**（用户批准：先功能后整理），
所以清单是**测试夹具**那一份，它把"哪一帧"与"十一个名字"放在一起。"""

FORMATION_MANIFEST = REPO / "tests" / "帧样本" / "点名" / MANIFEST_NAME
"""夹具清单的落点（`tests/帧样本/点名/编队页.json`）。"""


FORMATION_LEFT0: Final[int] = 470
"""上行第一张卡的**左缘**（原生像素）。下行第一张也是它 —— 两行左对齐。"""

FORMATION_PITCH: Final[int] = 271
"""编队页的**卡距**：相邻两张卡的左缘差。

**它与 `SQUAD_CARD_PITCH = 207.5` 不是同一个数**，别混：那个是**备战席**卡行的
间距，这个是**编队页**卡格的间距，两处是两套 UI。
"""

FORMATION_CARD_WIDTH: Final[int] = 262
"""一张卡的**宽度**（卡格的宽，含右边那条底纹）。

**它与 `FORMATION_PITCH` 不是同一个数**：卡距 = 卡宽 + 缝，见 `FORMATION_GAP`。
"""

FORMATION_GAP: Final[int] = 9
"""相邻两张卡之间的**缝**（原生像素）。

`FORMATION_CARD_WIDTH + FORMATION_GAP == FORMATION_PITCH`（`262 + 9 = 271`）——
这就是卡距的来历。三个数里改坏一个，`_check_geometry` 当场就报。
"""

FORMATION_ART_WIDTH: Final[int] = 214
"""卡里那块**立绘画**的宽度（`FORMATION_CARD_WIDTH` 减去右边 48 px 的卡底纹）。

**`214 ÷ 204 = 1.049`** —— 这就是"两侧同一像素尺度"的那个比（备战席卡 204 宽），
也是备战席那边身份区要搜尺度 1.05–1.15 的由来。
"""

FORMATION_ROWS: Final[tuple[tuple[int, int, int], ...]] = (
    (137, 617, 6),
    (642, 1122, 5),
)
"""两行 `(上缘, 下缘, 张数)` —— **上下缘是几何，张数不是**。

上下缘是**卡片自己的上下缘**（实测），这一层拿它当立绘块的上下界 ——
立绘铺满整张卡，所以块就是"卡内那块画"。

**张数随队伍人数变，所以它不住常量**：仓里那份示例是 `6 + 5`（11 人），而
`tests/帧样本/立绘两组/` 那一局是 `5 + 5`（10 人）—— 两帧的上下缘、左缘、卡距、
卡宽、立绘宽**都同值**，只有每行几张不同（实测：两行那张卡行右缘都落在常量算
出来的列上，只有第 6 格是空的）。所以这里那两个数只是**示例与默认**：清单声明了
`geometry.rows` 就按**清单的**张数切（见 `cut_formation_arts` 的 `counts`）。
"""

FORMATION_CARDS: Final[int] = sum(count for _, _, count in FORMATION_ROWS)
"""**仓里那份示例清单**切得出几块立绘（11）—— 不是"一帧编队页有几块"。"""

SUPPORT_NAME: Final[str] = "助战"
"""助战干员那一格在**库里**那条的名字（用户给的：编队页不印它的名字）。

它**不是读出来的**：那一格上印的是「助战干员 / SUPPORT UNIT」那个标签，不是干员名。
所以库里那一条统一叫「助战」—— 一位助战干员读出来就是「助战」，与它的真名无关。
"""

SUPPORT_BOX: Final[tuple[int, int, int, int]] = (2081, 125, 259, 505)
"""右上那一格「助战干员」的**橙色高亮框** `(x, y, 宽, 高)`（原生像素）。

**量出来的，不是算出来的**：在 `screenshots/run-05/` 的 `助战.png`（那一格**有人**）
与 `编队1-0.png`（那一格**空着**）上各自量橙色像素的包围盒
（`R > 170 且 50 < G < 170 且 B < 100`），**两帧同值**：
左 2081 · 右 2339（宽 259）· 上 125 · 下 629（高 505）。
⇒ 它是**那一格的框**，与那一格有没有人**无关**（`_check_support_box` 钉着）。

**这句话有网**：`tests/unit/test_formation_support_slot.py` 的
`test_the_box_is_measured_again_from_the_pixels` 在这三帧上**重新量一遍**包围盒，
逐帧比这里的常量；`test_each_edge_of_the_box_is_a_real_edge` 再钉四条边各自是真边。

它与卡片几何的两条关系（都对常量算）：

1. **上下各外扩 12 px**：`137 - 12 = 125`、`617 + 12 = 629` —— 它把**上行那一条卡片带**包住；
2. **它跨在第 7 列的格位上**：左缘 2081 < `470 + 271 × 6 = 2096` < 右缘 2340；
   而它**不是又一列卡**：宽 259 不等于 `FORMATION_CARD_WIDTH`，左缘也不落在
   `470 + 271k` 上（差 15 px）—— 那是那一格自己的版式。

**为什么整块取、不按卡片的取法**：卡片那条取区（`[卡左缘, 卡左缘 + 214)`）是从
「立绘画 = 卡宽减右边那条底纹」推出来的，而这一格**没有底纹、也不在列网格上**；
按类比去凑一个「214 宽的取区」就是**拿算出来的数当量出来的数**（[D-全局-25]）。
所以这里取**量到的框本身**：它的内容（立绘 + 底下那条「助战干员」带）就是这一格。
"""


def _check_support_box() -> None:
    """`SUPPORT_BOX` 的四条不变式 —— **全对常量算**（与 `_check_geometry` 那三条同一个做法）。

    它挡的是「常量被改坏了而没人看见」，因为常量才是代码实际跑的那一份。
    """
    x, y, width, height = SUPPORT_BOX
    top, bottom = FORMATION_ROWS[0][0], FORMATION_ROWS[0][1]
    if (y, y + height - 1) != (top - 12, bottom + 12):
        raise ValueError(
            "助战那一格的框上下要**各外扩 12 px**：上缘 " + repr(top - 12) + "、下缘 "
            + repr(bottom + 12) + "，而常量给的是 " + repr((y, y + height - 1))
        )
    seventh = FORMATION_LEFT0 + FORMATION_PITCH * 6
    if not x < seventh < x + width:
        raise ValueError(
            "助战那一格的框要**跨在第 7 列的格位上**（" + repr(seventh)
            + "），而常量给的 x 与宽是 " + repr((x, width))
        )
    if (x - FORMATION_LEFT0) % FORMATION_PITCH == 0:
        raise ValueError(
            "助战那一格的左缘落在卡片列网格上了（" + repr(x)
            + "）：它不是又一列卡，左缘不该是 470 + 271k"
        )
    if x < 0 or y < 0 or x + width > CANVAS_2800x1260.width or y + height > CANVAS_2800x1260.height:
        raise ValueError("助战那一格的框跑到画布外了：" + repr(SUPPORT_BOX))


def cut_support_art(frame: Frame) -> np.ndarray:
    """编队页右上那一格「助战干员」→ **一块立绘**（`(高, 宽, 3)` 的 RGB、是拷贝）。

    **它不判那一格有没有人**：空着的时候切出来的是那张**空占位**（灰剪影 + 一个加号，
    底下同样印着「助战干员 / SUPPORT UNIT」）。要哪一格由调用方说 ——
    与「每行几张」是同一个做法（见 `load_cardfaces` 的 `support=`）。

    **那一格有没有人可以从画面判**（这一单量到的两个数，都在 `SUPPORT_BOX` 的中央 80×80 里）：
    空着 **1821** 个白像素、平均饱和 **0.35**（那个加号是纯白）；有人 **399** 个白像素、
    平均饱和 **32.72**。**但这一版没有把它接成判据**（用户裁定：要动判据先报）。

    **这几个数指得出网**（不是只住在这里）：`tests/unit/test_formation_support_slot.py` 的
    `test_the_numbers_recorded_in_the_fixture_are_the_numbers_on_the_picture` 拿夹具那份真值
    **现算一遍**与画面比；`SUPPORT_BOX` 本身由 `test_the_box_is_measured_again_from_the_pixels`
    回画面上**重新量一次**（三帧）。上一版这里只有一条**字面量相等**（把常量抄一遍），
    **没有回画面上量**；改坏常量只有 `_check_support_box` 的自洽关系会响，
    而自洽关系挡不住一组数同时错。
    """
    if not isinstance(frame, Frame):
        raise TypeError(f"要切的是一帧，收到 {type(frame).__name__}")
    if frame.canvas != CANVAS_2800x1260:
        raise ValueError(
            f"助战那一格的几何是在 {CANVAS_2800x1260.width}×{CANVAS_2800x1260.height} 上"
            f"量的，这一帧是 {frame.canvas.width}×{frame.canvas.height}："
            "定死运行环境，不做分辨率适配"
        )
    _check_support_box()
    x, y, width, height = SUPPORT_BOX
    return np.ascontiguousarray(frame.pixels[y : y + height, x : x + width])




def cut_formation_arts(
    frame: Frame,
    *,
    counts: Sequence[int] | None = None,
) -> tuple[np.ndarray, ...]:
    """一帧编队页 → **每行几张块立绘**（`counts` 说几张），顺序就是 cut_order。

    `counts` 是**每行切几张**（上行在前、下行在后），假定两行左对齐、上下缘见
    `FORMATION_ROWS`；**不传就照常量那份示例**（`6 + 5`）。

    **为什么张数是参数、不是常量**：它随队伍人数变 —— 仓里那份示例是 6 + 5，
    而另一局是 5 + 5，两帧的坐标常量**同值**。清单里本来就有 `count` 这个字段
    （给人一眼看全），这里只是把它接到真正切的那一刀上。

    **它不认名字**：名字由调用方配。名字的来源是**人**（游戏印的是汉字，这一层
    不上通用 OCR），把它塞进这个函数等于让"切"去猜"谁"。

    每块是 `(高, 宽, 3)` 的 **RGB** uint8，而且**是拷贝**（不是帧的视图）——
    库会被拿着到处走，留一个指向 14 MB 帧的视图等于把整帧也拖着。

    画布必须是 `CANVAS_2800x1260`：这组坐标是在那张画布上量的，
    **不许按比例缩着用**（定死运行环境，不做分辨率适配）。
    """
    if not isinstance(frame, Frame):
        raise TypeError(f"要切的是一帧，收到 {type(frame).__name__}")
    if frame.canvas != CANVAS_2800x1260:
        raise ValueError(
            f"编队页的几何是在 {CANVAS_2800x1260.width}×{CANVAS_2800x1260.height} 上"
            f"量的，这一帧是 {frame.canvas.width}×{frame.canvas.height}："
            "定死运行环境，不做分辨率适配"
        )
    if counts is None:
        counts = tuple(count for _, _, count in FORMATION_ROWS)
    checked = _check_counts(counts, "要切的每行张数")
    arts: list[np.ndarray] = []
    for (row_top, row_bottom), count in zip(_row_bands(), checked):
        for index in range(count):
            x0 = FORMATION_LEFT0 + FORMATION_PITCH * index
            arts.append(
                np.ascontiguousarray(
                    frame.pixels[row_top:row_bottom, x0 : x0 + FORMATION_ART_WIDTH]
                )
            )
    return tuple(arts)


def load_formation_page(path: str | Path) -> Frame:
    """一个编队页 PNG → **一帧**（原生像素，一个都不缩）。

    走 `cv2.imdecode(np.fromfile(...))`：非 ASCII 路径上 `cv2.imread` 静默给
    None，而"解不出来"与"路径不对"在这一层都该是**响亮的错**。

    解出来是 **BGR**，而 `Frame.pixels` 是 **RGB** —— 这里翻过来。不翻的话库与
    战斗帧的通道序不一致，整条链会**安静地错**：相关系数照样算得出来，只是永远
    不高，而它看起来像"方法不行"。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"编队页是一个 PNG 文件，{path} 不是")
    decoded = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"{path} 解不成图：这一层只认它解得开的 PNG")
    height, width = int(decoded.shape[0]), int(decoded.shape[1])
    if (width, height) != (CANVAS_2800x1260.width, CANVAS_2800x1260.height):
        raise ValueError(
            f"{path} 是 {width}×{height}，而编队页的几何是在 "
            f"{CANVAS_2800x1260.width}×{CANVAS_2800x1260.height} 上量的："
            "定死运行环境，不做分辨率适配"
        )
    return Frame(
        canvas=CANVAS_2800x1260,
        pixels=np.ascontiguousarray(decoded[:, :, ::-1]),
    )


def load_cardfaces(
    manifest: str | Path = FORMATION_MANIFEST,
    *,
    names_from: str | Path | None = None,
    support: bool | None = None,
) -> tuple[tuple[str, np.ndarray], ...]:
    """清单 → **库**（`(名字, 立绘块)`，按 cut_order）。

    清单里那份 `geometry` 与这里的常量**必须说同一件事**：对不上就报错。
    同一组坐标两处各写一份、而且谁也不知道另一份变了没有，是这一层最容易烂的地方
    —— 检查比"信任"便宜。**唯一的例外是每行几张**：它随队伍人数变，所以清单声明
    几张就切几张（`_check_geometry` 把核过的张数原样交出来，就是这里传下去的那个）。

    名字的个数必须与切出来的块数一样：少一个名字就是**一条没有身份的空位**，
    而它在库里看起来仍然"有一条"。

    **`names_from` 是可选的那条路**（用户裁定 5）：给一个**一局名单产物**
    （`assets/roster/<局指纹>.json`，由 `roster_names` 读出来的那一份），就用它的
    `names`，清单里那份留着当兼容。**清单仍然要给**：`frame`、`geometry` 与张数都住它那儿，
    而名字与块数必须一样多 —— 两处对不上就报错，不猜谁对。

    **`support` 是右上那一格「助战干员」**（用户给的）：`True` 就把那一格的框切一块、以固定的名字 `SUPPORT_NAME`（「助战」）**接在库的最后一条**，名字与块数一起多一条。`None`（默认）读清单里的 `support` 那一项 —— **没有这一项就是「这一帧的助战位不进库」**。

    它与「每行几张」是**同一个做法**：都由调用方/清单说，链不判那一格有没有人（判法见 `cut_support_art`）。清单的 `names` **不许**已经写着「助战」—— 那个名字是这一条的位置。
    """
    manifest = Path(manifest)
    if not manifest.is_file():
        raise FileNotFoundError(f"清单是一个 JSON 文件，{manifest} 不是")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(record, Mapping):
        raise TypeError(f"{manifest} 的顶层要是一个对象，收到 {type(record).__name__}")
    for key in ("frame", "names"):
        if key not in record:
            raise KeyError(f"{manifest} 缺 {key!r}")
    counts = _check_geometry(record.get("geometry"), manifest)

    if support is None:
        support = record.get("support", False)
    if not isinstance(support, bool):
        raise TypeError(
            f"{manifest} 的 support 要是布尔（助战那一格有没有人），收到 {support!r}：没有这一项就是「不进库」，不是「不知道」"
        )

    frame_path = record["frame"]
    if not isinstance(frame_path, str) or not frame_path:
        raise TypeError(f"{manifest} 的 frame 要是相对仓库根的路径，收到 {frame_path!r}")
    names = record["names"]
    if names_from is not None:
        names = _names_from_roster(Path(names_from))
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise TypeError(f"{manifest} 的 names 要是名字的序列，收到 {type(names).__name__}")
    for position, name in enumerate(names):
        if not isinstance(name, str) or not name:
            raise TypeError(f"{manifest} 的 names[{position}] 要是非空字符串，收到 {name!r}")

    page = load_formation_page(REPO / frame_path)
    arts = cut_formation_arts(page, counts=counts)
    if len(names) != len(arts):
        raise ValueError(
            f"{manifest} 给了 {len(names)} 个名字，而编队页切出 {len(arts)} 块立绘："
            "对不上就是有卡位没有身份，而它在库里看起来仍然有一条"
        )
    if support:
        if SUPPORT_NAME in names:
            raise ValueError(
                f"{manifest} 的 names 里已经写着 {SUPPORT_NAME!r}：那个名字是**助战那一格**的位置，"
                "库里的助战条由 support 那一项接上，不该由清单再报一次"
            )
        arts = arts + (cut_support_art(page),)
        names = tuple(names) + (SUPPORT_NAME,)
    return tuple(zip(names, arts))

def _names_from_roster(path: Path) -> tuple[str, ...]:
    """一局名单产物 → 名字序列（用户裁定 5）。**只认那种形状**，别的形状报错。"""
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, Mapping) or "names" not in record:
        raise ValueError(
            str(path) + " 不是一份一局名单产物（没有 names 那一段）："
            "它该是 assets/roster/<局指纹>.json"
        )
    if record.get("kind") != "game_roster":
        raise ValueError(
            str(path) + " 的 kind 是 " + repr(record.get("kind"))
            + "，不是 game_roster —— 拿全量闭集当一局名单用会把库的键说错"
        )
    return load_roster_names(path)


def _row_bands() -> tuple[tuple[int, int], ...]:
    """两行的**上下缘** —— 这一层真正的几何（张数不在里面，见 `FORMATION_ROWS`）。

    **从 `FORMATION_ROWS` 现算，不另写一份**：两处各写一份，改一处漏一处就是
    「清单与常量抄同一个错」那一类，而这一层正是靠"两边能对上"挡漂移的。
    """
    return tuple((top, bottom) for top, bottom, _ in FORMATION_ROWS)


def _check_counts(counts: object, what: str) -> tuple[int, ...]:
    """每行切几张：**一行一个、每行至少一张、整行要装得下画布**。

    **张数不跟常量比**（它随队伍人数变：示例 `6 + 5`、另一局 `5 + 5`），能比的
    是"这么多张落不落得进一张 2800 宽的画布" —— 这是它唯一的硬边界。
    """
    if isinstance(counts, (str, bytes)) or not isinstance(counts, Sequence):
        raise TypeError(f"{what} 要是每行几张的序列，收到 {type(counts).__name__}")
    bands = _row_bands()
    if len(counts) != len(bands):
        raise ValueError(
            f"{what} 有 {len(counts)} 行，而这一层认的是 {len(bands)} 行（{bands}）"
        )
    out: list[int] = []
    for position, count in enumerate(counts):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{what}[{position}] 要是整数张数，收到 {count!r}")
        if count < 1:
            raise ValueError(f"{what}[{position}] 是 {count}：一行至少一张")
        right = FORMATION_LEFT0 + FORMATION_PITCH * (count - 1) + FORMATION_CARD_WIDTH
        if right > CANVAS_2800x1260.width:
            raise ValueError(
                f"{what}[{position}] 是 {count}：第 {count} 张卡的右缘落在 {right}，"
                f"画布只有 {CANVAS_2800x1260.width} 宽 —— 这一帧装不下这么多张"
            )
        out.append(count)
    return tuple(out)


_GEOMETRY_KEYS: Final[tuple[str, ...]] = (
    "left0",
    "pitch",
    "card_width",
    "gap",
    "art_width",
    "rows",
)
"""清单里那份 `geometry` **必须报全**的六个键。"""


def _check_geometry(recorded: object, manifest: Path) -> tuple[int, ...] | None:
    """清单里记的几何与这里的常量**对一遍**，再把清单自己的数**内部对一遍**。

    **返回清单声明的每行张数**（没声明几何就是 `None`，那时切法照常量那份示例）——
    这样"检查过的数"与"拿去切的数"是同一份，不会出现"核的是清单、切的是常量"。

    **清单里没有 `geometry` 就跳过** —— 那是「这份清单没有声明几何」，不是
    「声明了别的几何」。但只要**声明了**，就得**报全**：缺一个键会让后面每一条检查
    都**静默跳过**，而「跳过」看起来与「通过」一模一样。

    一层是**清单对常量**（左缘 · 卡距 · 卡宽 · 缝 · 立绘宽 · 两行的**上下缘**），
    挡**漂移**。**张数不在这一层里**（它随队伍人数变，见 `FORMATION_ROWS`）：清单说
    几张就切几张，只按 `_check_counts` 那三条硬边界核。所以这一层**挡不住"把张数
    写错"** —— 那是**每一帧自己的数**，没有常量可比，只有"装不装得下画布"这一条
    是硬的。

    再加**三条不变式**，**全对常量算**（不是对清单的值算）—— 它们挡的是
    「两边抄了同一个错」，也就是「清单对常量」那一层**永远看不见**的那一类：

    1. **两行等高** —— 历史：下行写的是 `(642, 1146, 5)`，504 高，而上行 480 高；
       **清单与常量两边都是 1146**，所以那一层一声不吭，多切进去的是卡下面那条
       队名栏。同款卡的两行不可能不一样高；
    2. **`FORMATION_ART_WIDTH < FORMATION_CARD_WIDTH`** —— 立绘画是卡里挖掉右边
       那条底纹剩下的那块，它比整张卡宽就没有意义；
    3. **`FORMATION_CARD_WIDTH + FORMATION_GAP == FORMATION_PITCH`** —— 卡距就是
       「卡宽 + 缝」（`262 + 9 = 271`），这就是卡距的来历。

    **不变式对常量算，因为常量才是代码实际跑的**；清单只负责「给人一眼看全」。
    单个数被改坏，三条路（常量 · 清单 · 不变式）全抓；只剩「一对数同时改坏且互相
    补偿」（`250 + 21`）漏网，而那不是顺手改得出来的。

    **上下缘没有像素网**：按行灰度落差找四条缘，全图最强的是 136 / 607 / 641 /
    1156 / 1175，而 137 与 1122 都不在前几名（1156、1175 是下面那条队名栏）——
    朴素规则会给出**另外一批错数**，比没有更坏。所以上下缘是「人肉实测值 + 这里的
    自洽约束」，**不装成有网**。
    """
    if recorded is None:
        return None
    if not isinstance(recorded, Mapping):
        raise TypeError(
            f"{manifest} 的 geometry 要是一个对象，收到 {type(recorded).__name__}"
        )
    missing = [key for key in _GEOMETRY_KEYS if key not in recorded]
    if missing:
        raise KeyError(
            f"{manifest} 的 geometry 缺 {'、'.join(missing)}：声明了几何就得报全，"
            "缺一个键会让后面每一条检查都静默跳过"
        )
    expected = {
        "left0": FORMATION_LEFT0,
        "pitch": FORMATION_PITCH,
        "card_width": FORMATION_CARD_WIDTH,
        "gap": FORMATION_GAP,
        "art_width": FORMATION_ART_WIDTH,
    }
    for key, value in expected.items():
        if recorded[key] != value:
            raise ValueError(
                f"{manifest} 的 geometry.{key} 写的是 {recorded[key]!r}，"
                f"而这一层的常量是 {value}：同一组坐标不许两处各说一套"
            )
    for key in expected:
        value = recorded[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{manifest} 的 geometry.{key} 要是整数像素，收到 {value!r}")
    rows = recorded["rows"]
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError(f"{manifest} 的 geometry.rows 要是两行的序列，收到 {rows!r}")
    seen: list[tuple[int, int, int]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"{manifest} 的 geometry.rows[{position}] 要是一个对象")
        span = row.get("y")
        count = row.get("count")
        if (
            isinstance(span, (str, bytes))
            or not isinstance(span, Sequence)
            or len(span) != 2
        ):
            raise TypeError(f"{manifest} 的 geometry.rows[{position}].y 要是一对数")
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(
                f"{manifest} 的 geometry.rows[{position}].count 要是整数张数，"
                f"收到 {count!r}"
            )
        seen.append((int(span[0]), int(span[1]), count))
    bands = tuple((top, bottom) for top, bottom, _ in seen)
    if bands != _row_bands():
        raise ValueError(
            f"{manifest} 的 geometry.rows 的上下缘是 {bands}，而这一层的常量是 "
            f"{_row_bands()}：同一组坐标不许两处各说一套（**张数不算**，它随队伍"
            "人数变，见 FORMATION_ROWS）"
        )
    counts = _check_counts(
        tuple(count for _, _, count in seen),
        f"{manifest} 的 geometry.rows 的每行张数",
    )
    heights = sorted({bottom - top for top, bottom, _ in seen})
    if len(heights) != 1:
        raise ValueError(
            f"{manifest} 的 geometry.rows 里两行的高度是 {heights}（下缘 - 上缘）："
            "同款卡的两行不可能不一样高，上缘与下缘都是卡自己的边"
        )
    if not FORMATION_ART_WIDTH < FORMATION_CARD_WIDTH:
        raise ValueError(
            f"这一层的常量里 art_width 是 {FORMATION_ART_WIDTH}、card_width 是 "
            f"{FORMATION_CARD_WIDTH}：立绘画是卡里挖掉右边那条底纹剩下的那块，"
            "它必须比整张卡窄"
        )
    if FORMATION_CARD_WIDTH + FORMATION_GAP != FORMATION_PITCH:
        raise ValueError(
            f"这一层的常量里 card_width + gap 是 {FORMATION_CARD_WIDTH} + "
            f"{FORMATION_GAP} = {FORMATION_CARD_WIDTH + FORMATION_GAP}，"
            f"而 pitch 是 {FORMATION_PITCH}：卡距就是「卡宽 + 缝」，"
            "这几个数不许各说一套"
        )
    # 闸门**对自己**：要求报全的键，必须与真正核过的键对得上。两个方向都堵 ——
    # 往 _GEOMETRY_KEYS 里加一个没人核的键（"必须报全、却没人比值、也没人验类型"，
    # 正是 card_width / gap 当年那个处境），或者往 expected 里加一个没进"报全"的。
    # 不把两处合成一处：rows 与其余五个不是一件事（一个是"必须有"，一个是"值必须
    # 等于常量"），合成一个会把两种含义揉在一起。
    if set(_GEOMETRY_KEYS) != set(expected) | {"rows"}:
        raise ValueError(
            "这一层要求报全的键，与它真正核过的键对不上："
            f"_GEOMETRY_KEYS = {set(_GEOMETRY_KEYS)}，"
            f"expected | {{'rows'}} = {set(expected) | {'rows'}}"
        )
    return counts
