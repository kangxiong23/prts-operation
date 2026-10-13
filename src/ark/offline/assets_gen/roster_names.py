"""编队页上的名字 —— 一局一次，产出「库的名单」。

# 它是什么

编队页那两行的每张卡，立绘下缘印着名字（白字压在深色带上）。这一层把它读出来，
产出**一局那份名单**（`assets/roster/<局指纹>.json`）。它**不进识别链** ——
识别链仍然离线、可复现；名字只在**建库那一步**用一次。

# 版面：十格竖排（用户裁定 1）

**为什么不是两行横排**：中文 OCR 不吃字间空格 —— 横排时相邻的名字会被并成一个词
（实测：上行 2/3/4 三格并成一个「歌蕾蒂娅荒芜拉普兰德凯尔希·思衡托」，而格 2/4 空着）。
所以拼图**一格一行、行间留白**：「词 → 哪一卡」靠**拼的位置**判，接口的位置开关
只当**交叉核对**（词数与格数对不上时，它指出是哪一格）。

**元凶是旧横排每格那 24 px 白边**（第十一单裁定 F）：名字右端到下一个名字左端只剩
`214 − 名字宽` px —— 短名字宽 74…148 ⇒ 分得开，两个长名字只剩 **14** ⇒ 合词。
**横排本身不是原因**，白边才是。

# 取区（原生 2800×1260，实测）

| 那一行 | 名字墨迹 y | 立绘里那句小白字 | 空档 |
| --- | --- | --- | --- |
| 上行（立绘 y 137..617） | **577..607** | y 550..576（`PLXY` / `HOKX` 那一档） | 上行卡位 2 只剩 **1 px** |
| 下行（立绘 y 642..1122） | **1081..1112** | y 1051..1073 | 8–10 px |

取区 = **`[卡左缘, 卡左缘 + 214)`**（用户裁定 4）：左端就是 `卡左缘` —— 再往左就是
**左邻卡那块亮东西**（它占 `[卡左缘 − 43, 卡左缘 − 1]`）。**十个格都有这条带**
（第十一单裁定 F 更正：先前那句「上排四格有、下排没有」是 min≥225 那道门槛的假象）；
带内**最右那一列**在上行 0/1/2 与下行 5/6/7 上是亮的（255 与 154–173），另外六格要到带内更深处才亮。

名字**右对齐**：墨迹右端 = 卡左缘 + `NAME_RIGHT_INSET`（208，十张卡逐一对上）。
**名字框 = `[卡左缘 + 9, 卡左缘 + 208]`、宽 200**：两个最长的名字（`荒芜拉普兰德`、
`凯尔希·思衡托`）左端都落在 `+9` —— 离取区左边界只剩 **9 px**，这就是「再往左就脏」的
全部余量；短名字的左端在 `+66`…`+144`。**长名字游戏会缩字**，所以名字一定放得进这 200 px
（用户给的事实）。取区右边留 6 px 深色，免得名字右端直接粘着一条白（用户裁定 5）。

**名字以外一律深色**（用户裁定 2/5）：上缘以上填深色（把立绘里那句小白字抹掉），
下缘以下也填深色（原图上那几行不是全暗的），**格与格之间也是深色空档、不许白边**。
**取区的四条边界都必须落在深色上**（用户裁定 D）：不许有一条亮线横穿取区 ——
上行 y=608 那一行整行是亮背景，所以上行的下缘取 **607**。

# 判决：只认名单（用户裁定 3）

1. 每个词**先归一化**（间隔号 `·`、全半角、空白、大小写）；
2. **精确命中**全量名单 ⇒ 采用；
3. 否则在名单里找**编辑距离 ≤ 1** 的候选（**多一个字 / 少一个字 / 错一个字**各算一个）：
   **恰好一个** ⇒ 采用并标**「改过」**；**零个或多个** ⇒ 落**未知**；
4. **不许**把长词切成几个已知名字的拼串（那是猜）—— 合词是版面/取区出错的信号，**要报错**；
5. **距离 2 不许**。每一格必须**恰好一个词**，多了少了都响亮报错。

**「未知」是一个值，不是一个异常**：判决照落、账照记；要不要写盘是 `write_roster` 的事，
它见到 `None` 就拒绝落盘 —— 一份带「未知」的名单当库的键用就是假事实。

# 拼图与缓存（用户裁定 4）

拼图**确定性**（同样的输入 ⇒ 逐字节相同的 PNG），所以缓存键就是**拼图字节的 blake2b**；
同一张图再读一次**调用 0 次**（与「画面没变就跳过」同一条纪律）。
缓存里存的是**接口给的原文** —— 加工过的那一份不叫原始响应。

# 传输（用户裁定 Q1）

**默认安全级先打**；只有握不上、且原因是 `SSL: WRONG_SIGNATURE_TYPE` 这一类
**握手签名**错误时，才对**那一次连接**用 `set_ciphers("DEFAULT@SECLEVEL=0")` 重试**一次**，
并把降级原因原文与用时记进账（`send_ocr` 回的 `OcrAttempt`）。**不是无条件降级** ——
默认那条路继续走，它自己会碰到通的那一窗（讯飞那个边缘时好时坏）。

两条路都**照验证书、都加密**；**任何情况下不许明文 http**（`endpoint` 只收 `https`）。
安全级 0 放宽的**不止** SHA-1 握手签名：链上的 SHA-1 签名、最小密钥长度一起放宽 ——
所以它是一条**有代价**的退路。实测这台机器上 `webapi.xfyun.cn` 的叶子证是
`sha256WithRSAEncryption` / RSA 2048 bit，两样都不弱；判别实验是
`AES256-GCM-SHA384`（纯 RSA 密钥交换，**没有**那句 ServerKeyExchange 签名）在默认安全级就通，
凡 ECDHE 套件只有安全级 0 才通 ⇒ 断的是那句握手上的 SHA-1 签名。

`post=` 是那条缝：喂一个假 `post` 就能**不联网**把「默认先试 / 只在那一类错误上降级」
整条阶梯走一遍（单测就是这么打的）。**那本账顺着 `send_ocr` → `transport` →
`RosterReading.attempt` 落进读数**（用户裁定 2）：走没走降级、原因原文、用了几秒，
调用方不用自己包一层 `send_ocr` 才看得到。

# 凭据（用户裁定 6）

`APP_ID_VAR` / `API_KEY_VAR` 两个**环境变量**，一个字都不进仓、不打印、不进日志。
**这条链要求变量在主机进程启动前就在环境里，或由调用方显式注入**：这台机器上两份都在
**机器域**，而 DSH 那个进程的环境块里没有它们（User 域没有、profile 也没有），所以子进程
得自己显式搬进来 —— **不去读注册表**。凭据缺了报的就是那句：也可能只是没继承到。

# 它不做什么

它不改识别链、不碰 `ark.engine` 的判定（只 `import` 那一帧的类型）；
它不上通用 OCR 去读**数字**（那是 `glyphs` 固定字形模板的事）。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from ark.engine.vision.frame import Frame

__all__ = [
    "NAME_ROWS",
    "NAME_LEFT_EDGE",
    "NAME_RIGHT_INSET",
    "NAME_SPANS",
    "CELL_HEIGHT",
    "FORMATION_LEFT0",
    "FORMATION_PITCH",
    "NAME_ART_WIDTH",
    "PAD_ABOVE",
    "PAD_BELOW",
    "CELL_GAP",
    "VERTICAL",
    "HORIZONTAL_V1",
    "APP_ID_VAR",
    "API_KEY_VAR",
    "DEFAULT_ENDPOINT",
    "ROSTER_DIR",
    "NAMES_FILE",
    "STATE_DIR",
    "RosterGrid",
    "NameDecision",
    "RosterReading",
    "RosterLayoutError",
    "OcrRefused",
    "OcrAttempt",
    "RELAXED_CIPHERS",
    "send_ocr",
    "name_boxes",
    "compose_sheet",
    "sheet_png",
    "sheet_digest",
    "parse_words",
    "assign_words",
    "normalize",
    "edit_distance",
    "roster_index",
    "decide",
    "read_names",
    "ask_ocr",
    "credentials_from_env",
    "load_roster_names",
    "write_roster",
]


REPO: Final[Path] = Path(__file__).resolve().parents[4]
"""仓库根：`src/ark/offline/assets_gen/` 往上四级（与 `cardface.py` 同一个根）。"""

ROSTER_DIR: Final[Path] = REPO / "assets" / "roster"
"""名单资产的落点（用户裁定 5）：**不是** `features/`。"""

NAMES_FILE: Final[Path] = ROSTER_DIR / "names.json"
"""全量**闭集**：判决里那句「精确命中全量名单」对的就是它。"""

STATE_DIR: Final[Path] = REPO / "data" / "roster_state"
"""**运行期状态**的家：**一局产物**落这儿（`data/` 是 gitignored 的运行时数据）。

**一局产物不进版本库、也不进 `assets/roster/`**（用户裁定）：它是**建库那一步的凭据**，
不是资产 —— 资产只有 `assets/roster/` 那一份（全量闭集 `names.json` + 库 `faces.json`）。
逐局产物往资产树里堆不是个事：库里跨局累积的是**卡面**，不是"哪一局用了谁"。
"""


NAME_ROWS: Final[tuple[tuple[int, int], ...]] = ((577, 607), (1081, 1112))
"""两行**名字墨迹**的上下缘（含），原生坐标 —— 实测，不是卡片几何推的。

**不取整条深色带**（用户裁定 2）：深色带上下都比名字宽（上行 557..615），把立绘里
那句小白字一起裁进来了。**上下界是按行的**：两行的值不一样，别拿一个常数套两行。

**上行下缘是 607、不是 608**（做7④ 量出来的）：**y=608 那一行整行都是亮背景**
（x 从 `卡左缘` 到 `+213` 全是 255）—— 它是名牌底的**下边界**，不是名字的一部分。
它在取区最左那 20 列上是**唯一的亮源**：留着它，「取区左边界那一列全深」就不成立。
名字墨迹真正的下缘（暗邻域掩码量的）是上行 605..607、下行 1111..1112。
"""

NAME_LEFT_EDGE: Final[int] = 0
"""取区**左端**到卡左缘的距离 —— **0**（用户裁定 4：`卡左缘` 就是最左的干净边界）。

再往左就是左邻卡那块亮东西（`[卡左缘 − 43, 卡左缘 − 1]`），所以取区一旦越过卡左缘
就把它带进来。**十个格都有这条带**（用户裁定 F）。两个最长的名字（`荒芜拉普兰德`、
`凯尔希·思衡托`）墨迹左端落在 `卡左缘 + 9`（**不是** +0 —— 用户裁定 F 以实测为准），
离取区左边界只剩 9 px，这就是"再往左就脏"的全部余量。"""

NAME_RIGHT_INSET: Final[int] = 208
"""名字**墨迹右端**到卡左缘的距离 —— 名字**右对齐**，十张卡逐一对上（208 = 678-470 = 948-741
= 1220-1012 = 1762-1554 …）。**它不是取区的右端**（那在 `卡左缘 + 214`）：右边留 6 px
深色，免得名字右端直接粘着一条白（用户裁定 5）。"""

FORMATION_LEFT0: Final[int] = 470
"""编队页两行第一张卡的左缘（与 `cardface.FORMATION_LEFT0` 同值）。"""

FORMATION_PITCH: Final[int] = 271
"""编队页的卡距（与 `cardface.FORMATION_PITCH` 同值）—— 不是备战席那个 207.5。"""

NAME_ART_WIDTH: Final[int] = 214
"""取区的宽（与 `cardface.FORMATION_ART_WIDTH` 同值）。"""

PAD_ABOVE: Final[int] = 6
"""名字上缘**以上**留几行 —— 这几行**填成深色**，把立绘里那句小白字消灭掉（用户裁定 2）。"""

PAD_BELOW: Final[int] = 6
"""名字下缘**以下**留几行 —— **也填成深色**（用户裁定 5：上下深色空档各 ≥ 6 px）。

名字下缘以下那几行在原图上**不是全暗的**（名牌底只因斜边而露出亮背景），所以必须压深，
不然取区最下那几行就是一片亮像素 —— 那正是「名字以外不许有亮像素」要挡的东西。"""

CELL_GAP: Final[int] = 25
"""竖排时格与格的**深色**空档（用户裁定 5：行间 ≥ 25 px）。**不许白边。**

中文 OCR 不吃字间空格，而白色空档在图片里就是"字与字之间的亮东西"；所以空档是深色的。
"""

NAME_SPANS: Final[tuple[int, ...]] = tuple(bottom - top + 1 for top, bottom in NAME_ROWS)
"""两行各留几行名字（上行 31、下行 32）—— **它们不一样**，所以格高按最长的那行算。"""

CELL_HEIGHT: Final[int] = max(NAME_SPANS) + PAD_ABOVE + PAD_BELOW
"""一格的高：最长那行的名字行数 + 上下深色空档 —— 44。

短的那行（上行）在下面**多抹一行深色**，把两行的格补成一样高（栅格要等格高）。
"""


@dataclass(frozen=True)
class RosterGrid:
    """拼图的格子几何 —— **拼法是我们的**，所以「词 → 哪一格」由它判（用户裁定 4）。

    `columns` 是「一行几格」：竖排是 1，横排是 5。`gap_x` 与 `gap_y` 分开，
    因为横排那张拼图的横向留白做在格**内**（白边 24 px），纵向才是独立的行间留白。
    """

    cell_width: int
    cell_height: int
    gap_x: int
    gap_y: int
    columns: int

    def slot_of(self, x: float, y: float) -> int | None:
        """这个位置落在第几格；落在**格间留白**里或越界就是 `None`。

        位置开关给的是一个框，这里用的是它的**中心** —— 框跨格（合词）时中心只落在一格，
        另外那几格就空了，于是「一格一词」那道闸门当场把它抓住。
        """
        if x < 0 or y < 0:
            return None
        step_x = self.cell_width + self.gap_x
        step_y = self.cell_height + self.gap_y
        row = int(y // step_y)
        col = int(x // step_x)
        if col >= self.columns:
            return None
        if y - row * step_y >= self.cell_height:
            return None
        if x - col * step_x >= self.cell_width:
            return None
        return row * self.columns + col


VERTICAL: Final[RosterGrid] = RosterGrid(NAME_ART_WIDTH, CELL_HEIGHT, 0, CELL_GAP, 1)
"""十格竖排（用户裁定 1/5）：格 214×44（含上下各 6 行深色空档），行间 25 px 深色。"""

HORIZONTAL_V1: Final[RosterGrid] = RosterGrid(262, 66, 0, 20, 5)
"""**旧**横排的几何 —— 只留给夹具里那一份响应（它就是按这个拼的）。"""

APP_ID_VAR: Final[str] = "xunfei_ocr_yinshua_APP_ID"
"""凭据的环境变量名（用户裁定 6）。"""
API_KEY_VAR: Final[str] = "xunfei_ocr_yinshua_APIKey"
"""凭据的环境变量名（用户裁定 6）。值一个字都不进仓、不打印、不进日志。"""

DEFAULT_ENDPOINT: Final[str] = "https://webapi.xfyun.cn/v1/service/v1/ocr/general"
"""默认端点 —— **https**（用户裁定 6：端点做参数，不许写死 http）。"""

RELAXED_CIPHERS: Final[str] = "DEFAULT@SECLEVEL=0"
"""降级那一条连接用的密码套件串（用户裁定 Q1②）。

它**只**把 OpenSSL 的安全级放到 0（因为讯飞那个边缘用 SHA-1 签握手上的 ServerKeyExchange）。
证书**照验**、握手**照加密** —— 放宽的是签名算法与最小密钥长度，不是"不看证书"。
见模块文档「传输」那一段：它是一条**有代价**的退路，不是默认。"""


class RosterLayoutError(Exception):
    """「一格一词」那道闸门：某一格不是**恰好一个词**（用户裁定 3d/3e）。

    多了是合词、少了是缺词 —— 两种都说明**版面或取区出了错**，而**不许**拿
    「把长词切成几个已知名字」去把账做平（用户裁定 3d）。
    """

    def __init__(self, missing: Sequence[int], extra: Sequence[int]) -> None:
        super().__init__(
            "拼图与响应对不上：「一格一词」不成立。缺词的格 " + repr(list(missing))
            + "，多于一个词的格 " + repr(list(extra))
            + " —— 多词是合词、缺词是取区切空，两种都要先修版面"
        )
        self.missing = tuple(missing)
        self.extra = tuple(extra)


class OcrRefused(Exception):
    """接口没给 `code == 0`：把 `code` 与 `desc` 原样带出来（不吞、不重试）。"""

    def __init__(self, code: object, desc: object) -> None:
        super().__init__("接口没成功：code=" + repr(code) + " desc=" + repr(desc))
        self.code = code
        self.desc = desc


@dataclass(frozen=True)
class OcrAttempt:
    """一次真调用的**账**（用户裁定 Q1③）：原文 · 走没走降级 · 降级原因原文 · 用时。

    `downgraded=False` 就是「默认安全级一次就成」；`True` 时 `reason` 是那条
    `WRONG_SIGNATURE_TYPE` 的**原文**，`first_seconds` 是第一次撞墙花了多久，
    `seconds` 是两次加起来。**报告里要写的就是这三样。**
    """

    raw: str
    downgraded: bool
    reason: str | None
    seconds: float
    first_seconds: float | None = None


@dataclass(frozen=True)
class NameDecision:
    """一格的结果：词 · 采用的名单条目（`None` = 未知）· 怎么来的。

    `how` 只有三种：**精确**（归一化之后逐字命中）、**改过**（编辑距离 1 的唯一条目）、
    **未知**。名字那一条**一定是名单里的原样**（采用的就是名单里那一份）。
    """

    slot: int
    word: str
    name: str | None
    how: str


@dataclass(frozen=True)
class RosterReading:
    """一次读数：名单 · 每格的账 · **调用了几次** · 拼图指纹 · 拼图字节 · **那本账**。

    `attempt` 是这一读**真打了网络**的那一次账（用户裁定 2）：撞缓存（`calls == 0`）
    或 transport 只吐一个字符串（夹具重放）时它是 `None` —— 那时说的是「**没有账**」，
    不是「走的默认那一路」。**读数的语义不因为多了这个字段而变一个字。**
    """

    names: tuple[str | None, ...]
    slots: tuple[NameDecision, ...]
    calls: int
    digest: str
    png: bytes
    attempt: OcrAttempt | None = None

    @property
    def counts(self) -> dict[str, int]:
        """三样数：**精确** / **改过** / **未知**。"""
        tally = {"精确": 0, "改过": 0, "未知": 0}
        for item in self.slots:
            tally[item.how] = tally.get(item.how, 0) + 1
        return tally

    @property
    def downgraded(self) -> bool:
        """这一读走没走降级（用户裁定 2）。**没打网络 ⇒ `False`**：没有降级这回事。"""
        return self.attempt is not None and self.attempt.downgraded

    @property
    def seconds(self) -> float | None:
        """这一读花在接口上的秒数。**没打网络就是 `None`** —— 别把 0 当成"很快"。"""
        return None if self.attempt is None else self.attempt.seconds


def name_boxes(counts: Sequence[int]) -> tuple[tuple[int, int, int, int], ...]:
    """每张卡的名字取区 `(x, y, 宽, 高)`，次序 = cut_order（上行在前、每行左到右）。

    `counts` 是**每行几张**（与 `cardface.cut_formation_arts` 同一个参数）：
    它随队伍人数变，仓里那份示例是 6 + 5，这一帧是 5 + 5 —— 上下缘、左缘、卡距、卡宽
    两帧同值，只有张数不同。
    """
    if isinstance(counts, (str, bytes)) or not isinstance(counts, Sequence):
        raise TypeError("每行几张要是整数序列，收到 " + type(counts).__name__)
    if len(counts) != len(NAME_ROWS):
        raise ValueError(
            "每行几张要正好 " + repr(len(NAME_ROWS)) + " 个数（名字行也是这么多），收到 "
            + repr(len(counts))
        )
    boxes: list[tuple[int, int, int, int]] = []
    for row, count in enumerate(counts):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("第 " + repr(row) + " 行几张要是非负整数，收到 " + repr(count))
        top, _bottom = NAME_ROWS[row]
        for index in range(count):
            left = FORMATION_LEFT0 + FORMATION_PITCH * index + NAME_LEFT_EDGE
            boxes.append(
                (
                    left,
                    top - PAD_ABOVE,
                    NAME_ART_WIDTH,
                    CELL_HEIGHT,
                )
            )
    return tuple(boxes)


def _cell(frame: Frame, box: tuple[int, int, int, int], span: int) -> np.ndarray:
    """一块取区：**名字以外一律压深**（用户裁定 2/5）—— 上缘以上、下缘以下都填成深色。

    `span` 是**这一行**名字占几行（`NAME_SPANS`）：两行不一样长，短的那行在下面多抹
    一行，好让每一格都是 `CELL_HEIGHT` 高（栅格要等格高）。
    """
    if not PAD_ABOVE <= span <= CELL_HEIGHT - PAD_BELOW:
        raise ValueError(
            "这一行名字占 " + repr(span) + " 行，塞不进 " + repr(CELL_HEIGHT)
            + " 高的格（上下各要留 " + repr(PAD_ABOVE) + " / " + repr(PAD_BELOW) + " 行）"
        )
    x, y, width, height = box
    cell = np.ascontiguousarray(frame.pixels[y : y + height, x : x + width])
    if cell.shape[0] != height or cell.shape[1] != width:
        raise ValueError(
            "取区落到了画布外：要 " + repr(box) + "，而这一帧是 "
            + repr(frame.pixels.shape) + " —— 编队页的几何是在 2800x1260 上量的"
        )
    cell[0:PAD_ABOVE, :, :] = 0
    cell[PAD_ABOVE + span :, :, :] = 0
    return cell


def compose_sheet(frame: Frame, *, counts: Sequence[int]) -> np.ndarray:
    """一帧编队页 → **一张竖排拼图**（一行一格、行间留白）。

    **确定性**（用户裁定 4）：同样的输入 ⇒ 逐字节相同的 PNG —— 缓存键就靠它。
    所以这里只做切片与拼接，**不插值、不缩放、不加任何抖动**。
    """
    if not isinstance(frame, Frame):
        raise TypeError("要拼的是一帧编队页，收到 " + type(frame).__name__)
    spans = [NAME_SPANS[row] for row, count in enumerate(counts) for _ in range(count)]
    cells = [
        _cell(frame, box, span) for box, span in zip(name_boxes(counts), spans)
    ]
    if not cells:
        raise ValueError("一帧编队页至少得切出一块名字取区，收到 0 块")
    height, width = cells[0].shape[0], cells[0].shape[1]
    rows = len(cells) * (height + CELL_GAP) - CELL_GAP
    sheet = np.zeros((rows, width, 3), np.uint8)
    for index, cell in enumerate(cells):
        at = index * (height + CELL_GAP)
        sheet[at : at + height, :, :] = cell
    return sheet


def sheet_png(sheet: np.ndarray) -> bytes:
    """拼图 → PNG 字节。压缩级别**写死**，否则同一张图两次编码未必同字节。"""
    if not isinstance(sheet, np.ndarray) or sheet.dtype != np.uint8 or sheet.ndim != 3:
        raise TypeError("拼图要是 (高, 宽, 3) 的 uint8 RGB，收到 " + type(sheet).__name__)
    ok, buffer = cv2.imencode(
        ".png", np.ascontiguousarray(sheet[:, :, ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 6]
    )
    if not ok:
        raise RuntimeError("拼图编不成 PNG")
    return buffer.tobytes()


def sheet_digest(png: bytes) -> str:
    """拼图字节 → 指纹（缓存键，也是产物文件名里那个「局指纹」）。"""
    return hashlib.blake2b(png, digest_size=16).hexdigest()


def parse_words(response: Mapping) -> tuple[tuple[str, float, float], ...]:
    """原始响应 → `(词, 中心 x, 中心 y)`，次序照接口给的次序。

    **位置开关必须开着**：关着就没有 location，分不清哪个词属于哪张卡 ——
    这里当场报错，不猜。`code` 不是 `0` 就报 `OcrRefused`（把接口的话原样带出来）。
    """
    if not isinstance(response, Mapping):
        raise TypeError("响应要是一个对象，收到 " + type(response).__name__)
    if str(response.get("code")) != "0":
        raise OcrRefused(response.get("code"), response.get("desc"))
    words: list[tuple[str, float, float]] = []
    data = response.get("data") or {}
    for block in data.get("block") or ():
        for line in block.get("line") or ():
            for word in line.get("word") or ():
                location = word.get("location")
                if not isinstance(location, Mapping):
                    raise ValueError(
                        "这一条词没有 location —— 位置开关没开，分不清它属于哪张卡："
                        + repr(word.get("content"))
                    )
                top_left = location.get("top_left") or {}
                bottom_right = location.get("right_bottom") or {}
                center_x = (float(top_left["x"]) + float(bottom_right["x"])) / 2.0
                center_y = (float(top_left["y"]) + float(bottom_right["y"])) / 2.0
                words.append((str(word.get("content", "")), center_x, center_y))
    return tuple(words)


def assign_words(
    words: Sequence[tuple[str, float, float]], grid: RosterGrid, *, slots: int
) -> tuple[tuple[str, ...], ...]:
    """每个格收到哪些词 —— **照记，不抛**；抛是 `read_names` 那道闸门的事。

    照着记是因为：账要能报出来（缺了哪几格、哪一格多了），而「要不要写盘」是另一件事。
    """
    gather: list[list[str]] = [[] for _ in range(slots)]
    for text, x, y in words:
        slot = grid.slot_of(x, y)
        if slot is None or slot >= slots:
            continue
        gather[slot].append(text)
    return tuple(tuple(item) for item in gather)


PUNCT: Final[tuple[str, ...]] = (
    "·", "・", "•", "．", ".", "。", "，", ",", "、", ";", "；", ":", "：",
    "!", "！", "?", "？", " ", chr(0x3000), chr(0x09), chr(0x0A), chr(0x0D),
    "-", "—", "–", "~", "～", "'", '"', "(", ")", "（", "）", "[", "]", "【", "】",
    "<", ">", "《", "》",
)
"""归一化时**抹掉**的字符（用户裁定 3a）：间隔号、各种标点、空白、括号。"""


def normalize(text: str) -> str:
    """归一化：全角→半角 · 抹掉标点与空白 · 大小写折成一套（用户裁定 3a）。

    **只做这些**：不做同音、不做简繁、不做字形近似 —— 那已经不是「归一化」了。
    """
    if not isinstance(text, str):
        raise TypeError("要归一化的是文本，收到 " + type(text).__name__)
    out: list[str] = []
    for char in text:
        if char in PUNCT:
            continue
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            char = chr(code - 0xFEE0)
        out.append(char)
    return "".join(out).casefold()


def edit_distance(first: str, second: str) -> int:
    """Levenshtein 距离（**多一字 / 少一字 / 错一字**各算 1）——只判 ≤ 1 那一段，超了就早退。"""
    if abs(len(first) - len(second)) > 1:
        return 2
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, 1):
        current = [i]
        for j, right in enumerate(second, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right))
            )
        if min(current) > 1:
            return 2
        previous = current
    return previous[-1]


def roster_index(names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """全量名单 → `归一化之后 → 名单里那一条（或几条）`。

    归一化之后**撞车**（两条名单归一化成同一个键）的值是一个**多于一条**的元组 ——
    判决那边见到就不算「精确命中」，而落**未知**：分不开就不许挑一个（用户裁定 3c）。
    """
    gather: dict[str, list[str]] = {}
    for name in names:
        if not isinstance(name, str) or not name:
            raise TypeError("名单里每一条要是非空字符串，收到 " + repr(name))
        gather.setdefault(normalize(name), []).append(name)
    return {key: tuple(value) for key, value in gather.items()}


def decide(word: str, index: Mapping[str, tuple[str, ...]]) -> tuple[str | None, str]:
    """一个词 → `(名单条目 | None, 怎么来的)`（用户裁定 3 的那四步）。"""
    key = normalize(word)
    if not key:
        return (None, "未知")
    exact = index.get(key)
    if exact is not None:
        if len(exact) == 1:
            return (exact[0], "精确")
        return (None, "未知")
    candidates: set[str] = set()
    for other, names in index.items():
        if abs(len(other) - len(key)) > 1:
            continue
        if edit_distance(other, key) <= 1:
            candidates.update(names)
    if len(candidates) == 1:
        return (next(iter(candidates)), "改过")
    return (None, "未知")


def credentials_from_env() -> tuple[str, str]:
    """凭据只从环境变量读（用户裁定 6）。缺哪个报哪个 —— **不编、不落盘、不打印值**。"""
    app_id = os.environ.get(APP_ID_VAR)
    api_key = os.environ.get(API_KEY_VAR)
    missing = [
        name for name, value in ((APP_ID_VAR, app_id), (API_KEY_VAR, api_key)) if not value
    ]
    if missing:
        raise RuntimeError(
            "凭据不在环境变量里：" + "、".join(missing)
            + "（用户裁定 6：只从环境变量读；也可能只是没继承到 —— 这条链要求变量在"
            + "主机进程启动前就在环境里，或由调用方显式注入）"
        )
    return str(app_id), str(api_key)


PostFn = Callable[..., str]
# 那条缝：`(endpoint, body, headers, *, relaxed, timeout_s) -> 原始响应文本`。


def _require_https(endpoint: str) -> None:
    """端点只收 `https`（用户裁定 Q1④：**任何情况下不许明文 http**）。"""
    if urllib.parse.urlsplit(endpoint).scheme != "https":
        raise ValueError(
            "端点只收 https（用户裁定 Q1④：任何情况下不许明文 http），收到 " + repr(endpoint)
        )


def _build_context(*, relaxed: bool) -> ssl.SSLContext:
    """一个**照验证书**的 TLS 上下文；`relaxed` 只是把安全级放到 0（用户裁定 Q1②）。"""
    context = ssl.create_default_context()
    if relaxed:
        context.set_ciphers(RELAXED_CIPHERS)
    return context


def _causes(bad: BaseException) -> list[BaseException]:
    """一条异常与挂在它上面的原因链（`__cause__` / `__context__` / `URLError.reason`）。"""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = bad
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        reason = getattr(current, "reason", None)
        if isinstance(reason, BaseException):
            current = reason
        else:
            current = current.__cause__ or current.__context__
    return chain


def _is_handshake_signature_error(bad: BaseException) -> bool:
    """是不是「握手上签名类型不对」那一类 —— 用户裁定 Q1②：**只**对它降级。

    别的错（超时、DNS、证书校验失败、HTTP 4xx、JSON 坏了）一律**照抛**：降级救不了它们，
    拿降级去重试就是拿额度瞎撞。
    """
    for item in _causes(bad):
        if getattr(item, "reason", None) == "WRONG_SIGNATURE_TYPE":
            return True
        text = str(item)
        if "WRONG_SIGNATURE_TYPE" in text or "wrong signature type" in text:
            return True
    return False


def _request_parts(
    png: bytes, *, app_id: str, api_key: str, language: str, location: str
) -> tuple[bytes, dict[str, str]]:
    """这一趟要发的东西：表单 + 那四个头。

    `X-CurTime` 是**现取**的，所以降级重试要**重建**一次 —— 别把过期的时间戳发第二遍。
    """
    param = base64.b64encode(
        json.dumps({"language": language, "location": location},
                   separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    cur_time = str(int(time.time()))
    checksum = hashlib.md5((api_key + cur_time + param).encode("utf-8")).hexdigest()
    body = urllib.parse.urlencode(
        {"image": base64.b64encode(png).decode("ascii")}
    ).encode("utf-8")
    headers = {
        "X-Appid": app_id,
        "X-CurTime": cur_time,
        "X-Param": param,
        "X-CheckSum": checksum,
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
    }
    return body, headers


def _post(
    endpoint: str,
    body: bytes,
    headers: Mapping[str, str],
    *,
    relaxed: bool,
    timeout_s: float,
) -> str:
    """打一次，回来**原始响应文本**（不加工、不裁剪）。`relaxed` 决定这条连接的安全级。"""
    request = urllib.request.Request(endpoint, data=body, method="POST", headers=dict(headers))
    context = _build_context(relaxed=relaxed)
    with urllib.request.urlopen(request, timeout=timeout_s, context=context) as response:
        return response.read().decode("utf-8", "replace")


def send_ocr(
    png: bytes,
    *,
    app_id: str,
    api_key: str,
    endpoint: str = DEFAULT_ENDPOINT,
    language: str = "cn|en",
    location: str = "true",
    timeout_s: float = 20.0,
    post: PostFn | None = None,
) -> OcrAttempt:
    """**默认安全级**先打一次；只在握手签名那一类错误上、对**那一次连接**降级重试一次。

    用户裁定 Q1：**不是无条件降级** —— 默认那条路继续走，它自己会碰到通的那一窗。
    `post=` 是那条缝：喂一个假的就能**不联网**把整条阶梯走一遍（单测就是那么打的）；
    真那条（`None`）走的两次都**照验证书、都加密**，且端点只收 `https`。
    """
    if location != "true":
        raise ValueError(
            "位置开关必须开着（位置是交叉核对唯一的手段），收到 location=" + repr(location)
        )
    _require_https(endpoint)
    send = _post if post is None else post
    started = time.monotonic()
    body, headers = _request_parts(
        png, app_id=app_id, api_key=api_key, language=language, location=location
    )
    try:
        raw = send(endpoint, body, headers, relaxed=False, timeout_s=timeout_s)
    except Exception as bad:
        if not _is_handshake_signature_error(bad):
            raise
        reason = str(bad)
        first = time.monotonic() - started
        body, headers = _request_parts(
            png, app_id=app_id, api_key=api_key, language=language, location=location
        )
        raw = send(endpoint, body, headers, relaxed=True, timeout_s=timeout_s)
        return OcrAttempt(
            raw=raw,
            downgraded=True,
            reason=reason,
            seconds=time.monotonic() - started,
            first_seconds=first,
        )
    return OcrAttempt(raw=raw, downgraded=False, reason=None, seconds=time.monotonic() - started)


def ask_ocr(
    png: bytes,
    *,
    app_id: str,
    api_key: str,
    endpoint: str = DEFAULT_ENDPOINT,
    language: str = "cn|en",
    location: str = "true",
    timeout_s: float = 20.0,
    post: PostFn | None = None,
) -> str:
    """打**一次**，回来的是**原始响应文本**（不加工、不裁剪）。

    它只是 `send_ocr` 那本账「只要原文」那一面：要看走了哪条路、降级没有，用 `send_ocr`。
    """
    return send_ocr(
        png,
        app_id=app_id,
        api_key=api_key,
        endpoint=endpoint,
        language=language,
        location=location,
        timeout_s=timeout_s,
        post=post,
    ).raw


def read_names(
    page: Frame,
    *,
    counts: Sequence[int],
    transport: Callable[[bytes], str | OcrAttempt],
    roster: Mapping[str, tuple[str, ...]],
    cache_dir: str | Path | None,
    grid: RosterGrid = VERTICAL,
    strict: bool = True,
) -> RosterReading:
    """一帧编队页 → 一局的名单（**一局一次**；同一张拼图再读 0 次调用）。

    `transport` 是那条缝：喂拼图的 PNG 字节、吐**原始响应文本**；也可以直接吐 `send_ocr`
    回的 `OcrAttempt`（那样那本账就顺着 `attempt` 落进读数，用户裁定 2）。给了它就不走网络，
    所以夹具那一份响应能原样喂进来当输入（网就打在它上面）。

    **缓存键 = 拼图字节的 blake2b**（用户裁定 4）：同一张图第二次进来 `calls` 就是 0。
    `cache_dir=None` = **不读也不写缓存**（秤/证据用）：那种场合喂进来的响应未必对应这张
    拼图（夹具里那一份就是旧版面的），写进缓存等于把它污染成「这张图的答案」。

    `strict=True`（产物那条路）在「一格一词」不成立时抛 `RosterLayoutError`；
    `strict=False` 只给秤/证据用 —— **判据一个字没变**，变的只是「发现缺词之后是抛还是记」。

    **账接在读数上**（用户裁定 2）：`attempt` / `downgraded` / `seconds` 说的是这一读
    **真打了网络**的那一次；撞缓存、或 transport 只吐一个字符串时它们说的是「没有账」，
    **不是**「走的默认那一路」—— 那两件事不一样，别混。
    """
    sheet = compose_sheet(page, counts=counts)
    png = sheet_png(sheet)
    digest = sheet_digest(png)
    path = None if cache_dir is None else Path(cache_dir) / (digest + ".json")
    attempt: OcrAttempt | None = None
    if path is not None and path.is_file():
        raw = path.read_text(encoding="utf-8")
        calls = 0
    else:
        delivered = transport(png)
        calls = 1
        if isinstance(delivered, OcrAttempt):
            attempt = delivered
            raw = delivered.raw
        else:
            raw = delivered
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(raw, encoding="utf-8", newline="\n")
    per_slot = assign_words(parse_words(json.loads(raw)), grid, slots=sum(counts))
    missing = [slot for slot, items in enumerate(per_slot) if not items]
    extra = [slot for slot, items in enumerate(per_slot) if len(items) > 1]
    if strict and (missing or extra):
        raise RosterLayoutError(missing, extra)
    decisions: list[NameDecision] = []
    for slot, items in enumerate(per_slot):
        if items:
            name, how = decide(items[0], roster)
        else:
            name, how = (None, "未知")
        decisions.append(
            NameDecision(slot=slot, word=(items[0] if items else ""), name=name, how=how)
        )
    return RosterReading(
        names=tuple(item.name for item in decisions),
        slots=tuple(decisions),
        calls=calls,
        digest=digest,
        png=png,
        attempt=attempt,
    )


def load_roster_names(path: str | Path = NAMES_FILE) -> tuple[str, ...]:
    """全量闭集 → 名字序列。文件里是 `{"version", "source", "names": [...]}`。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("全量名单是一个 JSON 文件，" + str(path) + " 不是")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, Mapping) or "names" not in record:
        raise ValueError(str(path) + " 里没有 names 那一段")
    names = record["names"]
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise TypeError(str(path) + " 的 names 要是序列，收到 " + type(names).__name__)
    return tuple(names)


def write_roster(
    reading: RosterReading,
    *,
    page: str,
    page_sha256: str,
    directory: str | Path = STATE_DIR,
    faces: Sequence[str | None] | None = None,
) -> Path:
    """**一局产物**落盘：`<局指纹>.json`。

    它**不是库**：库按干员累积、跨局（那一份住 `assets/roster/`，见 `face_library`）；
    一局产物只记**这一局队伍那 10 个**（名字 + 卡面引用 + 编队页出处），是**建库那一步
    的凭据** ⇒ 住运行期状态（`STATE_DIR`，`data/` 之下、**不进版本库**）。

    **见到「未知」就拒绝落盘**：一份带未知的名单当库的键用，就是把「没读出来」
    当成「查无此人」。

    **`faces` 就是那份「卡面引用」**（可选）：一格一个库里那条卡面的 id
    （`face_library.face_id`），那一局库里没有就给 `None`。给了就逐格多落一个 `face`、
    并写下它从哪份库解析（`faces_from`）；**不给就一个字节都不多写** ——
    夹具里那份样本是那么落的，它的哈希不许因为加了这一维而漂。
    """
    unknown = [item.slot for item in reading.slots if item.name is None]
    if unknown:
        raise ValueError(
            "这份名单里有 " + repr(len(unknown)) + " 格是未知（" + repr(unknown)
            + "）—— 拒绝落盘：库的名单不许带未知（用户裁定 3）"
        )
    if faces is not None and len(faces) != len(reading.slots):
        raise ValueError(
            "卡面引用要给每一格一个（" + repr(len(reading.slots)) + " 个），收到 "
            + repr(len(faces)) + " 个"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (reading.digest + ".json")
    slots: list[dict] = []
    for position, item in enumerate(reading.slots):
        entry = {"slot": item.slot, "word": item.word, "name": item.name, "how": item.how}
        if faces is not None:
            entry["face"] = faces[position]
        slots.append(entry)
    record = {
        "version": 1,
        "kind": "game_roster",
        "digest": reading.digest,
        "page": {"image": page, "sha256": page_sha256},
        "calls": reading.calls,
        "names": [item.name for item in reading.slots],
        "slots": slots,
    }
    if faces is not None:
        record["faces_from"] = "assets/roster/faces.json"
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
        newline="\n",
    )
    return path
