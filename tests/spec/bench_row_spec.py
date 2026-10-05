"""备战席那一行的规格符合性验收 —— **由规划者维护，不住在 tests/unit/**。

分工、纪律、跑法都在 [README](README.md) —— **不在这里重述**（同一条规定只写一次）。

## 真值从哪来

`tests/帧样本/备战席/` 那 10 张原生裁剪，规划者**逐帧目视确认**过。真值表在下面 `TRUTH`。
**执行者不许读本文件。**

**费用那一列只有 `shotNN` 那 6 帧写**（`cdNN` 那 4 帧是 `None`）：那 4 帧上有几张卡的费用
读数链**读不出来**，而那与"再部署那一列"是**两件事**（理由见批次 README §里面有什么）。

## 这一批覆盖到什么

- **卡数六档**：12 · 11 · 10 · 9 · 8 · 6；
- **费用一位数与两位数**：一位数 1 · 2 · 6，两位数 10–25（**只在 `shotNN` 那 6 帧上**）；
- **装置卡**：费用写 14 的那张，它不是干员（[视觉层/备战席点名](../../docs/10-规格/视觉层/备战席点名.md) §装置那一张长什么样）；
- **再部署中那一档**：`shot26` 最右那张 —— 红罩 + 白色环形进度 + 中央 `12.9`；
- **再部署那一档的另外三副面孔**（`cdNN` 那 4 帧）：**刚撤退**（白弧 31°，`cd01` 下标 7）、
  **真环落进判不了**（`cd08` 下标 5）、**亮卡面被误判**（`cd20` / `cd60` 下标 2）。

**没覆盖到**：灰掉那一档（费用不足）、空席（一张卡都没有）、3 位数费用。

## 这一批里的判据是「两个东西」而不是一个

再部署那一列的三档（是 / 否 / 判不了）在 `cdNN` 那 4 帧上**同时**被抓到两处毛病，而它们是**两种性质**：

- **假环**（`cd20` / `cd60` 下标 2）：那一张卡面很亮，环带上"最暗的那一度"也跟着亮，
  于是判据说"环在"。**这是假事实** —— 它把"不在再部署"说成"在"（R8）；
- **真环落进判不了**（`cd08` 下标 5）：环和数字都在（剩 16.3 s），读数链却报**判不了**。
  这一条**不是假事实**（判不了是"不知道"，是诚实的），它是**丢信息**（R9）。

两者同一个病根：**"环在不在"问的是"环带比它周围亮不亮"，不是"环带本身够不够亮"** ——
一张亮卡面能把绝对亮度顶到真环的高度（实测：同一张卡面上，真环那一次环带最暗 110 / 环外 29，
假环那一次 102 / 158，**绝对亮度只差 8、真值却反着**）。

## 还没写的是哪一半

R6 管"再部署剩余秒数"那一列 —— 它现在**读不出来**（中央那个数字的字体与出厂字形表
不是同一套，见 §尚未定），所以 R6 断言的是**"不许猜一个数"**，不是"等于 12.9"。
"""

from __future__ import annotations

import importlib
import json
import struct
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _entry in (str(REPO / "src"), str(_HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

BATCH = REPO / "tests" / "帧样本" / "备战席"
GLYPH_DIR = REPO / "assets" / "features" / "glyphs"

CANVAS_W, CANVAS_H = 2800, 1260


# ---------------------------------------------------------------------------
# 真值：规划者逐帧目视确认（10 帧）
#
# `cost` 是**从右往左**排的：下标 0 = 最右那张（代码里 `card` 0 的约定）；
# `cdNN` 那 4 帧**不写费用**（`None`），理由见模块文档。
# `device` 是"不是干员"的那些下标；`redeploy` 是"再部署中"的下标 → 中央那个秒数。
#
# `cd01` 下标 2 与 `cd20` 下标 2 是同一个干员、同一个费用：前者在再部署中（剩 54.7），
# 后者不在 —— 这一对就是 R8。
# ---------------------------------------------------------------------------

TRUTH: Mapping[str, Mapping[str, object]] = {
    "shot01": {
        "cards": 12,
        "cost": [25, 23, 23, 21, 18, 17, 14, 13, 13, 12, 12, 11],
        "device": [6],
        "redeploy": {},
    },
    "shot02": {
        "cards": 10,
        "cost": [25, 18, 17, 14, 13, 13, 12, 12, 11, 10],
        "device": [3],
        "redeploy": {},
    },
    "shot05": {
        "cards": 9,
        "cost": [18, 17, 14, 14, 13, 13, 12, 11, 10],
        "device": [2],
        "redeploy": {},
    },
    "shot07": {
        "cards": 8,
        "cost": [18, 17, 14, 14, 13, 12, 11, 10],
        "device": [2],
        "redeploy": {},
    },
    "shot21": {
        "cards": 11,
        "cost": [25, 23, 23, 17, 14, 13, 13, 12, 12, 11, 10],
        "device": [4],
        "redeploy": {},
    },
    "shot26": {
        "cards": 6,
        "cost": [18, 12, 11, 6, 2, 1],
        "device": [],
        "redeploy": {0: 12.9},
    },
    "cd01": {
        "cards": 9,
        "cost": None,
        "device": [],
        "redeploy": {0: 32.2, 1: 39.5, 2: 54.7, 5: 36.0, 7: 64.2, 8: 16.0},
    },
    "cd08": {
        "cards": 9,
        "cost": None,
        "device": [],
        "redeploy": {0: 12.6, 1: 19.9, 2: 35.0, 3: 69.4, 4: 57.2, 5: 16.3, 7: 44.6},
    },
    "cd20": {
        "cards": 11,
        "cost": None,
        "device": [],
        "redeploy": {3: 34.3, 4: 68.4, 5: 33.6, 6: 21.4, 9: 8.8},
    },
    "cd60": {
        "cards": 11,
        "cost": None,
        "device": [],
        "redeploy": {},
    },
}

PAIR = ("cd01", 2, "cd20", 2)
"""R8 用的那一对：**同一个干员、同一个费用**，一次在再部署中、一次不在。

两帧上的卡面像素几乎一样（实测整张卡的平均差 3.3 —— 差的是费用暗盒里那个数），
所以"这两张必须一个是一个否"这句话，只有**把环和它周围比**才答得出来。
"""


# ---------------------------------------------------------------------------
# 骨架
# ---------------------------------------------------------------------------


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


_LOADED: dict[str, object] = {}


def _mod(dotted: str):
    """import 一个还不一定存在的模块。不存在 = 这条检查红，不是检查炸了。"""
    if dotted not in _LOADED:
        try:
            _LOADED[dotted] = importlib.import_module(dotted)
        except BaseException as exc:  # noqa: BLE001 —— 缺模块就是这批检查的红法
            raise AssertionError(
                f"`{dotted}` 还不存在：{type(exc).__name__}: {exc}"
            ) from None
    return _LOADED[dotted]


def _index() -> dict:
    return json.loads((BATCH / "index.json").read_text(encoding="utf-8"))


def _frame_names() -> list[str]:
    """裁剪的文件名去掉扩展名 = 帧名（[基准集] §2）。"""
    return [Path(c["file"]).stem for c in _index()["crops"]]


def _entry(name: str) -> dict:
    for e in _index()["crops"]:
        if Path(e["file"]).stem == name:
            return e
    raise AssertionError(f"夹具里没有 {name}")


def _raw(name: str) -> bytes:
    """把夹具裁剪**贴回** 2800×1260 的黑底，编成设备层 §4 定义的原始 RGBA 字节。

    这样 ROI 的取法仍然是真的在被测 —— 没有把 ROI 预先替读数链做好。
    """
    e = _entry(name)
    x, y = e["origin"]
    w, h = e["size"]
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    canvas[y : y + h, x : x + w] = np.asarray(
        Image.open(BATCH / e["file"]).convert("RGB")
    )
    rgba = np.dstack([canvas, np.full((CANVAS_H, CANVAS_W), 255, np.uint8)])
    return struct.pack("<4I", CANVAS_W, CANVAS_H, 1, 1) + rgba.tobytes()


_READ_CACHE: dict[str, tuple] = {}


def _read(name: str):
    """走**设备层 §5 的重放路径**读一帧：字节 → 帧 → 读数链。同一帧只读一次。"""
    if name not in _READ_CACHE:
        frame = _mod("ark.engine.vision.frame").frame_from_raw(_raw(name))
        glyphs = _mod("ark.engine.vision.glyphs").GlyphTable.load(GLYPH_DIR)
        pitch = _mod("ark.engine.device.adb").SQUAD_CARD_PITCH
        _READ_CACHE[name] = _mod("ark.engine.vision.bench").read_squad_row(
            frame, glyphs, pitch=pitch
        )
    return _READ_CACHE[name]


# ---------------------------------------------------------------------------
# R 族：这一批长什么样
# ---------------------------------------------------------------------------


def check_batch_loads_with_its_manifest() -> str:
    """`index.json` 里那几样必填字段，加载器一样不少地读出来。

    出处是[基准集] §2 的字段表。**`facts` 没写就是没有** —— 这一批同样没记前提头。
    """
    batch = _mod("ark.offline.bench.batch").load_batch(BATCH)
    raw = _index()
    assert batch.version == raw["version"] >= 1, (
        f"版本号读成了 {batch.version!r}，清单里写的是 {raw['version']!r}"
    )
    assert (batch.canvas.width, batch.canvas.height) == (CANVAS_W, CANVAS_H), (
        f"画布读成了 {batch.canvas}，清单里写的是 {raw['canvas']}"
    )
    assert isinstance(batch.device, str) and batch.device, "机型没读出来 —— 它是出处"
    assert len(batch.cases) == len(raw["crops"]), (
        f"读出来 {len(batch.cases)} 条，清单里写了 {len(raw['crops'])} 条"
    )
    assert getattr(batch, "facts", None) is None, (
        "这一批没记前提头，加载器不许替它补一个 —— 那是编出来的环境事实"
    )
    return (
        f"v{batch.version} · 画布 {batch.canvas.width}×{batch.canvas.height} · "
        f"{len(batch.cases)} 条 · facts=None"
    )


def check_every_crop_stays_inside_the_canvas() -> str:
    """每条裁剪的原点与尺寸都落在画布内，**含两端口径**。

    出处是[基准集] §2：右缘 ≤ 宽 − 1、下缘 ≤ 高 − 1。写超了就是"含了一个不存在的像素"。
    """
    raw = _index()
    for crop in raw["crops"]:
        x0, y0 = crop["origin"]
        w, h = crop["size"]
        assert x0 >= 0 and y0 >= 0, f"{crop['file']} 原点 {crop['origin']} 越出左上角"
        assert x0 + w - 1 <= CANVAS_W - 1, (
            f"{crop['file']} 右缘 {x0 + w - 1} 越出画布宽 {CANVAS_W}"
        )
        assert y0 + h - 1 <= CANVAS_H - 1, (
            f"{crop['file']} 下缘 {y0 + h - 1} 越出画布高 {CANVAS_H}"
        )
    return f"{len(raw['crops'])} 条都在画布内（含两端口径）"


def check_truth_covers_exactly_the_batch() -> str:
    """真值表覆盖每一条裁剪、不多不少。

    出处是[基准集] §1：**期望值不住基准集里**，真值住 `tests/spec/`。两边对不上时，
    多出来的是"没有帧的真值"（永远绿不了也红不了），少掉的是"没有真值的帧"（验收
    时会被静默跳过）。
    """
    names = _frame_names()
    missing = [n for n in names if n not in TRUTH]
    extra = [n for n in TRUTH if n not in names]
    assert not missing, f"这几帧没有真值：{missing}"
    assert not extra, f"真值表里这几帧不在批次里：{extra}"
    return f"{len(names)} 帧，真值表逐帧对齐"


def check_truth_table_has_the_shape_it_claims() -> str:
    """真值表自己的形状：卡数 = 费用表的长度（没写费用那几帧跳过），各处下标都在卡数之内。

    出处是[视觉层/备战席] §卡片怎么排（**只有还没上场的才显示**，所以卡数会变）与
    §点名（**一张卡不一定是干员**）。这一条不碰像素，它保证**真值本身不自相矛盾** ——
    真值错了，拿它当标准的验收就一起错。
    """
    worst: list[str] = []
    no_cost: list[str] = []
    for name, row in TRUTH.items():
        cards = row["cards"]
        cost = row["cost"]
        if cost is None:
            no_cost.append(name)
        else:
            assert len(cost) == cards, (
                f"{name}：卡数写 {cards}，费用表有 {len(cost)} 个数"
            )
        for i in row["device"]:
            assert 0 <= i < cards, f"{name}：装置下标 {i} 不在 0..{cards - 1}"
        for i in row["redeploy"]:
            assert 0 <= i < cards, f"{name}：再部署下标 {i} 不在 0..{cards - 1}"
        worst.append(f"{name}:{cards}")
    counts = sorted({row["cards"] for row in TRUTH.values()})
    tail = f" ｜不写费用 {len(no_cost)} 帧（{', '.join(no_cost)}）" if no_cost else ""
    return f"卡数 {counts} ｜ " + " ".join(worst) + tail


def check_readings_match_the_truth_on_every_frame() -> str:
    """10 帧经重放路径读出来，**卡数 · 每张卡的费用（写了的那些帧）· 再部署状态**逐张一致。

    出处是[开发路线] 段一那条判据（"这套识别不看真机也能跑，而且对着基准集逐帧对得上"）
    与[离线验收] §1。真值是规划者逐帧目视确认的（见本文件模块文档），**执行者看不到它**。

    **一处不符也不放过**：这一条把每一处都收进来一起报，而不是撞到第一处就停 ——
    一个病根在几帧上出几处，报全了才看得出是"一处"还是"一片"。
    """
    from ark.contracts import Exact, Valued

    names = _frame_names()
    detail: list[str] = []
    wrong: list[str] = []
    for name in names:
        row = TRUTH[name]
        cards = _read(name)
        assert [c.index for c in cards] == list(range(len(cards))), (
            f"{name}：下标不是从 0 起连着排的 —— {[c.index for c in cards]}"
        )
        assert len(cards) == row["cards"], (
            f"{name}：读出 {len(cards)} 张卡，真值是 {row['cards']} 张"
        )
        if row["cost"] is not None:
            got = [c.cost for c in cards]
            bad = [
                f"k{i}={v!r}"
                for i, v in enumerate(got)
                if not (isinstance(v, Valued) and isinstance(v.payload, Exact))
            ]
            assert not bad, f"{name}：这几张卡的费用不是精确值 —— " + ", ".join(bad)
            values = [v.payload.value for v in got]
            assert values == list(row["cost"]), (
                f"{name}：费用读出 {values}，真值是 {list(row['cost'])}"
            )
        want_red = set(row["redeploy"])
        for card in cards:
            if card.redeploying != (card.index in want_red):
                said = {True: "再部署中", False: "不在再部署", None: "判不了"}[
                    card.redeploying
                ]
                wrong.append(
                    f"{name} 下标 {card.index} 读成「{said}」、真值是"
                    f"「{'再部署中' if card.index in want_red else '不在再部署'}」"
                )
        detail.append(f"{name}:{len(cards)}张")
    assert not wrong, (
        f"再部署那一列有 {len(wrong)} 处不符 ｜ " + "；".join(wrong)
    )
    return f"{len(names)} 帧逐张一致 ｜ " + " ".join(detail)


def check_countdown_seconds_are_never_guessed() -> str:
    """再部署剩余秒数：**要么精确等于真值，要么落未知 / 未采样 —— 不许是别的数**。

    出处是[状态层/我方干员] 那一行的形态（精确 / 未知 / 未采样）与"零假事实"这条验收标准。

    **这一条现在为什么不是"等于 12.9"**：实测中央那个数字的字体**不在出厂字形表里**
    （`1` 带底衬线、笔画更粗，最像的字形反而是 `2`），所以读数链如实落**未知**。
    那是一条**已知缺口**（[视觉层/备战席] §尚未定），不是这一条要抓的假事实 ——
    这一条抓的是"**为了交差猜一个数**"。缺口补上之后这一条照样绿。
    """
    from ark.contracts import Exact, Unknown, Unsampled, Valued

    checked = 0
    detail: list[str] = []
    for name, row in TRUTH.items():
        cards = _read(name)
        for card in cards:
            want = row["redeploy"].get(card.index)
            if want is None:
                continue
            checked += 1
            slot = card.seconds
            if isinstance(slot, Unsampled):
                detail.append(f"{name}#{card.index}=未采样")
                continue
            assert isinstance(slot, Valued), (
                f"{name} 第 {card.index} 张：秒数落成了 {slot!r}"
            )
            payload = slot.payload
            if isinstance(payload, Unknown):
                detail.append(f"{name}#{card.index}=未知")
                continue
            assert isinstance(payload, Exact), f"{name} 第 {card.index} 张：{payload!r}"
            assert abs(payload.value - want) < 0.05, (
                f"{name} 第 {card.index} 张：秒数读成 {payload.value}，真值是 {want} "
                "—— 猜错了比报未知危险"
            )
            detail.append(f"{name}#{card.index}={payload.value}")
    assert checked, "真值里一张再部署中的卡都没有，这一条什么都没验到"
    return f"{checked} 张再部署中 ｜ " + " ".join(detail)


def _ring_shortened(name: str, index: int) -> bytes:
    """把某张卡的环**擦短**：把亮弧那一大段涂成环自己的暗色，只在顶上留约 60°。

    这是**合成帧**，它编码的是一条实测事实：[视觉层/备战席] §判死与再部署 说
    **白弧量的是「已过比例」**（`shot26` 剩 12.9 s → 278°；用户实例图剩 68.6 s → 约 20°）。
    所以"刚撤退"的卡白弧就是短的，而它**仍然在再部署中** —— 环还在，红罩还在。

    真实素材里**没有这一档**（6 帧只有 `shot26` 那一张，而且已经过了 77%），
    所以只能合成。这是[tests/spec/README] 那张表里"第二张网"的做法：**验形状，不验值**。
    """
    e = _entry(name)
    x, y = e["origin"]
    w, h = e["size"]
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    canvas[y : y + h, x : x + w] = np.asarray(
        Image.open(BATCH / e["file"]).convert("RGB")
    )

    cx = 2696.0 - index * 207.5          # 卡中心 x（锚点 − k × 卡距）
    cy = 1045 + 116.5                    # 环心（卡顶下 116.5）
    dim = np.median(                    # 环的暗色：从它自己那一段暗弧上取
        [canvas[int(round(cy - 54 * np.cos(np.deg2rad(a)))),
                int(round(cx + 54 * np.sin(np.deg2rad(a))))]
         for a in range(300, 331, 3)],
        axis=0,
    )
    yy, xx = np.mgrid[int(cy - 60) : int(cy + 61), int(cx - 60) : int(cx + 61)]
    rr = np.hypot(xx - cx, yy - cy)
    ang = (np.degrees(np.arctan2(xx - cx, -(yy - cy)))) % 360
    hit = (rr >= 50) & (rr <= 58) & (ang >= 30) & (ang <= 330)
    block = canvas[int(cy - 60) : int(cy + 61), int(cx - 60) : int(cx + 61)]
    block[hit] = dim.astype(np.uint8)

    rgba = np.dstack([canvas, np.full((CANVAS_H, CANVAS_W), 255, np.uint8)])
    return struct.pack("<4I", CANVAS_W, CANVAS_H, 1, 1) + rgba.tobytes()


def _read_raw(raw: bytes):
    frame = _mod("ark.engine.vision.frame").frame_from_raw(raw)
    glyphs = _mod("ark.engine.vision.glyphs").GlyphTable.load(GLYPH_DIR)
    pitch = _mod("ark.engine.device.adb").SQUAD_CARD_PITCH
    return _mod("ark.engine.vision.bench").read_squad_row(frame, glyphs, pitch=pitch)


def check_redeploying_does_not_depend_on_the_arc_length() -> str:
    """白弧擦短之后，那一张**仍然**必须判成"再部署中"。

    出处是[视觉层/备战席] §判死与再部署：**判据只能是"环在不在"** —— 刚撤退的卡白弧
    接近 0°，按"白弧够不够长"判会把它说成"不在再部署"，**那是假事实**。

    合成帧的做法见 `_ring_shortened`：把亮弧压到约 60°，其余照旧（环还在、红罩还在）。
    """
    row = TRUTH["shot26"]
    index = next(iter(row["redeploy"]))
    cards = _read_raw(_ring_shortened("shot26", index))
    assert len(cards) == row["cards"], (
        f"擦短之后卡数从 {row['cards']} 变成 {len(cards)} —— 环被动过就数不出卡了"
    )
    card = cards[index]
    assert card.redeploying is True, (
        f"第 {index} 张的环还在（只是亮弧被压到约 60°），却判成 {card.redeploying!r} "
        "—— 按白弧长度判会把刚撤退的卡说成不在再部署"
    )
    return f"环擦短到约 60° 后仍判 {card.redeploying}（卡数 {len(cards)} 不变）"


def check_a_bright_card_face_is_not_a_ring() -> str:
    """同一个干员、同一个费用的两张卡，一次在再部署中、一次不在 —— 判据必须把它们分开。

    出处是[视觉层/备战席] §判死与再部署：**判据是「环在不在」**，而环是一道**画在卡面上的
    笔画**，所以"在不在"只能问**它比它周围亮不亮**：这两张卡面几乎一样（整张卡的平均差 3.3），
    真值却反着 —— 拿卡面的**绝对亮度**当判据，任何一条线都会把其中一张说错。
    """
    left, li, right, ri = PAIR
    a = _read(left)[li]
    b = _read(right)[ri]
    assert a.redeploying is True, (
        f"{left} 下标 {li} 真值是在再部署中（剩 54.7 s），读数链读成 {a.redeploying!r}"
    )
    assert b.redeploying is False, (
        f"{right} 下标 {ri} 没有环、没有红罩、卡面是全彩的，读数链却读成 {b.redeploying!r}"
        f" —— 它与 {left} 下标 {li} 是同一个干员、同一个费用，"
        "拿卡面的绝对亮度当判据分不开这两张"
    )
    return f"{left}#{li}={a.redeploying} ／ {right}#{ri}={b.redeploying}（同一张卡面）"


def check_a_real_ring_never_lands_in_the_undecided_band() -> str:
    """真环不许落「判不了」：环与中央数字都在的那些卡，读数链必须报「再部署中」。

    出处是[视觉层/备战席] §判死与再部署（"红罩 + 那个圆环在，就是再部署中"）与
    [状态层/我方干员] 那一行的三档。**这一条不是抓假事实** —— 报「判不了」是诚实的，
    代价是**丢信息**；而这里的证据是**肯定证据**（环在、数字在），
    所以"不知道"这个答案本身是错的。
    """
    undecided: list[str] = []
    total = 0
    for name, row in TRUTH.items():
        cards = _read(name)
        for index in sorted(row["redeploy"]):
            total += 1
            if cards[index].redeploying is not True:
                undecided.append(f"{name}#{index}={cards[index].redeploying!r}")
    assert not undecided, (
        f"{len(undecided)} 张卡的环与数字都在，读数链却没报「再部署中」："
        + "；".join(undecided)
    )
    return f"真环 {total} 张，一张都没落进判不了"


CHECKS: tuple[Check, ...] = (
    Check("R1", "采集与验收/基准集 §2（字段表）",
          "清单里的必填字段一样不少地读出来，没记的前提头不许补",
          check_batch_loads_with_its_manifest),
    Check("R2", "采集与验收/基准集 §2（含两端口径）",
          "每条裁剪的原点与尺寸都落在画布内",
          check_every_crop_stays_inside_the_canvas),
    Check("R3", "采集与验收/基准集 §1（期望值不住这里）",
          "真值表覆盖每一条裁剪，不多不少",
          check_truth_covers_exactly_the_batch),
    Check("R4", "视觉层/备战席 §卡片怎么排 + §点名",
          "真值表自己的形状自洽：卡数 = 费用表长度，下标都在卡数之内",
          check_truth_table_has_the_shape_it_claims),
    Check("R5", "00-总览/开发路线 段一（判据）+ 采集与验收/离线验收 §1",
          "10 帧经重放路径读出来，卡数 · 每张费用 · 再部署状态与真值逐张一致",
          check_readings_match_the_truth_on_every_frame),
    Check("R6", "状态层/我方干员（再部署倒计时的形态）+ 零假事实",
          "再部署剩余秒数要么精确等于真值、要么落未知/未采样，不许猜一个数",
          check_countdown_seconds_are_never_guessed),
    Check("R7", "视觉层/备战席 §判死与再部署（判据只能是「环在不在」）",
          "把亮弧擦短到约 60° 之后，那一张仍然判成再部署中",
          check_redeploying_does_not_depend_on_the_arc_length),
    Check("R8", "视觉层/备战席 §判死与再部署（环是笔画：只能问它比周围亮不亮）",
          "同一个干员、同一个费用的两张卡，一次在再部署中、一次不在，判据要分开",
          check_a_bright_card_face_is_not_a_ring),
    Check("R9", "视觉层/备战席 §判死与再部署 + 状态层/我方干员（三档）",
          "环与数字都在的那些卡，不许落「判不了」",
          check_a_real_ring_never_lands_in_the_undecided_band),
)


def main() -> int:
    failures: list[tuple[Check, str]] = []
    print(f"备战席那一行的规格符合性验收 —— {len(CHECKS)} 条\n")
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
