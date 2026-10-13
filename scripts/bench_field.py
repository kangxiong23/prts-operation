"""
场上我方干员那两条线：**基准集 + 面板真值批**。

    .venv\\Scripts\\python.exe scripts\\bench_field.py

两批，**各自独立**：

| 批 | 目录 | 它是什么 |
| --- | --- | --- |
| **基准集** | tests/帧样本/场上干员/ | 16 块裁剪 + 三条链的判据读数（两簇分布 / 空档 / 覆盖 / **全批**锚点摆动） |
| **面板真值批** | tests/帧样本/场上干员真值/ | 13 块裁剪 + **面板那一行的 生命 当前/上限** 逐帧对照 |

## 面板真值批为什么另开一批

第一批**已经拿它调过参**（HP_COLOUR_TOL / GAP_TOLERANCE / SKILL_FILL_TOL /
DIAMOND_* 都是从那一批量出来的），按 [基准集] §4 它**不再干净**。真值那一批要留成
**没被动过的干净基线**（version 1）—— 所以它是另一批，读数分开落。

**但这一版链已经改过一次，而且不是拿它改的**：技力那一侧换成了"颜色像不像填充色"
（`SKILL_FILL_TOL`）与 `GREEN_MAX_RG = 0`，那是拿 `data/probe/sp_series` 那五帧
（亮底、整条判成填满）定的，**这一批的 13 帧一帧都没参与定参**。这一批的读数在那之后
**重跑了一遍**，所以它现在对"**这一次**改动"来说已经不是"没见过"的了 ——
**下一轮再改链必须换一批**（[基准集] §4）。

## 一条硬纪律：**不许拿面板上的数字去调链**

面板上的数字是**真值**，只进「对照」那一步：链先把 13 帧读完、写进 rows，**之后**
才把真值读进来做减法。链的参数一个都不是从这一批来的 —— 真值只出现在 `TRUTH_PICKS`
里，`_truth_rows()` 里那一圈链调用**看不见**它。

## 配哪一根条

点场上干员开面板时，被点中的干员会被挪到屏幕较中心的位置（场景跟着一起挪），所以
**被选中的那根条 = 离画布中心最近的那根**（2800×1260 的中心 = (1400, 630)）。
`_bar_candidates()` 把一帧上「血条蓝」的横条都找出来，按这条规则挑最近的那根，并把
**第二名**一起写进 取材.json —— 挑错了要看得见。

## 它自己不解 PNG 给识别链

建批时读源帧用 PIL（住离线侧），跑读数时走 `ark.offline.bench` 的加载器 —— 与
`scripts/bench.py` 同一条路。

## 同一批跑两次，两份 JSON 逐字节一样

没有时间戳、没有绝对路径、没有随机量。
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.contracts import (  # noqa: E402
    Exact,
    Interval,
    LowerBound,
    PixelRect,
    SampledEmpty,
    Unknown,
    Unsampled,
    UpperBound,
    Valued,
)
from ark.engine.vision import field  # noqa: E402
from ark.engine.vision.frame import CANVAS_2800x1260, frame_from_raw  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402
from ark.offline.bench.feed import replay_frames  # noqa: E402

BATCH = REPO / "tests" / "帧样本" / "场上干员"
REPORT = BATCH / "判据读数.json"

TRUTH_BATCH = REPO / "tests" / "帧样本" / "场上干员真值"
TRUTH_REPORT = TRUTH_BATCH / "判据读数.json"

REGRESSION_BATCH = REPO / "tests" / "帧样本" / "场上干员回归"
"""**回归批**：`data/probe/sp_series` 那五帧，裁剪同样是 300×420。

它**不是**独立真值批 —— 技力那一侧的判据就是拿这一批的「亮底上整条判成填满」改的，
所以它**见过**，也**不能**再当"没被动过的基线"。它存在的意义只有一条：
以后谁再动技力那一侧，这五帧的读数**一字不许变**。

**真值不落在这里**：`取材.json` 里一个真值字段都没有（真值住在规划者那边）。
这一批**没有** `判据读数.json` —— 钉读数的是 `tests/unit/test_vision_field.py`
里那一段回归用例（那才是"读数不许变"这句话的载体）。
"""
CROP_WIDTH = 300
CROP_HEIGHT = 420
CROP_LEFT = 80
"""裁剪比取区（240×360）**大一圈**：左右各多 40、上下各多 20 上下。"""

CANVAS_CENTRE = (CANVAS_2800x1260.width / 2.0, CANVAS_2800x1260.height / 2.0)
"""画布中心 —— 配对那条规则的圆心。"""

CANDIDATE_MIN_LIT = 12
CANDIDATE_MIN_HEIGHT = 6
"""一根条要算候选，至少得有这么长的一段亮着、且横跨这么多行。

**12 是被这一批逼出来的**：p08–p11 与 sp_now 那五帧上，被点中的那根条只填了
**27 列**（面板 329/1801），而同一帧上还有一根填了 108 列的条—— 阈值一抬高
（我先用的 60），被点中的那根**根本不进候选**，配对就挑到别的东西上去了。
一句话：**填得很少的条仍然是一根条**。这条线只用来**找候选**，不判填充、
不判含义 —— 判那些的是链里的 MIN_FILL_COLUMNS。
"""

TEMPLATE_FRAME = REPO / "data" / "probe" / "rawcol_1.bin"
TEMPLATE_BOX = (1849 - 32, 243 - 32, 64, 64)
"""模板那一块在取材帧上的 (x, y, 宽, 高)。"""

DEVICE = "vivo V2338A"

# 一条一条：这一批要覆盖什么。键是帧名，值是 (源帧, 血条左上角 x, y, 这一条补的是什么)。
PICKS: tuple[tuple[str, str, int, int, str], ...] = (
    ("cd01_a", "data/precheck/cd/cd01.png", 1042, 629, "血条满 + 技力充能中（绿）"),
    ("cd01_b", "data/precheck/cd/cd01.png", 1025, 800, "血条满 + 技力生效中（橙实心）"),
    ("cd01_c", "data/precheck/cd/cd01.png", 1597, 476, "血条 0.884 + 技力弹药分格（满）"),
    ("cd12_a", "data/precheck/cd/cd12.png", 1597, 476, "血条 0.740 + 弹药分格"),
    ("shot01_a", "data/precheck/live/shot01.png", 1406, 486, "血条满 + 充能中 + 菱形不在"),
    ("shot02_a", "data/precheck/live/shot02.png", 1406, 486, "血条 0.288（很低）+ 充能中 + 菱形在"),
    ("shot04_a", "data/precheck/live/shot04.png", 1406, 486, "血条 0.301 + 技力条被别的条压住（下界）+ 菱形在"),
    ("shot21_a", "data/precheck/live/shot21.png", 1423, 777, "血条满 + 技力 0.863（没满）"),
    ("shot26_a", "data/precheck/live/shot26.png", 1222, 936, "血条满 + 弹药分格**没填满 → 下界**"),
    ("shot31_a", "data/precheck/live/shot31.png", 1609, 584, "血条 0.993 + 技力 0.527（没满）"),
    ("shot30_a", "data/precheck/live/shot30.png", 1429, 959, "血条满 + 充能中 + 菱形在（低分那一侧）"),
    ("shot22_a", "data/precheck/live/shot22.png", 1429, 959, "血条满 + 充能中 + 菱形在（最低分那一侧）"),
    ("board_a", "data/probe/_t5_board/board.raw", 1612, 629, "血条满 + 菱形在（另一个场景）"),
    ("board_b", "data/probe/_t5_board/board.raw", 877, 476, "血条 0.829 + 菱形在"),
    ("field_a", "data/probe/_t5_field/field.raw", 1612, 629, "血条满 + 菱形在"),
    ("field_b", "data/probe/_t5_field/field.raw", 877, 476, "血条 0.781 + 菱形在"),
)

# 面板真值批。元组是：帧名 / 源帧 / 期望的血条左上角 x / y / 生命当前 / 生命上限 /
# 技力当前（-1 = 这一帧里没有可读的 N/M）/ 技力上限 / 备注。
#
# **真值只看一次、只写在这一张表里**：它来自面板上「生命 当前/上限」那一行，
# 是我逐帧看图读出来的（读的地方记在 取材.json 的 truth_row 里，谁都能重读一遍）。
# 期望的血条坐标是我另跑一遍「血条蓝」扫描得到的，写在这里是为了**卡死配对**：
# 扫描结果与它不一致就直接抛，不许悄悄换一根条。
TRUTH_PICKS: tuple[tuple[str, str, int, int, int, int, int, int, str], ...] = (
    ("p00", "data/probe/panel_run/p00.png", 1487, 709, 2351, 2867, -1, -1, "面板 新约能天使；血条读到 120/146"),
    ("p01", "data/probe/panel_run/p01.png", 1487, 709, 2322, 2867, -1, -1, "同一场；血条 119/146"),
    ("p02", "data/probe/panel_run/p02.png", 1487, 709, 2867, 2867, -1, -1, "满血那一档"),
    ("p03", "data/probe/panel_run/p03.png", 1487, 709, 2839, 2867, -1, -1, "血条 145/146"),
    ("p04", "data/probe/panel_run/p04.png", 1487, 709, 2839, 2867, -1, -1, "与 p03 同一个数"),
    ("p05", "data/probe/panel_run/p05.png", 1487, 709, 2810, 2867, -1, -1, "血条 143/146"),
    ("p06", "data/probe/panel_run/p06.png", 1487, 709, 2781, 2867, -1, -1, "血条 142/146"),
    ("p07", "data/probe/panel_run/p07.png", 1487, 709, 2753, 2867, -1, -1, "血条 140/146"),
    ("p08", "data/probe/panel_run/p08.png", 1487, 709, 329, 1801, -1, -1, "另一场；血条 27/146"),
    ("p09", "data/probe/panel_run/p09.png", 1487, 709, 329, 1801, -1, -1, "与 p08 同一个数"),
    ("p10", "data/probe/panel_run/p10.png", 1487, 709, 329, 1801, -1, -1, "与 p08 同一个数"),
    ("p11", "data/probe/panel_run/p11.png", 1487, 709, 329, 1801, -1, -1, "与 p08 同一个数"),
    ("sp_now", "data/probe/sp_now.png", 1487, 709, 329, 1801, 16, 26, "面板 极境；技力按钮上读到 16/26"),
)

TRUTH_ROW_BOX: dict[str, tuple[int, int, int, int]] = {
    "p00": (100, 560, 1300, 655),
    "p01": (100, 560, 1300, 655),
    "p02": (100, 560, 1300, 655),
    "p03": (100, 560, 1300, 655),
    "p04": (100, 560, 1300, 655),
    "p05": (100, 560, 1300, 655),
    "p06": (100, 560, 1300, 655),
    "p07": (100, 560, 1300, 655),
    "p08": (100, 565, 1300, 660),
    "p09": (100, 565, 1300, 660),
    "p10": (100, 565, 1300, 660),
    "p11": (100, 565, 1300, 660),
    "sp_now": (100, 565, 1300, 660),
}
"""每一帧的**真值是从哪一块读出来的**（画布坐标，左闭右开）。"""


REGRESSION_PICKS: tuple[tuple[str, str, int, int, str], ...] = (
    ("s00", "data/probe/sp_series/s00.png", 1487, 709, "亮底：橙条纹地板"),
    ("s01", "data/probe/sp_series/s01.png", 1487, 709, "亮底：花丛"),
    ("s02", "data/probe/sp_series/s02.png", 1487, 709, "亮底：青地板"),
    ("s03", "data/probe/sp_series/s03.png", 1487, 709, "亮底：青地板"),
    ("s04", "data/probe/sp_series/s04.png", 1487, 709, "亮底：橙条纹地板"),
)
"""五帧：帧名 / 源帧 / 血条左上角 x / y / **这一帧的底是什么**（方便肉眼回去找）。

**这里不写"应该读多少"** —— 那是真值，住规划者那边。血条左上角是规划者给的，
建批时**当场自检**（见 `build_regression_batch`）：按它开的取区里必须真的有一根条，
而且左缘必须落在取区里的 (40, 320)。
"""


def _read_source(relative: str) -> np.ndarray:
    """读一张原生帧 → (高, 宽, 3) 的 RGB。**有损 JPG 一律不收**。"""
    path = REPO / relative
    if path.suffix.lower() in (".jpg", ".jpeg"):
        raise ValueError(f"{relative} 是有损 JPG：帧一律从原生帧裁，JPG 不用")
    if path.suffix.lower() in (".bin", ".raw"):
        raw = path.read_bytes()
        width, height = np.frombuffer(raw[:8], dtype="<u4")
        return (
            np.frombuffer(raw[16 : 16 + int(width) * int(height) * 4], dtype=np.uint8)
            .reshape(int(height), int(width), 4)[:, :, :3]
            .copy()
        )
    return np.asarray(Image.open(path).convert("RGB"))


def _sha256(path: Path) -> str:
    """源帧的指纹 —— **磁盘原样**，不折行尾：帧是二进制，折行尾没有意义。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    """落一份 JSON：**LF、UTF-8、缩进 2**。同一批跑两次逐字节一样。"""
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

def build_batch() -> dict[str, object]:
    """建批：落 PNG + index.json + 取材.json，返回清单。"""
    BATCH.mkdir(parents=True, exist_ok=True)
    crops: list[dict[str, object]] = []
    harvest: list[dict[str, object]] = []
    for name, relative, bar_x, bar_y, note in PICKS:
        source = _read_source(relative)
        origin_x = bar_x - CROP_LEFT
        origin_y = max(0, bar_y - 340)
        block = source[origin_y : origin_y + CROP_HEIGHT, origin_x : origin_x + CROP_WIDTH]
        if block.shape[:2] != (CROP_HEIGHT, CROP_WIDTH):
            raise ValueError(f"{name} 裁出来是 {block.shape[:2]}，落不进画布")
        file_name = f"{name}.png"
        Image.fromarray(block).save(BATCH / file_name)
        crops.append(
            {
                "file": file_name,
                "origin": [origin_x, origin_y],
                "size": [CROP_WIDTH, CROP_HEIGHT],
                "source": relative,
                "source_sha256": _sha256(REPO / relative),
            }
        )
        window = field.operator_ui_window(PixelRect(bar_x, bar_y, field.BAR_WIDTH, 16))
        harvest.append(
            {
                "file": file_name,
                "covers": note,
                "window": [int(window.x), int(window.y), int(window.width), int(window.height)],
                "window_from": (
                    "血条左上角 ("
                    + str(bar_x)
                    + ", "
                    + str(bar_y)
                    + ") − BARS_IN_WINDOW → operator_ui_window()"
                ),
                "source": relative,
            }
        )
    manifest = {
        "version": 1,
        "canvas": [CANVAS_2800x1260.width, CANVAS_2800x1260.height],
        "device": DEVICE,
        "note": (
            "场上我方干员那三条读数链的基准集。每一条是原生帧上 300×420 的一块，"
            "比取区（240×360）大一圈；血条左上角在每一块里的位置见 取材.json。"
        ),
        "crops": crops,
    }
    _write_json(BATCH / "index.json", manifest)
    _write_json(
        BATCH / "取材.json",
        {
            "note": (
                "裁剪的写法与取区的关系。取区由血条左上角折出来（operator_ui_window），"
                "裁剪再比取区大一圈 —— 这样测的是「窗里的哪一块是条」，不是「把条摆好给链子」。"
            ),
            "window_size": [field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT],
            "bars_in_window": list(field.BARS_IN_WINDOW),
            "crop_size": [CROP_WIDTH, CROP_HEIGHT],
            "crop_left_margin": CROP_LEFT,
            "crops": harvest,
        },
    )
    return manifest


def build_template() -> dict[str, object]:
    """从**取材帧**裁一次菱形模板，落进 assets/features/ready_diamond/。"""
    source = _read_source(str(TEMPLATE_FRAME.relative_to(REPO)))
    x, y, width, height = TEMPLATE_BOX
    block = np.ascontiguousarray(source[y : y + height, x : x + width])
    field.save_ready_diamond(
        block,
        field.READY_DIAMOND_PATH,
        source=str(TEMPLATE_FRAME.relative_to(REPO)).replace("\\", "/"),
        box=TEMPLATE_BOX,
    )
    Image.fromarray(block).save(BATCH / "菱形模板.png")
    return {
        "path": "assets/features/ready_diamond/diamond.npz",
        "source": str(TEMPLATE_FRAME.relative_to(REPO)).replace("\\", "/"),
        "box": list(TEMPLATE_BOX),
        "size": [height, width],
    }


# ---------------------------------------------------------------------------
# 槽 → JSON
# ---------------------------------------------------------------------------


def _payload_json(payload: object) -> dict[str, object]:
    if isinstance(payload, Exact):
        return {"form": "exact", "value": payload.value}
    if isinstance(payload, Interval):
        return {"form": "interval", "lo": payload.lo, "hi": payload.hi}
    if isinstance(payload, LowerBound):
        return {"form": "lower_bound", "lo": payload.lo}
    if isinstance(payload, UpperBound):
        return {"form": "upper_bound", "hi": payload.hi}
    if isinstance(payload, Unknown):
        return {"form": "unknown"}
    raise TypeError(f"这一批的载荷只有这五种，收到 {type(payload).__name__}")


def _slot_json(slot: object) -> dict[str, object]:
    if isinstance(slot, Unsampled):
        return {"state": "unsampled"}
    if isinstance(slot, SampledEmpty):
        return {"state": "empty", "confidence": slot.confidence}
    if isinstance(slot, Valued):
        return {
            "state": "valued",
            "payload": _payload_json(slot.payload),
            "confidence": slot.confidence,
        }
    raise TypeError(f"槽只有三态，收到 {type(slot).__name__}")


def _number(slot: object) -> float | None:
    """槽里的那个数（区间取下沿）。给分布统计用；不是数就给 None。"""
    if not isinstance(slot, Valued):
        return None
    payload = slot.payload
    if isinstance(payload, Exact) and isinstance(payload.value, (int, float)) and not isinstance(payload.value, bool):
        return float(payload.value)
    if isinstance(payload, Interval):
        return float(payload.lo)
    if isinstance(payload, LowerBound):
        return float(payload.lo)
    return None


def _spread(values: list[float]) -> dict[str, object]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "median": round(ordered[len(ordered) // 2], 6),
    }


def _gap(values: list[float]) -> dict[str, object]:
    """两簇之间的空档：把有序的读数按最大间隔切开。"""
    if len(values) < 2:
        return {"note": "读数太少，切不出空档"}
    ordered = sorted(set(values))
    best = (0.0, ordered[0], ordered[-1])
    for left, right in zip(ordered, ordered[1:]):
        if right - left > best[0]:
            best = (right - left, left, right)
    return {"largest_gap": round(best[0], 6), "below": round(best[1], 6), "above": round(best[2], 6)}


def _tally(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    """含义那一列各出现了几次。"""
    counts: dict[str, int] = {}
    for row in rows:
        slot = row[key]
        if slot["state"] != "valued":
            label = slot["state"]
        elif slot["payload"]["form"] == "exact":
            label = str(slot["payload"]["value"])
        else:
            label = slot["payload"]["form"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def _form(slot: object) -> str:
    """一个槽的**形态名**（三态或六种载荷），给"结论变没变"用。"""
    if isinstance(slot, Unsampled):
        return "unsampled"
    if isinstance(slot, SampledEmpty):
        return "empty"
    if isinstance(slot, Valued):
        return type(slot.payload).__name__
    return type(slot).__name__


def _label(slot: object) -> str:
    """一个槽在读数表里**显示成什么**：精确值就显示那个值（含义那个词 / 布尔），
    其余显示形态名。与 sensitivity 的 rows 里那几列用同一套写法 —— 两边不一样，
    "变没变"就会比较出**假的翻面**（Exact 对 充能中，永远不相等）。
    """
    if isinstance(slot, Valued) and isinstance(slot.payload, Exact):
        return str(slot.payload.value)
    return _form(slot)


def _payload_key(slot: object) -> tuple:
    """一个槽的"结论"：三态 / 载荷的**形态** / 精确值或含义那个词。

    **置信度不进这个键**：它随分数连续变，挪一格取区就变一点点，那是"读数的锐度"
    不是"结论"。**比例的末位也不进**：取区的行往下错一格，八行的带子就压到条边上一行，
    抗锯齿会让最后一列差 1/146 —— 那是归一化消不掉的**图案差**（[像素匹配诊断] §1），
    不是读数翻面。宽容窗口要吸收的是**翻面**（认成没条 / 认成别的含义 / 菱形从在变不在），
    所以判据定在形态那一级；那个小数在 health_delta 那一列单独报。
    """
    if isinstance(slot, Valued):
        payload = slot.payload
        if isinstance(payload, Exact) and not isinstance(payload.value, float):
            return ("exact", payload.value)
        return (type(payload).__name__,)
    return (type(slot).__name__,)


def _verdict(health: object, fill: object, meaning: object, diamond: object) -> tuple:
    """"结论"是什么：三条链各自的形态 + 精确值 / 含义。"""
    return (
        _payload_key(health),
        _payload_key(fill),
        _payload_key(meaning),
        _payload_key(diamond),
    )


SHIFTS: tuple[int, ...] = (-24, -16, -12, -4, -1, 0, 1, 4, 12, 16, 24)


def _sensitivity(template: np.ndarray) -> dict[str, object]:
    """**锚点摆动铺到整批**：每一条各自沿 x / y 挪 ±δ，看三条读数的**结论**变不变。

    这是"宽容窗口怎么吸收偏移"那句话的读数：取区原点是记录推出来的，一定带像素级
    误差；挪到 COL_TOL / ROW_TOL 以内，结论应当一字不变；挪到超过它，本来就该变
    （那时是"整格错位"，窗口已经开在别的干员上了）。

    **不看一整批只看一条是自欺**：同一条规则在不同条上撞到的东西不一样（旁边那个
    干员的条、条右边的亮背景、**压在条上的别的条与黄色菱形**），所以 16 条各挪一遍，逐条报。
    """
    harvest = json.loads((BATCH / "取材.json").read_text(encoding="utf-8"))["crops"]
    windows = {item["file"]: item["window"] for item in harvest}
    batch = load_batch(BATCH)
    crops: list[dict[str, object]] = []
    for case, raw in replay_frames(batch):
        frame = frame_from_raw(raw)
        entry = windows[case.name + ".png"]
        base = PixelRect(*entry)
        base_health = field.read_health(frame, base)
        base_fill, base_meaning = field.read_skill(frame, base)
        base_diamond = field.find_ready_diamond(frame, base, template)
        baseline = _verdict(base_health, base_fill, base_meaning, base_diamond)
        base_number = _number(base_health)
        rows: list[dict[str, object]] = []
        for axis in ("x", "y"):
            for delta in SHIFTS:
                shifted = PixelRect(
                    base.x + (delta if axis == "x" else 0),
                    base.y + (delta if axis == "y" else 0),
                    base.width,
                    base.height,
                )
                if shifted.x < 0 or shifted.y < 0:
                    continue
                if shifted.x + shifted.width > CANVAS_2800x1260.width:
                    continue
                if shifted.y + shifted.height > CANVAS_2800x1260.height:
                    continue
                health = field.read_health(frame, shifted)
                fill, meaning = field.read_skill(frame, shifted)
                diamond = field.find_ready_diamond(frame, shifted, template)
                number = _number(health)
                delta_number = (
                    None if number is None or base_number is None else abs(number - base_number)
                )
                same = _verdict(health, fill, meaning, diamond) == baseline and (
                    delta_number is None or delta_number <= 2.0 / field.BAR_WIDTH
                )
                rows.append(
                    {
                        "axis": axis,
                        "shift": delta,
                        "same": same,
                        "health_delta": (
                            None
                            if number is None or base_number is None
                            else round(number - base_number, 6)
                        ),
                        "health_form": _form(health),
                        "skill_form": _form(fill),
                        "meaning": meaning.payload.value
                        if isinstance(meaning, Valued) and isinstance(meaning.payload, Exact)
                        else _form(meaning),
                        "diamond": diamond.payload.value
                        if isinstance(diamond, Valued) and isinstance(diamond.payload, Exact)
                        else _form(diamond),
                    }
                )
        tolerance: list[dict[str, object]] = []
        for axis in ("x", "y"):
            limit = field.COL_TOL if axis == "x" else field.ROW_TOL
            inside = [r for r in rows if r["axis"] == axis and abs(int(r["shift"])) <= limit]
            broken = sorted({abs(int(r["shift"])) for r in inside if not r["same"]})
            tolerance.append(
                {
                    "axis": axis,
                    "tolerance": limit,
                    "deltas_within_tolerance": sorted({abs(int(r["shift"])) for r in inside}),
                    "all_same_within_tolerance": not broken,
                    "broken_at": broken,
                }
            )
        crops.append(
            {
                "name": case.name + ".png",
                "baseline": {
                    "health": repr(base_health),
                    "skill": repr((base_fill, base_meaning)),
                    "diamond": repr(base_diamond),
                },
                "baseline_forms": {
                    "health": _form(base_health),
                    "skill": _form(base_fill),
                    "meaning": _label(base_meaning),
                    "diamond": _label(base_diamond),
                },
                "tolerance": tolerance,
                "rows": rows,
            }
        )
    summary: dict[str, object] = {"crops": len(crops)}
    within: dict[str, object] = {}
    first_break: dict[str, object] = {}
    for axis in ("x", "y"):
        limit = field.COL_TOL if axis == "x" else field.ROW_TOL
        ok = sum(
            1
            for crop in crops
            if next(t for t in crop["tolerance"] if t["axis"] == axis)["all_same_within_tolerance"]
        )
        within[axis] = {"tolerance": limit, "ok": ok, "total": len(crops)}
        beyond = sorted(
            {
                abs(int(row["shift"]))
                for crop in crops
                for row in crop["rows"]
                if row["axis"] == axis and abs(int(row["shift"])) > limit and not row["same"]
            }
        )
        first_break[axis] = beyond[0] if beyond else None
    summary["within_tolerance"] = within
    summary["first_break_beyond_tolerance"] = first_break
    return {
        "what": (
            "取区整体沿 x 或 y 挪 ±δ 像素，看三条读数的**结论**变不变。判据："
            "三条链的形态 / 含义 / 菱形那个布尔一致，**且**血条那个比例挪动不超过 "
            "2/BAR_WIDTH（两步）—— 读数的行带压到条边上一行时，抗锯齿会让末位差一步"
        ),
        "shifts": list(SHIFTS),
        "crops": crops,
        "summary": summary,
    }

def _coverage(sensitivity: dict[str, object]) -> dict[str, object]:
    """覆盖到什么 / 没覆盖到什么 —— **不许拿别的东西凑**。"""
    within = sensitivity["summary"]["within_tolerance"]
    breaks = sensitivity["summary"]["first_break_beyond_tolerance"]
    broken = [
        {
            "name": crop["name"],
            "axis": row["axis"],
            "shift": row["shift"],
            "changed": [
                key + ":" + str(crop["baseline_forms"][key]) + "→" + str(row[field_key])
                for key, field_key in (
                    ("health", "health_form"),
                    ("skill", "skill_form"),
                    ("meaning", "meaning"),
                    ("diamond", "diamond"),
                )
                if str(crop["baseline_forms"][key]) != str(row[field_key])
            ]
            or ["比例的末位（形态没变）"],
        }
        for crop in sensitivity["crops"]
        for row in crop["rows"]
        if not row["same"] and abs(int(row["shift"])) <= (field.COL_TOL if row["axis"] == "x" else field.ROW_TOL)
    ]
    broken.sort(key=lambda b: (b["name"], b["axis"], abs(b["shift"]), b["shift"]))
    by_axis: dict[str, list[str]] = {"x": [], "y": []}
    for row in broken:
        by_axis[row["axis"]].append(
            row["name"] + "(" + ("%+d" % row["shift"]) + " " + "/".join(row["changed"]) + ")"
        )
    swing = {
        "count": len(broken),
        "by_axis": {axis: ", ".join(items) for axis, items in by_axis.items()},
        "rows": broken,
        "why": (
            "翻面**不等于判据坏了**：窄窗里每一条各自撞到的东西不一样（旁边那个干员的条、"
            "条右边那截亮背景、**压在条上的别的条与黄色菱形**）。逐条列出来，是为了让「宽容窗口到底有多宽」"
            "这句话有读数支撑 —— 不是拿一条帧上的一句总结代替整批。"
        ),
    }
    broken_line = "；".join(
        part
        for part in (
            ("x: " + swing["by_axis"]["x"]) if swing["by_axis"]["x"] else "",
            ("y: " + swing["by_axis"]["y"]) if swing["by_axis"]["y"] else "",
        )
        if part
    )
    if not broken_line:
        broken_line = "（一条都没翻）"
    return {
        "covered": [
            "血条：满（多帧）、0.740、0.781、0.829、0.884、0.993、0.288、0.301 —— 八档不同的填充",
            "血条降级：精确（15 条）· **下界**（shot04_a：填着的那一截被别的条压过）",
            "技力条：充能中（绿）、生效中（橙实心）、弹药分格（橙带竖线）三档都有",
            "技力条填充：满（12 条）· 0.863（shot21_a）· 0.808（shot01_a）· 0.527（shot31_a）"
            "· **下界 0.808**（shot26_a：弹药没填满）",
            "技力含义：充能中 12 条 / 生效中 1 条 / 弹药分格 3 条",
            "技力那个字段：生效中与弹药分格那 4 条**落未采样**",
            "菱形：在 8 条（分数 0.543–0.895）、不在 8 条（分数 0.129–0.279）；空档 0.279–0.543",
            "条被挡住的那一档：shot04_a（条上压着**别的条** → 缝 8 列以内，**跨过去接着算**，"
            "仍读满）、shot30/shot22（条右侧背景比条还亮）",
            "**锚点摆动铺到整批**："
            + str(sensitivity["crops"].__len__())
            + " 条**每一条各自**沿 x / y 挪 ±1 / ±4 / ±12 px：结论一字不变的有 "
            + str(within["x"]["ok"])
            + "/"
            + str(within["x"]["total"])
            + "（x）与 "
            + str(within["y"]["ok"])
            + "/"
            + str(within["y"]["total"])
            + "（y）；再往外挪才翻面，x 轴上最早翻面的是 ±"
            + (str(breaks["x"]) if breaks["x"] is not None else "（没翻）")
            + "、y 轴上是 ±"
            + (str(breaks["y"]) if breaks["y"] is not None else "（没翻）")
            + " px",
            "**±12 以内不是一条都不翻**："
            + str(swing["count"])
            + " 条在某一轴上翻了 —— "
            + broken_line
            + "。逐条明细见 sensitivity.swing_within_tolerance",
        ],
        "swing_within_tolerance": swing,
        "not_covered": [
            "**极低填充**：低于约 3% 的血条一帧都没有 —— MIN_FILL_COLUMNS 那一档只有合成用例",
            "**边界渐变 -> 区间**：血条两侧都没有「填着的那一截右边半亮」的真帧，两支区间只有合成用例",
            "**技力含义的「认不出」那一档**：**这三批**（16 + 13 + 5 = 34 帧）里一帧都没有 ——"
            " 真帧在**难例批**，落法与逐块读数见 `tests/帧样本/场上干员难例/判据读数.json` 的"
            " `coverage`（**覆盖文本的正文只在那一处**）；不许拿「取区里不止一根条」那种帧去凑",
            "**sp_of 的「下界 → 加宽成区间」那一支**：真帧上**一帧都不触发** —— 唯一的"
            "**下界**读数（shot26_a）含义是**弹药分格**，按字段规则直接落**未采样**，轮不到加宽；"
            "这一支同样只剩合成用例",
            "**技力条填充的区间那一支**：真帧上一帧都没有 —— 「半填」那条带子只有 SKILL_FILL_TOL（48）"
            "到 _SKILL_HALF_LIT_CEILING（56）那么窄，而实测的边界是**一步跳过去**的"
            "（填着那一侧最多 40，紧接着那一列就跳到 68 / 87 / 88 / 93 / 94 / 106 / 149）；"
            "这一支靠合成用例钉（取舍记在 field.py 那条常量里）",
            "**取区里不止一根条**（格里有好几根条、或者条只截了一部分）：这一批里**一帧都没有** —— "
            "这一档的取区本来就该由调用方给对；给错了会读出一个**没有意义**的比例，"
            "按用户对「很难判准」那一档的要求应当报**未采样**",
            "**空轨道单独可见**（血条接近空、能看清暗轨道）：**这三批**里一帧都没有 ——"
            "真帧在**难例批**；**「条在、只是空了」与「条不在」是分得开的**（空的条留下一条"
            "条形变暗区 → 条内比条外暗，判据 `field._looks_like_an_empty_track`）—— 逐块落法见"
            "难例批 `判据读数.json` 的 `coverage`（那里是正文）。极低填充仍然落未采样，但那是"
            "**另一条线**（填充连段短于 `MIN_FILL_COLUMNS`），不是「分不开」",
            "**技力条掉到零**（一条全暗的技力条）：一帧都没有 —— 难例批那一块是「空轨道被背景"
            "染了色」，不是「一条全暗的条」（见难例批 `判据读数.json` 的 `coverage`，那里是正文）",
            "**暂停态下的菱形**：这一批全是常速帧，菱形在暂停态下自己在动这件事没测",
            "**永续技能**：读得出「生效中」，而「没有还剩多久」这件事由数据库那一侧决定，"
            "这一批没有永续技能的帧",
        ],
        "stuck": (
            "「极低填充」与「血条被切断」两档在现有原生帧里一帧都找不到 —— 按任务单的"
            "【卡住】那一条报出来，**没有拿别的东西凑**；裁定下来了：**不必再去找**，"
            "这两档**认合成用例**（tests/unit/test_vision_field.py 里那两支合成用例）。"
        ),
    }


def run() -> int:
    manifest = build_batch()
    template_info = build_template()
    template = field.load_ready_diamond()
    batch = load_batch(BATCH)
    harvest = {
        item["file"]: item
        for item in json.loads((BATCH / "取材.json").read_text(encoding="utf-8"))["crops"]
    }

    rows: list[dict[str, object]] = []
    health_values: list[float] = []
    skill_values: list[float] = []
    scores: list[float] = []
    present: list[float] = []
    absent: list[float] = []
    for case, raw in replay_frames(batch):
        frame = frame_from_raw(raw)
        entry = harvest[case.name + ".png"]
        window = PixelRect(*entry["window"])
        health = field.read_health(frame, window)
        fill, meaning = field.read_skill(frame, window)
        diamond = field.find_ready_diamond(frame, window, template)
        score, dx, dy = field.diamond_score(frame, window, template)
        health_value = _number(health)
        fill_value = _number(fill)
        if health_value is not None:
            health_values.append(health_value)
        if fill_value is not None:
            skill_values.append(fill_value)
        scores.append(float(score))
        if isinstance(diamond, Valued) and isinstance(diamond.payload, Exact):
            (present if diamond.payload.value is True else absent).append(float(score))
        rows.append(
            {
                "name": case.name,
                "covers": entry["covers"],
                "window": entry["window"],
                "health": _slot_json(health),
                "skill_fill": _slot_json(fill),
                "skill_meaning": _slot_json(meaning),
                "sp": _slot_json(field.sp_of(fill, meaning)),
                "diamond": _slot_json(diamond),
                "diamond_score": round(score, 6),
                "diamond_at": [dx, dy],
            }
        )
        print(
            f"{case.name}  血条 {health!r}  技力 {fill!r} / {meaning!r}  菱形 {diamond!r} ({score:.4f})"
        )

    # 锚点摆动：**整批 16 条各自**挪 ±δ，看读数会不会变。
    sensitivity = _sensitivity(template)

    report = {
        "note": (
            "场上我方干员那三条读数链的判据读数。它**不判断够不够** —— 那是规划者的事；"
            "这里只放两簇的分布、空档、逐帧读数与覆盖说明。"
        ),
        "batch": "tests/帧样本/场上干员",
        "version": batch.version,
        "canvas": [batch.canvas.width, batch.canvas.height],
        "device": batch.device,
        "ruler": {
            "health": "逐列中位颜色离 HP_BLUE (110,190,228) 的逐通道最大差 ≤ HP_COLOUR_TOL=24；填着的那一截 = 自左缘起的连续亮列（容 GAP_TOLERANCE=4 列）",
            "skill": "逐列中位颜色离**这一条自己的填充色**（自左缘起跳过 2 列、取 8 列的中位颜色）的逐通道最大差 ≤ SKILL_FILL_TOL=48；填着的那一截 = 自左缘起的连续同色列（容 GAP_TOLERANCE=4 列）；被压住那一档另有缝宽容度 COVER_GAP_TOLERANCE=10（≤ 它 → 跨过去接着算；> 它 → 下界）；含义按填着那一截的逐列中位 (R−G)：≤ 0 绿 / ≥ 40 橙 / 正的那一段认不出",
            "diamond": "cv2.matchTemplate(TM_CCOEFF_NORMED, RGB 三通道)，模板 64×64，在取区里 ±60 px 的邻域上滑",
        },
        "window": {
            "size": [field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT],
            "pixels": field.OPERATOR_UI_WIDTH * field.OPERATOR_UI_HEIGHT,
            "bars_in_window": list(field.BARS_IN_WINDOW),
            "diamond_in_window": list(field.DIAMOND_IN_WINDOW),
            "bar_pixels": field.BAR_WIDTH * field.BAR_HEIGHT,
        },
        "window_from": "取区左上角 = 血条左上角 − BARS_IN_WINDOW；血条左上角由部署记录 + 投影给出（本层不产生它）",
        "dedup": {
            "key": "帧名（批内唯一）",
            "why": "同一个源帧里两根不同的条各是一条（例如 cd01 出了三条），所以去重按「哪一根条」而不是「哪一帧」",
        },
        "chains": {
            "health": {
                "reads": [{"name": row["name"], "slot": row["health"], "covers": row["covers"]} for row in rows],
                "distribution": _spread(health_values),
                "gap": _gap(health_values),
                "unsampled": [row["name"] for row in rows if row["health"]["state"] == "unsampled"],
                "note": "血量那个字段的形态是 精确/区间/下界/未采样 —— 这一层不产 Unknown。",
            },
            "skill": {
                "reads": [
                    {
                        "name": row["name"],
                        "fill": row["skill_fill"],
                        "meaning": row["skill_meaning"],
                        "sp": row["sp"],
                        "covers": row["covers"],
                    }
                    for row in rows
                ],
                "distribution": _spread(skill_values),
                "gap": _gap(skill_values),
                "meanings": _tally(rows, "skill_meaning"),
                "sp_forms": _tally(rows, "sp"),
                "note": (
                    "技力生效期间（含义 = 生效中 / 弹药分格）技力那个字段落未采样 —— 见 sp 那一列。"
                    "**读数本身是下界时**（弹药剩余那种），技力那个字段**加宽成区间**"
                    "（「下沿 … 1.0」）—— 下界不是那个字段允许的形态，不许原样透出去。上游的"
                    " skill_fill 仍然照实报下界：那是**读数**，不是字段。"
                ),
            },
            "diamond": {
                "reads": [
                    {
                        "name": row["name"],
                        "slot": row["diamond"],
                        "score": row["diamond_score"],
                        "at": row["diamond_at"],
                        "covers": row["covers"],
                    }
                    for row in rows
                ],
                "distribution": _spread(scores),
                "present": _spread(present),
                "absent": _spread(absent),
                "gap": {
                    "absent_max": round(max(absent), 6) if absent else None,
                    "present_min": round(min(present), 6) if present else None,
                },
                "lines": {
                    "solid": field.DIAMOND_SOLID_SCORE,
                    "absent": field.DIAMOND_ABSENT_SCORE,
                },
                "note": "两条线之间那一段落未采样（认不出），它绝不落 Exact(False) —— 那是漏检方向。",
            },
        },
        "template": template_info,
        "sensitivity": sensitivity,
        "coverage": _coverage(sensitivity),
    }
    _write_json(REPORT, report)
    print(f"\n报告落在 {REPORT.relative_to(REPO)}")
    print(f"批：{len(rows)} 条；血条 {_spread(health_values)}；菱形空档 {report['chains']['diamond']['gap']}")
    print(f"锚点摆动：{sensitivity['summary']['within_tolerance']}；再往外最早翻面 {sensitivity['summary']['first_break_beyond_tolerance']}")
    return 0

# ---------------------------------------------------------------------------
# 面板真值批
# ---------------------------------------------------------------------------


def _runs(row_mask: np.ndarray, minlen: int) -> list[tuple[int, int]]:
    """一行里所有长度 ≥ minlen 的连续亮段 → [(起点, 长度)]。"""
    index = np.flatnonzero(row_mask)
    if index.size == 0:
        return []
    out: list[tuple[int, int]] = []
    start = int(index[0])
    previous = int(index[0])
    for value in index[1:]:
        current = int(value)
        if current != previous + 1:
            if previous - start + 1 >= minlen:
                out.append((start, previous - start + 1))
            start = current
        previous = current
    if previous - start + 1 >= minlen:
        out.append((start, previous - start + 1))
    return out


def _bar_candidates(pixels: np.ndarray) -> list[tuple[int, int, int]]:
    """一帧上所有「血条蓝」的横条候选 → [(左缘 x, 上缘 y, 亮着多少列)]，按 y 再按 x 排。

    **它不是识别链的一部分**：配对（这一帧被点中的是哪个干员）是**建批**这一步的事，
    链只认取区。所以这一段逻辑住 scripts/ 里，不进 ark.engine.vision。
    """
    distance = np.abs(
        pixels.astype(np.int16) - np.asarray(field.HP_BLUE, dtype=np.int16)
    ).max(axis=2)
    mask = distance <= field.HP_COLOUR_TOL
    open_bands: dict[int, dict[str, int]] = {}
    closed: list[dict[str, int]] = []
    for row in range(mask.shape[0]):
        seen: set[int] = set()
        for start, length in _runs(mask[row], CANDIDATE_MIN_LIT):
            seen.add(start)
            band = open_bands.get(start)
            if band is None:
                open_bands[start] = {"x": start, "y0": row, "y1": row, "lit": length}
            else:
                band["y1"] = row
                band["lit"] = max(band["lit"], length)
        for start in [key for key in open_bands if key not in seen]:
            closed.append(open_bands.pop(start))
    closed.extend(open_bands.values())
    tall = [b for b in closed if b["y1"] - b["y0"] + 1 >= CANDIDATE_MIN_HEIGHT]
    tall.sort(key=lambda b: (b["y0"], b["x"]))
    return [(b["x"], b["y0"], b["lit"]) for b in tall]


def _distance_to_centre(bar: tuple[int, int, int]) -> float:
    """这根条的**中心**离画布中心多远（画布中心 = 配对规则那个圆心）。"""
    centre_x = bar[0] + field.BAR_WIDTH / 2.0
    centre_y = bar[1] + field.BAR_HEIGHT / 2.0
    return math.hypot(centre_x - CANVAS_CENTRE[0], centre_y - CANVAS_CENTRE[1])


def _pick_nearest_centre(
    candidates: list[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], tuple[int, int, int] | None, list[tuple[int, int, int]]]:
    """按「离画布中心最近」挑，并把第二名与**全部候选**一起给出来。"""
    if not candidates:
        raise ValueError("这一帧上一根「血条蓝」都没找到：配对规则无从落地")
    ordered = sorted(candidates, key=lambda b: (_distance_to_centre(b), -b[2], b[1], b[0]))
    return ordered[0], (ordered[1] if len(ordered) > 1 else None), ordered


def _truth_json(life_now: int, life_max: int, sp_now: int, sp_max: int) -> dict[str, object]:
    """面板上读到的那两个数 → JSON（比例在这里算一次，round 到 6 位）。"""
    return {
        "life": {"current": life_now, "max": life_max, "ratio": round(life_now / life_max, 6)},
        "sp": None
        if sp_now < 0
        else {"current": sp_now, "max": sp_max, "ratio": round(sp_now / sp_max, 6)},
    }


def build_truth_batch() -> dict[str, object]:
    """建真值那一批：落 PNG + index.json + 取材.json，返回清单。"""
    TRUTH_BATCH.mkdir(parents=True, exist_ok=True)
    crops: list[dict[str, object]] = []
    harvest: list[dict[str, object]] = []
    for name, relative, bar_x, bar_y, life_now, life_max, sp_now, sp_max, note in TRUTH_PICKS:
        source = _read_source(relative)
        picked, second, ordered = _pick_nearest_centre(_bar_candidates(source))
        if (picked[0], picked[1]) != (bar_x, bar_y):
            raise ValueError(
                f"{name}：按「离画布中心最近」挑中的是 {picked[:2]}，而表里钉的是 "
                f"{(bar_x, bar_y)} —— 配对规则与这一批对不上了，不许悄悄换一根条"
            )
        origin_x = bar_x - CROP_LEFT
        origin_y = max(0, bar_y - 340)
        block = source[origin_y : origin_y + CROP_HEIGHT, origin_x : origin_x + CROP_WIDTH]
        if block.shape[:2] != (CROP_HEIGHT, CROP_WIDTH):
            raise ValueError(f"{name} 裁出来是 {block.shape[:2]}，落不进画布")
        file_name = f"{name}.png"
        Image.fromarray(block).save(TRUTH_BATCH / file_name)
        crops.append(
            {
                "file": file_name,
                "origin": [origin_x, origin_y],
                "size": [CROP_WIDTH, CROP_HEIGHT],
                "source": relative,
                "source_sha256": _sha256(REPO / relative),
            }
        )
        window = field.operator_ui_window(PixelRect(bar_x, bar_y, field.BAR_WIDTH, 16))
        harvest.append(
            {
                "file": file_name,
                "source": relative,
                "bar": [bar_x, bar_y],
                "distance_to_centre": round(_distance_to_centre(picked), 3),
                "runner_up": None
                if second is None
                else {
                    "bar": [second[0], second[1]],
                    "distance_to_centre": round(_distance_to_centre(second), 3),
                },
                "candidates": [
                    {
                        "bar": [b[0], b[1]],
                        "lit": b[2],
                        "distance_to_centre": round(_distance_to_centre(b), 3),
                    }
                    for b in ordered
                ],
                "window": [
                    int(window.x),
                    int(window.y),
                    int(window.width),
                    int(window.height),
                ],
                "window_from": (
                    "血条左上角 ("
                    + str(bar_x)
                    + ", "
                    + str(bar_y)
                    + ") − BARS_IN_WINDOW → operator_ui_window()"
                ),
                "truth": _truth_json(life_now, life_max, sp_now, sp_max),
                "truth_row": list(TRUTH_ROW_BOX[name]),
                "note": note,
            }
        )
    manifest = {
        "version": 1,
        "canvas": [CANVAS_2800x1260.width, CANVAS_2800x1260.height],
        "device": DEVICE,
        "note": (
            "场上我方干员那条读数链的**面板真值批**：每一帧上详情面板是开着的，所以一帧里"
            "同时有面板那一行的 生命 当前/上限 与 被读的那根条。每一条是原生帧上 300×420 的"
            "一块，比取区（240×360）大一圈。它**不是**第一批，第一批已经拿它调过参。"
        ),
        "crops": crops,
    }
    _write_json(TRUTH_BATCH / "index.json", manifest)
    _write_json(
        TRUTH_BATCH / "取材.json",
        {
            "note": (
                "**这一批的真值来自面板**（生命 当前/上限；技力那一侧是场上技能按钮上的 N/M）。"
                "真值只在对照那一步用，链的参数一个都不是从这一批来的。"
            ),
            "window_size": [field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT],
            "bars_in_window": list(field.BARS_IN_WINDOW),
            "crop_size": [CROP_WIDTH, CROP_HEIGHT],
            "crop_left_margin": CROP_LEFT,
            "pairing": {
                "rule": (
                    "点场上干员开面板时，被点中的干员会被挪到屏幕较中心的位置"
                    "（场景跟着一起挪）→ 被选中的那根条 = 离画布中心最近的那根"
                ),
                "centre": [CANVAS_CENTRE[0], CANVAS_CENTRE[1]],
                "min_lit_columns": CANDIDATE_MIN_LIT,
                "min_height": CANDIDATE_MIN_HEIGHT,
            },
            "crops": harvest,
        },
    )
    return manifest


def build_regression_batch() -> dict[str, object]:
    """建回归那一批：落 PNG + index.json + 取材.json。**里面一个真值字段都没有。**

    建的时候**当场自检**：按表里那个血条左上角算出来的取区上，必须真的能找到一根条，
    而且它的左缘必须正好落在取区里的 `BARS_IN_WINDOW` 那个位置。取区算错了要在这里
    响亮地炸出来，不许等到"读数稳定"那句话把它盖过去。
    """
    REGRESSION_BATCH.mkdir(parents=True, exist_ok=True)
    crops: list[dict[str, object]] = []
    harvest: list[dict[str, object]] = []
    for name, relative, bar_x, bar_y, note in REGRESSION_PICKS:
        source = _read_source(relative)
        origin_x = bar_x - CROP_LEFT
        origin_y = max(0, bar_y - 340)
        block = source[origin_y : origin_y + CROP_HEIGHT, origin_x : origin_x + CROP_WIDTH]
        if block.shape[:2] != (CROP_HEIGHT, CROP_WIDTH):
            raise ValueError(f"{name} 裁出来是 {block.shape[:2]}，落不进画布")
        window = field.operator_ui_window(PixelRect(bar_x, bar_y, field.BAR_WIDTH, 16))
        region = block[
            int(window.y) - origin_y : int(window.y) - origin_y + field.OPERATOR_UI_HEIGHT,
            int(window.x) - origin_x : int(window.x) - origin_x + field.OPERATOR_UI_WIDTH,
        ]
        found = field._locate_health(region)
        if found is None:
            raise ValueError(f"{name}：按表里那个血条左上角开的取区上**找不到血条** —— 取区算错了")
        row, start, profile = found
        first = field._left_edge(field._blue_distance(profile) <= field.HP_COLOUR_TOL, start)
        if (start + first, row) != field.BARS_IN_WINDOW:
            raise ValueError(
                f"{name}：条找到的是取区里的 ({start + first}, {row})，而表里钉的是 "
                f"{field.BARS_IN_WINDOW}（x, y）—— 取区与血条左上角对不上"
            )
        file_name = f"{name}.png"
        Image.fromarray(block).save(REGRESSION_BATCH / file_name)
        crops.append(
            {
                "file": file_name,
                "origin": [origin_x, origin_y],
                "size": [CROP_WIDTH, CROP_HEIGHT],
                "source": relative,
                "source_sha256": _sha256(REPO / relative),
            }
        )
        harvest.append(
            {
                "file": file_name,
                "source": relative,
                "bar": [bar_x, bar_y],
                "window": [
                    int(window.x),
                    int(window.y),
                    int(window.width),
                    int(window.height),
                ],
                "window_from": (
                    "血条左上角 ("
                    + str(bar_x)
                    + ", "
                    + str(bar_y)
                    + ") − BARS_IN_WINDOW → operator_ui_window()"
                ),
                "note": note,
            }
        )
    manifest = {
        "version": 1,
        "canvas": [CANVAS_2800x1260.width, CANVAS_2800x1260.height],
        "device": DEVICE,
        "note": (
            "技力那一侧的**回归批**：五帧都是亮底（橙条纹地板 / 青地板 / 花丛）。"
            "它**不是**独立真值批 —— 判据是拿这一批的现象改的，所以它见过；"
            "它的用处是「以后技力那一侧的读数不许再变」。"
            "**这一批没有任何真值**（真值住规划者那边）。"
        ),
        "crops": crops,
    }
    _write_json(REGRESSION_BATCH / "index.json", manifest)
    _write_json(
        REGRESSION_BATCH / "取材.json",
        {
            "note": (
                "**这一批没有真值**：只有取区、来源与「这一帧的底是什么」。"
                "读数钉在 tests/unit/test_vision_field.py 的回归那一段里。"
            ),
            "window_size": [field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT],
            "bars_in_window": list(field.BARS_IN_WINDOW),
            "crop_size": [CROP_WIDTH, CROP_HEIGHT],
            "crop_left_margin": CROP_LEFT,
            "crops": harvest,
        },
    )
    return manifest


def run_regression() -> int:
    """建回归那一批，并把五帧的读数打一遍给人看。**不落读数文件。**"""
    manifest = build_regression_batch()
    harvest = json.loads((REGRESSION_BATCH / "取材.json").read_text(encoding="utf-8"))["crops"]
    windows = {item["file"]: item["window"] for item in harvest}
    for case, raw in replay_frames(load_batch(REGRESSION_BATCH)):
        frame = frame_from_raw(raw)
        window = PixelRect(*windows[case.name + ".png"])
        fill, meaning = field.read_skill(frame, window)
        print(f"{case.name}  技力 {fill!r} / {meaning!r}")
    print(
        f"$BSn回归批落在 {REGRESSION_BATCH.relative_to(REPO)}（{len(manifest['crops'])} 块）"
        " —— 它没有判据读数文件，读数钉在单元测试里"
    )
    return 0


def _truth_rows(template: np.ndarray) -> list[dict[str, object]]:
    """把真值那一批经离线那条路跑一遍 —— **这一圈里没有真值**。"""
    harvest = json.loads((TRUTH_BATCH / "取材.json").read_text(encoding="utf-8"))["crops"]
    windows = {item["file"]: item["window"] for item in harvest}
    batch = load_batch(TRUTH_BATCH)
    rows: list[dict[str, object]] = []
    for case, raw in replay_frames(batch):
        frame = frame_from_raw(raw)
        window = PixelRect(*windows[case.name + ".png"])
        health = field.read_health(frame, window)
        fill, meaning = field.read_skill(frame, window)
        diamond = field.find_ready_diamond(frame, window, template)
        score, dx, dy = field.diamond_score(frame, window, template)
        rows.append(
            {
                "name": case.name,
                "window": windows[case.name + ".png"],
                "health": _slot_json(health),
                "health_value": _number(health),
                "skill_fill": _slot_json(fill),
                "skill_fill_value": _number(fill),
                "skill_meaning": _slot_json(meaning),
                "sp": _slot_json(field.sp_of(fill, meaning)),
                "diamond": _slot_json(diamond),
                "diamond_score": round(score, 6),
                "diamond_at": [dx, dy],
            }
        )
    return rows


def _error_stats(values: list[float]) -> dict[str, object]:
    """误差分布：**中位、两端、单边还是双边**。

    正 = 链读高了（比面板多），负 = 链读低了。
    """
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    positive = [v for v in ordered if v > 0]
    negative = [v for v in ordered if v < 0]
    if not negative:
        sided = "单边（全部 ≥ 0：链读高了）"
    elif not positive:
        sided = "单边（全部 ≤ 0：链读低了）"
    else:
        sided = "双边（有高有低）"
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "max": round(ordered[-1], 6),
        "median": round(ordered[len(ordered) // 2], 6),
        "abs_max": round(max(abs(v) for v in ordered), 6),
        "sided": sided,
        "in_pixels": {
            "min": round(ordered[0] * field.BAR_WIDTH, 3),
            "max": round(ordered[-1] * field.BAR_WIDTH, 3),
            "median": round(ordered[len(ordered) // 2] * field.BAR_WIDTH, 3),
            "abs_max": round(max(abs(v) for v in ordered) * field.BAR_WIDTH, 3),
        },
    }


def _truth_coverage() -> dict[str, object]:
    """真值那一批覆盖到什么 / 没覆盖到什么。"""
    return {
        "truth_available": {
            "life": "全部 13 帧都有（面板那一行的 生命 当前/上限）",
            "sp": (
                "只有 sp_now 一帧有（场上那个技能按钮上的 16/26）—— 另外 12 帧上那个按钮"
                "根本没画出来，所以技力那一侧的误差样本量是 1"
            ),
        },
        "covered": [
            "血条八档：1.000（p02）、0.990、0.980、0.971、0.960、0.820、0.810、0.183",
            "**同一场内的连续帧**：p00–p07 一场、p08–p11 与 sp_now 另一场 —— 误差可以看趋势，不是一个孤立点",
            "**另一个场景**：p08–p11 与 sp_now 那一场是青地板，与 p00–p07 那一场不同",
        ],
        "not_covered": [
            "**只有一种真值来源**（面板那一行 + sp_now 那个按钮），没有第二种独立来源去交叉核",
            "**技力那一侧只有 1 帧**：12 帧上没有可读的 N/M",
            "**没有满/空的极值对**：最低那一档是 0.183，没有接近 0 的真值",
            "**没有暂停态 / 没有退场动画**：这一批全是常速帧",
        ],
        "pairing_caveat": (
            "**这一批 13 帧全是「点场上干员开面板」那一档。** 原来还有一帧，它是"
            "**备战席干员的详情页**（点的是备战席卡片，不是场上干员），按裁定**整个拿掉**了 ——"
            "这一帧**不属于本批的取帧场景**：配对规则（被选中的那根条 = 离画布中心最近的那根）"
            "**管不到它**，点备战席时整个场景朝固定的右上方挪、与被点的卡在哪无关。"
        ),
    }


def _truth_observed(frames: list[dict[str, object]]) -> dict[str, object]:
    """这一批**看出来的东西** —— 只报，**不改链**。

    拿这一批的数字回头改 field.py 的常量，这一批就不再是独立判据了；要改是另一次改动，
    改完这一批必须重采（[基准集] §4）。
    """
    sp_frame = next((f for f in frames if f["sp_error"] is not None), None)
    return {
        "health_unsampled": [
            f["name"] for f in frames if f["health"]["state"] == "unsampled"
        ],
        "sp_frame": None
        if sp_frame is None
        else {
            "name": sp_frame["name"],
            "truth": sp_frame["truth"]["sp"],
            "skill_fill": sp_frame["skill_fill"],
            "skill_meaning": sp_frame["skill_meaning"],
            "error": sp_frame["sp_error"],
            "error_px": round(float(sp_frame["sp_error"]) * field.BAR_WIDTH, 3),
        },
        "what_i_did_not_do": (
            "这一批的 13 帧**一帧都没有参与定参** —— 面板上的数字只进对照那一步，"
            "链的参数一个都不是从这一批来的。"
            "**要说清楚的一件事**：这一版链在**这一批之外**改过（技力那一侧的填充判据与 "
            "GREEN_MAX_RG 是拿 data/probe/sp_series 那五帧定的，见 field.SKILL_FILL_TOL），"
            "改完这一批的读数**重跑了一遍** —— 所以它对**下一次**改动才重新算「没见过」。"
            "下一轮再改链必须换一批（[基准集] §4）。"
        ),
    }


def run_truth() -> int:
    manifest = build_truth_batch()
    template = field.load_ready_diamond()
    harvest = json.loads((TRUTH_BATCH / "取材.json").read_text(encoding="utf-8"))["crops"]
    rows = _truth_rows(template)

    # **从这里往下才碰真值。**
    frames: list[dict[str, object]] = []
    health_errors: list[float] = []
    sp_errors: list[float] = []
    for row, item in zip(rows, harvest):
        if item["file"][:-4] != row["name"]:
            raise ValueError(f"顺序对不上：{item['file']} vs {row['name']}")
        life = item["truth"]["life"]
        sp = item["truth"]["sp"]
        health_error = (
            None if row["health_value"] is None else round(row["health_value"] - life["ratio"], 6)
        )
        sp_error = (
            None
            if sp is None or row["skill_fill_value"] is None
            else round(row["skill_fill_value"] - sp["ratio"], 6)
        )
        if health_error is not None:
            health_errors.append(health_error)
        if sp_error is not None:
            sp_errors.append(sp_error)
        frames.append(
            {
                "name": row["name"],
                "source": item["source"],
                "truth": item["truth"],
                "truth_row": item["truth_row"],
                "bar": item["bar"],
                "distance_to_centre": item["distance_to_centre"],
                "runner_up": item["runner_up"],
                "note": item["note"],
                "window": row["window"],
                "health": row["health"],
                "health_error": health_error,
                "health_error_px": (
                    None if health_error is None else round(health_error * field.BAR_WIDTH, 3)
                ),
                "skill_fill": row["skill_fill"],
                "skill_meaning": row["skill_meaning"],
                "sp": row["sp"],
                "sp_error": sp_error,
                "diamond": row["diamond"],
                "diamond_score": row["diamond_score"],
                "diamond_at": row["diamond_at"],
            }
        )
        print(
            f"{row['name']}  真值 {life['current']}/{life['max']}={life['ratio']:.6f}"
            f"  链 {row['health']!r}  误差 {health_error}  配对距离 {item['distance_to_centre']}"
        )

    report = {
        "note": (
            "面板真值批的判据读数：**链读出来的比例 vs 面板那一行的真值**，逐帧一个误差。"
            "它**不判断够不够** —— 这里只放误差分布、逐帧对照与覆盖说明。"
        ),
        "hard_rule": (
            "面板上的数字**只进对照这一步**：链先把 13 帧读完，之后才做减法。链的参数一个都"
            "不是从这一批来的 —— 拿它回头改 field.py 的常量，这一批就不再是独立判据。"
        ),
        "batch": "tests/帧样本/场上干员真值",
        "version": manifest["version"],
        "canvas": [CANVAS_2800x1260.width, CANVAS_2800x1260.height],
        "device": DEVICE,
        "how_truth": (
            "面板上「生命 当前/上限」那一行逐帧看图读出；技力那一侧是场上那个技能按钮上的"
            " N/M。每一帧**从哪一块读的**记在 truth_row 里，谁都能重读一遍。"
        ),
        "ruler": {
            "health": "逐列中位颜色离 HP_BLUE (110,190,228) 的逐通道最大差 ≤ HP_COLOUR_TOL=24；填着的那一截 = 自左缘起的连续亮列（容 GAP_TOLERANCE=4 列）",
            "skill": "逐列中位颜色离**这一条自己的填充色**（自左缘起跳过 2 列、取 8 列的中位颜色）的逐通道最大差 ≤ SKILL_FILL_TOL=48；填着的那一截 = 自左缘起的连续同色列（容 GAP_TOLERANCE=4 列）；被压住那一档另有缝宽容度 COVER_GAP_TOLERANCE=10（≤ 它 → 跨过去接着算；> 它 → 下界）；含义按填着那一截的逐列中位 (R−G)：≤ 0 绿 / ≥ 40 橙 / 正的那一段认不出",
            "diamond": "cv2.matchTemplate(TM_CCOEFF_NORMED, RGB 三通道)，模板 64×64，在取区里 ±60 px 的邻域上滑",
        },
        "window": {
            "size": [field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT],
            "bars_in_window": list(field.BARS_IN_WINDOW),
            "diamond_in_window": list(field.DIAMOND_IN_WINDOW),
        },
        "pairing": {
            "rule": "被选中的那根条 = 离画布中心最近的那根（点场上干员开面板时，被点中的干员会被挪到屏幕较中心）",
            "centre": [CANVAS_CENTRE[0], CANVAS_CENTRE[1]],
            "evidence": "每一帧的 bar / distance_to_centre / runner_up / 全部候选都在 取材.json 里 —— 挑错了看得见",
        },
        "frames": frames,
        "health_errors": dict(
            _error_stats(health_errors),
            rule="误差 = 链读出来的比例 − 面板 当前/上限；正 = 读高了，负 = 读低了",
            median_note="偶数个时取上中位",
        ),
        "sp_errors": dict(
            _error_stats(sp_errors),
            rule="误差 = 链读出来的技力条填充比例 − 技能按钮上的 N/M",
            note="这一批里只有 sp_now 一帧有可读的 N/M",
        ),
        "observed": _truth_observed(frames),
        "coverage": _truth_coverage(),
    }
    _write_json(TRUTH_REPORT, report)
    print(f"$BSn真值批报告落在 {TRUTH_REPORT.relative_to(REPO)}")
    print(f"血条误差 {report['health_errors']}")
    print(f"技力误差 {report['sp_errors']}")
    print(f"看出来什么 {json.dumps(report['observed']['sp_frame'], ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    code = run()
    code |= run_truth()
    code |= run_regression()
    raise SystemExit(code)

