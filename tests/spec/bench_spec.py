"""基准集与重放适配器的规格符合性验收 —— **由规划者维护，不住在 tests/unit/**。

与 `tests/unit/` 的分工：

| 目录 | 谁写 | 验证什么 |
| --- | --- | --- |
| `tests/unit/` | 执行者 | "实现按我的设计工作" |
| **本文件** | **规划者** | **"规格说的那些话成立"** |

## 两条纪律

1. **只用公开面。** 只 import 各层**对外承诺**的那些名字。碰 `_` 开头的内部名，
   验收就绑死在实现上了。
2. **每条检查都带规格出处。** 没有出处的检查是**发明规范**；出处对不上的检查是
   **发现了规格洞**。两种都要能在报告里看出来。

## 它是规划者的工具，不进任务消息

D-全局-22：`tests/spec/` 那套检查**只归规划者用**。执行者该建什么，从**规格文档**
里读；把它当"活清单"交出去，等于给它一道有答案的题——它会朝"让这些检查变绿"
过拟合，而不是朝"把这件事做对"。

唯一已经绿的是 B11（"engine 不 import PIL"）：它守的是**已经成立**的事。

## 跑法（不需要 pytest）

    python tests/spec/bench_spec.py

pytest 外壳是 `test_bench_spec.py`，它只是把下面那张表逐条变成用例。

## 这一批管的是"离线这条路通不通"，不是"读数准不准"

读数准不准归 `cost_spec.py`（D 族 / B 族）。这里问的是**同一批帧走另一条路**
（基准集 → 加载器 → 重放适配器 → 识别链）**会不会得到同一个结果**——那是
[离线验收] §1 那句"把录下来的帧喂回线上识别链路"的落点。

## 真值从哪来

**不在这里重抄一份。** 31 帧的真值住 `cost_spec.py`（同目录），B10 直接引用它——
抄一份就有两个真值表，改一处忘一处是静默漂移。
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import itertools
import json
import os
import re
import shutil
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _entry in (str(REPO / "src"), str(_HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

BATCH = REPO / "tests" / "帧样本" / "费用"
GLYPH_DIR = REPO / "assets" / "features" / "glyphs"

CANVAS_W, CANVAS_H = 2800, 1260


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


def _batch_mod():
    return _mod("ark.offline.bench.batch")


def _feed_mod():
    return _mod("ark.offline.bench.feed")


def _replay_mod():
    return _mod("ark.engine.device.replay")


_TMP_SEQ = itertools.count()


@contextlib.contextmanager
def _writable_tmp():
    """建一个**能进得去**的临时目录。

    本机沙箱下 `tempfile.mkdtemp()` 建出来的目录**连建它的进程自己都进不去**：
    它用 `os.mkdir(path, 0o700)`，而受限令牌下那份 DACL 不继承父目录。
    这是**本机环境的事，不是被测实现的事** —— 所以在这里绕开，而不是让实现去迁就。
    """
    base = REPO / "data" / "_spec_tmp"
    base.mkdir(parents=True, exist_ok=True)
    dest = base / f"bench{os.getpid()}_{next(_TMP_SEQ)}"
    dest.mkdir()
    try:
        yield dest
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def _raw_blank(width: int = 8, height: int = 4, value: int = 7) -> bytes:
    """造一帧原始字节（设备层 §4 的布局）。

    只做头 + 常量像素：这一批验的是**重放这条路**，不是解帧（解帧归 cost_spec 的
    F 族）。
    """
    header = struct.pack("<4I", width, height, 1, 1)
    return header + bytes([value, value, value, 255]) * (width * height)


def _slot_brief(slot: object) -> str:
    """把槽写成人能读的一小串，用在失败证据里。"""
    from ark.contracts import Exact, Interval, Valued

    if isinstance(slot, Valued):
        payload = slot.payload
        if isinstance(payload, Exact):
            return f"Exact({payload.value})"
        if isinstance(payload, Interval):
            return f"Interval({payload.lo},{payload.hi})"
        return f"Valued({type(payload).__name__})"
    return type(slot).__name__


def _index() -> dict:
    return json.loads((BATCH / "index.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# B 族：基准集长什么样
# ---------------------------------------------------------------------------


def check_batch_loads_with_its_manifest() -> str:
    """`index.json` 里那几样必填字段，加载器一样不少地读出来。

    出处是[基准集] §2 的字段表：`version` · `canvas` · `device` · `crops` 必填，
    `facts` 可选。**`facts` 没写就是没有** —— 费用那一批没记前提头，加载器不许替它
    补一个默认值（那是编出来的环境事实）。
    """
    batch = _batch_mod().load_batch(BATCH)
    raw = _index()
    assert batch.version == raw["version"] >= 1, (
        f"版本号读成了 {batch.version!r}，清单里写的是 {raw['version']!r}"
        "——「基准是第几版」要靠它回答"
    )
    assert (batch.canvas.width, batch.canvas.height) == (CANVAS_W, CANVAS_H), (
        f"画布读成了 {batch.canvas}，清单里写的是 {raw['canvas']}"
    )
    assert isinstance(batch.device, str) and batch.device, (
        f"机型读成了 {batch.device!r}：它是出处，必填"
    )
    names = [case.name for case in batch.cases]
    assert len(names) == len(raw["crops"]), (
        f"清单里有 {len(raw['crops'])} 条裁剪，加载器给出来 {len(names)} 条"
    )
    assert len(set(names)) == len(names), (
        f"帧名有重复：{sorted({n for n in names if names.count(n) > 1})}"
        "——帧名是批内唯一的标识，重复了报告就对不上帧"
    )
    assert batch.facts is None, (
        f"费用那一批的清单里没有 facts，加载器却给出了 {batch.facts!r}"
        "——没记的前提头不许补默认值"
    )
    return (
        f"v{batch.version} · 画布 {batch.canvas.width}×{batch.canvas.height} · "
        f"{batch.device} · {len(names)} 条裁剪（名字唯一）· 无前提头"
    )


def check_every_crop_stays_inside_the_canvas() -> str:
    """每条裁剪的原点 + 尺寸都落在画布内，**含两端口径**。

    出处是[基准集] §2 与[设备层] §4：右缘最大只能是画布最后一列、下缘最大只能是
    最后一行。写超了就是"含了一个不存在的像素"——这个坑规格自己踩过
    （顶部HUD与数字的 `cost_panel` 右缘写到 2800）。
    """
    batch = _batch_mod().load_batch(BATCH)
    bad = []
    for case in batch.cases:
        rect = case.rect
        right = rect.x + rect.width - 1
        bottom = rect.y + rect.height - 1
        if rect.x < 0 or rect.y < 0:
            bad.append(f"{case.name}: 左上角 ({rect.x},{rect.y}) 越出画布")
        if right > batch.canvas.width - 1:
            bad.append(f"{case.name}: 右缘 {right} 越出最后一列 {batch.canvas.width - 1}")
        if bottom > batch.canvas.height - 1:
            bad.append(f"{case.name}: 下缘 {bottom} 越出最后一行 {batch.canvas.height - 1}")
    assert not bad, "；".join(bad[:6])
    return f"{len(batch.cases)} 条裁剪都落在 {batch.canvas.width}×{batch.canvas.height} 内（含两端）"


def check_case_frame_is_a_raw_frame() -> str:
    """一条基准集给出来的**是设备层 §4 定义的那种原始帧**，不是裁剪本身。

    出处是[基准集] §3 那条"裁剪要贴回画布"：头 16 字节 = 4 个小端 uint32，后接
    宽 × 高 × 4 字节。长度对不上，下游解帧就会拒——**而那时候已经离这里很远了**。
    """
    batch = _batch_mod().load_batch(BATCH)
    case = batch.cases[0]
    raw = case.raw()
    width, height, pixel_format, color_space = struct.unpack_from("<4I", raw, 0)
    assert (width, height, pixel_format, color_space) == (CANVAS_W, CANVAS_H, 1, 1), (
        f"{case.name} 的头是 ({width}, {height}, {pixel_format}, {color_space})，"
        f"该是 ({CANVAS_W}, {CANVAS_H}, 1, 1)"
    )
    expected = 16 + width * height * 4
    assert len(raw) == expected, (
        f"{case.name} 给了 {len(raw)} 字节，该是 {expected}"
        "——拿裁剪本身当帧就会这样（ROI 的取法就没在被测了）"
    )
    return f"{case.name}: {len(raw)} 字节，头 ({width}, {height}, {pixel_format}, {color_space})"


def check_crop_is_pasted_back_pixel_for_pixel() -> str:
    """贴回画布是**逐像素**的：裁剪那一块与原 PNG 一个像素都不差。

    出处是[基准集] §3 那句"裁剪要贴回画布，不许直接把裁剪当帧"——**ROI 的取法
    仍然是真的在被测**。这条同时钉住帧名与文件名的关系：帧名就是文件名去扩展名。
    """
    batch = _batch_mod().load_batch(BATCH)
    frame_mod = _mod("ark.engine.vision.frame")
    bad = []
    checked = 0
    for case in batch.cases[:4]:
        png = BATCH / f"{case.name}.png"
        if not png.is_file():
            bad.append(f"{case.name}: 按帧名找不到 {png.name}")
            continue
        frame = frame_mod.frame_from_raw(case.raw())
        want = np.asarray(Image.open(png).convert("RGB"))
        rect = case.rect
        got = frame.pixels[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width]
        if got.shape != want.shape:
            bad.append(f"{case.name}: 贴回那块是 {got.shape}，裁剪是 {want.shape}")
            continue
        diff = int(np.abs(got.astype(int) - want.astype(int)).max())
        if diff != 0:
            bad.append(f"{case.name}: 贴回那块与 PNG 最大差 {diff}（缩放过就会这样）")
            continue
        checked += 1
    assert not bad, "；".join(bad[:6])
    return f"{checked} 条裁剪贴回后逐像素相等，最大差 0"


def check_batch_does_not_need_the_truth_file() -> str:
    """基准集**只依赖清单与裁剪**，不依赖真值文件。

    出处是[基准集] §1 那条"期望值不住这里"（与 D-全局-17 的取材 / 验收分开同源）。
    做法是把 `index.json` 与裁剪复制到一个没有 `取材.json`、没有 `README.md` 的
    目录里再加载一次——**"不许把真值读进来"这条只有这样才验得动**。
    """
    batch_mod = _batch_mod()
    full = batch_mod.load_batch(BATCH)
    with _writable_tmp() as tmp:
        dest = tmp / "费用"
        dest.mkdir()
        shutil.copy(BATCH / "index.json", dest / "index.json")
        for case in full.cases:
            shutil.copy(BATCH / f"{case.name}.png", dest / f"{case.name}.png")
        lean = batch_mod.load_batch(dest)
    assert (lean.version, lean.device) == (full.version, full.device), (
        "去掉真值文件之后版本 / 机型变了：加载器读了不该读的东西"
    )
    assert [c.name for c in lean.cases] == [c.name for c in full.cases], (
        "去掉真值文件之后帧名变了：加载器读了不该读的东西"
    )
    want = [(c.rect.x, c.rect.y, c.rect.width, c.rect.height) for c in full.cases]
    got = [(c.rect.x, c.rect.y, c.rect.width, c.rect.height) for c in lean.cases]
    assert got == want, "去掉真值文件之后裁剪位置变了：加载器读了不该读的东西"
    return f"只留 index.json + {len(full.cases)} 张裁剪，加载结果一模一样"


# ---------------------------------------------------------------------------
# B 族：重放适配器
# ---------------------------------------------------------------------------


def check_replay_device_has_the_six_methods() -> str:
    """重放适配器实现了设备端口那六个方法。

    出处是[设备层] §5 那张两列的表：`grab_frame` · 四个动词 · `device_facts`
    ——"这几个方法的名字与签名是接口的一部分"。
    """
    replay = _replay_mod()
    device = replay.ReplayDevice([_raw_blank()])
    wanted = ("grab_frame", "device_facts", "deploy", "retreat", "skill_on", "skill_off")
    missing = [name for name in wanted if not callable(getattr(device, name, None))]
    assert not missing, (
        f"重放适配器还缺 {missing}：轴上有几个动词，端口上就该有几个入口"
    )
    return f"六个方法都在（{type(device).__name__}）"


def check_verbs_only_write_a_ledger() -> str:
    """四个动词**只记一笔**，而且它**发不出命令**。

    出处是[设备层] §5 与[离线验收] §2 那张不一致清单的第一条："注入只记一笔、
    不真的点"。"发不出命令"这条查的是**构造面**：重放适配器收的是帧源与前提头，
    **没有 transport** —— 一个拿不到 transport 的适配器，在类型上就点不出去
    （与 W13 查 `AdbDevice` 三样必填同一个手法）。
    """
    from ark.contracts import GridCoord

    replay = _replay_mod()
    device = replay.ReplayDevice([_raw_blank()])
    device.deploy(0, GridCoord(1, 2), "右")
    device.retreat(GridCoord(3, 4))
    device.skill_on(GridCoord(3, 4))
    device.skill_off(GridCoord(3, 4))

    ledger = getattr(device, "ledger", None)
    assert isinstance(ledger, tuple), (
        f"流水账是 {type(ledger).__name__}：它该是一串**不可变**的记录"
    )
    assert len(ledger) == 4, f"调了四个动词，流水账里却有 {len(ledger)} 条"
    verbs = [entry.verb for entry in ledger]
    assert verbs == ["deploy", "retreat", "skill_on", "skill_off"], (
        f"流水账里的动词是 {verbs}：它该就是那四个方法名，不另发明一套词"
    )
    first = tuple(ledger[0].params)
    assert first == (0, GridCoord(1, 2), "右"), (
        f"deploy 那一笔记成了 {first}：该按签名顺序记 (card, at, facing)"
    )

    params = inspect.signature(replay.ReplayDevice.__init__).parameters
    assert "transport" not in params, (
        f"重放适配器的构造参数里有 transport（{sorted(params)}）：它一拿到就能发命令，"
        "而离线这一路一条都不许发"
    )
    assert not hasattr(device, "transport"), "重放适配器身上挂着 transport"
    return f"四条流水账 {verbs}；构造参数 {sorted(params)}，没有 transport"


def check_frames_come_out_in_order_and_then_stop() -> str:
    """帧**依次**吐；吐完再要就报错——**不许回卷、不许重吐**。

    出处是[设备层] §5 那句"`grab_frame()` 返回该帧号对应的录制帧"。[基准集] §3
    把这条收紧成"跑过一遍是一次性的"：回卷会让"跑过一遍"这句话变成假的
    （离线验收 §1 要的正是"这一批跑过了"）。
    """
    replay = _replay_mod()
    exhausted = getattr(replay, "ReplayExhausted", None)
    assert exhausted is not None, (
        "重放适配器没有 ReplayExhausted：帧吐完了要报**这一层自己的异常**"
        "（与 AdbTimeout 同类），不许静默给一帧旧的"
    )
    first, second = _raw_blank(value=1), _raw_blank(value=2)
    device = replay.ReplayDevice([first, second])
    assert device.grab_frame() == first, "第一帧不是帧源里的第一帧"
    assert device.grab_frame() == second, "第二帧不是帧源里的第二帧"
    try:
        again = device.grab_frame()
    except exhausted:
        return "两帧依次吐完，第三次报 ReplayExhausted"
    raise AssertionError(
        f"两帧吐完了还给了 {len(again)} 字节：回卷或重吐会让「跑过一遍」变成假的"
    )


def check_facts_are_replayed_or_refused() -> str:
    """前提头**记了就原样回放，没记就报错**——不许编一个。

    出处是[设备层] §5 那句"`device_facts()` 从录制的前提头里回放"。[基准集] §2
    把 `facts` 定成**可选**：费用那一批当时没记，所以那条路必须有"报错"这一支。
    编一个环境事实与编一个投影是同一种坏（端口与错误 §5）。
    """
    from ark.contracts import Canvas, DeviceFacts

    replay = _replay_mod()
    missing_exc = getattr(replay, "ReplayFactsMissing", None)
    assert missing_exc is not None, (
        "重放适配器没有 ReplayFactsMissing：录制没记前提头时要报错，不许编一个"
    )
    facts = DeviceFacts(resolution=Canvas(CANVAS_W, CANVAS_H), orientation="1")
    device = replay.ReplayDevice([_raw_blank()], facts=facts)
    got = device.device_facts()
    assert got == facts, (
        f"记进去的是 {facts!r}，回放出来的是 {got!r}：前提头要原样回放"
    )
    bare = replay.ReplayDevice([_raw_blank()])
    try:
        invented = bare.device_facts()
    except missing_exc:
        return f"记了就回放 {got.resolution.width}×{got.resolution.height}/{got.orientation}；没记就报 ReplayFactsMissing"
    except BaseException as other:  # noqa: BLE001 —— 报错了，但报错了类型
        raise AssertionError(
            f"没记前提头时该报 ReplayFactsMissing，实际 {type(other).__name__}: {other}"
        ) from None
    raise AssertionError(
        f"录制没记前提头，`device_facts()` 却给出了 {invented!r}"
        "——那是编出来的环境事实"
    )


# ---------------------------------------------------------------------------
# B 族：这条路本身
# ---------------------------------------------------------------------------


def check_readings_through_replay_match_the_truth() -> str:
    """**离线这条路是透明的**：31 帧经基准集 + 重放适配器喂进识别链，读数与真值全对。

    出处是[离线验收] §1 那句"把录下来的帧喂回线上识别链路，跑到槽为止"，加上
    [开发路线] 段一那条判据"对着基准集逐帧对得上"。

    真值**不在这里重抄**：它住同目录的 `cost_spec.py`。这条与 cost_spec 的 D1 / B1
    问的不是同一件事——那边问"读数器准不准"，这边问"**换一条路进来**会不会变"。
    """
    import cost_spec

    from ark.contracts import Exact, Unknown, Valued

    feed = _feed_mod()
    frame_mod = _mod("ark.engine.vision.frame")
    cost = _mod("ark.engine.vision.cost")
    glyphs = _mod("ark.engine.vision.glyphs")
    ui = _mod("ark.engine.vision.layout").UI
    table = glyphs.GlyphTable.load(GLYPH_DIR)
    batch = _batch_mod().load_batch(BATCH)

    digits_want = {f"shot{n:02d}": v for n, v in cost_spec.DIGITS.items()}
    fills_want = {f"shot{n:02d}": v for n, v in cost_spec.FILLS.items()}
    full = {f"shot{n:02d}" for n in cost_spec.FULL}

    bad: list[str] = []
    seen = 0
    for case, raw in feed.replay_frames(batch):
        seen += 1
        frame = frame_mod.frame_from_raw(raw)
        slot = cost.read_cost_digits(frame, table, ui=ui)
        want = digits_want.get(case.name)
        if want is None:
            bad.append(f"{case.name}: 真值表里没有这一帧")
        elif not (isinstance(slot, Valued) and isinstance(slot.payload, Exact)
                  and slot.payload.value == want):
            bad.append(f"{case.name}: 费用该读到 {want}，收到 {_slot_brief(slot)}")

        bar = cost.read_cost_bar(frame, ui=ui)
        if case.name in full:
            if not (isinstance(bar, Valued) and isinstance(bar.payload, Unknown)):
                bad.append(f"{case.name}: 满态该报 Unknown，收到 {_slot_brief(bar)}")
        elif not (isinstance(bar, Valued) and isinstance(bar.payload, Exact)
                  and bar.payload.value == fills_want.get(case.name)):
            bad.append(
                f"{case.name}: 填充该读到 {fills_want.get(case.name)} px，"
                f"收到 {_slot_brief(bar)}"
            )
    assert seen == len(batch.cases), (
        f"基准集有 {len(batch.cases)} 条，重放只吐了 {seen} 帧"
    )
    assert not bad, "；".join(bad[:6]) + (f"（共 {len(bad)} 帧不对）" if len(bad) > 6 else "")
    return f"{seen} 帧经重放适配器读出来，费用与填充全部与真值一致"


def check_engine_does_not_depend_on_pillow() -> str:
    """`engine` 与 `contracts` **不 import PIL**。

    出处是[基准集] §3 那条"PNG 解码住离线侧"：pyproject 里 pillow 挂在 `offline`
    那组额外依赖上，所以**引擎不装它也得能跑**。PNG → 原始帧这一步是离线的事，
    重放适配器收到的**已经是原始帧字节**。
    """
    pillow = re.compile(r"^\s*(?:import|from)\s+PIL\b", re.M)
    hits: list[str] = []
    scanned = 0
    for root in (REPO / "src" / "ark" / "engine", REPO / "src" / "ark" / "contracts"):
        for path in sorted(root.rglob("*.py")):
            scanned += 1
            if pillow.search(path.read_text(encoding="utf-8-sig")):
                hits.append(str(path.relative_to(REPO)))
    assert not hits, (
        f"{hits} import 了 PIL：pillow 是 offline 那组额外依赖，"
        "引擎不装它也得能跑——PNG 解码归 offline/bench/"
    )
    return f"扫了 {scanned} 个模块，engine 与 contracts 都不 import PIL"


def check_bench_script_goes_through_the_offline_path() -> str:
    """跑基准集的脚本走**基准集加载器 + 重放适配器**，不碰真机、不自己解 PNG。

    出处是[代码结构] 的"scripts/ 是开发期胶水"与[设备层] §5："离线重放不是特例
    代码，落地条件就是**换一个适配器**"。脚本自己拼 adb 或自己开 PNG，就等于把
    这条路复制了一份——那条路的下一处改动会漏掉它。
    """
    script = REPO / "scripts" / "bench.py"
    assert script.is_file(), f"{script.relative_to(REPO)} 不存在"
    text = script.read_text(encoding="utf-8-sig")
    assert "offline.bench" in text, (
        "跑基准集的脚本没走 offline/bench：加载基准集那条路只有一份"
    )
    assert "replay" in text, "跑基准集的脚本没走重放适配器"
    banned = [word for word in ("screencap", "AdbTransport", "PIL") if word in text]
    assert not banned, (
        f"脚本里出现了 {banned}：离线这条路不许碰真机，也不许自己解 PNG"
    )
    return "脚本走 offline/bench + 重放适配器，不出现 screencap / AdbTransport / PIL"


CHECKS: tuple[Check, ...] = (
    Check("B1", "采集与验收/基准集 §2（字段表）",
          "清单里的必填字段一样不少地读出来，没记的前提头不许补",
          check_batch_loads_with_its_manifest),
    Check("B2", "采集与验收/基准集 §2（含两端口径）+ 设备层 §4",
          "每条裁剪的原点与尺寸都落在画布内",
          check_every_crop_stays_inside_the_canvas),
    Check("B3", "采集与验收/基准集 §3 + 设备层 §4（原始帧的字节布局）",
          "一条基准集给出来的是原始帧，不是裁剪本身",
          check_case_frame_is_a_raw_frame),
    Check("B4", "采集与验收/基准集 §3（裁剪要贴回画布）",
          "贴回画布是逐像素的，帧名就是文件名去扩展名",
          check_crop_is_pasted_back_pixel_for_pixel),
    Check("B5", "采集与验收/基准集 §1（期望值不住这里）+ D-全局-17",
          "基准集只依赖清单与裁剪，不依赖真值文件",
          check_batch_does_not_need_the_truth_file),
    Check("B6", "设备层 §5（端口方法的名字与签名是接口的一部分）",
          "重放适配器实现了设备端口那六个方法",
          check_replay_device_has_the_six_methods),
    Check("B7", "设备层 §5 + 采集与验收/离线验收 §2（注入只记一笔）",
          "四个动词只记一笔，而且它发不出命令",
          check_verbs_only_write_a_ledger),
    Check("B8", "设备层 §5（帧依次吐）+ 基准集 §3（跑一遍是一次性的）",
          "帧依次吐，吐完再要就报错，不许回卷",
          check_frames_come_out_in_order_and_then_stop),
    Check("B9", "设备层 §5（前提头回放）+ 基准集 §2（facts 可选）",
          "前提头记了就原样回放，没记就报错，不许编",
          check_facts_are_replayed_or_refused),
    Check("B10", "采集与验收/离线验收 §1 + 开发路线 段一（判据）",
          "31 帧经重放适配器读出来，与真值全对",
          check_readings_through_replay_match_the_truth),
    Check("B11", "采集与验收/基准集 §3（PNG 解码住离线侧）",
          "engine 与 contracts 不 import PIL",
          check_engine_does_not_depend_on_pillow),
    Check("B12", "代码结构（scripts/ 是胶水）+ 设备层 §5（换一个适配器）",
          "跑基准集的脚本走这条路，不碰真机、不自己解 PNG",
          check_bench_script_goes_through_the_offline_path),
)


def main() -> int:
    failures: list[tuple[Check, str]] = []
    print(f"基准集与重放适配器规格符合性验收 —— {len(CHECKS)} 条\n")
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
