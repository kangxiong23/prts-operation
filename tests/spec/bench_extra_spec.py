"""T6 的第二张网 —— **规划者自己用的补充检查，不交给执行者**。

## 为什么单独一份，而不是并进 `bench_spec.py`

`bench_spec.py` 问的是"**规格说的那些话成立吗**"；这一份问的是"**它是不是只对那
唯一一批样本成立**"。两张网问的不是同一件事，所以分成两份——分开，这个区别就
一直看得见。

（历史：它当初另起一个文件，是因为 `bench_spec.py` 的哈希钉在 T6 的派工单里，
改它就等于让执行者手里的对齐基准当场失效。**D-全局-22 之后哈希改盖「先读」那一
列**，那条理由没有了，但分开的理由还在。）

**它故意没有 pytest 外壳**（没有 `test_bench_extra_spec.py`），所以
`pytest -q` **不收集它**——执行者跑自己的单元测试时不会连带跑它。跑法只有一条：

    python tests/spec/bench_extra_spec.py

## 它补的是什么

`bench_spec.py` 的 B1–B12 里，有一批是**对着费用那一批**验的，而那一批有个
巧合：**31 条裁剪的原点全是 `(2580, 840)`、尺寸全是 `220×136`、画布固定
`2800×1260`**。于是三件事其实**一次都没被真正测到**：

| 没被测到 | 为什么 |
| --- | --- |
| **原点**真的被读了 | 31 条原点一样，写死一个也照样过 |
| **尺寸**真的被读了 | 31 条尺寸一样，写死一个也照样过 |
| **画布**来自清单 | 只有一批，写死 `2800×1260` 也照样过 |

这正是"实现被调死在唯一一批样本上"的那一类。所以这一份用**合成的小批**
（按[基准集] §2 的格式现造）把这三样各自逼出来 —— 一个照着规格写的加载器，
对任何**合格式**的批都该给出同样的结果。

**这一份不引入任何新要求**：每条检查的出处都是已经写下来的那些话。跑出红来，
要么是实现没按规格写，要么是**规格自己没说清**（那就记成规格洞，下一轮补）。

纪律同 [README](README.md) —— **不在这里重述**（同一条规定只写一次）。
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import itertools
import json
import os
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

CANVAS_W, CANVAS_H = 2800, 1260


@dataclass(frozen=True)
class Check:
    ident: str
    spec: str
    claim: str
    probe: Callable[[], str]


_LOADED: dict[str, object] = {}


def _mod(dotted: str):
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
    """建一个**能进得去**的临时目录（理由同 `bench_spec.py`）。"""
    base = REPO / "data" / "_spec_tmp"
    base.mkdir(parents=True, exist_ok=True)
    dest = base / f"benchx{os.getpid()}_{next(_TMP_SEQ)}"
    dest.mkdir()
    try:
        yield dest
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def _raw_blank(width: int = 8, height: int = 4, value: int = 7) -> bytes:
    header = struct.pack("<4I", width, height, 1, 1)
    return header + bytes([value, value, value, 255]) * (width * height)


def _write_batch(
    root: Path,
    crops: list[tuple[str, tuple[int, int], tuple[int, int], int]],
    *,
    canvas: tuple[int, int] = (CANVAS_W, CANVAS_H),
    version: int = 1,
    device: str = "合成",
    facts: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """按[基准集] §2 的字段表现造一批。

    `crops` 每条是 `(帧名, 原点, 尺寸, 灰度值)` —— 灰度值让每条裁剪互不相同，
    这样"贴错了地方"与"贴错了大小"都能看出来。
    """
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, origin, size, value in crops:
        width, height = size
        Image.fromarray(np.full((height, width, 3), value, np.uint8)).save(
            root / f"{name}.png"
        )
        entries.append(
            {
                "file": f"{name}.png",
                "origin": list(origin),
                "size": [width, height],
                "source": f"合成/{name}",
                "source_sha256": "0" * 64,
            }
        )
    manifest: dict = {
        "version": version,
        "canvas": list(canvas),
        "device": device,
        "crops": entries,
    }
    if facts is not None:
        manifest["facts"] = facts
    if extra is not None:
        manifest.update(extra)
    (root / "index.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return root


def _paste_check(root: Path, case: object) -> str | None:
    """把一条基准集的帧解出来，与它那张 PNG 比 —— 不吻合就返回一句话。"""
    frame_mod = _mod("ark.engine.vision.frame")
    png = root / f"{case.name}.png"
    if not png.is_file():
        return f"{case.name}: 按帧名找不到 {png.name}"
    frame = frame_mod.frame_from_raw(case.raw())
    want = np.asarray(Image.open(png).convert("RGB"))
    rect = case.rect
    got = frame.pixels[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width]
    if got.shape != want.shape:
        return f"{case.name}: 贴回那块是 {got.shape}，裁剪是 {want.shape}"
    diff = int(np.abs(got.astype(int) - want.astype(int)).max())
    if diff != 0:
        return f"{case.name}: 贴回那块与 PNG 最大差 {diff}"
    return None


# ---------------------------------------------------------------------------
# X 族：把"原点 / 尺寸 / 画布"各自逼出来
# ---------------------------------------------------------------------------


def check_every_shipped_crop_is_pasted_back_pixel_for_pixel() -> str:
    """**31 条全量**逐像素贴回，不是抽 4 条。

    出处是[基准集] §3 那句"裁剪要贴回画布"。[基准集] §2 同时钉住帧名 = 文件名
    去扩展名，所以这里按帧名去找 PNG —— 名字对不上就会露出来。
    """
    batch = _batch_mod().load_batch(BATCH)
    bad = [msg for case in batch.cases if (msg := _paste_check(BATCH, case))]
    assert not bad, "；".join(bad[:6])
    return f"{len(batch.cases)} 条全部逐像素相等，最大差 0"


def check_crops_land_at_their_own_origin_and_size() -> str:
    """两条裁剪在**不同原点**、**不同尺寸**时，各自按自己的位置与大小贴回。

    出处是[基准集] §2 的 `origin` / `size` 两栏。费用那一批 31 条的原点与尺寸
    **完全一样**，所以"原点被读了"这件事在那一批上验不出来 —— 写死一个也照样过。
    这条用一个两条的小批把它逼出来。
    """
    batch_mod = _batch_mod()
    with _writable_tmp() as tmp:
        root = _write_batch(
            tmp / "两处",
            [
                ("left", (3, 5), (7, 4), 40),
                ("right", (900, 700), (11, 9), 200),
            ],
        )
        batch = batch_mod.load_batch(root)
        want = {"left": ((3, 5), (7, 4)), "right": ((900, 700), (11, 9))}
        bad = []
        for case in batch.cases:
            origin, size = want.get(case.name, ((None, None), (None, None)))
            got = (case.rect.x, case.rect.y)
            if got != origin:
                bad.append(f"{case.name}: 原点读成了 {got}，清单里写的是 {origin}")
            if (case.rect.width, case.rect.height) != size:
                bad.append(
                    f"{case.name}: 尺寸读成了 {(case.rect.width, case.rect.height)}，"
                    f"清单里写的是 {size}"
                )
            msg = _paste_check(root, case)
            if msg:
                bad.append(msg)
        assert not bad, "；".join(bad[:6])
        frames = [case.raw() for case in batch.cases]
    assert frames[0] != frames[1], (
        "两条不同位置、不同大小的裁剪造出了同一帧 —— 位置或尺寸没进到帧里"
    )
    return "两条裁剪各自落在自己的原点、各自是自己的大小，帧也互不相同"


def check_canvas_comes_from_the_manifest() -> str:
    """**画布来自清单**，不是写死的 2800×1260。

    出处是[基准集] §2：`canvas` 是必填字段，而 `facts` 里**只有** `orientation`
    —— 所以设备事实那个分辨率只能从 `canvas` 来（[设备层] §5：两个值都从设备报的
    那一条来，回放时也就是录制里记的那两个）。费用那一批只有一种画布，写死一个
    也照样过。
    """
    batch_mod = _batch_mod()
    replay = _replay_mod()
    with _writable_tmp() as tmp:
        root = _write_batch(
            tmp / "小画布",
            [("a", (2, 1), (4, 3), 60)],
            canvas=(64, 48),
            facts={"orientation": "3"},
        )
        batch = batch_mod.load_batch(root)
        assert (batch.canvas.width, batch.canvas.height) == (64, 48), (
            f"清单写 64×48，加载器读成 {batch.canvas}"
        )
        case = batch.cases[0]
        raw = case.raw()
        width, height, pixel_format, color_space = struct.unpack_from("<4I", raw, 0)
        assert (width, height, pixel_format, color_space) == (64, 48, 1, 1), (
            f"帧的头是 ({width}, {height}, {pixel_format}, {color_space})，该是 (64, 48, 1, 1)"
        )
        expected = 16 + 64 * 48 * 4
        assert len(raw) == expected, f"帧长 {len(raw)} 字节，该是 {expected}"
        facts = replay.ReplayDevice([raw], facts=batch.facts).device_facts()
        got = (facts.resolution.width, facts.resolution.height)
        assert got == (64, 48), f"设备事实的分辨率读成 {got}，清单写的是 (64, 48)"
        assert facts.orientation == "3", (
            f"设备事实的方向读成 {facts.orientation!r}，清单写的是 '3'"
        )
    return "换一张 64×48 的画布，帧的头、帧长、设备事实的分辨率都跟着变"


# ---------------------------------------------------------------------------
# X 族：清单怎么读
# ---------------------------------------------------------------------------


def check_extra_manifest_fields_do_not_change_loading() -> str:
    """清单里**批自己的额外字段**不影响加载。

    出处是[基准集] §2 最后那句："批自己的额外字段可以有（费用那一批多一个 `box`），
    加载器不读它们。"做法是同一批数据造两份清单，一份多几个字段，加载结果必须一样。
    """
    batch_mod = _batch_mod()
    crops = [("a", (1, 1), (3, 2), 50), ("b", (30, 20), (5, 5), 120)]
    with _writable_tmp() as tmp:
        plain = batch_mod.load_batch(_write_batch(tmp / "plain", crops))
        fancy = batch_mod.load_batch(
            _write_batch(
                tmp / "fancy",
                crops,
                extra={"box": [0, 0, 40, 40], "note": "这一批测什么", "作者": "规划者"},
            )
        )
    assert (plain.version, plain.device, plain.facts) == (
        fancy.version,
        fancy.device,
        fancy.facts,
    ), "多几个额外字段之后版本 / 机型 / 前提头变了：加载器读了不该读的字段"
    assert [c.name for c in plain.cases] == [c.name for c in fancy.cases], (
        "多几个额外字段之后帧名变了"
    )
    one = [(c.rect.x, c.rect.y, c.rect.width, c.rect.height) for c in plain.cases]
    two = [(c.rect.x, c.rect.y, c.rect.width, c.rect.height) for c in fancy.cases]
    assert one == two, "多几个额外字段之后裁剪位置变了"
    return "同一批数据加三个额外字段，加载结果一模一样"


def check_frame_order_follows_the_manifest_order() -> str:
    """帧**依次吐**的顺序就是清单里 `crops` 的顺序。

    出处是[基准集] §3（帧依次吐）与 §2（`crops` 是这一批的清单）。把清单里两条
    裁剪换个位置，吐出来的顺序要跟着换 —— 否则"第几帧"与清单就对不上了，
    而报告是按帧名写的。
    """
    feed = _feed_mod()
    with _writable_tmp() as tmp:
        root = _write_batch(
            tmp / "顺序",
            [("second", (10, 10), (3, 3), 90), ("first", (20, 20), (3, 3), 30)],
        )
        batch = _batch_mod().load_batch(root)
        names = [case.name for case, _ in feed.replay_frames(batch)]
    assert names == ["second", "first"], (
        f"清单里先写的是 second、后写的是 first，吐出来却是 {names}"
    )
    return "清单里两条换个位置，吐出来的顺序跟着换"


# ---------------------------------------------------------------------------
# X 族：重放适配器的边角
# ---------------------------------------------------------------------------


def check_the_frame_source_may_be_a_generator() -> str:
    """帧源可以是一个**生成器**，不要求是一个已经摆在内存里的列表。

    出处是[设备层] §5 那个词——它收的是"**帧源**"，不是"帧表"。段三要跑一整局，
    帧不能一次全驻留（一帧 14 MB）。所以这里传一个生成器，行为必须与列表一样。
    """
    replay = _replay_mod()
    exhausted = getattr(replay, "ReplayExhausted", None)
    assert exhausted is not None, "重放适配器没有 ReplayExhausted"
    frames = [_raw_blank(value=1), _raw_blank(value=2), _raw_blank(value=3)]
    device = replay.ReplayDevice(frame for frame in frames)
    got = [device.grab_frame() for _ in frames]
    assert got == frames, "生成器当帧源时吐出来的帧与列表不一致"
    try:
        device.grab_frame()
    except exhausted:
        return f"{len(frames)} 帧的生成器当帧源，依次吐完就报尽"
    raise AssertionError("生成器吐完了还给了帧：吐完就该报尽，与列表那条一样")


def check_an_empty_frame_source_is_immediately_exhausted() -> str:
    """**空帧源**第一次要帧就报尽，不许给一个空帧。

    出处是[基准集] §3 那句"帧依次吐、吐完再要就报错"——一帧都没有的时候，
    "吐完"在第一次调用时就已经成立。给一个 0 字节的"帧"是最坏的一支：
    它会一路走到解帧那里才炸，而那时候离这里已经很远。
    """
    replay = _replay_mod()
    exhausted = getattr(replay, "ReplayExhausted", None)
    assert exhausted is not None, "重放适配器没有 ReplayExhausted"
    device = replay.ReplayDevice([])
    try:
        got = device.grab_frame()
    except exhausted:
        return "空帧源第一次 grab_frame() 就报 ReplayExhausted"
    raise AssertionError(
        f"空帧源给了 {len(got)} 字节：那是编出来的一帧，不是「吐完了」"
    )


def check_replay_adapter_takes_no_assets_or_view() -> str:
    """重放适配器的构造面里**没有资产端口与视图**。

    出处是[设备层] §5 那条"它收帧源 + 可选的录制前提头，**不收资产端口与视图**
    ——它一个像素都不算、一条命令都不发，投影对它没有用途"。`AdbDevice` 那边三样
    必填（W13），这里两样都不该有；有了就说明有人把"重放"当成了另一种在线适配器。
    """
    replay = _replay_mod()
    params = inspect.signature(replay.ReplayDevice.__init__).parameters
    extra = [name for name in ("assets", "view") if name in params]
    assert not extra, (
        f"重放适配器的构造参数里有 {extra}（全部参数：{sorted(params)}）："
        "它一个像素都不算，投影对它没有用途"
    )
    return f"构造参数只有 {sorted(params)}，没有 assets / view"


def check_the_ledger_carries_no_frame_number() -> str:
    """流水账**不带帧号**，四个动词各自的参数个数与签名一致。

    出处是[设备层] §5 那段："四个动词只记一笔——动词就是那四个方法名、参数按签名
    顺序排，**不带帧号**（设备层拿不到逻辑帧）"。帧号由调用方在记录里补
    （[时序层] §4），设备层自己填一个只能是编的。
    """
    from ark.contracts import GridCoord

    replay = _replay_mod()
    device = replay.ReplayDevice([_raw_blank()])
    fresh = device.ledger
    assert isinstance(fresh, tuple) and len(fresh) == 0, (
        f"一次手都没动，流水账里已经有 {fresh!r}"
    )
    device.deploy(2, GridCoord(5, 6), "左")
    device.retreat(GridCoord(1, 1))
    device.skill_on(GridCoord(2, 2))
    device.skill_off(GridCoord(3, 3))
    arity = {"deploy": 3, "retreat": 1, "skill_on": 1, "skill_off": 1}
    bad = []
    for entry in device.ledger:
        if hasattr(entry, "frame"):
            bad.append(f"{entry.verb} 那一条带了帧号：设备层拿不到逻辑帧")
        want = arity.get(entry.verb)
        if want is None:
            bad.append(f"流水里出现了没见过的动词 {entry.verb!r}")
        elif len(tuple(entry.params)) != want:
            bad.append(
                f"{entry.verb} 记了 {len(tuple(entry.params))} 个参数，"
                f"签名是 {want} 个"
            )
    assert not bad, "；".join(bad)
    return f"空流水是空元组；四条 {list(arity)} 的参数个数都对，且都没有帧号"


def check_feed_goes_through_the_replay_adapter() -> str:
    """`replay_frames` **真的经过重放适配器**，不是自己读了帧。

    出处是[代码结构] §尚未落地 那一条："`bench/feed.py` 把一批帧**经重放适配器**
    逐帧吐出来"。这条要是不查，一个直接读文件、绕过适配器的实现照样能让 B10 全绿
    —— 而它会**悄悄丢掉**流水账与"吐完报尽"这两件事，那正是段三要用的。

    做法是**换掉 `ReplayDevice` 再看它有没有被用上**（同 `check_the_checks.py` 的
    手法）。补丁要**同时打到两个模块**：`feed` 可能是 `from … import ReplayDevice`
    （名字绑在自己身上），也可能是 `import … as replay`（名字还在那边）。
    """
    feed = _feed_mod()
    replay = _replay_mod()
    used: list[int] = []

    class _Spy:
        """只记"我被造出来了"，然后照常吐帧。"""

        def __init__(self, frames, *, facts=None):
            used.append(1)
            self._frames = iter(frames)

        def grab_frame(self):
            return next(self._frames)

        def device_facts(self):
            raise AssertionError("这条检查不查设备事实")

        def deploy(self, card, at, facing):
            raise AssertionError("这条检查不发动作")

        def retreat(self, at):
            raise AssertionError("这条检查不发动作")

        def skill_on(self, at):
            raise AssertionError("这条检查不发动作")

        def skill_off(self, at):
            raise AssertionError("这条检查不发动作")

    patched = []
    for module in (feed, replay):
        if hasattr(module, "ReplayDevice"):
            patched.append((module, module.ReplayDevice))
            module.ReplayDevice = _Spy
    assert patched, (
        "`ReplayDevice` 这个名字既不在 feed 也不在 replay 里，这条检查没法打补丁"
    )
    try:
        with _writable_tmp() as tmp:
            root = _write_batch(
                tmp / "小批", [("a", (1, 1), (3, 2), 40), ("b", (5, 4), (2, 2), 80)]
            )
            batch = _batch_mod().load_batch(root)
            got = [case.name for case, _ in feed.replay_frames(batch)]
    finally:
        for module, original in patched:
            module.ReplayDevice = original

    assert used, (
        "换掉 ReplayDevice 之后 replay_frames 照样跑完了：它自己读了帧，"
        "没经过重放适配器 —— 那样流水账与「吐完报尽」都不在这条路上"
    )
    assert got == ["a", "b"], f"换掉适配器之后帧序变了：{got}"
    return f"replay_frames 造了 {len(used)} 个重放适配器，帧序 {got}"


def _mutate(root: Path, change) -> Path:
    """把这一批的清单改一处再写回去（造坏清单用）。"""
    path = root / "index.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return root


def check_a_broken_manifest_is_refused_not_patched() -> str:
    """清单坏了**在加载时就拒绝**，不许就地修补。

    出处是[基准集] §2 那三句：字段表里的"**必填**" · "原点 + 尺寸**必须**落在画布
    内" · "帧名**批内唯一**"，加上那句"没有就是没有——加载器**不许补一个默认值**"。

    **"不许补一个默认值"这条要往宽了读**：缺一个字段当默认值、越出画布就裁到边上、
    尺寸不符就按 PNG 来 —— 三种都是**替调用方编了一个它没读到的值**，而下游会拿
    那个值去取 ROI。所以坏清单的出路只有一条：当场报错。

    这一条**锁的是行为、不是错误类型**：报什么都行（`ValueError` / `KeyError` /
    `FileNotFoundError` / 自己的异常），**只要不是"照样加载成功"**。
    """
    batch_mod = _batch_mod()
    base = [("a", (1, 1), (3, 2), 50), ("b", (10, 10), (4, 4), 90)]

    def refused(what: str, root: Path) -> str | None:
        try:
            batch_mod.load_batch(root)
        except Exception:  # noqa: BLE001 —— 报错就是那条出路，类型不查
            return None
        return what

    with _writable_tmp() as tmp:
        bad = []
        cases = [
            ("清单缺一个必填字段（device）",
             _mutate(_write_batch(tmp / "m1", base), lambda m: m.pop("device"))),
            ("裁剪缺一个必填字段（source_sha256）",
             _mutate(_write_batch(tmp / "m2", base),
                     lambda m: m["crops"][0].pop("source_sha256"))),
            ("裁剪越出画布（右下角那一块）",
             _mutate(_write_batch(tmp / "m3", base),
                     lambda m: m["crops"][1].__setitem__("origin", [2798, 1258]))),
            ("帧名批内重复（两条都叫 a）",
             _write_batch(tmp / "m4", [("a", (1, 1), (3, 2), 50),
                                       ("a", (10, 10), (4, 4), 90)])),
            ("清单写的尺寸与 PNG 实际尺寸不符",
             _mutate(_write_batch(tmp / "m5", base),
                     lambda m: m["crops"][0].__setitem__("size", [5, 5]))),
            ("裁剪文件根本不在",
             _mutate(_write_batch(tmp / "m6", base),
                     lambda m: m["crops"][0].__setitem__("file", "nope.png"))),
        ]
        for what, root in cases:
            msg = refused(what, root)
            if msg:
                bad.append(msg)
        assert not bad, (
            "这些坏清单照样加载成功了（就地修补 = 替调用方编了一个它没读到的值）："
            + "；".join(bad)
        )
        # 顺带：好清单必须仍然读得出来 —— 否则"什么都报错"也能让上面全过
        good = batch_mod.load_batch(_write_batch(tmp / "ok", base))
    assert len(good.cases) == 2, "好清单读出来是 {} 条".format(len(good.cases))
    return f"{len(cases)} 种坏清单全部被拒；同格式的好清单仍然读得出来"


def check_the_loader_folds_the_recorded_facts() -> str:
    """清单里那个映射**由加载器折成 `DeviceFacts`**，不许原样透给设备层。

    出处是[基准集] §2 那段"前提头在清单里只记画布装不下的那一项"：清单的 `canvas`
    就是录制时的分辨率，所以 `facts` 里只剩 `orientation`，而 `BenchBatch.facts`
    的类型是 **`DeviceFacts | None`**。

    这条与 X13 是一对：**折这件事只能发生在加载器里**。原样透出去，设备层就得去认
    "清单里那个键叫 `orientation`"——那是把离线那套文件格式塞进引擎
    （[代码结构] 规则三：引擎对外只认自己的语义）。费用那一批**没有 `facts`**，
    所以它在 B1 上两条路都返回 `None`，看不出差别 —— 只有造一批带前提头的才分得开。
    """
    from ark.contracts import DeviceFacts

    batch_mod = _batch_mod()
    with _writable_tmp() as tmp:
        root = _write_batch(
            tmp / "带前提头",
            [("a", (1, 1), (3, 2), 50)],
            canvas=(64, 48),
            facts={"orientation": "3"},
        )
        batch = batch_mod.load_batch(root)
    got = batch.facts
    assert isinstance(got, DeviceFacts), (
        f"BenchBatch.facts 是 {type(got).__name__}，该是 DeviceFacts"
        "——把清单里那个映射原样透出去，设备层就得去认离线的键名"
    )
    assert (got.resolution.width, got.resolution.height) == (64, 48), (
        f"前提头的分辨率是 {got.resolution}，该取清单的 canvas 64×48"
    )
    assert got.orientation == "3", f"前提头的方向是 {got.orientation!r}，该是 '3'"
    return f"清单里的 facts 折成了 {got!r}"


def check_the_device_layer_does_not_read_the_manifest_shape() -> str:
    """重放适配器**不认清单那套键名**，也**不在构造时预读帧源**。

    出处是[设备层] §5 那段（前提头的类型是 `DeviceFacts` 或 `None` · "它一个像素
    都不算、也不解帧头"）与[代码结构] 规则三。两半：

    - 传一个**清单形状的映射**进去，不许悄悄把它当前提头用 —— 认键名是加载器的事；
    - 构造的时候**不许去帧源里取一帧**。预读一帧就是把那个"一个像素都不算"破了：
      帧源在段三是一条惰性的流（录下来的整局），谁在构造时就动它，谁就把"取像"
      提前到了装配那一刻（[设备层] §5：只有"取一帧"，没有常驻取像）。
    """
    from ark.contracts import Canvas, DeviceFacts

    replay = _replay_mod()
    raw = _raw_blank(64, 48, 9)

    raised = False
    try:
        loose = replay.ReplayDevice([raw], facts={"orientation": "3"})
    except Exception:  # noqa: BLE001 —— 报错就是那条出路，类型不查
        raised = True
    if not raised:
        try:
            answered = loose.device_facts()
        except Exception:  # noqa: BLE001 —— 报错也算没认它
            answered = None
        assert answered is None, (
            f"把清单形状的映射（{{'orientation': '3'}}）传进去，它答出了 {answered!r}："
            "设备层不许认离线那套键名"
        )

    pulled: list[int] = []

    def source():
        pulled.append(1)
        yield raw

    facts = DeviceFacts(resolution=Canvas(width=64, height=48), orientation="1")
    device = replay.ReplayDevice(source(), facts=facts)
    assert not pulled, (
        "构造重放适配器的时候就去帧源里取了一帧：帧源是惰性的，"
        "第一帧该由第一次 grab_frame() 取（设备层 §5）"
    )
    assert device.grab_frame() == raw, "第一次取帧没拿到帧源里的那一帧"
    return "清单形状的映射不被认；构造时不预读帧源，第一帧由第一次取帧给"


CHECKS: tuple[Check, ...] = (
    Check("X1", "采集与验收/基准集 §3（裁剪要贴回画布）+ §2（帧名 = 文件名去扩展名）",
          "31 条裁剪全量逐像素贴回，不是抽几条",
          check_every_shipped_crop_is_pasted_back_pixel_for_pixel),
    Check("X2", "采集与验收/基准集 §2（origin / size 两栏）",
          "两条不同原点、不同尺寸的裁剪各自落位",
          check_crops_land_at_their_own_origin_and_size),
    Check("X3", "采集与验收/基准集 §2（canvas 必填 · facts 只有 orientation）",
          "画布来自清单，不是写死的 2800×1260",
          check_canvas_comes_from_the_manifest),
    Check("X4", "采集与验收/基准集 §2（批自己的额外字段可以有）",
          "清单里的额外字段不影响加载",
          check_extra_manifest_fields_do_not_change_loading),
    Check("X5", "采集与验收/基准集 §3（帧依次吐）+ §2（crops 是清单）",
          "帧序就是清单里 crops 的顺序",
          check_frame_order_follows_the_manifest_order),
    Check("X6", "设备层 §5（收的是「帧源」）",
          "帧源可以是生成器，行为与列表一样",
          check_the_frame_source_may_be_a_generator),
    Check("X7", "采集与验收/基准集 §3（吐完再要就报错）",
          "空帧源第一次要帧就报尽，不给空帧",
          check_an_empty_frame_source_is_immediately_exhausted),
    Check("X8", "设备层 §5（不收资产端口与视图）",
          "重放适配器的构造面里没有 assets / view",
          check_replay_adapter_takes_no_assets_or_view),
    Check("X9", "设备层 §5（参数按签名顺序排 · 不带帧号）+ 时序层 §4",
          "流水账不带帧号，四个动词的参数个数都对",
          check_the_ledger_carries_no_frame_number),
    Check("X10", "代码结构 §尚未落地（bench/feed.py 把一批帧经重放适配器吐出来）",
          "replay_frames 真的经过重放适配器",
          check_feed_goes_through_the_replay_adapter),
    Check("X11", "采集与验收/基准集 §2（必填 · 必须落在画布内 · 批内唯一 · 不许补默认值）",
          "坏清单在加载时就拒绝，不许就地修补",
          check_a_broken_manifest_is_refused_not_patched),
    Check("X12", "采集与验收/基准集 §2（前提头在清单里只记画布装不下的那一项）",
          "清单里的 facts 由加载器折成 DeviceFacts",
          check_the_loader_folds_the_recorded_facts),
    Check("X13", "设备层 §5（前提头是 DeviceFacts 或 None · 一个像素都不算）",
          "设备层不认清单那套键名，构造时也不预读帧源",
          check_the_device_layer_does_not_read_the_manifest_shape),
)


def main() -> int:
    failures: list[tuple[Check, str]] = []
    print(f"T6 补充检查（规划者自用，不交给执行者）—— {len(CHECKS)} 条\n")
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
