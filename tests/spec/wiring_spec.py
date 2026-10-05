"""接线与端口的规格符合性验收 —— **由规划者维护，不住在 tests/unit/**。

分工、纪律、跑法都在 [README](README.md) —— **不在这里重述**（同一条规定只写
一次）。T5 已经做完了，这批现在全绿。

## 这一批与别的验收的分界

- **不重复 `contracts_spec.py` 的活**：端口与错误的**类型**归那份；这里只问
  "**装配那一刻，谁接上了**"；
- **不重复 `import_rules.py` 的活**：import 关系归那份；这里问的是**行为**；
- **不重复 `port_completeness.py` 的活**：那份（执行者写，住 `tests/contract/`）
  是端口完备性的机器强制版；这里只确认**它存在、四个端口都查**。
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import itertools
import os
import re
import shutil
import struct
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ark.contracts import LogicalFrame, WallClockMs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
ENGINE_SRC = REPO / "src" / "ark"


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


# ---------------------------------------------------------------------------
# 懒加载：模块还不存在时要报成"红的"，不是"检查炸了"
# ---------------------------------------------------------------------------

_LOADED: dict[str, object] = {}


def _module(dotted: str):
    """import 一个还不一定存在的模块。不存在 = 这条检查红。"""
    if dotted not in _LOADED:
        try:
            _LOADED[dotted] = importlib.import_module(dotted)
        except ImportError as exc:
            raise AssertionError(
                f"`{dotted}` 还不存在：{type(exc).__name__}: {exc}"
            ) from None
    return _LOADED[dotted]


def _adb():
    return _module("ark.engine.device.adb")


def _wiring():
    return _module("ark.runtime.wiring")


_TMP_SEQ = itertools.count()


@contextlib.contextmanager
def _writable_tmp():
    """建一个**能进得去**的临时目录。

    本机沙箱下 `tempfile.mkdtemp()` 建出来的目录**连建它的进程自己都进不去**：
    它用 `os.mkdir(path, 0o700)`，而受限令牌下那份 DACL 不继承父目录。
    这是**本机环境的事，不是被测实现的事** —— 所以在这里绕开。
    """
    base = REPO / "data" / "_spec_tmp"
    base.mkdir(parents=True, exist_ok=True)
    dest = base / f"wiring{os.getpid()}_{next(_TMP_SEQ)}"
    dest.mkdir()
    try:
        yield dest
    finally:
        shutil.rmtree(dest, ignore_errors=True)


def _synthetic_raw(width: int, height: int) -> bytes:
    """造一帧原始字节（设备层 §4 的布局），喂假 transport 用。

    只做头 + 常量像素：这里验的是**通道**，不是解帧（解帧归 cost_spec 的 F 族）。
    """
    header = struct.pack("<4I", width, height, 1, 1)
    return header + bytes([7, 8, 9, 255]) * (width * height)


def _fake_transport(adb, raw: bytes, seen: list[tuple[str, ...]] | None = None,
                    scratch: Path | None = None):
    """一个假 transport：不碰真机，把 adb 参数表记下来。"""

    def run(argv: Sequence[str], timeout_ms: int) -> bytes:
        argv = tuple(argv)
        if seen is not None:
            seen.append(argv)
        if "pull" in argv and scratch is not None:
            Path(argv[-1]).write_bytes(raw)
            return b""
        return raw

    return adb.AdbTransport(run=run, timeout_ms=2000, work_dir=scratch)


# ---------------------------------------------------------------------------
# W 族：装配
# ---------------------------------------------------------------------------


def check_wiring_supplies_all_four_ports() -> str:
    """四个端口在装配那一刻各有一个实现者，缺一个就是欠账。"""
    wiring = _wiring()
    adb = _adb()
    with _writable_tmp() as scratch:
        built = wiring.build_wiring(_fake_transport(adb, _synthetic_raw(8, 4), scratch=scratch))
    missing = [
        name
        for name in ("device", "clock", "samples", "assets")
        if getattr(built, name, None) is None
    ]
    assert not missing, f"装配出来的东西缺了 {missing}：四个端口一个都不许缺"
    assert callable(getattr(built.device, "grab_frame", None)), "设备端口没有取帧"
    assert callable(getattr(built.clock, "wall_clock_ms", None)), "时钟端口没有墙钟"
    assert callable(getattr(built.samples, "submit", None)), "样本端口没有落样本"
    return "四个端口都接上了：device / clock / samples / assets"


def check_device_port_is_fully_wired() -> str:
    """设备端口的方法集齐了：取帧 · 设备事实 · 轴上的四个动词。"""
    wiring = _wiring()
    adb = _adb()
    with _writable_tmp() as scratch:
        device = wiring.build_wiring(
            _fake_transport(adb, _synthetic_raw(8, 4), scratch=scratch)
        ).device
    wanted = (
        "grab_frame",
        "device_facts",
        "deploy",
        "retreat",
        "skill_on",
        "skill_off",
    )
    missing = [name for name in wanted if not callable(getattr(device, name, None))]
    assert not missing, (
        f"设备端口还缺 {missing}：轴上有几个动词，端口上就该有几个入口"
        "（设备层 §1）—— 缺的那些要么接上，要么写进 PORT_GAPS"
    )
    gaps = wiring.PORT_GAPS
    assert isinstance(gaps, tuple), f"PORT_GAPS 要是元组，收到 {type(gaps).__name__}"
    leftover = [gap for gap in gaps if "DevicePort" in gap]
    assert not leftover, (
        f"设备端口已经接齐了，PORT_GAPS 里不该还有它的欠账：{leftover}"
    )
    return f"六个方法都在；PORT_GAPS 共 {len(gaps)} 条（设备端口 0 条）"


def check_clock_is_wall_clock_and_monotonic() -> str:
    """时钟只给墙钟毫秒，且不倒退。"""
    clock_module = _module("ark.engine.device.clock")
    clock = clock_module.SystemClock()
    first = clock.wall_clock_ms()
    time.sleep(0.01)
    second = clock.wall_clock_ms()
    assert isinstance(first, WallClockMs), (
        f"时钟返回了 {type(first).__name__}：逻辑帧不在这个端口里"
    )
    assert isinstance(second, WallClockMs), f"时钟返回了 {type(second).__name__}"
    assert not isinstance(first, LogicalFrame), "时钟返回了逻辑帧"
    assert second >= first, f"墙钟倒退了：{int(first)} → {int(second)}"
    return f"两次读数 {int(first)} → {int(second)} ms，都是 WallClockMs"


def check_sample_port_empty_impl_does_nothing() -> str:
    """样本端口的空实现什么都不做（离线验收不需要采集）。"""
    port_module = _module("ark.offline.hardcase.port")
    port = port_module.EmptySamplePort()
    result = port.submit(object())
    assert result is None, f"空实现的 submit 返回了 {result!r}"
    return "submit 收下一份不透明的样本，什么都不做"


def check_asset_port_empty_impl_does_not_invent_a_projection() -> str:
    """资产端口的空实现**不许编出一个投影**（端口与错误 §5 的补法规则 + 零假事实）。

    端口现在有第一个方法 `projection`（补法规则：每个方法由它的第一个用户层补），
    而这一轮**没有真的投影值** —— 标定是首轮探索编辑器的第一块。空实现只有两条
    诚实出路：**没有这个方法**（那就登记进 `PORT_GAPS`）、或者**调用就报错**。

    **返回一个编出来的 `Projection` 是最坏的一条**：设备层会照着它算像素、
    把手指放到一个没人验过的地方 —— 那是假事实，比"没接"坏得多。
    """
    port_module = _module("ark.offline.assets_gen.port")
    impl = port_module.EmptyAssetPort()
    method = getattr(impl, "projection", None)
    if method is None:
        return "空实现没有 projection：这一轮投影还没有来源，欠账要登记进 PORT_GAPS"
    from ark.contracts import View

    try:
        got = method(View.NORMAL)
    except BaseException as bad:  # noqa: BLE001 —— 报错就是那条诚实出路
        return f"空实现的 projection 调用就报错（{type(bad).__name__}）—— 没编出投影"
    raise AssertionError(
        f"空实现的 projection 返回了 {got!r}：这一轮没有真的投影值，"
        "返回一个编出来的投影会让设备层照着它去点，那是假事实"
    )


def check_missing_projection_never_becomes_a_click() -> str:
    """拿不到投影时，四个动词**一条命令都不许发**（设备层 §1 + 零假事实）。

    设备层 §1：像素在这个入口**内部**由投影算出。投影拿不到，就等于"不知道该把
    手指放到哪" —— 这时候**发一条 `input` 就是猜**。所以判据不是"报不报错"，
    而是"**有没有往设备上发东西**"：报错的方式可以变，手指不许乱动。
    """
    from ark.contracts import GridCoord

    wiring = _wiring()
    adb = _adb()
    seen: list[tuple[str, ...]] = []
    with _writable_tmp() as scratch:
        device = wiring.build_wiring(
            _fake_transport(adb, _synthetic_raw(8, 4), seen=seen, scratch=scratch)
        ).device
        for what, call in (
            ("deploy", lambda: device.deploy(0, GridCoord(1, 1), "右")),
            ("retreat", lambda: device.retreat(GridCoord(1, 1))),
            ("skill_on", lambda: device.skill_on(GridCoord(1, 1))),
            ("skill_off", lambda: device.skill_off(GridCoord(1, 1))),
        ):
            assert callable(getattr(device, what, None)), (
                f"{what} 还不存在 —— 方法没有的时候这条检查不算过（那是 W2 的事）"
            )
            before = len(seen)
            try:
                call()
            except BaseException:  # noqa: BLE001 —— 报错是允许的出路
                pass
            sent = seen[before:]
            assert not sent, (
                f"{what} 在拿不到投影时往设备上发了 {[' '.join(a) for a in sent]}："
                "不知道该点哪就不许点，猜出来的坐标是假事实"
            )
    return "四个动词在缺投影时一条命令都没发"


_ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]")


def check_no_local_paths_or_serials_in_sources() -> str:
    """引擎里不许出现本机的绝对路径与设备序列号。"""
    needles = ("platform-tools", "10AE6D1UM4002ZT")
    hits: list[str] = []
    scanned = sorted(ENGINE_SRC.rglob("*.py"))
    for path in scanned:
        text = path.read_text(encoding="utf-8-sig")
        where = path.relative_to(REPO)
        for needle in needles:
            if needle in text:
                hits.append(f"{where} 里有 {needle!r}")
        if _ABSOLUTE_PATH.search(text):
            hits.append(f"{where} 里有本机绝对路径")
    assert not hits, "；".join(hits) + "：换台机器就得改引擎"
    return f"扫了 {len(scanned)} 个模块，本机路径与序列号一个都没有"


# ---------------------------------------------------------------------------
# W 族：设备适配器
# ---------------------------------------------------------------------------


def check_wm_size_is_read_in_its_own_units() -> str:
    """`wm size` 两行都按它自己的口径解，有 `Override size` 时它必须在。"""
    adb = _adb()
    one = adb.parse_wm_size("Physical size: 1260x2800\n")
    two = adb.parse_wm_size("Physical size: 1260x2800\nOverride size: 882x1960\n")
    assert (one.physical.width, one.physical.height) == (1260, 2800), (
        f"物理那一行解成了 {one.physical}：`wm size` 是竖屏自然朝向，"
        "不许在这一步转置（转置要方向，方向来源还没定）"
    )
    assert one.override is None, f"只有一行却解出了 override={one.override}"
    assert two.override is not None, (
        "设过之后有两行，前提头校验要读 `Override size` —— 读 Physical 会以为"
        "画布还是 2800×1260"
    )
    assert (two.override.width, two.override.height) == (882, 1960), (
        f"覆盖那一行解成了 {two.override}"
    )
    return f"物理 {one.physical.width}×{one.physical.height}；覆盖 {two.override.width}×{two.override.height}"


def check_frame_is_taken_by_screencap_plus_pull() -> str:
    """取一帧走 `screencap`（不带 `-p`）+ `pull`，不走 `exec-out` 直出。

    **走装配点拿设备**（T5 修）：原来这里直接 `AdbDevice(transport)` 单参数构造，
    而设备层 §5 已经定了适配器要收三样（传输 · 资产端口 · 此刻的视图）—— 验收里
    那条单参数写法**逼着实现给后两样留默认值**，于是"没接资产端口的适配器"在类型上
    说得通了。规格检查不许反过来削规格（同一条教训：W8 曾经逼出一个 stdout 回退）。
    """
    wiring = _wiring()
    adb = _adb()
    raw = _synthetic_raw(8, 4)
    seen: list[tuple[str, ...]] = []
    with _writable_tmp() as scratch:
        transport = _fake_transport(adb, raw, seen=seen, scratch=scratch)
        got = wiring.build_wiring(transport).device.grab_frame()
    flat = [" ".join(argv) for argv in seen]
    assert any("screencap" in line for line in flat), f"没跑 screencap：{flat}"
    assert any("pull" in line for line in flat), f"没走 pull：{flat}"
    assert not any("exec-out" in line for line in flat), (
        f"用了 exec-out 直出：{flat} —— 原始帧 14 MB 走流式通道比 pull 慢"
    )
    assert not any("-p" in argv for argv in seen), (
        f"screencap 带了 -p（那是 PNG）：{flat}"
    )
    assert got == raw, f"取回的字节不是设备端那份：{len(got)} vs {len(raw)}"
    return f"{len(seen)} 条命令：{flat}；取回 {len(got)} 字节"


def check_timeout_is_reported_not_waited() -> str:
    """命令没返回要报超时，不是无限等。"""
    wiring = _wiring()
    adb = _adb()

    def slow(argv: Sequence[str], timeout_ms: int) -> bytes:
        time.sleep(0.25)
        return b""

    with _writable_tmp() as scratch:
        transport = adb.AdbTransport(run=slow, timeout_ms=50, work_dir=scratch)
        try:
            wiring.build_wiring(transport).device.grab_frame()
        except adb.AdbTimeout as caught:
            elapsed = getattr(caught, "elapsed_ms", None)
            assert isinstance(elapsed, int) and elapsed > 0, (
                f"AdbTimeout 没带正的 elapsed_ms，收到 {elapsed!r}"
            )
            return f"AdbTimeout(elapsed_ms={elapsed})"
        except BaseException as other:  # noqa: BLE001 —— 报清楚"炸错了类型"
            raise AssertionError(
                f"期望 AdbTimeout，实际 {type(other).__name__}: {other}"
            ) from None
    raise AssertionError("命令跑了 250 ms、超时设的是 50 ms，却没报超时")


# ---------------------------------------------------------------------------
# W 族：端到端与端口完备性
# ---------------------------------------------------------------------------


def check_device_adapter_requires_its_three_parts() -> str:
    """设备适配器构造时三样都要给全：传输 · 资产端口 · 此刻的视图（设备层 §5）。

    **不许给后两样留默认值**：一个"没接资产端口"的适配器在类型上说得通，但它其实
    **只能取帧与查设备事实** —— 那种对象会让"设备端口接上了"变成一句半真话（与
    [代码结构] §尚未落地 那条"留空 = 方法不存在"同一个道理）。缺投影时四个动词
    本来就不许发命令（W12），所以它没有存在的理由。

    **这条是 T5 补的**：原来 W8 / W9 直接 `AdbDevice(transport)` 单参数构造，
    逼着实现给后两样留默认值 —— 规格检查反过来削规格，同一个坑踩过两次
    （上一次是 `grab_frame` 的 stdout 回退）。
    """
    adb = _adb()
    params = inspect.signature(adb.AdbDevice.__init__).parameters
    for name in ("transport", "assets", "view"):
        assert name in params, (
            f"设备适配器的构造参数里没有 {name}：它要收三样（设备层 §5），"
            f"现在只有 {sorted(params)}"
        )
    optional = [
        name
        for name in ("assets", "view")
        if params[name].default is not inspect.Parameter.empty
    ]
    assert not optional, (
        f"{optional} 有默认值：一个没接资产端口的适配器只能取帧与查设备事实，"
        "而它会让「设备端口接上了」变成一句半真话 —— 缺投影时四个动词本来就不许发命令"
    )
    return f"三样都必填：{sorted(params)}"


def check_e2e_script_goes_through_the_ports() -> str:
    """端到端脚本走端口，不自己拼 adb 命令。"""
    script = REPO / "scripts" / "e2e_device_read.py"
    assert script.is_file(), f"{script.relative_to(REPO)} 不存在"
    text = script.read_text(encoding="utf-8-sig")
    assert "screencap" not in text, (
        "端到端脚本自己拼了 adb 命令 —— 它必须走端口：离线重放不是特例代码，"
        "落地条件就是换一个适配器"
    )
    assert "wiring" in text, "端到端脚本没走 runtime/wiring"
    return "脚本走 wiring，不自己拼 adb 命令"


def check_port_completeness_test_exists_and_covers_four() -> str:
    """端口完备性测试住在 tests/contract/，四个端口都查。"""
    path = REPO / "tests" / "contract" / "port_completeness.py"
    assert path.is_file(), f"{path.relative_to(REPO)} 不存在"
    text = path.read_text(encoding="utf-8-sig")
    missing = [
        name
        for name in ("DevicePort", "ClockPort", "SamplePort", "AssetPort")
        if name not in text
    ]
    assert not missing, f"端口完备性测试没查 {missing}"
    return "四个端口都查了"


CHECKS: tuple[Check, ...] = (
    Check("W1", "代码结构 §尚未落地（端口完备性测试）",
          "四个端口在装配那一刻各有一个实现者，一个不缺",
          check_wiring_supplies_all_four_ports),
    Check("W2", "设备层 §1（对外面就是轴的动词表）+ 代码结构 §尚未落地",
          "设备端口的方法集齐了：取帧 · 设备事实 · 轴上的四个动词",
          check_device_port_is_fully_wired),
    Check("W3", "契约层 §3 + 端口与错误 §5",
          "时钟只给墙钟毫秒，且不倒退", check_clock_is_wall_clock_and_monotonic),
    Check("W4", "端口与错误 §5 + 采集与验收/README §1",
          "样本端口的空实现什么都不做", check_sample_port_empty_impl_does_nothing),
    Check("W5", "端口与错误 §5（资产：补法规则）+ 零假事实",
          "资产端口的空实现不许编出一个投影",
          check_asset_port_empty_impl_does_not_invent_a_projection),
    Check("W6", "设备层 §6（前置）+ 代码结构 规则一",
          "引擎里不许出现本机的绝对路径与设备序列号",
          check_no_local_paths_or_serials_in_sources),
    Check("W7", "设备层 §4（两个口径不许混）",
          "`wm size` 两行都按自己的口径解，有 Override 时它在",
          check_wm_size_is_read_in_its_own_units),
    Check("W8", "设备层 §4 + D-全局-19",
          "取帧走 screencap（不带 -p）+ pull，不走 exec-out",
          check_frame_is_taken_by_screencap_plus_pull),
    Check("W9", "设备层 §5（墙钟超时属于这一层）",
          "命令没返回要报超时，不是无限等", check_timeout_is_reported_not_waited),
    Check("W10", "代码结构 规则二 + 设备层 §1",
          "端到端脚本走端口，不自己拼 adb 命令",
          check_e2e_script_goes_through_the_ports),
    Check("W11", "代码结构 §尚未落地（端口完备性测试）",
          "端口完备性测试在 tests/contract/，四个端口都查",
          check_port_completeness_test_exists_and_covers_four),
    Check("W12", "设备层 §1（像素在入口内部由投影算出）+ 零假事实",
          "拿不到投影时，四个动词一条命令都不许发",
          check_missing_projection_never_becomes_a_click),
    Check("W13", "设备层 §5（适配器收三样）",
          "设备适配器构造时传输 · 资产端口 · 此刻的视图都要给全",
          check_device_adapter_requires_its_three_parts),
)


def main() -> int:
    failures: list[tuple[Check, str]] = []
    print(f"接线与端口规格符合性验收 —— {len(CHECKS)} 条\n")
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
