"""设备适配器：`wm size` 的口径 · 超时 · 取帧的通道形状。

**一条真机命令都不发**：取帧那几条走 `AdbTransport` 的 `run` 缝喂假数据 ——
那个缝存在的理由就是让这件事能离线做。
"""

from __future__ import annotations

import dataclasses
import tempfile
import time
from pathlib import Path

import pytest

from ark.contracts import Canvas, GridCoord, Pixel, Projection, UI_LAYOUT, View
from ark.engine.device.adb import (
    DEFAULT_TIMEOUT_MS,
    DEVICE_TMP_DIR,
    FACING_VECTORS,
    FRAME_FILE_NAME,
    PANEL_RETREAT_AT,
    PANEL_SKILL_AT,
    PANEL_WAIT_MS,
    R_GRID,
    SQUAD_CARD_PITCH,
    SWIPE_DURATION_MS,
    AdbDevice,
    AdbTimeout,
    AdbTransport,
    WmSize,
    parse_dumpsys_input,
    parse_wm_size,
)

SCRATCH = Path(__file__).resolve().parents[2] / "data" / "_unit_tmp" / "device_adb"
"""本地临时落点：工作区里的 `data/`（gitignored 的运行时数据）。

**不许用 pytest 的 `tmp_path`**：它建在平台临时区，而本机沙箱下那个目录连建它的
进程自己都进不去（`PermissionError: [WinError 5]`）。落点放进工作区就没事 ——
验收脚本（`tests/spec/wiring_spec.py`）用的也是同一招。
"""


@pytest.fixture()
def work_dir() -> Path:
    """一个干净的本地落点：每次进来先把上一次留下的文件清掉。"""
    if SCRATCH.exists():
        for child in SCRATCH.iterdir():
            if child.is_file():
                child.unlink()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    yield SCRATCH
    for child in SCRATCH.iterdir():
        if child.is_file():
            child.unlink()


class Recorder:
    """假的「跑一条命令」：记下每条参数表，按命令给回数据。

    `write_local` 模拟「真的落盘了」那条路（取回时把文件写到本地），
    `delay_ms` 模拟「命令回来了但太晚」。
    """

    def __init__(
        self,
        frame: bytes = b"\x01\x02\x03\x04",
        *,
        delay_ms: int = 0,
        write_local: bool = False,
        stdout: bytes = b"",
    ) -> None:
        self.frame = frame
        self.delay_ms = delay_ms
        self.write_local = write_local
        self.stdout = stdout
        self.calls: list[tuple[list[str], int]] = []

    def __call__(self, argv, timeout_ms):
        self.calls.append((list(argv), timeout_ms))
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        if "pull" in argv:
            if self.write_local:
                Path(argv[-1]).write_bytes(self.frame)
            return self.frame
        return self.stdout

    @property
    def flat(self) -> list[str]:
        return [arg for argv, _ in self.calls for arg in argv]


class NoAssets:
    """假的资产端口：**什么都不给**。

    取帧与设备事实**不需要投影**，所以那两条用得上它；四个动词撞上它就是
    「投影还没有来源」那一支（`AssetsWithoutProjection` 是同一件事的另一种写法）。
    """


# ---------------------------------------------------------------------------
# wm size：两个口径，不许混
# ---------------------------------------------------------------------------


def test_parse_wm_size_keeps_its_own_units():
    parsed = parse_wm_size("Physical size: 1260x2800\n")
    assert parsed.physical == Canvas(width=1260, height=2800)
    assert parsed.override is None


def test_parse_wm_size_reads_both_lines_without_transposing():
    parsed = parse_wm_size("Physical size: 1260x2800\nOverride size: 882x1960\n")
    assert parsed.physical == Canvas(width=1260, height=2800)
    assert parsed.override == Canvas(width=882, height=1960)


def test_parse_wm_size_ignores_density_lines():
    parsed = parse_wm_size(
        "Physical size: 1260x2800\nPhysical density: 420\nOverride density: 420\n"
    )
    assert parsed.override is None


def test_parse_wm_size_tolerates_crlf_and_spaces():
    parsed = parse_wm_size("Physical size: 1260 x 2800\r\nOverride size:882x1960\r\n")
    assert parsed.physical == Canvas(width=1260, height=2800)
    assert parsed.override == Canvas(width=882, height=1960)


def test_parse_wm_size_without_physical_line_is_refused():
    with pytest.raises(ValueError):
        parse_wm_size("Override size: 882x1960\n")
    with pytest.raises(ValueError):
        parse_wm_size("error: device offline\n")


def test_parse_wm_size_refuses_a_non_string():
    with pytest.raises(TypeError):
        parse_wm_size(b"Physical size: 1260x2800")  # type: ignore[arg-type]


def test_wm_size_is_frozen_and_typed():
    parsed = parse_wm_size("Physical size: 1260x2800\n")
    assert isinstance(parsed, WmSize)
    with pytest.raises(dataclasses.FrozenInstanceError):
        parsed.override = Canvas(width=1, height=1)  # type: ignore[misc]
    with pytest.raises(TypeError):
        WmSize(physical=(1260, 2800), override=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 超时：报超时，不无限等
# ---------------------------------------------------------------------------


def test_adb_timeout_carries_a_positive_elapsed():
    assert AdbTimeout(0).elapsed_ms >= 1
    assert AdbTimeout(123).elapsed_ms == 123
    assert isinstance(AdbTimeout(5), Exception)


def test_transport_reports_timeout_when_the_command_returns_too_late():
    transport = AdbTransport(run=Recorder(delay_ms=60), timeout_ms=20)
    with pytest.raises(AdbTimeout) as caught:
        transport.run_command(["shell", "true"])
    assert caught.value.elapsed_ms > 0


def test_transport_reports_timeout_when_the_seam_raises_one():
    def slow(argv, timeout_ms):
        raise TimeoutError("假装这条命令没返回")

    transport = AdbTransport(run=slow, timeout_ms=1000)
    with pytest.raises(AdbTimeout):
        transport.run_command(["shell", "true"])


def test_transport_is_quiet_when_the_command_comes_back_in_time():
    transport = AdbTransport(run=Recorder(frame=b"ok"), timeout_ms=1000)
    assert transport.run_command(["shell", "true"]) == b""


# ---------------------------------------------------------------------------
# 参数表：完整、不含可执行文件、序列号只在有的时候有
# ---------------------------------------------------------------------------


def test_argv_carries_the_serial_but_not_the_executable():
    recorder = Recorder()
    transport = AdbTransport(run=recorder, serial="SERIAL-X", timeout_ms=1000)
    transport.run_command(["shell", "true"])
    argv, limit = recorder.calls[0]
    assert argv == ["-s", "SERIAL-X", "shell", "true"]
    assert limit == 1000


def test_argv_without_a_serial_omits_the_dash_s():
    recorder = Recorder()
    AdbTransport(run=recorder, timeout_ms=1000).run_command(["shell", "true"])
    assert recorder.calls[0][0] == ["shell", "true"]


def test_run_command_lets_the_caller_override_the_timeout():
    recorder = Recorder()
    AdbTransport(run=recorder, timeout_ms=1000).run_command(["shell", "true"], 50)
    assert recorder.calls[0][1] == 50


def test_defaults_are_the_documented_ones():
    transport = AdbTransport(run=Recorder())
    assert transport.adb == "adb"
    assert transport.serial is None
    assert transport.timeout_ms == DEFAULT_TIMEOUT_MS == 30_000


def test_default_work_dir_is_the_platform_temp_dir_not_a_fresh_one():
    transport = AdbTransport(run=Recorder())
    assert transport.work_dir == Path(tempfile.gettempdir())
    assert transport.work_dir.is_dir()


# ---------------------------------------------------------------------------
# 取帧：设备端写文件 + 取回，两侧都删
# ---------------------------------------------------------------------------


def test_grab_frame_goes_through_one_shell_and_one_pull(work_dir):
    frame = bytes(range(64))
    recorder = Recorder(frame=frame, write_local=True)
    transport = AdbTransport(run=recorder, serial="SERIAL-X", work_dir=work_dir)
    raw = AdbDevice(transport, NoAssets(), View.NORMAL).grab_frame()
    assert raw == frame
    remote = f"{DEVICE_TMP_DIR}/{FRAME_FILE_NAME}"
    commands = [argv for argv, _ in recorder.calls]
    # **两条命令，不许有第三条**（T4 定）：删旧文件与写新文件并进同一次 shell 调用。
    assert len(commands) == 2
    assert commands[0] == [
        "-s",
        "SERIAL-X",
        "shell",
        "rm",
        "-f",
        remote,
        "&&",
        "screencap",
        remote,
    ]
    assert commands[1] == [
        "-s",
        "SERIAL-X",
        "pull",
        remote,
        str(work_dir / FRAME_FILE_NAME),
    ]
    # 顺序是**先删再写**：写之前那个槽位必须是空的。
    assert commands[0].index("rm") < commands[0].index("screencap")
    flat = recorder.flat
    assert "screencap" in flat
    assert "pull" in flat
    assert "-p" not in flat
    assert "exec-out" not in flat


def test_grab_frame_never_issues_a_standalone_delete(work_dir):
    """取帧的参数表里**不许再出现单独的删除命令**（那是 ~200 ms 的两成）。"""
    recorder = Recorder(frame=b"x", write_local=True)
    AdbDevice(AdbTransport(run=recorder, work_dir=work_dir), NoAssets(),
              View.NORMAL).grab_frame()
    for argv, _ in recorder.calls:
        standalone = argv[0] == "shell" and argv[1] == "rm" and "screencap" not in argv
        assert not standalone


def test_grab_frame_refuses_to_hand_back_the_pull_stdout(work_dir):
    """取回之后本地没有那一份帧 → 报错。**不许退回 stdout**（T4.1 裁定二）。

    真取回命令的 stdout 是**进度文本**（下面这条假的就是照着它编的），把它当帧
    返回就是假事实的入口。这一条专门钉那个分支。
    """
    recorder = Recorder(frame=b"1 file pulled. 0 files skipped.", write_local=False)
    transport = AdbTransport(run=recorder, work_dir=work_dir)
    with pytest.raises(RuntimeError) as caught:
        AdbDevice(transport, NoAssets(), View.NORMAL).grab_frame()
    assert "stdout" in str(caught.value)
    assert not (work_dir / FRAME_FILE_NAME).exists()


def test_grab_frame_leaves_no_local_file_behind(work_dir):
    recorder = Recorder(frame=b"x" * 8, write_local=True)
    transport = AdbTransport(run=recorder, work_dir=work_dir)
    AdbDevice(transport, NoAssets(), View.NORMAL).grab_frame()
    assert not (work_dir / FRAME_FILE_NAME).exists()
    # 设备端那一份**不删**：它是固定槽位，靠同名覆盖（T4 定）—— 所以只有两条命令。
    assert len(recorder.calls) == 2


def test_grab_frame_clears_the_local_slot_before_taking_the_frame(work_dir):
    """本地那份**在发命令之前**就被清掉：陈帧会被当成这一帧读出来。"""
    stale = work_dir / FRAME_FILE_NAME
    stale.write_bytes(b"\xff" * 16)
    seen: list[bool] = []

    def spy(argv, timeout_ms):
        if "screencap" in argv:
            seen.append(stale.is_file())
        return b""

    with pytest.raises(RuntimeError):
        AdbDevice(AdbTransport(run=spy, work_dir=work_dir), NoAssets(),
                  View.NORMAL).grab_frame()
    assert seen == [False]


def test_grab_frame_reports_timeout_instead_of_waiting_forever(work_dir):
    transport = AdbTransport(
        run=Recorder(delay_ms=80), timeout_ms=20, work_dir=work_dir
    )
    with pytest.raises(AdbTimeout):
        AdbDevice(transport, NoAssets(), View.NORMAL).grab_frame()


def test_device_adapter_has_exactly_the_six_methods():
    device = AdbDevice(AdbTransport(run=Recorder()), NoAssets(), View.NORMAL)
    for name in ("grab_frame", "device_facts", "deploy", "retreat", "skill_on", "skill_off"):
        assert callable(getattr(device, name, None)), name
    assert not hasattr(device, "inject_gesture")


# ---------------------------------------------------------------------------
# 设备事实：一条 dumpsys input，画布取 logicalFrame、方向取那个整数
# ---------------------------------------------------------------------------

VIEWPORT_TEXT = (
    "some header\n"
    "  Viewport INTERNAL: displayId=0, uniqueId=local:463, port=131, "
    "orientation=1, logicalFrame=[0, 0, 2800, 1260], "
    "physicalFrame=[0, 0, 2800, 1260], deviceSize=[2800, 1260], isActive=[1]\n"
    "tail\n"
)


def test_parse_dumpsys_input_reads_the_viewport_line():
    """T5 裁定一与二：方向是**设备自己报的那个整数**，画布取 `logicalFrame`。"""
    facts = parse_dumpsys_input(VIEWPORT_TEXT)
    assert facts.resolution == Canvas(width=2800, height=1260)
    assert facts.orientation == "1"
    assert isinstance(facts.orientation, str)


def test_parse_dumpsys_input_keeps_the_orientation_literal():
    for raw, want in (("0", "0"), ("1", "1"), ("2", "2"), ("3", "3")):
        text = f"Viewport INTERNAL: orientation={raw}, logicalFrame=[0, 0, 100, 50]\n"
        assert parse_dumpsys_input(text).orientation == want


def test_parse_dumpsys_input_takes_the_canvas_from_the_frame_not_from_wm_size():
    """两个口径不许混：`wm size` 报 1260x2800，画布是 2800x1260（T5 裁定二）。"""
    facts = parse_dumpsys_input(VIEWPORT_TEXT)
    assert facts.resolution == Canvas(width=2800, height=1260)
    assert parse_wm_size("Physical size: 1260x2800\n").physical == Canvas(1260, 2800)


def test_parse_dumpsys_input_skips_lines_missing_a_field():
    text = (
        "Viewport INTERNAL: displayId=0, orientation=1\n"
        "Viewport EXTERNAL: logicalFrame=[0, 0, 800, 600]\n"
        "Viewport MAIN: orientation=0, logicalFrame=[0, 0, 640, 480]\n"
    )
    facts = parse_dumpsys_input(text)
    assert facts.resolution == Canvas(width=640, height=480)
    assert facts.orientation == "0"


def test_parse_dumpsys_input_refuses_when_there_is_no_viewport():
    with pytest.raises(ValueError):
        parse_dumpsys_input("error: device offline\n")
    with pytest.raises(ValueError):
        parse_dumpsys_input("Viewport INTERNAL: orientation=1\n")


def test_parse_dumpsys_input_refuses_a_non_string():
    with pytest.raises(TypeError):
        parse_dumpsys_input(b"Viewport INTERNAL: orientation=1")  # type: ignore[arg-type]


def test_device_facts_goes_through_one_dumpsys_and_needs_no_projection():
    """设备事实**不发手势**：所以没有投影也照样能查（T5 裁定五只约束四个动词）。"""
    recorder = Recorder(stdout=VIEWPORT_TEXT.encode())
    device = AdbDevice(AdbTransport(run=recorder), NoAssets(), View.NORMAL)
    facts = device.device_facts()
    assert facts.resolution == Canvas(width=2800, height=1260)
    assert facts.orientation == "1"
    assert [argv for argv, _ in recorder.calls] == [["shell", "dumpsys", "input"]]

# ---------------------------------------------------------------------------
# 四个动词：像素一律经投影，手势原语沉在这条类里面
# ---------------------------------------------------------------------------


class FakeAssets:
    """假的资产端口：给一套**线性**投影（格 → 像素 = 原点 + 格 × 100）。

    `refuses` 那支模拟真正的空实现（`EmptyAssetPort` 调用就报错）。
    """

    ORIGIN = (1000.0, 500.0)
    SCALE = 100.0

    def __init__(self, *, refuses: bool = False) -> None:
        self.refuses = refuses
        self.asked: list[View] = []

    def projection(self, view: View) -> Projection:
        self.asked.append(view)
        if self.refuses:
            raise RuntimeError("投影值还没有来源：标定是首轮探索编辑器的第一块")
        ox, oy = self.ORIGIN
        s = self.SCALE
        return Projection(
            view=view,
            grid_to_pixel=lambda gp: Pixel(ox + gp.x * s, oy + gp.y * s),
            pixel_to_grid=lambda px: GridCoord((px.x - ox) / s, (px.y - oy) / s),
        )


def _device(recorder: Recorder, assets: object, view: View = View.NORMAL):
    """三样都给全（设备层 §5 · T5 定）：**没有"省略资产端口"这一形态**。"""
    return AdbDevice(AdbTransport(run=recorder), assets, view)  # type: ignore[arg-type]


def _commands(recorder: Recorder) -> list[list[str]]:
    return [argv for argv, _ in recorder.calls]


def test_deploy_is_one_swipe_from_the_card_to_grid_plus_r():
    """设备层 §2：**一次直线 swipe** —— 起点是第 `card` 张卡，终点是 `格心 + r·û`。"""
    recorder = Recorder()
    device = _device(recorder, FakeAssets())
    device.deploy(0, GridCoord(3, 4), "右")

    # 卡 0 = **最右那张**（T5 定）：中心就是 UiLayout 那个锚点矩形的中心。
    anchor = UI_LAYOUT.rects["squad_card_rightmost"]
    start = (round(anchor.x + anchor.width / 2), round(anchor.y + anchor.height / 2))
    # 终点 = 格心 + 0.3 格向右 → 格 (3.3, 4) → 像素 (1330, 900)
    end = (round(1000 + 3.3 * 100), 500 + 4 * 100)
    assert _commands(recorder) == [
        ["shell", "input", "swipe", str(start[0]), str(start[1]),
         str(end[0]), str(end[1]), str(SWIPE_DURATION_MS)]
    ]
    assert R_GRID == 0.3


def test_deploy_endpoint_encodes_the_facing():
    """同一个终点**同时编码了格与朝向**：四个朝向各落一个方向。"""
    want = {
        "上": (1300, 870),
        "下": (1300, 930),
        "左": (1270, 900),
        "右": (1330, 900),
    }
    assert set(want) == set(FACING_VECTORS)
    for facing, end in want.items():
        recorder = Recorder()
        _device(recorder, FakeAssets()).deploy(0, GridCoord(3, 4), facing)
        argv = _commands(recorder)[0]
        assert (int(argv[-3]), int(argv[-2])) == end, facing


def test_card_index_counts_from_the_rightmost_card():
    """T5 定：**下标 0 = 最右那张**，第 k 张沿卡距左移 k 步（右对齐 ⇒ 锚点是常量）。"""
    starts = []
    for card in range(6):
        recorder = Recorder()
        _device(recorder, FakeAssets()).deploy(card, GridCoord(3, 4), "右")
        argv = _commands(recorder)[0]
        starts.append((int(argv[3]), int(argv[4])))
    # 第 0 张在最右，越往左 x 越小，每步正好一个卡距。
    anchor = UI_LAYOUT.rects["squad_card_rightmost"]
    assert starts[0] == (round(anchor.x + anchor.width / 2),
                         round(anchor.y + anchor.height / 2))
    for k in range(1, 6):
        # 每一步正好一个卡距 —— 逐个取整，所以差可以是 207 或 208。
        step = starts[k - 1][0] - starts[k][0]
        assert abs(step - SQUAD_CARD_PITCH) <= 1, (k, step)
        assert starts[k][1] == starts[0][1]


def test_deploy_refuses_a_negative_card():
    """**负下标报错、不猜**：下标从 0 起（0 = 最右那张），负数没有来源。

    **上限不设**（T5 补丁 2）：卡距量出来是常量（5 张卡与 10 张卡两档同值），
    而锚点与卡数无关 —— 所以第 5、第 9 张照算。
    """
    recorder = Recorder()
    device = _device(recorder, FakeAssets())
    for bad in (-1, -7):
        with pytest.raises(IndexError):
            device.deploy(bad, GridCoord(3, 4), "右")
    assert recorder.calls == []
    device.deploy(9, GridCoord(3, 4), "右")  # 10 张卡那一档也走得通
    assert len(recorder.calls) == 1


def test_deploy_refuses_an_unknown_facing_without_touching_the_device():
    recorder = Recorder()
    device = _device(recorder, FakeAssets())
    for bad in ("斜", "up", "", None):
        with pytest.raises((ValueError, TypeError)):
            device.deploy(0, GridCoord(3, 4), bad)  # type: ignore[arg-type]
    assert recorder.calls == []


def test_deploy_refuses_a_pixel_where_a_grid_belongs():
    recorder = Recorder()
    device = _device(recorder, FakeAssets())
    with pytest.raises(TypeError):
        device.deploy(0, Pixel(3, 4), "右")  # type: ignore[arg-type]
    assert recorder.calls == []


def test_retreat_is_two_taps_with_a_wait_between_them():
    """设备层 §2：**两次点 + 一次等面板** —— 第二下落在干员旁边那个键上。"""
    recorder = Recorder()
    device = _device(recorder, FakeAssets())
    device.retreat(GridCoord(3, 4))
    assert _commands(recorder) == [
        ["shell", "input", "tap", "1300", "900"],
        [
            "shell",
            "input",
            "tap",
            str(round(1000 + (3 + PANEL_RETREAT_AT.x) * 100)),
            str(round(500 + (4 + PANEL_RETREAT_AT.y) * 100)),
        ],
    ]
    assert PANEL_WAIT_MS > 0


def test_skill_on_and_off_tap_the_skill_key():
    for verb in ("skill_on", "skill_off"):
        recorder = Recorder()
        device = _device(recorder, FakeAssets())
        getattr(device, verb)(GridCoord(3, 4))
        assert _commands(recorder)[1] == [
            "shell",
            "input",
            "tap",
            str(round(1000 + (3 + PANEL_SKILL_AT.x) * 100)),
            str(round(500 + (4 + PANEL_SKILL_AT.y) * 100)),
        ]


def test_the_four_verbs_ask_for_the_view_they_were_built_with():
    assets = FakeAssets()
    recorder = Recorder()
    device = _device(recorder, assets, View.SLOW)
    device.retreat(GridCoord(3, 4))
    assert assets.asked == [View.SLOW]


def test_the_four_verbs_send_nothing_when_the_projection_refuses():
    """T5 裁定五：拿不到投影时**一条命令都不许发**。"""
    recorder = Recorder()
    device = _device(recorder, FakeAssets(refuses=True))
    for label, call in (
        ("deploy", lambda: device.deploy(0, GridCoord(1, 1), "右")),
        ("retreat", lambda: device.retreat(GridCoord(1, 1))),
        ("skill_on", lambda: device.skill_on(GridCoord(1, 1))),
        ("skill_off", lambda: device.skill_off(GridCoord(1, 1))),
    ):
        with pytest.raises(RuntimeError):
            call()
        assert recorder.calls == [], label


class AssetsWithoutProjection:
    """假的资产端口：**根本没有 projection 这个方法**（`EmptyAssetPort` 就是这一支）。

    「留空 = 方法不存在」（代码结构 §尚未落地），所以四个动词撞上的是
    `AttributeError` —— 那**不许**直接漏出去（读起来像 bug），要转成一句带话的报错。
    """


def test_the_four_verbs_send_nothing_when_the_method_does_not_exist():
    recorder = Recorder()
    device = _device(recorder, AssetsWithoutProjection())
    for label, call in (
        ("deploy", lambda: device.deploy(0, GridCoord(1, 1), "右")),
        ("retreat", lambda: device.retreat(GridCoord(1, 1))),
        ("skill_on", lambda: device.skill_on(GridCoord(1, 1))),
        ("skill_off", lambda: device.skill_off(GridCoord(1, 1))),
    ):
        with pytest.raises(RuntimeError) as bad:
            call()
        assert "投影还没有来源" in str(bad.value), label
        assert recorder.calls == [], label


def test_the_adapter_refuses_a_view_that_is_not_a_view():
    with pytest.raises(TypeError):
        AdbDevice(AdbTransport(run=Recorder()), FakeAssets(), "常速")  # type: ignore[arg-type]


def test_the_verbs_do_not_go_through_exec_out_or_a_second_adb():
    """手势原语沉在这条类里：发出去的只有 `shell input` 那一种形状。"""
    recorder = Recorder()
    device = _device(recorder, FakeAssets())
    device.deploy(0, GridCoord(3, 4), "右")
    device.retreat(GridCoord(3, 4))
    for argv in _commands(recorder):
        assert argv[:3] == ["shell", "input", "swipe"] or argv[:3] == [
            "shell",
            "input",
            "tap",
        ]
