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

from ark.contracts import Canvas
from ark.engine.device.adb import (
    DEFAULT_TIMEOUT_MS,
    DEVICE_TMP_DIR,
    FRAME_FILE_NAME,
    AdbDevice,
    AdbTimeout,
    AdbTransport,
    WmSize,
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
    ) -> None:
        self.frame = frame
        self.delay_ms = delay_ms
        self.write_local = write_local
        self.calls: list[tuple[list[str], int]] = []

    def __call__(self, argv, timeout_ms):
        self.calls.append((list(argv), timeout_ms))
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        if "pull" in argv:
            if self.write_local:
                Path(argv[-1]).write_bytes(self.frame)
            return self.frame
        return b""

    @property
    def flat(self) -> list[str]:
        return [arg for argv, _ in self.calls for arg in argv]


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
    raw = AdbDevice(transport).grab_frame()
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
    AdbDevice(AdbTransport(run=recorder, work_dir=work_dir)).grab_frame()
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
        AdbDevice(transport).grab_frame()
    assert "stdout" in str(caught.value)
    assert not (work_dir / FRAME_FILE_NAME).exists()


def test_grab_frame_leaves_no_local_file_behind(work_dir):
    recorder = Recorder(frame=b"x" * 8, write_local=True)
    transport = AdbTransport(run=recorder, work_dir=work_dir)
    AdbDevice(transport).grab_frame()
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
        AdbDevice(AdbTransport(run=spy, work_dir=work_dir)).grab_frame()
    assert seen == [False]


def test_grab_frame_reports_timeout_instead_of_waiting_forever(work_dir):
    transport = AdbTransport(
        run=Recorder(delay_ms=80), timeout_ms=20, work_dir=work_dir
    )
    with pytest.raises(AdbTimeout):
        AdbDevice(transport).grab_frame()


def test_device_adapter_has_no_invented_methods():
    device = AdbDevice(AdbTransport(run=Recorder()))
    assert callable(device.grab_frame)
    assert not hasattr(device, "inject_gesture")
    assert not hasattr(device, "device_facts")
