"""设备层的真机适配器：取像走 **设备端写文件 + `pull`**（设备层 §4 · D-全局-19）。

这个模块只做一件事：**把「取一帧」变成两条命令**。它不认识画面、也不认识画布 ——
那是视觉层的入口（`engine/vision/frame.py`）；它更不知道动作有没有生效，那是接口层
的事。设备层 §1 那条边界不能糊。

三样东西：

- `parse_wm_size`：**纯函数**，把 `wm size` 的两行按**它自己的口径**解出来。
  设备层 §4 说得很死：`wm size` 与画布差 90°，**两个口径不许混** —— 所以这里
  **不转置**，`Physical size: 1260x2800` 就解成 `Canvas(1260, 2800)`；
- `AdbTransport`：跑一条命令，拿回它的 stdout 字节。**墙钟超时属于这一层**
  （设备层 §5）；
- `AdbDevice`：设备端口的真机适配器。**这一轮只接 `grab_frame`** ——
  `inject_gesture` 与 `device_facts` 留空，理由与去向登记在 `runtime/wiring.py`
  的 `PORT_GAPS` 里（一个是签名不够，一个是「方向从哪来」没有来源，两条都还没
  裁定；自己补一个出来就是**发明规格**）。

## 那个 `run` 的缝

`AdbTransport` 收一个「跑一条命令」的可调用对象，**是为了离线验通道与超时**，
不是为了给端口开一个「我在重放」的开关 —— 契约层 §5 那条线绷得很紧：端口里
不许出现「现在真的是游戏在跑」这类信息。缝的契约只有三条：

    argv       完整参数表（含 `-s <serial>`，**不含可执行文件本身**）
    timeout_ms 这一次命令的超时
    返回       stdout 的原始字节；失败自己抛

不给它时**真跑**（走 `adb` 那个名字，不写绝对路径：换台机器就得改引擎的东西
不许出现在这里）。

## 墙钟读哪把时钟

**要 `perf_counter` 那一类、不要 `monotonic`** —— 规定与实测数在「契约层/刻度 §3」
（T4 定）。这里**只引用，不重述**：同一条规定出现两次，改的时候就会漏掉一处。
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final

from ark.contracts import Canvas

__all__ = [
    "AdbTimeout",
    "WmSize",
    "parse_wm_size",
    "AdbTransport",
    "AdbDevice",
    "DEFAULT_TIMEOUT_MS",
    "DEVICE_TMP_DIR",
    "FRAME_FILE_NAME",
]


DEFAULT_TIMEOUT_MS: Final[int] = 30_000
"""一条命令的默认墙钟上限（ms）。

实测一条命令的地板约 200 ms、取像往返中位 1.02–1.07 s（设备层 §3），所以 30 s
是「明显坏了」那条线，不是「慢」那条线。它的用处是**不许无限等**，不是掐着实测值卡。
"""

DEVICE_TMP_DIR: Final[str] = "/data/local/tmp"
"""设备端那个文件的落点。

`shell` 用户在这里有写权限、不需要 root，而且它**不是媒体库**（`/sdcard` 上的
东西会被媒体扫描器看见）。

**它是一个固定槽位，不每帧删**（设备层 §4 · T4 定）：同名写入直接覆盖、不累积，
而删它要多跑一条命令、约 200 ms（§3 的地板），占往返的两成。**本地那一份才每帧删**
（本机文件，不花命令）。
"""

FRAME_FILE_NAME: Final[str] = "ark_frame.raw"
"""设备端与本地两侧共用的文件名。原始帧定长 14,112,016 字节，`.raw` 是它自己。"""


class AdbTimeout(Exception):
    """**命令没在墙钟内返回**（设备层 §5）。带它自己的墙钟耗时。

    它**不是「中止」**：`Abort.frame` 是**逻辑帧**，而设备层拿不到逻辑帧 ——
    帧号由费用推算（契约层 §3），不是从时钟读来的。所以「命令没返回」在这一层的
    形状是**它自己那条异常**，带的也是这一层真正有的那个量：**墙钟毫秒**。
    把它折成中止是调用方的事（只有调用方知道跑到哪一帧）。
    """

    def __init__(self, elapsed_ms: int, detail: str | None = None) -> None:
        self.elapsed_ms = max(1, int(elapsed_ms))
        text = f"命令没在墙钟内返回（耗时 {self.elapsed_ms} ms）"
        super().__init__(f"{text}：{detail}" if detail else text)


@final
@dataclass(frozen=True)
class WmSize:
    """`wm size` 的输出，**按它自己的口径**。

    `physical` 是 `Physical size` 那一行；`override` 是有 `Override size`
    那一行时的它，否则 `None`（设备层 §4：设过之后 `wm size` 会报两行，而**前提头
    校验要读 `Override size`** —— 有它的时候 `Physical size` 是物理屏、不是当前
    画布，读错了会在降档之后仍以为画布是 2800×1260）。

    **两个字段都不转置。** 这台机器报 `1260x2800`（竖屏自然朝向），画布是
    `2800x1260`；转置要方向，而方向从哪来还没定（设备层「尚未定」）。
    """

    physical: Canvas
    override: Canvas | None

    def __post_init__(self) -> None:
        if not isinstance(self.physical, Canvas):
            raise TypeError(
                f"Physical size 要解成画布，收到 {type(self.physical).__name__}"
            )
        if self.override is not None and not isinstance(self.override, Canvas):
            raise TypeError(
                f"Override size 要解成画布或没有，收到 {type(self.override).__name__}"
            )


_SIZE_LINE: Final[re.Pattern[str]] = re.compile(
    r"(Physical|Override)\s+size\s*:\s*(\d+)\s*[xX]\s*(\d+)"
)
"""只认实测过的那两行。`Physical density` / `Override density` 里没有 `size`，落不进来。"""


def parse_wm_size(text: str) -> WmSize:
    """把 `wm size` 的输出解成 `WmSize`。**纯函数，不碰设备。**

        Physical size: 1260x2800   → Canvas(1260, 2800)
        Override size: 882x1960    → Canvas(882, 1960)

    **不转置**（设备层 §4）：想要 2800×1260 就自己去写 `1260x2800` —— 转置要方向，
    而方向来源还没定。两行都在时 `override` 非空。

    **没有 `Physical size` 这一行就报错**：它是基础那一行，缺了它说明这份输出
    根本不是 `wm size` 的输出（多半是别的命令的错误信息），按 RGBA 那样猜着解
    就是假事实。
    """
    if not isinstance(text, str):
        raise TypeError(f"wm size 的输出要是字符串，收到 {type(text).__name__}")
    physical: Canvas | None = None
    override: Canvas | None = None
    for kind, width, height in _SIZE_LINE.findall(text):
        if kind == "Physical":
            if physical is None:
                physical = Canvas(width=int(width), height=int(height))
        elif override is None:
            override = Canvas(width=int(width), height=int(height))
    if physical is None:
        raise ValueError(
            f"wm size 的输出里没有 `Physical size: 宽x高` 那一行，收到 {text!r}"
        )
    return WmSize(physical=physical, override=override)


class AdbTransport:
    """跑一条命令，拿回它的 stdout 字节。**墙钟超时属于这里**（设备层 §5）。

    两个参数值得说清：

    - `run` 是那个**缝**（见模块 docstring）：给了它就走它，不给就真跑 `adb`；
    - `work_dir` 是**本地临时文件的落点**。默认用**平台临时目录本身**，不用
      `tempfile.mkdtemp()`：本机沙箱下 `mkdtemp()` 建出来的目录连建它的进程
      自己都进不去（它用 `os.mkdir(path, 0o700)`，受限令牌下那份 DACL 不继承父
      目录）。**本地那份取完就删**（见 `AdbDevice.grab_frame`）。

    **它不知道自己在跟什么说话**：这条类里没有「现在真的是游戏在跑」这类信息
    （契约层 §5），所以换一个 `run` 就是换一条通道。
    """

    def __init__(
        self,
        *,
        adb: str = "adb",
        serial: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        work_dir: Path | None = None,
        run: Callable[[Sequence[str], int], bytes] | None = None,
    ) -> None:
        if not isinstance(adb, str) or not adb:
            raise TypeError(f"adb 的名字是非空字符串，收到 {adb!r}")
        if serial is not None and (not isinstance(serial, str) or not serial):
            raise TypeError(f"设备序列号要么不给，要么是非空字符串，收到 {serial!r}")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int):
            raise TypeError(f"超时是整数毫秒，收到 {type(timeout_ms).__name__}")
        if run is not None and not callable(run):
            raise TypeError(f"那条缝要是可调用的，收到 {type(run).__name__}")
        self.adb = adb
        self.serial = serial
        self.timeout_ms = timeout_ms
        self.work_dir = (
            Path(work_dir) if work_dir is not None else Path(tempfile.gettempdir())
        )
        self._run = run

    def argv_for(self, args: Sequence[str]) -> list[str]:
        """补上 `-s <serial>` 那一段，得到**完整**的参数表。

        **不含可执行文件本身**（缝的契约）：那是「怎么找到它」的事，归这条缝自己。
        不给 serial 时不加 `-s` —— 那表示「让它自己挑唯一那台」，而 `-s None`
        是一条错的命令。
        """
        head = ["-s", self.serial] if self.serial else []
        return head + [str(arg) for arg in args]

    def run_command(self, args: Sequence[str], timeout_ms: int | None = None) -> bytes:
        """跑一条命令，返回 stdout 的原始字节。**超了就报超时，不无限等。**

        超时那两条路都堵上：① **命令回来了但太晚** —— 缝不给中断能力时，唯一能做
        的就是量墙钟；② **命令自己抛了超时** —— `subprocess` 那条路会抛
        `TimeoutExpired`。两条都落成 `AdbTimeout`，带正的 `elapsed_ms`。
        """
        limit = self.timeout_ms if timeout_ms is None else timeout_ms
        argv = self.argv_for(args)
        started = time.perf_counter_ns()
        try:
            if self._run is None:
                data = self._spawn(argv, limit)
            else:
                data = self._run(argv, limit)
        except (subprocess.TimeoutExpired, TimeoutError) as expired:
            raise AdbTimeout(self._elapsed_ms(started), str(expired)) from expired
        elapsed_ms = self._elapsed_ms(started)
        if limit > 0 and elapsed_ms > limit:
            raise AdbTimeout(elapsed_ms)
        return data

    def _spawn(self, argv: Sequence[str], limit: int) -> bytes:
        """真跑一条命令。**可执行文件的名字只在这里被加上。**"""
        command = [self.adb, *argv]
        try:
            finished = subprocess.run(
                command,
                capture_output=True,
                timeout=(limit / 1000.0) if limit > 0 else None,
                check=False,
            )
        except FileNotFoundError as missing:
            raise RuntimeError(
                f"找不到 adb 可执行文件 {self.adb!r}（设备层 §6 的前置）：{missing}"
            ) from missing
        if finished.returncode != 0:
            tail = finished.stderr.decode("utf-8", "replace").strip()[:400]
            raise RuntimeError(
                f"命令返回 {finished.returncode}：{list(command)} ｜ {tail}"
            )
        return bytes(finished.stdout)

    @staticmethod
    def _elapsed_ms(started_ns: int) -> int:
        """墙钟耗时（ms，至少 1 —— 「没在墙钟内返回」不可能是 0 ms）。"""
        return max(1, (time.perf_counter_ns() - started_ns) // 1_000_000)


class AdbDevice:
    """设备端口的真机适配器：**这一轮只有取像那一条**。

    `grab_frame` 的形状是「一条 `shell` 写设备端文件 + 一条 `pull` 取回」，
    **不走 `exec-out` 直出**（设备层 §4 · D-全局-19）：原始帧 14 MB 走流式通道比
    取文件慢，而且 PowerShell 的 `>` 会毁掉二进制。

    `inject_gesture` 与 `device_facts` **不在这里**：它们的形状还没裁定，自己补
    一个出来就是发明规格。欠账登记在 `runtime/wiring.py` 的 `PORT_GAPS`。
    """

    def __init__(self, transport: AdbTransport) -> None:
        self._transport = transport

    def grab_frame(self) -> bytes:
        """取一帧，返回**原始帧**（不是归一化帧，设备层 §4）。

        **两条命令，不许有第三条**（设备层 §4 · T4 定）：

        1. 一条 `shell`：**先删旧文件、再 `screencap` 写新的**（原始 RGBA，
           **不带 `-p`**）—— 两步并进同一次调用；
        2. 一条 `pull`：把那一份取回本地。

        设备端那个文件是**固定槽位**：同名写入直接覆盖、不累积，所以**它不每帧删**
        —— 删它要多跑一条命令、约 200 ms（§3 的地板），占往返的两成。
        **本地那一份才每帧删**（本机文件，不花命令）。

        **本地那份读不出来就是取帧坏了**：不许退回 `pull` 的 stdout —— 真 `adb pull`
        的 stdout 是**进度文本**，把它当帧返回就是**假事实的入口**。
        """
        transport = self._transport
        remote = self._remote_path()
        local = self._local_path()
        parent = local.parent
        if not parent.is_dir():
            parent.mkdir(parents=True, exist_ok=True)
        # 先清掉上一次留下的**本地**那份：陈帧会被当成这一帧读出来。
        self._drop_local(local)
        transport.run_command(["shell", "rm", "-f", remote, "&&", "screencap", remote])
        try:
            transport.run_command(["pull", remote, str(local)])
            if not local.is_file():
                raise RuntimeError(
                    f"取回之后本地没有那一份帧（{local}）：取帧坏了。"
                    "不许把取回命令的 stdout 当帧 —— 那是进度文本，不是画面"
                )
            raw = local.read_bytes()
        finally:
            self._drop_local(local)
        return raw

    def _remote_path(self) -> str:
        """设备端那个槽位的路径。固定名字：只有「取一帧」，没有取像流。"""
        return f"{DEVICE_TMP_DIR}/{FRAME_FILE_NAME}"

    def _local_path(self) -> Path:
        """本地那一份的路径 —— 落点由 `transport.work_dir` 给。"""
        return Path(self._transport.work_dir) / FRAME_FILE_NAME

    def _drop_local(self, local: Path) -> None:
        """删本地那份。**删不掉不是故障**（那只是一份垃圾），不许盖掉真正的错。"""
        try:
            Path(local).unlink(missing_ok=True)
        except OSError:
            pass
