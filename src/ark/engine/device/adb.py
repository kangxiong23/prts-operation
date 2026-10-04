"""设备层的真机适配器：取一帧 + 轴上的四个动词 + 设备事实。

这个模块做三件事，**都在设备层 §1 那条边界之内**：

- **取一帧**：设备端写文件 + `pull`（§4 · D-全局-19）。它不认识画面、也不认识
  画布 —— 那是视觉层的入口（`engine/vision/frame.py`）；
- **轴上的四个动词**（部署 / 撤退 / 开技能 / 关技能）：**手势原语（按 / 拖 / 抬 /
  点 / 键）沉在这条类里面，端口上看不见**（时序层 §4）。参数是**格坐标与朝向**，
  像素在这里由投影算出 —— 所以这条类构造时要拿到「资产端口」与「此刻的视图」
  （T5 裁定）。**切视图 = 重新装配**，不是给端口加一个开关（代码结构 规则二）：
  慢放是另一套投影，而让每个调用方自己换算等于把那条纪律复制到每个调用点（§1）；
- **设备事实**（分辨率 / 方向）：查真机（§5）。两个值都从 `dumpsys input` 里
  那条 `Viewport ...` 行取 —— 它的 `orientation=` 就是方向，它的
  `logicalFrame=` 直接就是**画布口径**（T5 裁定：一个转置都不做）。

**它不知道动作有没有生效** —— 那是接口层的事（接口层/执行与失败 §3）。这条边界
不能糊。

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

## 拿不到投影就一条命令都不发

四个动词要像素，而像素只能由投影算（§1：像素在这个入口**内部**由投影算出）。
投影拿不到就等于「不知道该把手指放到哪」 —— 这时候发一条 `input` 就是**猜**。
所以四个动词**先要投影、再发命令**，要不到就直接抛出去（T5 裁定）。
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, final

from ark.contracts import (
    AssetPort,
    Canvas,
    DeviceFacts,
    GridCoord,
    Pixel,
    Projection,
    UI_LAYOUT,
    View,
)

__all__ = [
    "AdbTimeout",
    "WmSize",
    "parse_wm_size",
    "parse_dumpsys_input",
    "AdbTransport",
    "AdbDevice",
    "DEFAULT_TIMEOUT_MS",
    "DEVICE_TMP_DIR",
    "FRAME_FILE_NAME",
    "R_GRID",
    "FACING_VECTORS",
    "SWIPE_DURATION_MS",
    "PANEL_WAIT_MS",
    "PANEL_RETREAT_AT",
    "PANEL_SKILL_AT",
]


DEFAULT_TIMEOUT_MS: Final[int] = 30_000
"""一条命令的默认墙钟上限（ms）。

实测一条命令的地板约 200 ms、取像往返中位约 1.0 s（设备层 §3），所以 30 s 是
「明显坏了」那条线，不是「慢」那条线。它的用处是**不许无限等**，不是掐着实测值卡。
"""

DEVICE_TMP_DIR: Final[str] = "/data/local/tmp"
"""设备端那个文件的落点。

`shell` 用户在这里有写权限、不需要 root，而且它**不是媒体库**（`/sdcard` 上的
东西会被媒体扫描器看见）。

**它是一个固定槽位，不每帧删**（设备层 §4 · T4 定）：同名写入直接覆盖、不累积，
而删它要多跑一条命令、约 200 ms（§3 的地板）。**本地那一份才每帧删**（本机文件，
不花命令）。
"""

FRAME_FILE_NAME: Final[str] = "ark_frame.raw"
"""设备端与本地两侧共用的文件名。原始帧定长 14,112,016 字节，`.raw` 是它自己。"""

R_GRID: Final[float] = 0.3
"""部署终点相对格心的偏移 `r`（单位：格）。**估 0.3 格，真机待测。**

设备层 §2：终点是 `格心 + r·û`，同一个终点**同时编码了格与朝向**。`r` 被两边
夹住（上界：终点要落在可部署区域内；下界：四个朝向要分得开），而 §6 把「`r` 的
取值」列成待实测项 —— 这一轮**不测它**，只把它钉成一个**显式常量**。
"""

FACING_VECTORS: Final[Mapping[str, tuple[float, float]]] = MappingProxyType(
    {
        "上": (0.0, -1.0),
        "下": (0.0, 1.0),
        "左": (-1.0, 0.0),
        "右": (1.0, 0.0),
    }
)
"""朝向 → **格空间**里的单位向量 `û`。**四个，不含斜向**（时序层 §4）。

格空间的 y 向**下**为正（与屏幕一致），所以「上」是 `(0, -1)`。四个字面值是
设备层 §5 钉的，词表在时序层 §4。
"""

SWIPE_DURATION_MS: Final[int] = 400
"""部署那一次 `input swipe` 的 `duration` 参数（ms）。**真机待测**（设备层 §6）。

它是「手势耗时」那一栏的输入（§3）：`swipe` 的 `duration` 参数 vs 实际墙钟。
"""

PANEL_WAIT_MS: Final[int] = 400
"""点开干员之后**等面板滑出来**多久（ms）。**真机待测**（设备层 §6）。

设备层 §5：这个等待**不是判断**（等的是一件确定会发生的事），但它**也不能盲等**
—— 所以它是一个**显式常量**，不是一个循环。「面板出现要等多久」是 §6 的待实测项。
"""

PANEL_RETREAT_AT: Final[GridCoord] = GridCoord(0.0, 1.2)
"""面板上那个**撤退键**相对干员所在格的偏移（格）。**占位值，真机待测。**

视觉层/技能详情通道 §1：点开干员之后，「在干员**旁**显示技能按钮」—— 所以按钮的
位置是**格空间里的一个偏移**，像素经投影算，视图一换它跟着换。

**这个数是我挑的占位**：这一轮**没有面板打开的那一帧**，所以它没有实测来源
（§6 那张表里也没有这一项）。它不影响任何离线检查 —— 拿不到投影时这条命令根本
发不出去（T5 裁定五）。
"""

PANEL_SKILL_AT: Final[GridCoord] = GridCoord(0.0, 2.0)
"""面板上那个**技能键**相对干员所在格的偏移（格）。**占位值，真机待测**（同上）。"""


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

    **两个字段都不转置**（设备层 §4）：这台机器报 `1260x2800`（竖屏自然朝向），
    画布是 `2800x1260`。**画布口径不靠这里转** —— 设备事实取自 `dumpsys input`
    那条 `Viewport` 行的 `logicalFrame`（T5 定），所以这条解析器不必知道方向。
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

    **不转置**（设备层 §4）：想要 2800×1260 就自己去写 `1260x2800` —— 这台设备的
    `wm size` 与画布差 90°，而**画布口径从 `dumpsys input` 那条 `Viewport` 行来**
    （T5 定），不必在这里转。两行都在时 `override` 非空。

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


_VIEWPORT_LINE: Final[re.Pattern[str]] = re.compile(
    r"Viewport\s+\S+\s*:(?P<body>.*)$", re.MULTILINE
)
"""`dumpsys input` 里那条 `Viewport <名字>: ...` 行。名字（`INTERNAL` 之类）不认，
它随安卓版本变。"""

_ORIENTATION: Final[re.Pattern[str]] = re.compile(r"\borientation=(-?\d+)")
"""方向。**词表就是设备自己报的那个整数**：字面值 `"0"` `"1"` `"2"` `"3"`
（T5 裁定一）。"""

_LOGICAL_FRAME: Final[re.Pattern[str]] = re.compile(
    r"\blogicalFrame=\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"
)
"""`[左, 上, 右, 下]`。**它直接就是画布口径**（T5 裁定二）。"""


def parse_dumpsys_input(text: str) -> DeviceFacts:
    """把 `dumpsys input` 的输出解成设备事实。**纯函数，不碰设备。**

    取的是 `Viewport <名字>: ...` 那一行里的两个字段（T4.1 试出来、T5 裁定的那一对）：

        orientation=1                     → 方向就是**设备自己报的那个整数**
        logicalFrame=[0, 0, 2800, 1260]   → 画布口径**直接**就是它（宽 2800、高 1260）

    **一个转置都不做**（T5 裁定二）：`wm size` 报 `1260x2800`（竖屏自然朝向），
    而画布是 2800×1260 —— 差的那 90° 不必自己转，这条 `Viewport` 行已经把画布
    口径给出来了。两个口径不许混（设备层 §4）。

    **找不到就报错**：拿不到这两个字段说明这份输出不是那个形状，猜着解就是假事实。
    """
    if not isinstance(text, str):
        raise TypeError(f"dumpsys input 的输出要是字符串，收到 {type(text).__name__}")
    for match in _VIEWPORT_LINE.finditer(text):
        body = match.group("body")
        orientation = _ORIENTATION.search(body)
        frame = _LOGICAL_FRAME.search(body)
        if orientation is None or frame is None:
            continue
        left, top, right, bottom = (int(value) for value in frame.groups())
        return DeviceFacts(
            resolution=Canvas(right - left, bottom - top),
            orientation=orientation.group(1),
        )
    raise ValueError(
        "dumpsys input 的输出里没有一条同时带 orientation= 与 logicalFrame=[...] 的 "
        f"`Viewport` 行（设备事实就住那一行）：{text[:400]!r}"
    )


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


def _digits(point: Pixel) -> list[str]:
    """像素 → 两个整数字符串。`input` 只认整数，四舍五入到最近的那一个。"""
    return [str(int(round(point.x))), str(int(round(point.y)))]


class AdbDevice:
    """设备端口的真机适配器：**取一帧 + 轴上的四个动词 + 设备事实**（设备层 §5）。

    `grab_frame` 的形状是「一条 `shell` 写设备端文件 + 一条 `pull` 取回」，
    **不走 `exec-out` 直出**（设备层 §4 · D-全局-19）：原始帧 14 MB 走流式通道比
    取文件慢，而且 PowerShell 的 `>` 会毁掉二进制。

    `assets` 与 `view` 都是**构造时收进来的**（T5 裁定三）：四个动词要像素时向
    `assets.projection(view)` 要。**端口一个字不改** —— 切视图 = 重新装配
    （代码结构 规则二那句「换一套参数 = 换一个适配器」）。

    两个都**可以省略**（默认没有资产端口 + 常速视图）：取帧与设备事实**不需要投影**，
    所以 `AdbDevice(transport)` 对那两条成立；而四个动词在拿不到投影时**一条命令
    都不发**（T5 裁定五）—— 省略它们就是「没有投影」的一种。

    **replay 适配器对四个动作方法都只记一笔、不点**（契约层 §5）。
    """

    def __init__(
        self,
        transport: AdbTransport,
        assets: AssetPort | None = None,
        view: View = View.NORMAL,
    ) -> None:
        if not isinstance(view, View):
            raise TypeError(f"此刻的视图要是 View，收到 {type(view).__name__}")
        self._transport = transport
        self._assets = assets
        self._view = view

    # -- 设备管理 -----------------------------------------------------------

    def grab_frame(self) -> bytes:
        """取一帧，返回**原始帧**（不是归一化帧，设备层 §4）。

        **两条命令，不许有第三条**（设备层 §4 · T4 定）：

        1. 一条 `shell`：**先删旧文件、再 `screencap` 写新的**（原始 RGBA，
           **不带 `-p`**）—— 两步并进同一次调用；
        2. 一条 `pull`：把那一份取回本地。

        设备端那个文件是**固定槽位**：同名写入直接覆盖、不累积，所以**它不每帧删**
        —— 删它要多跑一条命令、约 200 ms（§3 的地板）。
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

    def device_facts(self) -> DeviceFacts:
        """设备事实：分辨率 · 方向（设备层 §5）。**查真机，不发手势。**

        一条 `dumpsys input`，取那条 `Viewport ...` 行（解析在 `parse_dumpsys_input`）：
        `orientation=` 就是方向（词表就是那个整数），`logicalFrame=` 直接就是画布。

        **它不需要投影** —— 它不发手势，所以拿不到投影也照样能查（T5 裁定五只约束
        四个动词）。
        """
        raw = self._transport.run_command(["shell", "dumpsys", "input"])
        return parse_dumpsys_input(raw.decode("utf-8", "replace"))

    # -- 轴上的四个动词 -----------------------------------------------------

    def deploy(self, card: int, at: GridCoord, facing: str) -> None:
        """部署：**一次直线 swipe**（设备层 §2）。

        起点 = 备战席第 `card` 张卡的中心（UiLayout 的卡框），终点 = `格心 + r·û`
        —— 同一个终点**同时编码了格与朝向**，所以四个朝向塌成一次直线拖动。

        `card` 是**此刻读到的那一张**，这条类**不许缓存它**（契约层 §5）。下标
        **0 = 最左那张卡**（与 UiLayout 那张表同一条口径）。
        """
        if not isinstance(facing, str) or facing not in FACING_VECTORS:
            raise ValueError(
                "朝向只有 " + "、".join(FACING_VECTORS) + f" 四个（不含斜向），收到 {facing!r}"
            )
        at = self._as_grid(at)
        projection = self._projection()
        start = self._card_center(card)
        ux, uy = FACING_VECTORS[facing]
        end = projection.to_pixels(
            GridCoord(at.x + R_GRID * ux, at.y + R_GRID * uy)
        )
        self._swipe(start, end, SWIPE_DURATION_MS)

    def retreat(self, at: GridCoord) -> None:
        """撤退：点那个格上的干员 → 等面板 → 点面板上的撤退（设备层 §2）。"""
        self._two_taps(at, PANEL_RETREAT_AT)

    def skill_on(self, at: GridCoord) -> None:
        """开技能：同撤退那个形状，点面板上的技能键。**仅对手动触发的技能成立**。"""
        self._two_taps(at, PANEL_SKILL_AT)

    def skill_off(self, at: GridCoord) -> None:
        """关技能：同上。**仅对可关的技能成立**（时序层 §4）。"""
        self._two_taps(at, PANEL_SKILL_AT)

    # -- 内部：像素与手势原语 -----------------------------------------------

    def _projection(self) -> Projection:
        """此刻这一套投影。**拿不到就抛** —— 调用方一条命令都别发（T5 裁定五）。

        三条"拿不到"，**报出来的是同一件事**：投影还没标定，所以像素算不出来。

        ① **没接资产端口**（`assets` 是空的 —— 取帧与设备事实不需要它，所以那两条
        命令允许单参数构造）；② 接了，但**那个实现上根本没有这个方法**（`EmptyAssetPort`
        就是这一支：T5 裁定四说「留空 = 方法不存在」）；③ 方法在、调用时抛。

        **② 不许落成一句光秃秃的 `AttributeError`** —— 那读起来像 bug，而它其实是
        「投影还没标定」。所以在这里转成一句带话的报错。
        """
        if self._assets is None:
            raise RuntimeError(
                "投影还没有来源：这条设备适配器没有接资产端口（取帧与设备事实不需要它，"
                "所以允许单参数构造）。像素算不出来，**一条命令都不发**"
            )
        make = getattr(self._assets, "projection", None)
        if not callable(make):
            raise RuntimeError(
                "投影还没有来源：资产端口上没有 projection 这个方法（留空 = 方法不存在，"
                "代码结构 §尚未落地），欠账登记在 runtime/wiring.py 的 PORT_GAPS 里 —— "
                "标定是首轮探索编辑器的第一块。像素算不出来，**一条命令都不发**"
            )
        return make(self._view)

    @staticmethod
    def _as_grid(at: GridCoord) -> GridCoord:
        """格坐标入口**只收格坐标**：喂像素进来就抛（契约层 §5）。"""
        if not isinstance(at, GridCoord):
            raise TypeError(
                f"这个入口收的是格坐标，收到 {type(at).__name__}："
                "像素在这一层内部由投影算出（设备层 §1）"
            )
        return at

    @staticmethod
    def _card_center(card: int) -> Pixel:
        """备战席第 `card` 张卡的**中心**（UiLayout 的卡框）。

        表里没有那一张就**报错，不猜**：那张表是就着 T5 段 0 那一帧（5 张卡）量的，
        而**卡距随卡数变**（见设备层 §6 与 T5 报告的规格洞）。
        """
        if isinstance(card, bool) or not isinstance(card, int):
            raise TypeError(f"卡下标是整数，收到 {type(card).__name__}")
        rect = UI_LAYOUT.rects.get(f"squad_card_{card}")
        if rect is None:
            raise IndexError(
                f"备战席卡框表里没有第 {card} 张卡：那张表是就着 T5 段 0 那一帧"
                "（5 张卡）量的，只有 0..4；别的卡数下卡距不一样"
            )
        return Pixel(rect.x + rect.width / 2, rect.y + rect.height / 2)

    def _two_taps(self, at: GridCoord, button_at: GridCoord) -> None:
        """撤退 / 开技能 / 关技能共用的形状：**两次点 + 一次等面板**（设备层 §2）。"""
        at = self._as_grid(at)
        projection = self._projection()
        self._tap(projection.to_pixels(at))
        time.sleep(PANEL_WAIT_MS / 1000.0)
        self._tap(
            projection.to_pixels(
                GridCoord(at.x + button_at.x, at.y + button_at.y)
            )
        )

    def _tap(self, point: Pixel) -> None:
        """手势原语「点」。**端口上看不见它**（时序层 §4）。"""
        self._transport.run_command(["shell", "input", "tap", *_digits(point)])

    def _swipe(self, start: Pixel, end: Pixel, duration_ms: int) -> None:
        """手势原语「拖」：按下与抬起是它的两端（设备层 §2）。

        `input swipe` 是**原子的** —— 手指无法跨越两条 adb 命令保持按下，所以
        部署必须是一次性手势，不能「先拖到格、停一下看清预览、再选方向」。
        """
        self._transport.run_command(
            [
                "shell",
                "input",
                "swipe",
                *_digits(start),
                *_digits(end),
                str(int(duration_ms)),
            ]
        )

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
