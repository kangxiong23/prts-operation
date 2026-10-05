"""设备端口的第二个适配器：离线重放（设备层 §5 那张表的 `replay` 一列）。

它与 `adb.py` 是**同一个端口的两个实现**，不是新端口、也不是特例代码 ——
「离线重放不是特例代码」这句话的落地形态就是**换一个适配器**（代码结构 规则二）。

| 端口方法 | 它怎么答 |
| --- | --- |
| `grab_frame()` | 返回帧源里的下一帧 |
| `deploy` / `retreat` / `skill_on` / `skill_off` | **只记一笔，不点** |
| `device_facts()` | 从录制的前提头里回放；**没记就报错** |

**它与 `adb` 适配器唯一的结构差别**（设备层 §5 · T6 定）：它收**帧源 + 可选的
录制前提头**，**不收资产端口与视图** —— 它一个像素都不算、一条命令都不发、
**也不解帧头**（像素布局对它没有用途）。所以这里没有 `_projection()`，也没有
"拿不到投影就报错"那一支。

**前提头的类型是 `DeviceFacts` 或 `None`**（设备层 §5 · T6 补）：把清单里那个
映射折成 `DeviceFacts` 是**加载器**的事（[基准集] §2）—— 让这一层去认"清单里
那个键叫 `orientation`"，等于把离线那套文件格式塞进引擎（代码结构 规则三：
引擎对外只认自己的语义）。传别的形状**报错，不许悄悄收下**。

**帧依次吐、吐完再要就报错、不许回卷**（基准集 §3）：回卷会让「这一批跑过了」
这句话变成假的（离线验收 §1）。所以这一层没有"重置"，也没有"跳到第几帧"。

**四个动词不带帧号**：设备层拿不到逻辑帧（设备层 §5 最后一条），帧号是控制层的
事。流水里只有**动词 + 按签名顺序的位置参数**。

**帧源是惰性的，而且构造时不预读**：一帧 14,112,016 字节，把一批全读进内存是
几百 MB —— 这条路本来就该一次只握一帧（设备层 §4：只有"取一帧"，没有"取像流"）。
所以 `__init__` 只把帧源收成迭代器，**第一帧由第一次 `grab_frame()` 取**。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ark.contracts import DeviceFacts, GridCoord

__all__ = [
    "ReplayDevice",
    "ReplayAction",
    "ReplayExhausted",
    "ReplayFactsMissing",
]


class ReplayExhausted(Exception):
    """帧吐完了（这一层自己的异常，与 `AdbTimeout` 同类）。

    它与 `AdbTimeout` 一样是**层内**的坏法，不是契约层 §5 的「中止」：中止要
    逻辑帧，而设备层拿不到（设备层 §5 最后一条）。
    """

    def __init__(self, frames: int) -> None:
        super().__init__(
            f"这一批的 {frames} 帧已经吐完了：重放不许回卷、不许重吐 —— "
            "回卷会让「这一批跑过了」这句话变成假的（基准集 §3）"
        )
        self.frames = frames


class ReplayFactsMissing(Exception):
    """录制没记前提头（设备层 §5）。

    **不许编一个**：与「不许编一个投影」同一条（端口与错误 §5）。前提头是环境
    前提，编出来的分辨率会让所有 ROI 在一个没人验过的画布上取。
    """

    def __init__(self) -> None:
        super().__init__(
            "录制里没有前提头：device_facts() 报错，不许编一个 —— "
            "与「不许编一个投影」同一条（设备层 §5 · 基准集 §2）"
        )


@dataclass(frozen=True)
class ReplayAction:
    """一条流水：**动词 + 按签名顺序的位置参数**（设备层 §5）。

    - `verb` 就是那四个方法名（`deploy` / `retreat` / `skill_on` /
      `skill_off`），**不是轴上的中文动词** —— 这里是设备层，轴上的词表在
      时序层 §4，翻译是调用方的事；
    - `params` 按签名顺序排，`deploy` 是 `(card, at, facing)`、其余三个是
      `(at,)`，**不带帧号**（设备层拿不到逻辑帧）。
    """

    verb: str
    params: tuple[object, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.verb, str) or not self.verb:
            raise TypeError(f"流水的动词是非空字符串，收到 {self.verb!r}")
        if not isinstance(self.params, tuple):
            raise TypeError(
                f"流水的参数要是一个元组，收到 {type(self.params).__name__}："
                "按签名顺序排，不是列表也不是裸值"
            )


class ReplayDevice:
    """设备端口的重放适配器（设备层 §5）。

    `ReplayDevice(frames, *, facts=None)`：

    - `frames` 是**一串原始帧字节**（设备层 §4 那种），依次吐；
    - `facts` 是**录制的前提头**，类型是 `DeviceFacts` 或 `None`
      （设备层 §5）—— 折好它是**加载器**的事（基准集 §2）。

    **它没有 transport、没有资产端口、没有视图**：这四个动词只往 `ledger` 上
    记一笔（离线验收 §2 那条"注入只记一笔、不真的点"），所以它一个像素都不算、
    一条命令都发不出去。这是它与 `AdbDevice` 唯一的结构差别，不是破例。
    """

    def __init__(
        self, frames: Iterable[bytes], *, facts: DeviceFacts | None = None
    ) -> None:
        if isinstance(frames, (bytes, bytearray, memoryview)):
            raise TypeError(
                "帧源是一串帧，不是一帧原始帧字节：把一帧当成一批喂进来，"
                "第一次取帧就会吐出第一个字节"
            )
        if facts is not None and not isinstance(facts, DeviceFacts):
            raise TypeError(
                "录制的前提头要么是 DeviceFacts、要么是 None，收到 "
                f"{type(facts).__name__}：把清单里那个映射折成 DeviceFacts 是"
                "**加载器**的事（基准集 §2）—— 引擎不认离线那套键名"
                "（代码结构 规则三）"
            )
        self._source = iter(frames)
        self._taken = 0
        self._ledger: list[ReplayAction] = []
        self._facts = facts

    # -- 设备管理 -----------------------------------------------------------

    def grab_frame(self) -> bytes:
        """取一帧：**帧源里的下一帧**，依次吐、不许回卷（设备层 §5）。

        吐完再要就是 `ReplayExhausted` —— 不是空帧、不是最后一帧的重复。
        **第一帧是这里取出来的**，不是构造时预读的。
        """
        try:
            raw = next(self._source)
        except StopIteration:
            raise ReplayExhausted(self._taken) from None
        self._taken += 1
        return raw

    def device_facts(self) -> DeviceFacts:
        """设备事实：**从录制的前提头里回放**（设备层 §5）。

        没记就报 `ReplayFactsMissing`，**不许编一个** —— 同"不许编一个投影"。
        返回的就是加载器折好的那一个（**原样**，一个字段都不动）。
        """
        if self._facts is None:
            raise ReplayFactsMissing()
        return self._facts

    # -- 轴上的四个动词：只记一笔 -------------------------------------------

    def deploy(self, card: int, at: GridCoord, facing: str) -> None:
        """部署：**只记一笔，不点**（设备层 §5）。参数按签名顺序排。"""
        self._record("deploy", (card, at, facing))

    def retreat(self, at: GridCoord) -> None:
        """撤退：**只记一笔，不点**。"""
        self._record("retreat", (at,))

    def skill_on(self, at: GridCoord) -> None:
        """开技能：**只记一笔，不点**。"""
        self._record("skill_on", (at,))

    def skill_off(self, at: GridCoord) -> None:
        """关技能：**只记一笔，不点**。"""
        self._record("skill_off", (at,))

    # -- 流水账 -------------------------------------------------------------

    @property
    def ledger(self) -> tuple[ReplayAction, ...]:
        """流水账：到目前为止记下的每一条，**按发生顺序**（设备层 §5）。

        它是属性、不是公开方法 —— 端口方法集之外的公开**方法**要先把契约层改了
        （代码结构 §尚未落地 的补法规则）。取到的是一份**快照**（元组），不是
        内部那个列表：调用方拿它去比对，不该能改到这一层。
        """
        return tuple(self._ledger)

    # -- 内部 ---------------------------------------------------------------

    def _record(self, verb: str, params: tuple[object, ...]) -> None:
        self._ledger.append(ReplayAction(verb=verb, params=params))
