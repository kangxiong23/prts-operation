"""§5 端口与错误：四个端口 + 三种错误。

这一节只有**边界**：端口是接口，错误是"不正常时工具怎么答"。实现（适配器）
住在 `engine/device/`、`offline/hardcase/` 里，由 `runtime/wiring.py` 接上；
这一层不 import 任何内部模块，端口也不许知道"谁在实现它"。

四个端口：

- **DevicePort**  设备：取一帧 · 注入手势 · 设备事实（分辨率 / 方向）；
- **ClockPort**   时钟：只给**墙钟毫秒**。**逻辑帧不在这个端口里** —— 它由费用
                  推算（时间基），不是从时钟读来的；
- **SamplePort**  样本：落一份样本。三层触发产出的是同一份东西，区别只在
                  "谁触发 · 留几条 · 落在哪个池子里"，那是 `offline/hardcase/` 的策略；
- **AssetPort**   资产：**方法集留空**，补法规则写在它的文档字符串里。

两条绷得最紧的线：

- **端口里不许出现"现在真的是游戏在跑"这类信息**：它必须纯到 replay 适配器能
  满足 —— "离线重放不是特例代码"的落地条件就是**换一个适配器**，不是给端口加
  一个"我在重放"的开关；
- **错误集里不许有"部分成功"这一支**：工具不报"我尽力了"，只报三态。

三种错误是三种**数据**，不是三条异常：降级本来就"照常返回"（它只是进存疑清单），
拒绝与中止同样是**回答的形状**，不是抛出来的东西。三种各自带自己那份数据，
**没有共同的基类** —— 有基类就会有人写一个 `except` 把它们一口吞掉。

叶子方向：ports → geometry（格坐标 / 画布）· scale（墙钟毫秒 / 逻辑帧）·
containers（中止要报"哪些动作已确认生效"，降级要带一条存疑清单条目）。端口是
这张图的**汇点**：它 import 别人，别人不 import 它。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, final

from ark.contracts.containers import Action, DoubtEntry
from ark.contracts.geometry import Canvas, GridCoord
from ark.contracts.scale import LogicalFrame, WallClockMs

__all__ = [
    "DevicePort",
    "ClockPort",
    "SamplePort",
    "AssetPort",
    "DeviceFacts",
    "Refusal",
    "Degradation",
    "Abort",
]


# ---------------------------------------------------------------------------
# 设备事实：设备端口那条"设备管理"的返回形状
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class DeviceFacts:
    """设备事实：**分辨率 · 方向**（设备层 §5）—— 它喂前提头校验（时序层 §3）。

    **不吻合就拒绝执行、不降级**（设备层 §4）：画布尺寸一歪，所有 ROI 与投影一起
    失效 —— 那不是"某个字段降级了"，是整段的坐标都不作数。

    - `resolution` 是**捕获分辨率本身**（Canvas：不缩放。缩放会引入插值误差，
      还会让"不吻合即拒绝"变成"先缩放再拒绝，等于已经适配过了"）；
    - `orientation` 是**屏幕方向**，它是设备报出来的环境前提，**取值词表不在这
      一层**（与 `end_reason` 同一个理由：那是环境 / 策略的事）。注意它与部署的
      **朝向**不是一回事 —— 朝向是四个方向（上下左右），两个词在规格里分得很清。
    """

    resolution: Canvas
    orientation: str

    def __post_init__(self) -> None:
        if not isinstance(self.resolution, Canvas):
            raise TypeError(
                f"设备事实的分辨率要是画布，收到 {type(self.resolution).__name__}"
            )
        if not isinstance(self.orientation, str) or not self.orientation:
            raise TypeError(f"设备事实的方向是非空字符串，收到 {self.orientation!r}")


# ---------------------------------------------------------------------------
# 四个端口
# ---------------------------------------------------------------------------


class DevicePort(Protocol):
    """设备端口：**整个项目里唯一接触真机的地方**（设备层 §1 的三层收在这里）。

    它的正确性判据只有一条：给一个动作，我把手指放对地方。**它不知道动作有没有
    生效** —— 那是视觉层与生效反馈的事（接口层/执行与失败 §3），这条边界不能糊。

    手势原语的参数是**格坐标与朝向**，不是像素：像素在这个入口**内部**由投影算
    出。理由是慢放是另一套投影 —— 让每个调用方自己换算，等于把"慢放下必须换一套"
    这条纪律复制到每一个调用点（设备层 §1）。
    """

    def grab_frame(self) -> bytes:
        """取一帧：**原始帧**，不是归一化帧（设备层 §4）。

        这一层只管"帧到手"，不该知道画布；归一化是视觉层的入口。**只有"取一帧"，
        没有"取像流"** —— 一次识别现抓一帧，没有常驻取像，所以也没有缓冲。
        replay 适配器返回该帧号对应的录制帧。
        """

    def inject_gesture(self, at: GridCoord, facing: str) -> None:
        """注入一次手势：**目标格坐标 + 朝向**（设备层 §1）。

        朝向**必填**：不选朝向 = 一次未完成的放置（时序层 §4）。四个朝向（上下
        左右，**不含斜向**），取值词表在时序层 §4 —— 这一层不发明它。

        replay 适配器**只记一笔，不点**："端口必须纯到 replay 能满足"这条要求
        具体就落在这一句上。
        """

    def device_facts(self) -> DeviceFacts:
        """设备事实：分辨率 · 方向（设备层 §5）。

        replay 适配器从录制的**前提头**里回放这两个值 —— 所以它们是环境前提，
        不是"这台机器此刻的状态"。
        """


class ClockPort(Protocol):
    """时钟端口：**只给墙钟毫秒**（契约层 §3）。

    **逻辑帧不在这个端口里**：它由费用推算（时间基），不是从时钟读来的。两种刻度
    混起来会产生一个**静默的错** —— 同一次 200 ms 的 ADB 延迟，常速下是 6 个游戏
    帧、慢放下只有 0.6 帧 —— 所以这里的返回类型是 `WallClockMs`，不是裸数。
    """

    def wall_clock_ms(self) -> WallClockMs:
        """此刻的墙钟毫秒。折成帧是控制层的事。"""


class SamplePort(Protocol):
    """样本端口：**方法集 = 一个方法**（契约层 §5）。

    位置是硬性欠的：触发点在线上（槽自己发现不确定、组装发现账不平），而实现住
    `offline/hardcase/`。**离线跑的时候它是空实现** —— 离线验收不需要采集。

    **为什么是一个方法，而不是按三层触发各开一个**：三层触发产出的**是同一份
    东西**（难例采集 §1 那七部分），区别只在"谁触发 · 留几条 · 落在哪个池子里"
    —— **上限、冻结、分池全是 `offline/hardcase/` 的事**。**端口是边界，策略在
    实现侧**；写成一个方法，端口就不必跟着触发层数变。
    """

    def submit(self, sample: object) -> None:
        """落一份样本。

        `sample` 是**不透明的**：那份数据长什么样（难例采集 §1 的七部分：原帧 ·
        逻辑标签 · 当时的槽 · 触发原因 · 存疑清单条目 · 成分表 · 戳）归实现侧定。
        那七样里没有一样是这一层的类型 —— 在这里把它们的形状写死，等于替采集侧
        发明一盘格式，而"留几条 · 落在哪个池子"本来就不是边界的事。所以这一层
        只声明"有一份东西要落"，不声明它长什么样。
        """


class AssetPort(Protocol):
    """资产端口：**方法集留空**，而且**长期是空的** —— 这不是欠账。

    补法规则：**每个方法由它的第一个用户层在用到时补；补的时候只加方法，不改
    端口本身。**

    为什么留空：五个访问点的返回类型**分属不同层**（投影在契约层 §2 · 格属性在
    状态层 / 地形 · 路线 / 特征库 / 背景帧在资产生成器）。现在定，等于替那些层
    发明类型。

    位置本身是硬的：`engine` 永远不 import `offline`（代码结构 规则一），而上面
    那些值全是 `offline/assets_gen/` 的产物、**视觉层与设备层都要读** —— 所以它们
    必须经由契约层暴露的某个东西到达，由 `runtime/wiring.py` 接上。**UI 布局常量
    不在这个端口里**：它不是生成物，它的类型与值就住 §2（geometry.UiLayout）。
    """


# ---------------------------------------------------------------------------
# 三种错误：各自带自己那份数据，没有共同基类
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class Refusal:
    """拒绝：**整段不执行**，附哪一项不吻合（接口层/执行与失败 §4）。

    排动作时：对象 / 格子 / 技能不合法 → **不写轴**；`继续到` 时：前提不吻合 /
    不是战斗界面 / 版本认不出 → **整段拒绝**。**不降级、不"尽力执行"** —— 部分
    执行比不执行更坏（AI 以为跑过了）。

    `item` 是"哪一项"（前提头的槽名 / 参数名），`detail` 是"怎么不吻合"。两个
    都给：只知道"被拒了"而不知道卡在哪一项，调用方只能把整段重猜一遍。
    """

    item: str
    detail: str

    def __post_init__(self) -> None:
        for what, value in (("拒绝的哪一项", self.item), ("拒绝的理由", self.detail)):
            if not isinstance(value, str) or not value:
                raise TypeError(f"{what}是非空字符串，收到 {value!r}")


@final
@dataclass(frozen=True)
class Degradation:
    """降级：**照常返回**，但进快照页眉的存疑清单（接口层/执行与失败 §4）。

    它带的就是**那一条存疑清单条目**（形状在 §4 的 `DoubtEntry`）：类别 · 涉及
    哪个字段或来源 · 严重度 · 涉及哪几帧。工具不因此少返回什么 —— 字段自身照旧
    带三态与形态，降级只是"顺带报一句我这儿不干净"。

    什么时候：动作没生效 · 时间不可信 · 字段识别降级或通道冲突 · 位置自检失败
    （脱同步）· 每片的账不平（I1–I4）。
    """

    doubt: DoubtEntry

    def __post_init__(self) -> None:
        if not isinstance(self.doubt, DoubtEntry):
            raise TypeError(
                f"降级要带一条存疑清单条目，收到 {type(self.doubt).__name__}"
            )


@final
@dataclass(frozen=True)
class Abort:
    """中止：跑到一半坏了 → 返回**跑到哪一帧** + **哪些动作已确认生效**（执行与失败 §4）。

    `frame` 是**逻辑帧**（轴上只许出现逻辑帧，§3）：裸 int 与墙钟都收不了，与
    动作 / 观测迹 / 存疑条目 / 页眉那四个帧戳位同一条纪律。

    `confirmed` 是**已确认生效的那些动作**，不是"应该生效的"：所以它**不假装
    跑完了**。空 tuple 是有意义的一种 —— 一帧都没跑到就断了。

    什么时候：设备断连 · 跑到一半画面变了。**墙钟超时也报中止**（设备层 §5），
    不是无限等。
    """

    frame: LogicalFrame
    confirmed: tuple[Action, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, LogicalFrame):
            raise TypeError(
                f"中止的帧号只许是逻辑帧，收到 {type(self.frame).__name__}："
                "墙钟只在记录里留痕"
            )
        if not isinstance(self.confirmed, tuple):
            raise TypeError(
                f"已确认生效的动作要用 tuple，收到 {type(self.confirmed).__name__}"
            )
        for action in self.confirmed:
            if not isinstance(action, Action):
                raise TypeError(
                    f"已确认生效的只能是动作，收到 {type(action).__name__}"
                )
