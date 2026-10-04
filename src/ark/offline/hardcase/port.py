"""样本端口的**空实现**：离线验收不需要采集（契约层 §5 · 采集与验收 §2）。

## 它为什么住这里，而不是住 `runtime/`

样本端口存在的**全部理由**是 `engine` 永远不 import `offline`（代码结构 规则一）
—— 触发点在线上（槽自己发现不确定、组装发现账不平），而实现住
`offline/hardcase/`。所以**空实现也是 `offline/hardcase/` 的实现**，
`runtime/wiring.py` 只把它接上。

## 它是空的，不是欠账

离线跑的时候没有东西要落盘。真实现（上限 · 冻结 · 分池）随采集器一起来，那时它
import 的东西会变多，而**端口一个字都不用改**（补法规则：只加方法，不改端口）。
"""

from __future__ import annotations

__all__ = ["EmptySamplePort"]


class EmptySamplePort:
    """样本端口的空实现：`submit` **什么都不做**。

    不落盘、不计数、不留状态 —— 「什么都不做」正是它的规格。一个会偷偷攒东西的
    空实现会让「离线跑过一遍」留下副作用，而离线验收不需要采集。
    """

    def submit(self, sample: object) -> None:
        """什么都不做。`sample` 是不透明的（它的形状归实现侧定），这里连看都不看。"""
        return None
