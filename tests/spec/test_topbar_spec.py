"""顶部两个数读数链规格符合性验收（pytest 外壳）。

检查逻辑在 `topbar_spec.py`，那份不依赖 pytest，可以直接跑：

    python tests/spec/topbar_spec.py

**分工**（详见那份文件的模块文档）：`tests/spec/` 由规划者维护，验证"规格说的
那些话成立"；`tests/unit/` 由执行者维护，验证"实现按我的设计工作"。两者不互相
import —— 混在一起，失败时就看不出是"实现错了"还是"验收标准错了"。

**它是规划者的工具，不进任务消息**（D-全局-22）：执行者不该照着它写实现。而它被
`pytest` 收集得到，所以这一层**不许把探针的源码漏进 traceback**（见下面那条注释）。
"""

from __future__ import annotations

import pytest
from topbar_spec import CHECKS, Check


@pytest.mark.parametrize("check", CHECKS, ids=[c.ident for c in CHECKS])
def test_spec_check(check: Check) -> None:
    """每条规格检查各一条用例：失败时能一眼看出是哪条、出处在哪。"""
    # **别在 `except` 里 raise**：那样 pytest 会把探针的源码行打进
    # "During handling of the above exception" 那一段 —— 那是 `tests/spec/`
    # 的一条侧信道，D-全局-17 那句"不许读"挡不住它（D-全局-22 注意②）。
    # 所以先把证据收进变量，出了 `except` 再报。
    evidence: str | None = None
    try:
        check.probe()
    except AssertionError as bad:
        evidence = str(bad)
    except BaseException as broken:  # noqa: BLE001 —— 探针自己炸了也要报清楚
        evidence = f"{type(broken).__name__}: {broken}"
    if evidence is not None:
        pytest.fail(f"{check.claim} ｜ 出处 {check.spec} ｜ 证据 {evidence}")
