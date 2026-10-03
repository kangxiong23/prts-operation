"""规格符合性验收（pytest 外壳）。

检查逻辑在 `contracts_spec.py`，那份不依赖 pytest，可以直接跑：

    python tests/spec/contracts_spec.py

**分工**（详见那份文件的模块文档）：`tests/spec/` 由规划者维护，验证"规格说的
那些话成立"；`tests/unit/` 由执行者维护，验证"实现按我的设计工作"。两者不互相
import —— 混在一起，失败时就看不出是"实现错了"还是"验收标准错了"。
"""

from __future__ import annotations

import pytest
from contracts_spec import CHECKS, Check


@pytest.mark.parametrize("check", CHECKS, ids=[c.ident for c in CHECKS])
def test_spec_check(check: Check) -> None:
    """每条规格检查各一条用例：失败时能一眼看出是哪条、出处在哪。"""
    try:
        check.probe()
    except AssertionError as bad:
        pytest.fail(f"{check.claim} ｜ 出处 {check.spec} ｜ 证据 {bad}")
