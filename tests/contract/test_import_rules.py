"""边界契约测试（pytest 外壳）。

检查逻辑在 `import_rules.py`，那份只用标准库，没装 pytest 也能跑：

    python tests/contract/import_rules.py

规则来自 `docs/10-规格/代码结构.md`。
"""

from __future__ import annotations

import pytest
from import_rules import ark_imports, modules, rule_for, violations

CASES = modules()


def test_no_violations() -> None:
    """一条汇总：任何一条规则被违反都会在这里失败。"""
    problems = violations()
    assert not problems, "导入规则违规：\n" + "\n".join(f"  - {p}" for p in problems)


@pytest.mark.parametrize("path,module", CASES, ids=[m for _, m in CASES])
def test_import_direction(path, module) -> None:
    """每个模块各一条，失败时能一眼看出是哪个模块越界了。"""
    allowed = rule_for(module)
    if allowed is None:
        return
    for imported in sorted(ark_imports(path)):
        ok = any(imported == a or imported.startswith(a + ".") for a in allowed)
        assert ok, f"{module} 不许 import {imported}（允许：{sorted(allowed)}）"
