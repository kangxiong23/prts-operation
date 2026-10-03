"""边界契约的检查逻辑。

只使用标准库，所以**不装 pytest 也能跑**：

    python tests/contract/import_rules.py

pytest 版本（`test_import_rules.py`）只是这份逻辑的外壳。规则本身写在
`docs/10-规格/代码结构.md`。

这里只检查 import 关系 —— 能被机器拦住的也只有这个。命名、词汇、职责
有没有越界，仍然靠 review。
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

# 每个包的允许集合；None 表示不限制。按前缀匹配，取最长的那条。
ALLOWED: dict[str, frozenset[str] | None] = {
    "ark.contracts": frozenset({"ark.contracts"}),
    # engine 的六个子包互不 import，各自只依赖契约层。
    # 把它们接到一起是 ark.runtime.wiring 的事。
    "ark.engine.device": frozenset({"ark.contracts", "ark.engine.device"}),
    "ark.engine.vision": frozenset({"ark.contracts", "ark.engine.vision"}),
    "ark.engine.state": frozenset({"ark.contracts", "ark.engine.state"}),
    "ark.engine.timeline": frozenset({"ark.contracts", "ark.engine.timeline"}),
    "ark.engine.control": frozenset({"ark.contracts", "ark.engine.control"}),
    "ark.engine.api": frozenset({"ark.contracts", "ark.engine.api"}),
    # 唯一的例外：桥接层需要算子清单。
    "ark.engine.mcp": frozenset(
        {"ark.contracts", "ark.engine.api", "ark.engine.mcp"}
    ),
    "ark.engine": frozenset({"ark.contracts", "ark.engine"}),
    # 离线可以用线上（重放要走线上那套识别链路），反向绝不允许。
    "ark.offline": frozenset({"ark.contracts", "ark.engine", "ark.offline"}),
    # 唯一的组装点：它谁都许 import。反过来说，只有它许 import 别人。
    "ark.runtime": None,
}

MCP_PACKAGE = "ark.engine.mcp"


def modules() -> list[tuple[Path, str]]:
    """列出 src/ark 下的每个模块，连同它的点分名。"""
    found: list[tuple[Path, str]] = []
    if not SRC.is_dir():
        return found
    for path in sorted(SRC.rglob("*.py")):
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        found.append((path, ".".join(parts) if parts else "ark"))
    return found


def ark_imports(path: Path, module: str) -> set[str]:
    """这个模块 import 了哪些 ark.* 名字。

    对 `from X import a` 记的是 **`X.a` 而不是 `X`**。只记 `X` 的话，
    `from ark import contracts` 会被误报成「不许 import ark」，
    把一种完全正常的写法全杀掉。

    **相对 import 要折算成绝对名再判。** 曾经把它整个跳过（"相对 import 不算"），
    那是个洞：`ark/contracts/__init__.py` 里写 `from ..engine import x` 是**逃出这一层**，
    而跳过等于放行。折算靠 `module`（文件自己的点分名）与 `path`（是不是 `__init__`）。

    用 utf-8-sig 读：Windows 上的编辑器与 PowerShell 常给文件加 BOM，
    而带 BOM 的文本交给 ast.parse 会抛 SyntaxError。

    已知边界：`from ark import engine` 之后再写 `engine.vision.x()` 查不出来，
    因为那是属性访问而不是 import。这是 import 分析的固有限制。
    """
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _absolute_base(module, path, node)
            if base is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(f"{base}.{alias.name}")
    return {n for n in names if n == "ark" or n.startswith("ark.")}


def _absolute_base(module: str, path: Path, node: ast.ImportFrom) -> str | None:
    """把一条 `from ... import` 的模块部分折算成绝对名。

    `level == 0` 本来就是绝对名。相对的那几种：`level=1` 是本包，每多一级上跳一层。
    上跳越过 `ark` 就返回 None（那不是本项目的事）。
    """
    if node.level == 0:
        return node.module
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    parts = package.split(".") if package else []
    keep = len(parts) - (node.level - 1)
    if keep <= 0:
        return None
    base = ".".join(parts[:keep])
    return f"{base}.{node.module}" if node.module else base


def rule_for(module: str) -> frozenset[str] | None:
    """取最长前缀匹配的那条规则的允许集合。"""
    best: str | None = None
    for prefix in ALLOWED:
        # 命中且比已选的更长（前缀匹配取最长的那条）
        if (module == prefix or module.startswith(prefix + ".")) and (
            best is None or len(prefix) > len(best)
        ):
            best = prefix
    # 这里 best 只可能是 None 或非空字符串，所以直接用它做布尔判断
    return ALLOWED[best] if best else None


def violations() -> list[str]:
    """返回所有违规，每条一句话。空列表表示全过。"""
    problems: list[str] = []
    cases = modules()

    # 一、依赖方向
    for path, module in cases:
        allowed = rule_for(module)
        if allowed is None:
            continue
        try:
            imported_names = sorted(ark_imports(path, module))
        except SyntaxError as exc:
            # 解析不了就报出来，不要崩。崩了会让人以为检查通过。
            problems.append(f"{module} 无法解析，导入规则未能检查：{exc}")
            continue
        for imported in imported_names:
            ok = any(imported == a or imported.startswith(a + ".") for a in allowed)
            if not ok:
                problems.append(
                    f"{module} 不许 import {imported}（允许：{sorted(allowed)}）"
                )

    # 二、只有桥接层可以带 MCP 字样
    for _, module in cases:
        if (
            module.startswith("ark.")
            and "mcp" in module.lower()
            and not module.startswith(MCP_PACKAGE)
        ):
            problems.append(f"只有 {MCP_PACKAGE} 可以提到 MCP，违规：{module}")

    # 三、engine 的子包必须逐个枚举
    # 不枚举的话，新子包会落到 `ark.engine` 那条宽松规则上，等于默认拿到
    # 访问整个 engine 的权限 —— 而这恰恰是本结构要防止的事。
    engine = SRC / "ark" / "engine"
    if engine.is_dir():
        for child in sorted(engine.iterdir()):
            if child.is_dir() and (child / "__init__.py").exists():
                name = f"ark.engine.{child.name}"
                if name not in ALLOWED:
                    problems.append(f"engine 子包未在 ALLOWED 中枚举：{name}")

    return problems


def main() -> int:
    cases = modules()
    problems = violations()
    if not problems:
        print(f"导入规则全部通过（检查了 {len(cases)} 个模块）")
        return 0
    print(f"发现 {len(problems)} 处违规：")
    for item in problems:
        print(f"  - {item}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
