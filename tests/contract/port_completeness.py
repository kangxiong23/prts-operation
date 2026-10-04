"""端口完备性：`contracts` 里定义的四个端口，在装配那一刻各有一个实现者。

只使用标准库，所以**不装 pytest 也能跑**：

    python tests/contract/port_completeness.py

## 它凭什么能判

端口是 `Protocol`，而 `Protocol` **不是** `runtime_checkable` —— 那是契约层
的事，不给它加装饰器。所以「有没有实现」不能问 `isinstance`，只能**按方法在不在**
判：拿端口自己声明的那几个方法名，逐个去实现者身上找。

## 它要分清「按设计是空的」与「忘了接」

这是它存在的**全部理由**。资产端口**方法集为空**、样本端口的空实现**什么都不做**
—— 那些是设计；而设备端口少两个方法，是因为**签名还没裁定**。两件事在「有没有实现」
这个问题下长得一模一样，所以欠账登记在 `ark.runtime.wiring.PORT_GAPS`：
**方法缺了，必须在登记里找得到它；登记里写了、而方法其实在，也报** —— 两边都要对得上。

规则本身写在 `docs/10-规格/代码结构.md`（§尚未落地）。它盯的是**装配那一刻**，
所以这里真的调用 `build_wiring` —— 而装配**不发任何命令**（构造一个 transport
不等于跑一条命令）。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ark.contracts import AssetPort, ClockPort, DevicePort, SamplePort  # noqa: E402
from ark.engine.device.adb import AdbTransport  # noqa: E402
from ark.runtime.wiring import PORT_GAPS, build_wiring  # noqa: E402

PORTS: dict[str, type] = {
    "DevicePort": DevicePort,
    "ClockPort": ClockPort,
    "SamplePort": SamplePort,
    "AssetPort": AssetPort,
}
"""四个端口，名字就是公开面上的名字（契约层 §5 四个）。"""

FIELDS: dict[str, str] = {
    "DevicePort": "device",
    "ClockPort": "clock",
    "SamplePort": "samples",
    "AssetPort": "assets",
}
"""端口 → `Wiring` 上那个字段。字段名与端口名不同（设备端口叫 `device`，
样本端口叫 `samples`），所以这张表不能省。"""


def method_names(port: type) -> tuple[str, ...]:
    """端口自己声明的那几个方法名。**从契约层取，不在这里抄一份。**"""
    return tuple(
        sorted(
            name
            for name, value in vars(port).items()
            if not name.startswith("_") and callable(value)
        )
    )


def public_methods(impl: object) -> tuple[str, ...]:
    """实现者身上的公开方法名（可调用的那些，下划线开头的不算）。"""
    return tuple(
        sorted(
            name
            for name in dir(impl)
            if not name.startswith("_") and callable(getattr(impl, name, None))
        )
    )


def gaps_of(entry: str) -> list[str]:
    """这条登记说的是哪个端口方法（在四个端口的全部方法名里找）。"""
    found: list[str] = []
    for port in PORTS.values():
        for name in method_names(port):
            if name in entry and name not in found:
                found.append(name)
    return found


def check() -> list[str]:
    """返回所有问题，每条一句话。空列表表示全过。"""
    problems: list[str] = []
    wiring = build_wiring(AdbTransport())
    implementations: dict[str, object | None] = {}
    for port_name, port in PORTS.items():
        impl = getattr(wiring, FIELDS[port_name], None)
        implementations[port_name] = impl
        if impl is None:
            problems.append(f"{port_name} 在 Wiring.{FIELDS[port_name]} 上没有实现者")
            continue
        declared = method_names(port)
        present = public_methods(impl)
        for name in declared:
            if name in present:
                continue
            if not any(name in entry for entry in PORT_GAPS):
                problems.append(
                    f"{port_name}.{name} 没有实现，也没登记进 PORT_GAPS —— "
                    "「忘了接」与「按设计留空」就是这么分开的"
                )
        for name in present:
            if name not in declared:
                problems.append(
                    f"{port_name} 的实现者 {type(impl).__name__} 多出了 {name}："
                    "端口方法集之外的公开方法要先把契约层改了（补法规则）"
                )
    for entry in PORT_GAPS:
        names = gaps_of(entry)
        if not names:
            problems.append(f"PORT_GAPS 这一条没有点名任何端口方法：{entry!r}")
        for name in names:
            for port_name, port in PORTS.items():
                if name not in method_names(port):
                    continue
                impl = implementations[port_name]
                if impl is not None and name in public_methods(impl):
                    problems.append(
                        f"PORT_GAPS 登记了 {name}，但它其实已经接上了 —— "
                        "登记要说的是「没接的是哪几条」"
                    )
    return problems


def main() -> int:
    wiring = build_wiring(AdbTransport())
    problems = check()
    print("端口完备性：contracts 里定义的四个端口，在装配那一刻各有一个实现者")
    for port_name, port in PORTS.items():
        impl = getattr(wiring, FIELDS[port_name], None)
        declared = method_names(port)
        name = type(impl).__name__ if impl is not None else "（没有）"
        if not declared:
            note = "方法集为空 —— 这一层按设计就是空的"
        else:
            missing = (
                [m for m in declared if m not in public_methods(impl)]
                if impl is not None
                else list(declared)
            )
            if missing:
                note = (
                    f"接上 {len(declared) - len(missing)} · "
                    f"登记欠账 {len(missing)}（{' · '.join(missing)}）"
                )
            else:
                note = f"接上 {len(declared)}"
        print(f"  {port_name:<11} 实现者 {name:<16} 方法 {len(declared)} 个：{note}")
    print(f"PORT_GAPS {len(PORT_GAPS)} 条")
    if problems:
        print(f"\n不成立 {len(problems)} 处：")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("四个端口都有实现者；没接的都在 PORT_GAPS 里写下来了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
