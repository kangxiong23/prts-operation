"""跑一批基准集：逐帧调费用那两个读数器，打印一张表 + 落一份 JSON。

    .venv\\Scripts\\python.exe scripts\\bench.py

**它走离线那条路，不碰真机**（代码结构 · 设备层 §5）：

    基准集目录 → 加载器 → 一帧一帧的原始帧 → 重放适配器 → 识别链

这个脚本自己**不解 PNG**、也不拼任何设备命令 —— 那两件事分别住在
`offline/bench/` 与 `engine/device/` 里（基准集 §3）。

## 打出来的那些行

**一行一帧，没有表头、没有汇总** —— 这一批是 31 帧，打出来就是 31 行。每行的列
依次是：**帧名 · 费用数字的槽 · 费用条的槽**。槽用它的原始形态写出来
（`Valued(payload=Exact(value=99), confidence=0.9975)` 这种），所以"三态落在
哪一支、载荷是什么形态、置信度多少"一眼看得见 —— 形态合并了就没法比对。

## 落的那一份 JSON

`data/bench/cost.json`（`data/` 是 gitignored 的运行时数据）。**同一批跑两次，
内容逐字节一样**：没有时间戳、没有绝对路径、没有随机量 —— 不然"和上次比一比"
这句话就没有意义了（离线验收 §3 的回归比对：判据是假事实数不许涨，证据是逐字段
并排留档）。

## 退出码

`1` 只给"这条链坏了"：清单读不出来 · 裁剪与清单对不上 · 帧解不出来 ·
写不下报告。**读不出费用不算失败** —— 那是"看不清"，不是"看错了"。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
"""仓库根。**包还没装**（`pyproject.toml` 的 `[project.scripts]` 还是注释状态），
所以这个脚本得自己把 `src/` 放进 `sys.path` —— 与 `scripts/e2e_device_read.py`
和 `data/probe/` 里那些脚本一致（代码结构 §尚未落地）。
"""
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.contracts import (  # noqa: E402
    DeviceFacts,
    Exact,
    Interval,
    LowerBound,
    Region,
    SampledEmpty,
    Unknown,
    Unsampled,
    UpperBound,
    Valued,
)
from ark.engine.vision.cost import read_cost_bar, read_cost_digits  # noqa: E402
from ark.engine.vision.frame import frame_from_raw  # noqa: E402
from ark.engine.vision.glyphs import DEFAULT_GLYPHS_PATH, GlyphTable  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402
from ark.offline.bench.feed import replay_frames  # noqa: E402

BATCH = REPO / "tests" / "帧样本" / "费用"
"""默认跑的那一批：第一批基准集 `v1`（基准集 §5）。

**它是拿命令行第一个参数换得掉的**，但默认值写在这里、不靠"当前目录" ——
换一批就是换一个目录，不是换一个工作目录。
"""

REPORT = REPO / "data" / "bench" / "cost.json"
"""报告的落点。**文件名里没有时间戳** —— 同一批跑两次要能逐字节比。"""


def _slot_json(slot: object) -> dict[str, object]:
    """一个槽 → 能进 JSON 的形状。**三态与六种载荷各自照实写，不许合并**：
    把 `Unknown` 和 `Unsampled` 写成同一个字符串，"和上次比一比"就比不出来了。
    """
    if isinstance(slot, Unsampled):
        return {"state": "unsampled"}
    if isinstance(slot, SampledEmpty):
        return {"state": "empty", "confidence": slot.confidence}
    if isinstance(slot, Valued):
        return {
            "state": "valued",
            "payload": _payload_json(slot.payload),
            "confidence": slot.confidence,
        }
    raise TypeError(f"槽只有三态，收到 {type(slot).__name__}")


def _payload_json(payload: object) -> dict[str, object]:
    """载荷 → 能进 JSON 的形状（契约层 §1 的六种）。"""
    if isinstance(payload, Exact):
        return {"form": "exact", "value": payload.value}
    if isinstance(payload, Interval):
        return {"form": "interval", "lo": payload.lo, "hi": payload.hi}
    if isinstance(payload, LowerBound):
        return {"form": "lower_bound", "lo": payload.lo}
    if isinstance(payload, UpperBound):
        return {"form": "upper_bound", "hi": payload.hi}
    if isinstance(payload, Unknown):
        return {"form": "unknown"}
    if isinstance(payload, Region):
        return {
            "form": "region",
            "center": [payload.center.x, payload.center.y],
            "radius": payload.radius,
        }
    raise TypeError(f"载荷只有六种，收到 {type(payload).__name__}")


def _facts_json(facts: DeviceFacts | None) -> dict[str, object] | None:
    """前提头 → 能进 JSON 的形状。**没记就是 `null`**（不许补一个）。

    它是加载器折好的那个 `DeviceFacts`（基准集 §2），所以两个字段都照实落：
    `resolution` 与 `orientation`。费用那一批没有前提头，落出来就是 `null`。
    """
    if facts is None:
        return None
    return {
        "resolution": [facts.resolution.width, facts.resolution.height],
        "orientation": facts.orientation,
    }


def _batch_label(root: Path) -> str:
    """报告里写的这一批是谁：**相对仓库根的路径**，不是绝对路径。

    绝对路径会让同一批在两台机器上落出两份不同的报告（离线验收 §3 的回归比对
    要的是"同一台机器上比一比"，但绝对路径连这一点都做不到 —— 换个检出目录就变）。
    """
    try:
        return root.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return root.name


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else BATCH

    try:
        batch = load_batch(root)
        glyphs = GlyphTable.load(DEFAULT_GLYPHS_PATH)
    except Exception as bad:  # noqa: BLE001 —— 清单或出厂资产读不出来 = 这条链坏了
        print(f"基准集读不出来：{type(bad).__name__}: {bad}", file=sys.stderr)
        return 1

    rows: list[dict[str, object]] = []
    for case, raw in replay_frames(batch):
        try:
            frame = frame_from_raw(raw)
        except Exception as bad:  # noqa: BLE001 —— 帧解不出来 = 这条链坏了
            print(f"解帧失败（{case.name}）：{type(bad).__name__}: {bad}", file=sys.stderr)
            return 1
        digits = read_cost_digits(frame, glyphs)
        bar = read_cost_bar(frame)
        rows.append(
            {"name": case.name, "digits": _slot_json(digits), "bar": _slot_json(bar)}
        )
        print(f"{case.name}  数字 {digits!r}  条 {bar!r}")

    report = {
        "batch": _batch_label(batch.root),
        "version": batch.version,
        "canvas": [batch.canvas.width, batch.canvas.height],
        "device": batch.device,
        "facts": _facts_json(batch.facts),
        "cases": rows,
    }
    try:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except Exception as bad:  # noqa: BLE001 —— 报告落不下去也是这条链坏了
        print(f"报告写不下去（{REPORT}）：{type(bad).__name__}: {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
