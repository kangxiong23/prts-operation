"""备战席那一行的规格符合性验收 —— **由规划者维护，不住在 tests/unit/**。

分工、纪律、跑法都在 [README](README.md) —— **不在这里重述**（同一条规定只写一次）。

## 真值从哪来

`tests/帧样本/备战席/` 的 6 张原生裁剪，规划者**逐帧目视确认**过三样：
**卡数 · 每张的费用 · 每张的状态**。真值表在下面 `TRUTH`。**执行者不许读本文件。**

## 这一批覆盖到什么

- **卡数六档**：12 · 11 · 10 · 9 · 8 · 6；
- **费用一位数与两位数**：一位数 1 · 2 · 6，两位数 10–25；
- **装置卡**：费用写 14 的那张，它不是干员（[视觉层/备战席点名](../../docs/10-规格/视觉层/备战席点名.md) §装置那一张长什么样）；
- **再部署中那一档**：`shot26` 最右那张 —— 红罩 + 白色环形进度 + 中央 `12.9`。

**没覆盖到**：灰掉那一档（费用不足）、空席（一张卡都没有）、3 位数费用。

## 还没写的是哪一半

读数链（`engine/vision/bench.py`）**还不存在**，所以这一层现在只有**真值 + 批次完整性**；
"读出来与真值全对"那一条要等读数链落地，**落进本文件，不另开一份**。
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _entry in (str(REPO / "src"), str(_HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

BATCH = REPO / "tests" / "帧样本" / "备战席"

CANVAS_W, CANVAS_H = 2800, 1260


# ---------------------------------------------------------------------------
# 真值：规划者逐帧目视确认（6 帧）
#
# `cost` 是**从右往左**排的：下标 0 = 最右那张（代码里 `card` 0 的约定）。
# `device` 是"不是干员"的那些下标；`redeploy` 是"再部署中"的下标 → 中央那个秒数。
# ---------------------------------------------------------------------------

TRUTH: Mapping[str, Mapping[str, object]] = {
    "shot01": {
        "cards": 12,
        "cost": [25, 23, 23, 21, 18, 17, 14, 13, 13, 12, 12, 11],
        "device": [6],
        "redeploy": {},
    },
    "shot02": {
        "cards": 10,
        "cost": [25, 18, 17, 14, 13, 13, 12, 12, 11, 10],
        "device": [3],
        "redeploy": {},
    },
    "shot05": {
        "cards": 9,
        "cost": [18, 17, 14, 14, 13, 13, 12, 11, 10],
        "device": [2],
        "redeploy": {},
    },
    "shot07": {
        "cards": 8,
        "cost": [18, 17, 14, 14, 13, 12, 11, 10],
        "device": [2],
        "redeploy": {},
    },
    "shot21": {
        "cards": 11,
        "cost": [25, 23, 23, 17, 14, 13, 13, 12, 12, 11, 10],
        "device": [4],
        "redeploy": {},
    },
    "shot26": {
        "cards": 6,
        "cost": [18, 12, 11, 6, 2, 1],
        "device": [],
        "redeploy": {0: 12.9},
    },
}


# ---------------------------------------------------------------------------
# 骨架
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """一条规格检查。

    - `ident`：稳定编号，报告与 pytest 用例名都用它；
    - `spec`：规格出处，具体到节；
    - `claim`：规格里的那句话，一句话；
    - `probe`：返回**证据字符串**；不成立时抛 AssertionError，消息就是证据。
    """

    ident: str
    spec: str
    claim: str
    probe: Callable[[], str]


_LOADED: dict[str, object] = {}


def _mod(dotted: str):
    """import 一个还不一定存在的模块。不存在 = 这条检查红，不是检查炸了。"""
    if dotted not in _LOADED:
        try:
            _LOADED[dotted] = importlib.import_module(dotted)
        except BaseException as exc:  # noqa: BLE001 —— 缺模块就是这批检查的红法
            raise AssertionError(
                f"`{dotted}` 还不存在：{type(exc).__name__}: {exc}"
            ) from None
    return _LOADED[dotted]


def _index() -> dict:
    return json.loads((BATCH / "index.json").read_text(encoding="utf-8"))


def _frame_names() -> list[str]:
    """裁剪的文件名去掉扩展名 = 帧名（[基准集] §2）。"""
    return [Path(c["file"]).stem for c in _index()["crops"]]


# ---------------------------------------------------------------------------
# R 族：这一批长什么样
# ---------------------------------------------------------------------------


def check_batch_loads_with_its_manifest() -> str:
    """`index.json` 里那几样必填字段，加载器一样不少地读出来。

    出处是[基准集] §2 的字段表。**`facts` 没写就是没有** —— 这一批同样没记前提头。
    """
    batch = _mod("ark.offline.bench.batch").load_batch(BATCH)
    raw = _index()
    assert batch.version == raw["version"] >= 1, (
        f"版本号读成了 {batch.version!r}，清单里写的是 {raw['version']!r}"
    )
    assert (batch.canvas.width, batch.canvas.height) == (CANVAS_W, CANVAS_H), (
        f"画布读成了 {batch.canvas}，清单里写的是 {raw['canvas']}"
    )
    assert isinstance(batch.device, str) and batch.device, "机型没读出来 —— 它是出处"
    assert len(batch.cases) == len(raw["crops"]), (
        f"读出来 {len(batch.cases)} 条，清单里写了 {len(raw['crops'])} 条"
    )
    assert getattr(batch, "facts", None) is None, (
        "这一批没记前提头，加载器不许替它补一个 —— 那是编出来的环境事实"
    )
    return (
        f"v{batch.version} · 画布 {batch.canvas.width}×{batch.canvas.height} · "
        f"{len(batch.cases)} 条 · facts=None"
    )


def check_every_crop_stays_inside_the_canvas() -> str:
    """每条裁剪的原点与尺寸都落在画布内，**含两端口径**。

    出处是[基准集] §2：右缘 ≤ 宽 − 1、下缘 ≤ 高 − 1。写超了就是"含了一个不存在的像素"。
    """
    raw = _index()
    for crop in raw["crops"]:
        x0, y0 = crop["origin"]
        w, h = crop["size"]
        assert x0 >= 0 and y0 >= 0, f"{crop['file']} 原点 {crop['origin']} 越出左上角"
        assert x0 + w - 1 <= CANVAS_W - 1, (
            f"{crop['file']} 右缘 {x0 + w - 1} 越出画布宽 {CANVAS_W}"
        )
        assert y0 + h - 1 <= CANVAS_H - 1, (
            f"{crop['file']} 下缘 {y0 + h - 1} 越出画布高 {CANVAS_H}"
        )
    return f"{len(raw['crops'])} 条都在画布内（含两端口径）"


def check_truth_covers_exactly_the_batch() -> str:
    """真值表覆盖每一条裁剪、不多不少。

    出处是[基准集] §1：**期望值不住基准集里**，真值住 `tests/spec/`。两边对不上时，
    多出来的是"没有帧的真值"（永远绿不了也红不了），少掉的是"没有真值的帧"（验收
    时会被静默跳过）。
    """
    names = _frame_names()
    missing = [n for n in names if n not in TRUTH]
    extra = [n for n in TRUTH if n not in names]
    assert not missing, f"这几帧没有真值：{missing}"
    assert not extra, f"真值表里这几帧不在批次里：{extra}"
    return f"{len(names)} 帧，真值表逐帧对齐"


def check_truth_table_has_the_shape_it_claims() -> str:
    """真值表自己的形状：卡数 = 费用表的长度，装置与再部署的下标都在卡数之内。

    出处是[视觉层/备战席] §卡片怎么排（**只有还没上场的才显示**，所以卡数会变）与
    §点名（**一张卡不一定是干员**）。这一条不碰像素，它保证**真值本身不自相矛盾** ——
    真值错了，拿它当标准的验收就一起错。
    """
    worst: list[str] = []
    for name, row in TRUTH.items():
        cards = row["cards"]
        cost = row["cost"]
        assert len(cost) == cards, (
            f"{name}：卡数写 {cards}，费用表有 {len(cost)} 个数"
        )
        for i in row["device"]:
            assert 0 <= i < cards, f"{name}：装置下标 {i} 不在 0..{cards - 1}"
        for i in row["redeploy"]:
            assert 0 <= i < cards, f"{name}：再部署下标 {i} 不在 0..{cards - 1}"
        worst.append(f"{name}:{cards}")
    counts = sorted({row["cards"] for row in TRUTH.values()})
    return f"卡数 {counts} ｜ " + " ".join(worst)


CHECKS: tuple[Check, ...] = (
    Check("R1", "采集与验收/基准集 §2（字段表）",
          "清单里的必填字段一样不少地读出来，没记的前提头不许补",
          check_batch_loads_with_its_manifest),
    Check("R2", "采集与验收/基准集 §2（含两端口径）",
          "每条裁剪的原点与尺寸都落在画布内",
          check_every_crop_stays_inside_the_canvas),
    Check("R3", "采集与验收/基准集 §1（期望值不住这里）",
          "真值表覆盖每一条裁剪，不多不少",
          check_truth_covers_exactly_the_batch),
    Check("R4", "视觉层/备战席 §卡片怎么排 + §点名",
          "真值表自己的形状自洽：卡数 = 费用表长度，下标都在卡数之内",
          check_truth_table_has_the_shape_it_claims),
)


def main() -> int:
    failures: list[tuple[Check, str]] = []
    print(f"备战席那一行的规格符合性验收 —— {len(CHECKS)} 条\n")
    for check in CHECKS:
        try:
            evidence = check.probe()
        except AssertionError as bad:
            failures.append((check, str(bad)))
            print(f"  ✗ {check.ident}  {check.claim}")
            print(f"      出处 {check.spec}")
            print(f"      证据 {bad}")
        except BaseException as broken:  # noqa: BLE001
            failures.append((check, f"{type(broken).__name__}: {broken}"))
            print(f"  ‼ {check.ident}  {check.claim}")
            print(f"      出处 {check.spec}")
            print(f"      检查本身炸了 {type(broken).__name__}: {broken}")
        else:
            print(f"  ✓ {check.ident}  {check.claim}")
            print(f"      {evidence}")

    print()
    if failures:
        print(f"不成立 {len(failures)} / {len(CHECKS)}：")
        for check, why in failures:
            print(f"  - {check.ident}（{check.spec}）{why}")
        return 1
    print(f"全部成立（{len(CHECKS)} 条）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
