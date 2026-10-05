"""倒计时数字那一列的规格符合性验收 —— **由规划者维护，不住在 tests/unit/**。

分工、纪律、跑法都在 [README](README.md) —— **不在这里重述**（同一条规定只写一次）。

## 这一批要回答什么

备战席卡片正中那串数字（`54.7`）现在**读不出来**：它用的不是出厂字形表那一套
（[视觉层/识别方法](../../docs/10-规格/视觉层/识别方法.md) §已定 —— `1` 带底衬线、笔画更粗）。
所以链上把它如实落**未知**。这一批是**那套粗体字形的取材来源 + 验收材料**，
真值表在下面 `TRUTH`。**执行者不许读本文件。**

## 真值从哪来

`tests/帧样本/倒计时数字/` 那 4 张原生裁剪，规划者**逐张目视确认**过：
**卡数 · 哪几张在再部署中 · 每张的秒数**。秒数在 3× 放大下逐位读过，
并且与**环上白弧的「已过比例」**交叉核过（`(1 − 弧/360) × 70 s` 与读到的秒数在
全部 22 张上相差 ≤ 0.7 s）—— 两边独立，所以是核对、不是同义反复。

## 这一批的硬条件：十个数字占全

**素材不占全十位，这套字形就建不出来。** 所以 D4 是这一批的存在理由，不是锦上添花。
`7` · `8` · `9` 三位最稀（取材各一个），而它们**在验收帧里又各出现一次** ——
最薄的三位正好用来验"会不会 generalize"。
"""

from __future__ import annotations

import importlib
import json
import struct
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _entry in (str(REPO / "src"), str(_HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

BATCH = REPO / "tests" / "帧样本" / "倒计时数字"
GLYPH_DIR = REPO / "assets" / "features" / "glyphs"

CANVAS_W, CANVAS_H = 2800, 1260
DIGITS = "0123456789."


# ---------------------------------------------------------------------------
# 真值：规划者逐张目视确认（4 帧）
#
# `countdown` 是"再部署中"的下标 → 中央那个秒数（从右往左，下标 0 = 最右那张）。
# `cards` 是这一帧数出来几张卡。
# ---------------------------------------------------------------------------

TRUTH: Mapping[str, Mapping[str, object]] = {
    "cd01": {
        "cards": 9,
        "countdown": {0: 32.2, 1: 39.5, 2: 54.7, 5: 36.0, 7: 64.2, 8: 16.0},
    },
    "cd09": {
        "cards": 10,
        "countdown": {0: 9.5, 1: 16.8, 2: 32.0, 3: 67.0, 4: 66.4, 5: 54.2, 6: 13.3, 8: 41.5},
    },
    "cd20": {
        "cards": 11,
        "countdown": {3: 34.3, 4: 68.4, 5: 33.6, 6: 21.4, 9: 8.8},
    },
    "cd30": {
        "cards": 11,
        "countdown": {3: 4.0, 4: 38.1, 5: 3.4},
    },
}


# ---------------------------------------------------------------------------
# 骨架
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """一条规格检查（形状与 `bench_row_spec.py` 那张表一样）。"""

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
    return [Path(c["file"]).stem for c in _index()["crops"]]


def _entry(name: str) -> dict:
    for e in _index()["crops"]:
        if Path(e["file"]).stem == name:
            return e
    raise AssertionError(f"夹具里没有 {name}")


def _raw(name: str) -> bytes:
    """把夹具裁剪**贴回** 2800×1260 的黑底，编成设备层 §4 定义的原始 RGBA 字节。

    这样 ROI 的取法仍然是真的在被测 —— 没有把 ROI 预先替读数链做好。
    """
    e = _entry(name)
    x, y = e["origin"]
    w, h = e["size"]
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    canvas[y : y + h, x : x + w] = np.asarray(
        Image.open(BATCH / e["file"]).convert("RGB")
    )
    rgba = np.dstack([canvas, np.full((CANVAS_H, CANVAS_W), 255, np.uint8)])
    return struct.pack("<4I", CANVAS_W, CANVAS_H, 1, 1) + rgba.tobytes()


_READ_CACHE: dict[str, tuple] = {}


def _read(name: str):
    """走设备层的重放路径读一帧：字节 → 帧 → 读数链。同一帧只读一次。"""
    if name not in _READ_CACHE:
        frame = _mod("ark.engine.vision.frame").frame_from_raw(_raw(name))
        glyphs = _mod("ark.engine.vision.glyphs").GlyphTable.load(GLYPH_DIR)
        pitch = _mod("ark.engine.device.adb").SQUAD_CARD_PITCH
        _READ_CACHE[name] = _mod("ark.engine.vision.bench").read_squad_row(
            frame, glyphs, pitch=pitch
        )
    return _READ_CACHE[name]


# ---------------------------------------------------------------------------
# D 族：这一批长什么样
# ---------------------------------------------------------------------------


def check_batch_loads_with_its_manifest() -> str:
    """`index.json` 里那几样必填字段，加载器一样不少地读出来（[基准集] §2）。"""
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
    """每条裁剪的原点与尺寸都落在画布内，含两端口径（[基准集] §2）。"""
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
    """真值表覆盖每一条裁剪、不多不少（[基准集] §1：期望值不住基准集里）。"""
    names = _frame_names()
    missing = [n for n in names if n not in TRUTH]
    extra = [n for n in TRUTH if n not in names]
    assert not missing, f"这几帧没有真值：{missing}"
    assert not extra, f"真值表里这几帧不在批次里：{extra}"
    return f"{len(names)} 帧，真值表逐帧对齐"


def check_the_truth_covers_every_digit() -> str:
    """**这一批的存在理由**：那串数字的十个字形与小数点，素材里一位都不许缺。

    出处是[视觉层/识别方法] §已定（"数字一律走固定字形模板"）与它的 §已知的缺口
    （字形表是照着样本建的）—— 一位没有样本，那一位就建不出来。

    **数字从真值里算出来，不手抄**（[README] §写一条检查的三条纪律 3）。
    """
    seen: dict[str, list[str]] = {}
    for name, row in TRUTH.items():
        for index, seconds in row["countdown"].items():
            for char in f"{seconds:.1f}":
                seen.setdefault(char, []).append(f"{name}#{index}")
    missing = sorted(set(DIGITS) - set(seen))
    assert not missing, (
        f"这几位的字形一张样本都没有：{missing} —— 这一批就是为了它们建的"
    )
    thin = sorted(c for c in DIGITS if len(seen[c]) <= 1)
    every = " ".join(f"{c}×{len(seen[c])}" for c in DIGITS)
    return f"{len(DIGITS)} 位都占到了 ｜ {every} ｜ 只有一张样本的：{thin}"


def check_the_truth_table_has_the_shape_it_claims() -> str:
    """真值表自己的形状：下标在卡数之内，秒数是**一位小数**、不是浮点尾巴。

    出处是[状态层/我方干员] 那一行（再部署倒计时是"精确"那种带小数的量）与
    [视觉层/识别方法] §已定（数字只有 0-9 与小数点）。这一条不碰像素 ——
    它保证**真值本身不自相矛盾**，真值错了验收就一起错。
    """
    for name, row in TRUTH.items():
        cards = row["cards"]
        for index, seconds in row["countdown"].items():
            assert 0 <= index < cards, f"{name}：下标 {index} 不在 0..{cards - 1}"
            text = f"{seconds:.1f}"
            assert abs(float(text) - seconds) < 1e-9, (
                f"{name}#{index} 的秒数 {seconds!r} 不是一位小数"
            )
            assert len(text) in (3, 4), f"{name}#{index} 的秒数写成 {text!r}，位数不对"
    total = sum(len(row["countdown"]) for row in TRUTH.values())
    return f"{len(TRUTH)} 帧 · 共 {total} 个倒计时读数，形状都自洽"


def check_a_countdown_is_only_read_from_a_redeploying_card() -> str:
    """前提：**有倒计时的那些卡必须先被判成"再部署中"**，秒数那一列才轮得到读。

    出处是[视觉层/备战席] §判死与再部署（"红罩 + 那个圆环在，就是再部署中"）与
    `read_squad_row` 的口径：不在再部署中的卡，秒数**一律 `Unsampled()`**。

    **这一条不是这一批的正事**，它是 D7 的前提 —— 前提塌了，D7 的红会指向错的地方。
    它用的是**两帧从没进过任何取材表的素材**（`cd09` · `cd20`），所以它同时是
    那条判据的一次**样本外**抽验。
    """
    wrong: list[str] = []
    for name, row in TRUTH.items():
        cards = _read(name)
        for card in cards:
            want = card.index in row["countdown"]
            if card.redeploying != want:
                said = {True: "再部署中", False: "不在再部署", None: "判不了"}[
                    card.redeploying
                ]
                wrong.append(
                    f"{name} 下标 {card.index} 读成「{said}」、真值是"
                    f"「{'再部署中' if want else '不在再部署'}」"
                )
    assert not wrong, f"{len(wrong)} 处不符 ｜ " + "；".join(wrong)
    total = sum(len(row["countdown"]) for row in TRUTH.values())
    return f"{len(TRUTH)} 帧逐张一致（其中 {total} 张在再部署中）"


def check_the_seconds_read_exactly() -> str:
    """**这一批的正事**：中央那串数字要逐位读出来，**精确等于真值**。

    出处是[状态层/我方干员] 那一行的三档（精确 / 未知 / 未采样）与
    [视觉层/备战席] §尚未定（"欠的是资产，不是链"）。

    **这一条现在是红的，而且红得对**：那套**粗体字形资产还没建**
    （`1` 带底衬线、最像的字形反而是 `2`），所以链上如实落**未知**。
    这一条抓的不是假事实、是**欠账** —— 资产落地的那一天它就绿。
    它同时钉住反面：**不许为了让这一条变绿而猜一个数**（猜错比报未知危险）。
    """
    from ark.contracts import Exact, Unknown, Unsampled, Valued

    detail: list[str] = []
    for name, row in TRUTH.items():
        cards = _read(name)
        for index, want in sorted(row["countdown"].items()):
            slot = cards[index].seconds
            if isinstance(slot, Unsampled):
                detail.append(f"{name}#{index}=未采样")
                continue
            assert isinstance(slot, Valued), f"{name}#{index}：秒数落成了 {slot!r}"
            payload = slot.payload
            if isinstance(payload, Unknown):
                detail.append(f"{name}#{index}=未知")
                continue
            assert isinstance(payload, Exact), f"{name}#{index}：{payload!r}"
            assert abs(payload.value - want) < 0.05, (
                f"{name}#{index}：秒数读成 {payload.value}，真值是 {want} "
                "—— 猜错了比报未知危险"
            )
            detail.append(f"{name}#{index}={payload.value}")
    unread = [d for d in detail if d.endswith("未知") or d.endswith("未采样")]
    assert not unread, (
        f"{len(unread)}/{len(detail)} 个倒计时没读出来："
        + "、".join(unread)
        + " —— 那套粗体字形资产还没建（[视觉层/备战席] §尚未定）"
    )
    return f"{len(detail)} 个倒计时全部精确读出 ｜ " + " ".join(detail)


CHECKS: tuple[Check, ...] = (
    Check("D1", "采集与验收/基准集 §2（字段表）",
          "清单里的必填字段一样不少地读出来，没记的前提头不许补",
          check_batch_loads_with_its_manifest),
    Check("D2", "采集与验收/基准集 §2（含两端口径）",
          "每条裁剪的原点与尺寸都落在画布内",
          check_every_crop_stays_inside_the_canvas),
    Check("D3", "采集与验收/基准集 §1（期望值不住这里）",
          "真值表覆盖每一条裁剪，不多不少",
          check_truth_covers_exactly_the_batch),
    Check("D4", "视觉层/识别方法 §已定（字形模板）+ §已知的缺口（表是照样本建的）",
          "那串数字的十个字形与小数点，这一批里一位都不缺",
          check_the_truth_covers_every_digit),
    Check("D5", "状态层/我方干员（倒计时的形状）+ 视觉层/识别方法 §已定",
          "真值表自己的形状自洽：下标在卡数内、秒数是一位小数",
          check_the_truth_table_has_the_shape_it_claims),
    Check("D6", "视觉层/备战席 §判死与再部署（有环才轮得到读秒）",
          "有倒计时的那些卡先被判成再部署中，秒数那一列才轮得到读",
          check_a_countdown_is_only_read_from_a_redeploying_card),
    Check("D7", "状态层/我方干员（精确 / 未知 / 未采样）+ 视觉层/备战席 §尚未定",
          "中央那串数字逐位读出、精确等于真值 —— 现在是欠账，不是假事实",
          check_the_seconds_read_exactly),
)


def main() -> int:
    failures: list[tuple[Check, str]] = []
    print(f"倒计时数字那一列的规格符合性验收 —— {len(CHECKS)} 条\n")
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
