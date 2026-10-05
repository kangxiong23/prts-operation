"""从取材帧造**倒计时那套粗体字形表**。

素材是**两批夹具**里 `取材.json` 给了真值的那几帧 —— 那是训练集：

| 批 | 帧 | 卡 |
| --- | --- | --- |
| `tests/帧样本/倒计时数字/` | `cd01` · `cd30` | 6 + 3 |
| `tests/帧样本/备战席/` | `shot26` · `cd01` · `cd08` · `cd60` | 1 + 6 + 7 + 0 |

`cd01` 两批都有、而且**源帧 sha256 一样**（同一帧），所以按 `(源帧 sha256, 卡下标)`
去重 —— 不去重就是让同一帧在「每个字形取平均」里占两票，那不是多了一份证据，
是把同一份证据数了两次。

**验收帧的值不在任何一份 `取材.json` 里**（`cd09` · `cd20` 只出现在
`index.json` 的 `crops` 里），所以这条路**读不到**它们 —— 想拿它们调表，
得先把真值手抄进来，而这一层没有那个口子。

**为什么这一步在离线侧**：它读夹具、写资产，是**生成物**；读数链只读表
（`ark.engine.vision.glyphs.load_countdown_glyphs`），不重建。所以这里能 import
线上那套切字 / 归一（`ark.engine.vision.bench.countdown_boxes`）—— 建表与读表
**共用同一份切法、同一套归一**，两边不一致是这一层最隐蔽的坏法。

**产物必须可重建**：同一批取材帧，跑两次逐像素一致（`tests/unit/test_vision_bench.py`
里有一条按字节比）。所以这里没有随机、没有时间戳、没有「取最像的那张」这种依赖
输入顺序的选择。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ark.contracts import squad_card_rect
from ark.engine.device.adb import SQUAD_CARD_PITCH
from ark.engine.vision.bench import countdown_boxes, squad_card_centres
from ark.engine.vision.frame import Frame, frame_from_raw
from ark.engine.vision.glyphs import GlyphTable, normalize
from ark.offline.bench.batch import load_batch

__all__ = [
    "build_countdown_glyphs",
    "load_countdown_harvest",
    "harvest_coverage",
    "CountdownHarvest",
    "COUNTDOWN_DEST",
    "HARVEST_BATCHES",
    "HARVEST_FILE",
    "INDEX_FILE",
]

HARVEST_FILE = "取材.json"
INDEX_FILE = "index.json"

REPO = Path(__file__).resolve().parents[4]
"""仓库根：`src/ark/offline/assets_gen/` 往上四级。"""

COUNTDOWN_DEST = REPO / "assets" / "features" / "glyphs_countdown"
"""粗体字形表的落点 —— 与 `assets/features/glyphs/` **并列**，同一个表文件名。

那个文件名住 `ark.engine.vision.glyphs.TABLE_FILENAME`（`GlyphTable.save` 自己拼），
这一层不另抄一份。
"""

HARVEST_BATCHES = (
    REPO / "tests" / "帧样本" / "倒计时数字",
    REPO / "tests" / "帧样本" / "备战席",
)
"""取材只许来自这两批。顺序是**确定**的 —— 产物可重建的前提。"""


@dataclass(frozen=True)
class CountdownHarvest:
    """一张取材卡：**哪一批 · 哪一帧 · 卡下标 · 真值秒数**。

    `seconds` 是规划者目视确认过的真值，而 `repr` 出来的那个十进制串就是这张卡上
    待切的那串字符（`12.9` → `"12.9"`）—— 小数点也算一位，切法要把它单独切出来。
    """

    batch: Path
    file: str
    index: int
    seconds: float

    @property
    def text(self) -> str:
        """这一张卡上那串字符（`12.9` → `"12.9"`）。"""
        return repr(self.seconds)

    @property
    def name(self) -> str:
        """报告里用的名字：`cd01#2` 这种。"""
        return f"{Path(self.file).stem}#{self.index}"


def load_countdown_harvest(
    batch_dirs: Sequence[Path] = HARVEST_BATCHES,
) -> tuple[CountdownHarvest, ...]:
    """两批夹具的 `取材.json` → 一张一张的取材卡，**按 (源帧 sha256, 卡下标) 去重**。

    `倒计时数字` 那一批的真值写在 `countdowns` 里，`备战席` 那一批写在
    `redeploying` 里 —— 两份的键名不同，但说的是同一件事（这张卡剩多少秒）。

    顺序：批按给定次序、批内按 `取材.json` 里那一条一条的次序。同一批取材帧跑两次，
    切出来的样本序列一模一样。
    """
    seen: dict[tuple[str, int], CountdownHarvest] = {}
    for batch_dir in batch_dirs:
        batch_dir = Path(batch_dir)
        index = json.loads((batch_dir / INDEX_FILE).read_text(encoding="utf-8"))
        by_file = {entry["file"]: entry["source_sha256"] for entry in index["crops"]}
        truth = json.loads((batch_dir / HARVEST_FILE).read_text(encoding="utf-8"))
        for entry in truth["harvest"]:
            file_name = entry["file"]
            items = entry.get("countdowns")
            if items is None:
                items = entry.get("redeploying") or []
            for item in items:
                key = (by_file[file_name], int(item["index"]))
                if key in seen:
                    continue
                seen[key] = CountdownHarvest(
                    batch=batch_dir,
                    file=file_name,
                    index=int(item["index"]),
                    seconds=float(item["seconds"]),
                )
    return tuple(seen.values())


def harvest_coverage(harvest: Sequence[CountdownHarvest]) -> dict[int, int]:
    """每个数字有几张样本 —— **从真值串数出来的，不切图**（报告与自检用）。"""
    counts: dict[int, int] = {}
    for item in harvest:
        for character in item.text:
            if character != ".":
                counts[int(character)] = counts.get(int(character), 0) + 1
    return counts


def _frames_of(batch_dir: Path) -> dict[str, Frame]:
    """一批夹具 → `{帧名: 帧}`（帧名不带扩展名，与 `crops[].file` 的 stem 一致）。

    走的是**读数链一样的那条路**：`load_batch` 把裁剪贴回画布、`frame_from_raw` 解成
    一帧 —— 建表与读表拿到的像素因此是同一份。
    """
    return {
        case.name: frame_from_raw(case.raw()) for case in load_batch(batch_dir).cases
    }


def build_countdown_glyphs(
    harvest: Sequence[CountdownHarvest],
    dest: Path = COUNTDOWN_DEST,
    *,
    pitch: float = SQUAD_CARD_PITCH,
) -> Path:
    """从取材卡造粗体字形表，写到 `dest`，返回落盘的文件路径。

    `pitch` 是卡距 —— 它住设备层的 `SQUAD_CARD_PITCH`，这里**照读**（不另抄一个数）。
    卡中心走 `squad_card_centres`、切字走 `countdown_boxes`：**与读数链同一份实现**。

    **切出来的块数必须等于位数**：不等就报错，不猜。切错了还硬造一张表，等于把一次
    错误固化进资产，而且从此看不出来。

    只有「一个数字都没覆盖到」是**致命**的：表里少一个键，那个数字就永远读不出来，
    而表看起来仍然是好的。
    """
    frames: dict[Path, dict[str, Frame]] = {}
    card_top = float(squad_card_rect(0, pitch).y)
    examples: dict[int, list[np.ndarray]] = {}
    for item in harvest:
        if item.batch not in frames:
            frames[item.batch] = _frames_of(item.batch)
        frame = frames[item.batch][Path(item.file).stem]
        centres = squad_card_centres(frame, pitch=pitch)
        if item.index >= len(centres):
            raise ValueError(
                f"{item.name}：这一帧只读出 {len(centres)} 张卡，"
                f"取材表里却写着下标 {item.index}"
            )
        ink, _, digits, dots = countdown_boxes(frame, centres[item.index][0], card_top)
        text = item.text
        wanted = [character for character in text if character != "."]
        if len(dots) != 1:
            raise ValueError(
                f"{item.name}（{text}）：环心里切出 {len(dots)} 块小数点，"
                "切错了不许硬造表"
            )
        dot = dots[0]
        whole = [box for box in digits if box[2] <= dot[0]]
        fraction = [box for box in digits if box[0] >= dot[2]]
        if not whole or len(fraction) != 1:
            raise ValueError(
                f"{item.name}（{text}）：整数位 {len(whole)} 块、小数位 {len(fraction)} 块，"
                "切错了不许硬造表"
            )
        ordered = whole + fraction
        if len(ordered) != len(wanted):
            raise ValueError(
                f"{item.name}（{text}）：{len(wanted)} 位却切出 {len(ordered)} 块笔画："
                "切错了不许硬造表"
            )
        for character, box in zip(wanted, ordered):
            examples.setdefault(int(character), []).append(
                normalize(ink[box[1]:box[3], box[0]:box[2]])
            )

    missing = sorted(set(range(10)) - set(examples))
    if missing:
        raise ValueError(
            f"取材卡覆盖不到这几个数字：{missing} —— 表里少一个键，"
            "那个数字就永远读不出来，而表看起来仍然是好的"
        )

    # normalize 给的是 0..1 的浮点墨迹；**乘回 255 再量化**成 uint8 —— 忘了乘就会把
    # 模板压成 0/1 两档（比对靠相关系数，还是能读对，但资产里存的已经不是那张字形图了，
    # 而它看起来仍然「有一张表」）。
    templates = {
        digit: np.clip(
            np.rint(np.stack(blocks).mean(axis=0) * 255.0), 0, 255
        ).astype(np.uint8)
        for digit, blocks in examples.items()
    }
    table = GlyphTable(digits=tuple(sorted(templates)), templates=templates)
    return table.save(Path(dest))


def main(argv: list[str] | None = None) -> int:
    r"""命令行入口：把**倒计时那套粗体字形表**重新生成一遍。

        .venv/Scripts/python.exe -m ark.offline.assets_gen.countdown

    默认取材那两批夹具、落到 `assets/features/glyphs_countdown/`。
    **这条路只写资产，不读验收套件** —— 取材帧与验收帧是分开的两批。
    """
    import argparse

    parser = argparse.ArgumentParser(description="从取材帧重建倒计时那套粗体字形表")
    parser.add_argument("--dest", type=Path, default=COUNTDOWN_DEST)
    args = parser.parse_args(argv)

    harvest = load_countdown_harvest()
    path = build_countdown_glyphs(harvest, args.dest)
    table = GlyphTable.load(path)
    coverage = harvest_coverage(harvest)
    print(
        f"取材 {len(harvest)} 张卡（{len({item.batch for item in harvest})} 批）"
        f" -> {path}（{path.stat().st_size} 字节，字形 {list(table.digits)}）"
    )
    print("每个数字几张样本：" + " ".join(f"{d}x{coverage[d]}" for d in sorted(coverage)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
