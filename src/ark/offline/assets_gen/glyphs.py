"""从取材帧造**出厂字形表**。

取材只许来自 `tests/帧样本/费用/取材.json` 里那 11 帧 —— 那是训练集。
剩下的 20 帧是验收用的测试集，**不许拿它们取材**（不然「自己写自己测」就没有
独立性了，见夹具 README）。

**为什么这一步在离线侧**：它读夹具、写资产，是**生成物**；读数链只读表
（`ark.engine.vision.glyphs.GlyphTable.load`），不重建。所以这里能 import 线上
那套切字 / 归一 / 比对的代码（`ark.engine.vision.glyphs`）—— 建表与读表**共用
同一个 INK_THRESHOLD、同一套归一**，两边不一致是这一层最隐蔽的坏法。

**产物必须可重建**：同一批取材帧，跑两次逐像素一致。所以这里没有随机、
没有时间戳、没有「取最像的那张」这种依赖输入顺序的选择。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image

from ark.contracts import Canvas, PixelRect
from ark.engine.vision.frame import Frame
from ark.engine.vision.glyphs import GlyphTable, glyph_boxes, ink_of, normalize
from ark.engine.vision.layout import UI

__all__ = ["build_digit_glyphs", "harvest_from_file", "HARVEST_FILE", "INDEX_FILE"]


HARVEST_FILE = "取材.json"
INDEX_FILE = "index.json"


def _canvas_of(
    shot_path: Path,
    origin: tuple[int, int],
    size: tuple[int, int],
    canvas_width: int,
    canvas_height: int,
) -> np.ndarray:
    """把一张裁剪**贴回**捕获分辨率那张画布上。

    夹具 README 说得明白：裁剪是原生像素，`index.json` 记了 origin，所以贴回去
    之后 **ROI 的取法仍然是真的在被测** —— 而不是把 ROI 预先替它做好。
    """
    patch = np.asarray(Image.open(shot_path).convert("RGB"))
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    left, top = origin
    height, width = patch.shape[:2]
    # index.json 的 size 是「含两端」的窗口宽高，而 PNG 里装的是**真实存在的
    # 那些像素**：窗口的右缘/下缘越出画布时，实际会少 1 列/行 —— 这份夹具正是
    # 如此（窗口写到 x=2800，而画布最后一列是 2799，所以文件是 220 列不是 221）。
    # 所以判据是「不大于窗口，且不比窗口小过那 1 列/行」，而不是严格相等。
    if width > size[0] or height > size[1]:
        raise ValueError(
            f"{shot_path.name} 的实际尺寸 {(width, height)} 比 index.json 里写的窗口 "
            f"{size} 还大"
        )
    if width < size[0] - 1 or height < size[1] - 1:
        raise ValueError(
            f"{shot_path.name} 的实际尺寸 {(width, height)} 比 index.json 里写的窗口 "
            f"{size} 小得太多（最多只该小 1 列/行）"
        )
    height = min(height, canvas_height - top)
    width = min(width, canvas_width - left)
    if height <= 0 or width <= 0:
        raise ValueError(f"{shot_path.name} 的原点 {origin} 落在画布之外")
    canvas[top:top + height, left:left + width] = patch[:height, :width]
    return canvas


def build_digit_glyphs(
    harvest: Mapping[int, int],
    dest: Path,
    *,
    crops_dir: Path,
    rect: PixelRect,
) -> Path:
    """从取材帧造字形表，写到 `dest`，返回落盘的文件路径。

    - `harvest`：`{源帧号: 该帧的费用整数}`（值由规划者目视确认）；
    - `crops_dir`：装着 `index.json`（每张裁剪的 origin）与 `shotNN.png`；
    - `rect`：费用数字那个矩形（`UI.rects["cost_digits"]`）。

    **切出来的块数必须等于位数**：不等就报错，不猜。切错了还硬造一张表，
    等于把一次错误固化进资产，而且从此看不出来。

    只有「一个数字都没覆盖到」是**致命**的：表里少一个键，那个数字就永远读不
    出来，而表看起来仍然是好的。单个取材帧切歪了，报错；缺整个数字，报错。
    """
    crops_dir = Path(crops_dir)
    index = json.loads((crops_dir / INDEX_FILE).read_text(encoding="utf-8"))
    canvas_width, canvas_height = (int(v) for v in index["canvas"])
    by_shot = {int(entry["shot"]): entry for entry in index["crops"]}

    examples: dict[int, list[np.ndarray]] = {}
    for shot in sorted(harvest):
        value = int(harvest[shot])
        text = str(value)
        entry = by_shot.get(shot)
        if entry is None:
            raise ValueError(f"取材帧 shot{shot:02d} 不在 {crops_dir / INDEX_FILE} 里")
        rgb = _canvas_of(
            crops_dir / entry["file"],
            tuple(int(v) for v in entry["origin"]),
            tuple(int(v) for v in entry["size"]),
            canvas_width,
            canvas_height,
        )
        frame = Frame(canvas=Canvas(width=canvas_width, height=canvas_height), pixels=rgb)
        roi = frame.crop(rect)
        ink = ink_of(roi)
        boxes = glyph_boxes(roi)
        if len(boxes) != len(text):
            raise ValueError(
                f"shot{shot:02d} 的费用是 {value}（{len(text)} 位），"
                f"在 {rect} 里却切出 {len(boxes)} 块笔画：切错了不许硬造表"
            )
        for character, (x0, y0, x1, y1) in zip(text, boxes):
            examples.setdefault(int(character), []).append(normalize(ink[y0:y1, x0:x1]))

    missing = sorted(set(range(10)) - set(examples))
    if missing:
        raise ValueError(
            f"取材帧覆盖不到这几个数字：{missing} —— 表里少一个键，"
            "那个数字就永远读不出来，而表看起来仍然是好的"
        )

    # normalize 给的是 0..1 的浮点墨迹；**乘回 255 再量化**成 uint8 —— 忘了乘
    # 就会把模板压成 0/1 两档（比对靠相关系数，还是能读对，但资产里存的已经不是
    # 那张字形图了，而它看起来仍然「有一张表」）。
    templates = {
        digit: np.clip(np.rint(np.stack(blocks).mean(axis=0) * 255.0), 0, 255).astype(np.uint8)
        for digit, blocks in examples.items()
    }
    table = GlyphTable(digits=tuple(sorted(templates)), templates=templates)
    return table.save(Path(dest))


def harvest_from_file(path: Path) -> dict[int, int]:
    """读 `取材.json`。放在这里是为了让「谁取材了」只有一个入口。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(item["shot"]): int(item["cost"]) for item in payload["harvest"]}


def main(argv: list[str] | None = None) -> int:
    r"""命令行入口：把**出厂字形表**重新生成一遍。

        .venv\Scripts\python.exe -m ark.offline.assets_gen.glyphs

    默认取材 `tests/帧样本/费用/取材.json`、落到 `assets/features/glyphs/`。
    **这条路只写资产，不读验收套件** —— 取材帧与验收帧是分开的两批。
    """
    import argparse

    repo = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="从取材帧重建出厂字形表")
    parser.add_argument("--crops", type=Path, default=repo / "tests" / "帧样本" / "费用")
    parser.add_argument("--dest", type=Path, default=repo / "assets" / "features" / "glyphs")
    args = parser.parse_args(argv)

    harvest = harvest_from_file(args.crops / HARVEST_FILE)
    path = build_digit_glyphs(
        harvest, args.dest, crops_dir=args.crops, rect=UI.rects["cost_digits"]
    )
    table = GlyphTable.load(path)
    print(
        f"取材 {len(harvest)} 帧 -> {path}"
        f"（{path.stat().st_size} 字节，字形 {list(table.digits)}）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
