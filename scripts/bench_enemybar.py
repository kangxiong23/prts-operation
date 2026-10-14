#!/usr/bin/env python
r"""敌人血条那一批基准集的**重跑**：裁剪 -> 读数。

    .venv\Scripts\python.exe scripts\bench_enemybar.py [批目录 …]

不给目录就跑默认那一批（`tests/帧样本/敌人血条`）；给了就逐批跑一遍，
每批各落一份自己的 `判据读数.json`。

它做三件事，一件都不做第四件：

1. **对出处**：`index.json` 里每条裁剪都记着源帧路径与**源帧 RAW sha256**。
   源帧还在本机时逐条比一遍（对不上就报，不静默改清单）；源帧早没了就跳过 ——
   出处是**只记不验**（基准集 §2）。
2. **折我方条的位置**：每一块里那些**我方蓝血条**（左缘 + 常量宽 146 + 8 行）就是
   调用方会交给链的那份记录。它从**裁剪自己的像素**上折，不碰源帧 ——
   所以源帧不在本机，这一份也照样重跑得出来。
3. **跑链、落 `判据读数.json`**：`read_enemy_bars` 逐块跑一遍，三个出口 +
   **条数（精确还是下界）** + 逐根填充比例与「分不分得开」+ 形状门与三条线的原样
   写下去。

**它不改判据、不改裁剪、不判「够不够」**：那份读数是给人核对的（哪几块判成在 /
不在 / 认不出、每一根量到多少、条数是精确还是下界），「够不够」是规划者的事。
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ark.contracts import (  # noqa: E402
    Exact,
    Interval,
    LowerBound,
    PixelRect,
    Unsampled,
)
from ark.engine.vision import enemybar  # noqa: E402
from ark.engine.vision.field import BAR_WIDTH, HP_BLUE, HP_COLOUR_TOL  # noqa: E402
from ark.engine.vision.frame import (  # noqa: E402
    CANVAS_2800x1260,
    frame_from_raw,
)
from ark.offline.bench.batch import load_batch  # noqa: E402
from ark.offline.bench.feed import replay_frames  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BATCH = REPO / "tests" / "帧样本" / "敌人血条"
MANIFEST_NAME = "index.json"
HARVEST_NAME = "取材.json"
REPORT_NAME = "判据读数.json"

OUR_BAR_ROWS = 8


def our_bars(frame) -> list[list[int]]:
    """这一帧里**我方蓝血条**的位置：左缘 + 常量宽 `BAR_WIDTH` + 8 行。"""
    pixels = frame.pixels.astype(np.int16)
    blue = np.abs(pixels - np.asarray(HP_BLUE, np.int16)).max(axis=2) <= HP_COLOUR_TOL
    runs = enemybar._long_runs(blue, OUR_BAR_ROWS).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
    found = []
    for index in range(1, count):
        x, y, width, height, _ = (int(value) for value in stats[index])
        if width < 20 or height > 12:
            continue
        found.append([x, y, BAR_WIDTH, OUR_BAR_ROWS])
    return sorted(found)


def slot_of(value) -> dict:
    """一个槽 -> 落盘用的那一份（形态就是载荷的类型）。"""
    if isinstance(value, Unsampled):
        return {"state": "unsampled"}
    payload = value.payload
    if isinstance(payload, Exact):
        form = {"form": "exact", "value": payload.value}
    elif isinstance(payload, Interval):
        form = {"form": "interval", "lo": payload.lo, "hi": payload.hi}
    elif isinstance(payload, LowerBound):
        form = {"form": "lower", "lo": payload.lo}
    else:
        form = {"form": type(payload).__name__}
    return {"state": "valued", "payload": form, "confidence": value.confidence}


def source_hash(path: pathlib.Path) -> str:
    """源帧的 **RAW sha256**（二进制按文件字节，不折算换行）。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources(manifest: dict) -> list[str]:
    """对一遍出处，给回「对不上 / 找不到」的那几条（对得上的一条都不提）。"""
    notes = []
    for crop in manifest["crops"]:
        source = REPO / crop["source"]
        if not source.is_file():
            notes.append("%s: 源帧不在本机，跳过（出处只记不验）" % crop["file"])
            continue
        got = source_hash(source)
        if got != crop["source_sha256"]:
            notes.append(
                "%s: 源帧指纹对不上 —— 清单 %s / 实际 %s"
                % (crop["file"], crop["source_sha256"][:12], got[:12])
            )
    return notes


def run_batch(root: pathlib.Path) -> None:
    """跑一批：对出处 -> 折我方条 -> 跑链 -> 落 `判据读数.json`。"""
    root = root.resolve()
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    harvest = json.loads((root / HARVEST_NAME).read_text(encoding="utf-8"))
    batch = load_batch(root)
    by_file = {item["file"]: item for item in harvest["crops"]}
    reads = []
    for case, raw in replay_frames(batch):
        frame = frame_from_raw(raw)
        ours = our_bars(frame)
        by_file[case.name + ".png"]["our_bars"] = ours
        result = enemybar.read_enemy_bars(
            frame, our_bars=[PixelRect(*rect) for rect in ours]
        )
        reads.append(
            {
                "name": case.name,
                "truth": by_file[case.name + ".png"]["truth"],
                "covers": by_file[case.name + ".png"]["covers"],
                "presence": slot_of(result.presence),
                "count": slot_of(result.count),
                "bars": [
                    {
                        "origin": [int(bar.rect.x), int(bar.rect.y)],
                        "total_columns": int(bar.rect.width),
                        "fill": slot_of(bar.fill),
                        "colour_distance": bar.distance,
                        "separated": bar.separated,
                    }
                    for bar in result.bars
                ],
                "cores": [
                    {
                        "origin": [int(core.rect.x), int(core.rect.y)],
                        "columns": int(core.rect.width),
                        "colour_distance": core.distance,
                        "masked": core.masked,
                    }
                    for core in result.cores
                ],
            }
        )

    report = {
        "note": "敌人血条那条链的判据读数。它不判断够不够 —— 那是规划者的事；这里只放逐帧读数、两簇的分布与空档、以及覆盖说明。",
        "batch": str(root.relative_to(REPO)).replace("\\", "/"),
        "version": 1,
        "canvas": [CANVAS_2800x1260.width, CANVAS_2800x1260.height],
        "device": batch.device,
        "ruler": {
            "colour": "逐像素离 ENEMY_FILL (218,106,53) 或 ENEMY_FILL_RED (197,41,40) 的逐通道最大差 "
            "<= SCREEN_TOL=%d —— **两族各筛一遍、不并掩膜**" % enemybar.SCREEN_TOL,
            "shape": "只留竖着连成 8 行的像素（前景与背景各扫一遍的连续长度），再按连通块取 "
            "高 <= %d；宽的门按族分：橙族 >= %d、红族 >= %d"
            % (
                enemybar.MAX_BAR_ROWS,
                enemybar.MIN_CANDIDATE_COLUMNS,
                enemybar.RED_MIN_COLUMNS,
            ),
            "columns": "块里那 8 行逐列取中位色，离 ENEMY_FILL 的逐通道最大差 <= %d 的列算作填充段"
            % enemybar.SCREEN_TOL,
            "exits": "分数 = 1 - 颜色距 / 255；>= 1-%d/255 为在，<= 1-%d/255 为不在，中间为未采样"
            % (enemybar.SOLID_TOL, enemybar.ABSENT_TOL),
            "rectangle": "矩形：每一列都亮着的那几行（条身）要装下整块的 >= %s 像素；"
            "内部均匀：那 8 行逐行取中位色，通道极差 <= %d"
            % (enemybar.RECT_FILL_MIN, enemybar.RECT_ROW_SPREAD_MAX),
            "short": "填充段短于 MIN_FILL_COLUMNS=%d 的候选**照样报在**，比例落未采样"
            % enemybar.MIN_FILL_COLUMNS,
            "track": "整长 = 填充段 + 空轨；空轨 = 与上下两块地形两侧都差 > TRACK_CONTRAST=%d 的那一段"
            % enemybar.TRACK_CONTRAST,
            "exclusion": "一块踩在我方血条那 8 行、或者紧贴它下面第 %d 行的技力条那 8 行上（行差 <= %d）且列范围夹在我方那一格里（列差 <= %d） -> 剔掉"
            % (enemybar.SKILL_ROW_OFFSET, enemybar.ROW_SLACK, enemybar.COL_SLACK),
            "merge": "同一行带上、中间那几列更靠近填充色（半透明遮挡压出来的混色）的两块接回一根条",
            "count": "条数：都分得开给 Exact；分不开的给 LowerBound（同一条带里相邻两根的整长交叠或相接、"
            "或中间至少有一列是混色 -> 那两块只算一根）。**分不开的不报精确**，也不高报",
        },
        "field": {"rows": [enemybar.FIELD_TOP, enemybar.FIELD_BOTTOM]},
        "reads": reads,
    }
    # **钉死 LF**：`write_text` 在 Windows 上默认把 `\n` 翻成 `\r\n`，而这个仓库
    # 里每一份文本都是 LF —— 不钉的话同一份落盘文件在两次重跑之间 RAW 字节不同。
    (root / HARVEST_NAME).write_text(
        json.dumps(harvest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / REPORT_NAME).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    for note in check_sources(manifest):
        print("[%s] 出处：%s" % (root.name, note))
    present = sum(
        1 for item in reads if item["presence"]["payload"]["value"] is True
    )
    absent = sum(
        1 for item in reads if item["presence"]["payload"]["value"] is False
    )
    unknown = len(reads) - present - absent
    print(
        "[%s] 逐帧判决：在 %d · 不在 %d · 认不出 %d（共 %d 块）"
        % (root.name, present, absent, unknown, len(reads))
    )
    print(
        "[%s] 条数：%s"
        % (
            root.name,
            ", ".join(
                "%s=%s:%s"
                % (
                    item["name"],
                    item["count"]["payload"]["form"],
                    item["count"]["payload"].get(
                        "value", item["count"]["payload"].get("lo")
                    ),
                )
                for item in reads
            ),
        )
    )
    print("[%s] 落盘：%s" % (root.name, (root / REPORT_NAME).relative_to(REPO)))


def main(dirs: list[pathlib.Path]) -> int:
    for root in dirs:
        run_batch(root)
    return 0


if __name__ == "__main__":
    given = [pathlib.Path(item) for item in sys.argv[1:]]
    raise SystemExit(main(given or [DEFAULT_BATCH]))
