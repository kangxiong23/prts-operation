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
   **条数（精确还是下界）** + **三栏各自的原样**：`bars`（逐根的填充段列数与比例，
   各自带形态与余量）· `clumps`（一堆的区域 / 哪一档 / 填充总列数 / 至少几根）·
   `unsampled`（有东西、但填了多少读不出来）+ 逐块筛出来的那些块 + 形状门与
   几条线的原样写下去。

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


def reading_of(reading) -> dict:
    """一条读数 -> 落盘用的那一份：**值（形态）** + **余量**。

    `margin` 是「**定下这次读数的两侧离判据的门有多远**」，不是概率；`None` = 这一条
    不是读出来的（比如「至少几根」是除法算的）。**这一层没有 0–1 的总分。**
    """
    return {"value": slot_of(reading.value), "margin": reading.margin}


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
                # **三栏互斥**（见判据的模块开头「三样读法」）：三样各自一眼分得开。
                "bars": [
                    {
                        "origin": [int(bar.fill_rect.x), int(bar.fill_rect.y)],
                        # **填充段列数**（不是整长 —— 这一版没有「整长」这个东西：
                        # 分母是按族查表的两档常量，见 `ruler["dial"]`）。
                        "fill_columns": int(bar.fill_rect.width),
                        "rows": int(bar.fill_rect.height),
                        "columns": reading_of(bar.columns),
                        "fill": reading_of(bar.fill),
                        "colour_distance": bar.distance,
                    }
                    for bar in result.bars
                ],
                "clumps": [
                    {
                        "region": [
                            int(clump.region.x),
                            int(clump.region.y),
                            int(clump.region.width),
                            int(clump.region.height),
                        ],
                        "tier": {
                            "name": clump.tier.name,
                            "full_columns": clump.tier.full_columns,
                            "body_rows": clump.tier.body_rows,
                        },
                        "columns": reading_of(clump.columns),
                        "least": reading_of(clump.least),
                    }
                    for clump in result.clumps
                ],
                "unsampled": [
                    {
                        "origin": [int(bar.fill_rect.x), int(bar.fill_rect.y)],
                        "fill_columns": int(bar.fill_rect.width),
                        "rows": int(bar.fill_rect.height),
                        "columns": reading_of(bar.columns),
                        "fill": reading_of(bar.fill),
                        "colour_distance": bar.distance,
                    }
                    for bar in result.unsampled
                ],
                "cores": [
                    {
                        "origin": [int(core.fill_rect.x), int(core.fill_rect.y)],
                        "columns": int(core.fill_rect.width),
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
        "version": 2,
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
            "columns": "块里**量到的条身那几行**逐列取中位色，离这一族参考色的逐通道最大差 <= SCREEN_TOL=%d 的列算作填充段，"
            "中间容 FILL_GAP=%d 列不亮（抗锯齿、一根窄遮挡）。**填充段列数**就是容得下那个缝的数："
            "干净 -> Exact · 右边还接着亮（没读到右端）-> LowerBound；**列数这一支不看缝**"
            "（缝底下是半透明遮挡压出来的，那几列算进这一段）—— 缝只压低它的**余量**"
            % (enemybar.SCREEN_TOL, enemybar.FILL_GAP),
            "body": "条身**按量到的行数**（不写死 8）：自那一块的行起，上下扩到「填充段那几列的中位色不再像这一族参考色」为止，封顶 MAX_BAR_ROWS=%d。"
            "每一根报出来的高（`EnemyBar.fill_rect.height`）就是这个**量到的行数**；它**比这一档更高**时"
            "（小怪 MOB_BODY_ROWS=%d / boss BOSS_BODY_ROWS=%d）这一根落 `unsampled` —— 跨档校验**只拦更高的那一头**"
            % (enemybar.MAX_BAR_ROWS, enemybar.MOB_BODY_ROWS, enemybar.BOSS_BODY_ROWS),
            "exits": "分数 = 1 - 颜色距 / 255；>= 1-%d/255 为在，<= 1-%d/255 为不在，中间为未采样"
            % (enemybar.SOLID_TOL, enemybar.ABSENT_TOL),
            "rectangle": "矩形：每一列都亮着的那几行（条身）要装下整块的 >= %s 像素；"
            "内部均匀：那 8 行逐行取中位色，通道极差 <= %d"
            % (enemybar.RECT_FILL_MIN, enemybar.RECT_ROW_SPREAD_MAX),
            "unsampled": "`unsampled` 那一栏落的是**未采样**（`Unsampled()`）：那一根确实"
            "有东西、但**填了多少读不出来** —— 填充段短于 MIN_FILL_COLUMNS=%d（条照报在、"
            "列数照量），或者条身行数比这一档更高。它与 `clumps` 是两件事"
            "（一堆 = 知道那一坨在哪、多大、至少几只）"
            % enemybar.MIN_FILL_COLUMNS,
            "dial": "分母是**按族查表的两档常量**：橙族 -> 小怪档 MOB_FILL_COLUMNS=%d、"
            "红族 -> boss 档 BOSS_FILL_COLUMNS=%d（都是**画布宽 2800 下的绝对像素**；别的分辨率未验，"
            "见模块开头「分辨率」那一节）。运行时**不量轨道**：比例 = 填充段列数 / 那一档的满条长"
            % (enemybar.MOB_FILL_COLUMNS, enemybar.BOSS_FILL_COLUMNS),
            "fill": "填充比例 = 填充段列数 / 那一档的满条长，按四支落形态：填充段短于 "
            "MIN_FILL_COLUMNS=%d -> 未采样；条身行数**比这一档更高** -> 未采样（跨档校验）；"
            "填充段中间有缝、或者右边没到头 -> LowerBound(缝前那一截 / 满条长)；其余 -> Exact"
            % enemybar.MIN_FILL_COLUMNS,
            "exclusion": "一块踩在我方血条那 8 行、或者紧贴它下面第 %d 行的技力条那 8 行上（行差 <= %d）且列范围夹在我方那一格里（列差 <= %d） -> 剔掉"
            % (enemybar.SKILL_ROW_OFFSET, enemybar.ROW_SLACK, enemybar.COL_SLACK),
            "merge": "同一行带上、中间那几列更靠近填充色（半透明遮挡压出来的混色）的两块接回一根条",
            "clump": "「一堆」（`clumps`）：**分不开**（同一条带里相邻两根的填充交叠 / 相接 / 中间至少有一列是混色）"
            "或者**填充总列数 > 那一档的满条长** -> 整坨报成**区域 + 至少几根 + 填充总列数**，"
            "**不给单根比例**。**按族归组、不许跨族**（所以每一堆都带 tier）。"
            "填充总列数 = 组里各根的填充段列数**相加**：组里有一条没读到右端 -> LowerBound、否则 Exact；"
            "至少几根 = ceil(填充总列数 / 这一档的满条长)，落 LowerBound —— **只是下界**，"
            "**上界由状态层拿账配、不从像素来**",
            "count": "条数：**一堆都没有** -> Exact(逐根 + 未采样的那些)；有一堆 -> LowerBound(逐根 + 未采样 + "
            "各堆的「至少几根」之和)，**两边不打架**。**这一支永远不高报**"
            "（分不开的整坨只按 ceil(填充总列数 / 满条长) 贡献）",
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
    print(
        "[%s] 三栏：%s"
        % (
            root.name,
            " · ".join(
                "%s bars=%d clumps=%d unsampled=%d"
                % (item["name"], len(item["bars"]), len(item["clumps"]), len(item["unsampled"]))
                for item in reads
                if item["bars"] or item["clumps"] or item["unsampled"]
            )
            or "（这一批三栏都空）",
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
