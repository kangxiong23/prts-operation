#!/usr/bin/env python
r"""敌人血条两批夹具的**取证图**：一个颜色一个意思，随重跑自动重出。

    .venv\Scripts\python.exe scripts\evidence_enemybar.py [批目录 …]

不给目录就跑默认那两批。它**读的是刚落盘的 `判据读数.json`** ——
`scripts\bench_enemybar.py` 每次重跑都会调它，所以图与读数**永远同步**
（从前它住在 gitignored 的 `data/probe/` 里手跑，那是隐患：图迟早与读数不同步）。

**幂等**：同一个 `判据读数.json` 跑两次，落盘的 JPG **逐字节相同** ——
`cv2.imencode` 是确定的，画法里没有随机数、没有时间戳。

**颜色**（一个颜色一个意思）：

- **绿** = `bars`（逐根读出来的；**短填充**那一根在标注里带 `small`）；
- **黄** = `unsampled`（有东西、但填了多少读不出来）；
- **洋红** = `clumps`（一堆：区域 + 至少几根 + 填充总列数 + **来路**）；
- **白细框** = 筛出来但**没进三栏**的块（被我方条挡掉的，或者颜色只落在中间带
  不当条的）。

标注里的 `h<行数>` 是**量到的条身**（不写死 8）；clump 那一栏带 `n>=<至少几根>`、
`cols=<总列数>` 与来路。**负例那一批里三栏都空的那几块不出图**（图上没有框、
只剩一个表头）；10-14 那一批逐块都出（它的负例也是点名的）。
"""

from __future__ import annotations

import json
import pathlib
import sys

import cv2
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]

# 白色细框画的是「筛出来、但颜色只落在中间带」的那些块 —— 它用的那道门与判据同源，
# 不在这里抄一个数。
sys.path.insert(0, str(REPO / "src"))

from ark.engine.vision.enemybar import SOLID_TOL  # noqa: E402

DEFAULT_BATCHES = (
    REPO / "tests" / "帧样本" / "敌人血条",
    REPO / "tests" / "帧样本" / "敌人血条-10-14",
)
"""不给目录时跑的那两批（与 `bench_enemybar` 的默认批一致）。"""

DRAW_EMPTY_BATCHES = ("敌人血条-10-14",)
"""三栏都空的那几块也出图的批次（负例也是点名的那一批）。"""

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
QUALITY = 80
"""落盘用的那几个常量：颜色就是「一个颜色一个意思」那张表。"""


def _box(pic, x, y, columns, rows, colour, label) -> None:
    """在 `(x, y)` 起、宽 `columns` 高 `rows` 的那一块外面画一个框 + 一行标注。"""
    cv2.rectangle(pic, (x - 6, y - 6), (x + columns + 6, y + rows + 5), colour, 3)
    cv2.putText(
        pic,
        label,
        (x - 6, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        colour,
        1,
        cv2.LINE_AA,
    )


def _slot(payload: dict):
    """一条读数落盘的那个槽 -> 它的数（精确 -> value，下界 / 区间 -> lo）。"""
    return payload.get("value", payload.get("lo"))


def draw(root: pathlib.Path | str) -> list[tuple[str, int]]:
    """读这一批的 `判据读数.json`，重出它的取证图 -> [(文件名, 字节数)]。

    **它只读落盘的那份读数**（不重跑链）：所以「图与读数同步」这件事是**结构性**的。
    """
    root = pathlib.Path(root).resolve()
    report = json.loads((root / "判据读数.json").read_text(encoding="utf-8"))
    harvest = json.loads((root / "取材.json").read_text(encoding="utf-8"))
    by_file = {item["file"]: item for item in harvest["crops"]}
    out: list[tuple[str, int]] = []
    for read in report["reads"]:
        if not any(read[key] for key in ("bars", "clumps", "unsampled")) and (
            root.name not in DRAW_EMPTY_BATCHES
        ):
            continue
        name = read["name"] + ".png"
        origin = by_file[name]["origin"]
        # `imdecode` + `np.fromfile` 而不是 `cv2.imread`：裁剪目录名里有非 ASCII 字。
        pic = cv2.imdecode(
            np.fromfile(str(root / name), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        for core in read["cores"]:
            x = core["origin"][0] - origin[0]
            y = core["origin"][1] - origin[1]
            if core["masked"] or core["colour_distance"] > SOLID_TOL:
                cv2.rectangle(
                    pic, (x - 4, y - 4), (x + core["columns"] + 4, y + 20), WHITE, 1
                )
        for bar in read["bars"]:
            x = bar["origin"][0] - origin[0]
            y = bar["origin"][1] - origin[1]
            label = "bar h%d" % int(bar["rows"])
            if bar.get("short_fill"):
                label += " small"
            _box(
                pic, x, y, int(bar["fill_columns"]), int(bar["rows"]), GREEN, label
            )
        for bar in read["unsampled"]:
            x = bar["origin"][0] - origin[0]
            y = bar["origin"][1] - origin[1]
            _box(
                pic, x, y, int(bar["fill_columns"]), int(bar["rows"]), YELLOW,
                "uns h%d" % int(bar["rows"]),
            )
        for clump in read["clumps"]:
            rx, ry, rw, rh = clump["region"]
            x, y = rx - origin[0], ry - origin[1]
            total = _slot(clump["columns"]["value"]["payload"])
            least = clump["least"]["value"]["payload"]["lo"]
            _box(
                pic,
                x,
                y,
                int(rw),
                int(rh),
                MAGENTA,
                "clump %s n>=%s cols=%s %s"
                % (clump["tier"]["name"], least, total, clump["route"]),
            )
        # 表头分两行：一条统计行 + 一条图例行。**图例要能整行看全** —— 它写着
        # 「一个颜色一个意思」那张表，一行放不下就会被裁掉尾巴（读者就只能靠 README）。
        cv2.rectangle(pic, (0, 0), (pic.shape[1] - 1, 48), BLACK, -1)
        payload = read["count"]["payload"]
        count = _slot(payload)
        presence = read["presence"]["payload"].get("value", "unsampled")
        cv2.putText(
            pic,
            "in=%s count=%s:%s bars=%d small=%d clumps=%d uns=%d"
            % (
                presence,
                payload["form"],
                count,
                len(read["bars"]),
                sum(1 for bar in read["bars"] if bar.get("short_fill")),
                len(read["clumps"]),
                len(read["unsampled"]),
            ),
            (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            YELLOW,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            pic,
            "green=bar  yellow=unsampled  magenta=clump  white=screened-only",
            (6, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            YELLOW,
            1,
            cv2.LINE_AA,
        )
        target = root / ("取证_" + read["name"] + ".jpg")
        target.write_bytes(
            cv2.imencode('.jpg', pic, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])[1].tobytes()
        )
        out.append((target.name, target.stat().st_size))
    return out


def main(dirs: list[pathlib.Path]) -> int:
    """逐批重出取证图，把每一张贴出来的那张图的名字与字节数报出来。"""
    for root in dirs:
        written = draw(root)
        print("[%s] 取证图 %d 张" % (root.name, len(written)))
        for name, size in written:
            print("  %-38s %8d B" % (name, size))
    return 0


if __name__ == "__main__":
    given = [pathlib.Path(item) for item in sys.argv[1:]]
    raise SystemExit(main(given or list(DEFAULT_BATCHES)))

