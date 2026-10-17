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

**每一块都出图，一个都不跳**（用户 10-15 的裁定）：负例（「不在」）那一张正是
「我看过、这里没有」的证据 —— 判决表上那几行没有它就没骨头；而且同一类东西两批
两种策略，将来没人说得清哪些块「该有图」。

**表头三行**（单 11 起），每一行都留得下看得见的余量：

1. **真值并排链的读数**（**青** + **黄**）：`truth=in n=1` / `truth=in n=2` /
   `truth=in n=?` / `truth=out n=0` 与 `read in=True count=…`。真值来自夹具
   `取材.json` 的 `truth` + `truth_count`（`bench` 把它们抄进了读数的 `reads`），
   是**画面事实**、不是链读出来的 —— **只写链的读数就是让图自己证明自己**，负例尤其
   如此：它得证明「这里确实没有」，而不是「链没读出来」。`n=` 那半个就是拿**真值几根**
   对**链读出几根**。`cv2.putText` 的 Hershey 字体画不出汉字，所以 `在 -> in`、
   `不在 -> out`、`null -> ?`（**只用 ASCII**）；碰上没见过的真值**报错**，不静默画
   占位符 —— 而且**在写任何文件之前**就报（见 `check_reads`）；
2. **颜色表** —— 这张图上的框各是什么意思，外加 `cyan=truth`；
3. **`(>= is a lower bound)`** —— 下界那个记号是 ASCII `>=`（画不出 `≥`），
   不解释会被读成「约等于」。

**表头不许盖住任何一块东西**：出图时先算一遍「每一件东西最高的那一行」有没有伸进表头
那一整条黑底（它是最后画的、盖满全宽），伸进去就**当场响**（`_guard_header`）、不静默盖住。

标注里带三样（都用 ASCII，`cv2.putText` 的 Hershey 字体画不出别的字）：

- **读数**：**精确只写数**（`bar 84`）· **下界带 `>=`**（`bar >=120`；框比读数宽时
  两个都写，`bar 60 >=50` —— 前面那个是**框**的宽、后面那个是**这次的主张**）。
  两者本来就**可以不一样**：框画的是**量到的那一段**、读数给的是**这次的主张**
  （有缝那一根框 60 列、读数只给 >=50）；
- `h<行数>` 是**量到的条身**（不写死 8）· `small` 是那个**小尺寸标记**；
- clump 那一栏：`n>=<至少几根>`、`cols=<总列数>`（精确）或 `cols>=<总列数>`（下界）、
  以及**来路**。
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

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
QUALITY = 80
"""落盘用的那几个常量：颜色就是「一个颜色一个意思」那张表。"""

CYAN = (255, 255, 0)
"""表头第一行那个**真值**的颜色 —— 它不画框，所以与上面那张颜色表不冲突。"""

FONT = cv2.FONT_HERSHEY_SIMPLEX
HEADER_SCALE = 0.45
HEADER_ROWS = 70
TEXT_X = 6
FIRST_Y = 17
SECOND_Y = 38
THIRD_Y = 59
TRUTH_GAP = 12
"""表头的几何（单 11 起三行）：黑底 `HEADER_ROWS` 行，三行字的基线在 17 / 38 / 59，
`truth=` 与链的读数**左右并排**（中间隔 `TRUTH_GAP`）。三行各自右边缘在真图上的实测
位置写在两份 README 的缺口里 —— 每一行都要留得下看得见的余量。"""

COLOUR_TABLE = (
    "green=bar  yellow=unsampled  magenta=clump  white=screened-only  cyan=truth"
)
BOUND_NOTE = "(>= is a lower bound)"
TRUTH_TEXT = {"在": "in", "不在": "out"}
"""表头那三行字与真值的写法。**只用 ASCII**：`cv2.putText` 的 Hershey 字体画不出别的字。"""


def _line(pic, text: str, y: int, colour, x: int = TEXT_X) -> None:
    """表头的一行字。**字号与左边距只有这一处** —— 三行各自的余量都由它兜着。"""
    cv2.putText(pic, text, (x, y), FONT, HEADER_SCALE, colour, 1, cv2.LINE_AA)


def _truth_text(value: str) -> str:
    """真值那一栏在图上写成什么样：`在 -> in`、`不在 -> out`。

    真值是**画面事实**（夹具 `取材.json` 的 `truth`），与链读出来的那几栏是两回事 ——
    图上把它**并排**写出来，图就不再**自己证明自己**。`cv2.putText` 的 Hershey 字体
    画不出汉字（**只用 ASCII**）。碰上没见过的真值**报错**：真值多了、或者改了口径，
    要有人看过，不许静默画一个占位符糊过去。
    """
    if value not in TRUTH_TEXT:
        raise ValueError(
            "真值 %r 没有对应的 ASCII 写法（在 -> in · 不在 -> out）" % value
        )
    return TRUTH_TEXT[value]


def _truth_count_text(value) -> str:
    """真值根数在图上写成什么样：整数照写，`None`（没数过 / 不数）写 `?`。

    **只用 ASCII**。整数以外的（字符串、布尔、负数）当场报错 —— 它不是一个估计，
    是**真值主本给的一个数**。
    """
    if value is None:
        return "?"
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("真值根数 %r 既不是整数也不是 null" % (value,))
    if value < 0:
        raise ValueError("真值根数 %r 是负数" % (value,))
    return str(value)

def _tops(read: dict, origin) -> list[tuple[str, int]]:
    """这一块上**画上去的每一件东西**最高的那一行（图像坐标）：框、白细框、还有标注。

    标注画在框的上方（基线在框顶再往上 10 行、字高 12），所以一件东西的「脚」要算到
    **框顶 - 22 行**。**表头那一整条黑底是最后画上去的、盖满全宽** —— 盖没盖住只用比**行**。
    """
    tops: list[tuple[str, int]] = []
    for core in read["cores"]:
        if core["masked"] or core["colour_distance"] > SOLID_TOL:
            tops.append(("core", core["origin"][1] - origin[1] - 4))
    for bar in read["bars"]:
        tops.append(("bar", bar["origin"][1] - origin[1] - 22))
    for bar in read["unsampled"]:
        tops.append(("uns", bar["origin"][1] - origin[1] - 22))
    for clump in read["clumps"]:
        tops.append(("clump", clump["region"][1] - origin[1] - 22))
    return tops


def _guard_header(tops: list[tuple[str, int]]) -> None:
    """表头盖住任何一件东西就**当场响** —— 缩字号、挪表头，或者换一块样本。

    表头是最后画的、盖满全宽，所以被盖住的那一件在图上是**没有痕迹**的：它不影响读数
    （读数在落盘的 JSON 里），却让这张「证据照」少了一件证据。
    """
    covered = [(what, top) for what, top in tops if top < HEADER_ROWS]
    if covered:
        raise ValueError(
            "表头（0..%d 行）盖住了 %s —— 缩字号、挪表头，或者换一块样本"
            % (HEADER_ROWS - 1, "、".join("%s@y=%d" % item for item in covered))
        )


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


def _shown(payload: dict) -> str:
    """一条读数在图上写成什么样：**精确只写数，下界 / 区间带前缀**。

    形态要写在图上：框画的是**量到的那一段**、读数给的是**这次的主张** —— 有缝的那
    一根框 60 列、读数给 `>=50`，图上不写形态，60 配 >=50 就成了打架。
    **只用 ASCII**：`cv2.putText` 的 Hershey 字体画不出别的字。
    """
    form = payload["form"]
    if form == "exact":
        return "%d" % payload["value"]
    if form == "lower":
        return ">=%d" % payload["lo"]
    return "%d..%d" % (payload["lo"], payload["hi"])


def _bar_label(kind: str, bar: dict) -> str:
    """一根的标注：`<kind> <框宽> <读数> h<行数>[ small]`。

    **只在两者字面不同的时候把框宽也摆出来**：精确的那一根框宽就是读数本身
    （`bar 84 h8`）；有缝的那一根框 60 列、读数只有 `>=50`，两个都得写
    （`bar 60 >=50 h8`）—— 框回答「在哪儿」、读数回答「多少」。
    """
    payload = bar["columns"]["value"]["payload"]
    columns = int(bar["fill_columns"])
    shown = _shown(payload)
    head = kind + " " + shown
    if int(_slot(payload)) != columns:
        # 框比读数宽（量到的那一段跨过了缝）：两个数都得摆出来。
        head = "%s %d %s" % (kind, columns, shown)
    label = head + " h%d" % int(bar["rows"])
    if bar.get("short_fill"):
        label += " small"
    return label


def _clump_label(clump: dict) -> str:
    """一堆的标注：`clump <哪一档> n>=<至少几根> cols=<总列数> <来路>`。

    `cols` 那一栏跟着**总列数**的形态走：精确 -> `cols=130` · 下界 -> `cols>=186`。
    """
    payload = clump["columns"]["value"]["payload"]
    shown = _shown(payload)
    total = ("cols=" + shown) if payload["form"] == "exact" else ("cols" + shown)
    least = clump["least"]["value"]["payload"]["lo"]
    return "clump %s n>=%s %s %s" % (
        clump["tier"]["name"],
        least,
        total,
        clump["route"],
    )


def _truth_label(read: dict) -> str:
    """图上真值那半句：`truth=in n=1` · `truth=in n=2` · `truth=in n=?` · `truth=out n=0`。"""
    return "truth=%s n=%s" % (
        _truth_text(read["truth"]),
        _truth_count_text(read["truth_count"]),
    )


def _readout_label(read: dict) -> str:
    """链的读数那半句：`read in=True count=exact:1 bars=1 small=0 clumps=0 uns=0`。"""
    payload = read["count"]["payload"]
    return "read in=%s count=%s:%s bars=%d small=%d clumps=%d uns=%d" % (
        read["presence"]["payload"].get("value", "unsampled"),
        payload["form"],
        _slot(payload),
        len(read["bars"]),
        sum(1 for bar in read["bars"] if bar.get("short_fill")),
        len(read["clumps"]),
        len(read["unsampled"]),
    )


def check_reads(reads: list[dict]) -> None:
    """把图上要写的那两半文字先拼一遍 ——**在写任何文件之前**把拼不出来的当场炸掉。

    `bench_enemybar` 落盘之前先调它：不然就是**半落盘**（读数更新了、图没更新），
    而且那种状态没人会发现。
    """
    for read in reads:
        _truth_label(read)
        _readout_label(read)


def draw(root: pathlib.Path | str) -> list[tuple[str, int]]:
    """读这一批的 `判据读数.json`，重出它的取证图 -> [(文件名, 字节数)]。

    **它只读落盘的那份读数**（不重跑链）：所以「图与读数同步」这件事是**结构性**的。
    """
    root = pathlib.Path(root).resolve()
    report = json.loads((root / "判据读数.json").read_text(encoding="utf-8"))
    harvest = json.loads((root / "取材.json").read_text(encoding="utf-8"))
    by_file = {item["file"]: item for item in harvest["crops"]}
    # **先校验、再动图**：真值画不出来就在这里炸，别落到「画了一半」。
    check_reads(report["reads"])
    out: list[tuple[str, int]] = []
    for read in report["reads"]:
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
            _box(
                pic,
                x,
                y,
                int(bar["fill_columns"]),
                int(bar["rows"]),
                GREEN,
                _bar_label("bar", bar),
            )
        for bar in read["unsampled"]:
            x = bar["origin"][0] - origin[0]
            y = bar["origin"][1] - origin[1]
            _box(
                pic,
                x,
                y,
                int(bar["fill_columns"]),
                int(bar["rows"]),
                YELLOW,
                _bar_label("uns", bar),
            )
        for clump in read["clumps"]:
            rx, ry, rw, rh = clump["region"]
            x, y = rx - origin[0], ry - origin[1]
            _box(
                pic, x, y, int(rw), int(rh), MAGENTA, _clump_label(clump)
            )
        # 表头三行（单 11）：真值并排链的读数 + 颜色表 + `>=` 的说明。
        # **每一行都要留得下看得见的余量** —— 颜色表那一行是「一个颜色一个意思」那张表，
        # 尾巴被裁掉读者就只能靠 README。
        _guard_header(_tops(read, origin))
        cv2.rectangle(pic, (0, 0), (pic.shape[1] - 1, HEADER_ROWS - 1), BLACK, -1)
        truth = _truth_label(read)
        readout = _readout_label(read)
        _line(pic, truth, FIRST_Y, CYAN)
        truth_width = cv2.getTextSize(truth, FONT, HEADER_SCALE, 1)[0][0]
        _line(pic, readout, FIRST_Y, YELLOW, TEXT_X + truth_width + TRUTH_GAP)
        _line(pic, COLOUR_TABLE, SECOND_Y, YELLOW)
        _line(pic, BOUND_NOTE, THIRD_Y, YELLOW)
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

