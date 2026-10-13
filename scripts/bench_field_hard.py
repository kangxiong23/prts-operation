"""`screenshots/run-01` 那一批**难例** → 与 `tests/帧样本/场上干员` 同形的批。

    .venv\\Scripts\\python.exe scripts\\bench_field_hard.py

## 这一批是什么

用户 2026-10-07 那一段连拍（`screenshots/run-01/manifest.csv`：58 帧、10 秒一张）里
挑出来的 12 张。**它是「补测」**：上一轮的问题有一部分是样例太少（老的两批里，
一格里只有一根条的情形占绝大多数），所以这一批专挑**一格里有好几根条、挨得近、
还被黄菱形压住**的帧。

文件名上有用户贴的标签，标签原文进 `取材.json` 的 `label`：

| 帧 | 用户贴的标签 |
| --- | --- |
| hard_多个技能条紧挨着 | 多个技能条紧挨着 |
| hard_多个技能条错位挨着 | 多个技能条错位挨着 |
| special_带弹药机制的有竖线间隔的橙色技能条 | 带弹药机制的有竖线间隔的橙色技能条 |
| special_黄色菱形挡住了技能条，很难判准，应报未采样 | 黄色菱形挡住了技能条，很难判准，**应报未采样** |

最后那一条是**期望输出**（用户写的），不是描述。

## 一条被测对象怎么挑：用**候选查找器**的全部输出，一个不筛

挑的方法**不带我的判断**：每一帧上跑 `bench_field._bar_candidates()`（执行者写的
建批侧候选查找器），它给出几根就做几块裁剪 —— 12 帧一共 52 块。

为什么这么挑：这一批要压的就是「**窗里不止一根条**」。查找器是**按「亮着的连段」
找候选**的，所以一格里有几根条它就给出几个候选，邻条会同时进窗。**我不筛**，
因为一筛就把要压的那一档筛掉了。

**它带两个已知的粗糙处**（都在读数里引出来，不在建批时悄悄抹掉）：

- **挨着的两条会连成一段**：条宽 146，而这一批里相邻两条的左缘只差 **138 px** ——
  也就是邻条会**盖掉本条的轨道尾巴 8 列**。查找器按「连段起点」记候选，于是
  一条 146 宽的条可能被记成 `lit=276`（两条连起来）。**裁剪只用 (x, y)**，
  所以这不影响取区；但 `lit` 那一列**不许当填充读数**。
- **同一根条可能被记成两个候选**（被背景切成两段）。两条裁剪仍然各是一块合法的窗，
  只是它们量的是同一根条 —— 逐块报，不合并。

所以：**这一批的 `bar` 是「候选查找器的输出」，不是「真值坐标」**。配对真值那一批
（`场上干员真值`）才有「离画布中心最近 = 被点中的那个」这条规矩，这一批没有。

## 交叉旁证：另一条算法给的填充列数

链（`read_health`）是「一带 8 行取逐列中位色 → 找左缘 → 自左缘起连续亮列（容
GAP_TOLERANCE 列）」。旁证换成**另一条路**：取条中间那一行，自候选左缘起**逐像素
一路数下去**（不许有断口、不取中位、不在 ±COL_TOL 里搜）。两条路**共用**颜色口径
（`HP_BLUE` / `HP_COLOUR_TOL`），算法不同。

它不是真值，**只是旁证**：两条路差出 2 列以上才值得看一眼。真值（面板上的
生命 当前/上限）在这一批**没有落** —— 这一批带上详情面板的帧有好几张，
要不要把它做成真值批，是**规划者的事**，不是这张单子的事。

## 这条旁证有一处**已知的错**：紧挨着的两条

用户给的画面事实：**紧挨着放置干员时，血条会小部分重叠，重叠那几列的底色会加深。**
条宽 146、相邻两条左缘只差 138 → 重叠 8 列。

旁证是「遇到第一个不是血条蓝的列就停」，所以它**会停在那道加深的接缝上**，把满条
数成 136 / 137 列；链那边 `GAP_TOLERANCE = 4` 跨过接缝接着算，报 146（满）——
**这一处是旁证错、链对**。所以读数里凡是 `tight_adjacent` 为真的块，
`delta_columns` **不许当成链的毛病**读。要判「满没满」只能靠同一帧里那根条的
**技力条**：技力条没满，它露在外面的轨道到哪儿结束，那一帧的条就有多宽。

## 判据

`判据读数.json` 里逐块落三条链的读数，另外落三条**可判**的线：

1. **满条应当读满**：旁证数到 ≥ 145 列的，链的比例应当是 1.0；
2. **不满的条，两条路不许差过 2 列**（链读到的那一截 vs 旁证数的列数）；
3. **链读出来的列数不许比旁证多**（多出来 = 跨过了本该停的地方）。

**它不判断"够不够"** —— 那是规划者的事。
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import bench_field as bf  # noqa: E402

from ark.engine.vision import field  # noqa: E402
from ark.engine.vision.frame import frame_from_raw  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402
from ark.offline.bench.feed import replay_frames  # noqa: E402

RUN = REPO / "screenshots" / "run-01"
BATCH = REPO / "tests" / "帧样本" / "场上干员难例"
REPORT = BATCH / "判据读数.json"

STAGING = REPO / "data" / "bench-staging" / BATCH.name
"""建批的**工作目录**：整批先在它里面建全，再一次性换到 `BATCH` 上。

**为什么不挨着 `BATCH` 放**：`tests/` 是纳管的，跑到一半崩了会在版本库里留一个
目录；`data/` 是 gitignored 的草稿区，而且是同一个卷 —— 改名就是原子的。
`_swap_in()` 换完之后把它删掉；下一次 `stage_batch()` 开工前先整个清一次，
所以上一次留下的残渣不会被当成这一批的东西换上去。
"""

CROP_WIDTH = bf.CROP_WIDTH
CROP_HEIGHT = bf.CROP_HEIGHT
CROP_LEFT = bf.CROP_LEFT

WIDTH_TOLERANCE = 2
"""两条路最多差两列 —— 一步是 1/146，两列是抗锯齿那一档。"""

DIAMOND_SCORE_MIN = 0.45
"""整帧搜菱形时的下限。老那两批里「在」是 0.543–0.895、「不在」是 0.129–0.279，
所以 0.45 落在空档里 —— 它只用来**找候选位置**，不当判定。"""

DIAMOND_BAND = (800, 350, 2000, 850)
"""只在中央战区那一带滑模板：整帧 2800×1260 太慢，而干员都在这一带。"""

DIAMOND_MIN_COLUMNS = 8
"""菱形盖住条**至少这么多列**才算「压在这根条上」，少于此只当擦边。"""

_write_json = bf._write_json
"""落 JSON 的写法与 `bench_field.py` **同一份实现**（LF、UTF-8、缩进 2、末尾换行）。"""


def _label_of(path: Path) -> str:
    """用户贴在文件名上的标签（`hard_` / `special_` 那一段）。其余帧没有标签。"""
    stem = path.stem
    for prefix in ("hard_", "special_"):
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return ""


def _frames() -> list[Path]:
    return sorted(RUN.glob("*.png"))


def _independent_fill(pixels: np.ndarray, bar_x: int, bar_y: int) -> int:
    """旁证：条中间那一行，自 `bar_x` 起**逐像素**数下去，遇到第一个不像血条蓝的就停。

    不许有断口、不取中位、不在 ±COL_TOL 里搜 —— 与链走的是两条路。
    """
    row = pixels[bar_y + field.BAR_HEIGHT // 2]
    distance = np.abs(
        row.astype(np.int16) - np.asarray(field.HP_BLUE, dtype=np.int16)
    ).max(axis=1)
    lit = distance <= field.HP_COLOUR_TOL
    count = 0
    for x in range(bar_x, min(bar_x + 3 * field.BAR_WIDTH, pixels.shape[1])):
        if not lit[x]:
            break
        count += 1
    return count


def _diamonds(source: np.ndarray) -> list[dict[str, object]]:
    """整帧搜黄菱形 → 候选位置（分数 ≥ DIAMOND_SCORE_MIN）。

    用的是链自己的模板（`field.load_ready_diamond()`）与 cv2 的归一化相关，
    但**搜索范围与邻域抑制都不走链** —— 链是被喂一个取区、在取区里 ±60 px 找；
    这一步是**建批侧**在为「这一帧的菱形在哪」取证，所以它不参与读数。
    """
    import cv2

    template = field.load_ready_diamond()
    height, width = template.shape[:2]
    x0, y0, x1, y1 = DIAMOND_BAND
    region = np.ascontiguousarray(source[y0:y1, x0:x1])
    score = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    work = score.copy()
    found: list[dict[str, object]] = []
    for _ in range(24):
        _, best, _, at = cv2.minMaxLoc(work)
        if best < DIAMOND_SCORE_MIN:
            break
        mx, my = int(at[0]) + x0, int(at[1]) + y0
        found.append(
            {"score": round(float(best), 4), "at": [mx, my], "size": [width, height]}
        )
        work[
            max(0, int(at[1]) - 20) : int(at[1]) + 20,
            max(0, int(at[0]) - 20) : int(at[0]) + 20,
        ] = -1.0
    return found


def _overlap_columns(
    diamond: dict[str, object], bar: list[int]
) -> tuple[int, int]:
    """菱形压在 (血条那一行 / 技力条那一行) 上各多少列。"""
    mx, my = (int(v) for v in diamond["at"])  # type: ignore[arg-type]
    dw, dh = (int(v) for v in diamond["size"])  # type: ignore[arg-type]
    bar_x, bar_y = int(bar[0]), int(bar[1])
    counts = []
    for top in (bar_y, bar_y + field.SKILL_ROW_OFFSET):
        rows = max(
            0,
            min(my + dh, top + field.BAR_HEIGHT) - max(my, top),
        )
        columns = max(
            0, min(mx + dw, bar_x + field.BAR_WIDTH) - max(mx, bar_x)
        )
        counts.append(columns if rows > 0 else 0)
    return counts[0], counts[1]


def _clear(path: Path) -> None:
    """把一个（可能不在的）目录整个删掉。"""
    if path.exists():
        shutil.rmtree(path)


def _build_into(where: Path) -> dict[str, object]:
    """把整批落进 `where`：`*.png` + `index.json` + `取材.json`。

    `where` 是**工作目录**（`stage_batch()` 建的那一个），不是现役目录 ——
    建到一半失败时，现役目录里连一个字节都还没动过。
    """
    crops: list[dict[str, object]] = []
    harvest: list[dict[str, object]] = []
    diamonds_by_frame: dict[str, list[dict[str, object]]] = {}
    for frame_index, path in enumerate(_frames(), start=1):
        source = bf._read_source(str(path.relative_to(REPO)).replace("\\", "/"))
        diamonds_by_frame[path.name] = _diamonds(source)
        candidates = bf._bar_candidates(source)
        if not candidates:
            raise ValueError(f"{path.name} 上一根候选条都没有：这一帧不该进补测批")
        for bar_index, (bar_x, bar_y, lit) in enumerate(candidates, start=1):
            name = f"r01_{frame_index:02d}_{bar_index:02d}"
            origin_x = bar_x - CROP_LEFT
            origin_y = max(0, bar_y - 340)
            block = source[origin_y : origin_y + CROP_HEIGHT, origin_x : origin_x + CROP_WIDTH]
            if block.shape[:2] != (CROP_HEIGHT, CROP_WIDTH):
                raise ValueError(f"{name} 裁出来是 {block.shape[:2]}，落不进画布")
            file_name = f"{name}.png"
            Image.fromarray(block).save(where / file_name)
            crops.append(
                {
                    "file": file_name,
                    "origin": [origin_x, origin_y],
                    "size": [CROP_WIDTH, CROP_HEIGHT],
                    "source": str(path.relative_to(REPO)).replace("\\", "/"),
                    "source_sha256": bf._sha256(path),
                }
            )
            window = field.operator_ui_window(
                bf.PixelRect(bar_x, bar_y, field.BAR_WIDTH, 16)
            )
            harvest.append(
                {
                    "file": file_name,
                    "frame": path.name,
                    "label": _label_of(path),
                    "bar": [int(bar_x), int(bar_y)],
                    "candidate_lit": int(lit),
                    "independent_fill": _independent_fill(source, bar_x, bar_y),
                    "distance_to_centre": round(
                        bf._distance_to_centre((bar_x, bar_y, lit)), 1
                    ),
                    "window": [
                        int(window.x),
                        int(window.y),
                        int(window.width),
                        int(window.height),
                    ],
                    "window_from": (
                        "候选条左上角 ("
                        + str(bar_x)
                        + ", "
                        + str(bar_y)
                        + ") − BARS_IN_WINDOW → operator_ui_window()"
                    ),
                    "source": str(path.relative_to(REPO)).replace("\\", "/"),
                }
            )
    manifest = {
        "version": 1,
        "canvas": [bf.CANVAS_2800x1260.width, bf.CANVAS_2800x1260.height],
        "device": bf.DEVICE,
        "note": (
            "场上我方干员那三条读数链的**补测批**：用户连拍里挑的 12 张难例"
            "（一格里有好几根条、挨得近、被黄菱形压住），按 `场上干员` 同一套尺寸裁成"
            " 300×420。它**不是**真值批（没有面板真值落进来），也**不是**基准集。"
        ),
        "crops": crops,
    }
    _write_json(where / "index.json", manifest)
    _write_json(
        where / "取材.json",
        {
            "note": (
                "被测对象 = `bench_field._bar_candidates()` 在整帧上的**全部输出**，"
                "一个不筛。所以 `bar` 是**候选坐标**而不是真值坐标；同一根条可能被记成"
                "两个候选（被背景切成两段），两条裁剪各是一块合法的窗。"
                "`candidate_lit` 是查找器给的连段长度，**不许当填充读数**"
                "（条宽 146，而相邻两条左缘只差 138 —— 邻条会盖掉本条的轨道尾巴）。"
            ),
            "window_size": [field.OPERATOR_UI_WIDTH, field.OPERATOR_UI_HEIGHT],
            "bars_in_window": list(field.BARS_IN_WINDOW),
            "crop_size": [CROP_WIDTH, CROP_HEIGHT],
            "crop_left_margin": CROP_LEFT,
            "pick_rule": (
                "每帧跑 bench_field._bar_candidates(source)（执行者写的候选查找器），"
                "它给出几根就裁几块；12 帧共 " + str(len(harvest)) + " 块"
            ),
            "frame_count": len(_frames()),
            "diamonds": diamonds_by_frame,
            "diamonds_note": (
                "整帧搜到的黄菱形候选位置（模板 = assets/features/ready_diamond，"
                "搜索带 = " + str(DIAMOND_BAND) + "，下限 " + str(DIAMOND_SCORE_MIN) + "）。"
                "**建批侧的取证**，不参与读数；用来判「用户标的『菱形挡住技能条』到底压在哪一根上」。"
            ),
            "crops": harvest,
        },
    )
    return manifest


def stage_batch() -> tuple[dict[str, object], Path]:
    """**把整批先建到工作目录里**，返回（清单, 工作目录）—— 现役目录一个字节都不动。

    上一版是「先把现役目录里的 `*.png` 全删掉、再一张张重建」：建到一半失败
    （例如某个 PNG 正被别的进程占着，`unlink()` 抛 `WinError 32`）就留下
    **盘上只剩几张、报告却还是上一版** 的局面 —— 报告说「52 块」而那 52 块已经不在
    盘上，是**假事实**，`load_batch()` 下一次读它直接抛 `FileNotFoundError`
    （实测：占住 `r01_12_05.png` 跑旧的那版，盘上剩 1 张、清单 52 条、51 条找不到）。
    现在先建全再换：中途失败就把工作目录清掉，**现役目录原样**。
    """
    _clear(STAGING)
    STAGING.mkdir(parents=True)
    try:
        manifest = _build_into(STAGING)
    except BaseException:
        shutil.rmtree(STAGING, ignore_errors=True)
        raise
    return manifest, STAGING


def _swap_in(staging: Path) -> None:
    """把工作目录整批换到现役目录上。**要么整批换过去，要么一个都不动。**

    换法只有两步目录改名：现役目录 → 备份目录，工作目录 → 现役目录。**第一步要是被
    占用挡住，它一个文件都没搬**（Windows 不许改一个里面有文件的删除共享没打开的
    目录的名字，实测抛 `WinError 5` 且原目录原样）；第二步动的是我们自己刚建的那个
    目录，失败就把备份改回来。

    `README.md`（手写的说明书）与**单独调 `build_batch()` 时**工作目录里没有的那份旧
    报告，先拷进工作目录再换 —— 它们照旧跟着整批一起搬过去，不会被丢掉。
    """
    for keep in ("README.md", REPORT.name):
        if (BATCH / keep).is_file() and not (staging / keep).is_file():
            shutil.copyfile(BATCH / keep, staging / keep)
    backup = STAGING.with_name(STAGING.name + "__old")
    if BATCH.exists():
        _clear(backup)
        BATCH.rename(backup)
    try:
        staging.rename(BATCH)
    except BaseException:
        if backup.exists():
            backup.rename(BATCH)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def build_batch() -> dict[str, object]:
    """建批：**先在工作目录里建全整批，再整批换上去**，返回清单。

    换上去的整批里有 `*.png` 与两份 JSON；单独调它时，现役目录里那份旧报告**不会丢**
    （换之前先拷进工作目录）。
    """
    manifest, staging = stage_batch()
    _swap_in(staging)
    return manifest


def run() -> int:
    _manifest, staging = stage_batch()
    template = field.load_ready_diamond()
    batch = load_batch(staging)
    harvest = {
        item["file"]: item
        for item in json.loads((staging / "取材.json").read_text(encoding="utf-8"))["crops"]
    }
    diamonds = json.loads((staging / "取材.json").read_text(encoding="utf-8"))["diamonds"]

    rows: list[dict[str, object]] = []
    for case, raw in replay_frames(batch):
        frame = frame_from_raw(raw)
        entry = harvest[case.name + ".png"]
        window = bf.PixelRect(*entry["window"])
        health = field.read_health(frame, window)
        fill, meaning = field.read_skill(frame, window)
        diamond = field.find_ready_diamond(frame, window, template)
        width = bf._number(health)
        chain_columns = None if width is None else round(width * field.BAR_WIDTH, 3)
        independent = int(entry["independent_fill"])
        on_bar = []
        for hit in diamonds.get(entry["frame"], []):
            bar_columns, skill_columns = _overlap_columns(hit, entry["bar"])
            if max(bar_columns, skill_columns) >= DIAMOND_MIN_COLUMNS:
                on_bar.append(
                    {
                        "score": hit["score"],
                        "at": hit["at"],
                        "bar_columns": bar_columns,
                        "skill_columns": skill_columns,
                    }
                )
        rows.append(
            {
                "name": case.name,
                "frame": entry["frame"],
                "label": entry["label"],
                "bar": entry["bar"],
                "candidate_lit": entry["candidate_lit"],
                "independent_fill": independent,
                "health": bf._slot_json(health),
                "health_columns": chain_columns,
                "delta_columns": (
                    None if chain_columns is None else round(chain_columns - independent, 3)
                ),
                "skill_fill": bf._slot_json(fill),
                "skill_meaning": bf._slot_json(meaning),
                "sp": bf._slot_json(field.sp_of(fill, meaning)),
                "diamond": bf._slot_json(diamond),
                "score": round(
                    float(field.diamond_score(frame, window, template)[0]), 6
                ),
                "diamonds_on_this_bar": on_bar,
            }
        )
        print(
            f"{case.name}  {entry['frame'][:28]:28s} 条{entry['bar']}  "
            f"旁证 {independent:3d} 列  链 {health!r}  "
            f"技力 {fill!r}/{meaning!r}  菱形 {diamond!r}"
        )

    full = [row for row in rows if int(row["independent_fill"]) == field.BAR_WIDTH]
    full_not_one = [
        row["name"]
        for row in full
        if row["health"].get("payload", {}).get("value") != 1.0
    ]
    wide = [
        row
        for row in rows
        if row["delta_columns"] is not None
        and abs(float(row["delta_columns"])) > WIDTH_TOLERANCE
    ]
    over = [
        row
        for row in rows
        if row["delta_columns"] is not None and float(row["delta_columns"]) > WIDTH_TOLERANCE
    ]
    unsampled = [row["name"] for row in rows if row["health"]["state"] == "unsampled"]
    tight: set[str] = set()
    for row in rows:
        for other in rows:
            if other is row or other["frame"] != row["frame"]:
                continue
            if abs(int(other["bar"][1]) - int(row["bar"][1])) > 2:
                continue
            if 0 < abs(int(other["bar"][0]) - int(row["bar"][0])) < field.BAR_WIDTH:
                tight.add(str(row["name"]))
    for row in rows:
        row["tight_adjacent"] = row["name"] in tight
    occluded = [row for row in rows if row["diamonds_on_this_bar"]]
    skill_unsampled = [
        row["name"] for row in rows if row["skill_fill"]["state"] == "unsampled"
    ]

    report = {
        "note": (
            "补测批的判据读数。**它不判断够不够** —— 只逐块落三条链的读数、"
            "另一条算法给的填充列数、以及三条可判的线。"
        ),
        "batch": "tests/帧样本/场上干员难例",
        "version": batch.version,
        "device": batch.device,
        "ruler": {
            "health": "链读出来的比例 × BAR_WIDTH = 链数到的列数；旁证 = 条中间那一行自候选左缘逐像素数下去、遇断口即停",
            "tolerance_columns": WIDTH_TOLERANCE,
        },
        "proof_caveat": (
            "旁证在**紧挨着的两条**上是错的：条宽 146、相邻两条左缘差 138 → 重叠 8 列，"
            "重叠处的底色加深，旁证会停在那道接缝上（把满条数成 136/137 列），"
            "而链按 GAP_TOLERANCE=4 跨过去报 146。凡 `tight_adjacent` 为真的块，"
            "`delta_columns` 不许当成链的毛病读。**另一处（同一个根因）**：条右端被压暗时旁证"
            "**在压暗那一列就停**（它只认 ≤ HP_COLOUR_TOL=24），而链报 146（满）—— 这一档"
            "**有三块**：`r01_06_04`（最后 6 列 25–29、条外 186，旁证数成 140）、"
            "`r01_05_04`（24 / 25 / 26 / 27 / 27、条外 191，旁证 142）、"
            "`r01_07_02`（32 / 36 / 40 / 41 / 41、条外 203，旁证 143）。"
            "这三处都是**旁证错、链对**。"
        ),
        "frame_count": len(_frames()),
        "crops": len(rows),
        "reads": rows,
        "lines": {
            "full_should_read_one": {
                "which": "旁证 ≥ " + str(field.BAR_WIDTH - 1) + " 列的块",
                "count": len(full),
                "not_one": full_not_one,
            },
            "delta_over_tolerance": {
                "count": len(wide),
                "rows": [
                    {
                        "name": row["name"],
                        "frame": row["frame"],
                        "bar": row["bar"],
                        "independent": row["independent_fill"],
                        "chain_columns": row["health_columns"],
                        "delta": row["delta_columns"],
                        "tight_adjacent": row["tight_adjacent"],
                    }
                    for row in wide
                ],
            },
            "chain_wider_than_proof": {
                "count": len(over),
                "names": [row["name"] for row in over],
            },
            "health_unsampled": {"count": len(unsampled), "names": unsampled},
            "skill_unsampled": {
                "count": len(skill_unsampled),
                "names": skill_unsampled,
            },
            "diamond_occluded": {
                "which": "有黄菱形压在血条那一行或技力条那一行上 ≥ "
                + str(DIAMOND_MIN_COLUMNS)
                + " 列的块",
                "count": len(occluded),
                "rows": [
                    {
                        "name": row["name"],
                        "frame": row["frame"],
                        "label": row["label"],
                        "bar": row["bar"],
                        "diamond": row["diamonds_on_this_bar"],
                        "health": row["health"],
                        "skill_fill": row["skill_fill"],
                        "skill_meaning": row["skill_meaning"],
                    }
                    for row in occluded
                ],
            },
        },
        "coverage": {
            "covered": [
                "一格里有好几根条：12 帧里 9 帧的候选数 ≥ 4（最多 6 根）",
                "相邻两条挨着（左缘差 138 px，条宽 146 → 邻条盖住本条轨道尾巴 8 列）",
                "上下错位挨着（相邻候选的 y 差 12 / 11 / 11 行）",
                "被黄菱形压住的那一档：用户标了「应报未采样」的那一帧",
                "详情面板开着的那一档（面板自己那条宽蓝条与脚底条同一高度）",
                "**「满」与「差最后一小截」分开**：`r01_04_03` 那一块的尾巴（8 列）"
                "是外来颜色（离填充色 30–44，按距离尺子算「填着」）—— 现在落"
                "`LowerBound(138/146)`（技能那个字段加宽成区间），**不再读成 `Exact(1.0)`**",
                "**技力含义的「认不出」那一档**（颜色两族都不是）现在有真帧了 —— **覆盖文本的正文"
                "就是这一条**（其余批次只引用它，不重抄）：`r01_01_05` / `r01_01_06`（技力条被详情"
                "界面的图标盖住）→ 技力那三个槽落**未采样**；`r01_06_04`（空轨道被背景染了色）→"
                " 技力落**精确 0**、**含义照实交 `Unknown`**（「认得出有东西、但读不准」那一族 ——"
                " **不是**未采样：技力未知与技力为 0 是两个完全不同的行为）、血条落**满**。"
                "**这两档的分法是「条内比条外暗了多少」**（`field._looks_like_an_empty_track`）："
                "空的条是半透明的、在那一格留下一条**条形变暗区** → 两条长边各自量一次，两端都要"
                "有落差、而且落差要落在半透明该有的那一段里（`r01_06_04` 是 **0.159 / 0.300**；"
                "`r01_01_05` 的左缘 **0.860** 过头了、`r01_01_06` 的右缘 **−0.155** 是反过来的）。"
                "所以 **「条在、只是空了」与「条不在」分得开**。**「空轨道单独可见」与「技力条掉到"
                "零」这两档就是 `r01_06_04`**：它是「条在、只是空了」，不是「一条全暗的条」；"
                "**空轨道的颜色落在一族里**（透出来的背景是绿或橙）时含义读得出，那一档仍然没有真帧",
            ],
            "not_covered": [
                "**没有真值**：这一批不带面板上的 生命 当前/上限，所以判不出「链读的比对不对」，"
                "只判得出「链与旁证同不同」",
                "**极低填充**（< 3%）仍然一帧都没有",
                "**没有可见接缝的外来尾巴**：外来色**平滑过渡**进填充色（与填充色连着、"
                "中间没有断口）时，最右那一截仍然算「本条填充色」 → 仍读满。这一档"
                "一帧都没有，现在也**判不出来**",
                "**「满条 + 两端落差落在带子里」**：一根**颜色两族都不是**、而两端落差又恰好"
                "落在半透明那一段里的**满条**，会被读成**空轨道 → 精确 0**（假事实）。"
                "这一档**一帧都没有**（`shot30_a` 原位就是这个形状，只是它今天到不了那一支）",
                "**那条带子的上下沿不是统计来的**：空条那一簇与被压住那一簇**各只有一块帧**"
                "（`r01_06_04` 对 `r01_01_05`），上下沿取的是两块之间的中项 —— 余量只有 1.7 倍",
                "**只有一边能量的块**：`r01_01_06` 的左缘被旁边那根条带偏（`_left_edge` 锁到了"
                "邻居的蓝带、左外那 8 列越出取区）→ 那一块只有**半边证据**，判据只由右缘一票定",
            ],
        },
    }
    _write_json(staging / "判据读数.json", report)
    _swap_in(staging)
    print()
    print(f"批：{len(rows)} 块；旁证满条 {len(full)} 块，其中读到不是 1.0 的 {full_not_one}")
    print(f"两条路差过 {WIDTH_TOLERANCE} 列的：{len(wide)} 块；链比旁证宽的：{len(over)} 块")
    print(f"血条落未采样的：{len(unsampled)} 块 {unsampled}")
    print(f"技力落未采样的：{len(skill_unsampled)} 块 {skill_unsampled}")
    print(f"黄菱形压住条的：{len(occluded)} 块 {[row['name'] for row in occluded]}")
    print(f"报告落在 {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
