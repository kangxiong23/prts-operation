
r"""同一个干员**两套立绘同时在库里**（20 条库）时的读数表。

    .venv\Scripts\python.exe scripts\bench_roster_twins.py

## 这一单问什么

`tests/帧样本/立绘两组/` 那四张是**两组**，每组「一张编队页 + 一张备战席横带」：
**人员与位置完全相同，只有立绘不同**。库把两组合成 **20 条**，于是每一对同名条目
是一对**双胞胎**（同一个干员的另一套立绘）。口径是 **「立绘就是身份」** ——
读出**另一套立绘**也算失败。

这一单**只量**：线 · 取区形状 · 模型 · 那条链 · 尺度区间 · 阈值**一个都不动**。

## 口径

- **库**：两组编队页各 10 条（`库1.json` / `库2.json`），合起来 20 条；标签是
  `组N·干员名` —— 同名两条会让「他在第几张卡」没有唯一答案，所以**必须唯一**；
- **取区是第二个自变量**：**干净卡那块框**（`roster.identity_region`，整框、不掩码、
  走线上那条链 `roster_scores`）与**红罩卡那块圆盘**（`roster.redcap_region` 加上
  `margin.MaskedRuler`）各测一份 —— **不让 `redeploying` 替这一单挑**（那是线上
  那条分支的事）。这一批的卡**全是干净卡**，所以这一条实际是**在干净卡上比两套取区**；
- **判据一个字没动**：最高分过 `ROSTER_SOLID_MATCH_SCORE` **而且** 余量拉开
  `ROSTER_AMBIGUITY_MARGIN` 才算精确，否则未知；**每一格都把次高分 · 余量 ·
  被哪条线挡的（`blocked_by`）落下来**，所以挪线可以离线重算；
- **三种库范围**并排：本组 10 条 · 另一组 10 条 · 全部 20 条。三者的分数**取自同一批
  20 条分数** —— 不是各跑一遍：一个库条目对一块取区的分数**与库里还有谁无关**
  （`roster_scores` 与 `MaskedRuler.scores` 都是逐条目独立算的），所以"子集"
  就是那把尺子只拿那几条库时会给的分数。

## 「另一套立绘 / 别人 / 读不出」怎么分

**真值住规划者那一侧**（这一单里执行者不填真值），所以这份表只落**能自证的**：

- **读不出**（未知）：不需要真值；
- **「另一组那 10 条」这一档**：卡上那一套**根本不在库里**（库里只有它的双胞胎），
  所以**凡是有判定的一律错** —— 这一条也不需要真值，那是这一档的定义；
- **读到哪一条**：逐卡落**标签 · 分数 · 余量**，再加上**双胞胎**给出的结构读数
  —— 同一位干员的两套立绘在**同一张卡**上是不是指到同一条库、单条那条路有没有指到
  另一条、同一条库在两张带上是不是指到同一个卡位。这是不需要真值就能判的那一半。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.engine.device.adb import SQUAD_CARD_PITCH  # noqa: E402
from ark.engine.vision import bench, margin  # noqa: E402
from ark.engine.vision.frame import frame_from_raw  # noqa: E402
from ark.engine.vision.roster import (  # noqa: E402
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
    find_squad_card,
    identity_region,
    redcap_region,
    roster_scores,
)
from ark.offline.assets_gen.cardface import load_cardfaces, load_formation_page  # noqa: E402
from ark.offline.bench.batch import load_batch  # noqa: E402

BATCH = REPO / "tests" / "帧样本" / "立绘两组"
OUT = BATCH / "点名.json"
PITCH = SQUAD_CARD_PITCH
"""卡距。它住设备层，视觉层不许 import 设备层，所以照调用方的样子传进来。"""

GROUPS = ("1", "2")
"""两组的组号 —— 目录里那一列文件的命名就是它。"""

SCOPE_SELF = "本组 10 条"
SCOPE_OTHER = "另一组 10 条"
SCOPE_ALL = "全部 20 条"
SCOPES = (SCOPE_SELF, SCOPE_OTHER, SCOPE_ALL)


def _round(value):
    return None if value is None else round(float(value), 6)


def _subset_names(labels, group: str):
    prefix = "组%s·" % group
    return frozenset(name for name in labels if name.startswith(prefix))


def _column(table, index: int):
    """第 index 张卡的 20 条分数 —— 降序，同分按库顺序（与两条路的稳定排序一致）。"""
    labels, scores, scales = table["labels"], table["scores"], table["scales"]
    rows = sorted(
        ((scores[row][index], labels[row], scales[row][index], row)
         for row in range(len(labels))),
        key=lambda item: (-item[0], item[3]),
    )
    return tuple((score, name, scale) for score, name, scale, _ in rows)


def _rank(scored, names):
    """从同一批分数里取**子集**的最高与次高（次序照原样的降序，稳定）。"""
    picked = [item for item in scored if item[1] in names]
    top = picked[0]
    second = picked[1] if len(picked) > 1 else (-2.0, None, None)
    return top, second


def _judge(top, second):
    """线上那条判据 —— **一处定义**，三条范围共用（不许另立一条）。"""
    gap = float(top[0]) - float(second[0])
    ok = float(top[0]) >= ROSTER_SOLID_MATCH_SCORE and gap >= ROSTER_AMBIGUITY_MARGIN
    return ok, gap


def _blocked(top, second) -> str | None:
    """这一格**被哪一条线挡的**（判定过了就返回 None）；两条都没够就两条都写。

    这两样以前分不开 —— 落下来之后，**挪线可以离线重算**。
    """
    if _judge(top, second)[0]:
        return None
    short = []
    if float(top[0]) < ROSTER_SOLID_MATCH_SCORE:
        short.append("分数不够")
    if float(top[0]) - float(second[0]) < ROSTER_AMBIGUITY_MARGIN:
        short.append("余量不够")
    return " + ".join(short) or None


def _reading(top, second) -> dict:
    ok, gap = _judge(top, second)
    return {
        "top": top[1],
        "top_score": _round(top[0]),
        "top_scale": _round(top[2]) if top[2] is not None else None,
        "second": second[1],
        "second_score": _round(second[0]) if second[1] is not None else None,
        "margin": _round(gap),
        "blocked_by": _blocked(top, second),
        "verdict": "精确" if ok else "未知",
    }


def _regions(faces):
    """两套取区的打分器 —— 矩形加打分函数。尺子**建一次**（同一条统计量）。"""
    _box, mask = margin.identity_disc_region()
    ruler = margin.MaskedRuler(mask, faces)
    return {
        "clean": (
            identity_region,
            "roster_scores —— 整框、不掩码，线上那条链",
            lambda region: roster_scores(region, faces),
        ),
        "redcap": (
            redcap_region,
            "margin.MaskedRuler —— 掩码里只有 %d px 算数" % int(mask.sum()),
            lambda region: tuple(ruler.scores(region)),
        ),
    }


def _table(frame, faces, centres, region) -> dict:
    """一张 **20×10** 的分数表：20 条库（固定的标签次序）× 10 张卡。"""
    rect, _statistic, scorer = region
    labels = [name for name, _ in faces]
    scores = {name: [] for name in labels}
    scales = {name: [] for name in labels}
    equator = None
    for position, centre in enumerate(centres):
        block = np.ascontiguousarray(frame.crop(rect(centre, pitch=PITCH)))
        scored = scorer(block)
        if position == 0:
            equator = [(float(s), n, None if k is None else float(k)) for s, n, k in scored]
        for score, name, scale in scored:
            scores[name].append(float(score))
            scales[name].append(None if scale is None else float(scale))
    return {
        "labels": labels,
        "scores": [[_round(scores[name][index]) for index in range(len(centres))]
                   for name in labels],
        "scales": [[_round(scales[name][index]) for index in range(len(centres))]
                   for name in labels],
        "_equator": equator,
    }


def _scopes(table, labels, centres, group: str) -> dict:
    """三条库范围下的 最高 · 次高 · 余量 · 判定（分数取自同一批）。"""
    other = "2" if group == "1" else "1"
    plan = ((SCOPE_SELF, _subset_names(labels, group)),
            (SCOPE_OTHER, _subset_names(labels, other)),
            (SCOPE_ALL, frozenset(labels)))
    out = {}
    for label, names in plan:
        rows = []
        for index in range(len(centres)):
            top, second = _rank(_column(table, index), names)
            rows.append({"index": index, "centre_x": _round(centres[index]),
                         **_reading(top, second)})
        out[label] = rows
    return out


def _single(table, labels, count: int):
    """单条那条路：拿一条库去问「在第几张卡」—— **恰有一张**过线才给下标。"""
    rows = []
    for row, label in enumerate(labels):
        column = [(table["scores"][row][index], index) for index in range(count)]
        best = min(column, key=lambda item: (-item[0], item[1]))
        hits = sorted(index for score, index in column
                      if score >= ROSTER_SOLID_MATCH_SCORE)
        rows.append({
            "library_entry": label,
            "slot": best[1] if len(hits) == 1 else None,
            "hits": hits,
            "best_score": _round(best[0]),
            "best_slot": best[1],
        })
    return rows


def main() -> int:
    batch = load_batch(BATCH)
    frames = {case.name: frame_from_raw(case.raw()) for case in batch.cases}

    pinned = {}
    for group in GROUPS:
        page = load_formation_page(BATCH / ("组%s_编队页.png" % group))
        pinned["组%s_编队页" % group] = bool(
            np.array_equal(page.pixels, frames["组%s_编队页" % group].pixels)
        )

    faces = load_cardfaces(BATCH / "库1.json") + load_cardfaces(BATCH / "库2.json")
    labels = [name for name, _ in faces]
    if len(set(labels)) != len(labels):
        raise ValueError("20 条库里出现了同名两条：那会让「他在第几张卡」没有唯一答案")

    report = {
        "note": (
            "同一个干员**两套立绘同时在库里**（20 条）时的读数表：**只量**，线 · 取区形状 · "
            "模型 · 那条链 · 尺度区间 · 阈值一个字节没动。口径是「立绘就是身份」"
            "（用户裁定）：读出另一套立绘也算失败。"
        ),
        "batch": "tests/帧样本/立绘两组",
        "harvest": "tests/帧样本/立绘两组/取材.json",
        "library": {
            "files": ["tests/帧样本/立绘两组/库1.json", "tests/帧样本/立绘两组/库2.json"],
            "order": labels,
            "note": (
                "两组编队页各 10 条合成一个库（20 条）。标签是 组N·干员名：同名两条会让"
                "「他在第几张卡」没有唯一答案，所以必须唯一。顺序就是这个顺序"
                "（两条路都按分数降序；同分时按库顺序稳定）。"
            ),
        },
        "entry": "ark.engine.vision.roster —— 与线上同一条链、同一块取区、同一条线",
        "region": {
            "clean": {
                "who": "roster.identity_region",
                "box_in_card": [52, 76, 100, 100],
                "pixels": 100 * 100,
            },
            "redcap": {
                "who": "roster.redcap_region",
                "box_in_card": list(margin.IDENTITY_DISC_BOX),
                "pixels": int(margin.identity_disc_mask().sum()),
            },
            "note": (
                "两块取区**分开记、不互相顶替**，而且**两套都跑** —— 不给 redeploying "
                "替这一单挑。这一批的卡全是干净卡（没有红罩），所以这一条实际是在**干净卡"
                "上比两套取区**。"
            ),
        },
        "criterion": {
            "score": ROSTER_SOLID_MATCH_SCORE,
            "margin": ROSTER_AMBIGUITY_MARGIN,
            "note": "最高分过线**而且**余量拉开才算精确；否则未知（**未知不带分数**）。",
        },
        "fixture_selfcheck": {
            "formation_page_paste_matches_png": pinned,
            "note": "基准集那条路把裁剪**贴回画布**；编队页是整帧，所以两者必须逐位相同。",
        },
        "frames": [],
    }

    print("=== 逐帧 ===")
    started = time.perf_counter()
    for group in GROUPS:
        frame_name = "组%s_备战席" % group
        frame = frames[frame_name]
        centres = [float(c) for c, _ in bench.squad_card_centres(frame, pitch=PITCH)]
        top_y = float(bench.squad_card_rect(0, PITCH).y)
        flags = [bench.redeploying(frame, c, card_top=top_y) for c in centres]
        regions = _regions(faces)
        entry = {
            "group": group,
            "frame": frame_name,
            "cards": len(centres),
            "centres": [_round(c) for c in centres],
            "redeploying": [None if f is None else bool(f) for f in flags],
            "regions": {},
        }
        for region_name in ("clean", "redcap"):
            table = _table(frame, faces, centres, regions[region_name])
            scopes = _scopes(table, labels, centres, group)
            entry["regions"][region_name] = {
                "who": regions[region_name][1],
                "table": {"labels": table["labels"], "scores": table["scores"],
                          "scales": table["scales"]},
                "scopes": scopes,
                "exact_under_other_group": [
                    {"index": row["index"], "read": row["top"], "score": row["top_score"],
                     "margin": row["margin"]}
                    for row in scopes[SCOPE_OTHER] if row["verdict"] == "精确"
                ],
                "exact_under_all": [
                    {"index": row["index"], "read": row["top"], "score": row["top_score"],
                     "margin": row["margin"], "second": row["second"],
                     "second_score": row["second_score"]}
                    for row in scopes[SCOPE_ALL] if row["verdict"] == "精确"
                ],
                "unknown_counts": {
                    scope: sum(1 for row in scopes[scope] if row["verdict"] == "未知")
                    for scope in SCOPES
                },
                "single_entry": _single(table, labels, len(centres)),
                "card0_column_rebuilt_equals_the_scorer": (
                    [(_round(s), n) for s, n, _k in table["_equator"]]
                    == [(_round(s), n) for s, n, _k in _column(table, 0)]
                ),
            }
            print("  %s %-6s 表 %d×%d  未知 %s"
                  % (frame_name, region_name, len(labels), len(centres),
                     entry["regions"][region_name]["unknown_counts"]))
        report["frames"].append(entry)

    online = {}
    for group in GROUPS:
        frame_name = "组%s_备战席" % group
        frame = frames[frame_name]
        online[frame_name] = [
            {"library_entry": name, "slot": find_squad_card(frame, name, faces, pitch=PITCH)}
            for name in labels
        ]
        derived = [{"library_entry": r["library_entry"], "slot": r["slot"]}
                   for r in report["frames"][int(group) - 1]["regions"]["clean"]["single_entry"]]
        print("  单条 %s：真入口的卡位 %s（与表里推出来的相同：%s）"
              % (frame_name, [r["slot"] for r in online[frame_name]], online[frame_name] == derived))

    operators = [name.split("·", 1)[1] for name in labels[: len(labels) // 2]]
    twins = []
    for operator in operators:
        row = {"operator": operator}
        for group in GROUPS:
            found = {r["library_entry"]: r["slot"]
                     for r in report["frames"][int(group) - 1]["regions"]["clean"]["single_entry"]}
            left = found["组1·" + operator]
            right = found["组2·" + operator]
            row["组%s 带上：组1 那条指到" % group] = left
            row["组%s 带上：组2 那条指到" % group] = right
            row["组%s 一致" % group] = left == right
        twins.append(row)

    cross = []
    for name in labels:
        slots = {group: next(
            r["slot"] for r in report["frames"][int(group) - 1]["regions"]["clean"]["single_entry"]
            if r["library_entry"] == name) for group in GROUPS}
        cross.append({"library_entry": name, "组1 带上指到": slots["1"],
                      "组2 带上指到": slots["2"], "一致": slots["1"] == slots["2"]})

    report["online_entry"] = online
    report["single_entry_structure"] = {
        "note": (
            "**不需要真值的那一半**：两带的人员与位置相同（用户给的），所以同一位干员的"
            "正确卡位在两条带上**是同一个**，而同一个干员的两套立绘也应该指到同一个卡位。"
            "**指不到一起就说明至少有一条指到了别的卡上**。取区取干净卡那块框"
            "（线上入口在这一批上走的就是它）。"
        ),
        "twins_on_one_band": twins,
        "one_entry_across_two_bands": cross,
    }
    report["seconds"] = _round(time.perf_counter() - started)

    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print()
    print("=== 整批 ===")
    print("  4 张表齐：%s" % sorted("%s/%s" % (f["frame"], r)
                                 for f in report["frames"] for r in ("clean", "redcap")))
    print("  夹具自证（编队页贴回画布 == 直接解 PNG）: %s" % pinned)
    print("  表里第 0 张那一列 == 直接问尺子的那一列: %s"
          % {f["frame"] + "/" + r: f["regions"][r]["card0_column_rebuilt_equals_the_scorer"]
             for f in report["frames"] for r in ("clean", "redcap")})
    print("  耗时 %.1f s" % report["seconds"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
