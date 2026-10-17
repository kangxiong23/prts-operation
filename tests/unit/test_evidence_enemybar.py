"""取证图表头那两半文字（`scripts/evidence_enemybar.py`）。

这一支负责「把真值画给人看」，从前**一条用例都没有** —— 出错了没人会红。这里只钉两件事：
真值那半句拼得对（在 / 不在 + **真值根数**，没数过的写 `?`），以及**表头不盖框**。

「盖没盖住」是这一支特有的坑：表头那一整条黑底是**最后**画上去的、盖满全宽，被它盖住的
东西在图上是**没有痕迹**的（读数在 JSON 里，所以点数照样绿）。出图时 `_guard_header` 会
当场响；这里拿两批**真夹具**的每一块把那条判据走一遍，并反面验证它真的会响。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evidence_enemybar as ev  # noqa: E402

LEFT = REPO / "tests" / "帧样本" / "敌人血条"
RIGHT = REPO / "tests" / "帧样本" / "敌人血条-10-14"
BATCHES = (LEFT, RIGHT)


def _read(root: Path, name: str) -> dict:
    """一块样张链的读数那一份。"""
    report = json.loads((root / "判据读数.json").read_text(encoding="utf-8"))
    return next(read for read in report["reads"] if read["name"] == name)


def _origin(root: Path, name: str) -> list[int]:
    """一块样张在源帧上的原点（图上的坐标要从它折）。"""
    harvest = json.loads((root / "取材.json").read_text(encoding="utf-8"))
    crops = {crop["file"]: crop for crop in harvest["crops"]}
    return crops[name + ".png"]["origin"]


def test_evidence_header_carries_truth_count_and_never_covers_a_box() -> None:
    """真值那半句带**根数**（没数过写 `?`）；两批每一块都不被表头压住。"""
    # **真值主本给的那几个数**（接线对不对，逐块点名）
    for root, name, want in (
        (LEFT, "在_raw_a_左", "truth=in n=1"),
        (LEFT, "在_raw_b_上", "truth=in n=2"),
        (LEFT, "不在_cd01", "truth=out n=0"),
        (RIGHT, "在_boss_红宽条", "truth=in n=3"),
        (RIGHT, "在_互叠1", "truth=in n=?"),
    ):
        assert ev._truth_label(_read(root, name)) == want
    # 链的那半句是原样抄的（真值那一半不许把它挤掉）
    assert ev._readout_label(_read(LEFT, "在_raw_b_上")) == (
        "read in=True count=lower:2 bars=0 small=0 clumps=1 uns=0"
    )

    # **表头不盖框**：两批每一块都走一遍出图时那条判据
    seen = 0
    for root in BATCHES:
        reads = json.loads((root / "判据读数.json").read_text(encoding="utf-8"))["reads"]
        ev.check_reads(reads)
        for read in reads:
            tops = ev._tops(read, _origin(root, read["name"]))
            ev._guard_header(tops)
            for what, top in tops:
                assert top >= ev.HEADER_ROWS, (root.name, read["name"], what, top)
            seen += 1
    assert seen == 18

    # 反面：把一件东西挪进表头，那条判据必须响（不然上面等于没测）
    with pytest.raises(ValueError):
        ev._guard_header([("bar", ev.HEADER_ROWS - 1)])

    # 反面：真值自己跟自己不自洽也得响（真值靠手写，这种错图上和点数里都不会露）
    bad = json.loads(json.dumps(_read(LEFT, "不在_cd01")))
    bad["truth_count"] = 2
    with pytest.raises(ValueError):
        ev.check_reads([bad])
    bad = json.loads(json.dumps(_read(LEFT, "在_raw_a_左")))
    bad["truth_count"] = 0
    with pytest.raises(ValueError):
        ev.check_reads([bad])
