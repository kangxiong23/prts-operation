"""建批那一侧「整批换上去」的原子性（`scripts/bench_field_hard.py`）。

这一条管的是**假事实**那一档。旧版 `build_batch()` 先把现役目录里的 `*.png` 全删掉、
再一张张重建：建到一半失败（某个 PNG 正被别的进程占着 → `unlink()` 抛 `WinError 32`）
就留下「盘上只剩几张、报告却还是上一版」—— 报告说 52 块，而那 52 块已经不在盘上。
实测（真帧、真句柄）：盘上剩 1 张、清单 52 条、51 条找不到，`load_batch()` 下一次读
它直接抛 `FileNotFoundError`。

所以这里钉三条：建到一半失败 ⇒ **现役目录一个字节不动**；现役的裁剪被占住 ⇒ 换不过去，
但**也没搬走一个文件**；单独调一次 `build_batch()` ⇒ 旧报告还在、孤儿没了。

用例跑的是**合成帧**（`_read_source` 换成纯色画布），不碰真素材，也不往仓库里写东西
—— 临时目录用 `test_vision_bench.py` 那份 `scratch`（**不用 pytest 的 `tmp_path`**：
本机沙箱下它建出来的目录连自己都进不去）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import bench_field_hard as bh  # noqa: E402

from ark.offline.bench.batch import load_batch  # noqa: E402

CANVAS = (1260, 2800, 3)
"""合成帧的尺寸（高, 宽, 通道）—— 与真帧同一张画布，裁剪才落得进去。"""

FRAMES = ("a.png", "b.png")
"""两帧，每帧各给一根候选条 ⇒ 四块裁剪。

名字不必真的存在于盘上：`_read_source` 与 `_sha256` 都会被用例换掉。"""

CROP = (420, 300)
"""裁剪的高、宽。"""


def _frame_paths() -> list[Path]:
    return [REPO / "screenshots" / "run-01" / name for name in FRAMES]


@pytest.fixture
def scratch() -> Iterator[Path]:
    """一个**自己能进去**的临时目录（照 `test_vision_bench.py` 那份）。"""
    root = Path(tempfile.gettempdir()) / "ark-bench-hard-tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex[:12]
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _fingerprint(where: Path) -> dict[str, str]:
    """目录里每个文件的指纹（折了行尾 —— 这一条只比变没变）。"""
    out: dict[str, str] = {}
    for item in sorted(where.rglob("*")):
        if item.is_file():
            raw = item.read_bytes()
            out[str(item.relative_to(where))] = hashlib.sha256(
                raw.replace(b"\r\n", b"\n")
            ).hexdigest()
    return out


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _plant_old_batch(live: Path) -> None:
    """造一批**旧的**：两张真能读的裁剪 + 清单 + 取材 + 一份报告 + 手写的 README。"""
    live.mkdir(parents=True, exist_ok=True)
    _write(live / "README.md", "手写的说明书\n")
    _write(live / "判据读数.json", '{"old": true}\n')
    crops = []
    for name in ("old_01", "old_02"):
        Image.fromarray(np.zeros(CROP + (3,), dtype=np.uint8)).save(live / (name + ".png"))
        crops.append(
            {
                "file": name + ".png",
                "origin": [0, 0],
                "size": [CROP[1], CROP[0]],
                "source": "screenshots/run-01/a.png",
                "source_sha256": "0" * 64,
            }
        )
    _write(
        live / "index.json",
        json.dumps(
            {"version": 1, "canvas": [2800, 1260], "device": "旧的", "crops": crops},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _write(live / "取材.json", '{"crops": []}\n')


def _synthetic(monkeypatch, live: Path, staging: Path, candidates) -> None:
    """把建批那一侧换成合成的：不读真帧、不搜菱形、候选由用例给定。"""
    blank = np.zeros(CANVAS, dtype=np.uint8)
    monkeypatch.setattr(bh, "BATCH", live)
    monkeypatch.setattr(bh, "STAGING", staging)
    monkeypatch.setattr(bh, "_frames", _frame_paths)
    monkeypatch.setattr(bh, "_diamonds", lambda source: [])
    monkeypatch.setattr(bh.bf, "_read_source", lambda relative: blank)
    monkeypatch.setattr(bh.bf, "_sha256", lambda path: "0" * 64)
    monkeypatch.setattr(bh.bf, "_bar_candidates", candidates)


def test_a_build_that_dies_midway_leaves_the_old_batch_alone(scratch, monkeypatch) -> None:
    """第二帧炸掉 ⇒ 现役目录**一个字节都没动**，而且工作目录被清掉了。"""
    live = scratch / "live"
    staging = scratch / "staging"
    _plant_old_batch(live)
    before = _fingerprint(live)
    seen = {"n": 0}

    def dies_on_the_second_frame(source):
        seen["n"] += 1
        if seen["n"] > 1:
            raise RuntimeError("第二帧建到一半炸了")
        return [(100, 200, 146)]

    _synthetic(monkeypatch, live, staging, dies_on_the_second_frame)
    with pytest.raises(RuntimeError):
        bh.build_batch()

    assert seen["n"] == 2, "第二帧那条路要真的走到"
    assert _fingerprint(live) == before, "现役目录被动了"
    assert len(load_batch(live).cases) == 2, "批还得是读得出来的"
    assert not staging.exists(), "失败之后工作目录要清掉，不许留半个批"


def test_a_standalone_build_keeps_the_old_report_and_drops_orphans(scratch, monkeypatch) -> None:
    """单独调一次 `build_batch()`：孤儿清掉、盘上 ↔ 清单一一对应、旧报告还在。"""
    live = scratch / "live"
    staging = scratch / "staging"
    _plant_old_batch(live)
    shutil.copyfile(live / "old_01.png", live / "r01_99_99.png")
    report = (live / "判据读数.json").read_bytes()
    _synthetic(monkeypatch, live, staging, lambda source: [(100, 200, 146), (400, 260, 120)])

    manifest = bh.build_batch()

    listed = sorted(str(item["file"]) for item in manifest["crops"])
    on_disk = sorted(item.name for item in live.glob("*.png"))
    assert on_disk == listed, "盘上的裁剪与清单对不上"
    assert not (live / "r01_99_99.png").is_file(), "孤儿还在"
    assert (live / "判据读数.json").read_bytes() == report, "旧报告被动了"
    assert (live / "README.md").is_file(), "手写的说明书没了"
    assert not staging.exists()
    assert not (scratch / (staging.name + "__old")).exists(), "备份目录没清掉"


def test_a_locked_crop_refuses_the_swap_without_moving_anything(scratch, monkeypatch) -> None:
    """现役的裁剪被占住 ⇒ 换不过去，但**一个文件都没搬走**（这就是那条假事实的反面）。"""
    live = scratch / "live"
    staging = scratch / "staging"
    _plant_old_batch(live)
    before = _fingerprint(live)
    _synthetic(monkeypatch, live, staging, lambda source: [(100, 200, 146)])

    with open(live / "old_02.png", "rb") as held:
        with pytest.raises(PermissionError):
            bh.build_batch()
        assert _fingerprint(live) == before, "换失败还搬走了东西"
        assert held.read(4) == b"\x89PNG", "占住的那个句柄一直是有效的"

    assert _fingerprint(live) == before
    assert len(load_batch(live).cases) == 2

