"""基准集加载与离线喂帧的单元测试。

这一批**不 import tests/spec**（D-全局-17：那是规划者的验收套件），也**不抄**
那 31 帧的真值 —— 真值住 `tests/spec/cost_spec.py`。这里测的是"清单读对没有 ·
前提头折成 `DeviceFacts` 没有 · 裁剪贴回画布没有 · 帧经重放适配器吐出来没有"。
"""

from __future__ import annotations

import json
import shutil
import struct
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from ark.contracts import Canvas, DeviceFacts, PixelRect
from ark.engine.device.replay import ReplayDevice
from ark.engine.vision.frame import (
    CHANNELS_PER_PIXEL,
    COLOR_SPACE_SRGB,
    PIXEL_FORMAT_RGBA8888,
    RAW_HEADER_BYTES,
    frame_from_raw,
)
from ark.offline.bench.batch import MANIFEST_NAME, BenchCase, load_batch
from ark.offline.bench.feed import replay_frames

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "tests" / "帧样本" / "费用"


@pytest.fixture
def scratch() -> Iterator[Path]:
    """一个**自己能进去**的临时目录。

    **不用 pytest 的 tmp_path**：它建目录时用 0o700，而 0o700 的 DACL 不继承
    父目录 —— 在受限令牌下（本机 DSH 沙箱就是这样），那个目录连创建它的进程
    自己都进不去。这里用**默认权限**建，父目录的 ACL 会继承下来。
    """
    root = Path(tempfile.gettempdir()) / "ark-offline-bench-tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex[:12]
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_batch(
    directory: Path,
    *,
    crops: list[dict[str, object]],
    canvas: tuple[int, int] = (64, 48),
    version: int = 1,
    device: str = "假机器",
    facts: object = None,
) -> Path:
    """写一批最小的基准集：清单 + 真 PNG。"""
    entries = []
    for position, crop in enumerate(crops):
        size = crop["size"]
        assert isinstance(size, tuple)
        Image.new("RGBA", size, crop.get("color", (10, 20, 30, 255))).save(
            directory / str(crop["file"])
        )
        entries.append(
            {
                "file": crop["file"],
                "origin": list(crop["origin"]),  # type: ignore[arg-type]
                "size": list(size),
                "source": f"data/假/shot{position:02d}.png",
                "source_sha256": "0" * 64,
            }
        )
    manifest: dict[str, object] = {
        "version": version,
        "canvas": list(canvas),
        "device": device,
        "crops": entries,
    }
    if facts is not None:
        manifest["facts"] = facts
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return directory


# ---------------------------------------------------------------------------
# 清单
# ---------------------------------------------------------------------------


def test_the_manifest_fields_are_read_as_recorded() -> None:
    batch = load_batch(BATCH)
    assert batch.version == 1
    assert batch.canvas == Canvas(width=2800, height=1260)
    assert batch.device == "vivo V2338A"
    assert batch.root == BATCH
    assert len(batch.cases) == 31
    # 这一批没记前提头 —— **没有就是没有**，不许补一个默认值。
    assert batch.facts is None


def test_the_frame_name_is_the_file_without_its_extension() -> None:
    names = [case.name for case in load_batch(BATCH).cases]
    assert names[0] == "shot01"
    assert names[-1] == "shot37"
    assert len(set(names)) == len(names)


def test_the_rect_is_the_origin_and_size_from_the_manifest() -> None:
    assert load_batch(BATCH).cases[0].rect == PixelRect(2580, 840, 220, 136)


def test_the_source_is_recorded_verbatim() -> None:
    """出处**只记不验**：原帧住在 gitignored 的 data/ 里，本来就可能已经不在了。"""
    case = load_batch(BATCH).cases[0]
    assert case.source == "data/precheck/live/shot01.png"
    assert (
        case.source_sha256
        == "721f0ff195be5838848430b190ec794d3c113816499adb568bf52a1b3705fae8"
    )


def test_a_batch_without_a_manifest_is_refused(scratch: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_batch(scratch)


def test_a_manifest_without_a_required_field_is_refused(scratch: Path) -> None:
    (scratch / MANIFEST_NAME).write_text(
        json.dumps({"version": 1, "canvas": [64, 48], "crops": []}), encoding="utf-8"
    )
    with pytest.raises(KeyError):
        load_batch(scratch)


def test_a_crop_that_leaves_the_canvas_is_refused(scratch: Path) -> None:
    _write_batch(
        scratch,
        crops=[{"file": "a.png", "origin": (60, 0), "size": (8, 8)}],
        canvas=(64, 48),
    )
    with pytest.raises(ValueError):
        load_batch(scratch)


def test_a_crop_whose_png_is_not_the_recorded_size_is_refused(scratch: Path) -> None:
    _write_batch(
        scratch,
        crops=[{"file": "a.png", "origin": (0, 0), "size": (8, 8)}],
        canvas=(64, 48),
    )
    Image.new("RGBA", (9, 8), (0, 0, 0, 255)).save(scratch / "a.png")
    with pytest.raises(ValueError):
        load_batch(scratch)


def test_two_crops_with_the_same_frame_name_are_refused(scratch: Path) -> None:
    _write_batch(
        scratch,
        crops=[
            {"file": "a.png", "origin": (0, 0), "size": (8, 8)},
            {"file": "a.png", "origin": (16, 0), "size": (8, 8)},
        ],
        canvas=(64, 48),
    )
    with pytest.raises(ValueError):
        load_batch(scratch)


# ---------------------------------------------------------------------------
# 一条 = 一张贴回画布的原始帧
# ---------------------------------------------------------------------------


def test_a_case_gives_a_raw_frame_not_the_crop() -> None:
    case = load_batch(BATCH).cases[0]
    raw = case.raw()
    assert len(raw) == RAW_HEADER_BYTES + 2800 * 1260 * CHANNELS_PER_PIXEL
    assert struct.unpack_from("<4I", raw, 0) == (
        2800,
        1260,
        PIXEL_FORMAT_RGBA8888,
        COLOR_SPACE_SRGB,
    )
    # 不是裁剪本身：裁剪只有 220×136 个像素。
    assert len(raw) != case.rect.width * case.rect.height * CHANNELS_PER_PIXEL
    assert frame_from_raw(raw).canvas == case.canvas


def test_the_crop_is_pasted_back_pixel_for_pixel() -> None:
    case = load_batch(BATCH).cases[0]
    raw = case.raw()
    with Image.open(BATCH / "shot01.png") as image:
        crop = image.convert("RGBA").tobytes()
    width = int(case.rect.width)
    height = int(case.rect.height)
    x0 = int(case.rect.x)
    y0 = int(case.rect.y)
    stride = 2800 * CHANNELS_PER_PIXEL
    row_bytes = width * CHANNELS_PER_PIXEL
    for row in range(height):
        at = RAW_HEADER_BYTES + (y0 + row) * stride + x0 * CHANNELS_PER_PIXEL
        assert raw[at : at + row_bytes] == crop[row * row_bytes : (row + 1) * row_bytes]


def test_everything_outside_the_crop_is_zero() -> None:
    case = load_batch(BATCH).cases[0]
    raw = case.raw()
    # 裁剪上面那一整片（y < 840）是零。
    assert raw[RAW_HEADER_BYTES : RAW_HEADER_BYTES + 400] == bytes(400)
    # 裁剪左边那一列（x = 2579）也是零 —— 裁剪自己从 2580 起，一直铺到画布右缘。
    x0 = int(case.rect.x)
    y0 = int(case.rect.y)
    assert x0 + int(case.rect.width) == 2800
    at = RAW_HEADER_BYTES + (y0 * 2800 + x0 - 1) * CHANNELS_PER_PIXEL
    assert raw[at : at + CHANNELS_PER_PIXEL] == bytes(4)


def test_a_case_without_pixels_refuses_to_invent_them() -> None:
    case = BenchCase(
        name="x",
        rect=PixelRect(0, 0, 4, 4),
        source="s",
        source_sha256="h",
        canvas=Canvas(width=8, height=8),
    )
    with pytest.raises(ValueError):
        case.raw()


# ---------------------------------------------------------------------------
# 经重放适配器吐出来
# ---------------------------------------------------------------------------


def test_frames_come_out_of_the_replay_adapter_in_manifest_order() -> None:
    batch = load_batch(BATCH)
    names = []
    for case, raw in replay_frames(batch):
        names.append(case.name)
        assert raw == case.raw()
    assert names == [case.name for case in batch.cases]


def test_replay_frames_refuses_something_that_is_not_a_batch() -> None:
    with pytest.raises(TypeError):
        next(replay_frames("费用"))  # type: ignore[arg-type]


def test_the_loader_folds_the_recorded_facts_into_device_facts(
    scratch: Path,
) -> None:
    """**折它的是加载器**（基准集 §2 · T6 补）：`resolution` 取 `canvas`、
    `orientation` 取清单里那一项。设备层收到的已经是 `DeviceFacts`。"""
    _write_batch(
        scratch,
        crops=[{"file": "a.png", "origin": (0, 0), "size": (8, 8)}],
        canvas=(64, 48),
        facts={"orientation": "1"},
    )
    batch = load_batch(scratch)
    assert batch.facts == DeviceFacts(
        resolution=Canvas(width=64, height=48), orientation="1"
    )
    assert [case.name for case, _ in replay_frames(batch)] == ["a"]
    case, raw = next(replay_frames(batch))
    device = ReplayDevice([raw], facts=batch.facts)
    # 原样回放：同一个对象，不是又折了一遍。
    assert device.device_facts() is batch.facts
    assert case.name == "a"


def test_a_facts_block_that_is_not_a_mapping_is_refused(scratch: Path) -> None:
    _write_batch(
        scratch,
        crops=[{"file": "a.png", "origin": (0, 0), "size": (8, 8)}],
        canvas=(64, 48),
        facts="1",
    )
    with pytest.raises(TypeError):
        load_batch(scratch)


def test_a_facts_block_without_an_orientation_is_refused(scratch: Path) -> None:
    """清单的 `canvas` 已经给了分辨率，前提头里只剩 `orientation` —— 它没了
    就是坏清单，**不许就地修补**（基准集 §2 最后那段）。"""
    _write_batch(
        scratch,
        crops=[{"file": "a.png", "origin": (0, 0), "size": (8, 8)}],
        canvas=(64, 48),
        facts={"resolution": [64, 48]},
    )
    with pytest.raises(KeyError):
        load_batch(scratch)
