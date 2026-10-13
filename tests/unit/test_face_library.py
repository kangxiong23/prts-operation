"""库（`face_library`）与那条端口缝的单元测试。

**它不碰 `screenshots/`**：卡面自己合成（每块一个常数颜色），库落在工作区里的一小块地上。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ark.offline.assets_gen import face_library as FL
from ark.offline.assets_gen.port import RosterAssetPort

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def scratch(request: pytest.FixtureRequest) -> Path:
    """**工作区里**的一小块地（平台的临时目录在本机沙箱下进不去）。"""
    path = REPO / "data" / "probe" / "pytest_scratch" / request.node.name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _art(slot: int) -> np.ndarray:
    """一块**合成**的卡面：整块一个常数颜色，槽不同颜色就不同。"""
    art = np.zeros((480, 214, 3), np.uint8)
    art[:, :, 0] = 10 + 20 * slot
    art[:, :, 1] = 30 + 20 * slot
    art[:, :, 2] = 50 + 20 * slot
    return art


def _land(scratch: Path, pairs, source=None):
    return FL.accumulate_cardfaces(
        list(pairs), source=source or {"page": "合成"},
        path=scratch / "faces.json", directory=scratch,
    )


# --- 一块与它的 id ---------------------------------------------------------


def test_png_round_trip_is_lossless() -> None:
    art = _art(3)
    assert np.array_equal(FL._decode(FL.png_bytes(art), where="合成"), art)


def test_face_id_follows_the_pixels() -> None:
    assert FL.face_id(_art(1)) == FL.face_id(_art(1))
    assert FL.face_id(_art(1)) != FL.face_id(_art(2))
    assert len(FL.face_id(_art(0))) == 32


def test_a_bad_block_is_refused() -> None:
    with pytest.raises(TypeError):
        FL.face_id(np.zeros((10, 10, 4), np.uint8))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        FL.face_id("不是数组")  # type: ignore[arg-type]


# --- 累积 -------------------------------------------------------------------


def test_accumulate_lands_then_hits(scratch: Path) -> None:
    pairs = [("甲", _art(0)), ("乙", _art(1))]
    first = _land(scratch, pairs)
    assert (len(first.added), len(first.hit)) == (2, 0)
    second = _land(scratch, pairs)
    assert (len(second.added), len(second.hit)) == (0, 2)
    assert second.library.ids == first.library.ids


def test_accumulate_is_idempotent_byte_for_byte(scratch: Path) -> None:
    pairs = [("甲", _art(0)), ("乙", _art(1))]
    _land(scratch, pairs)
    index = (scratch / "faces.json").read_bytes()
    stones = {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
              for item in (scratch / "faces").iterdir()}
    _land(scratch, pairs)
    assert (scratch / "faces.json").read_bytes() == index
    assert {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
            for item in (scratch / "faces").iterdir()} == stones


def test_one_name_may_keep_two_faces(scratch: Path) -> None:
    _land(scratch, [("甲", _art(0))])
    result = _land(scratch, [("甲", _art(5))])
    assert len(result.added) == 1
    assert len(result.library.by_name("甲")) == 2
    assert len(result.library.faces) == 2


def test_the_elite_dimension_is_an_open_hole_not_a_zero(scratch: Path) -> None:
    face = _land(scratch, [("甲", _art(0))]).added[0]
    assert face.elite is None and face.skin is None
    record = json.loads((scratch / "faces.json").read_text(encoding="utf-8"))
    assert record["count"] == 1
    assert record["key_gap"]
    assert record["faces"][0]["shape"] == [480, 214, 3]


# --- 读回来 -----------------------------------------------------------------


def test_load_face_images_keeps_the_order_and_the_pixels(scratch: Path) -> None:
    _land(scratch, [("甲", _art(0)), ("乙", _art(1)), ("丙", _art(2))])
    library = FL.load_face_library(scratch / "faces.json")
    images = FL.load_face_images(library, directory=scratch)
    assert [name for name, _ in images] == ["甲", "乙", "丙"]
    for slot, (_, art) in enumerate(images):
        assert np.array_equal(art, _art(slot))


def test_a_missing_library_file_is_an_empty_library(scratch: Path) -> None:
    assert FL.load_face_library(scratch / "faces.json").faces == ()


def test_load_face_images_refuses_a_missing_stone(scratch: Path) -> None:
    _land(scratch, [("甲", _art(0))])
    library = FL.load_face_library(scratch / "faces.json")
    next((scratch / "faces").iterdir()).unlink()
    with pytest.raises(FileNotFoundError):
        FL.load_face_images(library, directory=scratch)


def test_squad_faces_gives_bytes_and_the_order(scratch: Path) -> None:
    _land(scratch, [("甲", _art(0)), ("乙", _art(1))])
    payload = FL.squad_faces(scratch / "faces.json", directory=scratch)
    assert [item[0] for item in payload] == ["甲", "乙"]
    for slot, (_, raw, height, width) in enumerate(payload):
        assert (height, width) == (480, 214)
        assert len(raw) == 480 * 214 * 3
        assert raw == _art(slot).tobytes()


def test_squad_faces_refuses_an_empty_library(scratch: Path) -> None:
    with pytest.raises(ValueError):
        FL.squad_faces(scratch / "faces.json", directory=scratch)


# --- 端口那条缝 -------------------------------------------------------------


def test_the_port_serves_the_library(scratch: Path) -> None:
    _land(scratch, [("甲", _art(0)), ("乙", _art(1))])
    port = RosterAssetPort(scratch / "faces.json", directory=scratch)
    payload = port.squad_faces()
    assert [item[0] for item in payload] == ["甲", "乙"]
    assert payload[0][1] == _art(0).tobytes()


def test_the_port_remembers_what_it_read(scratch: Path) -> None:
    _land(scratch, [("甲", _art(0))])
    port = RosterAssetPort(scratch / "faces.json", directory=scratch)
    first = port.squad_faces()
    (scratch / "faces.json").write_text("{}", encoding="utf-8")
    assert port.squad_faces() is first, "库是资产，运行期不变：读一次就记住"


def test_the_port_refuses_an_unbuilt_library(scratch: Path) -> None:
    port = RosterAssetPort(scratch / "faces.json", directory=scratch)
    with pytest.raises(ValueError):
        port.squad_faces()


def test_the_port_only_exposes_squad_faces() -> None:
    """补法规则那道闸门的本地提醒：端口声明了几个方法，这里就只许有那几个。"""
    from ark.contracts import AssetPort

    declared = {
        name for name, value in vars(AssetPort).items()
        if not name.startswith("_") and callable(value)
    }
    present = {name for name in dir(RosterAssetPort) if not name.startswith("_")}
    assert present <= declared
    assert "squad_faces" in declared
