"""基准集：一个目录 = 一批帧 + 出处（采集与验收/基准集 §2）。

`load_batch(root)` 读 `index.json`，把每条裁剪**贴回画布**、编成设备层 §4
那种原始帧字节（头 16 字节 = 4 个小端 uint32 + 宽 × 高 × 4 RGBA）。

**为什么贴回画布、不直接把裁剪当帧**（基准集 §3）：理由只有一条 —— **ROI 的取法
仍然是真的在被测**。把 ROI 预先替识别链做好，测的就不是同一条路了。

**PNG 解码住这一层**（基准集 §3）：`engine` 不装 pillow 也得能跑（pillow 挂在
`pyproject.toml` 的 `offline` 那组额外依赖上）。所以
`engine/device/replay.py` 收到的**已经是原始帧字节**，它自己不解图。

**期望值不住这里**（基准集 §1 · D-全局-17）：这一层只读清单与裁剪，**一个真值
文件都不碰** —— 真值住 `tests/spec/`。`source` / `source_sha256` 是**出处**，
只记不验：原帧住在 gitignored 的 `data/` 里，本来就可能已经不在本机了。
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ark.contracts import Canvas, DeviceFacts, PixelRect
from ark.engine.vision.frame import (
    CHANNELS_PER_PIXEL,
    COLOR_SPACE_SRGB,
    PIXEL_FORMAT_RGBA8888,
    RAW_HEADER_BYTES,
)

__all__ = ["load_batch", "BenchBatch", "BenchCase", "MANIFEST_NAME"]


MANIFEST_NAME = "index.json"
"""清单的文件名：**一个目录 = 一批**，清单就在那一批的根上（基准集 §2）。"""

_CROP_KEYS = ("file", "origin", "size", "source", "source_sha256")
"""`crops` 每条的必填字段（基准集 §2 的第二张表）。"""

_BATCH_KEYS = ("version", "canvas", "device", "crops")
"""`index.json` 的必填字段（基准集 §2 的第一张表）。`facts` 不在里面 ——
它是可选的，**没有就是没有**。"""


@dataclass(frozen=True)
class BenchCase:
    """一条：**帧名 · 裁剪在画布上的矩形 · 出处 · 它属于哪张画布**（基准集 §2）。

    `raw()` 给的是**原始帧**（贴回画布之后的整张），不是裁剪本身 —— 裁剪只是
    它的一部分，其余部分是零（那一片没人拍过）。
    """

    name: str
    rect: PixelRect
    source: str
    source_sha256: str
    canvas: Canvas

    _pixels: bytes | None = field(default=None, repr=False, compare=False)
    """裁剪自己的 RGBA 字节（原生像素，一个都不缩）。

    它是私有的：`BenchCase` 对外那五样就是接口表里那五样，`raw()` 是唯一的
    公开方法。留 `None` 默认值是为了让"只拿着清单里那五样"也能构造出一条
    （那种条没有像素，`raw()` 会明说）。
    """

    def raw(self) -> bytes:
        """这一条的**原始帧字节**：裁剪贴回画布、其余像素为零（基准集 §3）。

        字节布局照设备层 §4：头 16 字节 + 宽 × 高 × 4，通道序 RGBA。**贴回去是
        逐像素的**：裁剪里 `(i, j)` 那个像素落在画布的 `(x + i, y + j)`，
        一个像素都不重采样、不缩放。
        """
        if self._pixels is None:
            raise ValueError(
                f"{self.name} 没有裁剪的像素：这一条不是 load_batch 读出来的，"
                "raw() 编不出它没读过的像素"
            )
        canvas = self.canvas
        width = int(self.rect.width)
        height = int(self.rect.height)
        x0 = int(self.rect.x)
        y0 = int(self.rect.y)
        frame = bytearray(
            RAW_HEADER_BYTES + canvas.width * canvas.height * CHANNELS_PER_PIXEL
        )
        struct.pack_into(
            "<4I",
            frame,
            0,
            canvas.width,
            canvas.height,
            PIXEL_FORMAT_RGBA8888,
            COLOR_SPACE_SRGB,
        )
        row_bytes = width * CHANNELS_PER_PIXEL
        stride = canvas.width * CHANNELS_PER_PIXEL
        for row in range(height):
            start = row * row_bytes
            at = RAW_HEADER_BYTES + (y0 + row) * stride + x0 * CHANNELS_PER_PIXEL
            frame[at : at + row_bytes] = self._pixels[start : start + row_bytes]
        return bytes(frame)


@dataclass(frozen=True)
class BenchBatch:
    """一批：**清单读出来的那些字段 + 每条**（基准集 §2）。

    `facts` 是**折好的 `DeviceFacts`**：`resolution` 取 `canvas`、
    `orientation` 取清单里那一项；**没有 `facts` 块就是 `None`** —— 加载器
    **不补默认值**（基准集 §2 那句"没有就是没有"）。**折它的是这一层，不是设备
    层**（基准集 §2 · T6 补）：让重放适配器去认"清单里那个键叫 orientation"，
    等于把离线那套文件格式塞进引擎（代码结构 规则三）。
    """

    root: Path
    version: int
    canvas: Canvas
    device: str
    facts: DeviceFacts | None
    cases: tuple[BenchCase, ...]


def load_batch(root: str | Path) -> BenchBatch:
    """读一批基准集：清单 → `BenchBatch`（基准集 §2）。

    **它只碰清单与裁剪**：`source` 那条路径指向的原帧可能早就没了（住在
    gitignored 的 `data/` 里），所以出处**只记不验**。真值文件更不碰 ——
    那住 `tests/spec/`（基准集 §1 · D-全局-17）。
    """
    batch_dir = Path(root)
    if not batch_dir.is_dir():
        raise FileNotFoundError(f"基准集是一个目录，{batch_dir} 不是")
    manifest_path = batch_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"{batch_dir} 里没有 {MANIFEST_NAME}：一批至少要有清单与裁剪（基准集 §2）"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError(
            f"{manifest_path} 的顶层要是一个对象，收到 {type(manifest).__name__}"
        )
    for key in _BATCH_KEYS:
        if key not in manifest:
            raise KeyError(f"{manifest_path} 缺必填字段 {key!r}（基准集 §2 的字段表）")

    version = _as_int(manifest["version"], "version")
    if version < 1:
        raise ValueError(f"version 从 1 起（改了内容就 +1），收到 {version}")
    width, height = _as_pair(manifest["canvas"], "canvas")
    canvas = Canvas(width=width, height=height)
    device = manifest["device"]
    if not isinstance(device, str) or not device:
        raise TypeError(f"device 是机型，要是非空字符串，收到 {device!r}")
    crops = manifest["crops"]
    if isinstance(crops, (str, bytes)) or not isinstance(crops, Sequence):
        raise TypeError(f"crops 要是每条的清单，收到 {type(crops).__name__}")

    cases: list[BenchCase] = []
    seen: set[str] = set()
    for position, entry in enumerate(crops):
        case = _read_case(batch_dir, entry, canvas, position)
        if case.name in seen:
            raise ValueError(
                f"帧名 {case.name!r} 在批内出现了两次：帧名批内唯一（基准集 §2）"
            )
        seen.add(case.name)
        cases.append(case)

    return BenchBatch(
        root=batch_dir,
        version=version,
        canvas=canvas,
        device=device,
        facts=_facts_from(manifest.get("facts"), canvas),
        cases=tuple(cases),
    )


def _facts_from(recorded: object, canvas: Canvas) -> DeviceFacts | None:
    """清单里那个映射 → `DeviceFacts`；**没有 `facts` 块就是 `None`**。

    **折它的是加载器**（基准集 §2 · T6 补）：清单的 `canvas` 就是录制时的分辨率
    （设备层 §4 那两个口径不许混），所以 `facts` 里只剩 `orientation` ——
    `resolution` 取 `canvas`、`orientation` 取 `facts.orientation`。

    **两处都不许补默认值**：没有 `facts` 块就是 `None`（不是"方向默认 0"）；
    映射里没有 `orientation` 就是坏清单（不是"方向不明"）—— 报错，不许就地
    修补（基准集 §2 最后那段）。
    """
    if recorded is None:
        return None
    if not isinstance(recorded, Mapping):
        raise TypeError(
            f"facts 是录制的前提头（一个映射）或者干脆没有，收到 "
            f"{type(recorded).__name__}：**没有就是没有**，加载器不许补一个默认值"
        )
    if "orientation" not in recorded:
        raise KeyError(
            "facts 里没有 orientation：清单的 canvas 已经给了分辨率，前提头里只剩"
            f"这一项（基准集 §2），记下来的键有 {sorted(recorded)}"
        )
    orientation = recorded["orientation"]
    if isinstance(orientation, bool) or not isinstance(orientation, (str, int)):
        raise TypeError(
            "facts.orientation 是设备报的那个整数（字面值见设备层 §5），收到 "
            f"{orientation!r}"
        )
    return DeviceFacts(resolution=canvas, orientation=str(orientation))


def _read_case(
    batch_dir: Path, entry: object, canvas: Canvas, position: int
) -> BenchCase:
    where = f"crops[{position}]"
    if not isinstance(entry, Mapping):
        raise TypeError(f"{where} 要是每条的清单，收到 {type(entry).__name__}")
    for key in _CROP_KEYS:
        if key not in entry:
            raise KeyError(f"{where} 缺必填字段 {key!r}（基准集 §2 的 crops 字段表）")

    file_name = entry["file"]
    if not isinstance(file_name, str) or not file_name:
        raise TypeError(f"{where}.file 是裁剪的 PNG，要是非空字符串，收到 {file_name!r}")
    name = Path(file_name).stem
    if not name:
        raise ValueError(f"{where}.file 去扩展名之后没有名字：{file_name!r}")

    x, y = _as_pair(entry["origin"], f"{where}.origin")
    width, height = _as_pair(entry["size"], f"{where}.size")
    rect = PixelRect(x, y, width, height)
    if x < 0 or y < 0 or x + width > canvas.width or y + height > canvas.height:
        raise ValueError(
            f"{where} 的裁剪 {rect} 落在画布 {canvas.width}×{canvas.height} 之外："
            "原点 + 尺寸必须含在画布内，右缘 ≤ 宽 − 1、下缘 ≤ 高 − 1（基准集 §2）"
        )

    source = entry["source"]
    if not isinstance(source, str) or not source:
        raise TypeError(f"{where}.source 是出处，要是非空字符串，收到 {source!r}")
    source_sha256 = entry["source_sha256"]
    if not isinstance(source_sha256, str) or not source_sha256:
        raise TypeError(
            f"{where}.source_sha256 是出处的指纹，要是非空字符串，收到 {source_sha256!r}"
        )

    png = batch_dir / file_name
    with Image.open(png) as image:
        actual = image.size
        if actual != (width, height):
            raise ValueError(
                f"{png} 的实际尺寸是 {actual}，清单写的是 {(width, height)}："
                "裁剪是原生像素、一个都不缩（基准集 §2）"
            )
        pixels = image.convert("RGBA").tobytes()
    return BenchCase(
        name=name,
        rect=rect,
        source=source,
        source_sha256=source_sha256,
        canvas=canvas,
        _pixels=pixels,
    )


def _as_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} 要是整数，收到 {type(value).__name__}")
    return value


def _as_pair(value: object, what: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{what} 要是一对整数，收到 {type(value).__name__}")
    if len(value) != 2:
        raise ValueError(f"{what} 要正好两个数，收到 {len(value)} 个")
    return _as_int(value[0], what), _as_int(value[1], what)
