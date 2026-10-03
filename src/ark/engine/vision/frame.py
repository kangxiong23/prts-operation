"""视觉层入口：设备层的**原始帧** → 画布上的像素。

字节布局（设备层 §4 · D-全局-19）：

    头 16 字节 = 4 个小端 uint32（宽 · 高 · 像素格式 · 色彩空间，实测 2800 1260 1 1）
    后接 宽 × 高 × 4 字节，**通道序 RGBA**

一帧定长 16 + 2800 × 1260 × 4 = **14,112,016** 字节。

**画布就是捕获分辨率本身**（契约层 §2）：这里不缩放、不归一化、不裁边 ——
解出来的像素与设备上那一片一一对应。所以 `Frame.pixels` 的 `(h, w)` 就是
前提头里的画布，不吻合即拒绝执行（[时序层] §3）。

**像素格式只认实测过的那一种。** 文档只写了「实测 1 1」，没有写「别的值合法」。
把没见过的格式按 RGBA 解出来就是**假事实**（「让不知道没法伪装成知道」），
所以这里对像素格式与色彩空间**报错拒绝**，而不是猜着解。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ark.contracts import Canvas, PixelRect

__all__ = [
    "Frame",
    "frame_from_raw",
    "RAW_HEADER_BYTES",
    "CHANNELS_PER_PIXEL",
    "PIXEL_FORMAT_RGBA8888",
    "COLOR_SPACE_SRGB",
    "CANVAS_2800x1260",
]


RAW_HEADER_BYTES = 16
"""头：4 个小端 uint32（宽 · 高 · 像素格式 · 色彩空间）。"""

CHANNELS_PER_PIXEL = 4
"""每像素 4 字节，通道序 RGBA。"""

PIXEL_FORMAT_RGBA8888 = 1
"""实测到的像素格式字段值。只认这一个。"""

COLOR_SPACE_SRGB = 1
"""实测到的色彩空间字段值。只认这一个。"""

CANVAS_2800x1260 = Canvas(width=2800, height=1260)
"""这台设备的捕获分辨率（vivo V2338A 横屏）。"""


@dataclass(frozen=True)
class Frame:
    """画布上的一帧：像素 + 它自己的尺寸。

    `pixels` 是 `(高, 宽, 3)` 的 uint8，**RGB**（不是 RGBA —— 识别链
    读的是颜色，透明度在这一层没有任何用途，留着它只会让下游多一个
    「这个通道要不要参与统计」的问题）。

    画布与像素的形状必须对得上：对不上就是解帧解错了，不该走到识别链里。
    """

    canvas: Canvas
    pixels: np.ndarray

    # 像素是 ndarray（不可哈希）：显式声明，比留一个到运行期才炸的生成哈希好。
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.canvas, Canvas):
            raise TypeError(f"帧的画布要是 Canvas，收到 {type(self.canvas).__name__}")
        pixels = self.pixels
        if not isinstance(pixels, np.ndarray):
            raise TypeError(f"帧的像素要是 ndarray，收到 {type(pixels).__name__}")
        if pixels.dtype != np.uint8:
            raise TypeError(f"帧的像素要是 uint8，收到 {pixels.dtype}")
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise TypeError(
                f"帧的像素要是 (高, 宽, 3) 的 RGB，收到 {pixels.shape}："
                "RGBA 在解帧时就该丢掉 alpha"
            )
        if pixels.shape[0] != self.canvas.height or pixels.shape[1] != self.canvas.width:
            raise ValueError(
                f"像素形状 {pixels.shape[:2]} 与画布 "
                f"{self.canvas.width}×{self.canvas.height} 对不上"
            )

    def crop(self, rect: PixelRect) -> np.ndarray:
        """按像素矩形裁一块 ROI，返回 `(高, 宽, 3)` 的 uint8（**是视图，不是拷贝**）。

        矩形是 `x / y / 宽 / 高`（契约层 §2），所以裁出来正好是
        `宽 × 高` 个像素。

        **贴到画布边缘时会裁掉越出的那一列/行**（numpy 的切片语义），不报错：
        UI 常量是「够用的窗口」，而文档自己的 `cost_panel` 右缘就写到了 2800
        （含两端口径下越出画布 1 px）。这里选择**如实返回画布上真实存在的那些
        像素**，而不是替调用方编一个不存在的像素。
        """
        if not isinstance(rect, PixelRect):
            raise TypeError(f"要裁的是像素矩形，收到 {type(rect).__name__}")
        x0 = int(rect.x)
        y0 = int(rect.y)
        x1 = x0 + int(rect.width)
        y1 = y0 + int(rect.height)
        if x1 <= 0 or y1 <= 0 or x0 >= self.canvas.width or y0 >= self.canvas.height:
            raise ValueError(
                f"矩形 {rect} 整个落在画布 {self.canvas.width}×{self.canvas.height} 之外"
            )
        return self.pixels[max(y0, 0):y1, max(x0, 0):x1]


def frame_from_raw(raw: bytes) -> Frame:
    """把设备层取回的一整帧原始字节解成 `Frame`。

    只做三件事：读头、验长度、把 RGBA 丢成 RGB。**一个像素都不重采样。**
    """
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise TypeError(f"原始帧要是字节，收到 {type(raw).__name__}")
    if len(raw) < RAW_HEADER_BYTES:
        raise ValueError(
            f"原始帧至少要有 {RAW_HEADER_BYTES} 字节的头，收到 {len(raw)} 字节"
        )
    width, height, pixel_format, color_space = struct.unpack_from("<4I", raw, 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"原始帧的宽高必须是正的，头里写的是 {width}×{height}")
    if pixel_format != PIXEL_FORMAT_RGBA8888:
        raise ValueError(
            f"原始帧的像素格式字段是 {pixel_format}，只认实测过的 "
            f"{PIXEL_FORMAT_RGBA8888}（RGBA8888）：没见过的格式按 RGBA 解就是假事实"
        )
    if color_space != COLOR_SPACE_SRGB:
        raise ValueError(
            f"原始帧的色彩空间字段是 {color_space}，只认实测过的 "
            f"{COLOR_SPACE_SRGB}（sRGB）"
        )
    expected = RAW_HEADER_BYTES + width * height * CHANNELS_PER_PIXEL
    if len(raw) != expected:
        raise ValueError(
            f"原始帧长度对不上：{width}×{height}×{CHANNELS_PER_PIXEL} + "
            f"{RAW_HEADER_BYTES} = {expected} 字节，收到 {len(raw)} 字节"
        )
    rgba = np.frombuffer(
        raw, dtype=np.uint8, count=width * height * CHANNELS_PER_PIXEL,
        offset=RAW_HEADER_BYTES,
    ).reshape(height, width, CHANNELS_PER_PIXEL)
    return Frame(canvas=Canvas(width=width, height=height), pixels=np.array(rgba[:, :, :3]))
