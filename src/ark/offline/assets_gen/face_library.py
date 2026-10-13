"""库：**按干员累积**的卡面（规格《资产生成器/卡面与指纹》§0 · §1 · §3）。

# 它是什么

**库跨局**：同一个（干员 × 精英化 × 皮肤）第二次遇到**直接命中**，不再落一份 ——
所以人的负担只跟"新出现的组合"成正比，不跟局数成正比（§3）。

**一局产物不是库**：一局产物只记"这一局队伍那 10 个"（名字 + 卡面引用 + 编队页出处），
它是**建库那一步的凭据**，住运行期状态（`roster_names.STATE_DIR`），**不进版本库**。
库才是资产，进 `assets/roster/`。

# 一块是什么

**编队页那张卡里的立绘**（`art_width` 宽 × 卡高），不是整张卡、也不是脸上那一小块：
比对的模板是**备战席卡里那个 100×100 的窄框**，它在库里的立绘上**滑着找位置**
（规格《视觉层/备战席点名》§判据的形状）。所以库里那一块必须**至少有一整块备战席卡那么大**
（`roster._guard_a_scale_fits` 是同一条闸门）。

体积（实测，run-04 那 10 块 480×214）：RAW 2.94 MB ｜ PNG（`IMWRITE_PNG_COMPRESSION=9`）
**1.56 MB**，一块约 **156 KB**。全量 418 位干员按这个数外推约 65 MB —— 这一版只落
"遇到过的那几块"（§2：不从全量列表采）。换更小的存法（比如只存立绘里那个窄框）是**另一单**，
换之前得先有一条网证明比对结果不变。

# 键与那个洞

**键 = 干员 × 精英化 × 皮肤**（§0）。**精英化这一维现在填不出来**（那一读只给名字），
所以这一版**按卡面像素认**：`id = blake2b(那一块立绘的 RGB 字节, 16)`。
同一套立绘第二次遇到 ⇒ 同一个 id ⇒ **命中**；同一个名字有两套立绘 ⇒ **两条 id**
（规格《视觉层/备战席点名》§多皮肤同时在库：立绘＝身份）。等精英化能读了，
把那一维加进键里就是加一维，落盘格式不用动。

# 落法

    assets/roster/
      names.json          全量闭集（那一读判决用的，住 `roster_names`）
      faces.json          库的索引：一条一笔账（id · 名字 · 出处 · 形状 · 两个哈希）
      faces/<id>.png      每一条一块，**无损 PNG**（比对要逐位可比，不许有损）

通道序照仓里那条惯例：**盘上是 PNG（编码器口径 BGR），内存里是 RGB**
（`cardface.load_formation_page` 同一套）。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

__all__ = [
    "REPO",
    "LIBRARY_DIR",
    "FACES_FILE",
    "FACES_SUBDIR",
    "PNG_COMPRESSION",
    "Face",
    "FaceLibrary",
    "Accumulation",
    "face_id",
    "png_bytes",
    "load_face_library",
    "load_face_images",
    "accumulate_cardfaces",
    "squad_faces",
]


REPO: Final[Path] = Path(__file__).resolve().parents[4]
"""仓库根：`src/ark/offline/assets_gen/` 往上四级（与 `cardface.py` 同一个根）。"""

LIBRARY_DIR: Final[Path] = REPO / "assets" / "roster"
"""库的落点 —— **进版本库**（规格 §0/§1）。`names.json` 也住这儿。"""

FACES_FILE: Final[Path] = LIBRARY_DIR / "faces.json"
"""库的索引。"""

FACES_SUBDIR: Final[str] = "faces"
"""一块卡面在库目录下的子目录名（`faces/<id>.png`）。"""

PNG_COMPRESSION: Final[int] = 9
"""落盘的 PNG 压缩级别。**写死**，否则同一块两次编码未必同字节（与 `cardface` 同一条纪律）。"""


def _guard_art(art: object, *, where: str) -> np.ndarray:
    """一块卡面的形状闸门：`(高, 宽, 3)` 的 uint8 **RGB**。"""
    if not isinstance(art, np.ndarray):
        raise TypeError("%s 要是一块卡面（ndarray），收到 %s" % (where, type(art).__name__))
    if art.dtype != np.uint8 or art.ndim != 3 or art.shape[2] != 3:
        raise TypeError(
            "%s 要是 (高, 宽, 3) 的 uint8 RGB，收到 %s 的 %s"
            % (where, art.dtype, getattr(art, "shape", art)),
        )
    return np.ascontiguousarray(art)


def face_id(art: np.ndarray) -> str:
    """一块卡面 → **id**（`blake2b`，16 字节十六进制）。

    **它就是这个洞的填法**（见模块文档「键与那个洞」）：精英化那一维读不出来，
    所以这一版拿**像素**当身份证 —— 同一套立绘第二次遇到就是同一个 id。
    """
    return hashlib.blake2b(
        _guard_art(art, where="要算 id 的卡面").tobytes(), digest_size=16
    ).hexdigest()


def png_bytes(art: np.ndarray) -> bytes:
    """一块卡面 → PNG 字节（**无损**，压缩级别写死）。"""
    checked = _guard_art(art, where="要编码的卡面")
    ok, buffer = cv2.imencode(
        ".png", checked[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]
    )
    if not ok:
        raise RuntimeError("卡面编不成 PNG")
    return buffer.tobytes()


def _decode(data: bytes, *, where: str) -> np.ndarray:
    """PNG 字节 → 一块 RGB 卡面。**走字节**（与 `cardface` 同一条理由）。"""
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("%s 解不成图" % where)
    return np.ascontiguousarray(decoded[:, :, ::-1])


@dataclass(frozen=True)
class Face:
    """库里**一条**：一块卡面 + 它的出处。

    `id` 是内容寻址的键（`face_id`）；`file` 是相对库目录的路径。
    `elite` / `skin` 现在是 `None` —— 那是规格 §0 那个键缺的两维（**洞，不是零**）。
    """

    id: str
    name: str
    elite: int | None
    skin: str | None
    file: str
    sha256: str
    pixels_sha256: str
    bytes: int
    shape: tuple[int, int, int]
    source: Mapping[str, object]

    def to_record(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "elite": self.elite,
            "skin": self.skin,
            "file": self.file,
            "sha256": self.sha256,
            "pixels_sha256": self.pixels_sha256,
            "bytes": self.bytes,
            "shape": list(self.shape),
            "source": dict(self.source),
        }


@dataclass(frozen=True)
class FaceLibrary:
    """库：一串 `Face`。**次序就是加进来的次序**（后面进来的排后面）。"""

    faces: tuple[Face, ...] = ()

    def by_name(self, name: str) -> tuple[Face, ...]:
        """这个名字在库里的**每一条** —— 同一个名字可以有两套立绘。"""
        return tuple(item for item in self.faces if item.name == name)

    def __contains__(self, item: object) -> bool:
        return any(face.id == item for face in self.faces)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.faces)


@dataclass(frozen=True)
class Accumulation:
    """一次累积的账：新库 · **新落下的** · **直接命中的**。"""

    library: FaceLibrary
    added: tuple[Face, ...]
    hit: tuple[Face, ...]


def load_face_library(path: str | Path = FACES_FILE) -> FaceLibrary:
    """库的索引 → `FaceLibrary`。**文件不在**（还没建库）⇒ 空库，不报错。"""
    path = Path(path)
    if not path.is_file():
        return FaceLibrary()
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, Mapping) or record.get("kind") != "roster_faces":
        raise ValueError(
            "%s 不是一份库的索引（kind 要是 roster_faces）" % (path,)
        )
    faces: list[Face] = []
    for position, item in enumerate(record.get("faces") or ()):
        if not isinstance(item, Mapping):
            raise TypeError("%s 的第 %d 条要是一个对象" % (path, position))
        shape = tuple(item.get("shape") or ())
        if len(shape) != 3:
            raise ValueError("%s 的第 %d 条的 shape 要是三个数" % (path, position))
        faces.append(
            Face(
                id=str(item["id"]),
                name=str(item["name"]),
                elite=item.get("elite"),
                skin=item.get("skin"),
                file=str(item["file"]),
                sha256=str(item["sha256"]),
                pixels_sha256=str(item["pixels_sha256"]),
                bytes=int(item["bytes"]),
                shape=(int(shape[0]), int(shape[1]), int(shape[2])),
                source=dict(item.get("source") or {}),
            )
        )
    return FaceLibrary(faces=tuple(faces))


def load_face_images(
    library: FaceLibrary, *, directory: str | Path = LIBRARY_DIR
) -> tuple[tuple[str, np.ndarray], ...]:
    """库 → **视觉层要的那份 `faces`**（`(名字, 立绘块)`，按库里的次序）。

    次序**不许重排**：`read_squad_identities` 拿它配对，换了次序就是换了配对。
    盘上的 PNG 是逐位无损的，所以这一趟出来的像素与当初切下来的**逐字节相同**。
    """
    directory = Path(directory)
    out: list[tuple[str, np.ndarray]] = []
    for face in library.faces:
        path = directory / face.file
        if not path.is_file():
            raise FileNotFoundError(
                "库的索引里有 %s（%s），但 %s 不在 —— 索引与石头对不上"
                % (face.id, face.name, path),
            )
        art = _decode(path.read_bytes(), where=str(path))
        if (art.shape[0], art.shape[1], art.shape[2]) != tuple(face.shape):
            raise ValueError(
                "%s 的形状是 %s，索引里写的是 %s" % (path, art.shape, face.shape)
            )
        out.append((face.name, art))
    return tuple(out)


def _write_library(library: FaceLibrary, *, path: Path) -> None:
    record = {
        "version": 1,
        "kind": "roster_faces",
        "note": "库按干员累积（规格 卡面与指纹 §3）：同一个组合第二次遇到直接命中，不再落一份。",
        "key": "干员 × 精英化 × 皮肤（规格 §0）",
        "key_gap": "【洞】精英化这一维现在填不出来（那一读只给名字）⇒ id 按**卡面像素**算（内容寻址）：同一套立绘第二次遇到是同一个 id；同一个名字有两套立绘就是两条 id。等精英化能读了再把它加进键里。",
        "block": "一块 = 编队页那张卡里的**立绘**（art_width × 卡高），不是整张卡、也不是脸上那一小块；比对的模板是备战席卡里那个 100×100 的窄框（规格 视觉层/备战席点名 §判据的形状）。",
        "channel_order": "盘上是 PNG（编码器口径 BGR），内存里是 RGB；无损，逐位可比。",
        "count": len(library.faces),
        "faces": [item.to_record() for item in library.faces],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
        newline=chr(10),
    )


def accumulate_cardfaces(
    pairs: Sequence[tuple[str, np.ndarray]],
    *,
    source: Mapping[str, object],
    path: str | Path = FACES_FILE,
    directory: str | Path = LIBRARY_DIR,
) -> Accumulation:
    """一局的那几块卡面 → **并进库** → 落盘；回一本账（新落下的 · 直接命中的）。

    **按像素去重**（见模块文档「键与那个洞」）：id 已经在库里 ⇒ **命中**，不重写文件、
    不加行；不在 ⇒ 落 `faces/<id>.png` 并加一行。**同一个名字的两套立绘都会留下**。

    写盘是**幂等**的：同一批再跑一次，文件字节与索引字节都不变（PNG 压缩级别写死、
    索引按库的次序序列化）。
    """
    path = Path(path)
    directory = Path(directory)
    library = load_face_library(path)
    known = {item.id: item for item in library.faces}
    added: list[Face] = []
    hit: list[Face] = []
    for name, art in pairs:
        if not isinstance(name, str) or not name:
            raise TypeError("卡面的名字要是非空字符串，收到 %r" % (name,))
        checked = _guard_art(art, where="库里的卡面「%s」" % name)
        identifier = face_id(checked)
        if identifier in known:
            hit.append(known[identifier])
            continue
        payload = png_bytes(checked)
        relative = "%s/%s.png" % (FACES_SUBDIR, identifier)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or target.read_bytes() != payload:
            target.write_bytes(payload)
        face = Face(
            id=identifier,
            name=name,
            elite=None,
            skin=None,
            file=relative,
            sha256=hashlib.sha256(payload).hexdigest(),
            pixels_sha256=hashlib.sha256(checked.tobytes()).hexdigest(),
            bytes=len(payload),
            shape=(int(checked.shape[0]), int(checked.shape[1]), int(checked.shape[2])),
            source=dict(source),
        )
        added.append(face)
        known[identifier] = face
    library = FaceLibrary(faces=tuple(list(library.faces) + added))
    _write_library(library, path=path)
    return Accumulation(library=library, added=tuple(added), hit=tuple(hit))


def squad_faces(
    path: str | Path = FACES_FILE, *, directory: str | Path = LIBRARY_DIR
) -> tuple[tuple[str, bytes, int, int], ...]:
    """库 → **端口那一份**（`(名字, RGB 原始字节, 高, 宽)`，按库里的次序）。

    给字节不给数组：端口那一层不 import numpy，而 `np.frombuffer` 出来就是**零拷贝**的一张图。
    **空库报错**，不静默给一个空 tuple —— 空库不是「库里没有这个人」，是**没建库**。
    """
    library = load_face_library(path)
    if not library.faces:
        raise ValueError(
            "库是空的（%s 不在，或者里面一条都没有）：空库不是「库里没有这个人」，是没建库"
            % (Path(path),),
        )
    images = load_face_images(library, directory=directory)
    return tuple(
        (name, art.tobytes(), int(art.shape[0]), int(art.shape[1]))
        for name, art in images
    )

