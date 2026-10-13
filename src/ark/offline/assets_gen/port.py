"""资产端口的第一个实现：**备战席点名要的那份库**（`assets/roster/`）。

补法规则照 `AssetPort` 的文档字符串：**每个方法由它的第一个用户层在用到时补；补的时候
只加方法、不改端口本身。** 这一轮补的是 `squad_faces`，它的用户层是**视觉层**
（`roster.read_squad_identities` 的 `faces`）—— 一侧是契约层那一处声明，另一侧就是这里。

**这个类原来叫 `EmptyAssetPort`、一个公开方法都没有**：那时端口的方法集是空的，
"空"就是全部真相。现在端口有两个方法、这里接了一个，所以名字改准 ——
**它不再是"空实现"，是"接了库的那一个"**。`projection` 仍然没接，欠账照旧登记在
`runtime/wiring.py` 的 `PORT_GAPS` 里；**不许**在这里摆一个会抛异常的方法，
一个在但一调就炸的方法会让「接上了」变成假的（`port_completeness` 判的是**方法级**）。

## 它为什么住这里

资产是 `offline/assets_gen/` 的产物，而 `engine` 永远不 import `offline`
（代码结构 规则一）。值必须经契约层到达、由 `runtime/wiring.py` 接上。

## 库只从盘上读一次

`squad_faces` 每次都重新解那 10 张 PNG 没必要：库是**资产**，运行期不变。
所以第一个调用读完就记住，同一个实例后面直接给（换库 = 换一个实例 ——
那正是"换一套参数 = 换一个适配器"那条纪律）。
"""

from __future__ import annotations

from pathlib import Path

from ark.offline.assets_gen.face_library import FACES_FILE, LIBRARY_DIR, squad_faces

__all__ = ["RosterAssetPort"]


class RosterAssetPort:
    """资产端口的实现：从 `assets/roster/` 读那份库，经 `squad_faces` 交给视觉层。"""

    def __init__(
        self, path: str | Path = FACES_FILE, *, directory: str | Path = LIBRARY_DIR
    ) -> None:
        self._path = Path(path)
        self._directory = Path(directory)
        self._faces: tuple[tuple[str, bytes, int, int], ...] | None = None

    def squad_faces(self) -> tuple[tuple[str, bytes, int, int], ...]:
        """库（`(名字, RGB 原始字节, 高, 宽)`）。**库没建好就抛**，不静默给空。"""
        if self._faces is None:
            self._faces = squad_faces(self._path, directory=self._directory)
        return self._faces

