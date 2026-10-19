"""哈希口径的**唯一定义处**：文本折 LF、二进制按磁盘原样（D-全局-18 + 二进制那一半）。

**为什么单独一个模块**：`scripts/filehash.py`（人用的核哈希工具）与
`scripts/check_source_hashes.py`（第五道闸门）必须说同一件事。各写一份后缀表，
迟早会各说一套 —— 而 9 条错口径正是"工具与纪律各说一套"的产物。

两处都 `import` 这一份，于是**改口径只有一处可改**。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

TEXT_SUFFIXES = frozenset(
    {
        ".json",
        ".md",
        ".py",
        ".txt",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".sh",
        ".ps1",
        ".c",
        ".h",
        ".cpp",
    }
)
"""**文本**那一半。别把二进制后缀加进来 —— 加了就是把 PNG 也折了（那 9 条就是这么来的）。"""


def convention(path: Path) -> str:
    """这个文件该按哪个口径算：**文本 LF 折算 / 二进制磁盘原样**（按后缀，一眼可见）。"""
    return "LF 折算" if path.suffix.lower() in TEXT_SUFFIXES else "磁盘原样"


def sha256_raw(path: Path) -> str:
    """文件的**磁盘原样**哈希。二进制走这一条。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lf(path: Path) -> str:
    """文件的**LF 折算**哈希（CRLF 折成 LF 之后再算）。文本走这一条。"""
    digest = hashlib.sha256()
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            data = tail + chunk
            # 末尾那个 CR 可能是下一块的 CRLF 的前一半：留到下一轮再折。
            cut = len(data) - 1 if data.endswith(b"\r") else len(data)
            digest.update(data[:cut].replace(b"\r\n", b"\n"))
            tail = data[cut:]
    digest.update(tail)
    return digest.hexdigest()


def expected_hash(path: Path) -> str:
    """按 `convention` 算出来的那个哈希 —— **这就是"该记哪个值"的唯一答案**。"""
    return sha256_lf(path) if convention(path) == "LF 折算" else sha256_raw(path)

