"""核哈希：算一个文件的 sha256。**口径按文件类型分两半**（D-全局-18 + 二进制那一半）：

| 文件 | 口径 | 为什么 |
| --- | --- | --- |
| **文本**（`.py` / `.json` / `.md` / …） | **LF 折算** | `autocrlf=true` 让磁盘字节与机器相关：同一份内容在两种行尾下 sha256 不同，直接核会误报「内容对不上」 |
| **二进制**（`.png` / `.jpg` / `.raw` / …） | **磁盘原样** | 换行在二进制里没有意义；折一次等于**把整个文件重写一遍再算** —— 那是另一个文件的哈希 |

**这一行是 9 条错口径的根**：从前这个脚本对**所有**文件都打「LF 折算（口径）」，
于是搬夹具的人照着"口径"那一栏抄，PNG 也被折了 LF。现在**类型决定口径**，
而且打印时把"哪一个才是口径"与"它为什么是口径"一起打出来。

**不要用 `Get-FileHash` 核** —— 它只给磁盘原样，文本那一半必然对不上。

用法：

    python scripts/filehash.py <路径> [更多路径…]
    python scripts/filehash.py <路径> --expect <sha256>

第一种打印两个口径与行尾构成，并**标出哪一个才是这个文件的**；第二种核对单个文件，
不一致就退出 1 —— 任务消息里的哈希就是这么核的。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash_conventions import (  # noqa: E402
    TEXT_SUFFIXES,  # noqa: F401
    convention,
    sha256_lf,
    sha256_raw,
)


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _describe(path: Path) -> tuple[str, str, str, str]:
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = raw.count(b"\r\n")
    bare = raw.count(b"\n") - crlf
    endings = "CRLF" if bare == 0 and crlf else ("LF" if crlf == 0 else "混合")
    return _sha256(lf), _sha256(raw), f"{endings}（CRLF {crlf} / 裸 LF {bare}）", convention(path)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2

    expect: str | None = None
    if "--expect" in argv:
        index = argv.index("--expect")
        if index + 1 >= len(argv):
            print("--expect 后面要跟一个 sha256", file=sys.stderr)
            return 2
        expect = argv[index + 1].strip().lower()
        argv = argv[:index] + argv[index + 2 :]

    paths = [Path(name) for name in argv]
    if expect is not None and len(paths) != 1:
        print("--expect 只能核一个文件", file=sys.stderr)
        return 2

    failed = False
    for path in paths:
        if not path.is_file():
            print(f"没有这个文件：{path}", file=sys.stderr)
            return 2
        canonical, raw_hash, endings, kind = _describe(path)
        right = canonical if kind == "LF 折算" else raw_hash
        other = raw_hash if kind == "LF 折算" else canonical
        if expect is not None:
            ok = right == expect
            failed = failed or not ok
            print(f"{'✓' if ok else '✗'} {path}    口径 {kind}")
            print(f"    算出来    {right}")
            if not ok:
                print(f"    期望      {expect}")
                print(f"    另一个口径 {other}   行尾 {endings}")
                print("    （只有标着「口径」的那一个算数；对不上才是内容变了）")
            continue
        print(f"{path}    口径 {kind}")
        print(f"    算出来（{kind}）  {right}")
        print(f"    另一个口径        {other}   行尾 {endings}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
