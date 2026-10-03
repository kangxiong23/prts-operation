"""核哈希：算一个文件的 sha256，口径是**按 LF 折算**。

派任务与交回时用的就是这个口径（[D-全局-18](../docs/20-决策记录/全局.md)）。
**不要用 `Get-FileHash` 核**：仓库里 `core.autocrlf=true`，于是磁盘字节是机器相关
的 —— git checkout 出来的文件是 CRLF，新写的文件是 LF。同一份内容在两种行尾下
sha256 不同，直接核会误报"内容对不上"。

用法：

    python scripts/filehash.py <路径> [更多路径…]
    python scripts/filehash.py <路径> --expect <sha256>

第一种打印两个口径（LF 折算 / 磁盘原样）与行尾构成；第二种核对单个文件，
不一致就退出 1 —— 任务消息里的哈希就是这么核的。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _describe(path: Path) -> tuple[str, str, str]:
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = raw.count(b"\r\n")
    bare = raw.count(b"\n") - crlf
    endings = "CRLF" if bare == 0 and crlf else ("LF" if crlf == 0 else "混合")
    return _sha256(lf), _sha256(raw), f"{endings}（CRLF {crlf} / 裸 LF {bare}）"


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
        canonical, raw_hash, endings = _describe(path)
        if expect is not None:
            ok = canonical == expect
            failed = failed or not ok
            print(f"{'✓' if ok else '✗'} {path}")
            print(f"    LF 折算   {canonical}")
            if not ok:
                print(f"    期望      {expect}")
                print(f"    磁盘原样  {raw_hash}   行尾 {endings}")
                print("    （磁盘原样与 LF 折算不同是正常的；对不上才是内容变了）")
            continue
        print(f"{path}")
        print(f"    LF 折算（口径）  {canonical}")
        print(f"    磁盘原样         {raw_hash}   行尾 {endings}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
