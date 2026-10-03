"""开发期小工具：看 docs 树的行数与 BOM。

    python .grill/docstat.py

按行数从多到少排，方便盯 200 行那条拆分阈值；末尾报告带 BOM 的文件（全树应为 0）。
"""

import pathlib
import sys

ROOT = pathlib.Path("docs")


def main() -> int:
    rows, bom = [], []
    for md in sorted(ROOT.rglob("*.md")):
        raw = md.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            bom.append(md.as_posix())
        rows.append((len(raw.decode("utf-8").splitlines()), md.as_posix()))

    rows.sort(reverse=True)
    for count, name in rows:
        flag = "  ← 逼近阈值" if count >= 175 else ""
        print(f"{count:4d}  {name}{flag}")
    print(f"\n共 {len(rows)} 份；带 BOM {len(bom)} 份")
    for name in bom:
        print(f"  BOM: {name}")
    return 1 if bom else 0


if __name__ == "__main__":
    sys.exit(main())
