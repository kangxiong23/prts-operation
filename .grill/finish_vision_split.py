# -*- coding: utf-8 -*-
"""收尾：补上 split_vision.py 漏掉的两处，然后扫全树确认没有残留的 视觉层.md 引用。

残留扫描是这次改动的主要风险守卫（改漏一处 = 指向一个已被删除的文件）。
扫到任何残留就退出 1，且**不删原文件**。
"""

import pathlib
import sys

ROOT = pathlib.Path(".")
SRC = ROOT / "docs/10-规格/视觉层.md"

FIX = [
    ("docs/10-规格/资产生成器/游戏数据库.md",
     "见 [视觉层](../视觉层.md) §7.1 |",
     "见 [视觉层](../视觉层/技能详情通道.md) |"),
    ("docs/10-规格/资产生成器/游戏数据库.md",
     "→ `../视觉层.md`",
     "→ `../视觉层/`"),
]

for path, old, new in FIX:
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    n = s.count(old)
    if n != 1:
        sys.exit("期望恰好 1 处，实际 %d 处：%s" % (n, path))
    p.write_text(s.replace(old, new), encoding="utf-8", newline="")
    print("补 %s" % path)

leftovers = []
for p in sorted(ROOT.glob("docs/**/*.md")):
    for i, line in enumerate(p.read_text(encoding="utf-8").split("\n"), 1):
        if "视觉层.md" in line:
            leftovers.append("%s:%d: %s" % (p.as_posix(), i, line.strip()))

if leftovers:
    print("仍有 %d 处残留，原文件保留：" % len(leftovers))
    for item in leftovers:
        print("  " + item)
    sys.exit(1)

print("残留 0 处")
SRC.unlink()
print("删除 %s" % SRC.as_posix())
