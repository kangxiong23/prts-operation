# -*- coding: utf-8 -*-
"""收尾 2：清掉 9 处代码体残留（不是 markdown 链接，所以第一遍没抓到），然后删原文件。

残留扫描仍是硬门：扫到任何 视觉层.md 就退出 1、不删原文件。
"""

import pathlib
import sys

ROOT = pathlib.Path(".")
SRC = ROOT / "docs/10-规格/视觉层.md"

FIX = [
    # 代码体路径：目录化后一律指向目录
    ("docs/10-规格/契约层.md", "`视觉层.md`、`状态层/`", "`视觉层/`、`状态层/`"),
    ("docs/10-规格/控制层.md", "→ `视觉层.md`", "→ `视觉层/`"),
    ("docs/10-规格/设备层.md", "→ `视觉层.md`", "→ `视觉层/`"),
    ("docs/10-规格/资产生成器/README.md", "→ `../视觉层.md`", "→ `../视觉层/`"),
    ("docs/10-规格/资产生成器/地形与关卡资产.md", "→ `../视觉层.md`", "→ `../视觉层/`"),
    # 目录树
    ("docs/README.md",
     "    视觉层.md\n",
     "    视觉层/                  ← 已拆目录：README（索引与坐标分工）+ 识别方法 / 场上我方干员 / 技能详情通道 / 备战席\n"),
]

for path, old, new in FIX:
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    n = s.count(old)
    if n != 1:
        sys.exit("期望恰好 1 处，实际 %d 处：%s -> %s" % (n, path, old))
    p.write_text(s.replace(old, new), encoding="utf-8", newline="")
    print("补 %s" % path)

# 三个新文件的出处说明：不写真名，免得又留一个已删文件的引用
for name in ["场上我方干员.md", "备战席.md", "技能详情通道.md"]:
    p = ROOT / "docs/10-规格/视觉层" / name
    s = p.read_text(encoding="utf-8")
    old = "> 状态：讨论中 ｜ 由 视觉层.md 拆出\n"
    if s.count(old) != 1:
        sys.exit("状态行不符：%s" % p.as_posix())
    p.write_text(s.replace(old, "> 状态：讨论中 ｜ 由旧的单文件拆出\n"),
                 encoding="utf-8", newline="")
    print("改状态行 %s" % name)

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
