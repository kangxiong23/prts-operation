# -*- coding: utf-8 -*-
"""一次性：把 docs/10-规格/视觉层.md 拆成目录。

只做机械动作：
  1. 按行区间原样搬 3 块正文到 docs/10-规格/视觉层/ 下；
  2. README.md 也原位搬（§4 / 尚未定 / 已定），只留一个 <!-- INDEX --> 占位；
  3. 目录内的相对链接统一加深一层（视觉层.md -> 视觉层/）；
  4. 迁移外部 12 处指向 视觉层.md 的链接；
  5. 删除原文件。

不做任何措辞改动。写完即由 scripts/check_docs.py 守卫，本脚本随后作废。
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(".")
SRC = ROOT / "docs/10-规格/视觉层.md"
OUT = ROOT / "docs/10-规格/视觉层"

text = SRC.read_text(encoding="utf-8")
lines = text.split("\n")
if lines[-1] != "":
    sys.exit("原文件末尾没有换行，行区间假设不成立，停下来人工看")


def block(a, b):
    """1-based 闭区间，原样返回（保留行尾换行）。"""
    return "\n".join(lines[a - 1:b]) + "\n"


def deepen(md):
    """目录内的相对链接加深一层。锚点与绝对 URL 不动。"""
    def rep(m):
        target = m.group(1)
        if target.startswith(("#", "http://", "https://", "mailto:")):
            return m.group(0)
        return "](../%s)" % target
    return re.sub(r"\]\(([^)]+)\)", rep, md)


HEAD = "> 状态：讨论中 ｜ 由 视觉层.md 拆出\n"

moved = {}

# 1. 场上我方干员.md  <- 原 §1 §2 §3（18..65）
body = block(18, 65).replace("## 1. 我方干员：要读的东西都画在图层最上层",
                             "## 1. 要读的东西都画在图层最上层", 1)
moved["场上我方干员.md"] = (
    "# 场上我方干员\n\n" + HEAD +
    "\n对应代码：`src/ark/engine/vision/`。本文件管**屏幕上我方干员那一块**要读什么。\n\n"
    "---\n\n" + body)

# 2. 备战席.md  <- 原 §5（75..102）
body = block(75, 102).replace("## 5. 判死与再部署", "## 判死与再部署", 1)
moved["备战席.md"] = (
    "# 备战席\n\n" + HEAD +
    "\n对应代码：`src/ark/engine/vision/`。本文件管**底部备战席那一栏**要读什么。\n\n"
    "---\n\n" + body)

# 3. 技能详情通道.md  <- 原 §7（111..193）
body = block(111, 193).replace("## 7. 技能详情通道", "# 技能详情通道", 1)
moved["技能详情通道.md"] = (
    "对应代码：`src/ark/engine/vision/`。\n\n" + HEAD +
    "\n---\n\n" + body)

# 4. README.md  <- 头部 + §4 + 尚未定 + 已定
resp = block(7, 17)          # 这份文件负责什么 + 不在这里
prec = block(66, 74).replace("## 4. 由此定下坐标精度的分工", "## 坐标精度的分工", 1)
todo = block(194, 200)
done = block(201, 206)
moved["README.md"] = (
    "# 视觉层\n\n"
    "> 状态：讨论中 ｜ 已拆目录；只写了**场上我方干员**、**技能详情通道**、**备战席**三块"
    "（都有实例图片支撑）；敌人、场地机制、顶部 HUD 还没写\n\n"
    "对应代码：`src/ark/engine/vision/`。\n\n"
    "---\n\n" + resp + "\n<!-- INDEX -->\n\n---\n\n" + prec + "\n---\n\n" + todo + "\n" + done)

OUT.mkdir(parents=True, exist_ok=True)
for name, content in moved.items():
    (OUT / name).write_text(deepen(content), encoding="utf-8", newline="")
    print("写出 %s（%d 行）" % (name, content.count("\n")))

# 5. 外部链接迁移：只换链接目标，不动句子
SITES = [
    ("docs/README.md",
     "[视觉层.md](10-规格/视觉层.md)",
     "[视觉层/](10-规格/视觉层/)"),
    ("docs/00-总览/开题状态.md",
     "见 [视觉层](../10-规格/视觉层.md) §1。",
     "见 [视觉层](../10-规格/视觉层/场上我方干员.md)。"),
    ("docs/10-规格/代码结构.md",
     "| [视觉层.md](视觉层.md) |",
     "| [视觉层/](视觉层/) |"),
    ("docs/10-规格/控制层.md",
     "见 [视觉层](视觉层.md) §7）",
     "见 [视觉层](视觉层/技能详情通道.md)）"),
    ("docs/10-规格/控制层.md",
     "读取的顺序前提见 [视觉层](视觉层.md) §7.5）",
     "读取的顺序前提见 [视觉层](视觉层/技能详情通道.md)）"),
    ("docs/10-规格/控制层.md",
     "（[视觉层](视觉层.md) §7.2）",
     "（[视觉层](视觉层/技能详情通道.md)）"),
    ("docs/10-规格/状态层/全局与HUD.md",
     "（[视觉层](../视觉层.md) §6 那条待查线索）",
     "（[视觉层](../视觉层/识别方法.md) 那条待查线索）"),
    ("docs/10-规格/状态层/README.md",
     "见 [视觉层](../视觉层.md)**（注册表待建）",
     "见 [视觉层/识别方法.md](../视觉层/识别方法.md)**"),
    ("docs/10-规格/状态层/README.md",
     "[`../视觉层.md`](../视觉层.md)",
     "[`../视觉层/`](../视觉层/)"),
    ("docs/10-规格/状态层/基础数值.md",
     "见[视觉层](../视觉层.md) §7.5",
     "见[视觉层](../视觉层/技能详情通道.md)"),
    ("docs/10-规格/资产生成器/游戏数据库.md",
     "见 [视觉层](../视觉层.md) §2）",
     "见 [视觉层](../视觉层/场上我方干员.md)）"),
    ("docs/10-规格/资产生成器/游戏数据库.md",
     "见 [视觉层](../视觉层.md) §7.1）",
     "见 [视觉层](../视觉层/技能详情通道.md)）"),
]

for path, old, new in SITES:
    p = ROOT / path
    s = p.read_text(encoding="utf-8")
    n = s.count(old)
    if n != 1:
        sys.exit("期望恰好 1 处，实际 %d 处：%s -> %s" % (n, path, old))
    p.write_text(s.replace(old, new), encoding="utf-8", newline="")
    print("迁移 %s  <- %s" % (path, old[:34]))

SRC.unlink()
print("删除 %s" % SRC.as_posix())
