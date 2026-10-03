"""docs 树的通用守卫：状态行、相对链接、已废术语。

只用标准库，没装 pytest 也能跑：

    python scripts/check_docs.py

**已废术语这一节是"判定规则只写一份"的地方**：改名与合并（格位 → 槽、出怪表 → 路线资产
这类）改漏一处是主要风险，所以它写成守卫，而不是靠人记得。
"""

import pathlib
import re
import sys
import urllib.parse

ROOT = pathlib.Path("docs")
VALID_STATUS = {"未开始", "讨论中", "已定", "已取代", "已归档"}
STATUS_RE = re.compile(r"^\s*(?:>\s*|-\s*)?状态：\s*\*{0,2}([^\s*｜|]+)")
LINK_RE = re.compile(r"\]\(([^)#]+)(?:#[^)]*)?\)")

# 已废术语：词 → (该改成什么, 允许出现的文件集合，相对 docs/ 的 posix 路径)
# 基线回改已做完（2026 批准），所以白名单只剩"引用别人说法"的那一处。
RETIRED = {
    "格位": ("「槽」——旧名与「格」撞车", set()),
    "质量等级": ("「取值形态」——旧名暗示存在全序", set()),
    "识别层": ("「视觉层」", set()),
    "出怪时刻表": ("「路线资产」——出怪已并进去", set()),
    "出怪表": ("「路线资产」", set()),
    "波次": ("「出怪」——游戏没有波次概念", {"50-外部借鉴/MAA借鉴清单.md"}),
}

# 行内出现这些字样，说明这一行是在讲"这个词已经废了"，不算违规。
RETIRE_MARKERS = (
    "不再",
    "已废",
    "废掉",
    "并进",
    "同时就是",
    "已落",
    "→",
    "曾叫",
    "已改",
    "旧名",
    "叫成",
    "没有",
    "保留",
    "没改",
)


def check_retired(docs):
    hits = []
    for md in docs:
        rel = md.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for term, (instead, allowed) in RETIRED.items():
                if term not in line or rel in allowed:
                    continue
                if any(marker in line for marker in RETIRE_MARKERS):
                    continue
                hits.append((rel, lineno, term, instead))
    return hits


def main() -> int:
    docs = sorted(ROOT.rglob("*.md"))
    missing_status, bad_status, broken = [], [], []

    for md in docs:
        text = md.read_text(encoding="utf-8")

        head = text.splitlines()[:8]
        found = next((m.group(1).strip() for m in map(STATUS_RE.match, head) if m), None)
        if found is None:
            missing_status.append(md)
        elif found not in VALID_STATUS:
            bad_status.append((md, found))

        for m in LINK_RE.finditer(text):
            target = m.group(1).strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            path = (md.parent / urllib.parse.unquote(target)).resolve()
            if not path.exists():
                broken.append((md, target))

    retired = check_retired(docs)

    print(f"文档 {len(docs)} 份")
    print(
        f"状态行缺失 {len(missing_status)}　取值非法 {len(bad_status)}　断链 {len(broken)}"
        f"　已废术语 {len(retired)}"
    )
    for md in missing_status:
        print(f"  缺状态行: {md.as_posix()}")
    for md, value in bad_status:
        print(f"  非法取值: {md.as_posix()} -> {value}")
    for md, target in broken:
        print(f"  断链: {md.as_posix()} -> {target}")
    for rel, lineno, term, instead in retired:
        print(f"  已废术语: {rel}:{lineno} 「{term}」 → {instead}")

    return 1 if (missing_status or bad_status or broken or retired) else 0


if __name__ == "__main__":
    sys.exit(main())
