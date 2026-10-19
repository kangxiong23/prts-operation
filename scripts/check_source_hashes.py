"""核夹具里每一处「源帧指纹」：**带路径的源，按磁盘原样的字节算 sha256**。

只使用标准库，所以**不装 pytest 也能跑**：

    python scripts/check_source_hashes.py

## 判据只认「字段名自己说明了它是谁的哈希」

本闸门核的是这一件事：**夹具说「这一条是从那个文件来的」，那就不许把哈希算成别的东西**。
所以它只认三种字段名，而且要求同一层里有对应的路径键：

| 字段名 | 路径键 | 它声称的是什么 |
| --- | --- | --- |
| `source_sha256` | `source` | 源帧（或源文件）的字节 |
| `page_sha256` | `page` | 那一张编队页的字节 |
| `sha256` | `image` | 那一张图的字节（`assets/roster/*.json` 的 `page` 块） |

别的 `*sha256` 字段（`raw_sha256` · `sheet_sha256` · `scratch_sha256_raw` ·
`producer_sha256_lf`）**不在这道闸门的判据里**，但它们**会进账、会被打印出来**：
不判红不等于不看见。

## 口径：文本折 LF，二进制按磁盘原样

| 源是什么 | 口径 | 为什么 |
| --- | --- | --- |
| 文本（`.py` / `.json` / `.md` / …） | **LF 折算** | git 的 `autocrlf` 让磁盘字节与机器相关，同一份内容在两种行尾下 sha256 不同 |
| 二进制（`.png` / `.jpg` / `.raw` / …） | **磁盘原样** | 换行在二进制里没有意义；折一次就是**把整个文件重写一遍再算** —— 那是另一个文件的哈希 |

**这一条就是从 9 条的错里长出来的**：那 9 条全是 PNG，却被按 LF 折算算了。
口径**按后缀判**（`TEXT_SUFFIXES`），不靠人记得 —— 这是本闸门要解决的第二个问题。

## 它不静默

`screenshots/` 不入库、**随时会被删**。所以分三个数报：**能核 N 条 / 核不到 M 条 /
源是描述串 K 条**。源文件不在本地时**只计数、不判红** —— 那不是"错了"，是"今天核不了"；
核不到的**逐条列出来**，免得「看着全绿」被读成「全都核过」。

判红只有一种情况：**源在本地、算得出哈希、而记下来的那个与它不同**。

## 「源」不一定是一条路径

有几处 `source` 写的是一句描述（「由上面那张 page 拼出来的（不是原图的一部分）」、
「webapi… 的原始响应文本」）。那些**不是路径**，不当路径解析、也不进「核不到」的账。
判据是**长不长得像一条路径**，不是**文件在不在**：按后者分，一个被删掉的源会静默
变成"描述串"，于是「核不到」这个数永远报不出东西来。

## 它不在单元测试里

去重后 137 个源文件、约 561 MB。`pytest` 默认收集的那两棵每跑一次就读一遍，
是拿几百 MB 的磁盘 IO 换一句"应该不会错"。所以它是**第五道闸门**，与另外四条并排。

**耗时**（本机）：整条闸门 **5.7 s**，其中哈希本身 **3.4 s**，其余是解释器启动与 JSON 解析。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash_conventions import (  # noqa: E402
    TEXT_SUFFIXES,  # noqa: F401
    convention,
    expected_hash,
    sha256_lf,
    sha256_raw,
)

REPO = Path(__file__).resolve().parents[1]

ROOTS = ("tests", "assets")
"""扫哪两棵：夹具与资产。`data/` 是 gitignored 的中间产物，不是"记下来的事实"。"""

DECLARED = {
    "source_sha256": "source",
    "page_sha256": "page",
    "sha256": "image",
}
"""这一道闸门认的字段：字段名 → 它对应的路径键。**别往里加口径不明的。**"""

PIXEL_FIELDS = frozenset({"pixels_sha256"})
"""**像素级**哈希：它压根不该等于文件字节哈希，所以**不进这道闸门**。

（`assets/roster/faces.json` 里那 10 条是"解码之后那块 RGB 的哈希"，用来证明同一套立绘
第二次遇到是同一块像素 —— 它本来就不该与 PNG 字节对上。这一条**不是缺陷**。）"""

LOOKS_LIKE_PATH = re.compile(r"^[^\s\u3000]*(?:/|\\\\)[^\s\u3000]*$")
"""一句描述**长什么样**：名字里有分隔符、且不含空白。"""


def _walk(node, trail):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, trail + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, trail + ("[%d]" % index,))
    else:
        yield trail, node


def _at(data, trail):
    node = data
    for key in trail:
        node = node[int(key[1:-1])] if key.startswith("[") else node[key]
    return node


def entries(repo: Path):
    """扫出来的每一处 `*sha256` 字段：`(哪份文件, 数据, 路径, 字段名, 记下来的值)`。

    **不按文件名猜**：扫 `tests/` 与 `assets/` 下所有 `.json`。**不写死某几个文件** ——
    新加一批夹具时不用回来改这一份。
    """
    out = []
    for root in ROOTS:
        base = repo / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            for trail, value in _walk(data, ()):
                if not trail or not isinstance(trail[-1], str):
                    continue
                if not trail[-1].endswith("sha256"):
                    continue
                out.append((path, data, trail, trail[-1], value))
    return out


def resolve(value: str, here: Path):
    """一条路径 → 仓内的那个文件（先按仓根，再按这份 JSON 自己所在的目录）。"""
    for base in (REPO, here.parent):
        candidate = base / value
        if candidate.is_file():
            return candidate.resolve()
    return None


def source_of(data, trail, here):
    """同一层里的路径 → `(来源, 状态)`。`here` 是这份 JSON 自己的路径。

    `状态` 只有三种，**这三种不能混**：

    - `"文件"` —— 解析得成仓内的一个文件；
    - `"描述"` —— 它压根**不是一条路径**（"由上面那张 page 拼出来的……"）；
    - `"不存在"` —— 它**长得像路径**、但那个文件今天不在本地（`screenshots/` 会被删）。
    """
    parent = _at(data, trail[:-1])
    if not isinstance(parent, dict):
        return None, "描述"
    key = DECLARED.get(trail[-1])
    for name in ([key] if key else []) + ["source", "page", "image", "file"]:
        value = parent.get(name)
        if not isinstance(value, str) or not value:
            continue
        found = resolve(value, here)
        if found is not None:
            return found, "文件"
        if LOOKS_LIKE_PATH.match(value):
            return value, "不存在"
        return value, "描述"
    return None, "描述"


def audit(repo: Path | None = None) -> dict:
    """跑一遍，回一本账。**判红的只有 `wrong`**；核不到只计数。"""
    global REPO
    if repo is not None:
        REPO = repo
    checked = 0
    wrong = []
    missing = []
    described = []
    undeclared = []
    by_convention = {}
    for path, data, trail, field, value in entries(REPO):
        where = "%s %s" % (path.relative_to(REPO), ".".join(trail))
        if field in PIXEL_FIELDS:
            continue
        if field not in DECLARED:
            undeclared.append("%s（字段名 %s）" % (where, field))
            continue
        source, state = source_of(data, trail, path)
        if state == "描述":
            described.append(where)
            continue
        if state == "不存在":
            missing.append("%s -> %s（文件不在）" % (where, source))
            continue
        kind = convention(source)
        checked += 1
        by_convention[kind] = by_convention.get(kind, 0) + 1
        got = expected_hash(source)
        if value != got:
            wrong.append(
                "%s -> %s：记的是 %s，%s算出来是 %s"
                % (where, source.relative_to(REPO), value, kind, got)
            )
    return {
        "checked": checked,
        "by_convention": by_convention,
        "wrong": wrong,
        "missing": missing,
        "described": described,
        "undeclared": undeclared,
    }


def main() -> int:
    report = audit(REPO)
    print(
        "能核 %d 条（磁盘原样 %d ｜ LF 折算 %d）｜ 核不到 %d 条 ｜ 源是描述串 %d 条"
        % (
            report["checked"],
            report["by_convention"].get("磁盘原样", 0),
            report["by_convention"].get("LF 折算", 0),
            len(report["missing"]),
            len(report["described"]),
        )
    )
    if report["missing"]:
        print("  核不到（源文件不在本地 —— **只列出来，不判红**）：")
        for item in report["missing"][:10]:
            print("    - %s" % item)
        if len(report["missing"]) > 10:
            print("    ……还有 %d 条" % (len(report["missing"]) - 10))
    if report["undeclared"]:
        print("  不在这道闸门的判据里（字段名没写明它是谁的哈希）：%d 条" % len(report["undeclared"]))
        for item in report["undeclared"][:10]:
            print("    - %s" % item)
    if report["wrong"]:
        print("")
        print("对不上 %d 条：" % len(report["wrong"]))
        for item in report["wrong"]:
            print("  - %s" % item)
        return 1
    print("带路径的源帧指纹全部对上")
    return 0


if __name__ == "__main__":
    sys.exit(main())
