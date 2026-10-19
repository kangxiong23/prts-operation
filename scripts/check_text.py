"""文本卫生：源码与文档里不许出现排版符号。

## 为什么有这条

一个真实的缺陷**活了两轮**：落文件时用 `«` `»` 当占位符、写完再替换成反引号，
**只替换了开引号、忘了闭引号**，于是 4 个文件里留下 80 处 `` `x» ``。

它躲过了当时**全部**四条守卫：

| 守卫 | 为什么不响 |
| --- | --- |
| `tests/spec/*.py` 验收 | 只跑代码，不读源码文本 |
| `pytest` | 同上 |
| `check_docs.py` | 只扫 `docs/`，且只看状态行 / 链接 / 废词 |
| `import_rules.py` | 只看 import 关系 |

docstring 里的坏字符**不执行**，所以没有任何一条会响。**这类东西只能靠扫文本抓。**

**同一类缺陷又发生了一次，这次是 `@@`**（落文件时用 `@@x@@` 当"这里要一个反引号"
的标记，批量替换时只转了一半）：4 处、2 个文件，全在 docstring 里，同样躲过了上面那四条
守卫。所以规矩加一条 —— **`@@` 在这个库里没有合法出现的地方**（这个库不处理补丁文本）。

## 规矩

- **源码 · 文档 · JSON 里一律不许出现 `«` `»` `‹` `›`** —— 它们不是这个项目用的
  排版符号，出现就说明有占位符没替换干净；
- **源码 · 文档 · JSON 里一律不许出现 `@@`** —— 同上：它是"这里要一个反引号"的补丁标记。
  **不许开豁免名单** —— 真找到一个合法出现的地方，那是要改规矩，不是要开豁免；
- **`.py` 里不许出现弯引号 `“ ” ‘ ’`** —— 它们**会真的把字符串写坏**
  （肉眼像直引号，解释器不认）；文档里用弯引号没问题，所以只禁源码。

## 为什么 JSON 也在里面

**夹具的 JSON 是"没有任何测试会读的那一层文本"**。`index.json` / `取材.json` / `点名.json`
里的 `note`、字段名、判决说明都是写给人看的 —— 没有一条防线会读它们。占位符落在那里，
和落在 docstring 里一模一样：**不执行、不报错、活到下一轮**。

**入仓的 JSON 一份不落**（今天 **77 份**：`tests/` 74 ＋ `assets/` 2 ＋ `dsh/` 1 ——
`git ls-files *.json` 数出来的正是这 77 份）。

**唯一不扫的是 `data/`**：它是 gitignored 的中间产物（今天里面躺着 **2304 份** JSON：
探针、scratch、落盘输出都在那儿），脚本随手写、随手删 —— 扫它等于拿草稿当交付物。
`src/` `scripts/` `docs/` `.grill/` 今天一份 JSON 都没有；`docs/` 的 `.md` 已经在
上面那 74 份文档里。

**规矩与文档那条相同**（只禁 `«»‹›` 与 `@@`），两条**故意不立**：

- **波浪线 `~`**：它在文本里合法（`≈`、`1~2` 都有人这么写），禁它抓不到这类事故；
- **弯引号 `“ ” ‘ ’`**：JSON 里那是正常中文标点，不写坏任何东西（`.py` 里才是错的）。

**不许开豁免名单**：真碰到一份"合法但含违禁字符"的 JSON，那是要改规矩或者改那份夹具，
不是要开名单。

## 跑法

    python scripts/check_text.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(".")
SOURCE_DIRS = ("src", "tests", "scripts", ".grill")
DOC_DIRS = ("docs",)
JSON_DIRS = ("tests", "assets", "dsh")
"""JSON 扫哪几棵：**入仓的 JSON 都在这儿** —— 夹具（`tests/`）· 资产（`assets/`）·
`dsh/` 那份手写的包清单。

`git ls-files *.json` 今天正是这 77 份；`data/` 是 gitignored 的中间产物（2304 份），**不扫**。"""

# 占位符残留：源码与文档都不许有
PLACEHOLDER = "«»‹›"
# 补丁标记残留：源码与文档都不许有（这个库不处理补丁文本，它没有合法出现的地方）
MARKER = "@@"
# 弯引号：只有源码里是错的（会写坏字符串字面量）
CURLY = "“”‘’"

# 扫描的单位是**记号**（字符串），不是字符：`@@` 是两个字符、一个记号，
# 按字符扫会把每一行 Python 的 `@dataclass` 全报成违规。
SOURCE_BANNED = tuple(PLACEHOLDER) + (MARKER,) + tuple(CURLY)
DOC_BANNED = tuple(PLACEHOLDER) + (MARKER,)
# JSON 与文档**同一条规矩**：占位符残留 · 补丁标记。波浪线 `~` 不立规矩（文本里合法，
# 禁它抓不到这类事故）；弯引号也不禁（JSON 的字符串是给人读的中文，`“”` 不写坏任何东西）。
JSON_BANNED = tuple(PLACEHOLDER) + (MARKER,)

# 本文件自己得写出这些字符才禁得了它们，所以**把自己排除在外**。
# 排除的是唯一一份"必须包含违禁字符"的文件 —— 它的正文不再被别的东西扫描。
SELF = pathlib.Path(__file__).resolve()


def scan(paths, banned: tuple[str, ...]) -> list[str]:
    hits = []
    for path in paths:
        if path.resolve() == SELF:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as bad:
            # **不许静默跳过**：解不出来本身就说明这个文件不是 UTF-8，
            # 而"不是 UTF-8"比"含违禁字符"更该报 —— 静默跳过等于给
            # 每一个非 UTF-8 文件开了后门（这条自己踩过）。
            hits.append(f"{path}: 不是 UTF-8，读不出来（{bad}）")
            continue
        except OSError as bad:
            hits.append(f"{path}: 读不了（{bad}）")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for token in banned:
                if token in line:
                    hits.append(f"{path}:{lineno}: 出现 {token!r} —— {line.strip()[:70]}")
    return hits


def main() -> int:
    py = [p for d in SOURCE_DIRS if (ROOT / d).is_dir() for p in (ROOT / d).rglob("*.py")]
    md = [p for d in DOC_DIRS if (ROOT / d).is_dir() for p in (ROOT / d).rglob("*.md")]
    js = [p for d in JSON_DIRS if (ROOT / d).is_dir() for p in (ROOT / d).rglob("*.json")]

    hits = scan(py, SOURCE_BANNED) + scan(md, DOC_BANNED) + scan(js, JSON_BANNED)
    print(f"源码 {len(py)} 份 ｜ 文档 {len(md)} 份 ｜ JSON {len(js)} 份")
    if hits:
        print(f"\n不干净 {len(hits)} 处：")
        for h in hits[:40]:
            print(f"  - {h}")
        if len(hits) > 40:
            print(f"  ……还有 {len(hits) - 40} 处")
        return 1
    print(
        f"占位符残留 0 处（{PLACEHOLDER}）｜ 补丁标记 0 处（{MARKER}）"
        f"｜ 源码弯引号 0 处（{CURLY}）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
