"""文本闸门把 **JSON 夹具**也扫进来了（`scripts/check_text.py`）。

**为什么单开一份**：这道闸门从前只扫 `.py` 与 `docs/*.md` —— **77 份 JSON 一份没扫**；
而夹具 JSON 的 `note` / 字段名正是「没有任何测试会读的那一层文本」：占位符落在那里不执行、
不报错，能活到下一轮（历史上那类事故活在 4 个文件 80 处里）。

三条网：

1. **含禁用字符的 JSON 会被抓** —— 这一条在把 JSON 加进来之前是红的（那时它扫不到）；
2. **规矩与文档同一条**（四个占位符字符 ＋ 补丁标记），**波浪线 `~` 与弯引号故意不立**；
3. **入仓的 JSON 一份不落**（`tests/` ＋ `assets/` ＋ `dsh/` 今天共 77 份），`data/` 的草稿不扫。

**这一份自己不许写出那四个字符与那个标记**（它在 `tests/` 里，闸门扫得到它）：一律用
码位拼出来（见下面的 `PLACEHOLDER` / `MARKER` / `CURLY`）—— **不是给本文件开豁免**，
是把"要喂给闸门的坏字符"当成数据造出来。

**不用 `tmp_path`**：本机 `data/pytest_scratch/pytest-of-ASUS/` 的 ACL 坏了
（`PermissionError`），仓里统一自带 scratch 目录（见 `test_source_hashes.py`）。
"""

from __future__ import annotations

import io
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import check_text as GATE  # noqa: E402

SCRATCH = REPO / "data" / "probe" / "pytest_scratch"

# 违禁字符本身：**用码位拼**，好让本文件自己干净。
PLACEHOLDER = chr(0xAB) + chr(0xBB) + chr(0x2039) + chr(0x203A)  # U+00AB U+00BB U+2039 U+203A
MARKER = "@" * 2  # 补丁标记
CURLY = chr(0x201C) + chr(0x201D) + chr(0x2018) + chr(0x2019)  # U+201C U+201D U+2018 U+2019


def _scratch(name: str) -> Path:
    """一条用例自己的小仓（先清空再建）。"""
    root = SCRATCH / name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    return root


def _run(root: Path) -> tuple[int, str]:
    """在这棵小仓上跑一遍闸门，回 `(退出码, 输出)`。"""
    old = GATE.ROOT
    GATE.ROOT = root
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            code = GATE.main()
    finally:
        GATE.ROOT = old
    return code, out.getvalue()


def _write(root: Path, where: str, text: str) -> None:
    path = root / where
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline=chr(10))


def _json_line(note: str) -> str:
    """一份最小的 JSON：只有一行 `note`。"""
    body = note.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))
    return "{" + chr(10) + '  "note": "' + body + chr(34) + chr(10) + "}" + chr(10)


def test_a_json_fixture_with_a_placeholder_is_red() -> None:
    """**这一条在把 JSON 加进来之前是红的**：只扫 `.py`/`.md` 的闸门看不见夹具 JSON。

    喂的字符正是那两次真实事故留下的（占位符残留 · 补丁标记），落在 `note` 里 ——
    正是「不执行、没人读」的那个位置。
    """
    root = _scratch("text_gate_json_red")
    bad = "本该是反引号：" + chr(96) + "x" + PLACEHOLDER[1] + " 而这里是 " + MARKER + " 标记"
    _write(root, "tests/帧样本/小批/节.json", _json_line(bad))
    code, text = _run(root)
    assert code == 1, text
    assert "节.json" in text
    assert PLACEHOLDER[1] in text, "要把抓到的那个字符打出来"
    assert MARKER in text, "要把补丁标记也打出来"
    assert "JSON 1 份" in text, "要把扫了几份 JSON 报出来"


def test_a_clean_json_fixture_is_green() -> None:
    """同一棵树、把坏字符拿掉 ⇒ 绿（**红不是「JSON 一进来就红」**）。"""
    root = _scratch("text_gate_json_green")
    _write(root, "tests/帧样本/小批/节.json", _json_line("这里本该是反引号：" + chr(96) + "x" + chr(96)))
    code, text = _run(root)
    assert code == 0, text
    assert "JSON 1 份" in text


def test_the_json_rule_is_the_docs_rule_and_the_tilde_is_not_in_it() -> None:
    """**规矩与文档同一条**：只禁那四个占位符字符与补丁标记。

    **波浪线故意不在里面**（用户裁定）：它在文本里合法，禁它抓不到这类事故；
    **弯引号也不在**：JSON 里那是正常中文标点（`.py` 里才是错的）。
    """
    assert GATE.JSON_BANNED == GATE.DOC_BANNED
    assert GATE.JSON_BANNED == tuple(PLACEHOLDER) + (MARKER,)
    assert GATE.JSON_BANNED == GATE.SOURCE_BANNED[:5]
    assert "~" not in GATE.JSON_BANNED
    assert chr(0xFF5E) not in GATE.JSON_BANNED
    assert not any(ch in GATE.JSON_BANNED for ch in CURLY)

    root = _scratch("text_gate_tilde")
    _write(root, "tests/行.json", _json_line("1~2 与 " + chr(0x2248) + " 都合法"))
    code, text = _run(root)
    assert code == 0, text


def test_the_json_trees_are_the_repos_own_json() -> None:
    """扫**入仓的 JSON**（`tests/` ＋ `assets/` ＋ `dsh/`）；`data/` 是 gitignored 的草稿，不扫。

    这一条两个方向都钉：`dsh/` 里放个坏 JSON ⇒ **红**（它在判据里）；
    `data/` 里放个坏 JSON ⇒ **绿**（那是草稿，不是交付物）。
    """
    assert GATE.JSON_DIRS == ("tests", "assets", "dsh")
    root = _scratch("text_gate_trees")
    bad = chr(96) + "x" + PLACEHOLDER[1]
    _write(root, "data/草稿.json", _json_line(bad))
    _write(root, "dsh/清单.json", _json_line("干净"))
    assert _run(root)[0] == 0, "data/ 的草稿不该被扫"
    _write(root, "dsh/清单.json", _json_line(bad))
    code, text = _run(root)
    assert code == 1, text
    assert "清单.json" in text, "dsh/ 在判据里"


def test_todays_repo_is_clean_in_all_three_kinds() -> None:
    """今天这一仓：源码 · 文档 · JSON 三栏都有数、0 处不干净（闸门自己的冒烟）。"""
    code, text = _run(REPO)
    assert code == 0, text
    assert "占位符残留 0 处" in text
    assert "JSON " in text
