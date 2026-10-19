"""源帧指纹的两条口径（`scripts/filehash.py` · `scripts/check_source_hashes.py`）。

**这两份脚本不是「顺手工具」**：9 条错口径就是从它们身上长出来的 —— `filehash.py` 从前
对**所有**文件都打「LF 折算（口径）」，而搬夹具的人照着「口径」那一栏抄，PNG 也被折了 LF。
所以这一份钉四件事：

1. **口径按后缀分**：`.png` 是磁盘原样、`.py` 是 LF 折算 —— 这是会响的那一条；
2. **闸门的账分得清**：能核 / 核不到 / 描述串，**三种不许混**（核不到不判红，但不许被吞）；
3. **闸门对不上就红**，而且报得出是哪一条、记的是什么、算出来是什么；
4. **「原始载荷」不许再叫 `raw_sha256`**：那三处（两张拼图 + 一张竖排重建）补上**自己说明
   它哈希的是谁**的名字与同一层的路径键之后，要从「不在判据里」挪进「能核」—— 值一个不改。

**它不跑全仓那一趟**（那要读 561 MB）：全仓那一趟是 `scripts/check_source_hashes.py` 的活。
这里只在一棵**临时造的小仓**上跑。

**为什么不用 `tmp_path`**：本机 `data/pytest_scratch/pytest-of-ASUS/` 那个目录的 ACL 坏了
（`PermissionError: [WinError 5]`，`Remove-Item` 也删不掉），`tmp_path` 一用就报错。
仓里别处也是自带 scratch 目录（见 `test_formation_support_slot.py`），这里沿用同一个做法：
**自己建、自己清**，落在 `data/pytest_scratch/<这一条的名字>/`。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import check_source_hashes as GATE  # noqa: E402
import filehash  # noqa: E402

SCRATCH = REPO / "data" / "probe" / "pytest_scratch"


def _fresh(name: str) -> Path:
    """一条用例自己的小仓（先清空再建）。"""
    root = SCRATCH / name
    shutil.rmtree(root, ignore_errors=True)
    (root / "shots").mkdir(parents=True)
    (root / "shots" / "shot.png").write_bytes(_payload())
    _write_json(
        root / "tests" / "帧样本" / "小批" / "index.json",
        {
            "version": 1,
            "crops": [
                {
                    "file": "a.png",
                    "source": "shots/shot.png",
                    "source_sha256": "0" * 64,
                }
            ],
        },
    )
    return root


def _payload() -> bytes:
    return bytes(range(256)) * 8


def _write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
        newline=chr(10),
    )


def _with_hash(root: Path, value: str) -> Path:
    _write_json(
        root / "tests" / "帧样本" / "小批" / "index.json",
        {
            "version": 1,
            "crops": [
                {
                    "file": "a.png",
                    "source": "shots/shot.png",
                    "source_sha256": value,
                }
            ],
        },
    )
    return root


def test_both_scripts_take_the_convention_from_one_module() -> None:
    """**口径只有一处定义**：两份脚本都从 `hash_conventions` 取，不只拿到一样的值。

    只比「值相等」是不够的 —— 各写一份表也能暂时相等。这里断的是**同一个对象**，
    所以以后往其中一份加后缀，另一份会自动跟上。
    """
    import hash_conventions

    assert GATE.TEXT_SUFFIXES is hash_conventions.TEXT_SUFFIXES
    assert filehash.TEXT_SUFFIXES is hash_conventions.TEXT_SUFFIXES
    assert filehash.convention is hash_conventions.convention
    assert GATE.convention is hash_conventions.convention
    assert GATE.expected_hash is hash_conventions.expected_hash

def test_the_convention_is_decided_by_the_suffix_not_by_habit() -> None:
    """**口径按后缀**：`.png` 走磁盘原样、`.py` 走 LF 折算 —— 两份脚本必须说同一件事。

    这一条是 9 条错的正面：从前 `filehash.py` 把 LF 折算标成「口径」（对所有文件），
    搬夹具的人照抄，于是 PNG 也被折了。现在类型决定口径，而且两份脚本用**同一份后缀表**。
    """
    assert GATE.convention(Path("a/b/助战.png")) == "磁盘原样"
    assert GATE.convention(Path("a/b/cardface.py")) == "LF 折算"
    assert GATE.convention(Path("a/b/index.json")) == "LF 折算"
    assert GATE.convention(Path("a/b/board.raw")) == "磁盘原样"
    assert filehash.convention(Path("a/b/助战.png")) == "磁盘原样"
    assert filehash.convention(Path("a/b/cardface.py")) == "LF 折算"
    assert filehash.TEXT_SUFFIXES == GATE.TEXT_SUFFIXES, "两份脚本的口径表不许各说一套"


def test_a_png_recorded_by_its_raw_bytes_passes() -> None:
    """按磁盘原样记的 PNG 指纹 ⇒ 核得过（这就是修完之后那 9 条的样子）。"""
    root = _with_hash(_fresh("source_hash_raw"), hashlib.sha256(_payload()).hexdigest())
    report = GATE.audit(root)
    assert report["checked"] == 1
    assert report["wrong"] == []
    assert report["missing"] == []
    assert report["by_convention"] == {"磁盘原样": 1}


def test_a_png_recorded_by_its_lf_folded_bytes_is_red() -> None:
    """**这一条就是那 9 条的形状**：PNG 按 LF 折算记 ⇒ 红，而且报得出记的与算的。

    喂进去的那个 `recorded` 是**真的旧值**（`助战.png` 的 LF 折算哈希）。
    """
    old = "e6ea38f3639343a79fb636dfd13682d16cf94cbffedcba4e5292f98206186c54"
    root = _with_hash(_fresh("source_hash_lf"), old)
    report = GATE.audit(root)
    assert report["checked"] == 1
    assert len(report["wrong"]) == 1
    message = report["wrong"][0]
    assert "source_sha256" in message
    assert "shot.png" in message
    assert old in message, "要把记下来的那个值打出来"
    assert "磁盘原样算出来是" in message, "要说清用哪个口径算的"


def test_a_missing_source_is_counted_and_named_but_not_red() -> None:
    """源文件不在本地 ⇒ **只计数、只点名、不判红**（`screenshots/` 会被删）。

    但**不许静默**：「核不到」这个数必须报出来，而且那一条要能指认出来。
    """
    root = _fresh("source_hash_missing")
    (root / "shots" / "shot.png").unlink()
    report = GATE.audit(root)
    assert report["checked"] == 0
    assert report["wrong"] == []
    assert len(report["missing"]) == 1
    assert "shots/shot.png" in report["missing"][0]


def test_a_description_is_neither_checked_nor_counted_as_missing() -> None:
    """一句描述**不是路径**：不进「能核」，也不进「核不到」。

    混进来会让「核不到 M 条」这个数失去意义 —— 那一栏是给「被删掉的源帧」用的。
    """
    root = _fresh("source_hash_described")
    _write_json(
        root / "tests" / "帧样本" / "小批" / "index.json",
        {
            "crops": [
                {
                    "file": "a.png",
                    "source": "由上面那张 page 拼出来的（不是原图的一部分）",
                    "source_sha256": "0" * 64,
                }
            ]
        },
    )
    report = GATE.audit(root)
    assert report["checked"] == 0
    assert report["missing"] == []
    assert len(report["described"]) == 1


def test_pixel_hashes_are_not_file_hashes() -> None:
    """**像素级哈希不进闸门** —— 它本来就不该等于文件字节哈希，那不是缺陷。

    `assets/roster/faces.json` 里那 10 条量的是「解码之后那块 RGB」，而同一个对象的
    `sha256` 量的是 PNG 字节：两者等于同一个数才是错的。
    """
    assert "pixels_sha256" in GATE.PIXEL_FIELDS
    assert "pixels_sha256" not in GATE.DECLARED
    assert GATE.DECLARED["source_sha256"] == "source"
    assert GATE.DECLARED["page_sha256"] == "page"


def test_a_field_that_does_not_say_whose_hash_it_is_is_listed_not_judged() -> None:
    """字段名没写明「这是哪个文件的哈希」的（`mystery_sha256` 这种）⇒ **进账、不判红**。

    补上 `拼图_sha256` / `sheet_sha256` 之后，「不在判据里」这一栏今天空了，**但这一栏
    与它的机制留着**：新夹具再冒出名字含糊的字段时，它会出现在 `undeclared` 里、跑闸门
    时被打出来。**不判红不等于不看见。**
    """
    assert "mystery_sha256" not in GATE.DECLARED
    root = _fresh("source_hash_undeclared")
    _write_json(
        root / "tests" / "帧样本" / "小批" / "index.json",
        {"crops": [{"file": "a.png", "mystery_sha256": "0" * 64}]},
    )
    report = GATE.audit(root)
    assert report["checked"] == 0
    assert report["wrong"] == []
    assert len(report["undeclared"]) == 1


def test_the_gate_only_reads_tests_and_assets() -> None:
    """闸门只扫 `tests/` 与 `assets/` —— `data/` 是 gitignored 的中间产物，不是「记下来的事实」。"""
    assert GATE.ROOTS == ("tests", "assets")


def test_a_sheet_hash_under_its_own_name_is_judged() -> None:
    """**"接上网"的那一步**：`拼图_sha256` / `sheet_sha256` 一旦在 `DECLARED` 里，
    记错就红、记对就进「能核」。

    从前它们叫 `raw_sha256` / 根本没有路径键，于是只能躺在「不在判据里」——
    **量得对不对，闸门一句话也说不了**。
    """
    assert GATE.CHANGED_IN_THIS_TURN == ("拼图_sha256", "sheet_sha256")
    assert GATE.DECLARED["拼图_sha256"] == "拼图"
    assert GATE.DECLARED["sheet_sha256"] == "sheet"

    for field, key in (("拼图_sha256", "拼图"), ("sheet_sha256", "sheet")):
        good = _with_sheet(_fresh("sheet_hash_" + field), key, field, hashlib.sha256(_payload()).hexdigest())
        report = GATE.audit(good)
        assert report["checked"] == 1, field
        assert report["wrong"] == [], field
        assert report["undeclared"] == [], field

        bad = _with_sheet(_fresh("sheet_hash_" + field + "_bad"), key, field, "0" * 64)
        report = GATE.audit(bad)
        assert report["checked"] == 1, field
        assert len(report["wrong"]) == 1, field
        assert field in report["wrong"][0], field


def _with_sheet(root: Path, key: str, field: str, value: str) -> Path:
    """一棵小仓：`shots/shot.png` 是那张"拼图"，同一层里 `key` 指路径、`field` 记哈希。"""
    _write_json(
        root / "tests" / "帧样本" / "小批" / "index.json",
        {"crops": [{"file": "a.png", key: "shots/shot.png", field: value}]},
    )
    return root


def test_the_three_unnamed_hashes_now_land_in_the_account() -> None:
    """**这一条在补映射之前是红的**：全仓那三处从前去 `undeclared`，不进 `checked`。

    今天它们**逐条落进「能核」且对上** —— 只算那 3 份 PNG（合计约 143 KB），
    **不跑全仓那一趟**（561 MB）。三处的名字与路径键都在夹具里看得见，值一个没改。
    """
    found = {}
    for path, data, trail, field, value in GATE.entries(REPO):
        if field not in GATE.CHANGED_IN_THIS_TURN:
            continue
        where = "%s %s" % (path.relative_to(REPO), ".".join(trail))
        found[where] = GATE.source_of(data, trail, path) + (value,)
    assert len(found) == 3, "今天仓里恰好三处：" + repr(sorted(found))
    for where, (source, state, value) in found.items():
        assert state == "文件", where
        assert value == GATE.expected_hash(source), where


def test_the_name_raw_no_longer_holds_a_payload_hash() -> None:
    """**`raw` 在这个仓里指"按磁盘原样算"这个口径**（`hash_conventions.py`），
    「原始载荷」不许再占这个名字：今天 76 份 JSON 里 `raw_sha256` 一处不剩。

    改名只动了字段名与同一层的路径键，**三个值一个没改**（值本身由上面那条用例核）。
    """
    counts = {}
    for _path, _data, _trail, field, _value in GATE.entries(REPO):
        counts[field] = counts.get(field, 0) + 1
    assert "raw_sha256" not in counts
    assert counts.get("拼图_sha256") == 2
    assert counts.get("sheet_sha256") == 1

