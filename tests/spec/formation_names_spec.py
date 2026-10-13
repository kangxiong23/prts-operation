"""编队页名字那一读的规格符合性验收 —— **由规划者维护，不住在 tests/unit/**。

分工、纪律、跑法都在 [README](README.md) —— **不在这里重述**（同一条规定只写一次）。

## 这一批要回答什么

编队页卡下缘那 10 个名字是**读出来**的，规格在
[资产生成器/编队页名字那一读](../../docs/10-规格/资产生成器/编队页名字那一读.md)；
一次真机调用留下的**两块石头**在 `tests/帧样本/编队页名字竖排/`。这一批盯四件事：

- **石头在不在、哈希对不对**（只留哈希证不了「当年发出去的是这张」）；
- **判决链在夹具上重放得对不对**（10 个词、一格一词、逐格落在真值上、验收线成立）；
- **主份真值与两份副本逐字一致**（两份夹具 ＋ 本文件的 `FORMATION_TRUTH`）；
- **记账如实**（两次尝试只计费一次）。

**执行者不许读本文件**（D-全局-22）。

## 真值从哪来

`FORMATION_TRUTH` 是规划者**逐格目视**那一帧读出来的（与 `roster_spec.py` 的 `ART_TRUTH`
同一个来路：游戏在编队页卡上印的是汉字，人读一遍最可靠），并与接口那一读交叉核过。

它按**拼图行序**（上行左→右、再下行左→右）—— **不是** roster 的卡位序（0 = 最右那张）：
**那两套之间的映射还没定**（规格 §5 那条欠账），所以本文件**不猜**它。
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import sys
from collections.abc import Callable

import numpy as np
from PIL import Image
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _entry in (str(REPO / "src"), str(_HERE)):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

SPEC = "10-规格/资产生成器/编队页名字那一读.md"
BATCH = REPO / "tests" / "帧样本" / "编队页名字竖排"
OLD_BATCH = REPO / "tests" / "帧样本" / "编队页名字"

FORMATION_TRUTH: tuple[str, ...] = (
    "圣聆初雪",
    "凛御银灰",
    "歌蕾蒂娅",
    "荒芜拉普兰德",
    "凯尔希·思衡托",
    "白面鸮",
    "塞雷娅",
    "温蒂",
    "维什戴尔",
    "极境",
)
"""卡位（**拼图行序**：上行左→右 0..4、再下行左→右 5..9）↔ 干员名。"""

SHEET = "拼图_竖排"
RESPONSE = "响应"


# ---------------------------------------------------------------------------
# 骨架
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """一条规格检查（形状与 `countdown_spec.py` 那张表一样）。"""

    ident: str
    spec: str
    claim: str
    probe: Callable[[], str]


_LOADED: dict[str, object] = {}


def _mod(dotted: str):
    """import 一个还不一定存在的模块。不存在 = 这条检查红，不是检查炸了。"""
    if dotted not in _LOADED:
        try:
            _LOADED[dotted] = importlib.import_module(dotted)
        except BaseException as exc:  # noqa: BLE001 —— 缺模块就是这批检查的红法
            raise AssertionError(
                f"`{dotted}` 还不存在：{type(exc).__name__}: {exc}"
            ) from None
    return _LOADED[dotted]


def _index(batch: Path = BATCH) -> dict:
    return json.loads((batch / "index.json").read_text(encoding="utf-8"))


def _lines() -> list[dict]:
    """响应原文里那一串词（**直接读数据**，不碰实现的内部名）。"""
    record = json.loads((BATCH / (RESPONSE + ".json")).read_text(encoding="utf-8"))
    if record.get("code") != "0":
        raise AssertionError(f"响应不是成功：code={record.get('code')!r}")
    return record["data"]["block"][0]["line"]


def _per_slot() -> list[list[str]]:
    """判决链的入口那一步（公开面）：原文 → 词 → 每格摊到的词。"""
    RN = _mod("ark.offline.assets_gen.roster_names")
    record = json.loads((BATCH / (RESPONSE + ".json")).read_text(encoding="utf-8"))
    return RN.assign_words(RN.parse_words(record), RN.VERTICAL, slots=len(FORMATION_TRUTH))


# ---------------------------------------------------------------------------
# 检查
# ---------------------------------------------------------------------------


def check_everything_listed_in_the_index_matches_its_file() -> str:
    """`crops`（两块石头）与 `products`（一局产物）**逐样**都要对得上实际文件。

    「目录里的东西都要能在 index 里找到」的另一半是：**index 里列着的东西都要真的在**，
    而且算出来的哈希与字节数与记的一致 —— 只写一个数、不盯着它，那个数会静默过期。
    """
    parts = []
    for section in ("crops", "products", "manifests"):
        for hit in _index().get(section, ()):
            path = BATCH / hit["file"]
            if not path.is_file():
                raise AssertionError(f"{section} 里列的 {hit['file']} 不在仓里")
            blob = path.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            if digest != hit["source_sha256"]:
                raise AssertionError(
                    f"{hit['file']} 的 sha256 对不上：算出来 {digest} ≠ index.json 里 "
                    f"{hit['source_sha256']}"
                )
            if len(blob) != hit["source_bytes"]:
                raise AssertionError(
                    f"{hit['file']} 的字节数对不上：算出来 {len(blob)} ≠ index.json 里 "
                    f"{hit['source_bytes']}"
                )
            parts.append(f"{path.name} {len(blob)} B {digest[:12]}…")
    if len(parts) < 3:
        raise AssertionError(
            f"index.json 里只列了 {len(parts)} 样，至少该有三样（拼图 · 响应 · 一局产物）"
        )
    return " ｜ ".join(parts)


def check_the_rows_are_unambiguous_from_the_boxes() -> str:
    spans = []
    for item in _lines():
        box = item["word"][0]["location"]
        spans.append((box["top_left"]["y"], box["right_bottom"]["y"]))
    if len(spans) != len(FORMATION_TRUTH):
        raise AssertionError(f"词数是 {len(spans)}，真值有 {len(FORMATION_TRUTH)} 条")
    gaps = [spans[index + 1][0] - spans[index][1] for index in range(len(spans) - 1)]
    if any(gap <= 0 for gap in gaps):
        raise AssertionError(f"有词框纵向重叠：空档 {gaps}")
    if any(spans[index][0] >= spans[index + 1][0] for index in range(len(spans) - 1)):
        raise AssertionError("十个词框的 y 不是单调递增 —— 行序就有歧义了")
    return f"{len(spans)} 个词框 y 单调递增 ｜ 相邻空档 {min(gaps)}–{max(gaps)} px"


def check_the_response_has_ten_words_one_per_slot() -> str:
    per_slot = _per_slot()
    empty = [slot for slot, items in enumerate(per_slot) if not items]
    many = [slot for slot, items in enumerate(per_slot) if len(items) > 1]
    if empty or many:
        raise AssertionError(f"「一格一词」不成立：缺词的格 {empty} ｜ 多于一个词的格 {many}")
    return f"{len(per_slot)} 格一格一词（缺词 {len(empty)} ｜ 多词 {len(many)}）"


def check_every_slot_lands_on_the_truth() -> str:
    RN = _mod("ark.offline.assets_gen.roster_names")
    index = RN.roster_index(RN.load_roster_names())
    per_slot = _per_slot()
    exact = 0
    edited: list[str] = []
    unknown: list[int] = []
    wrong: list[str] = []
    for slot, items in enumerate(per_slot):
        word = items[0] if items else ""
        name, how = RN.decide(word, index) if word else (None, "未知")
        truth = FORMATION_TRUTH[slot]
        if name is None:
            unknown.append(slot)
        elif name != truth:
            wrong.append(f"格{slot} 判成 {name}、真值 {truth}")
        elif how == "改过":
            edited.append(f"格{slot} {word}→{name}")
        else:
            exact += 1
    if unknown:
        raise AssertionError(f"有未知：{unknown}")
    if wrong:
        raise AssertionError(f"有错：{wrong}")
    if len(edited) > 1:
        raise AssertionError(f"「改过」有 {len(edited)} 格，超过验收线 1：{edited}")
    return (
        f"未知 {len(unknown)} ｜ 改过 {len(edited)} ｜ 错 {len(wrong)} ｜ 精确 {exact}"
        + (f" ｜ 改过那一格：{edited[0]}" if edited else "")
    )


def check_the_master_truth_agrees_with_both_copies() -> str:
    for batch in (OLD_BATCH, BATCH):
        copy = tuple(_index(batch)["truth_names"])
        if copy != FORMATION_TRUTH:
            raise AssertionError(f"{batch.name} 那份与主份不一致：{copy}")
    return f"主份与两份副本逐字一致（各 {len(FORMATION_TRUTH)} 条）"


def check_the_closed_set_covers_the_truth() -> str:
    RN = _mod("ark.offline.assets_gen.roster_names")
    names = RN.load_roster_names()
    missing = [name for name in FORMATION_TRUTH if name not in names]
    if missing:
        raise AssertionError(f"全量闭集里缺：{missing}")
    return f"闭集 {len(names)} 名，盖住真值 {len(FORMATION_TRUTH)} 条"


def check_the_two_attempts_are_recorded_truthfully() -> str:
    attempts = _index()["attempts"]
    if len(attempts) < 2:
        raise AssertionError(f"只记了 {len(attempts)} 次尝试")
    billed = [bool(item.get("billed")) for item in attempts]
    if sum(billed) != 1:
        raise AssertionError(f"计费次数不是 1：{billed}")
    if billed[0]:
        raise AssertionError("第一次（握手失败那次）不该计费")
    return (
        f"{len(attempts)} 次尝试 ｜ 计费 {sum(billed)} 次 ｜ "
        f"第一次 {'计费' if billed[0] else '不计费'}、第二次 {'计费' if billed[1] else '不计费'}"
    )


ORDER_BATCH = REPO / "tests" / "帧样本" / "编队页次序"
FACE_DIR = REPO / "assets" / "roster"
FACE_INDEX = FACE_DIR / "faces.json"


def check_the_cross_layer_order_is_one_rule() -> str:
    """合成帧那条**跨层**检查：第 i 个名字配的确实是第 i 张卡的立绘。

    两半各自独立读：**立绘那侧读颜色**（`index.json` 的 `paint.per_slot`），
    **名字那侧读响应里那个词**。两边都读出来之后，走**装载那条路**
    （`load_cardfaces(..., names_from=…)`）把它们配起来 —— 那条路是 `zip`，
    **静默**，所以必须有东西盯着它。
    """
    cardface = _mod("ark.offline.assets_gen.cardface")
    RN = _mod("ark.offline.assets_gen.roster_names")
    index = json.loads((ORDER_BATCH / "index.json").read_text(encoding="utf-8"))
    names = list(index["names_by_slot"])
    paint = [tuple(int(channel) for channel in item) for item in index["paint"]["per_slot"]]
    counts = list(index["region"]["counts"])

    record = json.loads((ORDER_BATCH / "合成响应.json").read_text(encoding="utf-8"))
    per_slot = RN.assign_words(RN.parse_words(record), RN.VERTICAL, slots=len(names))
    read_back = [items[0] if items else "" for items in per_slot]
    if read_back != names:
        raise AssertionError(f"名字那一侧读出来的次序不对：{read_back} ≠ {names}")

    template = json.loads(
        (REPO / "tests" / "帧样本" / "点名" / "编队页.json").read_text(encoding="utf-8")
    )
    geometry = json.loads(json.dumps(template["geometry"]))
    for row, count in zip(geometry["rows"], counts):
        row["count"] = count
    manifest = {
        "frame": "tests/帧样本/编队页次序/合成编队页.png",
        "canvas": template["canvas"],
        "geometry": geometry,
        "names": names,
        "same_run_as_battle_frames": False,
    }
    # 合成清单与合成产物落在 `data/`（**被 gitignore**）下的一小块里：沙箱里
    # 建临时目录会被拒，而这两个文件本来就只是"喂进去就扔"的料。
    scratch = REPO / "data" / "spec_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    manifest_path = scratch / "合成清单.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    product_path = scratch / "合成产物.json"
    product_path.write_text(
        json.dumps({"kind": "game_roster", "names": names}, ensure_ascii=False),
        encoding="utf-8",
    )
    library = cardface.load_cardfaces(manifest_path, names_from=product_path)

    if len(library) != len(names):
        raise AssertionError(f"库切出 {len(library)} 条，名字有 {len(names)} 条")
    for slot, ((name, block), want_name, want_colour) in enumerate(zip(library, names, paint)):
        if name != want_name:
            raise AssertionError(f"第 {slot} 条配的名字是 {name}，按次序该是 {want_name}")
        seen = set(map(tuple, block.reshape(-1, 3).tolist()))
        if seen != {want_colour}:
            raise AssertionError(
                f"第 {slot} 条的立绘不是一种颜色：{sorted(seen)[:3]}（期望 {want_colour}）"
            )
    return (
        f"{len(library)} 条**逐条**按同一套次序配上（名字 {names[0]}→{names[-1]}，"
        f"颜色 {paint[0]}→{paint[-1]}）"
    )


def check_the_face_library_is_content_addressed() -> str:
    """库**自带卡面**，而且每一块的 `id` 就是**它像素的哈希**（内容寻址）。

    内容寻址是这一版填「精英化那一维填不出来」那个洞的办法：同一套立绘第二次遇到
    就是同一个 `id`（直接命中），同一个名字两套立绘就是两条 `id`。所以"id 等于像素
    的哈希"这条**必须**有人盯着，否则内容寻址就只是个说法。
    """
    index = json.loads(FACE_INDEX.read_text(encoding="utf-8"))
    entries = index["faces"]
    if not entries:
        raise AssertionError("库索引里一条都没有")
    names = []
    for item in entries:
        path = FACE_DIR / item["file"]
        if not path.is_file():
            raise AssertionError(f"{item['file']} 不在仓里")
        blob = path.read_bytes()
        if len(blob) != item["bytes"]:
            raise AssertionError(f"{item['file']} 的字节数是 {len(blob)}，索引说 {item['bytes']}")
        if hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise AssertionError(f"{item['file']} 的文件 sha256 与索引对不上")
        pixels = np.asarray(Image.open(path).convert("RGB"))
        if list(pixels.shape) != list(item["shape"]):
            raise AssertionError(
                f"{item['file']} 的形状是 {list(pixels.shape)}，索引说 {item['shape']}"
            )
        raw = pixels.tobytes()
        if hashlib.sha256(raw).hexdigest() != item["pixels_sha256"]:
            raise AssertionError(f"{item['file']} 的像素 sha256 与索引对不上")
        digest = hashlib.blake2b(raw, digest_size=16).hexdigest()
        if digest != item["id"]:
            raise AssertionError(
                f"{item['file']} 的 id 不是它像素的哈希：算出来 {digest} ≠ 索引里 {item['id']}"
            )
        names.append(item["name"])
    return f"{len(entries)} 条，id 逐条等于像素的 blake2b ｜ 名字 {names[0]}…{names[-1]}"


def check_the_battle_product_does_not_land_in_the_asset_tree() -> str:
    """一局产物是**运行期凭据**，闭集才是资产 —— 两者的窝要分开（规格 §5）。"""
    RN = _mod("ark.offline.assets_gen.roster_names")
    default = inspect.signature(RN.write_roster).parameters["directory"].default
    parts = Path(default).parts
    if "assets" in parts:
        raise AssertionError(f"一局产物默认往资产树里写：{default}")
    names_file = Path(RN.NAMES_FILE)
    if names_file.parent.name != "roster":
        raise AssertionError(f"全量闭集该住 assets/roster/，实际 {names_file}")
    return (
        f"一局产物默认落 …{Path(*parts[-2:])}（不在 assets/ 下）｜ "
        f"闭集仍在 {names_file.parent.name}/{names_file.name}"
    )


def check_every_file_in_the_fixture_is_listed_in_the_index() -> str:
    """目录里有的，`index.json` 里也得有（只有 `index.json` 与 `README.md` 例外）。

    这条与 F1 是一对：**F1 管"列了的都在"，这条管"在的都被列了"**。少了后一半，
    往夹具里塞一个文件就能绕开索引 —— 而索引正是那份夹具"说得清自己"的地方。
    """
    listed: set[str] = set()
    for section in ("crops", "products", "manifests"):
        for hit in _index().get(section, ()):
            listed.add(hit["file"])
    present = {
        path.name
        for path in BATCH.iterdir()
        if path.is_file() and path.name not in ("index.json", "README.md")
    }
    missing = sorted(present - listed)
    if missing:
        raise AssertionError(f"目录里有、index.json 里没列：{missing}")
    stale = sorted(listed - present)
    if stale:
        raise AssertionError(f"index.json 里列了、目录里没有：{stale}")
    return f"{len(listed)} 样：列了的都在 ｜ 在的都被列了"


CHECKS: tuple[Check, ...] = (
    Check("F1", f"{SPEC} §4（index 里列着的都要在仓里、都对得上）",
          "index.json 里列的每一样（两块石头 ＋ 一局产物 ＋ 清单）都在仓里，"
          "算出来的 sha256 与字节数与记的一致",
          check_everything_listed_in_the_index_matches_its_file),
    Check("F2", f"{SPEC} §1（名字墨迹行）+ §3（行序无歧义）",
          "接口给的十个词框 y 单调递增、相邻框不重叠 —— 行序没有歧义",
          check_the_rows_are_unambiguous_from_the_boxes),
    Check("F3", f"{SPEC} §2（竖排一格一词）",
          "响应在夹具上重放：十格一格一词，缺词 0、多词 0",
          check_the_response_has_ten_words_one_per_slot),
    Check("F4", f"{SPEC} §3（验收线：未知 0 且 改过 ≤ 1 且 错 0）",
          "逐格判决落在真值上：没有未知、没有错、改过不超过一格",
          check_every_slot_lands_on_the_truth),
    Check("F5", f"{SPEC} §5（真值主份住 tests/spec/）",
          "主份真值与两份夹具副本逐字一致",
          check_the_master_truth_agrees_with_both_copies),
    Check("F6", "资产生成器/卡面与指纹 §名字怎么来（判决对全量闭集）",
          "全量闭集盖住这一帧的十个人（含皮肤名）",
          check_the_closed_set_covers_the_truth),
    Check("F7", f"{SPEC} §6（记账：两次尝试只计费一次）",
          "两次尝试如实记着，且只计费一次",
          check_the_two_attempts_are_recorded_truthfully),
    Check("F8", f"{SPEC} §5（库里的配对按位置来）",
          "跨层：第 i 个名字配的确实是第 i 张卡的立绘（装载那条路真的走一遍）",
          check_the_cross_layer_order_is_one_rule),
    Check("F9", f"{SPEC} §5（库自带卡面）+ 卡面与指纹 §3（累积、不重采）",
          "库索引里每一条：文件对、像素对、而且 id 就是它像素的 blake2b",
          check_the_face_library_is_content_addressed),
    Check("F10", f"{SPEC} §5（一局产物 ≠ 库）",
          "一局产物默认不落进资产树，而全量闭集仍在 assets/roster/",
          check_the_battle_product_does_not_land_in_the_asset_tree),
    Check("F11", f"{SPEC} §4（两块石头都留仓，索引说得清自己）",
          "夹具目录里有的，index.json 里也列了 —— 塞一个文件绕不过索引",
          check_every_file_in_the_fixture_is_listed_in_the_index),
)


def main() -> int:
    failures: list[tuple[Check, str]] = []
    print(f"编队页名字那一读的规格符合性验收 —— {len(CHECKS)} 条\n")
    for check in CHECKS:
        try:
            evidence = check.probe()
        except AssertionError as bad:
            failures.append((check, str(bad)))
            print(f"  ✗ {check.ident}  {check.claim}")
            print(f"      出处 {check.spec}")
            print(f"      证据 {bad}")
        except BaseException as broken:  # noqa: BLE001
            failures.append((check, f"{type(broken).__name__}: {broken}"))
            print(f"  ‼ {check.ident}  {check.claim}")
            print(f"      出处 {check.spec}")
            print(f"      检查本身炸了 {type(broken).__name__}: {broken}")
        else:
            print(f"  ✓ {check.ident}  {check.claim}")
            print(f"      {evidence}")

    print()
    if failures:
        print(f"不成立 {len(failures)} / {len(CHECKS)}：")
        for check, why in failures:
            print(f"  - {check.ident}（{check.spec}）{why}")
        return 1
    print(f"全部成立（{len(CHECKS)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
