"""第十单的秤：编队页那 10 个名字 —— 用夹具里那一份响应算，**不打接口**。

它回答四件事：取区对不对 · 判决链在**旧横排**那份响应上跑出什么 · 「一格一词」那道闸门
响不响 · 拼图确定性与缓存 1/0。**它不碰网络**（夹具那一段文本就是 transport 的返回值）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ark.offline.assets_gen import roster_names as RN  # noqa: E402
from ark.engine.device.adb import SQUAD_CARD_PITCH  # noqa: E402
from ark.engine.vision import roster as RS  # noqa: E402
from ark.offline.assets_gen import face_library as FL  # noqa: E402
from ark.offline.assets_gen.cardface import (  # noqa: E402
    cut_formation_arts,
    load_cardfaces,
    load_formation_page,
)

BATCH = REPO / "tests" / "帧样本" / "编队页名字"
SCRATCH = REPO / "data" / "probe"
INK_MIN = 150
"""「有墨」那条线 —— **只给做7的记账用**，不是判据（判据只有名单那一条）。"""


def main() -> int:
    index = json.loads((BATCH / "index.json").read_text(encoding="utf-8"))
    truth = index["truth_names"]
    counts = tuple(index["region"]["counts"])
    page = load_formation_page(REPO / index["page"]["image"])
    roster = RN.roster_index(RN.load_roster_names())
    print("全量闭集 %d 名 ｜ 归一化之后 %d 个键" % (len(RN.load_roster_names()), len(roster)))

    print()
    print("### 做7①：重建之后的竖排拼图")
    sheet = RN.compose_sheet(page, counts=counts)
    png = RN.sheet_png(sheet)
    again = RN.sheet_png(RN.compose_sheet(page, counts=counts))
    print("   尺寸 %d x %d ｜ 格 %d 宽 x %d 高 ｜ 行间 %d 深色 ｜ 上留 %d 下留 %d"
          % (sheet.shape[1], sheet.shape[0], RN.NAME_ART_WIDTH, RN.CELL_HEIGHT,
             RN.CELL_GAP, RN.PAD_ABOVE, RN.PAD_BELOW))
    print("   PNG %d 字节 ｜ RAW sha256 %s ｜ 指纹 %s"
          % (len(png), hashlib.sha256(png).hexdigest(), RN.sheet_digest(png)))
    print("   两次拼图逐字节相同：%s" % (png == again))
    (SCRATCH / "g12_sheet_vertical.png").write_bytes(png)

    print()
    print("### 做7②：十格各自的取区")
    boxes = RN.name_boxes(counts)
    for slot, (x, y, w, h) in enumerate(boxes):
        print("   格 %d   x %d..%d（宽 %d）   y %d..%d（高 %d）"
              % (slot, x, x + w - 1, w, y, y + h - 1, h))

    print()
    print("### 做7③：逐格墨迹的左右边缘（相对卡左缘）+ 记账")
    spans = [RN.NAME_SPANS[slot // 5] for slot in range(len(boxes))]
    for slot, (x, y, w, h) in enumerate(boxes):
        at = slot * (h + RN.CELL_GAP) + RN.PAD_ABOVE
        ink = sheet[at : at + spans[slot]].min(axis=2) >= INK_MIN
        cols = [i for i in range(w) if ink[:, i].any()]
        left, right = cols[0], cols[-1]
        note = ""
        if left <= 1:
            note = ("  ← 【记账】这一格可能被切了：墨迹落进取区最左 2 列"
                    "（原始区间 x %d..%d，页坐标 %d..%d）" % (left, right, x + left, x + right))
        print("   格 %d  卡左缘 %4d  墨迹 x %3d..%3d（宽 %3d）%s"
              % (slot, x, left, right, right - left + 1, note))

    print()
    print("### 做7④：取区左边界那一列（压深之后的格内，应该全深）+ 名字以外不许有亮像素")
    for slot, (x, y, w, h) in enumerate(boxes):
        at = slot * (h + RN.CELL_GAP)
        near = page.pixels[y + RN.PAD_ABOVE : y + RN.PAD_ABOVE + spans[slot], x - 1]
        print("   格 %d  取区左边界 x=%d 那一列（格内 y %d..%d）每通道最大 %s ｜ "
              "左邻那一列 x=%d（名字行上）每通道最大 %s"
              % (slot, x, y, y + h - 1, sheet[at : at + h, 0].max(axis=0).tolist(),
                 x - 1, near.max(axis=0).tolist()))
    stray = 0
    height = RN.CELL_HEIGHT
    bright = sheet.min(axis=2) > INK_MIN
    for slot in range(len(boxes)):
        top = slot * (height + RN.CELL_GAP)
        stray += int(bright[top : top + RN.PAD_ABOVE].sum())
        stray += int(bright[top + RN.PAD_ABOVE + spans[slot] : top + height].sum())
        if slot:
            stray += int(bright[top - RN.CELL_GAP : top].sum())
    print("   拼图里压在深色空档上的亮像素：%d 个（该是 0）" % stray)

    text = (BATCH / "响应_横排旧版.json").read_text(encoding="utf-8")
    holder = {"n": 0}

    def transport(png: bytes) -> str:
        holder["n"] += 1
        return text

    print()
    print("### 做2/做3：解析链 —— 夹具那份响应按**旧横排**几何解（strict=False 只记账）")
    reading = RN.read_names(
        page, counts=counts, transport=transport, roster=roster, cache_dir=None,
        grid=RN.HORIZONTAL_V1, strict=False,
    )
    print("   调用 %d 次 ｜ 词 %d 个 ｜ 格 %d 个"
          % (reading.calls, len(RN.parse_words(json.loads(text))), len(reading.slots)))
    for item in reading.slots:
        print("   格 %d  词 %-34s -> %-8s %s"
              % (item.slot, repr(item.word), repr(item.name), item.how))

    print()
    print("### 做4：对真值（真值是规划者给的，裁定 7）")
    exact = changed = unknown = wrong = 0
    for item, want in zip(reading.slots, truth):
        if item.name is None:
            unknown += 1
            verdict = "未知"
        elif item.name != want:
            wrong += 1
            verdict = "错（判成 " + item.name + "，真值 " + want + "）"
        elif item.how == "改过":
            changed += 1
            verdict = "改过 -> " + want
        else:
            exact += 1
            verdict = "对"
        print("   格 %d  真值 %-8s  %s" % (item.slot, want, verdict))
        if item.name is None and item.word:
            candidates = sorted(
                name for key, names in roster.items()
                if abs(len(key) - len(RN.normalize(item.word))) <= 1
                and RN.edit_distance(key, RN.normalize(item.word)) <= 1 for name in names
            )
            print("        距离 ≤1 的候选（%d 个）：%s" % (len(candidates), candidates))
    print("   对 %d ｜ 改过 %d ｜ 未知 %d ｜ 错 %d（合 %d）"
          % (exact, changed, unknown, wrong, exact + changed + unknown + wrong))

    print()
    print("### 一道闸门：「一格一词」不成立就响亮报错（strict=True，产物那条路）")
    try:
        RN.read_names(
            page, counts=counts, transport=transport, roster=roster, cache_dir=None,
            grid=RN.HORIZONTAL_V1, strict=True,
        )
        print("   【不对】它没报错")
    except RN.RosterLayoutError as bad:
        print("   RosterLayoutError：缺词的格 %s ｜ 多于一个词的格 %s"
              % (list(bad.missing), list(bad.extra)))

    print()
    print("### 缓存：同一张拼图再读一遍，调用几次")
    cache = SCRATCH / "g11_cache_demo"
    shutil.rmtree(cache, ignore_errors=True)
    holder["n"] = 0
    one = RN.read_names(page, counts=counts, transport=transport, roster=roster,
                        cache_dir=cache, strict=False)
    two = RN.read_names(page, counts=counts, transport=transport, roster=roster,
                        cache_dir=cache, strict=False)
    print("   第一次 calls=%d ｜ 第二次 calls=%d ｜ transport 一共被叫 %d 次"
          % (one.calls, two.calls, holder["n"]))
    shutil.rmtree(cache, ignore_errors=True)
    print("   （这个演示缓存已删：它存的是**旧版面**的响应，留着会污染这张竖排拼图的答案）")

    print()
    print("### 做9：竖排夹具（第十一单）—— **离线重放**，网络一个字都不用动")
    vertical = REPO / "tests" / "帧样本" / "编队页名字竖排"
    vindex = json.loads((vertical / "index.json").read_text(encoding="utf-8"))
    vtext = (vertical / "响应.json").read_text(encoding="utf-8")
    for item in vindex["crops"]:
        payload = (vertical / item["file"]).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        print("   %-16s %7d 字节  RAW sha256 %s  %s"
              % (item["file"], len(payload), digest,
                 "对上" if digest == item["source_sha256"] else "【对不上】"))
    seen = {"n": 0}

    def replay(png: bytes) -> str:
        seen["n"] += 1
        return vtext

    vreading = RN.read_names(
        page, counts=counts, transport=replay, roster=roster, cache_dir=None,
        grid=RN.VERTICAL, strict=True,
    )
    print("   calls=%d ｜ 词 %d 个 ｜ 十格一格一词：**闸门过了**"
          % (vreading.calls, len(RN.parse_words(json.loads(vtext)))))
    for item in vreading.slots:
        print("   格 %d  词 %-30s -> %-16s %s"
              % (item.slot, repr(item.word), repr(item.name), item.how))
    print("   判决链三样数 %s" % vreading.counts)
    vtruth = vindex["truth_names"]
    right = sum(1 for item, want in zip(vreading.slots, vtruth) if item.name == want)
    bad = sum(1 for item, want in zip(vreading.slots, vtruth)
              if item.name is not None and item.name != want)
    print("   照真值：对 %d ｜ 错 %d ｜ 未知 %d"
          % (right, bad, sum(1 for item in vreading.slots if item.name is None)))
    if list(vtruth) != list(truth) or seen["n"] != 1:
        print("   【红】两份夹具的 truth_names 逐字不一致，或者重放被叫了 %d 次" % seen["n"])
        return 1

    print()
    print("### A2 做3b：**真帧那一次** —— run-04 自己的清单（5 + 5）＋ 一局产物 ⇒ load_cardfaces")
    manifest = vertical / "编队页.json"
    manifest_record = json.loads(manifest.read_text(encoding="utf-8"))
    product = vertical / vindex["products"][0]["file"]
    cards = load_cardfaces(manifest, names_from=product)
    run_page = load_formation_page(REPO / manifest_record["frame"])
    direct = cut_formation_arts(
        run_page, counts=tuple(row["count"] for row in manifest_record["geometry"]["rows"])
    )
    print("   load_cardfaces(names_from=一局产物) 跑通：%d 条 ｜ 直接切出来 %d 块"
          % (len(cards), len(direct)))
    wrong = 0
    for slot, ((name, art), want) in enumerate(zip(cards, vtruth)):
        same_name = name == want
        same_pixels = art.tobytes() == direct[slot].tobytes()
        wrong += 0 if (same_name and same_pixels) else 1
        print("   格 %d  %-12s 真值 %-12s %s ｜ 立绘逐字节 == 第 %d 张卡的立绘：%s"
              % (slot, name, want, "对" if same_name else "【错】", slot, same_pixels))
    print("   逐条配对与真值一致：%s" % ("**是**" if wrong == 0 else "【否 %d 处】" % wrong))

    print()
    print("### A2 验收：**把编队页那一帧拿掉**，只用落下来的库，还认不认得出卡")
    library = FL.load_face_library()
    disk_faces = FL.load_face_images(library)
    print("   库从盘上读（%s）：%d 条 ｜ 次序 %s"
          % (FL.FACES_FILE.name, len(disk_faces), [item[0] for item in disk_faces]))
    identical = all(
        art.tobytes() == direct[slot].tobytes()
        for slot, (name, art) in enumerate(disk_faces) if name == cards[slot][0]
    )
    print("   盘上那块与当初切下来的**逐字节相同**：%s" % identical)
    battle = load_formation_page(REPO / "screenshots" / "run-04" / "干净卡对照.png")
    values = RS.read_squad_identities(battle, disk_faces, pitch=SQUAD_CARD_PITCH)
    exact = unknown = 0
    for index, value in enumerate(values):
        name = RS._exact_name(value)
        exact += 1 if name else 0
        unknown += 0 if name else 1
        print("   卡位 %d -> %s" % (index, name if name else "未知"))
    print("   备战席那一帧（run-04，与编队页**同源**）：精确 %d ｜ 未知 %d（合 %d）"
          % (exact, unknown, len(values)))
    print("   —— 这一趟**一个字节都没读编队页**：库来自 assets/roster/，帧来自备战席。")
    return 0 if (wrong == 0 and exact > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
