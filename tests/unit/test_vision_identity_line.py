"""A3（让「一条绝对线」成立）那一单的单元测试。

**不 import tests/spec**（D-全局-17）。这里钉十件事：

1. **报告有该有的每一块**（七条规格 + 分布 + 锐度 + 敏感性）；
2. **两把自写的尺子复现原来那两把**（掩码那一族对 margin.MaskedRuler，不掩码那一族对
   roster_scores，还有「掩码全开时复现线上那条链」）；
3. 每一行**两个端都有 44 对**，逐对四数在；
4. **两个端点真的是极值、出处指的是真有过的那一对**，比值就是那两个端点算出来的；
5. 没有一行有坏数（都是有限值、分数在 -2…1、错解分 < 1）；
6. **收位置那几行的胜者分数一位不变**（第 2 行的硬要求）；
7. 基线复现 预处理.json 那一批的读数（0.880038 / 0.722272 / 0.900440 / 0.761075，
   读数 11 / 8 块，身份区 19 / 8 块），而且**过线数不落盘** —— 由读者按当下的两条线现算；
8. **不掩码那一行就是线上那条链**（干净端 0.967544 / 0.559715）；
9. **锐度那一行带着它唯一那个「只磨不平移」的对照**，而且对照在两个端上都更差；
10. **分布那一族（不逐像素比）两个端点分不开**，各自丢掉了什么也写在旁边。

**它不重跑那七条规格**：那是 scripts/bench_redcap_identity.py 的活（这台机器上约五分钟）。
这里只核它落下来的那份 JSON，以及便宜的那几个事实。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ark.engine.vision.frame import frame_from_raw
from ark.engine.vision.margin import free_mask
from ark.engine.vision.roster import (
    IDENTITY_BOX_HEIGHT,
    IDENTITY_BOX_WIDTH,
    IDENTITY_BOX_X,
    IDENTITY_BOX_Y,
    ROSTER_AMBIGUITY_MARGIN,
    ROSTER_SOLID_MATCH_SCORE,
)
from ark.offline.bench.batch import load_batch

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "tests" / "帧样本" / "预处理"
BOX = (IDENTITY_BOX_X, IDENTITY_BOX_Y, IDENTITY_BOX_WIDTH, IDENTITY_BOX_HEIGHT)
CARD = (214, 204, 3)


def _report() -> dict:
    return json.loads((BATCH / "绝对线.json").read_text(encoding="utf-8"))


def _rows() -> dict:
    return {row["name"]: row for row in _report()["rows"]}


def _passes_line(entries: list) -> int:
    """只按**分数线**数过线 —— 判定栏不落盘，由这里按当下的常量现算。"""
    return sum(1 for e in entries if e["truth_score"] >= ROSTER_SOLID_MATCH_SCORE)


def _passes_both(entries: list) -> int:
    """**两条线一起**数过线（判定就是它）—— 落盘那份只有 truth_score / wrong_score / margin。"""
    return sum(
        1
        for e in entries
        if e["truth_score"] >= ROSTER_SOLID_MATCH_SCORE
        and e["margin"] >= ROSTER_AMBIGUITY_MARGIN
    )


def test_no_landed_end_carries_a_line_decision() -> None:
    """**由线决定的判定栏不许躺在落盘文件里** —— 逐对只有分数与余量。

    这是全仓那条新规矩在这一单上的把手：线一挪，判定要能离线重算，不必把整批重跑。
    """
    report = _report()
    verdicts = ("passes", "passes_line", "passes_both_rules")

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in verdicts, path + "." + key
                walk(value, path + "." + key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path + "[%d]" % index)

    walk(report, "绝对线.json")
    baseline = report["baseline"]["ends"]["capped"]
    assert "passes_line" not in baseline and "passes_both_rules" not in baseline


def test_the_batch_is_the_same_fixture_the_a2c_one_used() -> None:
    """这一单用的还是那 44 对 · 39 张裁剪那批；取区还是那个 100x100 的框、半径 62。"""
    batch = load_batch(BATCH)
    assert len(batch.cases) == 39
    harvest = json.loads((BATCH / "取材.json").read_text(encoding="utf-8"))
    assert len(harvest["pairs"]) == 44
    mask = _report()["mask"]
    assert tuple(mask["box"]) == BOX
    assert mask["radius"] == 62
    assert mask["pixels"] == int(free_mask(BOX, radius=62).sum()) == 552
    assert mask["of_box"] == 10000


def test_the_report_has_every_block() -> None:
    """七件【做】各落一块，外加判据说明。"""
    report = _report()
    assert set(report) == {
        "note",
        "batch",
        "harvest",
        "library",
        "criterion",
        "chains",
        "mask",
        "model",
        "self_check",
        "windows",
        "baseline",
        "rows",
        "distribution",
        "sharpness_share",
        "sensitivity",
    }
    assert list(report["windows"]) == [0, 1, 2, 4, 8]
    assert report["baseline"]["name"] == "基线-掩码552"
    names = {row["name"] for row in report["rows"]}
    for expected in (
        "基线-掩码552",
        "基线-不掩码10000",
        "取区-环带回填·半径45",
        "取区原点·x-1",
        "取区原点·x+1",
        "锐度-磨参考图·sigma1",
        "锐度-只磨不平移·sigma1",
    ):
        assert expected in names, expected


def test_the_two_self_written_rulers_reproduce_the_reference_chains() -> None:
    """并排的必须是同一条链：自写的 masked 对上 margin.MaskedRuler，
    自写的 full 对上 roster_scores，掩码全开时 masked 再对上 roster_scores。"""
    check = _report()["self_check"]
    assert check["full_vs_roster_scores"] == 0.0
    assert check["masked_vs_margin_masked"] < 1e-5
    assert check["masked_all_open_vs_roster_scores"] < 1e-5


def test_every_row_reports_both_ends_with_44_pairs() -> None:
    """罩着端与干净端并排；每端 44 对，逐对四数在，而且自洽。"""
    report = _report()
    for row in report["rows"]:
        assert set(row["ends"]) == {"capped", "clean"}
        for side in ("capped", "clean"):
            end = row["ends"][side]
            assert end["pairs"] == 44
            assert len(end["per_pair"]) == 44
            assert end["library"] >= 2
            for entry in end["per_pair"]:
                assert -2.0 <= entry["truth_score"] <= 1.0
                assert -2.0 <= entry["wrong_score"] < 1.0
                assert entry["wrong_name"] != entry["truth"]
                assert abs(
                    entry["margin"] - (entry["truth_score"] - entry["wrong_score"])
                ) < 2e-6
                assert math.isfinite(entry["winner_score"])


def test_the_two_endpoints_are_the_extremes_and_the_ratio_follows() -> None:
    """端点就是端点：真值最低分、错解最高分；出处指的是真有过的那一对。"""
    report = _report()
    for row in report["rows"]:
        for side in ("capped", "clean"):
            end = row["ends"][side]
            per = end["per_pair"]
            assert end["min_truth"]["score"] == min(e["truth_score"] for e in per)
            assert end["max_wrong"]["score"] == max(e["wrong_score"] for e in per)
            assert any(
                e["key"] == end["min_truth"]["pair"]
                and e["truth_score"] == end["min_truth"]["score"]
                and e["truth"] == end["min_truth"]["entry"]
                for e in per
            ), row["name"]
            assert any(
                e["key"] == end["max_wrong"]["pair"]
                and e["wrong_score"] == end["max_wrong"]["score"]
                and e["wrong_name"] == end["max_wrong"]["entry"]
                for e in per
            ), row["name"]
            assert (
                abs(
                    end["ratio"]
                    - (1.0 - end["min_truth"]["score"]) / (1.0 - end["max_wrong"]["score"])
                )
                < 1e-6
            ), row["name"]
            assert (
                abs(end["gap"] - (end["min_truth"]["score"] - end["max_wrong"]["score"]))
                < 1e-6
            ), row["name"]
            assert end["min_truth"]["frame"] == end["min_truth"]["pair"].split("#")[0]
            assert end["max_wrong"]["frame"] == end["max_wrong"]["pair"].split("#")[0]


def test_every_row_carries_the_before_state_inline() -> None:
    """验收要「每一行都带改前 / 改后两态」—— 改前那一态就落在同一行的 vs_baseline 里，
    所以只读这一份就能把每一行的两态并排比出来。"""
    report = _report()
    names = {row["name"] for row in report["rows"]}
    for row in report["rows"]:
        assert row["baseline_ref"] in names, row["name"]
        block = row["vs_baseline"]
        assert block["baseline"] == row["baseline_ref"]
        assert set(block["ends"]) == {"capped", "clean"}
        reference = next(
            item for item in report["rows"] if item["name"] == row["baseline_ref"]
        )
        for side in ("capped", "clean"):
            before = block["ends"][side]
            after = row["ends"][side]
            root = reference["ends"][side]
            assert before["min_truth"] == root["min_truth"]["score"]
            assert before["max_wrong"] == root["max_wrong"]["score"]
            assert before["ratio"] == root["ratio"]
            assert before["median_margin"] == root["median_margin"]
            assert abs(before["delta_ratio"] - (after["ratio"] - before["ratio"])) < 1e-6
            assert (
                abs(
                    before["delta_median_margin"]
                    - (after["median_margin"] - before["median_margin"])
                )
                < 1e-6
            )
    # 基线自己那一行的改前参数就是它自己，差当然是 0
    for side in ("capped", "clean"):
        assert report["baseline"]["vs_baseline"]["ends"][side]["delta_ratio"] == 0.0


def test_no_row_has_a_broken_number() -> None:
    """没有一行有坏数（掩码太小那几档最容易塌 —— 分母塌了就会冒出天文数字）。"""
    report = _report()
    for row in report["rows"]:
        for side in ("capped", "clean"):
            end = row["ends"][side]
            for key in ("ratio", "gap", "median_margin", "min_margin", "max_truth"):
                assert math.isfinite(end[key]), (row["name"], side, key)
            assert end["min_margin"] <= end["median_margin"]
            # **判定不落盘**：这一端只有逐对的 truth_score / wrong_score / margin。
            assert _passes_both(end["per_pair"]) <= _passes_line(end["per_pair"])
            assert _passes_line(end["per_pair"]) <= end["pairs"]
            assert "passes_line" not in end and "passes_both_rules" not in end


def test_the_window_rows_keep_the_winner_score_bit_for_bit() -> None:
    """第 2 行的硬要求：收位置之后**胜者的分数一位不变**（它本来就在自己的最优上）。

    对不上就说明第二遍动的不是「位置自由」那一件，这一行整段作废。
    """
    report = _report()
    free = {row["spec"]: row for row in report["rows"] if row["window"] is None}
    windowed = [row for row in report["rows"] if row["window"] is not None]
    assert len(windowed) == 3 * len(report["windows"])
    for row in windowed:
        base = free[row["spec"]]
        assert row["baseline_ref"] == row["spec"]
        for side in ("capped", "clean"):
            for got, root in zip(row["ends"][side]["per_pair"], base["ends"][side]["per_pair"]):
                assert got["winner_score"] == root["winner_score"], row["name"]
                assert got["winner_name"] == root["winner_name"], row["name"]


def test_the_baseline_reproduces_the_a2c_batch() -> None:
    """基线就是 A2c 那一批的读数（同一条链、同一个取区）。

    **也是这一单最要紧的一个数**：按分数线算是 44/44（真值最低分 0.880038 已经在
    0.85 之上，而错解最高分只到 0.722272）—— 而**两条规则一起**只过 `_passes_both(...)` 那么多。
    也就是说这 44 对上过不去的是**余量那一条**，不是分数那一条。
    **这两个数由这一条自己按当下的两条线算**：落盘那份只有逐对的 truth_score /
    margin，一个判定布尔都没有。
    """
    baseline = _report()["baseline"]
    capped = baseline["ends"]["capped"]
    clean = baseline["ends"]["clean"]
    assert abs(capped["min_truth"]["score"] - 0.880038) < 1e-5
    assert capped["min_truth"]["pair"] == "cd10#0"
    assert capped["min_truth"]["entry"] == "斩业星熊"
    assert capped["min_truth"]["slot"] == 0
    assert abs(capped["max_wrong"]["score"] - 0.722272) < 1e-5
    assert capped["max_wrong"]["pair"] == "cd04#8"
    assert abs(clean["min_truth"]["score"] - 0.900440) < 1e-5
    assert abs(clean["max_wrong"]["score"] - 0.761075) < 1e-5
    assert _passes_line(capped["per_pair"]) == 44
    assert _passes_line(clean["per_pair"]) == 44
    # **两条线一起**那个数由这一条自己算（两端本来就该一样）：余量线 0.25 下是 40/44，
    # 也就是说过不去的那 4 对全是**余量那一条**挡的。
    assert _passes_both(capped["per_pair"]) == _passes_both(clean["per_pair"]) == 40
    for end in (capped, clean):
        assert "passes_line" not in end and "passes_both_rules" not in end
    # 互不相同的画面有几块：读数键（诊断文档 §5 那个 11 / 8）与画面键各报一份
    assert (capped["distinct_readings"], clean["distinct_readings"]) == (11, 8)
    assert (capped["distinct_identity_images"], clean["distinct_identity_images"]) == (19, 8)
    assert (capped["distinct_card_images"], clean["distinct_card_images"]) == (19, 13)
    # 基线两个比值都小于 1（这一条是算术，不是「够 / 不够」的结论）
    assert 0.4 < capped["ratio"] < 0.5
    assert 0.4 < clean["ratio"] < 0.5


def test_the_unmasked_baseline_is_the_online_chain() -> None:
    """不掩码那一条是线上那条链（roster_scores）；干净端 0.967544 / 0.559715。

    **罩着端不掩码分不开**（两个端点重叠：真值最低 0.341192 小于 错解最高 0.377541），
    但**逐对的胜者仍然是真值** —— 端点重叠与「这一对排对了」是两件事，别互相读反。
    """
    row = _rows()["基线-不掩码10000"]
    assert row["window"] is None
    clean = row["ends"]["clean"]
    capped = row["ends"]["capped"]
    assert abs(clean["min_truth"]["score"] - 0.967544) < 1e-5
    assert abs(clean["max_wrong"]["score"] - 0.559715) < 1e-5
    assert clean["ratio"] < 0.1
    assert capped["ratio"] > 1.0
    assert capped["winner_is_truth"] is True
    assert _passes_line(capped["per_pair"]) == 0


def test_the_sharpness_row_carries_its_only_blur_control() -> None:
    """第 3 行：把两边磨到可比那一条，**必须**带着「只磨不平移」的对照。

    对照在**两个端上**都比「磨到可比」那一条差（比值更大）—— 光把画面磨软是在扣分。
    """
    rows = _rows()
    treated = rows["锐度-磨参考图·sigma1"]
    control = rows["锐度-只磨不平移·sigma1"]
    assert treated["params"]["art_sigma"] == 1.0
    assert treated["params"]["query_sigma"] == 0.0
    assert control["params"]["art_sigma"] == 1.0
    assert control["params"]["query_sigma"] == 1.0
    assert treated["window"] is None and control["window"] is None
    for side in ("capped", "clean"):
        assert control["ends"][side]["ratio"] > treated["ends"][side]["ratio"]
    share = _report()["sharpness_share"]
    assert len(share["align_offsets_px"]) == 9
    assert share["blur_kernel"] == [0.125, 0.75, 0.125]
    assert "warpAffine" in share["align_method"]
    assert "不位移" in share["blur_method"]
    assert len(share["per_pair"]) == 88
    for side in ("capped", "clean"):
        block = share[side]
        assert block["pairs"] == 44
        assert block["align_gain_median"] >= 0.0
        assert block["align_gain_positive"] <= block["pairs"]
        assert abs(block["blur_gain_median"]) < 0.01
    assert "blur_gain" in share["value_of_sharpness"]


def test_the_distribution_family_is_not_pixelwise_and_does_not_separate() -> None:
    """第 4 行：不逐像素比的那一族，两种取法各一行，**各自丢掉了什么**写在旁边。

    两种取法在**两个端上**都分不开（比值 > 1）—— 与仓库里同族那两次试的结果一致
    （红色占比 · 亮度，都分不开）。
    """
    families = _report()["distribution"]["families"]
    assert len(families) >= 2
    for family in families:
        assert family["name"]
        assert family["drops"] and family["region"]
        for side in ("capped", "clean"):
            end = family["ends"][side]
            assert end["pairs"] == 44
            assert len(end["per_pair"]) == 44
            assert end["ratio"] > 1.0, family["name"]
            assert math.isfinite(end["ratio"])
            assert "布局" in family["drops"]


def test_the_r45_row_is_flagged_where_the_free_winner_is_not_the_truth() -> None:
    """第 5 行（把环带换回来）：**罩着端的自由搜索会认错人**。

    这一行的逐对胜者不是真值的有 7 对（干净端 0 对）。规划者的【卡住】那一条说的就是它：
    这一行的「收位置」第二遍前提不成立 —— 不是第二遍坏了，是**这一块取区本身挑错了人**。
    """
    row = _rows()["取区-环带回填·半径45"]
    assert row["params"]["radius"] == 45
    assert row["window"] is None
    assert row["ends"]["clean"]["winner_is_truth"] is True
    assert row["ends"]["capped"]["winner_is_truth"] is False
    wrong = [
        e["key"]
        for e in row["ends"]["capped"]["per_pair"]
        if e["truth_score"] < e["wrong_score"]
    ]
    assert len(wrong) == 7
    assert row["ends"]["capped"]["ratio"] > 1.0
    assert row["ends"]["clean"]["ratio"] < 0.2


def test_the_sensitivity_block_reports_each_dial() -> None:
    """四档自变量各自的摆动：收位置那个窗、低通强度、掩码半径、取区原点。"""
    sensitivity = _report()["sensitivity"]
    assert set(sensitivity["box_origin_ratio"]) == {"-1", "0", "+1"}
    assert set(sensitivity["radius_ratio"]) == {"45", "62"}
    assert sensitivity["sigma_baseline"]
    assert sensitivity["window_median_margin"]
    for label, block in sensitivity["box_origin_ratio"].items():
        assert set(block) == {"capped", "clean"}
        for side in ("capped", "clean"):
            assert math.isfinite(block[side])
    capped = {block["capped"] for block in sensitivity["box_origin_ratio"].values()}
    assert len(capped) == 3
    windowed = sensitivity["window_median_margin"]
    assert windowed["基线-掩码552±0"]["capped"] > windowed["基线-掩码552±8"]["capped"]


def test_the_model_and_the_chains_are_the_ones_this_batch_uses() -> None:
    """模型还是那两个标定对定的四个参数；两把尺子各自的统计量写明了。"""
    report = _report()
    assert report["model"]["pairs"] == 2
    assert report["model"]["parameters"] == 4
    assert report["model"]["fit_from"] == ["cd14#1 -> cd15#1", "cd22#9 -> cd23#9"]
    assert "MaskedRuler" in report["chains"]["masked"]
    assert "roster_scores" in report["chains"]["full"]
