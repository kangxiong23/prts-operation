"""A2c（红罩反解之后的「灰」能不能靠像素变换救回来）那一单的单元测试。

**不 import tests/spec**（D-全局-17）。这里钉八件事：

1. **样本能加载**（`load_batch`），而且是「配对」不是随便一堆卡；
2. **取区与掩码**（身份区 · 半径 62 · 552 px · 原点来源）；
3. **标尺行那条"做不到一位不变"的取证** —— `cv2.resize` 不是平移等变的，
   四条标尺行的摆动漫不出界，而且**没有哪一行是下限**；
4. **预处理.json 那份读数表自洽**：行数、逐对齐全、全灰度与两边 lam=0 同值、
   **没有任何一行比「什么都不做」多过线**；
5. **三条对照**（罩着·掩码 / 干净·掩码 / 干净·不掩码）—— 红罩的账与掩码的账要分开；
6. **逐对余量差**（罩着 − 干净，同一块掩码）；
7. **去重那一栏**：几块画面、几块过线、去重键；
8. **`pairs[].clean_*` 是哪把尺子**（线上那条链、不掩码）以及**压缩事实**那一栏。

**它不重跑那 17 行**：那是 `scripts/bench_redcap_preprocess.py` 的活（十几分钟）。这里只核
它落下来的那份 JSON，以及便宜的那几个事实。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ark.engine.vision.frame import frame_from_raw
from ark.engine.vision.margin import free_mask
from ark.engine.vision.redcap import fit_redcap
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
    return json.loads((BATCH / "预处理.json").read_text(encoding="utf-8"))


def _passes(entries: list) -> int:
    """**判定由读者按当下的两条线算** —— 落盘那份只有 score / runner_up / margin。

    这是全仓那条新规矩在这一单上的把手：线一挪，判定要能离线重算，不必把整批重跑。
    """
    return sum(
        1
        for entry in entries
        if entry["score"] >= ROSTER_SOLID_MATCH_SCORE
        and entry["margin"] >= ROSTER_AMBIGUITY_MARGIN
    )


def test_no_landed_row_carries_a_line_decision() -> None:
    """**由线决定的判定栏不许躺在落盘文件里** —— 逐条只有分数与余量。"""
    report = _report()
    verdicts = ("passes", "passes_line", "passes_both_rules", "passes_distinct_images",
                "passes_distinct_combos")

    def walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in verdicts, path + "." + key
                walk(value, path + "." + key)
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, path + "[%d]" % index)

    walk(report, "预处理.json")
    assert report["criterion"]["score"] == ROSTER_SOLID_MATCH_SCORE
    assert report["criterion"]["margin"] == ROSTER_AMBIGUITY_MARGIN


def test_the_batch_loads_as_a_bench_batch() -> None:
    """样本进版本库、能被 `load_batch` 那套路子加载，而且每条都是一整张卡。"""
    batch = load_batch(BATCH)
    assert batch.version == 1
    assert (batch.canvas.width, batch.canvas.height) == (2800, 1260)
    assert batch.facts is None
    assert len(batch.cases) == 39
    for case in batch.cases:
        assert (case.rect.width, case.rect.height) == (CARD[1], CARD[0])
        assert case.rect.y == 1045


def test_the_crops_sit_on_a_per_slot_constant_origin() -> None:
    """原点不是常量网格，而是**这一帧像素读出来的卡心减 102** —— 同一个卡位上是同一个数。"""
    batch = load_batch(BATCH)
    per_slot: dict[int, set[int]] = {}
    for case in batch.cases:
        slot = int(case.name.rsplit("_", 1)[1])
        per_slot.setdefault(slot, set()).add(int(case.rect.x))
    assert sorted(per_slot) == [0, 1, 4, 6, 8, 9]
    assert all(len(values) == 1 for values in per_slot.values())
    # 与常量网格（2594 - slot * 207.5 取整）比：差 0…2 px，所以"用哪一个"必须写清。
    offsets = {
        slot: sorted(values)[0] - int(2594 - slot * 207.5) for slot, values in per_slot.items()
    }
    assert offsets == {0: 0, 1: 0, 4: -1, 6: -1, 8: -1, 9: -1}


def test_the_harvest_is_44_pairs_with_two_calibration_pairs() -> None:
    """配对表：44 对、两个标定对、每个真值都过线上那两条线。"""
    harvest = json.loads((BATCH / "取材.json").read_text(encoding="utf-8"))
    pairs = harvest["pairs"]
    assert len(pairs) == 44
    assert Counter(pair["set"] for pair in pairs) == {"validation": 42, "calibration": 2}
    assert {pair["capped"]["key"] for pair in pairs if pair["set"] == "calibration"} == {
        "cd14#1",
        "cd22#9",
    }
    assert all(pair["truth_confident"] for pair in pairs)
    assert Counter(pair["slot"] for pair in pairs) == {0: 4, 1: 2, 4: 26, 6: 7, 8: 3, 9: 2}


def test_the_model_is_fitted_from_the_two_calibration_pairs_only() -> None:
    """模型只吃那两个标定对 —— 四个参数、残差在那个量级。"""
    harvest = json.loads((BATCH / "取材.json").read_text(encoding="utf-8"))
    blocks = {}
    for case in load_batch(BATCH).cases:
        frame = frame_from_raw(case.raw())
        x0, y0 = int(case.rect.x), int(case.rect.y)
        blocks[case.name] = np.ascontiguousarray(
            frame.pixels[y0 : y0 + CARD[0], x0 : x0 + CARD[1]]
        )
    pairs = [
        (
            blocks[Path(pair["capped"]["file"]).stem],
            blocks[Path(pair["clean"]["file"]).stem],
        )
        for pair in harvest["pairs"]
        if pair["set"] == "calibration"
    ]
    assert len(pairs) == 2
    model = fit_redcap(pairs)
    assert model.parameters == 4
    assert model.pairs == 2
    assert 0.15 < model.slope < 0.25
    assert 8.0 < model.residual < 11.0
    assert model.pixels == 2 * 22752


def test_the_mask_is_the_countdown_disk_at_radius_62() -> None:
    """取区：身份区里排掉倒计时圆盘。半径 62 是钉死的，所有行同一个。"""
    assert int(free_mask(BOX, radius=62).sum()) == 552
    assert int(free_mask(BOX, radius=60).sum()) == 748
    assert int(free_mask(BOX, radius=64).sum()) == 402


def test_the_report_rows_are_all_there_and_pairwise_complete() -> None:
    """17 行并排、行行 44 对、逐对四样都在；对照行就是第 0 行。"""
    report = _report()
    rows = report["rows"]
    assert len(rows) == 17
    assert rows[0]["name"] == "对照"
    assert report["baseline"]["name"] == "对照"
    assert [entry["score"] for entry in report["baseline"]["per_pair"]] == [
        entry["score"] for entry in rows[0]["per_pair"]
    ]
    for row in rows:
        assert "error_ratio_median" not in row, row["name"]
        assert len(row["per_pair"]) == len(report["pairs"])
        for entry in row["per_pair"]:
            assert -1.0 <= entry["score"] <= 1.0
            assert abs(entry["margin"] - (entry["score"] - entry["runner_up"])) < 2e-6
            # **判定不落盘**：逐对只有 score / name / runner_up / margin，
            # 过没过线由这里按当下的两条线算。
            assert "passes" not in entry
    assert len(report["pairs"]) == 44


def test_nothing_beats_doing_nothing() -> None:
    """**这一单的答案**：没有任何一行比「什么都不做」多过线。

    （判据是最高分过 0.85 **而且**余量拉开 0.25；余量是这一单要救的那一半。）
    """
    report = _report()
    baseline = _passes(report["rows"][0]["per_pair"])
    assert baseline > 0
    for row in report["rows"]:
        assert _passes(row["per_pair"]) <= baseline, row["name"]


def test_the_grey_row_is_the_same_computation_as_both_sides_lambda_zero() -> None:
    """全灰度与「两边都降 lam=0」是同一次计算 —— 两行的逐对分数必须逐位相同。"""
    report = _report()
    rows = {row["name"]: row for row in report["rows"]}
    grey = rows["全灰度"]["per_pair"]
    zero = rows["降饱和-两边 lam=0"]["per_pair"]
    assert len(grey) == len(zero)
    for one, other in zip(grey, zero):
        assert one["score"] == other["score"]
        assert one["margin"] == other["margin"]


def test_desaturation_never_helps_at_any_level_or_placement() -> None:
    """降饱和度：三种放法、三档，**没有一档把中位余量抬上去** —— 颜色是活的。"""
    report = _report()
    rows = {row["name"]: row for row in report["rows"]}
    base = rows["对照"]["median_margin"]
    for placement in ("只参考图", "只查询图", "两边"):
        star = rows["降饱和-%s lam=lam*" % placement]["median_margin"]
        slope = rows["降饱和-%s lam=slope" % placement]["median_margin"]
        zero = rows["降饱和-%s lam=0" % placement]["median_margin"]
        assert star <= base + 1e-9, placement
        assert slope <= star + 1e-9, placement
        assert zero <= slope + 1e-9, placement


def test_the_invariance_probe_says_resize_is_not_shift_exact() -> None:
    """**标尺行那条门做得到不到**：做不到，而且原因是 `cv2.resize` 不是平移等变的。

    整体仿射在数学上确实会被尺子约掉（它先减均值、再除能量），但尺子**先 resize**：
    把整块画面加一个整数再放大、与先放大再加那个整数，差 ±1 个灰度级。所以
    "一位不变"在不换尺子的前提下做不到。读数落在 `预处理.json` 的 `invariance` 里。
    """
    report = _report()
    invariance = report["invariance"]
    deviations = invariance["resize_deviation_u8"]
    assert len(deviations) == 11
    assert invariance["resize_shift_exact"] is False
    assert max(deviations) == 1
    rows = {row["name"]: row for row in report["rows"]}
    evidence = rows["取证-填零+整数平移（只查询图）"]["delta_vs_compare"]
    assert evidence["compare_to"] == "取证-填零对照（只查询图）"
    assert 0.0 < evidence["max_abs"] < 1e-3


def test_the_ruler_rows_move_by_rounding_not_by_pixels() -> None:
    """四条「标尺」行的摆动钉在这条界里 —— 而且**没有哪一行是下限**。

    界取 **3e-3**：这一单里**真正动过像素**的那些行（降饱和那九条）最小也动了 4.976e-3，
    所以 3e-3 把「取整 / 夹 / resize 那一份」与「真动」分开。而
    `标尺-整体仿射（只查询图）` 比 `取证-填零+整数平移` 还小 —— 所以那条也不是下限，
    它们只是同一个量级的样本（上一版把它写成了下限，这一版收回）。
    """
    report = _report()
    rows = {row["name"]: row for row in report["rows"]}
    for name in (
        "标尺-整体仿射（两边）",
        "标尺-整体仿射（只参考图）",
        "标尺-整体仿射（只查询图）",
        "标尺-整体压对比（两边）",
    ):
        assert rows[name]["compare_to"] == 0, name
        assert 0.0 < rows[name]["delta_vs_compare"]["max_abs"] < 3e-3, name
    assert (
        rows["标尺-整体仿射（只查询图）"]["delta_vs_compare"]["max_abs"]
        < rows["取证-填零+整数平移（只查询图）"]["delta_vs_compare"]["max_abs"]
    )
    for row in report["rows"]:
        if row["name"].startswith("降饱和-"):
            assert row["delta_vs_compare"]["max_abs"] > 3e-3, row["name"]


def test_the_error_column_is_measured_on_the_masked_pixels() -> None:
    """误差那一栏在**排掉倒计时图形之后**的像素上量：逐对逐通道 rms / std / 比值。"""
    report = _report()
    assert len(report["error"]) == 44
    mask_pixels = report["mask"]["pixels"]
    assert mask_pixels == 552
    for entry in report["error"]:
        assert len(entry["rms"]) == 3
        assert len(entry["std"]) == 3
        assert len(entry["ratio"]) == 3
        for channel in range(3):
            assert entry["rms"][channel] > 0.0
            assert abs(
                entry["ratio"][channel] - entry["rms"][channel] / entry["std"][channel]
            ) < 1e-4


def _controls() -> dict:
    """三条对照按名字收成一张表 —— 每一条都必须写清它的取区与它走的链。"""
    table = {}
    for control in _report()["controls"]:
        assert control["region"] and control["chain"]
        assert control["of"] == 44
        assert len(control["per_pair"]) == 44
        identity = control["identity_region"]
        assert identity["distinct_images"] <= control["distinct_images"]
        assert _passes(control["per_pair"]) <= control["of"]
        for entry in control["per_pair"]:
            # **判定不落盘**：逐对只有 score / name / runner_up / margin，
            # 过没过线由这里按当下的两条线算。
            assert "passes" not in entry
        table[control["name"]] = control
    return table


def test_the_three_controls_separate_the_cap_from_the_mask() -> None:
    """三条对照并排：罩着·掩码 / 干净·掩码 / 干净·不掩码。

    **红罩的账与掩码的账必须分开**：真正扔掉信息的是**掩码本身**（552 / 10000 = 5.5%），
    它在相关之前就扔掉了。**这不是「红罩卡认不出来」** —— 干净端不掩码那一条 44/44 全过。
    """
    controls = _controls()
    assert list(controls) == ["罩着·掩码", "干净·掩码", "干净·不掩码"]
    capped = controls["罩着·掩码"]
    masked = controls["干净·掩码"]
    full = controls["干净·不掩码"]
    # **过线数不落盘**：下面这三个数由这里按当下的两条线从逐对 score / margin 现算。
    # 余量线还在 0.30 时前两条各是 31/44；挪到 0.25 之后是 40/44 —— 挪回来的正是
    # 「分数够、余量差一点」那 9 对。**掩码那一刀仍然很贵**（见下面中位余量）。
    assert (capped["of"], masked["of"], full["of"]) == (44, 44, 44)
    assert _passes(capped["per_pair"]) == 40
    assert _passes(masked["per_pair"]) == 40
    assert _passes(full["per_pair"]) == 44
    assert abs(masked["median_margin"] - 0.4001) < 1e-3
    assert abs(full["median_margin"] - 0.5932) < 1e-3
    # 掩码那一刀很贵，红罩那一层几乎不花钱。
    assert full["median_margin"] - masked["median_margin"] > 0.1
    assert abs(masked["median_margin"] - capped["median_margin"]) < 0.05
    # 干净端不掩码就是线上那条链：它必须复现 取材.json 的 clean_*。
    for entry, pair in zip(full["per_pair"], _report()["pairs"]):
        assert abs(entry["score"] - pair["clean_score"]) < 1e-6
        assert abs(entry["margin"] - pair["clean_margin"]) < 1e-6


def test_the_carry_from_the_cap_is_nearly_nothing() -> None:
    """逐对余量差（罩着 − 干净，同一块掩码）—— 红罩对这把尺子几乎不留痕。"""
    report = _report()
    carry = report["delta_capped_minus_clean_masked"]
    assert carry["of"] == 44 and len(carry["per_pair"]) == 44
    assert -0.10 < carry["min"] < 0.0 < carry["max"] < 0.10
    assert abs(carry["median_distinct_images"]) < 0.01
    for entry in carry["per_pair"]:
        assert (
            abs(entry["delta"] - (entry["capped_margin"] - entry["clean_masked_margin"]))
            < 2e-6
        )


def test_the_dedup_column_says_how_many_distinct_images() -> None:
    """去重那一栏：几块画面 · 几块过线 —— 「40/44」里有一大份是重复时段的贡献。"""
    report = _report()
    redundancy = report["redundancy"]
    assert "sha256" in redundancy["dedup_key"]
    assert redundancy["crops"] == 39
    assert redundancy["distinct_crops"] == 32
    assert redundancy["pairs"] == 44
    assert redundancy["capped_images"] == 19
    assert redundancy["clean_images"] == 13
    assert "sha256" in redundancy["dedup_key_identity_region"]
    assert redundancy["distinct_pair_images"] == 29
    for side, expected in (("capped", 19), ("clean", 13)):
        groups = redundancy[side + "_groups"]
        assert len(groups) == expected, side
        assert len({group["key"] for group in groups}) == expected, side
        assert sum(group["pairs"] for group in groups) == 44, side
        assert all(len(group["key"]) == 12 and group["example"] for group in groups), side
    controls = _controls()
    # 整块卡面那个键
    assert controls["罩着·掩码"]["distinct_images"] == 19
    assert controls["干净·掩码"]["distinct_images"] == 13
    assert controls["干净·不掩码"]["distinct_images"] == 13
    # **按块算的过线数不落盘**：文件里只有每对的分数与余量，没有画面分组，
    # 所以这里只钉块数 —— 罩着端 19 块、干净端 13 块（重复时段把 44 对压到这么多）。
    assert [control["distinct_images"] for control in controls.values()] == [19, 13, 13]
    # 身份区那个键（那把尺子真正看到的那一块）—— 干净端比整卡少。
    assert redundancy["capped_images_identity_region"] == 19
    assert redundancy["clean_images_identity_region"] == 8
    assert controls["罩着·掩码"]["identity_region"]["distinct_images"] == 19
    assert controls["干净·掩码"]["identity_region"]["distinct_images"] == 8
    assert controls["干净·不掩码"]["identity_region"]["distinct_images"] == 8
    masked_identity = controls["干净·掩码"]["identity_region"]
    full_identity = controls["干净·不掩码"]["identity_region"]
    # 按块算的过线数不落盘；块数钉在这里，判别力那条读在下面两行的中位余量上。
    assert masked_identity["distinct_images"] == 8
    assert full_identity["distinct_images"] == 8
    assert abs(masked_identity["median_margin_distinct_images"] - 0.3483) < 1e-3
    assert abs(controls["罩着·掩码"]["identity_region"]["median_margin_distinct_images"]
               - 0.2813) < 1e-3
    # 余量线还在 0.30 时这一条是 `<`（红罩那一档按块算过不了）；挪到 0.25 之后
    # 0.2813 过了线 —— **这一条钉的是「线的位置」，不是「判别力」的结论**。
    assert controls["罩着·掩码"]["identity_region"]["median_margin_distinct_images"] > (
        ROSTER_AMBIGUITY_MARGIN
    )
    for row in report["rows"]:
        assert row["distinct_images"] == 19
        assert row["median_margin_distinct_images"] <= row["median_margin"] + 1.0


def test_the_clean_readings_say_which_ruler_they_came_from() -> None:
    """`pairs[].clean_*` 是**线上那条链（不掩码）**的读数 —— 这个坑必须堵住。

    按「这把尺子（掩码）在干净端的读数」去读那一栏，会得出**完全相反**的结论：
    0.5932（不掩码）读成 0.4001（掩码），正好把「掩码很贵、红罩几乎不花钱」看反。
    """
    report = _report()
    assert "10000" in report["pairs_note"]
    assert "552" in report["pairs_note"]
    for entry in report["pairs"]:
        assert "roster_scores" in entry["clean_read_by"]
        assert "不掩码" in entry["clean_read_by"]
        assert "MaskedRuler" in entry["clean_masked"]["read_by"]
        assert "552" in entry["clean_masked"]["read_by"]
    controls = _controls()
    assert controls["干净·不掩码"]["median_margin"] > (
        controls["干净·掩码"]["median_margin"] + 0.1
    )


def test_the_compression_field_records_three_regions() -> None:
    """罩子不是一块均匀的滤镜：三个取区、三种有效压缩，各自的取区都写明了。"""
    report = _report()
    regions = {region["name"]: region for region in report["compression"]["regions"]}
    assert set(regions) == {"整卡立绘区", "身份区（这把尺子的取区）", "职业图标区"}
    for region in report["compression"]["regions"]:
        assert region["region"] and region["pairs"] and region["source"]
    assert abs(regions["整卡立绘区"]["values"]["slope"] - report["model"]["slope"]) < 1e-12
    identity = regions["身份区（这把尺子的取区）"]["values"]
    icon = regions["职业图标区"]["values"]
    assert identity["slope_min"] <= identity["slope_median"] <= identity["slope_max"]
    assert identity["std_ratio_min"] <= identity["std_ratio_median"] <= identity["std_ratio_max"]
    assert 0.10 < identity["slope_median"] < 0.20
    assert 0.10 < identity["std_ratio_median"] < 0.20
    assert abs(icon["slope_min"] - 0.43) < 1e-3
    assert abs(icon["slope_max"] - 0.6142) < 1e-3
    assert len(icon["slope"]) == 12
    # 三个数真的不一样：职业图标那一块比身份区松得多。
    assert icon["slope_min"] > identity["slope_max"]
    assert report["model"]["slope"] > identity["slope_max"]
