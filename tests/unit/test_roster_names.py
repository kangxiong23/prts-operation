"""编队页名字那一读（第十单）的单元测试。

**它不碰 `screenshots/`**（那目录不进版本库）：要帧的地方一律自己合成一帧，
要响应的地方要么用**夹具里那一份真响应**（`tests/帧样本/编队页名字/`），要么用
**明说合成的**竖排响应（单测构造，不是任何一次真调用的产物）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import ssl
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from ark.engine.vision.frame import CANVAS_2800x1260, Frame
from ark.offline.assets_gen import cardface
from ark.offline.assets_gen import roster_names as RN
from ark.offline.assets_gen.cardface import load_cardfaces, load_formation_page

REPO = Path(__file__).resolve().parents[2]
BATCH = REPO / "tests" / "帧样本" / "编队页名字"
INDEX = json.loads((BATCH / "index.json").read_text(encoding="utf-8"))
TRUTH = tuple(INDEX["truth_names"])
COUNTS = tuple(INDEX["region"]["counts"])
RESPONSE = (BATCH / "响应_横排旧版.json").read_text(encoding="utf-8")
SKINS = ("圣聆初雪", "凛御银灰", "荒芜拉普兰德", "凯尔希·思衡托")


def _synthetic_page() -> Frame:
    """一帧**合成**的编队页：固定种子，所以它自己也是可复现的（不是真截图）。"""
    rng = np.random.default_rng(7)
    pixels = rng.integers(0, 256, (1260, 2800, 3), dtype=np.uint8)
    return Frame(canvas=CANVAS_2800x1260, pixels=np.ascontiguousarray(pixels))


def _location(x0: float, y0: float, x1: float, y1: float) -> dict:
    return {"top_left": {"x": x0, "y": y0}, "right_bottom": {"x": x1, "y": y1}}


def _vertical_text(items: dict) -> str:
    """**合成**一份竖排版面的响应文本：`{格: [词, ...]}`（单测自己造的，不是真响应）。"""
    lines = []
    height = RN.VERTICAL.cell_height
    gap = RN.VERTICAL.gap_y
    for slot in sorted(items):
        words = items[slot]
        top = slot * (height + gap) + 8
        bottom = top + 28
        if len(words) == 1:
            boxes = [_location(20, top, 180, bottom)]
        else:
            boxes = [_location(20, top, 80, bottom), _location(100, top, 180, bottom)]
        lines.append(
            {"word": [{"content": word, "location": box} for word, box in zip(words, boxes)]}
        )
    return json.dumps(
        {"code": "0", "desc": "success", "data": {"block": [{"type": "text", "line": lines}]}},
        ensure_ascii=False,
    )


@pytest.fixture(scope="module")
def roster() -> dict:
    return RN.roster_index(RN.load_roster_names())


@pytest.fixture(scope="module")
def page() -> Frame:
    return _synthetic_page()

@pytest.fixture
def scratch(request: pytest.FixtureRequest) -> Path:
    """**工作区里**的一小块临时地（与别的单测同一套做法）：

    平台的临时目录在本机沙箱下连建它的进程自己都进不去，所以落点显式给在工作区里
    （`data/` 是 gitignored 的运行时数据）。
    """
    path = REPO / "data" / "probe" / "pytest_scratch" / request.node.name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- 取区 -----------------------------------------------------------------


def test_boxes_start_at_the_card_left_edge() -> None:
    boxes = RN.name_boxes(COUNTS)
    assert len(boxes) == 10
    assert boxes[0] == (RN.FORMATION_LEFT0, 577 - RN.PAD_ABOVE, 214, RN.CELL_HEIGHT)
    assert boxes[5][1] == 1081 - RN.PAD_ABOVE
    for index, box in enumerate(boxes):
        assert box[0] == RN.FORMATION_LEFT0 + RN.FORMATION_PITCH * (index % 5)
        assert box[2] == RN.NAME_ART_WIDTH
        assert box[0] + RN.NAME_ART_WIDTH > box[0] + RN.NAME_RIGHT_INSET


def test_boxes_guard_the_counts() -> None:
    with pytest.raises(TypeError):
        RN.name_boxes("55")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RN.name_boxes((5,))
    with pytest.raises(ValueError):
        RN.name_boxes((5, -1))
    with pytest.raises(ValueError):
        RN.name_boxes((5, True))


def test_sheet_presses_everything_but_the_name_dark(page: Frame) -> None:
    sheet = RN.compose_sheet(page, counts=COUNTS)
    height = RN.CELL_HEIGHT
    assert sheet.shape == (10 * (height + RN.CELL_GAP) - RN.CELL_GAP, 214, 3)
    for slot in range(10):
        top = slot * (height + RN.CELL_GAP)
        assert not sheet[top : top + RN.PAD_ABOVE].any(), "上缘以上那几行要全黑"
        assert not sheet[top + height - RN.PAD_BELOW : top + height].any(), "下缘以下那几行要全黑"
        if slot:
            assert not sheet[top - RN.CELL_GAP : top].any(), "格与格之间要全黑，不许白边"
    assert RN.CELL_GAP >= 25 and RN.PAD_ABOVE >= 6 and RN.PAD_BELOW >= 6


def test_sheet_is_deterministic_byte_for_byte(page: Frame) -> None:
    first = RN.sheet_png(RN.compose_sheet(page, counts=COUNTS))
    second = RN.sheet_png(RN.compose_sheet(page, counts=COUNTS))
    assert first == second
    assert RN.sheet_digest(first) == RN.sheet_digest(second)


def test_compose_refuses_a_non_frame() -> None:
    with pytest.raises(TypeError):
        RN.compose_sheet(object(), counts=COUNTS)  # type: ignore[arg-type]


# --- 解析 -----------------------------------------------------------------


def test_parse_words_reads_text_and_centre() -> None:
    text = _vertical_text({0: ["圣聆初雪"]})
    words = RN.parse_words(json.loads(text))
    assert len(words) == 1
    assert words[0][0] == "圣聆初雪"
    assert words[0][1] == pytest.approx(100.0)
    assert words[0][2] == pytest.approx(22.0)


def test_parse_words_refuses_a_nonzero_code() -> None:
    with pytest.raises(RN.OcrRefused):
        RN.parse_words({"code": "10105", "desc": "illegal client_ip"})


def test_parse_words_insists_on_location() -> None:
    bad = {"code": "0", "data": {"block": [{"line": [{"word": [{"content": "白面鸮"}]}]}]}}
    with pytest.raises(ValueError):
        RN.parse_words(bad)


# --- 归一化与距离 ---------------------------------------------------------


def test_normalize_folds_punctuation_fullwidth_and_case() -> None:
    assert RN.normalize("凯尔希·思衡托") == RN.normalize("凯尔希思衡托")
    assert RN.normalize("ＡＢＣ") == "abc"
    assert RN.normalize(" 白 面 鸮 ") == "白面鸮"
    assert RN.normalize("Mon3tr") == RN.normalize("MON3TR")


def test_edit_distance_counts_an_extra_missing_or_wrong_char_as_one() -> None:
    assert RN.edit_distance("白面号", "白面鸮") == 1
    assert RN.edit_distance("白面", "白面鸮") == 1
    assert RN.edit_distance("白面鸮鸮", "白面鸮") == 1
    assert RN.edit_distance("白面", "白面鸮鸮") == 2
    assert RN.edit_distance("塞娅雷", "塞雷娅") == 2


# --- 判决 -----------------------------------------------------------------


def test_decide_exact_hit(roster: dict) -> None:
    assert RN.decide("凛御银灰", roster) == ("凛御银灰", "精确")


def test_decide_one_edit_is_marked_changed(roster: dict) -> None:
    assert RN.decide("白面号", roster) == ("白面鸮", "改过")


def test_decide_two_candidates_is_unknown() -> None:
    index = RN.roster_index(["白面鸮", "白面鸡"])
    assert RN.decide("白面号", index) == (None, "未知")


def test_decide_distance_two_is_unknown(roster: dict) -> None:
    assert RN.decide("塞娅雷", roster) == (None, "未知")


def test_a_merged_word_is_never_cut_into_known_names(roster: dict) -> None:
    merged = "歌蕾蒂娅荒芜拉普兰德凯尔希·思衡托"
    assert RN.decide(merged, roster) == (None, "未知")


def test_a_name_colliding_after_normalization_is_unknown() -> None:
    index = RN.roster_index(["凯尔希·思衡托", "凯尔希思衡托"])
    assert RN.decide("凯尔希思衡托", index) == (None, "未知")


# --- 闸门 -----------------------------------------------------------------


def test_gate_raises_when_a_slot_is_empty_or_crowded(page: Frame, roster: dict, scratch: Path) -> None:
    text = _vertical_text({0: ["圣聆初雪"], 1: ["凛御银灰", "极境"]})
    with pytest.raises(RN.RosterLayoutError) as bad:
        RN.read_names(
            page, counts=COUNTS, transport=lambda _png: text, roster=roster,
            cache_dir=scratch, strict=True,
        )
    assert 1 in bad.value.extra
    assert 0 not in bad.value.missing and 0 not in bad.value.extra
    assert set(bad.value.missing) == set(range(2, 10))


def test_gate_accepts_exactly_one_word_per_slot(page: Frame, roster: dict, scratch: Path) -> None:
    text = _vertical_text({slot: [name] for slot, name in enumerate(TRUTH)})
    reading = RN.read_names(
        page, counts=COUNTS, transport=lambda _png: text, roster=roster,
        cache_dir=scratch, strict=True,
    )
    assert reading.names == TRUTH
    assert reading.counts == {"精确": 10, "改过": 0, "未知": 0}
    assert reading.calls == 1


def test_the_fixture_response_is_a_layout_fault_not_a_verdict_fault(roster: dict) -> None:
    words = RN.parse_words(json.loads(RESPONSE))
    per_slot = RN.assign_words(words, RN.HORIZONTAL_V1, slots=10)
    assert [len(items) for items in per_slot] == [1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
    decisions = [
        RN.decide(items[0], roster) if items else (None, "未知") for items in per_slot
    ]
    exact = sum(1 for (name, how), want in zip(decisions, TRUTH) if how == "精确" and name == want)
    changed = sum(1 for (name, how), want in zip(decisions, TRUTH) if how == "改过" and name == want)
    unknown = sum(1 for name, _ in decisions if name is None)
    wrong = sum(1 for name, _ in decisions if name is not None and name not in TRUTH)
    assert (exact, changed, unknown, wrong) == (6, 1, 3, 0)


# --- 缓存与产物 -----------------------------------------------------------


def test_cache_makes_the_second_read_free(page: Frame, roster: dict, scratch: Path) -> None:
    seen = {"calls": 0}
    text = _vertical_text({slot: [name] for slot, name in enumerate(TRUTH)})

    def transport(_png: bytes) -> str:
        seen["calls"] += 1
        return text

    first = RN.read_names(page, counts=COUNTS, transport=transport, roster=roster,
                          cache_dir=scratch, strict=True)
    second = RN.read_names(page, counts=COUNTS, transport=transport, roster=roster,
                           cache_dir=scratch, strict=True)
    assert (first.calls, second.calls, seen["calls"]) == (1, 0, 1)
    assert (scratch / (first.digest + ".json")).is_file()


def test_no_cache_dir_means_always_calling(page: Frame, roster: dict) -> None:
    seen = {"calls": 0}
    text = _vertical_text({slot: [name] for slot, name in enumerate(TRUTH)})

    def transport(_png: bytes) -> str:
        seen["calls"] += 1
        return text

    reading = RN.read_names(page, counts=COUNTS, transport=transport, roster=roster,
                            cache_dir=None, strict=True)
    assert (reading.calls, seen["calls"]) == (1, 1)


def test_write_roster_refuses_an_unknown(page: Frame, roster: dict, scratch: Path) -> None:
    text = _vertical_text({0: ["圣聆初雪"]})
    reading = RN.read_names(page, counts=COUNTS, transport=lambda _png: text, roster=roster,
                            cache_dir=None, strict=False)
    with pytest.raises(ValueError):
        RN.write_roster(reading, page="x", page_sha256="y", directory=scratch)


def test_write_roster_lands_the_names(page: Frame, roster: dict, scratch: Path) -> None:
    text = _vertical_text({slot: [name] for slot, name in enumerate(TRUTH)})
    reading = RN.read_names(page, counts=COUNTS, transport=lambda _png: text, roster=roster,
                            cache_dir=None, strict=True)
    path = RN.write_roster(reading, page="p", page_sha256="s", directory=scratch)
    assert path.name == reading.digest + ".json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["kind"] == "game_roster"
    assert tuple(record["names"]) == TRUTH
    assert [item["how"] for item in record["slots"]] == ["精确"] * 10
    data = path.read_bytes()
    assert data.endswith(b"}\n"), "落盘要以换行收尾"
    assert b"\r\n" not in data, "落盘必须是裸 LF（D-全局-18）"


# --- 名单资产 -------------------------------------------------------------


def test_the_closed_set_covers_every_truth_name() -> None:
    names = RN.load_roster_names()
    assert len(names) == INDEX["count"] if "count" in INDEX else len(names) > 400
    for name in TRUTH:
        assert name in names, name
    for skin in SKINS:
        assert skin in names, skin


def test_roster_index_maps_normalized_keys(roster: dict) -> None:
    assert roster[RN.normalize("凯尔希·思衡托")] == ("凯尔希·思衡托",)


def test_load_cardfaces_can_eat_a_roster_product(scratch: Path) -> None:
    default = load_cardfaces()
    flipped = [name for name, _ in default][::-1]
    product = scratch / "game.json"
    product.write_text(
        json.dumps({"kind": "game_roster", "names": flipped}, ensure_ascii=False),
        encoding="utf-8",
    )
    eaten = load_cardfaces(names_from=product)
    assert [name for name, _ in eaten] == flipped
    assert len(eaten) == len(default)


def test_load_cardfaces_refuses_the_closed_set(scratch: Path) -> None:
    with pytest.raises(ValueError):
        load_cardfaces(names_from=RN.NAMES_FILE)


# --- 凭据与端点 -----------------------------------------------------------


def test_ask_ocr_refuses_location_off() -> None:
    with pytest.raises(ValueError):
        RN.ask_ocr(b"x", app_id="a", api_key="b", location="false")


def test_credentials_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RN.APP_ID_VAR, raising=False)
    monkeypatch.delenv(RN.API_KEY_VAR, raising=False)
    with pytest.raises(RuntimeError):
        RN.credentials_from_env()
    monkeypatch.setenv(RN.APP_ID_VAR, "id")
    monkeypatch.setenv(RN.API_KEY_VAR, "key")
    assert RN.credentials_from_env() == ("id", "key")


def test_default_endpoint_is_https() -> None:
    assert RN.DEFAULT_ENDPOINT.startswith("https://")


def test_the_fixture_files_are_present() -> None:
    assert (BATCH / "响应_横排旧版.json").is_file()
    assert (BATCH / "拼图_横排旧版.png").is_file()
    assert (BATCH / "index.json").is_file()
    assert INDEX["known_fault"]


# --- 传输阶梯：默认先试 / 只在那一类错误上降级（用户裁定 Q1）--------------

VERTICAL_BATCH = REPO / "tests" / "帧样本" / "编队页名字竖排"
OK = "RAW-原始响应文本"
"""假的那条缝吐的**原文** —— `send_ocr` 不解析它，所以它不必是 JSON。"""


class _FakePost:
    """假的那条缝（用户裁定 Q1：不联网可测）：记下每次是谁被叫的、走的哪条路。"""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.seen: list[bool] = []

    def __call__(self, endpoint, body, headers, *, relaxed, timeout_s):
        self.seen.append(relaxed)
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def _handshake_error() -> Exception:
    """**假**的握手签名错 —— 原文照抄实测那条，但它不联网。"""
    inner = ssl.SSLError(1, "[SSL: WRONG_SIGNATURE_TYPE] wrong signature type (_ssl.c:1000)")
    return urllib.error.URLError(inner)


def _verify_error() -> Exception:
    """另一类 TLS 错：**不该**触发降级（降级救不了它）。"""
    inner = ssl.SSLError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    return urllib.error.URLError(inner)


def test_the_relaxed_string_is_only_a_security_level() -> None:
    assert RN.RELAXED_CIPHERS == "DEFAULT@SECLEVEL=0"


def test_the_relaxed_context_still_verifies_and_still_encrypts() -> None:
    strict = RN._build_context(relaxed=False)
    relaxed = RN._build_context(relaxed=True)
    for context in (strict, relaxed):
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True
    assert len(relaxed.get_ciphers()) > len(strict.get_ciphers())


def test_send_ocr_takes_the_default_path_and_does_not_downgrade() -> None:
    post = _FakePost([OK])
    attempt = RN.send_ocr(b"png", app_id="a", api_key="b", post=post)
    assert post.seen == [False]
    assert attempt.downgraded is False
    assert attempt.reason is None
    assert attempt.first_seconds is None
    assert attempt.raw == OK
    assert attempt.seconds >= 0.0


def test_send_ocr_downgrades_once_on_wrong_signature_type() -> None:
    post = _FakePost([_handshake_error(), OK])
    attempt = RN.send_ocr(b"png", app_id="a", api_key="b", post=post)
    assert post.seen == [False, True]
    assert attempt.downgraded is True
    assert "WRONG_SIGNATURE_TYPE" in (attempt.reason or "")
    assert attempt.reason == str(_handshake_error())
    assert attempt.first_seconds is not None
    assert attempt.raw == OK


def test_send_ocr_does_not_downgrade_on_other_ssl_errors() -> None:
    post = _FakePost([_verify_error(), "never"])
    with pytest.raises(Exception):
        RN.send_ocr(b"png", app_id="a", api_key="b", post=post)
    assert post.seen == [False]
    assert post.script == ["never"]


def test_send_ocr_does_not_retry_a_third_time() -> None:
    post = _FakePost([_handshake_error(), _handshake_error()])
    with pytest.raises(Exception):
        RN.send_ocr(b"png", app_id="a", api_key="b", post=post)
    assert post.seen == [False, True]


def test_send_ocr_refuses_plaintext_endpoint() -> None:
    post = _FakePost(["never"])
    with pytest.raises(ValueError):
        RN.send_ocr(
            b"png", app_id="a", api_key="b", endpoint="http://webapi.xfyun.cn/x", post=post
        )
    assert post.seen == []


def test_ask_ocr_returns_the_raw_text_of_send_ocr() -> None:
    post = _FakePost([_handshake_error(), OK])
    assert RN.ask_ocr(b"png", app_id="a", api_key="b", post=post) == OK
    assert post.seen == [False, True]


# --- 竖排夹具（第十一单）--------------------------------------------------


def test_the_vertical_fixture_sheet_matches_its_record() -> None:
    index = json.loads((VERTICAL_BATCH / "index.json").read_text(encoding="utf-8"))
    record = next(item for item in index["crops"] if item["file"] == "拼图_竖排.png")
    digest = hashlib.sha256((VERTICAL_BATCH / "拼图_竖排.png").read_bytes()).hexdigest()
    assert digest == record["source_sha256"]
    assert index["grid"]["cell"] == [RN.NAME_ART_WIDTH, RN.CELL_HEIGHT]
    assert index["grid"]["gap_y"] == RN.CELL_GAP
    assert index["region"]["counts"] == [5, 5]


def test_the_vertical_fixture_says_where_it_came_from() -> None:
    index = json.loads((VERTICAL_BATCH / "index.json").read_text(encoding="utf-8"))
    assert index["page"]["image"] == "screenshots/run-04/编队界面.png"
    assert index["page"]["sha256"]
    assert index["region"]["rows"] == [[577, 607], [1081, 1112]]
    assert index["truth_names"]
    assert (VERTICAL_BATCH / "README.md").is_file()


# --- 那本账接进读数（用户裁定 2）------------------------------------------


def _vertical_truth_text() -> str:
    return _vertical_text({slot: [name] for slot, name in enumerate(TRUTH)})


def _attempt(*, downgraded: bool, seconds: float = 1.5) -> RN.OcrAttempt:
    return RN.OcrAttempt(
        raw=_vertical_truth_text(),
        downgraded=downgraded,
        reason="<urlopen error [SSL: WRONG_SIGNATURE_TYPE] wrong signature type>" if downgraded else None,
        seconds=seconds,
        first_seconds=0.5 if downgraded else None,
    )


def test_a_plain_text_transport_leaves_no_account(page: Frame, roster: dict) -> None:
    reading = RN.read_names(
        page, counts=COUNTS, transport=lambda _png: _vertical_truth_text(),
        roster=roster, cache_dir=None, strict=True,
    )
    assert reading.attempt is None
    assert reading.downgraded is False
    assert reading.seconds is None


def test_the_account_lands_in_the_reading(page: Frame, roster: dict) -> None:
    attempt = _attempt(downgraded=True, seconds=11.72)
    reading = RN.read_names(
        page, counts=COUNTS, transport=lambda _png: attempt,
        roster=roster, cache_dir=None, strict=True,
    )
    assert reading.attempt is attempt
    assert reading.downgraded is True
    assert reading.seconds == 11.72
    assert reading.counts == {"精确": 10, "改过": 0, "未知": 0}


def test_an_attempt_on_the_default_path_is_not_a_downgrade(page: Frame, roster: dict) -> None:
    attempt = _attempt(downgraded=False)
    reading = RN.read_names(
        page, counts=COUNTS, transport=lambda _png: attempt,
        roster=roster, cache_dir=None, strict=True,
    )
    assert reading.attempt is attempt
    assert reading.downgraded is False
    assert reading.seconds == 1.5


def test_the_account_never_changes_the_reading(page: Frame, roster: dict) -> None:
    text = _vertical_truth_text()
    plain = RN.read_names(
        page, counts=COUNTS, transport=lambda _png: text,
        roster=roster, cache_dir=None, strict=True,
    )
    wrapped = RN.read_names(
        page, counts=COUNTS,
        transport=lambda _png: RN.OcrAttempt(
            raw=text, downgraded=True, reason="x", seconds=9.0,
        ),
        roster=roster, cache_dir=None, strict=True,
    )
    assert plain.names == wrapped.names
    assert plain.slots == wrapped.slots
    assert plain.counts == wrapped.counts
    assert plain.digest == wrapped.digest
    assert plain.png == wrapped.png
    assert (plain.calls, wrapped.calls) == (1, 1)


def test_a_cache_hit_leaves_no_account(page: Frame, roster: dict, scratch: Path) -> None:
    attempt = _attempt(downgraded=True)
    first = RN.read_names(
        page, counts=COUNTS, transport=lambda _png: attempt,
        roster=roster, cache_dir=scratch, strict=True,
    )
    second = RN.read_names(
        page, counts=COUNTS, transport=lambda _png: attempt,
        roster=roster, cache_dir=scratch, strict=True,
    )
    assert (first.calls, first.downgraded, first.seconds) == (1, True, 1.5)
    assert (second.calls, second.attempt, second.downgraded, second.seconds) == (0, None, False, None)


# --- 产物样本（收单 ①）-----------------------------------------------------

PRODUCT = VERTICAL_BATCH / "d9df3cb41184af1792cf330568024b49.json"


def test_the_landed_product_is_named_like_write_roster_would() -> None:
    record = json.loads(PRODUCT.read_text(encoding="utf-8"))
    assert PRODUCT.name == record["digest"] + ".json"
    assert record["digest"] == "d9df3cb41184af1792cf330568024b49"
    assert record["kind"] == "game_roster"
    assert record["calls"] == 1
    assert tuple(record["names"]) == TRUTH
    assert [item["how"] for item in record["slots"]][5] == "改过"
    assert PRODUCT.read_bytes().endswith(b"}\n")


def test_the_landed_product_is_eaten_by_the_roster_seam() -> None:
    """装载那条缝（`_names_from_roster`）吃它 —— 这是 **真产物**，不是单测现造的。

    **为什么走私有那条**（用户裁定 4）：公开的 `load_cardfaces(names_from=…)` 还要一份
    **同帧**的清单（张数对得上），而仓里那份样例清单是 6+5、这一帧是 5+5 —— 现在没有同帧的，
    所以这里直接打 `load_cardfaces` 拿产物时**调的那个函数**。**这不是推荐做法**：
    以后有了同帧清单，该走公开那条；而那两份夹具本来就不是同一帧，拿 A 帧的名字喂 B 帧的
    立绘库**本来就该拒**。
    """
    assert cardface._names_from_roster(PRODUCT) == TRUTH
    with pytest.raises(ValueError):
        cardface._names_from_roster(RN.NAMES_FILE)
