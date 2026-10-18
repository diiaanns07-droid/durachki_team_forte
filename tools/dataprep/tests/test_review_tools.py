"""vlm_types / vlm_probe / types_review / contact_sheets — без GPU и без реальных данных."""
from __future__ import annotations

import csv

import numpy as np
import pytest


def test_parse_json_variants(dp):
    vt = dp("vlm_types")
    d, ok = vt.parse_json('мусор {"type_id": "V03", "confidence": 0.7,} хвост')
    assert ok and d["type_id"] == "V03"
    d, ok = vt.parse_json('"type_id": "V05", "confidence": 0.4 — без скобок')
    assert not ok and d == {"type_id": "V05", "confidence": "0.4"}


@pytest.mark.parametrize("raw,want", [("V1", "V01"), ("v04 — дверь", "V04"), ("NEW: другое", "NEW"),
                                      ("новый тип", "NEW"), ("", "UNK"), ("V09", "UNK")])
def test_norm_type(dp, raw, want):
    assert dp("vlm_types").norm_type(raw) == want


def test_norm_oop_and_fnum(dp):
    vt = dp("vlm_types")
    assert vt.norm_oop("peak — в разгаре") == "peak" and vt.norm_oop("кассир встаёт") == "other"
    assert vt.fnum("0,5") == 0.5 and np.isnan(vt.fnum(None)) and vt.fnum("x", 0) == 0


@pytest.mark.parametrize("raw,want", [(["yes", 12], ("yes", 12.0)), (["Нет", None], ("no", None)),
                                      ({"answer": "да", "t": "7"}, ("yes", 7.0)), ("maybe", ("unclear", None))])
def test_probe_norm_ans(dp, raw, want):
    assert dp("vlm_probe").norm_ans(raw) == want


def _event(vid, t, typ="V04"):
    return dict(video_id=vid, t_sec=t, type_id=typ, alt_type_id="", what_is_visible="w", actors="a", objects="o",
                new_type="", confidence=0.95, onset_or_peak="peak", event_time_sec=t)


def test_confirmed_type_mapping(dp):
    tr = dp("types_review")
    assert tr.confirmed_type("v03") == "V03" and tr.confirmed_type("NEW") == "V99"
    assert tr.confirmed_type("NONE") == "" and tr.confirmed_type("UNSURE") == "" and tr.confirmed_type(None) == ""


def test_confirmed_type_follows_catalog(dp):
    tr = dp("types_review")
    assert tr.confirmed_type("V09") == "V09"  # V09 есть в каталоге A13
    assert tr.confirmed_type("V10") == ""  # неизвестный id не становится разметкой
    assert tr.confirmed_type("V09", valid={"V01", "V99"}) == ""  # список задаёт каталог, а не код


def test_type_catalog_yaml_fallback_and_missing(dp, tmp_path):
    tc = dp("type_catalog")
    (tmp_path / "norms").mkdir()
    (tmp_path / "norms/catalog.yaml").write_text(
        "version: 1\ntypes:\n  V09:\n    name: \"Личные вещи\"\n  V01:\n    name: \"Посторонний\"\n"
        "  V99:\n    name: \"Иное\"\n", encoding="utf-8")
    types, src = tc.load_types(None, tmp_path)
    assert list(types) == ["V01", "V09", "V99"] and src.endswith("catalog.yaml")
    types, src = tc.load_types(None, tmp_path / "nowhere")
    assert "V09" in types and "V99" in types and src == tc.FALLBACK_SOURCE
    with pytest.raises(FileNotFoundError):
        tc.load_types(tmp_path / "missing.yaml", tmp_path)


def test_label_rows_vlm_never_becomes_ground_truth(dp):
    tr = dp("types_review")
    rows = tr.label_rows([_event("X-1", 22.0), _event("X-2", 10.0)], {}, {})
    assert all(r["type_id"] == "" and r["confirmed"] == 0 and r["source"] == "vlm" for r in rows)
    assert [r["vlm_type_id"] for r in rows] == ["V04", "V04"]


def test_label_rows_confirmed_new_none_and_conflict(dp):
    tr = dp("types_review")
    ev = [_event("X-1", 22.0), _event("X-2", 10.0), _event("X-4", 57.0), _event("X-47", 68.0), _event("X-5", 3.0)]
    conf = {("X-1", 22.0): {"type_id": "V08", "confirmed": "1"},
            ("X-2", 10.0): {"type_id": "NEW", "new_type": "описание", "confirmed": "1"},
            ("X-5", 3.0): {"type_id": "NONE", "confirmed": "1", "notes": "не видно"}}
    rows = {r["video_id"]: r for r in tr.label_rows(ev, conf, {"X-4": "X-47", "X-47": "X-4"})}
    assert rows["X-1"]["type_id"] == "V08" and rows["X-1"]["source"] == "human"
    assert rows["X-2"]["type_id"] == "V99" and rows["X-2"]["new_type"] == "описание"
    assert rows["X-5"]["type_id"] == "" and rows["X-5"]["notes"].startswith("HUMAN:NONE")
    assert rows["X-4"]["notes"].startswith("UNRESOLVED") and rows["X-47"]["type_id"] == ""
    decided = tr.label_rows(ev[2:4], {}, {"X-4": "X-47", "X-47": "X-4"}, {"pair:X-4+X-47": {"decision": "two_events"}})
    assert all(r["notes"].startswith("PAIR_DECISION:two_events") for r in decided)
    unclear = tr.label_rows(ev[2:3], {}, {"X-4": "X-47"}, {"pair:X-4+X-47": {"decision": "unclear"}})
    assert unclear[0]["notes"].startswith("UNRESOLVED")


def test_read_confirmed_only_confirmed_rows(dp, tmp_path):
    tr = dp("types_review")
    p = tmp_path / "export.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:  # браузер пишет с BOM
        w = csv.DictWriter(f, fieldnames=tr.COLS)
        w.writeheader()
        w.writerow(dict(video_id="X-1", t_sec="22", type_id="V08", confirmed=1))
        w.writerow(dict(video_id="X-2", t_sec="10", type_id="", draft_type_id="V01", confirmed=0))
    assert set(tr.read_confirmed(p)) == {("X-1", 22.0)}


def test_contact_sheet_on_synthetic_video(dp, tmp_path):
    cv2 = pytest.importorskip("cv2")
    cs = dp("contact_sheets")
    vid = tmp_path / "synthetic.avi"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 96))
    for i in range(30):  # 3 с, 10 fps
        w.write(np.full((96, 64, 3), i * 8, np.uint8))
    w.release()
    assert cs.parse_times("overview:4", 2.0) == [0.25, 0.75, 1.25, 1.75]
    out = tmp_path / "sheet.jpg"
    size = cs.make_sheet(str(vid), str(out), "0,1,2,9", scale=1.0)  # 9 с — за пределами ролика → пустая клетка
    assert out.exists() and size == (4 * 64 + 3 * 4, 96)
