import pytest

from ml.vlm.parsing import (coerce_conf, coerce_type_id, discover_schema, extract_json, normalize_discover,
                            normalize_verify, regex_fallback_verify, verify_schema)

ALLOWED = {"V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08", "V99"}


@pytest.mark.parametrize("text", [
    '{"visible": true, "type_id": "V05", "confidence": 0.8}',
    'Вот ответ:\n```json\n{"visible": true, "type_id": "V05", "confidence": 0.8}\n```\nГотово.',
    "<think>hmm {not json}</think>{\"visible\": true, \"type_id\": \"V05\", \"confidence\": 0.8}",
    "{'visible': True, 'type_id': 'V05', 'confidence': 0.8}",
    '{visible: true, type_id: "V05", confidence: 0.8,}',
    '{"visible": true, "type_id": "V05", "confidence": 0.8, "observed_facts": ["Кассир держит',
    '{"visible": true, "type_id": "V05", "confidence": 0.8, "observed_facts": [',
])
def test_extract_json_variants(text):
    r = extract_json(text)
    assert r.ok, r
    assert r.data["visible"] is True and r.data["type_id"] == "V05"


@pytest.mark.parametrize("text", ["", "Sorry, I can't", "visible maybe", "[1, 2, 3]", "{{{{"])
def test_extract_json_garbage(text):
    assert not extract_json(text).ok


def test_extract_prefers_largest_object():
    r = extract_json('{"a": 1} и ещё {"visible": false, "type_id": "V01", "confidence": 0.3}')
    assert r.ok and r.data["type_id"] == "V01"


def test_braces_inside_strings_do_not_break():
    r = extract_json('{"visible": true, "uncertain_because": "скобка } внутри", "type_id": "V02"}')
    assert r.ok and r.data["type_id"] == "V02"


@pytest.mark.parametrize("v,exp", [(0.7, 0.7), ("0.7", 0.7), ("70%", 0.7), (85, 0.85), ("high", 0.8),
                                   ("0,6", 0.6), (-0.2, None), (150, None), ("nan", None), (None, None),
                                   (True, None), (float("nan"), None)])
def test_coerce_conf(v, exp):
    assert coerce_conf(v) == exp


def test_coerce_type_id_enum():
    assert coerce_type_id("v5", ALLOWED) == "V05"
    assert coerce_type_id("V-03", ALLOWED) == "V03"
    assert coerce_type_id("V12", ALLOWED) is None  # вне enum — не подменяем молча
    assert coerce_type_id("phone", ALLOWED) is None


def test_normalize_verify_validates_ranges():
    d, err = normalize_verify({"visible": "да", "type_id": "V12", "confidence": 150, "best_frame": 9,
                               "visible_frames": [1, 2, 9], "observed_facts": "Кассир у стола; Телефон в руке"},
                              ALLOWED, n_frames=5)
    assert err == ""
    assert d["visible"] is True
    assert d["type_id"] is None and "type_id_invalid" in d["flags"]
    assert d["confidence"] == 0.5 and d["confidence_missing"] and "confidence_invalid" in d["flags"]
    assert d["best_frame"] is None and "best_frame_out_of_range" in d["flags"]
    assert d["visible_frames"] == [1, 2] and "visible_frames_out_of_range" in d["flags"]
    assert d["observed_facts"] == ["Кассир у стола", "Телефон в руке"]


def test_normalize_verify_requires_visible():
    d, err = normalize_verify({"type_id": "V05"}, ALLOWED)
    assert d is None and err == "missing_visible"


def test_normalize_discover():
    items = ["phone", "other"]
    d, _ = normalize_discover({"findings": [{"item": "phone", "observed_facts": ["Телефон в руке кассира"],
                                             "confidence": 0.9, "frame": 7},
                                            {"item": "unknown_item", "confidence": 1.0},
                                            "garbage"]}, items, n_frames=4)
    assert len(d["findings"]) == 1
    f = d["findings"][0]
    assert f["item"] == "phone" and f["frame"] is None
    assert {"item_not_in_enum", "finding_not_object", "frame_out_of_range"} <= set(d["flags"])
    d2, _ = normalize_discover({"phone": True, "other": False}, items)
    assert [f["item"] for f in d2["findings"]] == ["phone"]
    assert normalize_discover({"foo": 1}, items)[0] is None


def test_regex_fallback():
    d = regex_fallback_verify('visible: yes, type_id: V05, confidence: 0.7 (broken', ALLOWED)
    assert d["visible"] is True and d["type_id"] == "V05" and d["confidence"] == 0.7
    assert d["observed_facts"] == [] and "regex_fallback" in d["flags"]
    assert regex_fallback_verify("nothing here", ALLOWED) is None


def test_schemas_have_enums_and_ranges():
    s = verify_schema(sorted(ALLOWED), 5)
    assert s["properties"]["type_id"]["enum"] == sorted(ALLOWED)
    assert s["properties"]["best_frame"]["maximum"] == 5
    ds = discover_schema(["phone", "other"], 4)
    assert ds["properties"]["findings"]["items"]["properties"]["item"]["enum"] == ["phone", "other"]
