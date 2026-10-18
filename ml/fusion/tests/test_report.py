"""Описания без домыслов, нормы (verified), строгий ответ жюри, UI-отчёт, полнота анализа (error vs clean)."""
from __future__ import annotations

import json
import re

import pytest

import ml.fusion.core as core
from ml.fusion import apply_completeness, fuse, result_to_jury, result_to_ui_report
from ml.fusion.catalog import FALLBACK_TYPES, is_verified, resolve_catalog, rule_refs
from ml.fusion.config import load_config
from ml.fusion.describe import build_description, clean_facts, format_time, has_forbidden
from ml.fusion.serialize import event_from_dict, event_to_dict, result_to_api

from .conftest import FakeClip, ev

CFG = load_config()
SENTENCE = re.compile(r"[^.!?…]+[.!?…]")


def test_description_structure_and_no_forbidden_words():
    facts = ["кассир украл пачку денег", "посетитель находится за кассовым столом",
             "умышленно отвернулся от камеры", "посетитель находится за кассовым столом"]
    text = build_description("V01", FALLBACK_TYPES["V01"], facts, 22.4, CFG)
    assert not has_forbidden(text, CFG)
    for word in ("украл", "умышленно", "мошенничество", "кража"):
        assert word not in text.lower()
    assert "посетитель находится за кассовым столом" in text
    assert text.endswith("Требуется проверка человеком.")
    assert "возможного нарушения" in text
    assert 1 <= len(SENTENCE.findall(text)) <= 3
    assert "00:22,4" in text


def test_template_from_catalog_and_safe_placeholders():
    info = {"name": "Тест", "description_template": "На {t} зафиксировано: {facts}. Ссылка {unknown}"}
    text = build_description("V01", info, ["дверь открыта"], 65.0, CFG)
    assert text.startswith("На 01:05 зафиксировано: дверь открыта.")
    assert "{" not in text and text.endswith("Это признаки возможного нарушения — требуется проверка человеком.")


def test_template_with_forbidden_sentence_is_dropped():
    info = {"name": "Тест", "description_template": "Это мошенничество кассира. На {t}: {facts}."}
    text = build_description("V99", info, ["клиент у сейфа"], 5.0, CFG)
    assert "мошенни" not in text.lower() and "клиент у сейфа" in text


def test_word_boundary_does_not_overmatch():
    assert not has_forbidden("во дворе видна дверь, ворота закрыты", CFG)
    assert has_forbidden("вор у кассы", CFG)
    assert clean_facts(["Дверь открыта.", "дверь открыта", ""], CFG) == ["Дверь открыта"]


def test_only_verified_norm_cited_in_description(catalog, no_frames):
    res = fuse(FakeClip(), [ev("V01", t=5.0), ev("V05", t=15.0, facts=["кассир держит телефон у уха"])], [],
               catalog, "job")
    by_type = {v.type_id: v for v in res.violations}
    assert "(норма: Тестовый документ, п. 1)" in by_type["V01"].description
    assert "норма" not in by_type["V05"].description           # verified: false → не цитируем
    assert by_type["V05"].rules and by_type["V05"].rules[0].verified is False


def test_verified_strict_parsing(catalog):
    assert is_verified(True) and is_verified("true") and not is_verified("false") and not is_verified(None)
    types, _ = resolve_catalog(catalog)
    assert rule_refs(types, "V99")[0].verified is False      # verified: "false" строкой
    bad = {"types": {"V01": {"name": "x", "rules": [{"document": "", "url": ""}, "мусор",
                                                    {"document": "Д", "url": "https://adilet.zan.kz/x"}]}}}
    refs = rule_refs(resolve_catalog(bad)[0], "V01")
    assert len(refs) == 1 and refs[0].clause is None and refs[0].verified is False


def test_jury_serializer_strict_schema(catalog):
    res = fuse(FakeClip(), [ev("V01", t=5.0)], [], catalog, "job-1")
    out = result_to_jury(res)
    json.dumps(out)
    assert {"job_id", "status", "verdict", "violations"} <= set(out)
    v = out["violations"][0]
    assert {"type", "timecode_sec", "description", "evidence_frames"} <= set(v)
    assert isinstance(v["timecode_sec"], float) and v["evidence_frames"]
    for f in v["evidence_frames"]:
        assert set(f) == {"timestamp_sec", "image_base64"} or {"timestamp_sec", "image_base64"} <= set(f)
    try:
        import contracts.serialize  # noqa: F401
    except ImportError:  # локальный fallback — ровно поля ТЗ
        assert set(out) == {"job_id", "status", "verdict", "violations"}
        assert set(v) == {"type", "timecode_sec", "description", "evidence_frames"}


def test_api_and_ui_report_keep_norm_status(catalog, no_frames):
    res = fuse(FakeClip(), [ev("V05", t=5.0, facts=["кассир держит телефон"])], [], catalog, "job")
    api = result_to_api(res)
    json.dumps(api)
    assert api["violations"][0]["rules"][0]["verified"] is False
    ui = result_to_ui_report(res)
    json.dumps(ui)
    assert ui["violations"][0]["rules"][0]["status_ru"].startswith("НЕ подтверждено")
    assert "review_hints" in ui["ui"] and ui["disclaimer"]


def test_error_vs_clean(catalog, no_frames):
    clean = apply_completeness(fuse(FakeClip(), [], [], catalog, "a"),
                               {"ingest": "ok", "detect": "ok", "rules": "ok", "temporal": "ok", "vlm": "ok"})
    assert clean.status == "done" and clean.verdict == "no_violation" and clean.meta["analysis_complete"]

    broken = apply_completeness(fuse(FakeClip(), [], [], catalog, "b"),
                                {"ingest": "ok", "detect": "failed: CUDA OOM", "rules": "ok"})
    assert broken.status == "failed" and broken.verdict is None and broken.violations == []
    assert broken.meta["failure"]["failed_stages"] == ["detect"]
    assert result_to_jury(broken)["verdict"] is None

    missing = apply_completeness(fuse(FakeClip(), [], [], catalog, "c"), {"ingest": "ok"})
    assert missing.status == "failed" and set(missing.meta["failure"]["failed_stages"]) == {"detect", "rules"}


def test_incomplete_but_found_violation_kept(catalog, no_frames):
    res = apply_completeness(fuse(FakeClip(), [ev(t=5.0)], [], catalog, "d"),
                             {"ingest": "ok", "detection": {"status": "error", "error": "timeout"}, "rules": "ok"})
    assert res.status == "done" and res.verdict == "violation" and res.meta["analysis_complete"] is False
    assert res.meta["ui"]["incomplete"]["failed_stages"] == ["detect"]


def test_optional_stage_failure_is_degraded_not_failed(catalog, no_frames):
    res = apply_completeness(fuse(FakeClip(), [], [], catalog, "e"),
                             {"ingest": True, "detect": True, "rules": True, "temporal": "timeout", "vlm": False})
    assert res.status == "done" and res.verdict == "no_violation" and res.meta["degraded"] == ["temporal", "vlm"]


def test_emergency_mode_never_hidden_clean(catalog, monkeypatch, no_frames):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(core, "fuse_scores", boom)
    res = fuse(FakeClip(), [], [], catalog, "f")
    assert res.status == "failed" and res.verdict is None
    res2 = fuse(FakeClip(), [ev(t=5.0, conf=0.9)], [], catalog, "g")
    assert res2.status == "done" and res2.verdict == "violation" and res2.meta["analysis_complete"] is False


def test_event_dict_roundtrip():
    e = ev("V03", t=7.0, bboxes={"safe_open": (0.1, 0.2, 0.3, 0.4)}, track_ids=[3], extra={"k": 1})
    back = event_from_dict(json.loads(json.dumps(event_to_dict(e))))
    assert back.type_id == "V03" and back.bboxes["safe_open"] == (0.1, 0.2, 0.3, 0.4) and back.track_ids == [3]


@pytest.mark.parametrize("t,expected", [(0.0, "00:00"), (22.4, "00:22,4"), (125.0, "02:05")])
def test_format_time(t, expected):
    assert format_time(t) == expected


@pytest.mark.parametrize("claim", ["кассир пересчитала 50 000 тенге", "сумма операции не совпадает",
                                   "проверяет подлинность купюр", "лицо без допуска", "неуполномоченное лицо у сейфа",
                                   "сделал это с умыслом", "намеренно отвернулся"])
def test_no_amount_authenticity_permission_intent(claim):
    text = build_description("V01", FALLBACK_TYPES["V01"], [claim, "человек находится за кассовым столом"], 5.0, CFG)
    assert claim not in text and "человек находится за кассовым столом" in text and not has_forbidden(text, CFG)


def test_a13_template_keeps_honest_norm_sentence():
    info = {"name": "Посторонний в зоне кассира",
            "description_template": "{t}: наблюдение для проверки человеком — {facts}. Нормативное основание не "
                                    "подтверждено; по видео нельзя установить допуск и личность человека."}
    text = build_description("V01", info, ["человек находится за кассовым столом"], 22.4, CFG)
    assert "Нормативное основание не подтверждено" in text and "допуск" in text   # дисклеймер не вырезан
    assert "возможного нарушения" in text and len(SENTENCE.findall(text)) == 3
    assert text.endswith("требуется проверка человеком.")


def test_fallback_names_are_catalog_banking_names():
    assert FALLBACK_TYPES["V01"]["name"] == "Посторонний в зоне кассира"
    assert FALLBACK_TYPES["V09"]["name"] == "Личные вещи на рабочем месте кассира"
