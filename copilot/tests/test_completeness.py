"""Неполный анализ ≠ «нарушений нет»; нет сведений о полноте ≠ «полный» (unknown ≠ false)."""
from __future__ import annotations

import base64

import pytest

from copilot.guard import UNVERIFIED_ANSWER
from copilot.llm import ScriptedLLM
from copilot.prompts import completeness, normalize_result
from copilot.report import report
from copilot.service import CopilotService

NO_DATA = "По имеющимся данным это установить нельзя"


def _clean(meta=None, **top):
    return {"job_id": "c", "status": "done", "verdict": "no_violation", "violations": [], "meta": meta or {}, **top}


@pytest.mark.parametrize("res, state", [
    (_clean({"completeness": {"complete": True}}), "complete"),
    (_clean({"completeness": {"complete": False, "reason": "detect: timeout"}}), "incomplete"),
    (_clean({"completeness": {"complete": True, "required_stages": ["detect"], "stages": {"detect": "failed"}}}),
     "incomplete"),
    (_clean({"completeness": {"complete": True, "analyzed_until_sec": 30, "duration_sec": 60}}), "incomplete"),
    (_clean({}, analysis_complete=False), "incomplete"),
    (_clean({"partial": True, "complete": False}), "incomplete"),
    (_clean({"coverage": 0.6}), "incomplete"),
    (_clean({"coverage": {"ratio": 1.0}}), "complete"),
    (_clean({"completeness": True}), "complete"),
    (_clean({}), "unknown"),
    ({"job_id": "f", "status": "failed", "verdict": None, "violations": []}, "incomplete"),
    ({"job_id": "p", "status": "processing"}, "pending"),
])
def test_completeness_states(res, state) -> None:
    assert completeness(normalize_result(res))["state"] == state


def test_ratio_and_reason() -> None:
    c = completeness(normalize_result(_clean({"completeness": {"complete": False, "analyzed_until_sec": 30,
                                                                "duration_sec": 60, "reason": "budget"}})))
    assert c["ratio"] == pytest.approx(0.5) and "budget" in c["reason"]


def test_clean_but_incomplete_is_not_no_violation(cfg, corpus) -> None:
    res = _clean({"completeness": {"complete": False, "analyzed_until_sec": 30, "duration_sec": 60}})
    r = CopilotService(cfg=cfg, corpus=corpus).answer(res, "Есть ли нарушения?", [])
    assert r["answer"].startswith(NO_DATA) and "не означает" in r["answer"] and "50%" in r["answer"]


def test_clean_without_completeness_info_is_caveated(cfg, corpus) -> None:
    r = CopilotService(cfg=cfg, corpus=corpus).answer(_clean({}), "Есть ли нарушения?", [])
    assert "не обнаружено" in r["answer"] and "Полнота анализа не подтверждена" in r["answer"]


def test_llm_cannot_say_no_violations_when_incomplete(cfg, corpus) -> None:
    res = _clean({"completeness": {"complete": False}})
    llm = ScriptedLLM(["Нарушений нет, всё в порядке."])
    r = CopilotService(cfg=cfg, corpus=corpus, llm=llm).answer(res, "Итог?", [])
    assert r["answer"] == UNVERIFIED_ANSWER and r["grounded"] is False
    assert any("противоречит результату" in x for x in r["removed_citations"])


def test_llm_cannot_deny_reported_violation(cfg, corpus, job_violation) -> None:
    llm = ScriptedLLM(["Нарушений не обнаружено. Кассир держит телефон у уха [V1]."])
    r = CopilotService(cfg=cfg, corpus=corpus, llm=llm).answer(job_violation, "Итог?", [])
    assert "Нарушений не обнаружено" not in r["answer"] and "телефон у уха" in r["answer"]


def test_scoped_or_local_claims_are_kept(cfg, corpus, job_violation) -> None:
    job = {**job_violation, "meta": {**job_violation["meta"], "completeness": {"complete": True}}}
    llm = ScriptedLLM(["Кассир держит телефон у уха [V1]. Других нарушений не обнаружено."])
    r = CopilotService(cfg=cfg, corpus=corpus, llm=llm).answer(job, "Итог?", [])
    assert "Других нарушений не обнаружено" in r["answer"]


def test_llm_cannot_invent_violation_on_clean_complete(cfg, corpus) -> None:
    res = _clean({"completeness": {"complete": True}})
    llm = ScriptedLLM(["Обнаружено нарушение порядка работы с сейфом."])
    r = CopilotService(cfg=cfg, corpus=corpus, llm=llm).answer(res, "Итог?", [])
    assert "Обнаружено нарушение" not in r["answer"]


def test_report_incomplete_banner(cfg, corpus) -> None:
    md = report(_clean({"completeness": {"complete": False, "reason": "detect: timeout"}}), corpus=corpus, cfg=cfg)
    assert "Анализ неполный" in md and "сделать нельзя" in md and "НЕПОЛНЫЙ анализ" in md


def test_meta_compaction_strips_images_keeps_explanations() -> None:
    jpeg = base64.b64encode(b"\xff\xd8\xff" + b"0" * 3000 + b"\xff\xd9").decode()
    res = normalize_result({"job_id": "m", "status": "done", "verdict": "violation", "violations": [],
                            "timeline": [[0, 0.1]] * 500,
                            "meta": {"ui": {"evidence_overlays": [{"violation_index": 0, "frames": [jpeg],
                                                                   "shapes": [{"x": 1}], "facts": ["факт"],
                                                                   "sources": {"rules": 0.7}, "vlm": "supports"}],
                                            "review_hints": [{"type_id": "V05", "facts": ["f"], "reasons_ru": ["ниже порога"]}]},
                                     "scores_raw": {"big": list(range(1000))},
                                     "warnings": ["w"], "completeness": {"complete": True}}})
    dumped = str(res)
    assert jpeg[:100] not in dumped and "timeline" not in res and "scores_raw" not in res["meta"]
    ov = res["meta"]["ui"]["evidence_overlays"][0]
    assert ov["facts"] == ["факт"] and ov["sources"] == {"rules": 0.7} and "frames" not in ov and "shapes" not in ov
    assert res["meta"]["ui"]["review_hints"][0]["reasons_ru"] == ["ниже порога"]
